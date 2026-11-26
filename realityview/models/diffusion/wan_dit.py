"""The Wan2.1 text-to-video diffusion backbone, re-written so that it loads the official weights and takes
our two conditions.

Structure and parameter names follow ``wan/modules/model.py`` of https://github.com/Wan-Video/Wan2.1
(Apache License 2.0, commit 9737cba9): ``patch_embedding``, ``text_embedding``, ``time_embedding``,
``time_projection``, ``blocks.<i>.{self_attn, norm3, cross_attn, ffn, modulation}``, ``head``. Differences:

* inputs are batched tensors of one size instead of lists of videos (no padding, no sequence lengths);
* attention uses ``torch.nn.functional.scaled_dot_product_attention`` (no flash-attn package);
* no dependency on diffusers;
* every block has a second cross-attention, ``feat_attn`` with ``norm_feat``, that looks at the camera
  feature tokens of the latent frame's own conditioning moment. Its output projection starts at zero, so a
  freshly built model computes exactly what the official one does;
* an optional tensor is added to the patch embedding (the target-camera ray encoding);
* optional gradient checkpointing per block.

Precision is as in the original: the residual stream, the time embedding and the modulation stay in single
precision, the attention and feed-forward layers run under the caller's autocast (bf16).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

WAN_1_3B = dict(dim=1536, ffn_dim=8960, freq_dim=256, num_heads=12, num_layers=30, in_dim=16, out_dim=16,
                text_dim=4096, text_len=512, patch_size=(1, 2, 2), eps=1e-6)
# A small network of the same structure for tests on the CPU.
WAN_TINY = dict(dim=48, ffn_dim=96, freq_dim=32, num_heads=4, num_layers=2, in_dim=16, out_dim=16,
                text_dim=32, text_len=8, patch_size=(1, 2, 2), eps=1e-6)
CONFIGS = {"wan2.1-t2v-1.3b": WAN_1_3B, "tiny": WAN_TINY}


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    half = dim // 2
    position = position.to(torch.float64)
    freqs = torch.pow(10000, -torch.arange(half, dtype=torch.float64, device=position.device) / half)
    sinusoid = torch.outer(position, freqs)
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


def rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = torch.outer(torch.arange(max_seq_len, dtype=torch.float64),
                        1.0 / torch.pow(theta, torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_grid(freqs: torch.Tensor, grid) -> torch.Tensor:
    """Rotary factors of every token of an (f, h, w) grid: complex (f*h*w, 1, head_dim / 2)."""
    f, h, w = grid
    c = freqs.shape[1]
    parts = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    return torch.cat([parts[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                      parts[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                      parts[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(f * h * w, 1, -1)


def rope_apply(x: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
    """x (B, L, heads, head_dim) -> the same, rotated. Computed in single precision."""
    b, s, n, d = x.shape
    with torch.autocast(x.device.type, enabled=False):
        xc = torch.view_as_complex(x.float().reshape(b, s, n, d // 2, 2))
        return torch.view_as_real(xc * factors.to(xc.dtype)).flatten(3)


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.type_as(x) * self.weight


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


def _attend(q, k, v, key_mask=None):
    """q (B, Lq, heads, d), k, v (B, Lk, heads, d) -> (B, Lq, heads * d). key_mask (B, Lk) bool: True = the
    key takes part."""
    mask = None if key_mask is None else key_mask[:, None, None, :]
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
    return out.transpose(1, 2).flatten(2)


class WanSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps)
        self.norm_k = WanRMSNorm(dim, eps=eps)

    def forward(self, x, rope):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        q = rope_apply(q, rope).to(v.dtype)
        k = rope_apply(k, rope).to(v.dtype)
        return self.o(_attend(q, k, v))


class WanCrossAttention(WanSelfAttention):
    def forward(self, x, context, key_mask=None):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        return self.o(_attend(q, k.to(q.dtype), v.to(q.dtype), key_mask))


class WanAttentionBlock(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, num_heads: int, eps: float = 1e-6, feature_attention: bool = True):
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True)
        self.cross_attn = WanCrossAttention(dim, num_heads, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)
        if feature_attention:
            self.norm_feat = WanLayerNorm(dim, eps, elementwise_affine=True)
            self.feat_attn = WanCrossAttention(dim, num_heads, eps)
        else:
            self.norm_feat = self.feat_attn = None

    def forward(self, x, e, rope, context, feat_tokens=None, feat_mask=None):
        """x (B, L, C) single precision; e (B, 6, C); context (B, Lt, C) text tokens;
        feat_tokens (B, F, Lc, C) camera feature tokens per latent frame (L = F * tokens per frame), or None;
        feat_mask (B, Lc) bool, True = the token exists (None = all)."""
        with torch.autocast(x.device.type, enabled=False):
            e = (self.modulation.float() + e.float()).chunk(6, dim=1)
        y = self.self_attn(self.norm1(x).float() * (1 + e[1]) + e[0], rope)
        x = x + y.float() * e[2]
        x = x + self.cross_attn(self.norm3(x), context).float()
        if feat_tokens is not None and self.feat_attn is not None:
            b, f, lc, c = feat_tokens.shape
            q = self.norm_feat(x).reshape(b * f, -1, c)
            km = None if feat_mask is None else feat_mask[:, None, :].expand(b, f, lc).reshape(b * f, lc)
            y = self.feat_attn(q, feat_tokens.reshape(b * f, lc, c), km).reshape(b, -1, c)
            x = x + y.float()
        y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
        return x + y.float() * e[5]


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size, eps: float = 1e-6):
        super().__init__()
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        with torch.autocast(x.device.type, enabled=False):
            e = (self.modulation.float() + e.float().unsqueeze(1)).chunk(2, dim=1)
            return self.head(self.norm(x).float() * (1 + e[1]) + e[0])


class WanDiT(nn.Module):
    """forward(x, t, text, feat_tokens=None, extra=None) -> prediction with the shape of x.

        x            (B, 16, F, H, W) noisy latents
        t            (B,) time step in 0..1000 (= 1000 x noise level)
        text         (B, L, text_dim) or (L, text_dim) text-encoder output, L <= text_len (padded with zeros
                     to text_len like the original)
        feat_tokens  (B, F, Lc, dim) camera feature tokens of each latent frame, or None (no such condition)
        feat_mask    (B, Lc) bool, True = the token exists (samples with fewer input views are padded)
        extra        (B, F * H/2 * W/2, dim) added to the patch embedding, or None
    """

    def __init__(self, dim=1536, ffn_dim=8960, freq_dim=256, num_heads=12, num_layers=30, in_dim=16, out_dim=16,
                 text_dim=4096, text_len=512, patch_size=(1, 2, 2), eps=1e-6, feature_attention=True,
                 grad_checkpoint=False):
        super().__init__()
        self.dim, self.freq_dim, self.text_len, self.out_dim = dim, freq_dim, text_len, out_dim
        self.patch_size = tuple(patch_size)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([WanAttentionBlock(dim, ffn_dim, num_heads, eps, feature_attention)
                                     for _ in range(num_layers)])
        self.head = Head(dim, out_dim, self.patch_size, eps)
        d = dim // num_heads
        assert d % 2 == 0
        self.register_buffer("freqs", torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)),
                                                 rope_params(1024, 2 * (d // 6))], dim=1), persistent=False)
        self._rope_cache = {}

    # ------------------------------------------------------------------ weights
    def init_feature_attention(self) -> None:
        """Start every ``feat_attn`` as a copy of the block's text cross-attention with a zero output
        projection (call after the pretrained weights are loaded)."""
        for blk in self.blocks:
            if blk.feat_attn is None:
                continue
            blk.feat_attn.load_state_dict(blk.cross_attn.state_dict())
            blk.norm_feat.load_state_dict(blk.norm3.state_dict())
            nn.init.zeros_(blk.feat_attn.o.weight)
            nn.init.zeros_(blk.feat_attn.o.bias)

    def load_pretrained(self, path: str) -> None:
        """Load the official ``diffusion_pytorch_model.safetensors``; the new modules are then initialised
        from it (``init_feature_attention``)."""
        from safetensors.torch import load_file

        state = load_file(path)
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [k for k in missing if ".feat_attn." not in k and ".norm_feat." not in k]
        if missing or unexpected:
            raise RuntimeError(f"{path} does not fit: missing {missing[:5]}, unexpected {list(unexpected)[:5]}")
        self.init_feature_attention()

    # ------------------------------------------------------------------ forward
    def _rope(self, grid, device):
        key = (tuple(grid), str(device))
        if key not in self._rope_cache:
            self._rope_cache = {key: rope_grid(self.freqs.to(device), grid).to(torch.complex64)}
        return self._rope_cache[key]

    def forward(self, x, t, text, feat_tokens: Optional[torch.Tensor] = None, extra: Optional[torch.Tensor] = None,
                feat_mask: Optional[torch.Tensor] = None):
        b = x.shape[0]
        x = self.patch_embedding(x.to(self.patch_embedding.weight.dtype))
        grid = x.shape[2:]
        x = x.flatten(2).transpose(1, 2).float()
        if extra is not None:
            x = x + extra.float()
        rope = self._rope(grid, x.device)
        with torch.autocast(x.device.type, enabled=False):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        if text.dim() == 2:
            text = text.unsqueeze(0).expand(b, -1, -1)
        if text.shape[1] < self.text_len:
            text = torch.cat([text, text.new_zeros(b, self.text_len - text.shape[1], text.shape[2])], dim=1)
        context = self.text_embedding(text.to(self.text_embedding[0].weight.dtype))
        if feat_tokens is not None and feat_tokens.shape[1] != grid[0]:
            raise ValueError(f"feat_tokens has {feat_tokens.shape[1]} frames, the latents have {grid[0]}")
        for blk in self.blocks:
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(blk, x, e0, rope, context, feat_tokens, feat_mask, use_reentrant=False)
            else:
                x = blk(x, e0, rope, context, feat_tokens, feat_mask)
        x = self.head(x, e)
        return self.unpatchify(x, grid)

    def unpatchify(self, x, grid):
        b = x.shape[0]
        f, h, w = grid
        p = self.patch_size
        x = x.view(b, f, h, w, *p, self.out_dim)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        return x.reshape(b, self.out_dim, f * p[0], h * p[1], w * p[2])
