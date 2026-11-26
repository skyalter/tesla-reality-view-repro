"""The stage-2 model: frozen stage-1 network (backbone + depth head), frozen video autoencoder, the Wan2.1
diffusion backbone with the feature cross-attention, and the conditioner.

One training sample (see ``data/stage2_dataset.py``) is a clip of T frames seen from the target camera plus
M conditioning moments, each with the usual 14 input views. Per latent frame the diffusion backbone looks at
the tokens of one conditioning moment: latent frame j uses moment ``min(j // cond_every, M - 1)``.

    forward(batch, sigma, noise, drop) -> {"pred", "target", "latents"}      training
    sample(batch, ...) -> latents;  decode(latents) -> uint8 video           inference

Batch keys used here (B = samples): images (B, M, N, 3, H, W) uint8, K, ref_T_cam (B, M, N, ...) relative to
each moment's own reference (what the backbone was trained with), dt (B, M, N), cam_id, slot (B, M, N),
view_valid (B, N) bool, clip_T_cam (B, M, N, 4, 4), src_time (B, M, N), target (B, T, 3, Ht, Wt) uint8 (or
``latents``), tgt_clip_T_cam (B, F, 4, 4), tgt_K (B, 3, 3), tgt_ego_T_cam (B, 4, 4), tgt_time (B, F).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from . import schedule
from .condition import Conditioner, pool_features, target_rays, token_geometry, token_log_depth
from .wan_dit import CONFIGS, WanDiT

SCOPES = ("adapters", "adapters+self_attn", "full")


class VideoVAE(nn.Module):
    """The Wan2.1 video autoencoder (frozen): 4x in time, 8x in space, 16 channels.

    encode: uint8 video (B, T, 3, H, W), T = 1 + 4k -> normalised latents (B, 16, 1 + k, H / 8, W / 8)
    decode: latents -> uint8 video (B, T, 3, H, W)
    """
    MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
    STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
           3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
    time_stride, space_stride, channels = 4, 8, 16

    def __init__(self, path: Optional[str] = None):
        super().__init__()
        from .wan.vae import WanVAE_

        self.model = WanVAE_(dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                             temperal_downsample=[False, True, True], dropout=0.0)
        if path:
            self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.register_buffer("mean", torch.tensor(self.MEAN), persistent=False)
        self.register_buffer("std", torch.tensor(self.STD), persistent=False)
        self.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        scale = [self.mean, 1.0 / self.std]
        out = []
        with torch.autocast(video.device.type, enabled=False):
            for v in video:
                x = v.float().div(127.5).sub(1.0).permute(1, 0, 2, 3).unsqueeze(0)      # (1, 3, T, H, W)
                out.append(self.model.encode(x, scale).float())
        return torch.cat(out)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        scale = [self.mean, 1.0 / self.std]
        out = []
        with torch.autocast(latents.device.type, enabled=False):
            for z in latents:
                x = self.model.decode(z.float().unsqueeze(0), scale).clamp(-1, 1)        # (1, 3, T, H, W)
                out.append(((x[0].permute(1, 0, 2, 3) + 1) * 127.5).round().to(torch.uint8))
        return torch.stack(out)


class Stage2Model(nn.Module):
    """
    Args:
        stage1: a ``Stage1Model`` (backbone + depth head). It is frozen here and kept in evaluation mode.
        dit: the diffusion backbone (``WanDiT``).
        vae: the video autoencoder (``VideoVAE``), or any module with ``encode`` / ``decode`` of the same
            shapes; None when the batches bring ``latents``.
        text: (L, text_dim) encoding of the fixed prompt.
        cond_every: one conditioning moment per this many latent frames.
        use_depth: the token geometry uses the depth head (3-D point and where it lands in the target image).
        scope: which parameters train, one of ``SCOPES``.
        amp_dtype: autocast dtype of the conditioner and the diffusion backbone on the GPU; None = float32.
        moment_chunk: conditioning moments per pass of the stage-1 network (memory).
    """

    def __init__(self, stage1: nn.Module, dit: WanDiT, vae: Optional[nn.Module], text: torch.Tensor, *,
                 cond_every: int = 2, use_depth: bool = True, scope: str = "adapters+self_attn",
                 amp_dtype: Optional[torch.dtype] = torch.bfloat16, moment_chunk: int = 2, n_cams: int = 7,
                 n_slots: int = 2):
        super().__init__()
        self.stage1 = stage1.requires_grad_(False).eval()
        self.dit = dit
        self.vae = vae
        self.register_buffer("text", text.float(), persistent=False)
        bb = stage1.backbone
        self.cond = Conditioner(2 * bb.embed_dim, len(bb.out_layers), dit.dim, n_cams=n_cams, n_slots=n_slots,
                                use_depth=use_depth)
        self.cond_every = int(cond_every)
        self.amp_dtype = amp_dtype
        self.moment_chunk = int(moment_chunk)
        self.scope = None
        self.set_scope(scope)

    # ------------------------------------------------------------------ what trains
    def set_scope(self, scope: str) -> None:
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}")
        self.scope = scope
        self.dit.requires_grad_(scope == "full")
        for name, p in self.dit.named_parameters():
            if ".feat_attn." in name or ".norm_feat." in name:
                p.requires_grad_(True)
            elif scope == "adapters+self_attn" and ".self_attn." in name:
                p.requires_grad_(True)
        self.cond.requires_grad_(True)

    def param_groups(self) -> dict:
        """Trainable parameters by part: ``new`` (conditioner and feature cross-attention) and ``pretrained``
        (whatever of the original diffusion backbone trains)."""
        new = list(self.cond.parameters())
        pre = []
        for name, p in self.dit.named_parameters():
            if not p.requires_grad:
                continue
            (new if ".feat_attn." in name or ".norm_feat." in name else pre).append(p)
        return {"new": new, "pretrained": pre}

    def frozen_prefixes(self) -> tuple:
        """Names of state-dict entries that never change and are rebuilt from the pretrained files: a
        checkpoint can leave them out."""
        frozen = ["stage1.", "vae."]
        frozen += ["dit." + name for name, p in self.dit.named_parameters() if not p.requires_grad]
        return tuple(frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        self.stage1.eval()
        if self.vae is not None:
            self.vae.eval()
        return self

    # ------------------------------------------------------------------ pieces
    def _autocast(self, device):
        on = self.amp_dtype is not None and device.type == "cuda"
        return torch.autocast(device.type, dtype=self.amp_dtype if on else None, enabled=on)

    def moment_of_frame(self, n_latent: int, n_moments: int, device) -> torch.Tensor:
        return (torch.arange(n_latent, device=device) // self.cond_every).clamp(max=n_moments - 1)

    @torch.no_grad()
    def backbone_outputs(self, images, ref_T_cam, K, dt):
        """Stage-1 network on the M moments of one sample (each a set of views):
        (pooled features (M, n, L, levels * D) in half precision, token log depth (M, n, L))."""
        pooled, log_depth = [], []
        for s in range(0, images.shape[0], self.moment_chunk):
            sl = slice(s, s + self.moment_chunk)
            out = self.stage1(images[sl], ref_T_cam[sl], K[sl], dt[sl], with_reference=False,
                              return_backbone_features=True)
            self.patch_hw = tuple(out["patch_hw"])
            pooled.append(pool_features(out["backbone_feats"], self.patch_hw, torch.float16))
            log_depth.append(token_log_depth(out["depth"], self.patch_hw))
            del out
        return torch.cat(pooled), torch.cat(log_depth)

    def conditions(self, batch: dict, n_latent: int, tgt_hw, drop: Optional[torch.Tensor] = None):
        """(feat_tokens (B, F, N * L, dim), feat_mask (B, N * L) bool, extra (B, F * hw, dim))."""
        images = batch["images"]
        B, M, N = images.shape[:3]
        dev = images.device
        mof = self.moment_of_frame(n_latent, M, dev)
        L = self.cond.tokens_per_view
        tokens, masks = [], []
        for b in range(B):
            valid = batch["view_valid"][b] if "view_valid" in batch else torch.ones(N, dtype=torch.bool, device=dev)
            idx = valid.nonzero().squeeze(1)
            feats, log_depth = self.backbone_outputs(images[b][:, idx], batch["ref_T_cam"][b][:, idx],
                                                     batch["K"][b][:, idx], batch["dt"][b][:, idx])
            geom = token_geometry(log_depth, batch["K"][b][:, idx],
                                  batch["clip_T_cam"][b][:, idx], batch["src_time"][b][:, idx],
                                  batch["tgt_clip_T_cam"][b], batch["tgt_K"][b], tgt_hw, batch["tgt_time"][b], mof,
                                  images.shape[-2:], self.patch_hw, use_depth=self.cond.use_depth)
            with self._autocast(dev):
                tok = self.cond.feature_tokens(feats, batch["cam_id"][b][:, idx], batch["slot"][b][:, idx])
                tok = self.cond.frame_tokens(tok, geom, mof).float()                # (F, n * L, dim)
            # Both branches touch every parameter of the conditioner (with a zero factor where it does not
            # count), so that distributed training never sees a parameter without a gradient.
            null = self.cond.null_token.float()
            if drop is not None and bool(drop[b]):
                tok = null.expand_as(tok) + 0.0 * tok
            else:
                tok = tok + 0.0 * null.sum()
            full = tok.new_zeros(n_latent, N * L, tok.shape[-1])
            full[:, :tok.shape[1]] = tok
            mask = torch.zeros(N * L, dtype=torch.bool, device=dev)
            mask[:tok.shape[1]] = True
            tokens.append(full)
            masks.append(mask)
        feat_mask = torch.stack(masks)
        rays = target_rays(batch["tgt_K"], batch["tgt_ego_T_cam"], tgt_hw,
                           patch_px=self.vae_stride * self.dit.patch_size[1])
        with self._autocast(dev):
            extra = self.cond.target_embedding(rays, n_latent).float()
        return torch.stack(tokens), (None if bool(feat_mask.all()) else feat_mask), extra

    @property
    def vae_stride(self) -> int:
        return getattr(self.vae, "space_stride", 8)

    def denoise(self, x, sigma, feat_tokens, feat_mask, extra):
        with self._autocast(x.device):
            return self.dit(x, sigma * schedule.NUM_TRAIN_STEPS, self.text, feat_tokens, extra, feat_mask).float()

    def latents_of(self, batch: dict) -> torch.Tensor:
        if "latents" in batch:
            return batch["latents"].float()
        return self.vae.encode(batch["target"])

    # ------------------------------------------------------------------ training
    def forward(self, batch: dict, sigma: torch.Tensor, noise: Optional[torch.Tensor] = None,
                drop: Optional[torch.Tensor] = None) -> dict:
        """sigma (B,) noise levels; noise like the latents (drawn here when None); drop (B,) bool: samples
        whose feature tokens are replaced by the null token."""
        latents = self.latents_of(batch)
        if noise is None:
            noise = torch.randn_like(latents)
        tgt_hw = (latents.shape[-2] * self.vae_stride, latents.shape[-1] * self.vae_stride)
        feat_tokens, feat_mask, extra = self.conditions(batch, latents.shape[2], tgt_hw, drop)
        pred = self.denoise(schedule.add_noise(latents, noise, sigma), sigma, feat_tokens, feat_mask, extra)
        return {"pred": pred, "target": schedule.velocity_target(latents, noise), "latents": latents}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def sample(self, batch: dict, n_frames: int, tgt_hw, steps: int = 50, guidance: float = 3.0, shift: float = 5.0,
               generator=None, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Latents (B, 16, F, h, w) of a video of ``n_frames`` frames at ``tgt_hw`` for the conditions in
        ``batch``. ``guidance`` 1 = no guidance; otherwise null + guidance * (conditioned - null)."""
        B = batch["images"].shape[0]
        dev = batch["images"].device
        F_lat = (n_frames - 1) // 4 + 1
        shape = (B, self.dit.out_dim, F_lat, tgt_hw[0] // self.vae_stride, tgt_hw[1] // self.vae_stride)
        feat_tokens, feat_mask, extra = self.conditions(batch, F_lat, tgt_hw)
        null_tokens = self.cond.null_token.float().expand_as(feat_tokens) if guidance != 1.0 else None

        def predict(x, t):
            sigma = t / schedule.NUM_TRAIN_STEPS
            v = self.denoise(x, sigma, feat_tokens, feat_mask, extra)
            if null_tokens is None:
                return v
            v0 = self.denoise(x, sigma, null_tokens, feat_mask, extra)
            return v0 + guidance * (v - v0)

        return schedule.sample_euler(predict, shape, steps, shift, generator=generator, device=dev, noise=noise)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents)


def build_dit(name: str = "wan2.1-t2v-1.3b", weights: Optional[str] = None, grad_checkpoint: bool = True) -> WanDiT:
    """The diffusion backbone; with ``weights`` (the official safetensors file) it starts from them and the
    new attention layers start as zero-output copies, without it everything is random (tests)."""
    dit = WanDiT(**CONFIGS[name], grad_checkpoint=grad_checkpoint)
    if weights:
        dit.load_pretrained(weights)
    else:
        dit.init_feature_attention()
    return dit
