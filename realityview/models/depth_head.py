"""DA3's DualDPT depth head, main branch only (no auxiliary ray branch), with its per-pixel features exposed.

Same layers and parameter names as the main branch of ``da3/dualdpt.py``, so the DA3 weights
(``head.*`` without ``*_aux*``) load strictly. Per view:

    4 token maps (h*w, 2C) -> LayerNorm -> 1x1 conv to out_channels[i] -> + UV position embedding
    -> resize to the patch grid x4, x2, x1, /2 -> 3x3 conv to ``features`` -> RefineNet 4 -> 3 -> 2 -> 1
    -> output_conv1 (3x3, features -> features/2) -> bilinear to H x W (align_corners=True)
    -> + UV position embedding = per-pixel feature F (features/2 channels: 128 LARGE, 64 BASE)
    -> output_conv2 (3x3 -> 32, ReLU, 1x1 -> 2): channel 0 = l, channel 1 = q
    depth = exp(l) (DA3 units), depth_conf = exp(q) + 1

``F`` is what a later head (appearance) attaches to. The head always runs in float32 with autocast
disabled, as DA3 does (``da3.py`` line 139).
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .da3.build import PATCH_SIZE, model_config
from .da3.dpt import _make_fusion_block, _make_scratch
from .da3.utils.head_utils import create_uv_grid, custom_interpolate, position_grid_to_embed


def _add_pos_embed(x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
    """UV position embedding of DualDPT (``dualdpt.py`` ``_add_pos_embed``)."""
    pw, ph = x.shape[-1], x.shape[-2]
    pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
    pe = position_grid_to_embed(pe, x.shape[1]) * ratio
    pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
    return x + pe


class DepthHead(nn.Module):
    def __init__(self, dim_in: int, features: int = 256, out_channels: Sequence[int] = (256, 512, 1024, 1024),
                 output_dim: int = 2, patch_size: int = PATCH_SIZE):
        super().__init__()
        self.patch_size = patch_size
        self.feature_dim = features // 2
        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList([nn.Conv2d(dim_in, oc, kernel_size=1) for oc in out_channels])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])
        self.scratch = _make_scratch(list(out_channels), features, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)
        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_dim, kernel_size=1, stride=1, padding=0),
        )

    @classmethod
    def from_model_name(cls, name: str) -> "DepthHead":
        h = model_config(name)["head"]
        if h["type"] != "dualdpt":
            raise ValueError(f"{name} has no DualDPT head")
        return cls(h["dim_in"], h["features"], h["out_channels"], h["output_dim"])

    # ------------------------------------------------------------------------------------------------------
    def _features(self, feats, H: int, W: int) -> torch.Tensor:
        """(M, h*w, 2C) x 4 -> per-pixel feature F (M, features/2, H, W)."""
        M, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized = []
        for i in range(4):
            x = self.norm(feats[i])
            x = x.permute(0, 2, 1).reshape(M, C, ph, pw)
            x = self.projects[i](x)
            x = _add_pos_embed(x, W, H)
            resized.append(self.resize_layers[i](x))
        l1, l2, l3, l4 = (getattr(self.scratch, f"layer{i}_rn")(r) for i, r in enumerate(resized, start=1))
        out = self.scratch.refinenet4(l4, size=l3.shape[2:])
        out = self.scratch.refinenet3(out, l3, size=l2.shape[2:])
        out = self.scratch.refinenet2(out, l2, size=l1.shape[2:])
        out = self.scratch.refinenet1(out, l1)
        out = self.scratch.output_conv1(out)
        out = custom_interpolate(out, (ph * self.patch_size, pw * self.patch_size), mode="bilinear",
                                 align_corners=True)
        return _add_pos_embed(out, W, H)

    def _chunk(self, feats, H, W, keep_features: bool):
        f = self._features(feats, H, W)
        logits = self.scratch.output_conv2(f)
        return (logits, f) if keep_features else (logits,)

    def forward(self, feats, image_hw, *, chunk_size: Optional[int] = 8, return_features: bool = False,
                grad_checkpoint: bool = False) -> dict:
        """
        Args:
            feats: list of 4 tensors (B, N, h*w, 2C) from the backbone.
            image_hw: (H, W) of the input images.
            chunk_size: run at most this many views at a time (B*N are split; None = all at once). The
                default 8 is what DA3 uses; with TF32 convolutions (PyTorch default) the result depends
                slightly on the chunking, so 8 reproduces DA3 bit for bit.
            return_features: also return F.
            grad_checkpoint: recompute each chunk in the backward pass (``use_reentrant=False``).
        Returns:
            log_depth (B, N, H, W)  l, the head's raw output, DA3 depth = exp(l)
            depth (B, N, H, W)      exp(l)
            depth_conf (B, N, H, W) exp(q) + 1
            features (B, N, features/2, H, W) if requested
        """
        H, W = image_hw
        B, N, P, C = feats[0].shape
        flat = [f.reshape(B * N, P, C) for f in feats]
        M = B * N
        step = M if not chunk_size else chunk_size
        logits, fmaps = [], []
        with torch.autocast(device_type=flat[0].device.type, enabled=False):
            flat = [f.float() for f in flat]
            for s in range(0, M, step):
                part = [f[s:s + step] for f in flat]
                if grad_checkpoint and torch.is_grad_enabled():
                    res = checkpoint(self._chunk, part, H, W, return_features, use_reentrant=False)
                else:
                    res = self._chunk(part, H, W, return_features)
                logits.append(res[0])
                if return_features:
                    fmaps.append(res[1])
            logits = torch.cat(logits, 0) if len(logits) > 1 else logits[0]
            log_depth = logits[:, 0]
            out = {
                "log_depth": log_depth.reshape(B, N, H, W),
                "depth": torch.exp(log_depth).reshape(B, N, H, W),
                "depth_conf": (torch.exp(logits[:, 1]) + 1).reshape(B, N, H, W),
            }
            if return_features:
                f = torch.cat(fmaps, 0) if len(fmaps) > 1 else fmaps[0]
                out["features"] = f.reshape(B, N, *f.shape[1:])
        return out
