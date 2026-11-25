"""Sky scores from DA3METRIC-LARGE, per image, for offline sky masks.

DA3METRIC-LARGE is a single-view model (no cross-view attention): ViT-L, DPT head with a depth output and a
one-channel sky output with ReLU activation, at the input resolution. ``SkyScorer`` returns that raw sky output
(non-negative, not a probability; no threshold is applied here). For reference, DA3 itself uses two different
thresholds: ``sky < 0.3`` is "not sky" inside the model (``utils/alignment.py``), and the API returns
``sky >= 0.5`` as its boolean mask (``utils/io/output_processor.py``).

The scorer runs like the official API: ImageNet normalisation, backbone under bfloat16 autocast, head in
float32, ``torch.no_grad``. Images are processed independently, in chunks.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .cameras import normalize_images
from .da3.build import (account, build_backbone, build_original_head, component, freeze_, load_strict,
                        read_weights, weights_path)

MODEL_NAME = "da3metric-large"
# uint8 storage: value = round(score * SKY_UINT8_SCALE), clipped to 255, i.e. steps of 0.01 up to 2.55.
# On AV2 images the raw score stays below about 1.5 (sky is mostly around 1, ground exactly 0).
SKY_UINT8_SCALE = 100.0


class SkyScorer(nn.Module):
    def __init__(self, weights: Optional[str] = None, amp_dtype: Optional[torch.dtype] = torch.bfloat16):
        super().__init__()
        sd, _ = read_weights(weights or weights_path(MODEL_NAME))
        self.backbone = build_backbone(MODEL_NAME)
        self.head = build_original_head(MODEL_NAME)
        load_strict(self.backbone, component(sd, "backbone"))
        load_strict(self.head, component(sd, "head"))
        self.load_report = account(sd, ["backbone", "head"], [])
        self.amp_dtype = amp_dtype
        freeze_(self)
        self.eval()

    @torch.no_grad()
    def forward(self, images: torch.Tensor, chunk_size: int = 16, return_depth: bool = False):
        """
        Args:
            images: (M, 3, H, W) RGB, uint8 or float 0..1; H and W multiples of 14 (518 x 392 for AV2).
        Returns:
            sky (M, H, W) float32 raw sky score; with ``return_depth`` also the network depth (M, H, W)
            (DA3METRIC convention: metric depth = focal_px * depth / 300).
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must be (M, 3, H, W), got {tuple(images.shape)}")
        H, W = images.shape[-2:]
        dev = next(self.parameters()).device
        skies, depths = [], []
        for s in range(0, images.shape[0], chunk_size):
            x = normalize_images(images[s:s + chunk_size].to(dev))[None]  # (1, m, 3, H, W)
            with torch.autocast(device_type=dev.type, dtype=self.amp_dtype or torch.float32,
                                enabled=self.amp_dtype is not None):
                feats, _ = self.backbone(x, cam_token=None, export_feat_layers=[])
            with torch.autocast(device_type=dev.type, enabled=False):
                out = self.head(feats, H, W, patch_start_idx=0)  # DA3 default chunking (8 views, B = 1)
            skies.append(out["sky"][0].float())
            depths.append(out["depth"][0].float())
        sky = torch.cat(skies)
        return (sky, torch.cat(depths)) if return_depth else sky


def sky_score_to_uint8(score) -> np.ndarray:
    """Raw sky score (tensor or array) to uint8 for storage: round(score * 100), clipped to 0..255."""
    if isinstance(score, torch.Tensor):
        score = score.detach().float().cpu().numpy()
    return np.clip(np.rint(np.asarray(score, np.float32) * SKY_UINT8_SCALE), 0, 255).astype(np.uint8)


def sky_score_from_uint8(q) -> np.ndarray:
    """Inverse of ``sky_score_to_uint8`` (float32; saturated pixels come back as 2.55)."""
    return np.asarray(q, np.float32) / SKY_UINT8_SCALE
