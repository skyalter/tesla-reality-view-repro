"""Stage 1, second part: the Gaussian head on the frozen backbone.

    frozen backbone (after the depth training)  ->  depth head: trunk + depth branch (keeps training)
                                                                      |  per-pixel feature F
                                                                      v
                                                      appearance branch: size, rotation, colour, opacity
    one Gaussian per input pixel, at the predicted depth (see ``realityview/render/gaussians.py``)

``AppearanceBranch`` attaches to the per-pixel feature F of ``DepthHead`` (it does not share the last depth
convolutions). Its last layer starts at zero weights with biases chosen so that an untrained Gaussian sits on
the predicted surface, is about one pixel large, has the colour of its pixel and opacity 0.12.

``ExposureCorrection`` gives a 3x3 colour matrix and an offset per input view, predicted from the mean of
the view's last backbone features; it starts as the identity. It is applied to a rendered image only before
that image is compared with a photo, and absorbs exposure and white-balance differences between cameras so
that the geometry does not have to. (A simplification of Instant NuRec's per-camera correction, which uses
cross-attention instead of the mean.)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..render.gaussians import RAW_CHANNELS, assemble
from .cameras import DEFAULT_POSE_SCALE
from .stage1 import Stage1Model


class AppearanceBranch(nn.Module):
    """Per-pixel feature F (M, C, H, W) -> raw Gaussian parameters (M, 11, H, W): [a (3), q (4), f (3), o (1)]."""

    def __init__(self, dim_in: int, hidden: int = 64, size_bias: float = -2.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, hidden, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, RAW_CHANNELS, kernel_size=1),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        with torch.no_grad():
            last.bias.zero_()
            last.bias[0:3] = size_bias      # size: about 1.04 pixel footprints
            last.bias[3] = 1.0              # rotation: identity (w, x, y, z)

    def forward(self, features: torch.Tensor, chunk_size: Optional[int] = 8, grad_checkpoint: bool = False):
        M = features.shape[0]
        step = M if not chunk_size else chunk_size
        parts = []
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = features.float()
            for s in range(0, M, step):
                part = features[s:s + step]
                if grad_checkpoint and torch.is_grad_enabled():
                    parts.append(checkpoint(self.net, part, use_reentrant=False))
                else:
                    parts.append(self.net(part))
        return torch.cat(parts) if len(parts) > 1 else parts[0]


class ExposureCorrection(nn.Module):
    """Backbone tokens of each view (B, N, P, D) -> colour matrix A (B, N, 3, 3) and offset b (B, N, 3)."""

    def __init__(self, dim_tokens: int, hidden: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(dim_tokens)
        self.mlp = nn.Sequential(nn.Linear(dim_tokens, hidden), nn.GELU(), nn.Linear(hidden, 12))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, tokens: torch.Tensor):
        with torch.autocast(device_type=tokens.device.type, enabled=False):
            d = self.mlp(self.norm(tokens.float().mean(dim=-2)))
        eye = torch.eye(3, device=d.device, dtype=d.dtype)
        return eye + d[..., :9].reshape(*d.shape[:-1], 3, 3), d[..., 9:]

    @staticmethod
    def apply(rgb: torch.Tensor, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """rgb (M, 3, H, W) in 0..1, A (M, 3, 3), b (M, 3) -> clamp(A rgb + b, 0, 1)."""
        return (torch.einsum("mij,mjhw->mihw", A, rgb) + b[:, :, None, None]).clamp(0.0, 1.0)


class Stage1HeadModel(nn.Module):
    """Frozen backbone + depth head + appearance branch (+ exposure correction, background colour).

    Args:
        model_name, weights: as ``Stage1Model`` (the original DA3 weights; they define the architecture and
            are what the model holds when no ``backbone_checkpoint`` is given).
        backbone_checkpoint: a checkpoint of the backbone training (``last.pt`` of a stage1_backbone run);
            its backbone, depth head and global depth factor replace the original weights.
        pose_scale: metres per DA3 unit; must be the value the backbone was trained with.
        kappa: the global depth factor when there is no checkpoint (a checkpoint brings its own).
        train_depth_branch: the depth head keeps training (default); False freezes it.
        exposure: build the exposure correction.
        head_checkpoint / head_chunk_size: recompute the head per chunk of views in the backward pass.
        scale_range, max_scale_rel: size limits of a Gaussian, see ``render.gaussians.assemble``.
    The backbone is always frozen and stays in evaluation mode.
    """

    def __init__(self, model_name: str = "da3-large-1.1", weights: Optional[str] = None, *,
                 load_weights: bool = True, backbone_checkpoint: Optional[str] = None,
                 pose_scale: float = DEFAULT_POSE_SCALE, kappa: float = 1.0, train_depth_branch: bool = True,
                 exposure: bool = True, head_checkpoint: bool = False, head_chunk_size: Optional[int] = 8,
                 scale_range=(0.1, 8.0), max_scale_rel: float = 0.02, depth_min: float = 0.5,
                 depth_max: float = 500.0):
        super().__init__()
        self.stage1 = Stage1Model(model_name, weights, load_weights=load_weights, pose_scale=pose_scale,
                                  freeze="frozen", train_head=train_depth_branch, frozen_reference=False,
                                  head_checkpoint=head_checkpoint, head_chunk_size=head_chunk_size,
                                  depth_min=depth_min, depth_max=depth_max)
        with torch.no_grad():
            self.stage1.log_kappa.fill_(float(torch.log(torch.tensor(float(kappa)))))
            self.stage1.log_kappa_init.copy_(self.stage1.log_kappa)
        self.backbone_checkpoint = None
        if backbone_checkpoint:
            self.backbone_checkpoint = self.load_backbone_checkpoint(backbone_checkpoint, pose_scale)
        self.stage1.log_kappa.requires_grad_(False)
        self.appearance = AppearanceBranch(self.stage1.head.feature_dim)
        self.exposure = ExposureCorrection(2 * self.stage1.backbone.embed_dim) if exposure else None
        self.background_logit = nn.Parameter(torch.zeros(3))
        self.scale_range = tuple(float(x) for x in scale_range)
        self.max_scale_rel = float(max_scale_rel)
        self.head_checkpoint = bool(head_checkpoint)
        self.head_chunk_size = head_chunk_size

    def load_backbone_checkpoint(self, path: str, pose_scale: float) -> dict:
        """Load backbone, depth head and depth factor from a checkpoint of the backbone training."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        trained = (state.get("cfg") or {}).get("model") or {}
        if "pose_scale" in trained and abs(float(trained["pose_scale"]) - float(pose_scale)) > 1e-9:
            raise ValueError(f"the backbone was trained with pose_scale {trained['pose_scale']}, "
                             f"this model is built with {pose_scale}")
        weights = {k: v for k, v in state["model"].items() if not k.startswith("reference.")}
        missing, unexpected = self.stage1.load_state_dict(weights, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"backbone checkpoint does not fit: missing {missing[:5]}, unexpected {unexpected[:5]}")
        return {"path": str(path), "step": int(state.get("step", -1)), "code_version": state.get("code_version")}

    # ------------------------------------------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        self.stage1.backbone.eval()         # frozen: never in training mode
        return self

    def param_groups(self) -> dict:
        """Trainable parameters by part: depth_head, appearance (with the background colour), exposure."""
        groups = {"depth_head": [p for p in self.stage1.head.parameters() if p.requires_grad],
                  "appearance": list(self.appearance.parameters()) + [self.background_logit]}
        if self.exposure is not None:
            groups["exposure"] = list(self.exposure.parameters())
        return groups

    def forward(self, images, ref_T_cam, K, dt, keep, mover=None) -> dict:
        """
        Inputs as ``Stage1Model.forward`` plus, both (B, N, H, W) bool:
            keep    the pixels that produce a Gaussian
            mover   pixels on moving objects or persons (their Gaussians are flagged), or None

        Returns:
            depth, log_depth, depth_conf   (B, N, H, W), as ``Stage1Model``
            gaussians    list of B ``Gaussians`` (reference frame)
            exposure     (A (B, N, 3, 3), b (B, N, 3)) or None
            background   (3,) colour behind the Gaussians, 0..1
        """
        out = self.stage1(images, ref_T_cam, K, dt, with_reference=False, return_features=True,
                          return_backbone_features=self.exposure is not None)
        feats = out["features"]
        B, N, C, H, W = feats.shape
        raw = self.appearance(feats.reshape(B * N, C, H, W), self.head_chunk_size, self.head_checkpoint)
        raw = raw.reshape(B, N, RAW_CHANNELS, H, W)
        colours = images.float() / 255.0 if images.dtype == torch.uint8 else images.float()
        gaussians = [assemble(raw[b], colours[b], out["depth"][b], K[b], ref_T_cam[b], keep[b],
                              None if mover is None else mover[b], scale_range=self.scale_range,
                              max_scale_rel=self.max_scale_rel) for b in range(B)]
        result = {"depth": out["depth"], "log_depth": out["log_depth"], "depth_conf": out["depth_conf"],
                  "gaussians": gaussians, "exposure": None, "background": torch.sigmoid(self.background_logit)}
        if self.exposure is not None:
            result["exposure"] = self.exposure(out["backbone_feats"][-1])
        return result
