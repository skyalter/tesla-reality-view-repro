"""Stage 1 model: DA3 backbone + DA3 depth head, metric depth along the optical axis, and a frozen reference.

    z = pose_scale * kappa * exp(l)      clamped to [depth_min, depth_max] (default 0.5 .. 500 m)

``l`` is the depth head's raw output, ``pose_scale`` the same metres-per-unit used for the pose translations
(default 20 m), ``kappa`` a learnable global factor stored as ``log_kappa`` (initial value 0, so kappa = 1).
At initialisation (time encoding zero, kappa 1) ``z / pose_scale`` equals the original DA3 depth.

Frozen reference: at construction the model keeps frozen copies of everything that can change in training
and sits after the shared frozen part (the trainable blocks, the camera encoder if the first trainable
block is at or before ``alt_start``, the head). In ``forward`` the frozen leading layers run once; their
state feeds both the trainable branch and, under ``torch.no_grad()``, the frozen copy, which gives
``ref_depth``: the depth the original DA3 (no time encoding, kappa = initial value) predicts for the same
input. This follows OccAny (``occany/model/model_da3.py`` ``init_aux_branch``), but our copy also gets the
camera tokens from a frozen copy of the camera encoder, because our trainable range includes it.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .backbone import DA3Backbone
from .cameras import DEFAULT_POSE_SCALE
from .da3.build import component, frozen_copy, load_strict, read_weights, weights_path
from .depth_head import DepthHead


class FrozenReference(nn.Module):
    """Frozen copies of the parts of ``Stage1Model`` that training can change, taken at construction."""

    def __init__(self, model: "Stage1Model"):
        super().__init__()
        bb = model.backbone
        self.start = bb.first_trainable_layer()
        trainable = set(bb.trainable_blocks)
        self.blocks = nn.ModuleDict({str(i): frozen_copy(bb.vit.blocks[i])
                                     for i in range(self.start, bb.depth) if i in trainable})
        self.needs_cam = self.start <= bb.alt_start
        self.cam_enc = frozen_copy(bb.cam_enc) if (self.needs_cam and bb.train_cam_enc) else None
        head_trainable = any(p.requires_grad for p in model.head.parameters())
        self.head = frozen_copy(model.head) if head_trainable else None
        # frozen parts after ``start`` are the same modules in both branches; a plain dict keeps them out of
        # this module's parameters and state_dict
        self._shared = {
            "blocks": {i: bb.vit.blocks[i] for i in range(self.start, bb.depth) if i not in trainable},
            "cam_enc": bb.cam_enc if (self.needs_cam and not bb.train_cam_enc) else None,
            "head": None if head_trainable else model.head,
        }

    def block_map(self) -> Dict[int, nn.Module]:
        m = dict(self._shared["blocks"])
        m.update({int(k): v for k, v in self.blocks.items()})
        return m

    def camera_encoder(self):
        return self.cam_enc if self.cam_enc is not None else self._shared["cam_enc"]

    def depth_head(self):
        return self.head if self.head is not None else self._shared["head"]


class Stage1Model(nn.Module):
    """Backbone + depth head for stage 1.

    Args:
        model_name: "da3-large-1.1" (default) or "da3-base".
        weights: ``model.safetensors`` path; default ``$RV_WEIGHTS_ROOT/<model_name>/model.safetensors``.
        pose_scale: metres per DA3 unit, for pose translations and for the depth.
        freeze: backbone preset ("default", "late", "frozen"), see ``DA3Backbone``.
        train_head: the depth head is trainable (default True).
        frozen_reference: build the frozen reference branch.
        grad_checkpoint: checkpoint each backbone block; head_checkpoint / head_chunk_size: same for the head
            (per chunk of views).
        depth_min, depth_max: clamp of z in metres.
    """

    def __init__(self, model_name: str = "da3-large-1.1", weights: Optional[str] = None, *,
                 load_weights: bool = True, pose_scale: float = DEFAULT_POSE_SCALE, freeze: str = "default",
                 train_head: bool = True, frozen_reference: bool = True, grad_checkpoint: bool = False,
                 head_checkpoint: bool = False, head_chunk_size: Optional[int] = 8,
                 amp_dtype: Optional[torch.dtype] = torch.bfloat16, depth_min: float = 0.5,
                 depth_max: float = 500.0):
        super().__init__()
        sd = read_weights(weights or weights_path(model_name))[0] if load_weights else None
        self.backbone = DA3Backbone(model_name, state_dict=sd, load_weights=load_weights, pose_scale=pose_scale,
                                    freeze=freeze, grad_checkpoint=grad_checkpoint, amp_dtype=amp_dtype)
        self.head = DepthHead.from_model_name(model_name)
        if load_weights:
            load_strict(self.head, component(sd, "head", drop=("_aux",)))
        del sd
        self.head.requires_grad_(train_head)
        self.log_kappa = nn.Parameter(torch.zeros(()))
        self.register_buffer("log_kappa_init", torch.zeros(()))
        self.pose_scale = float(pose_scale)
        self.depth_min, self.depth_max = float(depth_min), float(depth_max)
        self.head_checkpoint = bool(head_checkpoint)
        self.head_chunk_size = head_chunk_size
        self.reference = FrozenReference(self) if frozen_reference else None

    # ------------------------------------------------------------------------------------------------------
    def param_groups(self) -> Dict[str, list]:
        """Trainable parameters by part, for per-part learning rates: backbone (blocks, camera encoder,
        time encoding), head, scale (log_kappa)."""
        bb = [p for p in self.backbone.parameters() if p.requires_grad]
        hd = [p for p in self.head.parameters() if p.requires_grad]
        return {"backbone": bb, "head": hd, "scale": [self.log_kappa]}

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _metric(self, head_out: dict, log_kappa: torch.Tensor) -> dict:
        l = head_out["log_depth"]
        log_z = l + (math.log(self.pose_scale) + log_kappa)
        z = (self.pose_scale * torch.exp(log_kappa)) * head_out["depth"]
        return {"depth": z.clamp(self.depth_min, self.depth_max), "log_depth": log_z}

    def forward(self, images, ref_T_cam, K, dt=None, *, with_reference: Optional[bool] = None,
                return_features: bool = False, return_backbone_features: bool = False) -> dict:
        """
        Inputs as ``DA3Backbone.forward`` (images (B, N, 3, H, W) uint8 or 0..1; ref_T_cam (B, N, 4, 4)
        camera-to-reference, OpenCV axes, metres; K (B, N, 3, 3) pixels; dt (B, N) seconds).

        Returns a dict, all (B, N, H, W) float32 unless noted:
            depth        z in metres along the optical axis, clamped to [depth_min, depth_max]
            log_depth    log z before clamping (use this in log-space losses: the clamp has no gradient)
            depth_conf   DA3 depth confidence exp(q) + 1 (>= 1)
            rel_depth    exp(l), the depth in DA3 units (what the original model outputs)
            features     (B, N, 128, H, W) per-pixel head features F   [return_features]
            backbone_feats / cam_tokens / patch_hw   backbone outputs   [return_backbone_features]
            ref_depth, ref_log_depth   frozen reference, no gradient   [with_reference, default on if built]
        """
        bb = self.backbone
        use_ref = (self.reference is not None) if with_reference is None else with_reference
        if use_ref and self.reference is None:
            raise ValueError("this model was built without the frozen reference")
        state = bb.prepare(images)
        hw = state["image_hw"]
        cam = bb.camera_tokens(ref_T_cam, K, dt, hw)
        k = self.reference.start if use_ref else 0
        shared = bb.run_blocks(state, 0, k, cam_token=cam if k > bb.alt_start else None)
        main = bb.finalize(bb.run_blocks(shared, k, bb.depth, cam_token=cam))
        h = self.head(main["feats"], hw, chunk_size=self.head_chunk_size, return_features=return_features,
                      grad_checkpoint=self.head_checkpoint)
        out = self._metric(h, self.log_kappa)
        out["depth_conf"] = h["depth_conf"]
        out["rel_depth"] = h["depth"]
        if return_features:
            out["features"] = h["features"]
        if return_backbone_features:
            out["backbone_feats"] = main["feats"]
            out["cam_tokens"] = main["cam_tokens"]
            out["patch_hw"] = main["patch_hw"]
        if use_ref:
            ref = self.reference
            with torch.no_grad():
                rcam = None
                if ref.needs_cam:
                    rcam = bb.camera_tokens(ref_T_cam, K, dt, hw, cam_enc=ref.camera_encoder(), use_time=False)
                rf = bb.finalize(bb.run_blocks(shared, k, bb.depth, blocks=ref.block_map(), cam_token=rcam))
                rh = ref.depth_head()(rf["feats"], hw, chunk_size=self.head_chunk_size)
                r = self._metric(rh, self.log_kappa_init)
            out["ref_depth"] = r["depth"]
            out["ref_log_depth"] = r["log_depth"]
        return out
