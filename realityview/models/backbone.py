"""The DA3 multi-view backbone with known poses, a time encoding, freezing presets and gradient checkpointing.

Inputs (B samples, N views each, all views of a batch the same size, H and W multiples of 14):

    images     (B, N, 3, H, W)  RGB, uint8 0..255 or float 0..1 (ImageNet normalisation is done here)
    ref_T_cam  (B, N, 4, 4)     camera-to-reference, OpenCV camera axes, metres (see ``cameras.py``)
    K          (B, N, 3, 3)     intrinsics in pixels of the H x W images
    dt         (B, N)           seconds, image time minus reference time (t - 0.5 s gives about -0.5)

Output of ``forward`` (a dict):

    feats       list of 4 tensors (B, N, h*w, 2C) float32, h = H/14, w = W/14, row-major patches; layers
                ``out_layers`` (LARGE 11, 15, 19, 23; BASE 5, 7, 9, 11). Channels [0, C) are the output of the
                last single-view layer (not normalised), [C, 2C) the output of that layer after the
                backbone's final LayerNorm. This is exactly what DA3 hands to its heads (C = 1024 LARGE, 768 BASE).
    cam_tokens  list of 4 tensors (B, N, 2C): the camera token of the same layers (before the LayerNorm).
    patch_hw    (h, w);  image_hw (H, W)

How the poses enter: ``ref_T_cam`` with translation divided by ``pose_scale`` (default 20 m) is inverted to
world-to-camera and given to DA3's camera encoder, which runs in float32; the time encoding of ``dt`` is
added to its output; the result replaces each view's first token at layer ``alt_start`` (8 for LARGE),
exactly where DA3 puts it. Because poses are always given, DA3 never reorders the views
(``vision_transformer.py`` only reorders when ``cam_token`` is None), so output view i is input view i.

The time encoding is 16 sine/cosine pairs of ``dt`` followed by a Linear(32 -> C) whose weight and bias
start at zero, so a freshly loaded backbone gives bit-identical outputs to DA3.

Precision: the transformer blocks run under ``torch.autocast(bfloat16)`` (residual stream stays float32, as
in DA3); the camera encoder and time encoding run in float32 with autocast disabled.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .cameras import DEFAULT_POSE_SCALE, da3_extrinsics, normalize_images
from .da3.build import (PATCH_SIZE, account, build_backbone, build_cam_enc, component, freeze_, load_strict,
                        model_config, read_weights, weights_path)

FREEZE_PRESETS = ("default", "late", "frozen")


class TimeEncoding(nn.Module):
    """sin/cos of dt at ``n_freqs`` periods (log-spaced, ``min_period``..``max_period`` s), then a zero Linear."""

    def __init__(self, dim: int, n_freqs: int = 16, min_period: float = 0.1, max_period: float = 10.0):
        super().__init__()
        periods = torch.logspace(math.log10(min_period), math.log10(max_period), n_freqs, dtype=torch.float64)
        self.register_buffer("omega", (2 * math.pi / periods).float(), persistent=False)
        self.proj = nn.Linear(2 * n_freqs, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        a = dt.float()[..., None] * self.omega
        return self.proj(torch.cat([a.sin(), a.cos()], dim=-1))


class DA3Backbone(nn.Module):
    """DA3 ViT + camera encoder + time encoding. See the module docstring for inputs and outputs.

    Args:
        model_name: "da3-large-1.1" or "da3-base".
        weights: path of ``model.safetensors``; default ``$RV_WEIGHTS_ROOT/<model_name>/model.safetensors``.
        state_dict: already read weights (``da3.build.read_weights``), instead of ``weights``.
        load_weights: False builds a randomly initialised network (tests that only count parameters).
        pose_scale: metres per DA3 translation unit.
        freeze: "default" (layers alt_start..last, camera encoder, time encoding trainable; LARGE 8-23),
            "late" (the last quarter of the layers only; LARGE 18-23), "frozen" (nothing). Patch embedding,
            positional embedding, the learned camera tokens, layers before alt_start and the final LayerNorm
            are always frozen. ``set_freeze`` can also take an explicit block range.
        grad_checkpoint: recompute each transformer block in the backward pass (``use_reentrant=False``).
        amp_dtype: autocast dtype of the blocks; None runs them in float32.
    """

    def __init__(self, model_name: str = "da3-large-1.1", weights: Optional[str] = None, *,
                 state_dict: Optional[Dict[str, torch.Tensor]] = None, load_weights: bool = True,
                 pose_scale: float = DEFAULT_POSE_SCALE, freeze: str = "default", grad_checkpoint: bool = False,
                 amp_dtype: Optional[torch.dtype] = torch.bfloat16, time_encoding: bool = True,
                 n_time_freqs: int = 16):
        super().__init__()
        cfg = model_config(model_name)
        if cfg["head"]["type"] != "dualdpt":
            raise ValueError(f"{model_name} is not a multi-view DA3 model")
        self.model_name = model_name
        self.dinov2 = build_backbone(model_name)
        self.cam_enc = build_cam_enc(model_name)
        self.load_report = None
        if load_weights:
            if state_dict is None:
                state_dict, _ = read_weights(weights or weights_path(model_name))
            load_strict(self.dinov2, component(state_dict, "backbone"))
            load_strict(self.cam_enc, component(state_dict, "cam_enc"))
            # the head is loaded by DepthHead; cam_dec and the auxiliary ray branch are dropped on purpose
            self.load_report = account(state_dict, ["backbone", "cam_enc", "head"], ["cam_dec.", "_aux"])
        vit = self.vit
        self.embed_dim = vit.embed_dim
        self.depth = len(vit.blocks)
        self.alt_start = vit.alt_start
        self.out_layers = list(cfg["net"]["out_layers"])
        self.time_enc = TimeEncoding(self.embed_dim, n_time_freqs) if time_encoding else None
        self.pose_scale = float(pose_scale)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.amp_dtype = amp_dtype
        self.set_freeze(freeze)

    @property
    def vit(self):
        return self.dinov2.pretrained

    # ------------------------------------------------------------------------------------------------------
    # freezing
    # ------------------------------------------------------------------------------------------------------
    def set_freeze(self, preset: str = "default", *, blocks: Optional[Iterable[int]] = None,
                   cam_enc: Optional[bool] = None, time_enc: Optional[bool] = None) -> None:
        """Apply a preset; ``blocks`` / ``cam_enc`` / ``time_enc`` override its parts."""
        if preset == "default":
            b, c, t = range(self.alt_start, self.depth), True, True
        elif preset == "late":
            b, c, t = range(self.depth - self.depth // 4, self.depth), False, False
        elif preset == "frozen":
            b, c, t = (), False, False
        else:
            raise ValueError(f"freeze preset must be one of {FREEZE_PRESETS}, got {preset!r}")
        b = sorted(set(blocks if blocks is not None else b))
        c = c if cam_enc is None else cam_enc
        t = t if time_enc is None else time_enc
        if any(i < 0 or i >= self.depth for i in b):
            raise ValueError(f"block index out of range 0..{self.depth - 1}: {b}")
        freeze_(self)
        for i in b:
            self.vit.blocks[i].requires_grad_(True)
        self.cam_enc.requires_grad_(c)
        if self.time_enc is not None:
            self.time_enc.requires_grad_(t)
        self.freeze_preset = preset
        self.trainable_blocks = b
        self.train_cam_enc = bool(c)
        self.train_time_enc = bool(t and self.time_enc is not None)

    def first_trainable_layer(self) -> int:
        """Index of the first block whose input can carry a gradient (``depth`` if none can)."""
        k = min(self.trainable_blocks) if self.trainable_blocks else self.depth
        if self.train_cam_enc or self.train_time_enc:
            k = min(k, self.alt_start)
        return k

    # ------------------------------------------------------------------------------------------------------
    # forward pieces (also used by the frozen reference in stage1.py)
    # ------------------------------------------------------------------------------------------------------
    def _autocast(self, device):
        return torch.autocast(device_type=device.type, dtype=self.amp_dtype or torch.float32,
                              enabled=self.amp_dtype is not None)

    def camera_tokens(self, ref_T_cam, K, dt, image_hw, cam_enc: Optional[nn.Module] = None,
                      use_time: bool = True) -> torch.Tensor:
        """(B, N, C) camera tokens in float32: DA3 camera encoder (+ time encoding)."""
        cam_enc = cam_enc or self.cam_enc
        with torch.autocast(device_type=ref_T_cam.device.type, enabled=False):
            tok = cam_enc(da3_extrinsics(ref_T_cam, self.pose_scale), K.float(), image_hw)
            if use_time and self.time_enc is not None and dt is not None:
                tok = tok + self.time_enc(dt)
        return tok

    def prepare(self, images: torch.Tensor) -> dict:
        """Patch tokens + positional embedding, and the RoPE positions; the state the block loop starts from."""
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"images must be (B, N, 3, H, W), got {tuple(images.shape)}")
        B, N, _, H, W = images.shape
        if H % PATCH_SIZE or W % PATCH_SIZE:
            raise ValueError(f"H and W must be multiples of {PATCH_SIZE}, got {H}x{W}")
        x = normalize_images(images)
        vit = self.vit
        with self._autocast(x.device):
            tokens = vit.prepare_tokens_with_masks(x)
        pos, pos_nodiff = vit._prepare_rope(B, N, H, W, x.device)
        return {"x": tokens, "local_x": None, "pos": pos, "pos_nodiff": pos_nodiff, "outs": {},
                "image_hw": (H, W)}

    def _block(self, x, blk, kind, pos):
        fn = self.vit.process_attention
        if (self.grad_checkpoint and torch.is_grad_enabled()
                and (x.requires_grad or any(p.requires_grad for p in blk.parameters()))):
            return checkpoint(fn, x, blk, kind, pos, use_reentrant=False)
        return fn(x, blk, kind, pos)

    def run_blocks(self, state: dict, start: int, end: int, blocks=None,
                   cam_token: Optional[torch.Tensor] = None) -> dict:
        """Blocks ``start``..``end - 1`` as in DA3's ``_get_intermediate_layers_not_chunked`` (camera-token path).

        ``blocks`` maps a layer index to a block (default: this backbone's own blocks). Returns a new state;
        the input state is not modified, so two branches can continue from the same state.
        """
        vit = self.vit
        blocks = vit.blocks if blocks is None else blocks
        x, local_x, outs = state["x"], state["local_x"], dict(state["outs"])
        with self._autocast(x.device):
            for i in range(start, end):
                if i < vit.rope_start or vit.rope is None:
                    g_pos, l_pos = None, None
                else:
                    g_pos, l_pos = state["pos_nodiff"], state["pos"]
                if self.alt_start != -1 and i == self.alt_start:
                    if cam_token is None:
                        raise ValueError("camera tokens are needed from layer alt_start on")
                    # DA3 writes x[:, :, 0] = cam_token in place; a concatenation gives the same values and
                    # keeps the shared state intact for the frozen reference branch.
                    x = torch.cat([cam_token[:, :, None].to(x.dtype), x[:, :, 1:]], dim=2)
                if self.alt_start != -1 and i >= self.alt_start and i % 2 == 1:
                    x = self._block(x, blocks[i], "global", g_pos)
                else:
                    x = self._block(x, blocks[i], "local", l_pos)
                    local_x = x
                if i in self.out_layers:
                    outs[i] = torch.cat([local_x, x], dim=-1) if vit.cat_token else x
        return {**state, "x": x, "local_x": local_x, "outs": outs}

    def finalize(self, state: dict) -> dict:
        """Apply the final LayerNorm to the global half and drop the camera token (``get_intermediate_layers``)."""
        vit = self.vit
        C = self.embed_dim
        H, W = state["image_hw"]
        feats, cams = [], []
        with self._autocast(state["x"].device):
            for i in self.out_layers:
                o = state["outs"][i]
                cams.append(o[:, :, 0])
                if o.shape[-1] == C:
                    f = vit.norm(o)
                else:
                    f = torch.cat([o[..., :C], vit.norm(o[..., C:])], dim=-1)
                feats.append(f[:, :, 1 + vit.num_register_tokens:])
        return {"feats": feats, "cam_tokens": cams, "patch_hw": (H // PATCH_SIZE, W // PATCH_SIZE),
                "image_hw": (H, W)}

    def forward(self, images, ref_T_cam, K, dt=None) -> dict:
        state = self.prepare(images)
        cam = self.camera_tokens(ref_T_cam, K, dt, state["image_hw"])
        state = self.run_blocks(state, 0, self.depth, cam_token=cam)
        return self.finalize(state)

    def forward_without_poses(self, images, ref_view_strategy: str = "saddle_balanced") -> dict:
        """DA3's own pose-free path (learned camera tokens; views may be reordered internally and restored)."""
        x = normalize_images(images)
        with self._autocast(x.device):
            out, _ = self.dinov2(x, cam_token=None, export_feat_layers=[], ref_view_strategy=ref_view_strategy)
        H, W = x.shape[-2:]
        return {"feats": [o[0] for o in out], "cam_tokens": [o[1] for o in out],
                "patch_hw": (H // PATCH_SIZE, W // PATCH_SIZE), "image_hw": (H, W)}


def feats_to_maps(feats, patch_hw):
    """(B, N, h*w, D) token tensors to (B, N, D, h, w) feature maps."""
    h, w = patch_hw
    return [f.unflatten(2, (h, w)).permute(0, 1, 4, 2, 3) for f in feats]
