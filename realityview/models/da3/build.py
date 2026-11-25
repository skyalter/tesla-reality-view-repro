"""Build Depth Anything 3 networks by model name and load their weights strictly.

This replaces ``depth_anything_3/model/da3.py`` + ``cfg.py`` + ``api.py`` for our use: no omegaconf, no
Hugging Face hub, no ``torch.inference_mode``. The architecture parameters below are copied from
``src/depth_anything_3/configs/<name>.yaml`` (commit 3d835ec) and are checked against the ``config.json``
that ships with each weight file (``check_config``).

Weight files (``model.safetensors``) name every tensor ``model.<component>.<...>``; the components are
``backbone`` (DinoV2 wrapper, tensors under ``backbone.pretrained.``), ``cam_enc``, ``cam_dec`` and ``head``.
Loading is strict: every tensor in the file must be consumed by a module or listed as deliberately dropped
(``cam_dec`` and the auxiliary ray branch ``head.*_aux*``), and every module parameter must come from the file.

The DA3 files store six LayerNorm parameters of the auxiliary branch as aliases of
``head.scratch.output_conv2_aux.0.2.*`` (listed in the safetensors metadata, no data of their own). They are
only needed when the full original DualDPT (with the auxiliary branch) is built; ``fill_aliases`` copies them.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from safetensors import safe_open

from .cam_dec import CameraDec
from .cam_enc import CameraEnc
from .dinov2.dinov2 import DinoV2
from .dpt import DPT
from .dualdpt import DualDPT

PATCH_SIZE = 14
WEIGHTS_ENV = "RV_WEIGHTS_ROOT"

# Copied from src/depth_anything_3/configs/{da3-base,da3-large,da3metric-large}.yaml.
MODEL_CONFIGS: Dict[str, dict] = {
    "da3-base": {
        "net": dict(name="vitb", out_layers=[5, 7, 9, 11], alt_start=4, qknorm_start=4, rope_start=4,
                    cat_token=True),
        "head": dict(type="dualdpt", dim_in=1536, output_dim=2, features=128, out_channels=[96, 192, 384, 768]),
        "cam_enc": dict(dim_out=768),
        "cam_dec": dict(dim_in=1536),
    },
    "da3-large-1.1": {  # same yaml as da3-large; the 1.1 weights were retrained, the structure is unchanged
        "net": dict(name="vitl", out_layers=[11, 15, 19, 23], alt_start=8, qknorm_start=8, rope_start=8,
                    cat_token=True),
        "head": dict(type="dualdpt", dim_in=2048, output_dim=2, features=256,
                     out_channels=[256, 512, 1024, 1024]),
        "cam_enc": dict(dim_out=1024),
        "cam_dec": dict(dim_in=2048),
    },
    "da3metric-large": {
        "net": dict(name="vitl", out_layers=[4, 11, 17, 23], alt_start=-1, qknorm_start=-1, rope_start=-1,
                    cat_token=False),
        "head": dict(type="dpt", dim_in=1024, output_dim=1, features=256, out_channels=[256, 512, 1024, 1024]),
    },
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def model_config(name: str) -> dict:
    if name not in MODEL_CONFIGS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODEL_CONFIGS)}")
    return MODEL_CONFIGS[name]


def weights_path(name: str, root: Optional[str] = None) -> str:
    """``<root>/<name>/model.safetensors``; ``root`` defaults to the environment variable RV_WEIGHTS_ROOT."""
    root = root or os.environ.get(WEIGHTS_ENV)
    if not root:
        raise FileNotFoundError(f"no weights root given and {WEIGHTS_ENV} is not set")
    return os.path.join(root, name, "model.safetensors")


def check_config(name: str, config_json_path: str) -> None:
    """Raise if the ``config.json`` shipped with the weights differs from ``MODEL_CONFIGS[name]``."""
    with open(config_json_path) as f:
        cfg = json.load(f)["config"]
    ours = model_config(name)
    for part in ("net", "head", "cam_enc", "cam_dec"):
        theirs = cfg.get(part)
        if part not in ours:
            if theirs is not None:
                raise ValueError(f"{name}: config.json has a {part} that we do not build")
            continue
        theirs = {k: v for k, v in theirs.items() if k != "__object__"}
        mine = {k: v for k, v in ours[part].items() if k != "type"}
        if theirs != mine:
            raise ValueError(f"{name}: {part} differs: config.json {theirs} vs ours {mine}")
    head_cls = cfg["head"]["__object__"]["name"]
    if head_cls.lower() != ours["head"]["type"]:
        raise ValueError(f"{name}: head class {head_cls} vs ours {ours['head']['type']}")


# ----------------------------------------------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------------------------------------------
def build_backbone(name: str) -> DinoV2:
    return DinoV2(**model_config(name)["net"])


def build_cam_enc(name: str) -> CameraEnc:
    return CameraEnc(**model_config(name)["cam_enc"])


def build_cam_dec(name: str) -> CameraDec:
    return CameraDec(**model_config(name)["cam_dec"])


def build_original_head(name: str) -> nn.Module:
    """The head exactly as DA3 builds it (DualDPT with its auxiliary ray branch, or DPT with the sky branch)."""
    h = dict(model_config(name)["head"])
    kind = h.pop("type")
    return DualDPT(**h) if kind == "dualdpt" else DPT(**h)


# ----------------------------------------------------------------------------------------------------------
# weights
# ----------------------------------------------------------------------------------------------------------
def read_weights(path: str, device: str | torch.device = "cpu") -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """All tensors with the leading ``model.`` removed, and the alias map from the file metadata."""
    sd: Dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device=str(device)) as f:
        meta = f.metadata() or {}
        for k in f.keys():
            if not k.startswith("model."):
                raise ValueError(f"unexpected tensor name {k!r} in {path}")
            sd[k[len("model."):]] = f.get_tensor(k)
    aliases = {}
    for k, v in meta.items():
        if k.startswith("model.") and isinstance(v, str) and v.startswith("model."):
            aliases[k[len("model."):]] = v[len("model."):]
    return sd, aliases


def fill_aliases(sd: Dict[str, torch.Tensor], aliases: Dict[str, str]) -> Dict[str, torch.Tensor]:
    """Return a copy of ``sd`` where every aliased name that has no tensor of its own gets its target's tensor."""
    out = dict(sd)
    for k, target in aliases.items():
        if k not in out:
            out[k] = out[target].clone()
    return out


def component(sd: Dict[str, torch.Tensor], prefix: str, drop: Iterable[str] = ()) -> Dict[str, torch.Tensor]:
    """Tensors under ``prefix.`` with the prefix removed, minus names containing any of ``drop``."""
    p = prefix + "."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p) and not any(d in k for d in drop)}


def load_strict(module: nn.Module, sd: Dict[str, torch.Tensor]) -> None:
    res = module.load_state_dict(sd, strict=True)
    assert not res.missing_keys and not res.unexpected_keys, res


def account(sd: Dict[str, torch.Tensor], used_prefixes: Iterable[str], dropped: Iterable[str]) -> dict:
    """Check that every tensor in ``sd`` is either used (by prefix) or deliberately dropped.

    ``dropped`` are substrings / prefixes (``cam_dec.``, ``_aux``). Returns counts; raises on a leftover tensor.
    """
    used_prefixes, dropped = tuple(used_prefixes), tuple(dropped)
    n_used = n_dropped = 0
    left = []
    for k in sd:
        if any(d in k for d in dropped):
            n_dropped += 1
        elif any(k.startswith(p + ".") for p in used_prefixes):
            n_used += 1
        else:
            left.append(k)
    if left:
        raise ValueError(f"{len(left)} tensors neither loaded nor dropped, e.g. {left[:5]}")
    return {"file": len(sd), "used": n_used, "dropped": n_dropped}


def load_backbone_parts(name: str, path: str, with_cam_dec: bool = False):
    """Backbone, camera encoder (and optionally camera decoder) of a multi-view DA3 model, strictly loaded.

    Returns ``(backbone, cam_enc, cam_dec_or_None, head_main_sd, report)``. ``head_main_sd`` is the state dict of
    the DualDPT main branch (auxiliary tensors removed), for ``realityview.models.depth_head.DepthHead``.
    """
    sd, _ = read_weights(path)
    backbone, cam_enc = build_backbone(name), build_cam_enc(name)
    load_strict(backbone, component(sd, "backbone"))
    load_strict(cam_enc, component(sd, "cam_enc"))
    cam_dec = None
    dropped = ["_aux"]
    if with_cam_dec:
        cam_dec = build_cam_dec(name)
        load_strict(cam_dec, component(sd, "cam_dec"))
    else:
        dropped.append("cam_dec.")
    used = ["backbone", "cam_enc", "head"] + (["cam_dec"] if with_cam_dec else [])
    report = account(sd, used, dropped)
    head_sd = component(sd, "head", drop=("_aux",))
    return backbone, cam_enc, cam_dec, head_sd, report


# ----------------------------------------------------------------------------------------------------------
# the original model, assembled from the copied code (reference for tests)
# ----------------------------------------------------------------------------------------------------------
class DA3Original(nn.Module):
    """The DA3 network as ``DepthAnything3Net.forward`` (``model/da3.py`` 100-153) runs it, depth part only.

    Built from the copied, unmodified DinoV2 / CameraEnc / DualDPT code; the head keeps its auxiliary branch.
    ``forward`` takes ImageNet-normalised images (B, N, 3, H, W), optional world-to-camera extrinsics
    (B, N, 4, 4) and intrinsics (B, N, 3, 3), and returns ``{"depth", "depth_conf"}`` (B, N, H, W).
    The official API wraps the call in ``torch.no_grad()`` and ``torch.autocast(bfloat16)``; callers that
    want to reproduce it do the same. Only used as the reference in tests.
    """

    def __init__(self, name: str, path: str, with_cam_dec: bool = False):
        super().__init__()
        cfg = model_config(name)
        if cfg["head"]["type"] != "dualdpt":
            raise ValueError("DA3Original is for the multi-view models")
        sd, aliases = read_weights(path)
        sd = fill_aliases(sd, aliases)
        self.backbone = build_backbone(name)
        self.cam_enc = build_cam_enc(name)
        self.head = build_original_head(name)
        load_strict(self.backbone, component(sd, "backbone"))
        load_strict(self.cam_enc, component(sd, "cam_enc"))
        load_strict(self.head, component(sd, "head"))
        self.cam_dec = None
        if with_cam_dec:
            self.cam_dec = build_cam_dec(name)
            load_strict(self.cam_dec, component(sd, "cam_dec"))
        account(sd, ["backbone", "cam_enc", "head"] + (["cam_dec"] if with_cam_dec else []),
                [] if with_cam_dec else ["cam_dec."])
        self.eval()

    def forward(self, x, extrinsics=None, intrinsics=None, ref_view_strategy="saddle_balanced"):
        if extrinsics is not None:
            with torch.autocast(device_type=x.device.type, enabled=False):
                cam_token = self.cam_enc(extrinsics, intrinsics, x.shape[-2:])
        else:
            cam_token = None
        feats, _ = self.backbone(x, cam_token=cam_token, export_feat_layers=[],
                                 ref_view_strategy=ref_view_strategy)
        H, W = x.shape[-2], x.shape[-1]
        with torch.autocast(device_type=x.device.type, enabled=False):
            out = self.head(feats, H, W, patch_start_idx=0)
            if self.cam_dec is not None:
                from .utils.geometry import affine_inverse
                from .utils.transform import pose_encoding_to_extri_intri

                pose_enc = self.cam_dec(feats[-1][1])
                c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H, W))
                out["extrinsics"] = affine_inverse(c2w)  # (B, N, 3, 4) world-to-camera, as da3.py stores it
                out["intrinsics"] = ixt
        out["feats"] = feats
        return out


def freeze_(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def frozen_copy(module: nn.Module) -> nn.Module:
    return freeze_(copy.deepcopy(module))
