"""Checkpoints: the model, the optimizer and the step, written atomically."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import torch


def code_version() -> str:
    """The git commit of the code that is running, or "unknown" outside a git checkout."""
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def save_checkpoint(path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, cfg: dict,
                    extra: dict | None = None) -> None:
    """Parameters that never train (requires_grad False) and buffers are saved too: a checkpoint loads on its
    own, without the pretrained weights it started from."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "cfg": cfg,
             "code_version": code_version(), **(extra or {})}
    torch.save(state, tmp)
    os.replace(tmp, path)       # a crash while writing never leaves a half-written file under the final name


def load_checkpoint(path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state


def find_resume(setting: str, run_dir) -> Path | None:
    """"auto": run_dir/last.pt if it exists; "none": never; anything else: that file, which must exist."""
    if setting == "none":
        return None
    if setting == "auto":
        last = Path(run_dir) / "last.pt"
        return last if last.is_file() else None
    path = Path(setting)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint to resume from not found: {path}")
    return path
