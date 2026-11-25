"""Training loop shared by every stage: configuration, multi-GPU setup, sampling, checkpoints."""
from .config import apply_overrides, load_config, merge
from .trainer import Trainer

__all__ = ["Trainer", "apply_overrides", "load_config", "merge"]
