"""Configuration files: YAML, an optional parent file, command-line overrides, environment variables.

    base: other.yaml            # optional; path relative to this file; this file overrides it
    task: stage1_backbone
    run_dir: ${RV_RUNS}/stage1  # ${NAME} is replaced from the environment when the file is loaded

Overrides are "dotted.key=value" strings; the value is parsed as YAML, so numbers, booleans, lists and
null work as written: optimizer.weight_decay=0.1  model.unfreeze_from=18  data.input_offsets=[0,-10]
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

_UNSET = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def merge(base: dict, override: dict) -> dict:
    """`override` on top of `base`; dictionaries are merged key by key, everything else is replaced."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def set_dotted(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def apply_overrides(cfg: dict, overrides) -> dict:
    out = merge({}, cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got: {item}")
        key, text = item.split("=", 1)
        set_dotted(out, key.strip(), yaml.safe_load(text))
    return out


def expand_env(value):
    """Replace ${NAME} in every string; a name missing from the environment is an error."""
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, str) and "${" in value:
        missing = [name for name in _UNSET.findall(value) if name not in os.environ]
        if missing:
            raise KeyError(f"environment variable(s) {', '.join(missing)} not set, needed by: {value}")
        return _UNSET.sub(lambda m: os.environ[m.group(1)], value)
    return value


def read_yaml(path) -> dict:
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: the top level must be a mapping")
    parent = cfg.pop("base", None)
    if parent:
        cfg = merge(read_yaml(path.parent / parent), cfg)
    return cfg


def load_config(path, overrides=None) -> dict:
    """Read a file (with its parents), apply overrides, then fill in environment variables."""
    return expand_env(apply_overrides(read_yaml(path), overrides))
