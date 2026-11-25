"""What can be trained. A task tells the training loop which model, data and losses to use.

The interface is described at the top of realityview/engine/trainer.py.
"""
from __future__ import annotations

import importlib

TASKS = {
    "toy": "realityview.tasks.toy:ToyTask",
}


def build_task(cfg: dict):
    name = cfg.get("task")
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; known tasks: {', '.join(sorted(TASKS))}")
    module, attr = TASKS[name].split(":")
    return getattr(importlib.import_module(module), attr)(cfg)
