#!/usr/bin/env python3
"""Train a task from a configuration file.

    python scripts/train.py --config configs/toy.yaml --set run_dir=runs/toy total_steps=200
    torchrun --nproc_per_node=8 scripts/train.py --config configs/stage1_backbone.yaml --set run_dir=...

A run that stopped early (time limit, STOP file, signal) continues when the same command is run again.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from realityview.engine import Trainer, load_config  # noqa: E402
from realityview.tasks import build_task  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML file, see configs/")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override configuration values, e.g. total_steps=200 optimizer.weight_decay=0.1")
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    if not cfg.get("run_dir"):
        parser.error("run_dir is not set: add it to the configuration or pass --set run_dir=<directory>")
    trainer = Trainer(build_task(cfg), cfg)
    trainer.install_signal_handlers()
    result = trainer.fit()
    if trainer.info.is_main:
        print("result: " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
