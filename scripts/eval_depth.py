#!/usr/bin/env python3
"""Depth errors of a backbone against LiDAR, on the fixed evaluation moments of a configuration.

    # the original pretrained network (the baseline)
    python scripts/eval_depth.py --config configs/stage1_backbone.yaml --out baseline.json
    # a trained one
    python scripts/eval_depth.py --config configs/stage1_backbone.yaml --checkpoint <run>/last.pt --out trained.json

--set works as in scripts/train.py, e.g. --set model.pose_scale=6.7 data.eval_per_log=4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from realityview.engine import load_config  # noqa: E402
from realityview.engine.checkpoint import load_checkpoint  # noqa: E402
from realityview.tasks import build_task  # noqa: E402

ROWS = (("single", "LiDAR single sweep"), ("single_aligned", "  same, scale-aligned per sample"),
        ("single_moving", "  car moving"), ("single_standing", "  car standing"),
        ("acc", "LiDAR accumulated"), ("orig_single", "original network, single sweep"),
        ("orig_single_aligned", "  same, scale-aligned per sample"), ("orig_single_moving", "  car moving"),
        ("orig_single_standing", "  car standing"), ("orig_acc", "original network, accumulated"))


def report(metrics: dict) -> str:
    lines = [f"samples {metrics.get('samples', 0):.0f}    kappa {metrics.get('kappa', float('nan')):.4f}", "",
             f"{'':36s}{'AbsRel':>9s}{'d<1.25':>9s}{'RMSE log':>10s}"]
    for key, label in ROWS:
        if f"absrel_{key}" in metrics:
            lines.append(f"{label:36s}{metrics[f'absrel_{key}']:9.4f}{metrics[f'delta1_{key}']:9.4f}"
                         f"{metrics[f'rmse_log_{key}']:10.4f}")
    for key, label in (("single", "this network"), ("orig_single", "original network")):
        parts = sorted((k.split("/", 1)[1], v) for k, v in metrics.items() if k.startswith(f"absrel_{key}/"))
        if parts:
            bands = [(n, v) for n, v in parts if n[0].isdigit()]
            bands.sort(key=lambda item: int(item[0].split("-")[0]))
            cams = [(n, v) for n, v in parts if not n[0].isdigit()]
            lines += ["", f"AbsRel on the single sweep, {label}:",
                      "  by distance  " + "  ".join(f"{n} {v:.4f}" for n, v in bands),
                      "  by camera    " + "  ".join(f"{n.replace('ring_', '')} {v:.4f}" for n, v in cams)]
    others = [(k, v) for k, v in sorted(metrics.items())
              if "/" not in k and not k.startswith(("absrel_", "delta1_", "rmse_log_", "pixels_")) and k not in ("samples", "kappa")]
    if others:
        lines += ["", "other numbers:"] + [f"  {k:28s}{v:10.4f}" for k, v in others]
    scales = sorted((k.split("/", 1)[1], v) for k, v in metrics.items() if k.startswith("scale/"))
    if scales:
        lines += ["", "median of LiDAR / prediction per camera (should be close to each other):",
                  "  " + "  ".join(f"{n.replace('ring_', '')} {v:.3f}" for n, v in scales)]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="a checkpoint written by scripts/train.py; default: the pretrained weights")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", default=None, help="write all numbers to this JSON file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    task = build_task(cfg)
    model = task.build_model()
    step = None
    if args.checkpoint:
        step = load_checkpoint(args.checkpoint, model, None, getattr(task, "checkpoint_skip", ()))["step"]
    device = torch.device(args.device)
    model.to(device).eval()
    dataset = task.build_eval_dataset()
    if dataset is None:
        parser.error("the configuration has no evaluation logs (data.eval_logs)")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers,
                        collate_fn=getattr(task, "collate", None),
                        multiprocessing_context="spawn" if args.workers > 0 else None)
    with torch.no_grad():
        metrics = task.evaluate(model, loader, device)
    print(f"checkpoint: {args.checkpoint or 'none (pretrained weights)'}" + (f", step {step}" if step is not None else ""))
    print(report(metrics))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"checkpoint": args.checkpoint, "step": step, "config": cfg, "metrics": metrics}, indent=1) + "\n")
        print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
