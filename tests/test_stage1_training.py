"""Backbone training on real processed data with the small backbone.

Needs a GPU, RV_WEIGHTS_ROOT (with da3-base/) and RV_PROCESSED_ROOT holding the test log with sky scores
(RV_TEST_LOG / RV_TEST_SPLIT, default: the first log of data_lists/av2_val_report20.txt); skipped otherwise.

On a GPU a few operations (bilinear upsampling in the depth head, half-precision matrix products) do not
give bit-identical results from run to run, so runs are compared with a tolerance that is first measured:
two uninterrupted runs must agree with each other as well as an interrupted run agrees with them. The exact
(bit-identical) version of the same test runs on the CPU in test_engine.py.
"""
import contextlib
import gc
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

import _util
import torch

from realityview.engine import Trainer, merge
from realityview.tasks import build_task

STEPS = 6


def setting():
    weights = os.environ.get("RV_WEIGHTS_ROOT")
    root = os.environ.get("RV_PROCESSED_ROOT")
    if not weights or not root:
        raise unittest.SkipTest("RV_WEIGHTS_ROOT and RV_PROCESSED_ROOT not set")
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    if not os.path.isfile(os.path.join(weights, "da3-base", "model.safetensors")):
        raise unittest.SkipTest("da3-base weights missing")
    split = os.environ.get("RV_TEST_SPLIT", "val")
    log_id = os.environ.get("RV_TEST_LOG")
    if not log_id:
        with open(os.path.join(_util.REPO, "data_lists", "av2_val_report20.txt")) as f:
            log_id = f.readline().strip()
    if not os.path.isfile(os.path.join(root, "logs", split, log_id, "sky", "DONE")):
        raise unittest.SkipTest(f"{split}/{log_id} has no sky scores under RV_PROCESSED_ROOT")
    return {
        "task": "stage1_backbone", "seed": 5, "total_steps": STEPS, "num_workers": 2, "deterministic": True,
        "schedule": {"warmup_steps": 2, "min_lr": 1.0e-6}, "log_every": 1, "checkpoint_every": 100, "eval_every": 0,
        "model": {"name": "da3-base", "weights_root": weights},
        "data": {"root": root, "train_split": split, "train_logs": [log_id], "train_first": None,
                 "eval_split": split, "eval_logs": [log_id], "eval_first": None, "eval_per_log": 2},
    }


def run(cfg, run_dir, **changes):
    cfg = merge(cfg, {"run_dir": run_dir, **changes})
    with contextlib.redirect_stdout(io.StringIO()):
        result = Trainer(build_task(cfg), cfg).fit()
    gc.collect()                    # give the GPU memory back before the next run
    torch.cuda.empty_cache()
    return result


def losses(run_dir):
    with open(os.path.join(run_dir, "train_log.jsonl")) as f:
        return [json.loads(line)["loss"] for line in f]


def worst(a, b):
    return max(abs(x - y) / max(abs(x), 1e-9) for x, y in zip(a, b))


class TestStage1Training(unittest.TestCase):
    def test_interrupted_and_distributed_runs_match_an_uninterrupted_one(self):
        cfg = setting()
        with tempfile.TemporaryDirectory() as tmp:
            straight, again, resumed, ddp = (os.path.join(tmp, n) for n in ("straight", "again", "resumed", "ddp"))
            a = run(cfg, straight)
            b = run(cfg, again)
            first = run(cfg, resumed, stop_at_step=3)
            self.assertEqual((first["finished"], first["step"]), (False, 3))
            c = run(cfg, resumed, stop_at_step=3)
            self.assertTrue(a["finished"] and b["finished"] and c["finished"])
            self.assertEqual(len(losses(resumed)), STEPS)

            # the multi-GPU code path with a single process, started the way a cluster job starts it
            command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=1",
                       os.path.join(_util.REPO, "scripts", "train.py"), "--config",
                       os.path.join(_util.REPO, "configs", "stage1_overfit_base.yaml"), "--set", f"run_dir={ddp}"]
            flat = {"seed": cfg["seed"], "total_steps": STEPS, "num_workers": 2, "deterministic": True,
                    "schedule.warmup_steps": 2, "log_every": 1, "checkpoint_every": 100, "eval_every": 0,
                    "eval_at_start": False, "keep_every": 0,
                    "model.weights_root": cfg["model"]["weights_root"], **{f"data.{k}": v for k, v in cfg["data"].items()}}
            command += [f"{k}={json.dumps(v)}" for k, v in flat.items()]
            done = subprocess.run(command, capture_output=True, text=True, timeout=900,
                                  env=dict(os.environ, PYTHONNOUSERSITE="1"))
            self.assertEqual(done.returncode, 0, done.stdout[-3000:] + done.stderr[-3000:])
            self.assertIn("distributed", done.stdout)
            with open(os.path.join(ddp, "done.json")) as f:
                d = json.load(f)

            noise = worst(losses(straight), losses(again))
            gap_resumed = worst(losses(straight), losses(resumed))
            gap_ddp = worst(losses(straight), losses(ddp))
            metric = "absrel_single"
            print(f"\n[stage-1 training, DA3-BASE, {STEPS} steps] losses per step, uninterrupted: "
                  + " ".join(f"{x:.4f}" for x in losses(straight)))
            print(f"  largest relative difference of a step's loss: two uninterrupted runs {noise:.2e}, "
                  f"interrupted at step 3 and continued {gap_resumed:.2e}, distributed path {gap_ddp:.2e}")
            print(f"  {metric} after training: uninterrupted {a[metric]:.5f} / {b[metric]:.5f}, "
                  f"interrupted and continued {c[metric]:.5f}, distributed path {d[metric]:.5f}; "
                  f"original network {a['absrel_orig_single']:.5f}")
            tolerance = max(5 * noise, 2e-3)
            self.assertLess(gap_resumed, tolerance)
            self.assertLess(gap_ddp, tolerance)
            self.assertLess(abs(c[metric] - a[metric]), max(5 * abs(b[metric] - a[metric]), 2e-3))
            self.assertLess(abs(d[metric] - a[metric]), max(5 * abs(b[metric] - a[metric]), 2e-3))
            self.assertTrue(all(x == x and abs(x) < 1e3 for x in losses(straight)))      # finite


if __name__ == "__main__":
    unittest.main()
