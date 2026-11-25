"""Head training on real processed data with the small backbone: a few steps, an evaluation, a checkpoint
that can be loaded again.

Needs a GPU, RV_WEIGHTS_ROOT (with da3-base/) and RV_PROCESSED_ROOT holding the test log with sky scores
(RV_TEST_LOG / RV_TEST_SPLIT, default: the first log of data_lists/av2_val_report20.txt); skipped otherwise.
LPIPS is part of the run only when torchvision's VGG weights are already in the torch hub cache (the test
does not download them).
"""
import contextlib
import gc
import io
import json
import os
import tempfile
import unittest

import _util
import torch

from realityview.engine import Trainer, merge
from realityview.tasks import build_task
from realityview.tasks.stage1_head import load_trained_model


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
    vgg = os.path.join(torch.hub.get_dir(), "checkpoints", "vgg16-397923af.pth")
    lpips_weight = 0.05 if os.path.isfile(vgg) else 0.0
    return {
        "task": "stage1_head", "seed": 3, "total_steps": 3, "num_workers": 2,
        "schedule": {"warmup_steps": 1, "min_lr": 1.0e-6}, "log_every": 1, "checkpoint_every": 100, "eval_every": 0,
        "model": {"name": "da3-base", "weights_root": weights, "head_checkpoint": True, "exposure_delay_steps": 1},
        "data": {"root": root, "train_split": split, "train_logs": [log_id], "train_first": None,
                 "eval_split": split, "eval_logs": [log_id], "eval_first": None, "eval_per_log": 1},
        "loss": {"lpips": {"weight": lpips_weight, "ramp_steps": 2, "max_views": 2}},
        "eval": {"panels": 2},
    }


class TestStage1HeadTraining(unittest.TestCase):
    def test_a_short_run(self):
        cfg = setting()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = merge(cfg, {"run_dir": tmp})
            with contextlib.redirect_stdout(io.StringIO()):
                result = Trainer(build_task(cfg), cfg).fit()
            self.assertTrue(result["finished"])
            self.assertEqual(result["step"], 3)
            with open(os.path.join(tmp, "train_log.jsonl")) as f:
                logs = [json.loads(line) for line in f]
            self.assertEqual(len(logs), 3)
            for name in ("loss_lidar", "loss_sky", "loss_photo", "loss_render_depth", "psnr", "share_compared"):
                self.assertIn(name, logs[0])
            self.assertEqual(logs[0]["lr_exposure"], 0.0)            # the exposure correction starts later
            self.assertGreater(logs[-1]["lr_exposure"], 0.0)
            self.assertGreater(logs[0]["share_compared"], 0.3)
            # the evaluation: one moment, targets t, t + 0.5 s and t + 1 s
            self.assertEqual(result["samples"], 3.0)
            for name in ("t", "p10", "p20"):
                self.assertIn(f"psnr_{name}", result)
                self.assertGreater(result[f"compared_{name}"], 0.2)
            self.assertEqual(result["compared_t"], 1.0)              # at t everything outside the ego body
            self.assertGreater(result["psnr_t"], result["psnr_p20"])
            images = os.listdir(os.path.join(tmp, "eval_images", "step_000003"))
            self.assertEqual(len(images), 2)
            # the checkpoint holds the whole model and can be rebuilt without the run's paths
            model, task = load_trained_model(os.path.join(tmp, "last.pt"))
            self.assertEqual(task.step, 3)
            self.assertIsNone(model.backbone_checkpoint)
            state = torch.load(os.path.join(tmp, "last.pt"), map_location="cpu", weights_only=False)
            self.assertTrue(torch.equal(model.appearance.net[0].weight, state["model"]["appearance.net.0.weight"]))
            del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
