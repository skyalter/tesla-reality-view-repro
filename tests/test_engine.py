"""The training loop on a toy task: it learns, it resumes exactly, and the multi-GPU code path works."""
from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

from realityview.engine import Trainer, load_config, merge  # noqa: E402
from realityview.engine.sampler import StepBatchSampler, epoch_permutation  # noqa: E402
from realityview.engine.trainer import lr_at, split_param_groups  # noqa: E402
from realityview.tasks.toy import ToyTask  # noqa: E402

BASE = {"task": "toy", "seed": 3, "device": "cpu", "total_steps": 40, "batch_size": 8, "num_workers": 0,
        "schedule": {"warmup_steps": 5, "min_lr": 1.0e-5}, "log_every": 1, "checkpoint_every": 7,
        "deterministic": True, "toy": {"train_size": 96, "eval_size": 32, "width": 16, "lr": 1.0e-2, "dropout": 0.2}}


class SignalAtStep(ToyTask):
    """Sends this process the signal Slurm sends before a time limit, in the middle of a chosen step."""

    def training_step(self, model, batch, step):
        if step == self.cfg["signal_at_step"]:
            os.kill(os.getpid(), signal.SIGUSR1)
        return super().training_step(model, batch, step)


def run(run_dir, task_class=ToyTask, **changes):
    cfg = merge(BASE, {"run_dir": run_dir, **changes})
    with contextlib.redirect_stdout(io.StringIO()):
        trainer = Trainer(task_class(cfg), cfg)
        trainer.install_signal_handlers()
        return trainer.fit()


def weights(run_dir):
    return torch.load(os.path.join(run_dir, "last.pt"), map_location="cpu", weights_only=False)


def log_records(run_dir):
    with open(os.path.join(run_dir, "train_log.jsonl")) as f:
        return [json.loads(line) for line in f]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SamplerTest(unittest.TestCase):
    def test_permutation_depends_on_seed_and_epoch_only(self):
        a = epoch_permutation(50, seed=1, epoch=4)
        self.assertEqual(a.tolist(), epoch_permutation(50, seed=1, epoch=4).tolist())
        self.assertEqual(sorted(a.tolist()), list(range(50)))
        self.assertNotEqual(a.tolist(), epoch_permutation(50, seed=1, epoch=5).tolist())
        self.assertNotEqual(a.tolist(), epoch_permutation(50, seed=2, epoch=4).tolist())

    def test_ranks_share_nothing_within_an_epoch(self):
        samplers = [StepBatchSampler(length=41, batch_size=2, rank=r, world_size=4, seed=0) for r in range(4)]
        per_rank = 41 // 4          # 10 samples per rank and epoch = 5 steps of 2
        seen = [i for s in samplers for step in range(per_rank // 2) for i in s.batch_at(step)]
        self.assertEqual(len(seen), 40)
        self.assertEqual(len(set(seen)), 40)

    def test_iterating_from_a_later_step_gives_the_same_batches(self):
        full = StepBatchSampler(30, 4, rank=1, world_size=2, seed=7)
        late = StepBatchSampler(30, 4, rank=1, world_size=2, seed=7, start_step=9)
        expected = [full.batch_at(s) for s in range(9, 15)]
        iterator = iter(late)
        self.assertEqual([next(iterator) for _ in range(6)], expected)

    def test_too_few_samples_is_an_error(self):
        with self.assertRaises(ValueError):
            StepBatchSampler(3, 1, rank=0, world_size=4, seed=0)


class ScheduleTest(unittest.TestCase):
    def test_delay_warmup_and_cosine(self):
        kwargs = dict(base_lr=1.0, warmup_steps=10, total_steps=110, min_lr=0.01, delay_steps=0)
        self.assertAlmostEqual(lr_at(0, **kwargs), 0.1)
        self.assertAlmostEqual(lr_at(9, **kwargs), 1.0)
        self.assertAlmostEqual(lr_at(10, **kwargs), 1.0)
        self.assertAlmostEqual(lr_at(60, **kwargs), 0.01 + 0.99 * 0.5)
        self.assertAlmostEqual(lr_at(110, **kwargs), 0.01)
        delayed = dict(kwargs, delay_steps=20)
        self.assertEqual(lr_at(19, **delayed), 0.0)
        self.assertAlmostEqual(lr_at(20, **delayed), 0.1)
        self.assertAlmostEqual(lr_at(110, **delayed), 0.01)

    def test_biases_and_norms_are_not_decayed_and_frozen_parameters_are_left_out(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4), torch.nn.Linear(4, 1))
        for p in model[2].parameters():
            p.requires_grad_(False)
        groups = split_param_groups([{"name": "all", "params": list(model.parameters()), "lr": 0.1}], 0.05, True)
        self.assertEqual([(g["weight_decay"], sorted(p.ndim for p in g["params"])) for g in groups],
                         [(0.05, [2]), (0.0, [1, 1, 1])])


class ConfigTest(unittest.TestCase):
    def test_parent_file_overrides_and_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "parent.yaml"), "w") as f:
                f.write("task: toy\noptimizer: {weight_decay: 0.05, betas: [0.9, 0.95]}\nrun_dir: ${RV_TEST_RUNS}/a\n")
            with open(os.path.join(tmp, "child.yaml"), "w") as f:
                f.write("base: parent.yaml\noptimizer: {weight_decay: 0.1}\ntotal_steps: 5\n")
            os.environ["RV_TEST_RUNS"] = "/somewhere"
            try:
                cfg = load_config(os.path.join(tmp, "child.yaml"), ["total_steps=7", "toy.width=8", "data.offsets=[0,-10]"])
            finally:
                del os.environ["RV_TEST_RUNS"]
            self.assertEqual(cfg["optimizer"], {"weight_decay": 0.1, "betas": [0.9, 0.95]})
            self.assertEqual((cfg["total_steps"], cfg["toy"]["width"], cfg["data"]["offsets"]), (7, 8, [0, -10]))
            self.assertEqual(cfg["run_dir"], "/somewhere/a")
            with self.assertRaises(KeyError):
                load_config(os.path.join(tmp, "child.yaml"))        # RV_TEST_RUNS is no longer set


class TrainerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def assert_same_weights(self, a, b):
        wa, wb = weights(self.path(a)), weights(self.path(b))
        self.assertEqual(wa["step"], wb["step"])
        for key in wa["model"]:
            self.assertTrue(torch.equal(wa["model"][key], wb["model"][key]), f"{key} differs between {a} and {b}")

    def test_the_loss_goes_down(self):
        result = run(self.path("plain"), eval_at_start=True, total_steps=400, log_every=50,
                     toy={"lr": 2.0e-2, "dropout": 0.0})
        self.assertTrue(result["finished"])
        with open(self.path("plain/eval_log.jsonl")) as f:
            evals = [json.loads(line) for line in f]
        self.assertEqual([e["step"] for e in evals], [0, 400])
        self.assertLess(evals[1]["mse"], 0.2 * evals[0]["mse"])
        self.assertTrue(os.path.isfile(self.path("plain/done.json")))

    def test_a_run_interrupted_by_a_signal_continues_to_the_same_result(self):
        run(self.path("straight"))
        first = run(self.path("resumed"), SignalAtStep, signal_at_step=17)
        self.assertFalse(first["finished"])
        self.assertEqual(first["step"], 18)                    # the step the signal arrived in was completed
        self.assertFalse(os.path.exists(self.path("resumed/done.json")))
        second = run(self.path("resumed"), SignalAtStep, signal_at_step=-1)
        self.assertTrue(second["finished"])
        self.assert_same_weights("straight", "resumed")
        straight, resumed = log_records(self.path("straight")), log_records(self.path("resumed"))
        self.assertEqual([r["step"] for r in resumed], list(range(1, 41)))
        self.assertEqual([r["loss"] for r in straight], [r["loss"] for r in resumed])
        self.assertEqual(run(self.path("straight"))["mse"], second["mse"])     # nothing left to do: evaluates again

    def test_a_stop_file_stops_the_run_and_it_continues_to_the_same_result(self):
        run(self.path("straight"))
        first = run(self.path("resumed"), toy={"stop_file_at_step": 22})
        self.assertEqual((first["finished"], first["step"]), (False, 23))
        self.assertFalse(os.path.exists(self.path("resumed/STOP")))
        self.assertTrue(run(self.path("resumed"))["finished"])
        self.assert_same_weights("straight", "resumed")

    def test_stop_at_step_stops_once_and_the_same_command_continues(self):
        run(self.path("straight"))
        first = run(self.path("resumed"), stop_at_step=15)
        self.assertEqual((first["finished"], first["step"]), (False, 15))
        self.assertTrue(run(self.path("resumed"), stop_at_step=15)["finished"])
        self.assert_same_weights("straight", "resumed")

    def test_a_time_limit_stops_the_run_with_a_checkpoint(self):
        result = run(self.path("limited"), max_minutes=1e-9)
        self.assertEqual((result["finished"], result["step"]), (False, 0))
        self.assertEqual(weights(self.path("limited"))["step"], 0)

    def test_data_loading_processes_do_not_change_the_result(self):
        run(self.path("inline"), total_steps=12)
        run(self.path("workers"), total_steps=12, num_workers=2)
        self.assert_same_weights("inline", "workers")

    def test_the_distributed_path_with_one_process_gives_the_same_result(self):
        run(self.path("plain"))
        env = {"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0", "MASTER_ADDR": "127.0.0.1",
               "MASTER_PORT": str(free_port())}
        os.environ.update(env)
        try:
            cfg = {**BASE, "run_dir": self.path("ddp")}
            with contextlib.redirect_stdout(io.StringIO()):
                trainer = Trainer(ToyTask(cfg), cfg)
                self.assertTrue(trainer.info.distributed)
                self.assertIsInstance(trainer.ddp_model, torch.nn.parallel.DistributedDataParallel)
                self.assertTrue(trainer.fit()["finished"])
        finally:
            for key in env:
                del os.environ[key]
        self.assert_same_weights("plain", "ddp")

    def test_parts_the_model_rebuilds_can_be_left_out_of_checkpoints(self):
        class SkipFirst(ToyTask):
            checkpoint_skip = ("0.",)

            def build_model(self):
                model = super().build_model()
                torch.nn.init.constant_(model[0].weight, 0.25)      # "pretrained" and never trained
                model[0].requires_grad_(False)
                return model

            def param_groups(self, model):
                return [{"name": "rest", "params": [p for m in list(model)[1:] for p in m.parameters()], "lr": self.lr}]

        run(self.path("straight"), SkipFirst)
        run(self.path("resumed"), SkipFirst, toy={"stop_file_at_step": 9})
        saved = weights(self.path("resumed"))
        self.assertEqual(saved["step"], 10)
        self.assertFalse(any(key.startswith("0.") for key in saved["model"]))
        run(self.path("resumed"), SkipFirst)
        a, b = weights(self.path("straight")), weights(self.path("resumed"))
        self.assertTrue(all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"]))
        with self.assertRaises(RuntimeError):                       # a checkpoint that lacks other parts is refused
            run(self.path("resumed"), ToyTask)

    def test_the_sync_command_runs_after_checkpoints_and_at_the_end(self):
        marker = self.path("synced.txt")
        run(self.path("sync"), total_steps=10, checkpoint_every=4,
            sync_command=f'echo "$STEP $(basename "$RUN_DIR")" >> "{marker}"')
        with open(marker) as f:
            lines = f.read().split("\n")
        self.assertEqual(lines[-2], "10 sync")                 # the last sync is waited for
        self.assertIn("4 sync", lines)


class TwoProcessTest(unittest.TestCase):
    """The real thing on CPU: two processes started by torchrun, gradients averaged between them."""

    def launch(self, run_dir, *extra):
        command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
                   "--master_port", str(free_port()), os.path.join(REPO, "scripts", "train.py"),
                   "--config", os.path.join(REPO, "configs", "toy.yaml"),
                   "--set", f"run_dir={run_dir}", "total_steps=30", "checkpoint_every=10", "log_every=5",
                   "deterministic=true", *extra]
        env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONNOUSERSITE="1")
        done = subprocess.run(command, capture_output=True, text=True, timeout=600, env=env)
        self.assertEqual(done.returncode, 0, done.stdout[-2000:] + done.stderr[-2000:])
        return done.stdout

    def test_two_processes_train_and_resume_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            straight, resumed = os.path.join(tmp, "straight"), os.path.join(tmp, "resumed")
            out = self.launch(straight)
            self.assertIn("2 process(es)", out)
            self.assertIn('"finished": true', out)
            records = log_records(straight)
            self.assertLess(records[-1]["loss"], records[0]["loss"])
            out = self.launch(resumed, "toy.stop_file_at_step=13")
            self.assertIn('"finished": false', out)
            self.assertEqual(weights(resumed)["step"], 14)
            out = self.launch(resumed)
            self.assertIn("resumed from", out)
            self.assertIn('"finished": true', out)
            a, b = weights(straight), weights(resumed)
            self.assertEqual((a["step"], a["world_size"]), (30, 2))
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)


if __name__ == "__main__":
    unittest.main()
