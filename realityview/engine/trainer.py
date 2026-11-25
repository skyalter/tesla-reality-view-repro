"""A small training loop: multi-GPU through DistributedDataParallel, exact resume, Slurm-friendly.

The loop knows nothing about the model. Everything specific comes from a task object:

    class Task:
        def __init__(self, cfg): ...
        def build_model(self) -> torch.nn.Module
        def build_train_dataset(self) -> torch.utils.data.Dataset          # map-style
        def build_eval_dataset(self)                                       # a Dataset or None
        def param_groups(self, model) -> list[dict]                        # {"name", "params", "lr"} each
        def training_step(self, model, batch, step) -> (loss, {name: float})
        def evaluate(self, model, loader, device) -> {name: float}
        collate = None                                                     # optional collate function
        checkpoint_skip = ()                                               # optional: state-dict name prefixes that
                                                                           # are not saved (rebuilt by build_model)
        def on_device(self, device): ...                                   # optional, called once after build_model

Three things make an interrupted run continue exactly where an uninterrupted one would be:
  * which samples a step sees is a pure function of (seed, step, rank, number of ranks);
  * random numbers inside a step are re-seeded from (seed, step, rank);
  * a checkpoint holds the model, the optimizer and the step, and is written atomically.
The dataset has to play along: a sample must depend on its index only, not on what was read before it.

A run stops early, with a checkpoint, when a file named STOP appears in the run directory, when
`max_minutes` of wall-clock time are used up, or on SIGUSR1 / SIGTERM. Running the same command again
continues it.
"""
from __future__ import annotations

import json
import math
import os
import random
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import distributed as dist
from .checkpoint import find_resume, load_checkpoint, save_checkpoint
from .config import merge
from .sampler import StepBatchSampler

DEFAULTS = {
    "seed": 0,
    "device": "auto",           # auto | cuda | cpu
    "dist_backend": "auto",     # auto | nccl | gloo
    "total_steps": 1000,
    "batch_size": 1,            # per process (per GPU)
    "num_workers": 2,           # data-loading processes per training process
    "optimizer": {"betas": [0.9, 0.95], "weight_decay": 0.05,
                  "no_decay_1d": True},     # biases and normalisation weights are not decayed
    "schedule": {"warmup_steps": 100, "min_lr": 1.0e-6},
    "grad_clip": 1.0,           # 0 = off
    "log_every": 10,
    "checkpoint_every": 500,
    "keep_every": 0,            # also keep step_XXXXXXXX.pt every this many steps; 0 = only last.pt
    "eval_every": 0,            # 0 = evaluate only at the end
    "eval_at_start": False,     # also evaluate before the first step of a fresh run
    "eval_batch_size": 1,
    "resume": "auto",           # "auto": continue from run_dir/last.pt if present; "none"; or a path
    "max_minutes": 0,           # stop with a checkpoint after this much wall-clock time; 0 = no limit
    "sync_command": "",         # shell command, run in the background after each checkpoint and once at the
                                # end; it sees RUN_DIR and STEP in its environment
    "find_unused_parameters": False,
    "allow_tf32": True,
    "deterministic": False,     # deterministic cuDNN, no TF32; slower, for tests
}


def lr_at(step: int, base_lr: float, warmup_steps: int, total_steps: int, min_lr: float, delay_steps: int = 0) -> float:
    """Zero for `delay_steps`, linear warm-up, then cosine decay to `min_lr`. `step` counts finished steps."""
    step -= delay_steps
    if step < 0:
        return 0.0
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    floor = min(min_lr, base_lr)
    span = max(1, total_steps - delay_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / span)
    return floor + (base_lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def step_seed(seed: int, step: int, rank: int) -> int:
    return (seed * 1_000_003 + step * 9_973 + rank) % (2**63 - 1)


def to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(v, device) for v in batch)
    return batch


def ignore_stop_signals(_worker_id: int) -> None:
    """For data-loading processes: the training process decides when to stop, they just follow it."""
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, signal.SIG_IGN)


def split_param_groups(groups, weight_decay: float, no_decay_1d: bool) -> list[dict]:
    """One optimizer group per (task group, decayed or not). Parameters that do not train are left out."""
    out = []
    for i, group in enumerate(groups):
        name = group.get("name", f"group{i}")
        params = [p for p in group["params"] if p.requires_grad]
        common = {k: group[k] for k in ("lr", "warmup_steps", "delay_steps") if k in group}
        decay = [p for p in params if p.ndim > 1 or not no_decay_1d]
        plain = [p for p in params if p.ndim <= 1 and no_decay_1d]
        if decay:
            out.append({"name": name, "params": decay, "weight_decay": group.get("weight_decay", weight_decay), **common})
        if plain:
            out.append({"name": name, "params": plain, "weight_decay": 0.0, **common})
    if not out:
        raise ValueError("no trainable parameters")
    return out


class Trainer:
    def __init__(self, task, cfg: dict):
        self.cfg = cfg = merge(DEFAULTS, cfg)
        self.task = task
        self.run_dir = Path(cfg["run_dir"])
        self.info = dist.setup(cfg["device"], cfg["dist_backend"])
        self.device = self.info.device
        self.stop_requested = False
        self.started = time.time()
        self._sync_process = None
        if cfg["deterministic"]:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        tf32 = bool(cfg["allow_tf32"]) and not cfg["deterministic"]
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        if self.info.is_main:
            self.run_dir.mkdir(parents=True, exist_ok=True)

        seed_everything(cfg["seed"])
        self.model = task.build_model().to(self.device)
        if hasattr(task, "on_device"):
            task.on_device(self.device)
        opt = cfg["optimizer"]
        self.groups = split_param_groups(task.param_groups(self.model), opt["weight_decay"], opt["no_decay_1d"])
        self.optimizer = torch.optim.AdamW(
            [{"params": g["params"], "lr": g["lr"], "weight_decay": g["weight_decay"]} for g in self.groups],
            betas=tuple(opt["betas"]))
        self.step = 0

        self.skip_prefixes = tuple(getattr(task, "checkpoint_skip", ()))
        resume = find_resume(cfg["resume"], self.run_dir)
        if resume is not None:
            state = load_checkpoint(resume, self.model, self.optimizer, self.skip_prefixes)
            self.step = state["step"]
            self.log(f"resumed from {resume} at step {self.step}")
            for key, now in (("world_size", self.info.world_size), ("batch_size", cfg["batch_size"]), ("seed", cfg["seed"])):
                if state.get(key, now) != now:
                    self.log(f"WARNING: {key} was {state[key]} when the checkpoint was written and is {now} now; "
                             "training continues, but not on the sample sequence of the earlier run")
        self.ddp_model = dist.wrap(self.model, self.info, cfg["find_unused_parameters"])

        dataset = task.build_train_dataset()
        self.sampler = StepBatchSampler(len(dataset), cfg["batch_size"], self.info.rank, self.info.world_size,
                                        cfg["seed"], start_step=self.step)
        workers = cfg["num_workers"]
        self.loader = DataLoader(dataset, batch_sampler=self.sampler, num_workers=workers,
                                 collate_fn=getattr(task, "collate", None), pin_memory=self.device.type == "cuda",
                                 worker_init_fn=ignore_stop_signals if workers > 0 else None,
                                 multiprocessing_context="spawn" if workers > 0 else None)
        self.eval_dataset = task.build_eval_dataset() if self.info.is_main else None
        summary = {}
        for g in self.groups:
            entry = summary.setdefault(g["name"], [0, g["lr"]])
            entry[0] += sum(p.numel() for p in g["params"])
        self.log(f"{len(dataset)} training samples; trainable parameters: "
                 + ", ".join(f"{name} {n / 1e6:.2f}M (lr {lr:g})" for name, (n, lr) in summary.items()))

    # ------------------------------------------------------------------ helpers
    def log(self, message: str) -> None:
        if self.info.is_main:
            print(message, flush=True)

    def request_stop(self, *_args) -> None:
        """Finish the current step, write a checkpoint and return. Wired to SIGUSR1 and SIGTERM."""
        self.stop_requested = True

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(sig, self.request_stop)

    def should_stop(self) -> bool:
        """The same answer on every process: one of them was asked to stop, or the main one ran out of time."""
        local = self.stop_requested
        if self.info.is_main:
            limit = self.cfg["max_minutes"]
            if limit and time.time() - self.started > limit * 60:
                local = True
            stop_file = self.run_dir / "STOP"
            if stop_file.exists():
                stop_file.unlink()
                local = True
        return dist.any_process(local, self.info)

    def set_lr(self) -> dict:
        sch = self.cfg["schedule"]
        lrs = {}
        for spec, group in zip(self.groups, self.optimizer.param_groups):
            group["lr"] = lr_at(self.step, spec["lr"], spec.get("warmup_steps", sch["warmup_steps"]),
                                self.cfg["total_steps"], sch["min_lr"], spec.get("delay_steps", 0))
            lrs[f"lr_{spec['name']}"] = group["lr"]
        return lrs

    def checkpoint(self) -> None:
        if not self.info.is_main:
            return
        extra = {"world_size": self.info.world_size, "batch_size": self.cfg["batch_size"], "seed": self.cfg["seed"]}
        save_checkpoint(self.run_dir / "last.pt", self.model, self.optimizer, self.step, self.cfg, extra,
                        self.skip_prefixes)
        keep = self.cfg["keep_every"]
        if keep and self.step % keep == 0:
            save_checkpoint(self.run_dir / f"step_{self.step:08d}.pt", self.model, self.optimizer, self.step,
                            self.cfg, extra, self.skip_prefixes)
        self.sync()

    def sync(self, wait: bool = False) -> None:
        """Start the sync command in the background. If the previous one is still running, skip (or, with
        `wait`, wait for it first and then also wait for the new one)."""
        command = self.cfg["sync_command"]
        if not command or not self.info.is_main:
            return
        previous = self._sync_process
        if previous is not None:
            if wait:
                previous.wait()
            elif previous.poll() is None:
                self.log(f"step {self.step}: the previous sync is still running, not starting another")
                return
            if previous.returncode != 0:
                self.log(f"WARNING: sync command failed with exit code {previous.returncode}, see {self.run_dir / 'sync.log'}")
        env = dict(os.environ, RUN_DIR=str(self.run_dir), STEP=str(self.step))
        with (self.run_dir / "sync.log").open("a") as out:
            out.write(f"== step {self.step}\n")
            out.flush()
            self._sync_process = subprocess.Popen(command, shell=True, env=env, stdout=out, stderr=subprocess.STDOUT)
        if wait:
            code = self._sync_process.wait()
            if code != 0:
                self.log(f"WARNING: sync command failed with exit code {code}, see {self.run_dir / 'sync.log'}")
            self._sync_process = None

    def append_json(self, name: str, record: dict) -> None:
        if self.info.is_main:
            with (self.run_dir / name).open("a") as f:
                f.write(json.dumps(record) + "\n")

    def evaluate(self) -> dict:
        if not self.info.is_main or self.eval_dataset is None:
            return {}
        workers = min(2, self.cfg["num_workers"])
        loader = DataLoader(self.eval_dataset, batch_size=self.cfg["eval_batch_size"], shuffle=False,
                            num_workers=workers, collate_fn=getattr(self.task, "collate", None),
                            worker_init_fn=ignore_stop_signals if workers > 0 else None,
                            multiprocessing_context="spawn" if workers > 0 else None)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            metrics = self.task.evaluate(self.model, loader, self.device)
        self.model.train(was_training)
        self.append_json("eval_log.jsonl", {"step": self.step, **metrics})
        self.log(f"eval at step {self.step}: " + "  ".join(f"{k} {v:.4f}" for k, v in metrics.items()
                                                           if isinstance(v, float) and "/" not in k))
        return metrics

    # ------------------------------------------------------------------ main loop
    def fit(self) -> dict:
        cfg = self.cfg
        total = cfg["total_steps"]
        self.ddp_model.train()
        params = [p for group in self.optimizer.param_groups for p in group["params"]]
        self.log(f"training from step {self.step} to step {total}: {self.info.world_size} process(es), "
                 f"{cfg['batch_size']} sample(s) each, device {self.device}, "
                 f"{'distributed' if self.info.distributed else 'not distributed'}")
        last_metrics: dict = {}
        if cfg["eval_at_start"] and self.step == 0:
            last_metrics = self.evaluate()
        iterator = iter(self.loader)
        data_time = step_time = 0.0
        counted = 0
        stopped = False
        while self.step < total:
            if self.should_stop():
                stopped = True
                break
            t0 = time.time()
            batch = to_device(next(iterator), self.device)
            t1 = time.time()
            # Randomness inside a step depends only on (seed, step, rank), not on how the run got here.
            seed_everything(step_seed(cfg["seed"], self.step, self.info.rank))
            lrs = self.set_lr()
            loss, logs = self.task.training_step(self.ddp_model, batch, self.step)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss is {loss.item()} at step {self.step}")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"] if cfg["grad_clip"] else float("inf"))
            self.optimizer.step()
            self.step += 1
            data_time += t1 - t0
            step_time += time.time() - t1
            counted += 1

            if self.step % cfg["log_every"] == 0 or self.step == total:
                record = {"step": self.step, "loss": dist.mean_across_processes(loss.detach(), self.info),
                          "grad_norm": float(grad_norm), "sec_per_step": step_time / counted,
                          "data_sec_per_step": data_time / counted, **lrs, **logs}
                if self.device.type == "cuda":
                    record["max_mem_gb"] = torch.cuda.max_memory_allocated(self.device) / 2**30
                data_time = step_time = 0.0
                counted = 0
                self.append_json("train_log.jsonl", record)
                self.log(f"step {self.step}/{total}  loss {record['loss']:.4f}  grad {record['grad_norm']:.2f}  "
                         f"{record['sec_per_step']:.2f} s/step (+{record['data_sec_per_step']:.2f} s waiting for data)")
            if cfg["eval_every"] and self.step % cfg["eval_every"] == 0 and self.step < total:
                last_metrics = self.evaluate()
            if self.step % cfg["checkpoint_every"] == 0 and self.step < total:
                self.checkpoint()
        del iterator        # stops the data-loading processes

        self.checkpoint()
        if stopped:
            self.log(f"stopped early: checkpoint written at step {self.step}; run the same command to continue")
        else:
            last_metrics = self.evaluate()
            if self.info.is_main:
                (self.run_dir / "done.json").write_text(json.dumps({"step": self.step, **last_metrics}) + "\n")
        self.sync(wait=True)
        dist.cleanup(self.info)
        return {"step": self.step, "finished": not stopped, **last_metrics}
