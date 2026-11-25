"""Stage 1, first part: train the backbone on depth alone (no rendering).

    model   DA3 backbone + DA3 depth head, metric depth z = pose_scale * kappa * exp(l)
    data    one moment of one log: 7 cameras x the input moments, LiDAR depth, sky scores, ego mask
    loss    LiDAR depth + sky pushed far away + stay close to the frozen original network where there is no LiDAR

Everything that is still open in the design is a configuration value; the defaults below are the
recommended ones. A configuration file only has to name what it changes.
"""
from __future__ import annotations

import hashlib
import math
import os

import torch
from torch.utils.data import Dataset, Subset

from ..engine.config import merge
from ..engine.distributed import unwrap
from ..losses import depth as D

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STANDING_BELOW_M = 0.5      # largest ego displacement within +-1 s below this: the car is standing

DEFAULTS = {
    "model": {
        "name": "da3-large-1.1",    # or da3-base (for debugging on a small GPU)
        "weights_root": None,       # directory holding <name>/model.safetensors
        "pose_scale": 20.0,         # metres per DA3 unit, for the pose translations and the depth
        "kappa": 1.0,               # initial value of the learned global depth factor
        "freeze": "default",        # default: cross-view layers (LARGE 8-23), camera encoder, time encoding
                                    # late: the last quarter of the layers (LARGE 18-23); frozen: nothing
        "train_head": True,
        "reference": True,          # keep a frozen copy of the original network (anchor loss, comparison)
        "grad_checkpoint": False,   # trade time for memory in the backbone / in the head
        "head_checkpoint": False,
        "lr_backbone": 2.0e-5,
        "lr_head": 2.0e-4,
        "lr_scale": 1.0e-2,
    },
    "data": {
        "root": None,               # processed root (.../av2_v1)
        "train_split": "train",
        "train_logs": "data_lists/av2_train_stage1.txt",    # file with one log id per line, or a list of ids
        "train_first": None,        # only the first N logs of the list
        "eval_split": "val",
        "eval_logs": "data_lists/av2_val_dev10.txt",
        "eval_first": None,
        "eval_per_log": 8,          # fixed moments per log, spread over the log
        "input_offsets": [0, -10],  # input moments in front-center frames relative to t (20 Hz): t and t - 0.5 s
        "front_center_mode": "crop",    # crop | split | letterbox
        "depth_layer": "acc",       # LiDAR layer used for training: acc (11 sweeps) | single
        "sky": True,                # read sky scores (they must have been computed)
        "retries": 3,               # a sample that fails to load is replaced by the next one, this many times
    },
    "loss": {
        "lidar": {"weight": 1.0, "range": [0.5, 200.0],
                  "confidence": False, "confidence_lambda": 0.2, "confidence_from_step": 2000,
                  "confidence_clamp": [1.0, 10.0]},
        "sky": {"weight": 0.1, "distance": 400.0, "low": 0.3, "high": 0.5, "window": 5},
        "anchor": {"weight": 0.1, "align": "sample", "min_points": 200, "max_depth": 80.0},
    },
}


class Retry(Dataset):
    """A sample that fails to load is reported and replaced by the next one."""

    def __init__(self, dataset, tries: int):
        self.dataset, self.tries = dataset, max(1, tries)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        for attempt in range(self.tries):
            try:
                return self.dataset[(index + attempt) % len(self.dataset)]
            except Exception as error:  # noqa: BLE001 - any failure of one sample must not end a long run
                print(f"WARNING: sample {index + attempt} failed to load ({error!r})", flush=True)
                last = error
        raise RuntimeError(f"{self.tries} samples in a row failed to load, starting at {index}") from last


def read_logs(spec, first=None) -> list:
    """Log ids from a list, or from a file with one id per line (a relative path is relative to the repository)."""
    if isinstance(spec, (list, tuple)):
        logs = [str(x) for x in spec]
    else:
        path = spec if os.path.isabs(spec) else os.path.join(REPO, spec)
        with open(path) as f:
            logs = [line.split()[0] for line in f if line.strip() and not line.startswith("#")]
    return logs[:first] if first else logs


def fixed_moments(dataset, per_log: int) -> list:
    """Indices of `per_log` samples of every log, at fixed places spread over the log."""
    from ..data.splits import val_fixed_moments

    by_log = {}
    for index, (log_index, moment) in enumerate(dataset.samples):
        by_log.setdefault(log_index, []).append((moment, index))
    chosen = []
    for log_index, items in sorted(by_log.items()):
        wanted = val_fixed_moments(dataset.n_moments[log_index], per_log)
        picked = {min(items, key=lambda item: abs(item[0] - t))[1] for t in wanted}
        chosen.extend(sorted(picked))
    return chosen


class Stage1BackboneTask:
    checkpoint_skip = ("reference.",)       # the frozen copy is rebuilt from the pretrained weights

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.m = merge(DEFAULTS["model"], cfg.get("model") or {})
        self.d = merge(DEFAULTS["data"], cfg.get("data") or {})
        self.l = merge(DEFAULTS["loss"], cfg.get("loss") or {})
        for section, key in ((self.m, "weights_root"), (self.d, "root")):
            if not section[key]:
                raise ValueError(f"{'model' if section is self.m else 'data'}.{key} is not set")
        if self.l["anchor"]["weight"] > 0 and not self.m["reference"]:
            raise ValueError("loss.anchor.weight > 0 needs model.reference: true")

    # ------------------------------------------------------------------ model
    def build_model(self):
        from ..models.stage1 import Stage1Model

        m = self.m
        weights = os.path.join(m["weights_root"], m["name"], "model.safetensors")
        model = Stage1Model(m["name"], weights, pose_scale=m["pose_scale"], freeze=m["freeze"],
                            train_head=m["train_head"], frozen_reference=m["reference"],
                            grad_checkpoint=m["grad_checkpoint"], head_checkpoint=m["head_checkpoint"])
        with torch.no_grad():
            model.log_kappa.fill_(math.log(m["kappa"]))
            model.log_kappa_init.fill_(math.log(m["kappa"]))
        return model

    def param_groups(self, model):
        parts = model.param_groups()
        groups = [{"name": "backbone", "params": parts["backbone"], "lr": self.m["lr_backbone"]},
                  {"name": "head", "params": parts["head"], "lr": self.m["lr_head"]},
                  {"name": "scale", "params": parts["scale"], "lr": self.m["lr_scale"], "warmup_steps": 1}]
        return [g for g in groups if g["params"]]

    # ------------------------------------------------------------------ data
    def _dataset(self, split, logs, evaluation: bool):
        from ..data.dataset import Stage1Dataset

        d = self.d
        return Stage1Dataset(d["root"], split, logs, input_offsets=tuple(d["input_offsets"]),
                             front_center_mode=d["front_center_mode"], depth_layer=d["depth_layer"],
                             load_single_layer=evaluation, with_sky=d["sky"])

    def train_logs(self) -> list:
        return read_logs(self.d["train_logs"], self.d["train_first"])

    def eval_logs(self) -> list:
        return read_logs(self.d["eval_logs"], self.d["eval_first"]) if self.d["eval_logs"] else []

    def build_train_dataset(self):
        d = self.d
        train, evaluation = self.train_logs(), self.eval_logs()
        if d["train_split"] != d["eval_split"] and set(train) & set(evaluation):
            raise RuntimeError("the same log is in the training and the evaluation list")
        dataset = self._dataset(d["train_split"], train, False)
        if dataset.skipped:
            raise RuntimeError(f"training logs that are not ready: {dataset.skipped}")
        return Retry(dataset, d["retries"])

    def build_eval_dataset(self):
        d = self.d
        if not d["eval_logs"]:
            return None
        dataset = self._dataset(d["eval_split"], self.eval_logs(), True)
        if dataset.skipped:
            raise RuntimeError(f"evaluation logs that are not ready: {dataset.skipped}")
        return Subset(dataset, fixed_moments(dataset, d["eval_per_log"]))

    def checkpoint_extra(self) -> dict:
        """Which logs a checkpoint was trained and evaluated on: counts and a checksum of the id lists."""
        def digest(ids):
            return hashlib.sha256("\n".join(ids).encode()).hexdigest()

        train, evaluation = self.train_logs(), self.eval_logs()
        return {"data_lists": {"train_split": self.d["train_split"], "train_logs": len(train),
                               "train_sha256": digest(train), "eval_split": self.d["eval_split"],
                               "eval_logs": len(evaluation), "eval_sha256": digest(evaluation)}}

    # ------------------------------------------------------------------ loss
    def _classes(self, batch):
        sky = self.l["sky"]
        return D.pixel_classes(batch["depth"], batch["depth_flag"], batch["ego_mask"], batch.get("sky"),
                               lidar_range=tuple(self.l["lidar"]["range"]), sky_low=sky["low"],
                               sky_high=sky["high"], sky_window=sky["window"])

    def losses(self, out: dict, batch: dict, step: int):
        """(total, {term: value}) from a model output and a batch. Also used to measure the terms in evaluation."""
        lidar, sky, anchor = self.l["lidar"], self.l["sky"], self.l["anchor"]
        log_z = out["log_depth"]
        classes = self._classes(batch)
        conf = None
        if lidar["confidence"] and step >= lidar["confidence_from_step"]:
            conf = out["depth_conf"].clamp(*lidar["confidence_clamp"])
        terms = {"lidar": D.lidar_depth_loss(log_z, batch["depth"], classes["lidar"], conf, lidar["confidence_lambda"])}
        total = lidar["weight"] * terms["lidar"]
        if "sky" in batch and sky["weight"] > 0:
            norms = D.ray_norms(batch["K"], *log_z.shape[-2:])
            terms["sky"] = D.sky_distance_loss(log_z, norms, classes["sky"], sky["distance"])
            total = total + sky["weight"] * terms["sky"]
        if "ref_log_depth" in out and anchor["weight"] > 0:
            aligned, valid = D.align_reference(out["ref_log_depth"], batch["depth"], classes["lidar"],
                                               anchor["align"], anchor["min_points"], anchor["max_depth"])
            terms["anchor"] = D.anchor_loss(log_z, aligned, classes["anchor"] & valid)
            total = total + anchor["weight"] * terms["anchor"]
        return total, terms, classes

    def training_step(self, model, batch, step):
        out = model(batch["images"], batch["ref_T_cam"], batch["K"], batch["dt"],
                    with_reference=self.l["anchor"]["weight"] > 0)
        total, terms, classes = self.losses(out, batch, step)
        logs = {f"loss_{name}": float(value.detach()) for name, value in terms.items()}
        logs["kappa"] = float(unwrap(model).log_kappa.detach().exp())
        for name, mask in classes.items():
            logs[f"share_{name}"] = float(mask.float().mean())
        return total, logs

    # ------------------------------------------------------------------ evaluation
    @staticmethod
    def summary(m: dict) -> str:
        """One line for the console; everything else is in eval_log.jsonl."""
        def pair(key, original):
            text = f"{m[key]:.4f}" if key in m else "-"
            return text + (f" (original {m[original]:.4f})" if original in m else "")

        parts = [f"AbsRel {pair('absrel_single', 'absrel_orig_single')}",
                 f"scale-aligned {pair('absrel_single_aligned', 'absrel_orig_single_aligned')}",
                 f"d<1.25 {pair('delta1_single', 'delta1_orig_single')}",
                 f"overlap {pair('overlap_median', 'orig_overlap_median')}"]
        if "sky_median_m" in m:
            parts.append(f"sky {m['sky_median_m']:.0f} m")
        parts.append(f"kappa {m.get('kappa', float('nan')):.3f}")
        return "  |  ".join(parts) + f"  [{m.get('samples', 0):.0f} samples, LiDAR single sweep]"

    @staticmethod
    def _eval_mask(depth, flag, ego, depth_range):
        return ((depth >= depth_range[0]) & (depth <= depth_range[1]) & ~ego
                & ((flag.to(torch.int64) & D.SEE_THROUGH) == 0))

    def evaluate(self, model, loader, device) -> dict:
        """Depth errors against LiDAR on the fixed evaluation moments.

        Names: `single` is the single-sweep LiDAR layer (the cleaner one), `acc` the accumulated layer;
        `aligned` means the prediction of every sample was first scaled by the median of LiDAR / prediction;
        `orig` is the frozen original network on the same inputs; `standing` / `moving` split the samples by
        the ego displacement within +-1 s. Keys with a "/" are the breakdowns by distance and by camera."""
        from ..data.config import CAMS
        from ..engine.trainer import to_device

        depth_range = tuple(self.l["lidar"]["range"])
        metrics = D.DepthMetrics(CAMS)
        has_reference = getattr(model, "reference", None) is not None
        sums, counts = {}, {}

        def add(name, value):
            sums[name] = sums.get(name, 0.0) + float(value)
            counts[name] = counts.get(name, 0) + 1

        for batch in loader:
            batch = to_device(batch, device)
            out = model(batch["images"], batch["ref_T_cam"], batch["K"], batch["dt"], with_reference=has_reference)
            cam_id = batch["cam_id"]
            single = self._eval_mask(batch["depth_single"], batch["depth_single_flag"], batch["ego_mask"], depth_range)
            acc = self._eval_mask(batch["depth"], batch["depth_flag"], batch["ego_mask"], depth_range)
            versions = [("", out["depth"])] + ([("orig_", out["ref_depth"])] if has_reference else [])
            for prefix, z in versions:
                metrics.update(prefix + "single", z, batch["depth_single"], single, cam_id)
                for b in range(z.shape[0]):     # the same error, split by whether the car is standing
                    group = "standing" if float(batch["displacement_m"][b]) < STANDING_BELOW_M else "moving"
                    metrics.update(f"{prefix}single_{group}", z[b:b + 1], batch["depth_single"][b:b + 1],
                                   single[b:b + 1], cam_id[b:b + 1])
                metrics.update(prefix + "acc", z, batch["depth"], acc, cam_id)
                scale = D.median_scale(z, batch["depth_single"], single)
                metrics.update(prefix + "single_aligned", z * scale[:, None, None, None], batch["depth_single"],
                               single, cam_id)
                for b in range(z.shape[0]):
                    add(prefix + "scale_log", math.log(float(scale[b])))
                    add(prefix + "scale_log_sq", math.log(float(scale[b])) ** 2)
                    valid = ~batch["ego_mask"][b]
                    if "sky" in batch:
                        valid = valid & (batch["sky"][b] < self.l["sky"]["low"])
                    overlap = D.overlap_consistency(z[b], batch["K"][b], batch["ref_T_cam"][b], valid, batch["slot"][b])
                    if overlap is not None:
                        add(prefix + "overlap_median", overlap[0])
                        add(prefix + "overlap_within_5pct", overlap[1])
                    for cam in range(len(CAMS)):
                        views = cam_id[b] == cam
                        mask = single[b] & views[:, None, None]
                        if int(mask.sum()) >= 50:
                            ratio = (batch["depth_single"][b][mask] / z[b][mask]).median()
                            add(f"{prefix}scale/{CAMS[cam]}", ratio)
            _, terms, classes = self.losses(out, batch, step=0)
            for name, value in terms.items():
                add(f"loss_{name}", value)
            if "sky" in batch and classes["sky"].any():
                norms = D.ray_norms(batch["K"], *out["depth"].shape[-2:])
                add("sky_median_m", (out["depth"] * norms)[classes["sky"]].median())

        result = metrics.compute()
        for name in sums:
            if not name.endswith(("scale_log", "scale_log_sq")):
                result[name] = sums[name] / counts[name]
        for prefix in ("", "orig_") if has_reference else ("",):
            if counts.get(prefix + "scale_log"):
                n = counts[prefix + "scale_log"]
                mean = sums[prefix + "scale_log"] / n
                result[prefix + "scale_median_like"] = math.exp(mean)        # geometric mean of the per-sample factors
                result[prefix + "scale_log_std"] = math.sqrt(max(sums[prefix + "scale_log_sq"] / n - mean ** 2, 0.0))
        if "loss_anchor" in result:      # how far the depth has moved from the original where there is no LiDAR
            result["drift_from_original"] = result["loss_anchor"]
        result["kappa"] = float(model.log_kappa.exp())
        result["samples"] = float(counts.get("scale_log", 0))
        return result
