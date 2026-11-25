"""Offline processing of one Argoverse 2 log: index, images, LiDAR depth, boxes, checks.

The main process builds the tables; images and depth are produced by a process pool in time-ordered
chunks, so every worker keeps the few sweeps it needs in memory. Output goes to ``<log_id>.partial/`` and
is renamed to ``<log_id>/`` only when everything (including DONE) is written.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time

import cv2
import numpy as np
import pandas as pd

from . import av2_io
from .boxes import interpolate_boxes, prepare_tracks
from .config import CAMS, config_sha256
from .images import ego_masks_for_log, resize_to_jpeg, stored_intrinsics, stored_size
from .index import build_frames_table, build_moments_table
from .lidar_depth import SweepCache, depth_for_image
from .store import depth_relpath, depth_to_bytes, img_relpath, sha256_file, write_bytes


class IncompleteLogError(RuntimeError):
    pass


def code_commit(repo_dir: str) -> str:
    try:
        sha = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", repo_dir, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------------------------------- log context
class LogContext:
    """Everything about one raw log that the workers need; built once per process and log."""

    def __init__(self, log_path: str, cfg: dict, ego_mask_dir: str | None):
        self.path = log_path
        self.cfg = cfg
        self.poses = av2_io.read_poses(log_path)
        self.calib, self.ego_T_sensor = av2_io.read_calibration(log_path)
        self.sweep_ts, self.sweep_files = av2_io.list_sweeps(log_path)
        self.cam_ts, self.cam_files = {}, {}
        for cam in CAMS:
            self.cam_ts[cam], self.cam_files[cam] = av2_io.list_camera_frames(log_path, cam)
        self.tracks = prepare_tracks(av2_io.read_annotations(log_path), self.poses, cfg)
        self.ego_masks = ego_masks_for_log(ego_mask_dir, CAMS, cfg)
        self.K = {c: stored_intrinsics(self.calib[c].K, (self.calib[c].width, self.calib[c].height), c, cfg)
                  for c in CAMS}
        self._sweeps = None

    @property
    def sweeps(self) -> SweepCache:
        if self._sweeps is None:
            self._sweeps = SweepCache(self.sweep_ts, self.sweep_files, self.poses, self.tracks, self.cfg)
        return self._sweeps


_WORKER_CTX: dict = {}


def _worker_init():
    cv2.setNumThreads(1)


def _get_ctx(log_path, cfg, ego_mask_dir) -> LogContext:
    key = (log_path, json.dumps(cfg, sort_keys=True), ego_mask_dir)
    if key not in _WORKER_CTX:
        _WORKER_CTX.clear()
        _WORKER_CTX[key] = LogContext(log_path, cfg, ego_mask_dir)
    return _WORKER_CTX[key]


def process_chunk(args) -> list[dict]:
    """Worker task: resize images and compute both depth layers for a time-ordered list of images."""
    log_path, cfg, ego_mask_dir, out_dir, items = args
    ctx = _get_ctx(log_path, cfg, ego_mask_dir)
    stats = []
    for cam, frame, ts, k0 in items:
        t0 = time.perf_counter()
        jpg = resize_to_jpeg(ctx.cam_files[cam][frame], cam, cfg)
        write_bytes(os.path.join(out_dir, img_relpath(cam, ts)), jpg)
        t1 = time.perf_counter()
        w, h = stored_size(cam, cfg)
        acc, single = depth_for_image(ctx.sweeps, int(ts), int(k0), ctx.K[cam], ctx.calib[cam].ego_T_cam,
                                      (h, w), ctx.ego_masks[cam], cfg)
        write_bytes(os.path.join(out_dir, depth_relpath("acc", cam, ts)), depth_to_bytes(acc))
        write_bytes(os.path.join(out_dir, depth_relpath("single", cam, ts)), depth_to_bytes(single))
        t2 = time.perf_counter()
        stats.append({
            "cam": cam, "frame": int(frame),
            "n_valid_px": int((~ctx.ego_masks[cam]).sum()),
            "n_acc": len(acc), "n_single": len(single),
            "n_acc_seethrough": int(((acc.flag & 4) > 0).sum()),
            "n_single_seethrough": int(((single.flag & 4) > 0).sum()),
            "n_acc_sync": int(((acc.flag & 2) > 0).sum()),
            "img_bytes": len(jpg),
            "t_img_s": t1 - t0, "t_depth_s": t2 - t1,
        })
    return stats


def _chunks(items: list, n_workers: int) -> list[list]:
    if n_workers <= 1:
        return [items]
    size = int(np.clip(len(items) // (n_workers * 3), 32, 400))
    return [items[i:i + size] for i in range(0, len(items), size)]


# ----------------------------------------------------------------------------------------------- one log
def process_log(raw_root: str, split: str, log_id: str, out_root: str, cfg: dict, ego_mask_dir: str | None,
                pool=None, n_workers: int = 1, commit: str = "unknown") -> dict:
    """Process one log into ``<out_root>/logs/<split>/<log_id>/``. Returns a summary dict."""
    from .checks import run_log_checks

    log_path = av2_io.log_dir(raw_root, split, log_id)
    if not os.path.isdir(log_path):
        raise IncompleteLogError(f"log directory not found: {split}/{log_id}")
    missing = av2_io.check_log_complete(log_path)
    if missing:
        raise IncompleteLogError(f"{split}/{log_id} is incomplete, missing: {', '.join(missing)}")

    final_dir = os.path.join(out_root, "logs", split, log_id)
    tmp_dir = final_dir + ".partial"
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)  # leftovers of an interrupted run of this script
    os.makedirs(tmp_dir)
    timings = {}
    t_start = time.perf_counter()

    t = time.perf_counter()
    ctx = LogContext(log_path, cfg, ego_mask_dir)
    frames = build_frames_table(ctx.cam_ts, ctx.poses, ctx.sweep_ts)
    moments = build_moments_table(ctx.cam_ts, ctx.poses, ctx.sweep_ts, cfg)
    sweep_idx = ctx.poses.index(ctx.sweep_ts)
    sweeps = pd.DataFrame({
        "k": np.arange(len(ctx.sweep_ts), dtype=np.int32), "timestamp_ns": ctx.sweep_ts,
        "qw": ctx.poses.quat[sweep_idx, 0], "qx": ctx.poses.quat[sweep_idx, 1],
        "qy": ctx.poses.quat[sweep_idx, 2], "qz": ctx.poses.quat[sweep_idx, 3],
        "tx_m": ctx.poses.trans[sweep_idx, 0], "ty_m": ctx.poses.trans[sweep_idx, 1],
        "tz_m": ctx.poses.trans[sweep_idx, 2],
        "n_annotations": [int((ctx.tracks["timestamp_ns"] == s).sum()) for s in ctx.sweep_ts],
    })
    timings["index_s"] = time.perf_counter() - t

    t = time.perf_counter()
    boxes = interpolate_boxes(ctx.tracks, frames["timestamp_ns"].to_numpy(np.int64), cfg)
    boxes.insert(0, "cam", frames["cam"].to_numpy()[boxes["q"].to_numpy()])
    boxes.insert(1, "frame", frames["frame"].to_numpy()[boxes["q"].to_numpy()])
    boxes.insert(2, "timestamp_ns", frames["timestamp_ns"].to_numpy()[boxes["q"].to_numpy()])
    boxes = boxes.drop(columns=["q"])
    timings["boxes_s"] = time.perf_counter() - t

    # images + depth, time-ordered so each chunk touches few sweeps
    t = time.perf_counter()
    order = np.argsort(frames["timestamp_ns"].to_numpy(), kind="stable")
    items = [(frames["cam"].iat[i], int(frames["frame"].iat[i]), int(frames["timestamp_ns"].iat[i]),
              int(frames["k0"].iat[i])) for i in order]
    tasks = [(log_path, cfg, ego_mask_dir, tmp_dir, c) for c in _chunks(items, n_workers)]
    if pool is None:
        _WORKER_CTX.clear()
        _WORKER_CTX[(log_path, json.dumps(cfg, sort_keys=True), ego_mask_dir)] = ctx
        results = [process_chunk(a) for a in tasks]
    else:
        results = pool.map(process_chunk, tasks, chunksize=1)
    st = pd.DataFrame([s for r in results for s in r])
    frames = frames.merge(st, on=["cam", "frame"], how="left", validate="one_to_one")
    timings["images_depth_s"] = time.perf_counter() - t
    timings["images_cpu_s"] = float(st["t_img_s"].sum())
    timings["depth_cpu_s"] = float(st["t_depth_s"].sum())
    frames = frames.drop(columns=["t_img_s", "t_depth_s"])

    os.makedirs(os.path.join(tmp_dir, "sky"), exist_ok=True)
    frames.to_parquet(os.path.join(tmp_dir, "frames.parquet"), index=False)
    moments.to_parquet(os.path.join(tmp_dir, "moments.parquet"), index=False)
    sweeps.to_parquet(os.path.join(tmp_dir, "sweeps.parquet"), index=False)
    boxes.to_parquet(os.path.join(tmp_dir, "boxes.parquet"), index=False)

    meta = {
        "log_id": log_id, "split": split, "city": av2_io.city_of_log(log_path),
        "cams": list(CAMS),
        "n_frames": {c: int(len(ctx.cam_ts[c])) for c in CAMS},
        "n_sweeps": int(len(ctx.sweep_ts)),
        "full_size_wh": {c: [ctx.calib[c].width, ctx.calib[c].height] for c in CAMS},
        "stored_size_wh": {c: list(stored_size(c, cfg)) for c in CAMS},
        "K_full": {c: ctx.calib[c].K.tolist() for c in CAMS},
        "K_stored": {c: ctx.K[c].tolist() for c in CAMS},
        "pixel_centre_convention": "integer (pixel u covers [u-0.5, u+0.5)); add 0.5 to cx, cy for gsplat",
        "ego_T_cam": {c: ctx.calib[c].ego_T_cam.tolist() for c in CAMS},
        "ego_T_lidar": {k: ctx.ego_T_sensor[k].tolist() for k in ("up_lidar", "down_lidar") if k in ctx.ego_T_sensor},
        "config": cfg, "config_sha256": config_sha256(cfg), "code_commit": commit,
        "sky": "not produced in this version (needs the DA3METRIC model); sky/ is reserved",
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "opencv": cv2.__version__},
    }
    with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)

    t = time.perf_counter()
    checks = run_log_checks(log_path, tmp_dir, ctx, cfg, raw_root=raw_root, split=split)
    timings["checks_s"] = time.perf_counter() - t
    timings["total_s"] = time.perf_counter() - t_start
    with open(os.path.join(tmp_dir, "CHECKS.json"), "w") as f:
        json.dump(checks, f, indent=1, sort_keys=True)

    # DONE: sha256 of every file, timings, overall check result
    hashes = {}
    for root, _, names in os.walk(tmp_dir):
        for n in names:
            full = os.path.join(root, n)
            hashes[os.path.relpath(full, tmp_dir).replace(os.sep, "/")] = sha256_file(full)
    sizes = _sizes(tmp_dir)
    done = {"log_id": log_id, "split": split, "config_sha256": config_sha256(cfg), "code_commit": commit,
            "checks_passed": bool(checks["passed"]), "n_workers": n_workers, "timings_s": timings,
            "bytes": sizes, "files": dict(sorted(hashes.items()))}
    with open(os.path.join(tmp_dir, "DONE"), "w") as f:
        json.dump(done, f, indent=1)
    if os.path.isdir(final_dir):
        shutil.rmtree(final_dir)  # a previous output of this same log being redone (--redo)
    os.replace(tmp_dir, final_dir)
    return {"log_id": log_id, "split": split, "status": "done", "checks_passed": bool(checks["passed"]),
            "failed_checks": [k for k, v in checks["checks"].items() if v.get("passed") is False],
            "timings_s": timings, "bytes": sizes}


def _sizes(d: str) -> dict:
    out = {"img": 0, "depth_acc": 0, "depth_single": 0, "boxes": 0, "tables_other": 0, "total": 0}
    for root, _, names in os.walk(d):
        for n in names:
            full = os.path.join(root, n)
            rel = os.path.relpath(full, d).replace(os.sep, "/")
            b = os.path.getsize(full)
            out["total"] += b
            if rel.startswith("img/"):
                out["img"] += b
            elif rel.startswith("depth/acc/"):
                out["depth_acc"] += b
            elif rel.startswith("depth/single/"):
                out["depth_single"] += b
            elif rel == "boxes.parquet":
                out["boxes"] += b
            else:
                out["tables_other"] += b
    return out


def is_done(out_root: str, split: str, log_id: str, cfg: dict) -> bool:
    f = os.path.join(out_root, "logs", split, log_id, "DONE")
    if not os.path.isfile(f):
        return False
    try:
        with open(f) as fh:
            return json.load(fh).get("config_sha256") == config_sha256(cfg)
    except Exception:
        return False
