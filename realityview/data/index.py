"""Per-log index: which camera frames form a moment, and which moments can be samples.

A moment is a front-center frame i. Each other camera contributes its frame nearest in time (the seven
cameras expose one after another, up to ~22.5 ms apart). t - 0.5 s is front-center frame i - 10.
Every image keeps its own timestamp; nothing is snapped to a common time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CAMS, FRONT_CENTER


def nearest_index(sorted_ts: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Index of the nearest element of ``sorted_ts`` for every query timestamp (ties -> earlier)."""
    j = np.searchsorted(sorted_ts, query)
    j0 = np.clip(j - 1, 0, len(sorted_ts) - 1)
    j1 = np.clip(j, 0, len(sorted_ts) - 1)
    pick1 = np.abs(sorted_ts[j1] - query) < np.abs(sorted_ts[j0] - query)
    return np.where(pick1, j1, j0)


def sweep_before(sweep_ts: np.ndarray, ts) -> np.ndarray:
    """k0: the last sweep that started at or before ``ts``; -1 if the camera frame precedes all sweeps."""
    return np.searchsorted(sweep_ts, np.asarray(ts, np.int64), side="right") - 1


def build_frames_table(cam_ts: dict[str, np.ndarray], pose_table, sweep_ts: np.ndarray) -> pd.DataFrame:
    """One row per camera image: camera, frame index, timestamp, exact ego pose, k0."""
    rows = []
    for cam in CAMS:
        ts = cam_ts[cam]
        idx = pose_table.index(ts)
        k0 = sweep_before(sweep_ts, ts)
        rows.append(pd.DataFrame({
            "cam": cam,
            "frame": np.arange(len(ts), dtype=np.int32),
            "timestamp_ns": ts,
            "qw": pose_table.quat[idx, 0], "qx": pose_table.quat[idx, 1],
            "qy": pose_table.quat[idx, 2], "qz": pose_table.quat[idx, 3],
            "tx_m": pose_table.trans[idx, 0], "ty_m": pose_table.trans[idx, 1], "tz_m": pose_table.trans[idx, 2],
            "k0": np.maximum(k0, 0).astype(np.int32),
            "k0_before_first_sweep": k0 < 0,
        }))
    return pd.concat(rows, ignore_index=True)


def build_moments_table(cam_ts: dict[str, np.ndarray], pose_table, sweep_ts: np.ndarray, cfg: dict) -> pd.DataFrame:
    """One row per front-center frame i (the moment), with the frame of every camera and validity flags.

    Columns: i, idx_<cam>, ts_<cam>, dt_ms_<cam> (camera time - front-center time), k0_<cam>,
    max_abs_dt_ms, complete, valid_stage1, valid_stage2, disp_t05_m, disp_win_max_m, n_sweeps_window.
    """
    icfg = cfg["index"]
    past = int(icfg["past_offset_frames"])
    tol_ns = float(icfg["group_tol_ms"]) * 1e6
    margin = int(icfg["stage2_margin_frames"])
    win = int(icfg["disp_window_frames"])
    each_side = int(cfg["lidar"]["sweeps_each_side"])

    fc_ts = cam_ts[FRONT_CENTER]
    n = len(fc_ts)
    out = {"i": np.arange(n, dtype=np.int32)}
    max_dt = np.zeros(n)
    for cam in CAMS:
        j = nearest_index(cam_ts[cam], fc_ts) if cam != FRONT_CENTER else np.arange(n)
        ts = cam_ts[cam][j]
        dt = (ts - fc_ts).astype(np.float64)
        out[f"idx_{cam}"] = j.astype(np.int32)
        out[f"ts_{cam}"] = ts
        out[f"dt_ms_{cam}"] = dt / 1e6
        out[f"k0_{cam}"] = np.maximum(sweep_before(sweep_ts, ts), 0).astype(np.int32)
        max_dt = np.maximum(max_dt, np.abs(dt))
    complete = max_dt <= tol_ns
    out["max_abs_dt_ms"] = max_dt / 1e6
    out["complete"] = complete

    i = np.arange(n)
    prev_ok = np.zeros(n, bool)
    prev_ok[past:] = complete[: n - past]
    valid1 = (i >= past) & complete & prev_ok
    valid2 = valid1 & (i >= margin) & (i <= n - 1 - margin)
    out["valid_stage1"] = valid1
    out["valid_stage2"] = valid2

    pos = pose_table.trans[pose_table.index(fc_ts)]
    disp = np.full(n, np.nan)
    disp[past:] = np.linalg.norm(pos[past:] - pos[: n - past], axis=1)
    out["disp_t05_m"] = disp
    wmax = np.zeros(n)
    for d in range(1, win + 1):
        a = np.linalg.norm(pos[d:] - pos[:-d], axis=1)
        wmax[:-d] = np.maximum(wmax[:-d], a)  # i vs i + d
        wmax[d:] = np.maximum(wmax[d:], a)  # i vs i - d
    out["disp_win_max_m"] = wmax

    k0 = sweep_before(sweep_ts, fc_ts)
    lo = np.maximum(k0 - each_side, 0)
    hi = np.minimum(k0 + each_side, len(sweep_ts) - 1)
    out["n_sweeps_window"] = np.where(k0 >= 0, hi - lo + 1, 0).astype(np.int32)
    return pd.DataFrame(out)
