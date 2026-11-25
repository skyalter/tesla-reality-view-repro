"""Shared helpers for the model tests (files ``test_models_*.py``).

Environment variables (tests that need them are skipped when they are not set):

    RV_WEIGHTS_ROOT   directory holding da3-base/, da3-large-1.1/, da3metric-large/ (each with
                      model.safetensors and config.json)
    AV2_RAW_ROOT      Argoverse 2 Sensor root holding val/ (real images, calibration and poses)
    RV_TEST_LOG       log id under $AV2_RAW_ROOT/val (default: first line of data_lists/av2_val_report20.txt)
    RV_TEST_OUT       optional directory for the sky-score visualisations

A CUDA device is required by every test that runs a full model.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

CAMS = ("ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left", "ring_side_right",
        "ring_rear_left", "ring_rear_right")
H, W = 392, 518
FC_FULL_H = 684       # front-center 2048 x 1550 (portrait) resized to width 518
FC_ROW0 = 146         # crop rows 146 .. 537 of the resized portrait image (same as the data reader)


def weights_root() -> str:
    root = os.environ.get("RV_WEIGHTS_ROOT")
    if not root:
        raise unittest.SkipTest("RV_WEIGHTS_ROOT not set")
    return root


def weights(name: str) -> str:
    p = os.path.join(weights_root(), name, "model.safetensors")
    if not os.path.isfile(p):
        raise unittest.SkipTest(f"{p} missing")
    return p


def need_cuda():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA not available")
    return torch.device("cuda")


def log_dir() -> str:
    root = os.environ.get("AV2_RAW_ROOT")
    if not root:
        raise unittest.SkipTest("AV2_RAW_ROOT not set")
    log_id = os.environ.get("RV_TEST_LOG")
    if not log_id:
        with open(os.path.join(REPO, "data_lists", "av2_val_report20.txt")) as f:
            log_id = f.readline().strip()
    d = os.path.join(root, "val", log_id)
    if not os.path.isdir(d):
        raise unittest.SkipTest(f"{d} missing")
    return d


def _quat_wxyz_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _se3(row):
    T = np.eye(4)
    T[:3, :3] = _quat_wxyz_to_mat([row["qw"], row["qx"], row["qy"], row["qz"]])
    T[:3, 3] = [row["tx_m"], row["ty_m"], row["tz_m"]]
    return T


def load_av2_sample(frame: int = 100, offsets=(0, -10), cams=CAMS):
    """A stage-1 style sample read from the raw log: 7 cameras x moments (front-center frame + offset).

    Returns numpy arrays: images uint8 (N, 3, 392, 518) RGB, K float32 (N, 3, 3), ref_T_cam float32
    (N, 4, 4) camera -> front-center camera at the first moment (OpenCV axes, metres), dt float32 (N,)
    seconds, and the list of (camera, timestamp_ns). Slot-major order (all cameras of offset 0 first).
    Images are resized with INTER_AREA; intrinsics follow the integer pixel-centre convention of the data
    reader (fx' = fx * s, cx' = (cx + 0.5) * s - 0.5; front-center cy -= 146).
    """
    import cv2
    import pandas as pd

    d = log_dir()
    intr = pd.read_feather(os.path.join(d, "calibration", "intrinsics.feather")).set_index("sensor_name")
    ext = pd.read_feather(os.path.join(d, "calibration", "egovehicle_SE3_sensor.feather")).set_index("sensor_name")
    poses = pd.read_feather(os.path.join(d, "city_SE3_egovehicle.feather")).set_index("timestamp_ns")
    stamps = {c: np.array(sorted(int(f[:-4]) for f in os.listdir(os.path.join(d, "sensors", "cameras", c))))
              for c in cams}
    fc = stamps["ring_front_center"]
    imgs, Ks, city_T_cam, ts_all, names = [], [], [], [], []
    for off in offsets:
        t_fc = fc[frame + off]
        for c in cams:
            ts = stamps[c][np.argmin(np.abs(stamps[c] - t_fc))]
            im = cv2.imread(os.path.join(d, "sensors", "cameras", c, f"{ts}.jpg"), cv2.IMREAD_COLOR)
            h0, w0 = im.shape[:2]
            r = intr.loc[c]
            K = np.array([[r.fx_px, 0, r.cx_px], [0, r.fy_px, r.cy_px], [0, 0, 1]], np.float64)
            if c == "ring_front_center":
                sx, sy = W / w0, FC_FULL_H / h0
                im = cv2.resize(im, (W, FC_FULL_H), interpolation=cv2.INTER_AREA)[FC_ROW0:FC_ROW0 + H]
            else:
                sx, sy = W / w0, H / h0
                im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            K[0, 0] *= sx
            K[1, 1] *= sy
            K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
            K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
            if c == "ring_front_center":
                K[1, 2] -= FC_ROW0
            imgs.append(im[:, :, ::-1].transpose(2, 0, 1))
            Ks.append(K)
            city_T_cam.append(_se3(poses.loc[ts]) @ _se3(ext.loc[c]))
            ts_all.append(ts)
            names.append((c, int(ts)))
    city_T_cam = np.stack(city_T_cam)
    ref_T_cam = np.linalg.inv(city_T_cam[0]) @ city_T_cam
    dt = (np.array(ts_all, np.int64) - ts_all[0]) / 1e9
    return (np.ascontiguousarray(np.stack(imgs)), np.stack(Ks).astype(np.float32),
            ref_T_cam.astype(np.float32), dt.astype(np.float32), names)


def to_batch(sample, device):
    """Numpy sample to (1, N, ...) tensors on ``device``."""
    imgs, K, T, dt, _ = sample
    return (torch.from_numpy(imgs)[None].to(device), torch.from_numpy(K)[None].to(device),
            torch.from_numpy(T)[None].to(device), torch.from_numpy(dt)[None].to(device))


def rel_diff(a: torch.Tensor, b: torch.Tensor):
    """(max, mean) of |a - b| / |b|."""
    r = ((a.double() - b.double()).abs() / b.double().abs().clamp_min(1e-12))
    return float(r.max()), float(r.mean())
