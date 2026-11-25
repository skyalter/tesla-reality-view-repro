"""Readers for the raw Argoverse 2 Sensor files of one log.

Only pandas / pyarrow / numpy; the ``av2`` package is not needed. Things that are easy to get wrong:

- LiDAR ``x, y, z`` are float16 on disk; they are converted to float64 before any transform.
- LiDAR points are already ego-motion compensated to the sweep timestamp (the file name, start of the
  sweep) and expressed in the ego frame at that time. ``offset_ns`` must NOT be used to compensate again;
  it only tells when each point was measured (used to pick points on moving objects).
- Poses are looked up exactly by timestamp; every camera frame and sweep has its own pose row.
- ``annotations.feather`` is read with pandas so that ``track_uuid`` is kept.
- Images are already undistorted; the distortion coefficients in ``intrinsics.feather`` are ignored.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CAMS
from .geometry import se3_from_quat_trans

UP_LIDAR_LASERS = range(0, 32)  # laser_number 0..31: up_lidar; 32..63: down_lidar (inferred from geometry)


def log_dir(raw_root: str, split: str, log_id: str) -> str:
    return os.path.join(raw_root, split, log_id)


def check_log_complete(path: str, cams=CAMS) -> list[str]:
    """Return what is missing for full processing (empty list = complete)."""
    missing = []
    for rel in ("annotations.feather", "city_SE3_egovehicle.feather",
                "calibration/intrinsics.feather", "calibration/egovehicle_SE3_sensor.feather"):
        if not os.path.isfile(os.path.join(path, rel)):
            missing.append(rel)
    lidar = os.path.join(path, "sensors", "lidar")
    if not glob.glob(os.path.join(lidar, "*.feather")):
        missing.append("sensors/lidar/*.feather")
    for cam in cams:
        if not glob.glob(os.path.join(path, "sensors", "cameras", cam, "*.jpg")):
            missing.append(f"sensors/cameras/{cam}/*.jpg")
    return missing


def _ts_from_names(paths: list[str], ext: str) -> np.ndarray:
    return np.array([int(os.path.basename(p)[: -len(ext)]) for p in paths], np.int64)


def list_camera_frames(path: str, cam: str) -> tuple[np.ndarray, list[str]]:
    """Sorted timestamps (ns) and file paths of one camera."""
    files = sorted(glob.glob(os.path.join(path, "sensors", "cameras", cam, "*.jpg")))
    ts = _ts_from_names(files, ".jpg")
    order = np.argsort(ts, kind="stable")
    return ts[order], [files[i] for i in order]


def list_sweeps(path: str) -> tuple[np.ndarray, list[str]]:
    files = sorted(glob.glob(os.path.join(path, "sensors", "lidar", "*.feather")))
    ts = _ts_from_names(files, ".feather")
    order = np.argsort(ts, kind="stable")
    return ts[order], [files[i] for i in order]


@dataclass
class Sweep:
    timestamp_ns: int
    xyz: np.ndarray  # (N, 3) float64, ego frame at timestamp_ns
    laser_number: np.ndarray  # (N,) uint8
    offset_ns: np.ndarray  # (N,) int64, measurement time = timestamp_ns + offset_ns
    intensity: np.ndarray  # (N,) uint8

    @property
    def is_up(self) -> np.ndarray:
        return self.laser_number < 32


def read_sweep(file: str) -> Sweep:
    d = pd.read_feather(file)
    ts = int(os.path.basename(file)[: -len(".feather")])
    return Sweep(
        timestamp_ns=ts,
        xyz=d[["x", "y", "z"]].to_numpy(np.float64),
        laser_number=d["laser_number"].to_numpy(np.uint8),
        offset_ns=d["offset_ns"].to_numpy(np.int64),
        intensity=d["intensity"].to_numpy(np.uint8),
    )


class PoseTable:
    """city_SE3_egovehicle, looked up exactly by timestamp (no interpolation)."""

    def __init__(self, df: pd.DataFrame):
        df = df.sort_values("timestamp_ns", kind="stable")
        self.ts = df["timestamp_ns"].to_numpy(np.int64)
        if len(np.unique(self.ts)) != len(self.ts):
            raise ValueError("duplicate timestamps in city_SE3_egovehicle")
        self.quat = df[["qw", "qx", "qy", "qz"]].to_numpy(np.float64)
        self.trans = df[["tx_m", "ty_m", "tz_m"]].to_numpy(np.float64)
        self.T = se3_from_quat_trans(self.quat, self.trans)

    def index(self, ts) -> np.ndarray:
        ts = np.atleast_1d(np.asarray(ts, np.int64))
        i = np.searchsorted(self.ts, ts)
        i = np.clip(i, 0, len(self.ts) - 1)
        bad = self.ts[i] != ts
        if bad.any():
            raise KeyError(f"{int(bad.sum())} timestamps have no exact pose, e.g. {int(ts[bad][0])}")
        return i

    def has(self, ts) -> np.ndarray:
        ts = np.atleast_1d(np.asarray(ts, np.int64))
        i = np.clip(np.searchsorted(self.ts, ts), 0, len(self.ts) - 1)
        return self.ts[i] == ts

    def city_T_ego(self, ts) -> np.ndarray:
        """(N, 4, 4) for an array of timestamps, or (4, 4) for a scalar."""
        out = self.T[self.index(ts)]
        return out[0] if np.ndim(ts) == 0 else out


def read_poses(path: str) -> PoseTable:
    return PoseTable(pd.read_feather(os.path.join(path, "city_SE3_egovehicle.feather")))


@dataclass
class CameraCalib:
    name: str
    K: np.ndarray  # 3x3 at the original resolution
    width: int
    height: int
    ego_T_cam: np.ndarray  # 4x4


def read_calibration(path: str) -> tuple[dict[str, CameraCalib], dict[str, np.ndarray]]:
    """Camera intrinsics/extrinsics, and ego_T_sensor for every sensor (incl. up_lidar/down_lidar)."""
    intr = pd.read_feather(os.path.join(path, "calibration", "intrinsics.feather")).set_index("sensor_name")
    extr = pd.read_feather(os.path.join(path, "calibration", "egovehicle_SE3_sensor.feather")).set_index("sensor_name")
    ego_T = {}
    for name, r in extr.iterrows():
        ego_T[str(name)] = se3_from_quat_trans([r.qw, r.qx, r.qy, r.qz], [r.tx_m, r.ty_m, r.tz_m])
    cams = {}
    for name, r in intr.iterrows():
        name = str(name)
        if name not in ego_T:
            continue
        K = np.array([[r.fx_px, 0.0, r.cx_px], [0.0, r.fy_px, r.cy_px], [0.0, 0.0, 1.0]], np.float64)
        cams[name] = CameraCalib(name, K, int(r.width_px), int(r.height_px), ego_T[name])
    return cams, ego_T


def read_annotations(path: str) -> pd.DataFrame:
    """All 3D boxes of the log with track_uuid. Boxes are in the ego frame at their timestamp_ns."""
    a = pd.read_feather(os.path.join(path, "annotations.feather"))
    a["track_uuid"] = a["track_uuid"].astype(str)
    a["category"] = a["category"].astype(str)
    return a.sort_values(["track_uuid", "timestamp_ns"], kind="stable").reset_index(drop=True)


def city_of_log(path: str) -> str:
    """Three-letter city code from the map file name, e.g. ``..._PIT_city_71109.json`` -> PIT."""
    files = glob.glob(os.path.join(path, "map", "log_map_archive_*.json"))
    if not files:
        return "UNKNOWN"
    stem = os.path.basename(files[0])
    try:
        return stem.split("____")[1].split("_")[0]
    except IndexError:
        return "UNKNOWN"
