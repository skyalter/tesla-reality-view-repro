"""Sparse LiDAR depth for every camera image.

Transform chain for a point of sweep k seen by a camera image at time tc (all float64):

    p_ego(sweep k) --city_T_ego(ts_k)--> city --inv(city_T_ego(tc))--> ego(tc) --inv(ego_T_cam)--> cam --K--> pixel

The points are already compensated to the sweep start ts_k, so ``offset_ns`` is never used to move points.
It is only used to keep, inside dynamic boxes, the points measured within 25 ms of the camera time.

Two layers are produced per image:
- accumulated: static points (outside dynamic boxes) of sweeps k0-5..k0+5, plus synchronised points inside
  dynamic boxes from sweeps k0-1..k0+1;
- single: static points of sweep k0 only, plus the same synchronised dynamic points.
k0 is the last sweep that started at or before the camera time. Per pixel the nearest point is kept.
Flags: 1 = point from sweep k0, 2 = synchronised point inside a dynamic box, 4 = suspected see-through
(depth > 1.2 x the minimum depth in the 5x5 neighbourhood). Depth range 0.5..250 m; points on the ego body
(mask) or outside the image are dropped.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from .av2_io import read_sweep
from .boxes import points_in_boxes
from .config import FLAG_CENTER_SWEEP, FLAG_SEETHROUGH, FLAG_SYNC_DYNAMIC
from .geometry import se3_inv, transform_points
from .images import pixel_of, project


@dataclass
class SparseDepth:
    vu: np.ndarray  # (N, 2) uint16, row and column
    z: np.ndarray  # (N,) float16, metres along the optical axis
    flag: np.ndarray  # (N,) uint8

    def __len__(self):
        return len(self.z)

    def densify(self, hw):
        H, W = hw
        depth = np.zeros((H, W), np.float32)
        flag = np.zeros((H, W), np.uint8)
        v = self.vu[:, 0].astype(np.int64)
        u = self.vu[:, 1].astype(np.int64)
        depth[v, u] = self.z.astype(np.float32)
        flag[v, u] = self.flag
        return depth, flag


def empty_depth() -> SparseDepth:
    return SparseDepth(np.zeros((0, 2), np.uint16), np.zeros(0, np.float16), np.zeros(0, np.uint8))


class SweepCache:
    """Loads sweeps of one log lazily: city-frame points, measurement times, dynamic-box membership."""

    def __init__(self, sweep_ts: np.ndarray, sweep_files: list[str], poses, tracks: pd.DataFrame, cfg: dict,
                 max_items: int = 24):
        self.sweep_ts = sweep_ts
        self.files = sweep_files
        self.poses = poses
        self.cfg = cfg
        self.max_items = max_items
        self._cache: OrderedDict[int, dict] = OrderedDict()
        dyn = tracks[tracks["dynamic"]]
        self._dyn_by_ts = {int(t): g for t, g in dyn.groupby("timestamp_ns")}
        self._all_by_ts = {int(t): g for t, g in tracks.groupby("timestamp_ns")}

    def boxes_at(self, k: int, dynamic_only: bool = True) -> pd.DataFrame:
        src = self._dyn_by_ts if dynamic_only else self._all_by_ts
        return src.get(int(self.sweep_ts[k]), pd.DataFrame(columns=["tx_m"]))

    def get(self, k: int) -> dict:
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        sw = read_sweep(self.files[k])
        lc = self.cfg["lidar"]
        dyn = points_in_boxes(sw.xyz, self.boxes_at(k), lc["lidar_box_margin_lw_m"], lc["lidar_box_margin_h_m"])
        T = self.poses.city_T_ego(sw.timestamp_ns)
        item = {
            "city": transform_points(T, sw.xyz),
            "t_meas": sw.timestamp_ns + sw.offset_ns,
            "dyn": dyn,
            "ego": sw.xyz,
            "up": sw.is_up,
        }
        self._cache[k] = item
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return item


def zbuffer(pix: np.ndarray, z: np.ndarray, n_pix: int):
    """Index of the nearest point for every occupied pixel (deterministic for ties: lowest index)."""
    if len(z) == 0:
        return np.zeros(0, np.int64)
    D = np.full(n_pix, np.inf)
    np.minimum.at(D, pix, z)
    cand = np.nonzero(z <= D[pix])[0]
    _, first = np.unique(pix[cand], return_index=True)
    return cand[first]


def seethrough_flags(vu: np.ndarray, z: np.ndarray, hw, window: int, ratio: float) -> np.ndarray:
    H, W = hw
    D = np.full((H, W), 1e9, np.float32)
    D[vu[:, 0], vu[:, 1]] = z
    mn = cv2.erode(D, np.ones((window, window), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=1e9)
    return D[vu[:, 0], vu[:, 1]] > ratio * mn[vu[:, 0], vu[:, 1]]


def _layer(sel, pix, z, is_k0, is_sync, hw, lc) -> SparseDepth:
    H, W = hw
    idx = np.nonzero(sel)[0]
    if len(idx) == 0:
        return empty_depth()
    win = idx[zbuffer(pix[idx], z[idx], H * W)]
    win = win[np.argsort(pix[win], kind="stable")]
    p = pix[win]
    vu = np.stack([p // W, p % W], 1)
    zz = z[win]
    flag = np.where(is_k0[win], FLAG_CENTER_SWEEP, 0) | np.where(is_sync[win], FLAG_SYNC_DYNAMIC, 0)
    st = seethrough_flags(vu, zz.astype(np.float32), hw, int(lc["seethrough_window"]), float(lc["seethrough_ratio"]))
    flag = flag | np.where(st, FLAG_SEETHROUGH, 0)
    return SparseDepth(vu.astype(np.uint16), zz.astype(np.float16), flag.astype(np.uint8))


def depth_for_image(cache: SweepCache, cam_ts: int, k0: int, K: np.ndarray, ego_T_cam: np.ndarray, hw,
                    ego_mask: np.ndarray | None, cfg: dict) -> tuple[SparseDepth, SparseDepth]:
    """Accumulated and single-sweep sparse depth of one image (K at the stored resolution, hw = (H, W))."""
    lc = cfg["lidar"]
    H, W = hw
    n_sw = len(cache.sweep_ts)
    each = int(lc["sweeps_each_side"])
    sync_each = int(lc["sync_sweeps_each_side"])
    tol = int(round(float(lc["sync_tol_ms"]) * 1e6))
    cam_T_city = se3_inv(cache.poses.city_T_ego(cam_ts) @ ego_T_cam)
    zs, pixs, k0s, syncs = [], [], [], []
    for k in range(max(k0 - each, 0), min(k0 + each, n_sw - 1) + 1):
        s = cache.get(k)
        sync = s["dyn"] & (np.abs(s["t_meas"] - cam_ts) <= tol) if abs(k - k0) <= sync_each else np.zeros(len(s["dyn"]), bool)
        keep = ~s["dyn"] | sync
        pc = transform_points(cam_T_city, s["city"][keep])
        u, v, z = project(K, pc)
        ok = (z >= lc["z_min_m"]) & (z <= lc["z_max_m"]) & (u >= -0.5) & (u < W - 0.5) & (v >= -0.5) & (v < H - 0.5)
        vi, ui = pixel_of(u[ok], v[ok])
        zk = z[ok]
        sk = sync[keep][ok]
        if ego_mask is not None:
            off = ~ego_mask[vi, ui]
            vi, ui, zk, sk = vi[off], ui[off], zk[off], sk[off]
        zs.append(zk)
        pixs.append(vi * W + ui)
        k0s.append(np.full(len(zk), k == k0))
        syncs.append(sk)
    if not zs:
        return empty_depth(), empty_depth()
    z = np.concatenate(zs)
    pix = np.concatenate(pixs)
    is_k0 = np.concatenate(k0s)
    is_sync = np.concatenate(syncs)
    acc = _layer(np.ones(len(z), bool), pix, z, is_k0, is_sync, hw, lc)
    single = _layer(is_k0 | is_sync, pix, z, is_k0, is_sync, hw, lc)
    return acc, single


def two_lidar_consistency(cache: SweepCache, k: int, cam_ts: int, rng_m=(3.0, 40.0), min_height_m=0.5) -> dict:
    """Median nearest-neighbour distances between static surface points, in the ego frame at ``cam_ts``.

    - ``up_vs_down``: up_lidar vs down_lidar points of sweep k;
    - ``sweep_vs_next``: sweep k vs sweep k+1, both moved to camera time (exercises the per-sweep chain).
    Static points: 3..40 m horizontal range, > 0.5 m above the ego origin, outside every box (+0.3 m).
    """
    from scipy.spatial import cKDTree

    lc = cache.cfg["lidar"]
    ego_T_city = se3_inv(cache.poses.city_T_ego(cam_ts))

    def static(kk):
        s = cache.get(kk)
        anyb = points_in_boxes(s["ego"], cache.boxes_at(kk, dynamic_only=False),
                               lc["lidar_box_margin_lw_m"], lc["lidar_box_margin_h_m"])
        r = np.linalg.norm(s["ego"][:, :2], axis=1)
        keep = (r >= rng_m[0]) & (r <= rng_m[1]) & (s["ego"][:, 2] > min_height_m) & ~anyb
        return transform_points(ego_T_city, s["city"][keep]), s["up"][keep]

    out = {}
    p, up = static(k)
    if up.sum() > 100 and (~up).sum() > 100:
        d, _ = cKDTree(p[up]).query(p[~up], k=1)
        out["up_vs_down_median_m"] = float(np.median(d))
    if k + 1 < len(cache.sweep_ts):
        q, _ = static(k + 1)
        if len(p) > 100 and len(q) > 100:
            d, _ = cKDTree(p).query(q, k=1)
            out["sweep_vs_next_median_m"] = float(np.median(d))
    return out
