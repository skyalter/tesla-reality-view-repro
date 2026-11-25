"""3D boxes: city-frame tracks, speed and motion flags, interpolation to camera times, box masks.

Boxes in ``annotations.feather`` are at sweep timestamps (10 Hz) in the ego frame of that sweep.
1. Each box goes to the city frame with the ego pose of its own timestamp.
2. Speed of a track at an annotation: displacement of the box centre over ~1 s (+-0.5 s, clipped to the
   track) divided by the time span. Spans shorter than 0.25 s give no speed. A box is ``moving`` if its
   speed is > 0.5 m/s or unknown; ``person_like`` categories are dynamic regardless of speed;
   ``dynamic = moving | person_like``.
3. Interpolation to a camera timestamp: centre linear, orientation slerp, size from the earlier box,
   at most 50 ms of extrapolation beyond the ends of a track (flagged ``extrapolated``).
Masks are drawn at read time from the interpolated table (``draw_box_mask``), not stored.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from .config import PERSON_LIKE
from .geometry import quat_to_rotmat, rotmat_to_quat, se3_from_quat_trans, slerp

MASK_MOVING = 1
MASK_PERSON = 2
MASK_ANY = 4


def annotations_to_city(ann: pd.DataFrame, poses) -> pd.DataFrame:
    """Add city-frame centre (cx, cy, cz) and orientation (cqw..cqz) to every annotation row."""
    a = ann.copy()
    T = poses.city_T_ego(a["timestamp_ns"].to_numpy(np.int64))  # (N, 4, 4), raises if a pose is missing
    if len(a) == 0:
        for c in ("cx", "cy", "cz", "cqw", "cqx", "cqy", "cqz"):
            a[c] = np.zeros(0)
        return a
    ctr = a[["tx_m", "ty_m", "tz_m"]].to_numpy(np.float64)
    c_city = np.einsum("nij,nj->ni", T[:, :3, :3], ctr) + T[:, :3, 3]
    R_box = quat_to_rotmat(a[["qw", "qx", "qy", "qz"]].to_numpy(np.float64))
    q_city = rotmat_to_quat(T[:, :3, :3] @ R_box)
    a["cx"], a["cy"], a["cz"] = c_city[:, 0], c_city[:, 1], c_city[:, 2]
    a["cqw"], a["cqx"], a["cqy"], a["cqz"] = q_city[:, 0], q_city[:, 1], q_city[:, 2], q_city[:, 3]
    return a


def estimate_motion(a: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add vx, vy, vz, speed (city frame, NaN if unknown), moving, person_like, dynamic."""
    bc = cfg["boxes"]
    half = float(bc["speed_half_window_s"])
    min_span = float(bc["speed_min_span_s"])
    thr = float(bc["moving_speed_mps"])
    a = a.sort_values(["track_uuid", "timestamp_ns"], kind="stable").reset_index(drop=True)
    vel = np.full((len(a), 3), np.nan)
    eps = 0.02
    for _, idx in a.groupby("track_uuid", sort=False).indices.items():
        ts = a["timestamp_ns"].to_numpy(np.int64)[idx] / 1e9
        c = a[["cx", "cy", "cz"]].to_numpy(np.float64)[idx]
        lo = np.searchsorted(ts, ts - half - eps, side="left")
        hi = np.searchsorted(ts, ts + half + eps, side="right") - 1
        span = ts[hi] - ts[lo]
        ok = span >= min_span
        v = np.full((len(idx), 3), np.nan)
        v[ok] = (c[hi[ok]] - c[lo[ok]]) / span[ok, None]
        vel[idx] = v
    a["vx"], a["vy"], a["vz"] = vel[:, 0], vel[:, 1], vel[:, 2]
    a["speed"] = np.linalg.norm(vel[:, :2], axis=1)  # horizontal speed; NaN stays NaN
    a["moving"] = ~(a["speed"].to_numpy() <= thr)  # unknown -> moving
    a["person_like"] = a["category"].isin(PERSON_LIKE).to_numpy()
    a["dynamic"] = a["moving"].to_numpy() | a["person_like"].to_numpy()
    return a


def prepare_tracks(ann: pd.DataFrame, poses, cfg: dict) -> pd.DataFrame:
    return estimate_motion(annotations_to_city(ann, poses), cfg)


BOX_COLUMNS = ["track_uuid", "category", "cx", "cy", "cz", "cqw", "cqx", "cqy", "cqz",
               "length_m", "width_m", "height_m", "vx", "vy", "vz", "speed",
               "moving", "person_like", "dynamic", "extrapolated"]


def interpolate_boxes(tracks: pd.DataFrame, query_ts: np.ndarray, cfg: dict) -> pd.DataFrame:
    """Boxes of every track at every query timestamp (ns). Returns rows with column ``q`` = query index."""
    bc = cfg["boxes"]
    max_ex = int(round(float(bc["max_extrapolation_s"]) * 1e9))
    max_gap = int(round(float(bc["max_gap_s"]) * 1e9))
    query_ts = np.asarray(query_ts, np.int64)
    parts = []
    ts_all = tracks["timestamp_ns"].to_numpy(np.int64)
    num = tracks[["cx", "cy", "cz", "cqw", "cqx", "cqy", "cqz", "length_m", "width_m", "height_m",
                  "vx", "vy", "vz", "speed"]].to_numpy(np.float64)
    mov = tracks["moving"].to_numpy(bool)
    per = tracks["person_like"].to_numpy(bool)
    for uuid, idx in tracks.groupby("track_uuid", sort=True).indices.items():
        ts = ts_all[idx]
        lo_t, hi_t = ts[0] - max_ex, ts[-1] + max_ex
        q = np.nonzero((query_ts >= lo_t) & (query_ts <= hi_t))[0]
        if len(q) == 0:
            continue
        qt = query_ts[q]
        j = np.searchsorted(ts, qt, side="right") - 1  # last annotation at or before qt
        before = j < 0
        after = j >= len(ts) - 1
        a_i = np.clip(j, 0, len(ts) - 1)
        b_i = np.clip(j + 1, 0, len(ts) - 1)
        inside = ~before & ~after
        gap = ts[b_i] - ts[a_i]
        keep = ~inside | (gap <= max_gap)
        # exact hits on the last annotation count as inside
        exact_last = after & (qt == ts[-1])
        A = num[idx][a_i]
        B = num[idx][b_i]
        alpha = np.zeros(len(q))
        alpha[inside] = (qt[inside] - ts[a_i][inside]) / np.maximum(gap[inside], 1)
        ctr = A[:, 0:3] + alpha[:, None] * (B[:, 0:3] - A[:, 0:3])
        quat = slerp(A[:, 3:7], B[:, 3:7], alpha)
        vel = A[:, 10:13] + alpha[:, None] * (B[:, 10:13] - A[:, 10:13])
        speed = A[:, 13] + alpha * (B[:, 13] - A[:, 13])
        # extrapolation beyond the ends: move along the end velocity when known, else hold
        ex = (before | after) & ~exact_last
        if ex.any():
            end = np.where(before, 0, len(ts) - 1)
            E = num[idx][end]
            dt = (qt - ts[end]) / 1e9
            v_end = np.nan_to_num(E[:, 10:13])
            ctr[ex] = E[ex, 0:3] + v_end[ex] * dt[ex, None]
            quat[ex] = E[ex, 3:7]
            vel[ex] = E[ex, 10:13]
            speed[ex] = E[ex, 13]
        m = mov[idx][a_i] | mov[idx][b_i]
        p = per[idx][a_i]
        sel = keep
        if not sel.any():
            continue
        parts.append(pd.DataFrame({
            "q": q[sel].astype(np.int32), "track_uuid": uuid, "category": tracks["category"].iloc[idx[0]],
            "cx": ctr[sel, 0], "cy": ctr[sel, 1], "cz": ctr[sel, 2],
            "cqw": quat[sel, 0], "cqx": quat[sel, 1], "cqy": quat[sel, 2], "cqz": quat[sel, 3],
            "length_m": A[sel, 7], "width_m": A[sel, 8], "height_m": A[sel, 9],
            "vx": vel[sel, 0], "vy": vel[sel, 1], "vz": vel[sel, 2], "speed": speed[sel],
            "moving": m[sel], "person_like": p[sel], "dynamic": (m | p)[sel], "extrapolated": ex[sel],
        }))
    if not parts:
        return pd.DataFrame({c: [] for c in ["q"] + BOX_COLUMNS})
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["q", "track_uuid"], kind="stable").reset_index(drop=True)


def points_in_boxes(pts: np.ndarray, boxes: pd.DataFrame, margin_lw: float, margin_h: float) -> np.ndarray:
    """Which ego-frame points lie inside any of the given ego-frame boxes (annotation rows), grown by margins."""
    inside = np.zeros(len(pts), bool)
    if len(boxes) == 0 or len(pts) == 0:
        return inside
    ctr = boxes[["tx_m", "ty_m", "tz_m"]].to_numpy(np.float64)
    R = quat_to_rotmat(boxes[["qw", "qx", "qy", "qz"]].to_numpy(np.float64))
    half = np.stack([boxes["length_m"].to_numpy() / 2 + margin_lw / 2,
                     boxes["width_m"].to_numpy() / 2 + margin_lw / 2,
                     boxes["height_m"].to_numpy() / 2 + margin_h / 2], axis=1)
    rad = np.linalg.norm(half, axis=1)
    for b in range(len(boxes)):
        near = np.nonzero(np.abs(pts - ctr[b]).max(axis=1) <= rad[b])[0]
        if len(near) == 0:
            continue
        loc = (pts[near] - ctr[b]) @ R[b]
        ins = np.all(np.abs(loc) <= half[b], axis=1)
        inside[near[ins]] = True
    return inside


def _box_surface_samples(n: int) -> np.ndarray:
    """Points on the surface of the unit cube [-1, 1]^3, n x n per face."""
    g = np.linspace(-1.0, 1.0, n)
    a, b = np.meshgrid(g, g, indexing="ij")
    a, b = a.ravel(), b.ravel()
    one = np.ones_like(a)
    faces = [np.stack(x, 1) for x in (
        (one, a, b), (-one, a, b), (a, one, b), (a, -one, b), (a, b, one), (a, b, -one))]
    return np.concatenate(faces, 0)


def draw_box_mask(boxes: pd.DataFrame, cam_T_city: np.ndarray, K: np.ndarray, wh, cfg: dict,
                  bits=None) -> np.ndarray:
    """uint8 mask at (H, W): bit 1 moving, bit 2 person-like, bit 4 any box.

    Each box is grown (length/width + margin, height + margin), swept along its velocity by
    |v| * velocity_stretch_s (centred), sampled on its six faces, projected (points behind the camera are
    dropped), filled as a convex hull, then dilated. Boxes partly outside the image are still drawn.
    ``bits`` optionally restricts which boxes are drawn, e.g. MASK_MOVING | MASK_PERSON.
    """
    mc = cfg["masks"]
    W, H = int(wh[0]), int(wh[1])
    mask = np.zeros((H, W), np.uint8)
    if len(boxes) == 0:
        return mask
    unit = _box_surface_samples(int(mc["face_samples"]))
    stretch = float(mc["velocity_stretch_s"])
    ctr = boxes[["cx", "cy", "cz"]].to_numpy(np.float64)
    R = quat_to_rotmat(boxes[["cqw", "cqx", "cqy", "cqz"]].to_numpy(np.float64))
    half = np.stack([boxes["length_m"].to_numpy() + mc["margin_lw_m"],
                     boxes["width_m"].to_numpy() + mc["margin_lw_m"],
                     boxes["height_m"].to_numpy() + mc["margin_h_m"]], 1) / 2
    vel = np.nan_to_num(boxes[["vx", "vy", "vz"]].to_numpy(np.float64))
    moving = boxes["moving"].to_numpy(bool)
    person = boxes["person_like"].to_numpy(bool)
    lim = np.array([4 * W, 4 * H])
    for b in range(len(boxes)):
        val = MASK_ANY | (MASK_MOVING if moving[b] else 0) | (MASK_PERSON if person[b] else 0)
        if bits is not None and not (val & bits):
            continue
        local = unit * half[b]
        world = local @ R[b].T + ctr[b]
        shift = vel[b] * stretch / 2
        pts = np.concatenate([world - shift, world + shift], 0) if np.any(shift) else world
        pc = pts @ cam_T_city[:3, :3].T + cam_T_city[:3, 3]
        pc = pc[pc[:, 2] > 0.1]
        if len(pc) < 3:
            continue
        uv = np.stack([K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]], 1)
        if (uv[:, 0].max() < -0.5 or uv[:, 0].min() > W - 0.5 or uv[:, 1].max() < -0.5 or uv[:, 1].min() > H - 0.5):
            continue
        uv = np.clip(uv, -lim, lim + np.array([W, H]))
        hull = cv2.convexHull(np.round(uv).astype(np.int32))
        layer = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(layer, hull, 1)
        mask |= layer * np.uint8(val)
    d = int(mc["dilate_px"])
    if d > 0 and mask.any():
        k = np.ones((2 * d + 1, 2 * d + 1), np.uint8)
        out = np.zeros_like(mask)
        for bit in (MASK_MOVING, MASK_PERSON, MASK_ANY):
            out |= (cv2.dilate((mask & bit > 0).astype(np.uint8), k) * bit).astype(np.uint8)
        mask = out
    return mask


def cam_T_city_from_row(row, ego_T_cam: np.ndarray) -> np.ndarray:
    """cam_T_city for one frames-table row (exact ego pose at the image timestamp)."""
    from .geometry import se3_inv

    city_T_ego = se3_from_quat_trans([row["qw"], row["qx"], row["qy"], row["qz"]],
                                     [row["tx_m"], row["ty_m"], row["tz_m"]])
    return se3_inv(city_T_ego @ ego_T_cam)
