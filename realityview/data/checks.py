"""Data checks that do not need a model, and overlay images.

Per-log checks (run at the end of preprocessing, written to CHECKS.json):
  completeness, timestamps, image_cache, intrinsics, lidar_two_sensor, lidar_coverage, motion,
  ego_mask_points, depth_files, boxes.
Global checks (scripts/check_data.py): split lists disjoint, reproducibility, sample reading.
Not implemented here (need a model or the cluster): sky score check, frozen-DA3 scale check, transfer check.
"""
from __future__ import annotations

import glob
import os

import cv2
import numpy as np
import pandas as pd

from .config import CAMS, FRONT_CENTER
from .images import crop_intrinsics, project, resize_array, stored_size
from .lidar_depth import two_lidar_consistency
from .store import LogStore, depth_relpath, img_relpath

MIN_FRAMES_PER_CAM = 318
MIN_SWEEPS = 155
FRAME_INTERVAL_MS = (49.0, 51.0)
MAX_GROUP_DT_MS = 25.0
MIN_STRICT_MOMENTS = 270
# The design asked for PSNR > 40 dB against a direct resize, but JPEG quality 95 alone gives 34-40 dB on these
# images (busy foliage is lowest), so 40 dB fails on correct output. The strict test is instead that
# re-encoding the direct resize gives the stored bytes exactly; PSNR > 30 dB is kept as a loose sanity bound.
IMG_PSNR_DB = 30.0
INTRINSICS_TOL_PX = 0.01
LIDAR_NN_MEDIAN_M = 0.15
COVERAGE_SINGLE = (0.04, 0.10)
COVERAGE_ACC = (0.15, 0.45)
SEETHROUGH_MAX = 0.15
STATIC_MOVING_FRAC_MAX = 0.02


def _res(passed, **kw):
    d = {"passed": None if passed is None else bool(passed)}
    d.update(kw)
    return d


def check_completeness(log_path: str, raw_root: str | None, split: str | None) -> dict:
    counts = {c: len(glob.glob(os.path.join(log_path, "sensors", "cameras", c, "*.jpg"))) for c in CAMS}
    n_sweeps = len(glob.glob(os.path.join(log_path, "sensors", "lidar", "*.feather")))
    files_ok = all(os.path.isfile(os.path.join(log_path, f)) for f in (
        "annotations.feather", "city_SE3_egovehicle.feather", "calibration/intrinsics.feather",
        "calibration/egovehicle_SE3_sensor.feather"))
    ok = files_ok and min(counts.values()) >= MIN_FRAMES_PER_CAM and n_sweeps >= MIN_SWEEPS
    out = {"frames_per_cam": counts, "n_sweeps": n_sweeps, "meta_files_present": files_ok,
           "threshold": f">= {MIN_FRAMES_PER_CAM} frames per camera, >= {MIN_SWEEPS} sweeps"}
    # byte sizes against MANIFEST.tsv when the download left one at the raw root
    man = os.path.join(raw_root, "MANIFEST.tsv") if raw_root else None
    if man and os.path.isfile(man) and split:
        m = pd.read_csv(man, sep="\t")
        prefix = f"{split}/{os.path.basename(log_path.rstrip('/'))}/"
        m = m[m["relative_path"].str.startswith(prefix)]
        m = m[~m["relative_path"].str.contains("/stereo_")]
        bad = [r for r in m.itertuples() if not os.path.isfile(os.path.join(raw_root, r.relative_path))
               or os.path.getsize(os.path.join(raw_root, r.relative_path)) != r.bytes]
        out["manifest_files"] = int(len(m))
        out["manifest_mismatch"] = len(bad)
        ok = ok and not bad
    else:
        out["manifest_files"] = None
    return _res(ok, **out)


def check_timestamps(ctx, moments: pd.DataFrame) -> dict:
    missing = int(sum((~ctx.poses.has(ctx.cam_ts[c])).sum() for c in CAMS) + (~ctx.poses.has(ctx.sweep_ts)).sum())
    iv = {}
    for c in CAMS:
        d = np.diff(ctx.cam_ts[c]) / 1e6
        iv[c] = [float(d.min()), float(d.max())]
    iv_ok = all(FRAME_INTERVAL_MS[0] <= a and b <= FRAME_INTERVAL_MS[1] for a, b in iv.values())
    # ``complete`` is defined by the same tolerance, so the group check is on the moments away from the two
    # ends of the log: there every camera must have its partner frame within MAX_GROUP_DT_MS. Only the first
    # and last moment may lack a partner (the camera started late or stopped early).
    n = len(moments)
    interior = moments.iloc[1:n - 1] if n > 2 else moments.iloc[0:0]
    max_dt_interior = float(interior["max_abs_dt_ms"].max()) if len(interior) else float("nan")
    n_incomplete_interior = int((~interior["complete"]).sum())
    n_s1 = int(moments["valid_stage1"].sum())
    n_s2 = int(moments["valid_stage2"].sum())
    ok = (missing == 0 and iv_ok and max_dt_interior <= MAX_GROUP_DT_MS and n_incomplete_interior == 0
          and n_s2 >= MIN_STRICT_MOMENTS)
    return _res(ok, missing_poses=missing, frame_interval_ms=iv, max_group_dt_ms_interior=max_dt_interior,
                max_group_dt_ms_all=float(moments["max_abs_dt_ms"].max()) if n else float("nan"),
                n_moments=int(n), n_incomplete=int((~moments["complete"]).sum()),
                n_incomplete_interior=n_incomplete_interior, n_valid_stage1=n_s1, n_valid_stage2=n_s2,
                threshold=f"0 missing poses; intervals in {FRAME_INTERVAL_MS} ms; group dt <= {MAX_GROUP_DT_MS} ms "
                          f"for every moment except the first and last; valid_stage2 >= {MIN_STRICT_MOMENTS}")


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def check_image_cache(ctx, store: LogStore, frames: pd.DataFrame, cfg: dict, n_psnr: int = 20, seed: int = 0) -> dict:
    count_ok = all(int((frames["cam"] == c).sum()) == len(ctx.cam_ts[c]) for c in CAMS)
    bad = []
    for r in frames.itertuples():
        b = store.read(img_relpath(r.cam, r.timestamp_ns))
        im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        w, h = stored_size(r.cam, cfg)
        if im is None or im.shape[:2] != (h, w):
            bad.append(f"{r.cam}/{r.timestamp_ns}")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(frames), size=min(n_psnr, len(frames)), replace=False)
    ps = []
    same_bytes = 0
    for i in sorted(pick):
        r = frames.iloc[i]
        ref = resize_array(cv2.imread(ctx.cam_files[r.cam][int(r.frame)]), r.cam, cfg)
        stored = store.read(img_relpath(r.cam, r.timestamp_ns))
        re_enc = cv2.imencode(".jpg", ref, [cv2.IMWRITE_JPEG_QUALITY, int(cfg["images"]["jpeg_quality"])])[1].tobytes()
        same_bytes += int(re_enc == stored)
        im = cv2.imdecode(np.frombuffer(stored, np.uint8), cv2.IMREAD_COLOR)
        ps.append(_psnr(ref, im))
    ok = count_ok and not bad and same_bytes == len(ps) and min(ps) > IMG_PSNR_DB
    return _res(ok, counts_match=count_ok, undecodable_or_wrong_size=bad[:10], n_bad=len(bad),
                reencoded_identical=f"{same_bytes}/{len(ps)}",
                psnr_db_min=float(min(ps)), psnr_db_median=float(np.median(ps)),
                threshold=f"all decode at stored size; on {len(ps)} random images the direct resize re-encodes to "
                          f"the stored bytes and PSNR (JPEG loss only) > {IMG_PSNR_DB} dB")


def intrinsics_error_px(K_full, full_wh, K_stored, stored_wh, crop_row0: int = 0, n: int = 1000, seed: int = 0) -> float:
    """Max |projection with stored K - (full-res projection mapped to the stored grid)| over random points."""
    rng = np.random.default_rng(seed)
    pts = np.stack([rng.uniform(-20, 20, n), rng.uniform(-10, 10, n), rng.uniform(1, 80, n)], 1)
    u0, v0, _ = project(np.asarray(K_full), pts)
    sx = stored_wh[0] / full_wh[0]
    sy = stored_wh[1] / full_wh[1]
    u_ref = (u0 + 0.5) * sx - 0.5
    v_ref = (v0 + 0.5) * sy - 0.5 - crop_row0
    u1, v1, _ = project(crop_intrinsics(K_stored, row0=crop_row0), pts)
    return float(max(np.abs(u1 - u_ref).max(), np.abs(v1 - v_ref).max()))


def check_intrinsics(ctx, cfg: dict) -> dict:
    errs = {}
    for c in CAMS:
        full = (ctx.calib[c].width, ctx.calib[c].height)
        errs[c] = intrinsics_error_px(ctx.calib[c].K, full, ctx.K[c], stored_size(c, cfg))
    c = FRONT_CENTER
    full = (ctx.calib[c].width, ctx.calib[c].height)
    r0 = int(cfg["images"]["front_center_crop_row0"])
    errs["front_center_crop"] = intrinsics_error_px(ctx.calib[c].K, full, ctx.K[c], stored_size(c, cfg), crop_row0=r0)
    errs["front_center_split_lower"] = intrinsics_error_px(ctx.calib[c].K, full, ctx.K[c], stored_size(c, cfg),
                                                           crop_row0=stored_size(c, cfg)[1] - cfg["images"]["landscape_wh"][1])
    ok = max(errs.values()) < INTRINSICS_TOL_PX
    return _res(ok, max_error_px=errs, threshold=f"< {INTRINSICS_TOL_PX} px")


def check_lidar_two_sensor(ctx, n_sweeps: int = 6) -> dict:
    ks = np.linspace(5, len(ctx.sweep_ts) - 7, n_sweeps).round().astype(int)
    fc = ctx.cam_ts[FRONT_CENTER]
    rows = []
    for k in ks:
        # a front-center frame exposed during sweep k: the camera time differs from the sweep time
        j = int(np.searchsorted(fc, ctx.sweep_ts[k], side="right"))
        cam_ts = int(fc[min(j, len(fc) - 1)])
        r = two_lidar_consistency(ctx.sweeps, int(k), cam_ts)
        r["k"] = int(k)
        r["cam_minus_sweep_ms"] = (cam_ts - int(ctx.sweep_ts[k])) / 1e6
        rows.append(r)
    ud = [r["up_vs_down_median_m"] for r in rows if "up_vs_down_median_m" in r]
    sn = [r["sweep_vs_next_median_m"] for r in rows if "sweep_vs_next_median_m" in r]
    ok = bool(ud) and bool(sn) and max(ud) <= LIDAR_NN_MEDIAN_M and max(sn) <= LIDAR_NN_MEDIAN_M
    return _res(ok, per_sweep=rows, up_vs_down_max_m=max(ud) if ud else None,
                sweep_vs_next_max_m=max(sn) if sn else None,
                threshold=f"median NN distance <= {LIDAR_NN_MEDIAN_M} m (static points, 3-40 m, > 0.5 m high)")


def check_coverage(frames: pd.DataFrame) -> dict:
    land = frames[frames["cam"] != FRONT_CENTER]
    cov_single = land["n_single"] / land["n_valid_px"]
    cov_acc = land["n_acc"] / land["n_valid_px"]
    st = frames["n_acc_seethrough"] / frames["n_acc"].clip(lower=1)
    m1, m2, m3 = float(cov_single.median()), float(cov_acc.median()), float(st.median())
    per_cam = {c: {"single": float((g["n_single"] / g["n_valid_px"]).median()),
                   "acc": float((g["n_acc"] / g["n_valid_px"]).median())} for c, g in frames.groupby("cam")}
    ok = COVERAGE_SINGLE[0] <= m1 <= COVERAGE_SINGLE[1] and COVERAGE_ACC[0] <= m2 <= COVERAGE_ACC[1] and m3 <= SEETHROUGH_MAX
    return _res(ok, landscape_single_median=m1, landscape_acc_median=m2, acc_seethrough_frac_median=m3,
                per_cam_median=per_cam,
                threshold=f"landscape single in {COVERAGE_SINGLE}, acc in {COVERAGE_ACC}, see-through <= {SEETHROUGH_MAX}")


def check_motion(tracks: pd.DataFrame, moving_speed: float) -> dict:
    g = tracks.groupby("track_uuid")
    first = g[["cx", "cy"]].first()
    last = g[["cx", "cy"]].last()
    net = np.hypot(last["cx"] - first["cx"], last["cy"] - first["cy"])
    still_ids = net.index[net < 0.5]
    s = tracks[tracks["track_uuid"].isin(still_ids) & tracks["speed"].notna()]
    frac = float((s["speed"] > moving_speed).mean()) if len(s) else 0.0
    dyn_frac = float(tracks["dynamic"].mean()) if len(tracks) else 0.0
    unknown = float(tracks["speed"].isna().mean()) if len(tracks) else 0.0
    return _res(frac <= STATIC_MOVING_FRAC_MAX, still_tracks=int(len(still_ids)), still_track_boxes=int(len(s)),
                still_judged_moving_frac=frac, n_tracks=int(len(net)), n_boxes=int(len(tracks)),
                dynamic_box_frac=dyn_frac, unknown_speed_box_frac=unknown,
                threshold=f"<= {STATIC_MOVING_FRAC_MAX} of boxes of tracks with net displacement < 0.5 m judged moving")


def check_ego_points_and_depth(store: LogStore, frames: pd.DataFrame, ego_masks: dict, cfg: dict) -> tuple[dict, dict]:
    in_mask = 0
    n_checked = 0
    missing = []
    out_of_range = 0
    zr = (cfg["lidar"]["z_min_m"] - 0.01, cfg["lidar"]["z_max_m"] + 0.25)
    for r in frames.itertuples():
        for layer in ("acc", "single"):
            rel = depth_relpath(layer, r.cam, r.timestamp_ns)
            try:
                d = store.depth(layer, r.cam, r.timestamp_ns)
            except Exception:
                missing.append(rel)
                continue
            z = d.z.astype(np.float32)
            out_of_range += int(((z < zr[0]) | (z > zr[1])).sum())
            if ego_masks[r.cam].any():
                in_mask += int(ego_masks[r.cam][d.vu[:, 0], d.vu[:, 1]].sum())
                n_checked += 1
    ego = _res(in_mask == 0, points_in_ego_mask=in_mask, depth_files_checked=n_checked, threshold="0 points")
    dep = _res(not missing and out_of_range == 0, missing=missing[:10], n_missing=len(missing),
               z_out_of_range=out_of_range, threshold="every image has both layers; z within range")
    return ego, dep


def check_boxes(boxes: pd.DataFrame, frames: pd.DataFrame) -> dict:
    per_img = boxes.groupby(["cam", "frame"]).size()
    return _res(None, n_rows=int(len(boxes)), images_with_boxes=int(len(per_img)), images=int(len(frames)),
                boxes_per_image_median=float(per_img.median()) if len(per_img) else 0.0,
                extrapolated_frac=float(boxes["extrapolated"].mean()) if len(boxes) else 0.0,
                dynamic_frac=float(boxes["dynamic"].mean()) if len(boxes) else 0.0,
                note="informational")


def check_sky(store: LogStore, frames: pd.DataFrame, cfg: dict, n_sample: int = 40, seed: int = 0,
              threshold: float = 0.5) -> dict:
    """Sky scores of a log: one file per image, the right size, and no sky at the bottom of the pictures.

    passed is None when the log has no sky scores yet (scripts/sky_scores.py has not run on it)."""
    from .images import stored_size
    from .store import SKY_DONE, sky_relpath

    if not store.exists(SKY_DONE):
        return _res(None, reason="no sky scores yet (scripts/sky_scores.py)")
    done = store.json(SKY_DONE)
    items = [(c, int(t)) for c, t in zip(frames["cam"], frames["timestamp_ns"])]
    missing = [f"{c}/{t}" for c, t in items if not store.exists(sky_relpath(c, t))]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(items), size=min(n_sample, len(items)), replace=False)
    wrong_size, top, bottom, whole = [], [], [], []
    for k in pick:
        cam, ts = items[int(k)]
        if f"{cam}/{ts}" in missing:
            continue
        m = store.sky(cam, ts)
        w, h = stored_size(cam, cfg)
        if m.shape != (h, w):
            wrong_size.append(f"{cam}/{ts}: {m.shape}")
            continue
        is_sky = m >= round(threshold * 100)
        rows = max(1, h // 8)
        top.append(float(is_sky[:rows].mean()))
        bottom.append(float(is_sky[-rows:].mean()))
        whole.append(float(is_sky.mean()))
    bottom_share = float(np.mean(bottom)) if bottom else None
    ok = not missing and not wrong_size and done.get("images") == len(items) and bottom_share is not None \
        and bottom_share < 0.02
    return _res(ok, n_images=len(items), n_missing=len(missing), missing=missing[:10], wrong_size=wrong_size[:10],
                done=done, sampled=len(whole), threshold=threshold,
                sky_share_top_eighth=float(np.mean(top)) if top else None, sky_share_bottom_eighth=bottom_share,
                sky_share_whole=float(np.mean(whole)) if whole else None,
                criterion="every image has a score of its own size; under 2 % sky in the bottom eighth of the pictures")


def run_log_checks(log_path: str, out_dir: str, ctx, cfg: dict, raw_root=None, split=None) -> dict:
    store = LogStore(out_dir)
    frames = store.parquet("frames.parquet")
    moments = store.parquet("moments.parquet")
    boxes = store.parquet("boxes.parquet")
    checks = {
        "completeness": check_completeness(log_path, raw_root, split),
        "timestamps": check_timestamps(ctx, moments),
        "image_cache": check_image_cache(ctx, store, frames, cfg),
        "intrinsics": check_intrinsics(ctx, cfg),
        "lidar_two_sensor": check_lidar_two_sensor(ctx),
        "lidar_coverage": check_coverage(frames),
        "motion": check_motion(ctx.tracks, float(cfg["boxes"]["moving_speed_mps"])),
        "boxes": check_boxes(boxes, frames),
    }
    checks["ego_mask_points"], checks["depth_files"] = check_ego_points_and_depth(store, frames, ctx.ego_masks, cfg)
    passed = all(v["passed"] is not False for v in checks.values())
    return {"passed": passed, "checks": checks,
            "not_run": {"sky": "scripts/sky_scores.py, then scripts/check_data.py",
                        "frozen_da3_scale": "scripts/eval_depth.py (needs the model)",
                        "transfer_sha256": "cluster-side",
                        "overlays": "scripts/check_data.py --overlays (human review)"}}


# ----------------------------------------------------------------------------------------------- overlays
MOSAIC = [["ring_front_left", "ring_front_center", "ring_front_right"],
          ["ring_side_left", None, "ring_side_right"],
          ["ring_rear_left", None, "ring_rear_right"]]


def _depth_colors(z, zmin=2.0, zmax=80.0):
    t = np.clip(np.log(np.maximum(z, 1e-3) / zmin) / np.log(zmax / zmin), 0, 1)
    return cv2.applyColorMap((t * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)


def overlay_mosaic(store: LogStore, frames: pd.DataFrame, moment: pd.Series, cfg: dict, kind: str,
                   ego_masks: dict, layer: str = "acc", boxes: pd.DataFrame | None = None,
                   meta: dict | None = None, sky_low: float = 0.3, sky_high: float = 0.5) -> np.ndarray:
    """3x3 mosaic of the 7 cameras of one moment (front-center cropped to 518x392).

    kind "lidar": sparse depth points coloured by log depth (2-80 m); see-through points drawn grey.
    kind "dyn": box masks (red = moving, yellow = person-like, blue outline = any box) and the ego mask.
    kind "sky": sky scores (blue = sky, score >= sky_high; yellow = uncertain, sky_low <= score < sky_high).
    """
    from .boxes import MASK_ANY, MASK_MOVING, MASK_PERSON, cam_T_city_from_row, draw_box_mask
    from .images import front_center_view

    W, H = cfg["images"]["landscape_wh"]
    canvas = np.full((3 * H, 3 * W, 3), 255, np.uint8)
    fr = frames.set_index(["cam", "frame"])
    for r_, row in enumerate(MOSAIC):
        for c_, cam in enumerate(row):
            if cam is None:
                continue
            j = int(moment[f"idx_{cam}"])
            ts = int(moment[f"ts_{cam}"])
            im = cv2.imdecode(np.frombuffer(store.read(img_relpath(cam, ts)), np.uint8), cv2.IMREAD_COLOR)
            K = np.array(meta["K_stored"][cam])
            ego = ego_masks[cam]
            if kind == "lidar":
                d = store.depth(layer, cam, ts)
                dense, flag = d.densify(im.shape[:2])
                if cam == FRONT_CENTER:
                    im, K, (dense, flag, ego) = front_center_view(im, K, "crop", cfg, [dense, flag, ego])
                im = (im * 0.6).astype(np.uint8)
                v, u = np.nonzero(dense > 0)
                z = dense[v, u]
                col = _depth_colors(z)
                st = (flag[v, u] & 4) > 0
                col[st] = (160, 160, 160)
                im[v, u] = col
            elif kind == "sky":
                score = store.sky(cam, ts)
                if cam == FRONT_CENTER:
                    im, K, (score, ego) = front_center_view(im, K, "crop", cfg, [score, ego])
                out = im.astype(np.float32)
                sure, maybe = score >= round(sky_high * 100), (score >= round(sky_low * 100)) & (score < round(sky_high * 100))
                out[sure] = out[sure] * 0.45 + np.array((255, 120, 0), np.float32) * 0.55
                out[maybe] = out[maybe] * 0.3 + np.array((0, 230, 255), np.float32) * 0.7
                im = out.astype(np.uint8)
                cnt, _ = cv2.findContours(sure.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(im, cnt, -1, (255, 255, 255), 1)
                if ego.any():
                    cnt, _ = cv2.findContours(ego.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(im, cnt, -1, (255, 0, 255), 1)
            else:
                frow = fr.loc[(cam, j)]
                bx = boxes[(boxes["cam"] == cam) & (boxes["frame"] == j)]
                mask = draw_box_mask(bx, cam_T_city_from_row(frow, np.array(meta["ego_T_cam"][cam])), K,
                                     (im.shape[1], im.shape[0]), cfg)
                if cam == FRONT_CENTER:
                    im, K, (mask, ego) = front_center_view(im, K, "crop", cfg, [mask, ego])
                out = im.astype(np.float32)
                for bit, color in ((MASK_ANY, (255, 80, 0)), (MASK_MOVING, (0, 0, 255)), (MASK_PERSON, (0, 220, 255))):
                    sel = (mask & bit) > 0
                    a = 0.25 if bit == MASK_ANY else 0.45
                    out[sel] = out[sel] * (1 - a) + np.array(color, np.float32) * a
                out[ego] = out[ego] * 0.4 + np.array((255, 0, 255), np.float32) * 0.6
                im = out.astype(np.uint8)
            if ego.any() and kind == "lidar":
                cnt, _ = cv2.findContours(ego.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(im, cnt, -1, (255, 0, 255), 1)
            cv2.putText(im, cam.replace("ring_", ""), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            canvas[r_ * H:(r_ + 1) * H, c_ * W:(c_ + 1) * W] = im
    lines = [f"log {meta['log_id'][:8]} ({meta['split']}, {meta['city']}), moment i={int(moment['i'])}"]
    if kind == "lidar":
        lines += [f"LiDAR depth, {layer} layer, log colour 2-80 m", "grey: suspected see-through (flag 4)",
                  "magenta outline: ego mask"]
        bar = _depth_colors(np.exp(np.linspace(np.log(2), np.log(80), W - 80)))
        canvas[H + 150:H + 170, W + 40:2 * W - 40] = bar[None]
        for zt in (2, 5, 10, 20, 40, 80):
            x = W + 40 + int(np.log(zt / 2) / np.log(40) * (W - 81))
            cv2.putText(canvas, str(zt), (x - 6, H + 188), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    elif kind == "sky":
        lines += [f"sky score of DA3METRIC-LARGE", f"blue with white outline: sky (score >= {sky_high})",
                  f"yellow: uncertain ({sky_low} <= score < {sky_high}), not supervised",
                  "magenta outline: ego mask"]
    else:
        lines += ["red: moving box (speed > 0.5 m/s or unknown)", "yellow: person-like", "blue tint: any box",
                  "magenta: ego mask", "boxes interpolated to each camera time"]
    for n, t in enumerate(lines):
        cv2.putText(canvas, t, (W + 10, H + 30 + 22 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return canvas
