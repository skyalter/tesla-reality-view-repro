#!/usr/bin/env python3
"""Check processed logs and draw overlays for human review.

Usage (from the repository root):
    PYTHONPATH=. python scripts/check_data.py --out-root <processed>/av2_v1 --split val \
        [--logs <list or ids>] [--overlays] [--read-samples 500] \
        [--repro <log_id> --raw-root <av2 sensor root>]

What it does:
- collects every log's CHECKS.json (written by preprocess.py) into one table;
- split lists: data_lists/av2_train_stage1.txt, av2_val_dev10.txt, av2_val_report20.txt pairwise disjoint;
- --overlays: per log, two 7-camera mosaics of one moment (LiDAR depth on the photos; box masks on the
  photos), written to <out-root>/overlays/<split>/;
- --read-samples N: reads N stage-1 samples with the training reader and reports errors and speed;
- --repro LOG: processes LOG again into a scratch root and compares the sha256 of every file except DONE.
A JSON summary goes to <out-root>/checks_<split>.json. Exit status 1 if anything failed.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from realityview.data.checks import overlay_mosaic  # noqa: E402
from realityview.data.config import CAMS, default_config, load_root_config  # noqa: E402
from realityview.data.splits import check_root_splits, default_lists, overlaps, read_list, val_fixed_moments  # noqa: E402,E501
from realityview.data.store import LogStore, find_log  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", default=os.environ.get("RV_PROCESSED_ROOT"))
    p.add_argument("--split", required=True)
    p.add_argument("--logs", default=None, help="list file or comma-separated ids; default: all processed logs")
    p.add_argument("--overlays", action="store_true")
    p.add_argument("--overlay-dir", default=None, help="where overlays go (default <out-root>/overlays/<split>)")
    p.add_argument("--moment", type=int, default=None, help="front-center frame for overlays (default: middle)")
    p.add_argument("--read-samples", type=int, default=0)
    p.add_argument("--repro", default=None, help="log id to process twice and compare")
    p.add_argument("--raw-root", default=os.environ.get("AV2_RAW_ROOT"))
    return p.parse_args()


def resolve_logs(args):
    if args.logs is None:
        d = os.path.join(args.out_root, "logs", args.split)
        names = sorted(os.listdir(d)) if os.path.isdir(d) else []
        return [n[:-4] if n.endswith(".zip") else n for n in names if not n.endswith(".partial")]
    if os.path.isfile(args.logs):
        return read_list(args.logs)
    return [x.strip() for x in args.logs.split(",") if x.strip()]


def ego_masks(root, cfg):
    out = {}
    for cam in CAMS:
        from realityview.data.images import stored_size

        w, h = stored_size(cam, cfg)
        f = os.path.join(root, "ego_masks", f"{cam}.png")
        m = cv2.imread(f, cv2.IMREAD_GRAYSCALE) if os.path.isfile(f) else None
        out[cam] = (m > 0) if m is not None else np.zeros((h, w), bool)
    return out


def draw_overlays(args, cfg, log_id, store):
    meta = store.json("meta.json")
    frames = store.parquet("frames.parquet")
    moments = store.parquet("moments.parquet")
    boxes = store.parquet("boxes.parquet")
    if args.moment is not None:
        i = args.moment
    else:
        v = moments.index[moments["valid_stage2"]]
        i = int(v[len(v) // 2])
    m = moments.iloc[i]
    od = args.overlay_dir or os.path.join(args.out_root, "overlays", args.split)
    os.makedirs(od, exist_ok=True)
    em = ego_masks(args.out_root, cfg)
    paths = []
    for kind in ("lidar", "dyn"):
        img = overlay_mosaic(store, frames, m, cfg, kind, em, layer="acc", boxes=boxes, meta=meta)
        p = os.path.join(od, f"{log_id}_i{i:03d}_{kind}.jpg")
        cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        paths.append(p)
    img = overlay_mosaic(store, frames, m, cfg, "lidar", em, layer="single", boxes=boxes, meta=meta)
    p = os.path.join(od, f"{log_id}_i{i:03d}_lidar_single.jpg")
    cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    paths.append(p)
    return paths


def read_samples(args, logs, n):
    import torch

    from realityview.data.dataset import Stage1Dataset

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    ds = Stage1Dataset(args.out_root, args.split, logs, require_checks=False, allow_missing_logs=True)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(ds))
    idx = np.concatenate([idx] * (n // max(len(idx), 1) + 1))[:n]
    errors, times = [], []
    shapes = None
    ranges = {"depth_max": 0.0, "depth_min_pos": 1e9, "dt_min": 1e9, "dt_max": -1e9, "trans_max_m": 0.0,
              "coverage_mean": []}
    for k in idx:
        t = time.perf_counter()
        try:
            s = ds[int(k)]
        except Exception as e:
            errors.append(f"{k}: {e!r}")
            continue
        times.append(time.perf_counter() - t)
        if shapes is None:
            shapes = {kk: (list(v.shape), str(v.dtype)) for kk, v in s.items() if hasattr(v, "shape")}
        d = s["depth"]
        pos = d[d > 0]
        ranges["depth_max"] = max(ranges["depth_max"], float(d.max()))
        if len(pos):
            ranges["depth_min_pos"] = min(ranges["depth_min_pos"], float(pos.min()))
        ranges["dt_min"] = min(ranges["dt_min"], float(s["dt"].min()))
        ranges["dt_max"] = max(ranges["dt_max"], float(s["dt"].max()))
        ranges["trans_max_m"] = max(ranges["trans_max_m"], float(s["ref_T_cam"][:, :3, 3].norm(dim=1).max()))
        ranges["coverage_mean"].append(float((d > 0).float().mean()))
    ranges["coverage_mean"] = float(np.mean(ranges["coverage_mean"])) if ranges["coverage_mean"] else None
    t = np.array(times)
    return {"passed": not errors and len(times) == n, "n_requested": n, "n_read": len(times),
            "dataset_len": len(ds), "skipped_logs": ds.skipped, "errors": errors[:20], "n_errors": len(errors),
            "ms_per_sample_mean": float(t.mean() * 1e3) if len(t) else None,
            "ms_per_sample_p90": float(np.percentile(t, 90) * 1e3) if len(t) else None,
            "samples_per_s": float(1 / t.mean()) if len(t) else None, "shapes": shapes, "ranges": ranges}


def repro_check(args, log_id):
    from realityview.data.pipeline import code_commit, process_log

    cfg = default_config()
    scratch = os.path.join(args.out_root, "_repro_tmp")
    if os.path.isdir(scratch):
        shutil.rmtree(scratch)
    process_log(args.raw_root, args.split, log_id, scratch, cfg,
                os.path.join(REPO, "assets", "ego_masks", "argoverse2"), commit=code_commit(REPO))
    a = json.load(open(os.path.join(args.out_root, "logs", args.split, log_id, "DONE")))["files"]
    b = json.load(open(os.path.join(scratch, "logs", args.split, log_id, "DONE")))["files"]
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    shutil.rmtree(scratch)  # the scratch copy made by this check
    return {"passed": not diff, "log_id": log_id, "n_files": len(a), "differing": diff[:20], "n_differing": len(diff)}


def main():
    args = parse_args()
    cfg = load_root_config(args.out_root)
    logs = resolve_logs(args)
    out = {"split": args.split, "logs": {}, "global": {}}
    print(f"{'log':38s} {'checks':7s} failed")
    for log_id in logs:
        path = find_log(args.out_root, args.split, log_id)
        if path is None:
            out["logs"][log_id] = {"status": "not processed"}
            print(f"{log_id:38s} {'MISSING':7s}")
            continue
        store = LogStore(path)
        ck = store.json("CHECKS.json")
        failed = [k for k, v in ck["checks"].items() if v.get("passed") is False]
        out["logs"][log_id] = {"passed": ck["passed"], "failed": failed}
        print(f"{log_id:38s} {'ok' if ck['passed'] else 'FAIL':7s} {', '.join(failed)}")
        if args.overlays:
            try:
                out["logs"][log_id]["overlays"] = draw_overlays(args, cfg, log_id, store)
            except Exception:
                out["logs"][log_id]["overlay_error"] = traceback.format_exc()
                print(traceback.format_exc())
    lists = default_lists(REPO)
    ov = overlaps(lists)
    try:
        root_lists = check_root_splits(args.out_root)
        root_ok, root_err = True, None
    except ValueError as e:
        root_lists, root_ok, root_err = {}, False, str(e)
    out["global"]["split_lists_disjoint"] = {"passed": not ov and root_ok,
                                             "sizes": {k: len(v) for k, v in lists.items()}, "overlaps": ov,
                                             "root_splits": {k: len(v) for k, v in root_lists.items()},
                                             "root_splits_error": root_err}
    print("split lists disjoint:", not ov and root_ok)
    if args.split == "val":
        rows = []
        for log_id in logs:
            path = find_log(args.out_root, args.split, log_id)
            if path is None:
                continue
            mom = LogStore(path).parquet("moments.parquet")
            for k, i in enumerate(val_fixed_moments(len(mom))):
                rows.append(f"{log_id}\t{k}\t{i}\t{bool(mom['valid_stage2'].iloc[i])}\n")
        os.makedirs(os.path.join(args.out_root, "splits"), exist_ok=True)
        with open(os.path.join(args.out_root, "splits", "val_fixed.tsv"), "w") as f:
            f.write("log_id\tk\tmoment\tvalid_stage2\n" + "".join(rows))
        out["global"]["val_fixed"] = {"passed": all(r.rstrip().endswith("True") for r in rows),
                                      "n_samples": len(rows)}
    if args.read_samples:
        r = read_samples(args, logs, args.read_samples)
        out["global"]["read_samples"] = r
        print("read samples:", {k: r[k] for k in ("passed", "n_read", "n_errors", "ms_per_sample_mean", "samples_per_s")})
    if args.repro:
        r = repro_check(args, args.repro)
        out["global"]["reproducibility"] = r
        print("reproducibility:", r["passed"], r["n_differing"], "differing files")
    with open(os.path.join(args.out_root, f"checks_{args.split}.json"), "w") as f:
        json.dump(out, f, indent=1)
    bad = [k for k, v in out["logs"].items() if not v.get("passed")]
    bad += [k for k, v in out["global"].items() if not v["passed"]]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
