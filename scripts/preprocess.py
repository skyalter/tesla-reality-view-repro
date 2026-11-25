#!/usr/bin/env python3
"""Offline preprocessing of Argoverse 2 Sensor logs for stage 1 (images, LiDAR depth, boxes, index).

Usage (from the repository root, nothing installed):
    PYTHONPATH=. python scripts/preprocess.py --raw-root <av2 sensor root> --out-root <processed>/av2_v1 \
        --split val --logs data_lists/av2_val_report20.txt --workers 8 [--first N]

Each log goes to <out-root>/logs/<split>/<log_id>/ and is finished when its DONE file exists; finished
logs are skipped on a rerun (use --redo to force). Incomplete raw logs (e.g. training logs downloaded with
poses and boxes only) are reported and skipped, and make the exit status non-zero.
Sky scores are not produced by this version (they need the DA3METRIC model); sky/ is left empty.
A JSON report of the run is written to <out-root>/runs/.
"""
from __future__ import annotations

import os

# one thread per process: the pool provides the parallelism (set before numpy / OpenCV load)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import cv2  # noqa: E402

from realityview.data.config import CAMS, default_config, init_root_config  # noqa: E402
from realityview.data.images import ego_masks_for_log  # noqa: E402
from realityview.data.pipeline import (IncompleteLogError, _worker_init, code_commit, is_done,  # noqa: E402
                                       process_log)
from realityview.data.splits import read_list  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-root", default=os.environ.get("AV2_RAW_ROOT"),
                   help="Argoverse 2 Sensor root holding train/ val/ (env AV2_RAW_ROOT)")
    p.add_argument("--out-root", default=os.environ.get("RV_PROCESSED_ROOT"),
                   help="output root, one per config version, e.g. .../av2_v1 (env RV_PROCESSED_ROOT)")
    p.add_argument("--split", required=True, choices=["train", "val", "test"])
    p.add_argument("--logs", default=None,
                   help="log list file (one id per line, or a TSV with log_id) or comma-separated ids; "
                        "default: every log directory under <raw-root>/<split>")
    p.add_argument("--first", type=int, default=None, help="only the first N logs of the list")
    p.add_argument("--workers", type=int, default=1, help="processes for images and depth (spawned)")
    p.add_argument("--ego-mask-dir", default=os.path.join(REPO, "assets", "ego_masks", "argoverse2"))
    p.add_argument("--redo", action="store_true", help="reprocess logs that already have DONE")
    return p.parse_args()


def resolve_logs(args) -> list[str]:
    if args.logs is None:
        d = os.path.join(args.raw_root, args.split)
        ids = sorted(n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n)))
    elif os.path.isfile(args.logs):
        ids = read_list(args.logs)
    else:
        ids = [x.strip() for x in args.logs.split(",") if x.strip()]
    return ids[: args.first] if args.first else ids


def main():
    args = parse_args()
    if not args.raw_root or not args.out_root:
        sys.exit("--raw-root and --out-root (or AV2_RAW_ROOT / RV_PROCESSED_ROOT) are required")
    cfg = default_config()
    init_root_config(args.out_root, cfg)
    commit = code_commit(REPO)
    with open(os.path.join(args.out_root, "CODE_COMMIT.txt"), "a") as f:
        f.write(f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} {commit}\n")
    # dataset-level ego masks at the stored resolution, one per camera (all-zero where the car is not seen);
    # the reader requires all seven. A root never mixes masks: existing ones must be identical.
    em_dir = os.path.join(args.out_root, "ego_masks")
    os.makedirs(em_dir, exist_ok=True)
    for cam, m in ego_masks_for_log(args.ego_mask_dir, CAMS, cfg).items():
        f = os.path.join(em_dir, f"{cam}.png")
        if os.path.isfile(f):
            old = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if old is None or old.shape != m.shape or not ((old > 0) == m).all():
                sys.exit(f"{f} differs from the mask made from --ego-mask-dir; use a new output root")
        else:
            cv2.imwrite(f, m.astype("uint8") * 255)

    logs = resolve_logs(args)
    pool = None
    if args.workers > 1:
        pool = mp.get_context("spawn").Pool(args.workers, initializer=_worker_init)
    else:
        _worker_init()
    results = []
    t_run = time.perf_counter()
    try:
        for n, log_id in enumerate(logs, 1):
            tag = f"[{n}/{len(logs)}] {args.split}/{log_id}"
            if not args.redo and is_done(args.out_root, args.split, log_id, cfg):
                print(f"{tag}: already done, skipped", flush=True)
                results.append({"log_id": log_id, "split": args.split, "status": "skipped_done"})
                continue
            try:
                r = process_log(args.raw_root, args.split, log_id, args.out_root, cfg, args.ego_mask_dir,
                                pool=pool, n_workers=args.workers, commit=commit)
                ck = "checks passed" if r["checks_passed"] else f"CHECKS FAILED: {', '.join(r['failed_checks'])}"
                print(f"{tag}: done in {r['timings_s']['total_s']:.1f} s, {r['bytes']['total'] / 2**20:.0f} MiB, {ck}",
                      flush=True)
            except IncompleteLogError as e:
                r = {"log_id": log_id, "split": args.split, "status": "incomplete", "error": str(e)}
                print(f"{tag}: INCOMPLETE, skipped: {e}", flush=True)
            except Exception as e:  # keep going with the other logs, but report
                r = {"log_id": log_id, "split": args.split, "status": "error", "error": repr(e),
                     "traceback": traceback.format_exc()}
                print(f"{tag}: ERROR: {e!r}", flush=True)
            results.append(r)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    summary = {k: sum(1 for r in results if r["status"] == k) for k in ("done", "skipped_done", "incomplete", "error")}
    summary["checks_failed"] = sum(1 for r in results if r["status"] == "done" and not r["checks_passed"])
    report = {"started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "split": args.split,
              "workers": args.workers, "commit": commit, "wall_s": time.perf_counter() - t_run,
              "summary": summary, "logs": results}
    os.makedirs(os.path.join(args.out_root, "runs"), exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with open(os.path.join(args.out_root, "runs", f"preprocess_{args.split}_{stamp}.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(summary))
    bad = summary["incomplete"] + summary["error"] + summary["checks_failed"]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
