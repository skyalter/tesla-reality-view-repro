#!/usr/bin/env python3
"""Cut the photos of the 7 ring cameras to the size of the stage-2 targets (default 832 x 480).

The processed data of stage 1 holds 518 x 392 images, too small for a target, so this reads the original
images again. Each image is cropped to the aspect ratio of the target and resized (INTER_AREA):

    landscape cameras (2048 x 1550):  the middle 2048 x 1182 rows
    front-center (1550 x 2048):       1550 x 894 rows around the centre of the band stage 1 uses

    python scripts/make_video_frames.py --raw-root <AV2 sensor root> --processed-root <processed root> \\
        --split train --logs data_lists/av2_train_stage1.txt --first 150 --out <dir> --workers 8

Output, per log in <out>/<split>/<log id>/:
    <camera>/<timestamp>.jpg   RGB photo at the target size (JPEG quality 95)
    <camera>/ego_mask.png      where the ego vehicle is (255), from the masks of the processed root
    cameras.json               per camera: K at the target size (integer pixel centres), size, crop
    DONE                       when the log is complete (such logs are skipped)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def crop_rows(cam: str, raw_wh, out_wh, cfg: dict) -> tuple[int, int]:
    """(first row, number of rows) of the crop in the original image."""
    from realityview.data.config import FRONT_CENTER

    W0, H0 = raw_wh
    rows = min(H0, int(round(W0 * out_wh[1] / out_wh[0] / 2)) * 2)
    if cam == FRONT_CENTER:
        stored_h = int(cfg["images"]["front_center_wh"][1])
        band_h = int(cfg["images"]["landscape_wh"][1])
        centre = (int(cfg["images"]["front_center_crop_row0"]) + band_h / 2) * H0 / stored_h
    else:
        centre = H0 / 2
    row0 = int(round(centre - rows / 2))
    return max(0, min(H0 - rows, row0)), rows


def one_camera(job):
    log_path, cam, out_dir, out_wh, row0, rows, quality = job
    from realityview.data.av2_io import list_camera_frames

    os.makedirs(out_dir, exist_ok=True)
    ts, files = list_camera_frames(log_path, cam)
    n = 0
    for t, f in zip(ts, files):
        dst = os.path.join(out_dir, f"{int(t)}.jpg")
        if os.path.isfile(dst):
            continue
        im = cv2.imread(f, cv2.IMREAD_COLOR)
        if im is None:
            raise IOError(f"cannot decode {f}")
        small = cv2.resize(im[row0:row0 + rows], tuple(out_wh), interpolation=cv2.INTER_AREA)
        tmp = dst + ".tmp.jpg"
        if not cv2.imwrite(tmp, small, [cv2.IMWRITE_JPEG_QUALITY, quality]):
            raise IOError(f"cannot write {tmp}")
        os.replace(tmp, dst)
        n += 1
    return cam, len(files), n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", default=os.environ.get("AV2_RAW_ROOT"))
    ap.add_argument("--processed-root", default=os.environ.get("RV_PROCESSED_ROOT"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--logs", nargs="+", required=True, help="a log list file or log ids")
    ap.add_argument("--first", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, nargs=2, default=(832, 480), metavar=("W", "H"))
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    if not args.raw_root or not args.processed_root:
        ap.error("--raw-root and --processed-root are needed")

    from realityview.data.av2_io import log_dir, read_calibration
    from realityview.data.config import CAMS, load_root_config
    from realityview.data.dataset import load_root_ego_masks
    from realityview.data.images import crop_intrinsics, scale_intrinsics

    logs = []
    for item in args.logs:
        if os.path.isfile(item):
            logs += [line.split()[0] for line in open(item) if line.strip() and not line.startswith("#")]
        else:
            logs.append(item)
    if args.first:
        logs = logs[:args.first]
    cfg = load_root_config(args.processed_root)
    ego = load_root_ego_masks(args.processed_root, cfg)
    out_wh = tuple(args.size)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for log_id in logs:
            folder = os.path.join(args.out, args.split, log_id)
            if os.path.isfile(os.path.join(folder, "DONE")):
                print(f"{log_id}: done, skipped", flush=True)
                continue
            path = log_dir(args.raw_root, args.split, log_id)
            calib, _ = read_calibration(path)
            t0 = time.time()
            jobs, cameras = [], {}
            for cam in CAMS:
                c = calib[cam]
                row0, rows = crop_rows(cam, (c.width, c.height), out_wh, cfg)
                K = scale_intrinsics(crop_intrinsics(c.K, row0, 0), (c.width, rows), out_wh)
                cameras[cam] = {"K": K.tolist(), "width": out_wh[0], "height": out_wh[1],
                                "K_convention": "integer pixel centres",
                                "crop": {"row0": row0, "rows": rows, "raw_width": c.width, "raw_height": c.height}}
                os.makedirs(os.path.join(folder, cam), exist_ok=True)
                m = cv2.resize(ego[cam].astype(np.uint8), (c.width, c.height), interpolation=cv2.INTER_NEAREST)
                m = cv2.resize(m[row0:row0 + rows], out_wh, interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(folder, cam, "ego_mask.png"), m * 255)
                jobs.append((path, cam, os.path.join(folder, cam), out_wh, row0, rows, args.quality))
            done = pool.map(one_camera, jobs)
            with open(os.path.join(folder, "cameras.json"), "w") as f:
                json.dump(cameras, f, indent=2)
            Path(folder, "DONE").write_text(json.dumps({"images": int(sum(n for _, n, _ in done))}) + "\n")
            print(f"{log_id}: {sum(n for _, n, _ in done)} images ({sum(w for _, _, w in done)} written), "
                  f"{time.time() - t0:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
