#!/usr/bin/env python3
"""Sky scores for the stored images of processed logs, from DA3METRIC-LARGE (needs a GPU).

Usage (from the repository root):
    PYTHONPATH=. python scripts/sky_scores.py --out-root <processed>/av2_v1 --split val \
        --weights <weights dir>/da3metric-large/model.safetensors [--logs <list or ids>] [--redo]

For every image of a log, img/<cam>/<ts>.jpg, it writes sky/<cam>/<ts>.png: the raw sky output of the model
(not a probability; ground and buildings are 0, sky is around 1) times 100, as uint8, at the stored
resolution. No threshold is applied here. The portrait front-center image (518x684) is padded to 518x686
for the network and the result is cut back. A log is finished when sky/DONE exists; finished logs are
skipped. Only loose log directories are handled: compute the scores before packing a log into a zip.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from realityview.data.pipeline import code_commit  # noqa: E402
from realityview.data.splits import read_list  # noqa: E402
from realityview.data.store import SKY_DONE, LogStore, img_relpath, sha256_file, sky_relpath  # noqa: E402
from realityview.models.sky import SKY_UINT8_SCALE, SkyScorer, sky_score_to_uint8  # noqa: E402

PATCH = 14


class StoredImages(Dataset):
    """The stored images of one log that share one size, as RGB uint8 tensors."""

    def __init__(self, log_dir: str, items):
        self.log_dir, self.items = log_dir, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        cam, ts = self.items[i]
        im = cv2.imread(os.path.join(self.log_dir, img_relpath(cam, ts)), cv2.IMREAD_COLOR)
        if im is None:
            raise IOError(f"cannot read {img_relpath(cam, ts)}")
        return torch.from_numpy(np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))), i


def pad_to_patch(images: torch.Tensor) -> torch.Tensor:
    """Pad H and W up to multiples of 14 by repeating the last row / column."""
    h, w = images.shape[-2:]
    ph, pw = (-h) % PATCH, (-w) % PATCH
    if ph == 0 and pw == 0:
        return images
    return torch.nn.functional.pad(images.float(), (0, pw, 0, ph), mode="replicate").round().to(images.dtype)


def score_log(log_dir: str, scorer: SkyScorer, batch: int, workers: int, info: dict) -> dict:
    store = LogStore(log_dir)
    frames = store.parquet("frames.parquet")
    items = [(c, int(t)) for c, t in zip(frames["cam"], frames["timestamp_ns"])]
    sizes = {}
    for cam in sorted(set(c for c, _ in items)):
        first = next(t for c, t in items if c == cam)
        sizes[cam] = cv2.imread(os.path.join(log_dir, img_relpath(cam, first)), cv2.IMREAD_COLOR).shape[:2]
    for cam in sizes:
        os.makedirs(os.path.dirname(os.path.join(log_dir, sky_relpath(cam, 0))), exist_ok=True)
    started = time.perf_counter()
    for size in sorted(set(sizes.values())):
        group = [(c, t) for c, t in items if sizes[c] == size]
        loader = DataLoader(StoredImages(log_dir, group), batch_size=batch, num_workers=workers,
                            multiprocessing_context="spawn" if workers > 0 else None)
        for images, index in loader:
            h, w = images.shape[-2:]
            sky = scorer(pad_to_patch(images), chunk_size=batch)[:, :h, :w]
            for row, i in zip(sky_score_to_uint8(sky), index.tolist()):
                cam, ts = group[i]
                if not cv2.imwrite(os.path.join(log_dir, sky_relpath(cam, ts)), row, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
                    raise IOError(f"cannot write {sky_relpath(cam, ts)}")
    done = {"images": len(items), "seconds": round(time.perf_counter() - started, 1), **info}
    with open(os.path.join(log_dir, SKY_DONE), "w") as f:
        json.dump(done, f, indent=1)
    return done


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-root", default=os.environ.get("RV_PROCESSED_ROOT"))
    p.add_argument("--split", required=True)
    p.add_argument("--logs", default=None, help="list file or comma-separated ids; default: all processed logs of the split")
    p.add_argument("--weights", required=True, help="model.safetensors of DA3METRIC-LARGE")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=6, help="image-reading processes")
    p.add_argument("--redo", action="store_true", help="recompute logs that already have sky/DONE")
    args = p.parse_args()
    if not args.out_root:
        p.error("--out-root (or RV_PROCESSED_ROOT) is required")
    if not torch.cuda.is_available():
        sys.exit("sky scores need a GPU")

    split_dir = os.path.join(args.out_root, "logs", args.split)
    if args.logs is None:
        logs = sorted(n for n in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, n)) and not n.endswith(".partial"))
    elif os.path.isfile(args.logs):
        logs = read_list(args.logs)
    else:
        logs = [x.strip() for x in args.logs.split(",") if x.strip()]

    scorer = None
    info = {"model": "da3metric-large", "weights_sha256": None, "uint8_scale": SKY_UINT8_SCALE,
            "code_version": code_commit(REPO)}
    failed = 0
    for n, log_id in enumerate(logs, 1):
        log_dir = os.path.join(split_dir, log_id)
        tag = f"[{n}/{len(logs)}] {args.split}/{log_id}"
        if not os.path.isfile(os.path.join(log_dir, "DONE")):
            print(f"{tag}: not processed yet (no DONE), skipped", flush=True)
            failed += 1
            continue
        if os.path.isfile(os.path.join(log_dir, SKY_DONE)) and not args.redo:
            print(f"{tag}: sky scores already there, skipped", flush=True)
            continue
        if scorer is None:
            info["weights_sha256"] = sha256_file(args.weights)
            scorer = SkyScorer(args.weights).cuda()
        try:
            done = score_log(log_dir, scorer, args.batch, args.workers, info)
            print(f"{tag}: {done['images']} images in {done['seconds']:.0f} s", flush=True)
        except Exception as error:  # noqa: BLE001 - one bad log must not stop the others
            print(f"{tag}: ERROR {error!r}", flush=True)
            failed += 1
    print(f"sky scores: {len(logs) - failed} of {len(logs)} logs complete", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
