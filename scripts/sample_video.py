#!/usr/bin/env python3
"""Sample videos from a trained stage-2 diffusion model.

    python scripts/sample_video.py --checkpoint <run>/last.pt --out <dir> \\
        --split val --logs <log id or list file> --clips 4 \\
        --set data.root=<processed root> data.chase_root=<render_targets output> \\
              model.weights_root=<weights dir> model.stage1.head_checkpoint=<last.pt of the stage1_head run>

The checkpoint holds the trained parts and the configuration of the run; ``--set`` replaces the paths in
it (they are those of the machine the run was made on). Clips are read like the training samples, so the
logs need the processed data and, for the chase camera, the output of scripts/render_targets.py (it gives
the camera and the picture to compare with); for a real camera (``--camera``) they need the output of
scripts/make_video_frames.py instead.

Output, per clip, in <out>/:
    <log>_<moment>_<camera>.mp4     the target on the left, the sample on the right (``--no-target``: the
                                    sample alone); for the chase camera the ego vehicle is laid over both
                                    when the chase folder has ego_overlay.png (``--no-ego`` turns it off)
    samples_<camera>.json           what was sampled and with which settings
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def lay_over(frames: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """frames (T, H, W, 3) uint8, overlay (H, W, 4) uint8 with straight alpha -> frames with the overlay."""
    a = overlay[None, :, :, 3:4].astype(np.float32) / 255.0
    out = overlay[None, :, :, :3].astype(np.float32) * a + frames.astype(np.float32) * (1.0 - a)
    return out.round().clip(0, 255).astype(np.uint8)


def read_overlay(folder: str, size) -> np.ndarray | None:
    import cv2

    from realityview.data.stage2_dataset import fit_image

    path = os.path.join(folder, "ego_overlay.png")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED) if os.path.isfile(path) else None
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        return None
    return fit_image(np.ascontiguousarray(img[:, :, [2, 1, 0, 3]]), size)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="last.pt (or a kept checkpoint) of a stage2_diffusion run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default=None, help="default: the evaluation split of the run")
    ap.add_argument("--logs", nargs="+", default=None, help="a log list file or log ids; default: those of the run")
    ap.add_argument("--first", type=int, default=None, help="only the first N logs")
    ap.add_argument("--camera", default="chase", help='"chase" or the name of a real camera, e.g. ring_front_center')
    ap.add_argument("--clips", type=int, default=4, help="number of clips, spread evenly over the logs")
    ap.add_argument("--starts", type=int, nargs="+", default=None,
                    help="instead of --clips: positions in the list of clip starts (0 = the first clip)")
    ap.add_argument("--steps", type=int, default=None, help="default: that of the run")
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--shift", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-target", action="store_true")
    ap.add_argument("--no-ego", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="replace values of the stored configuration")
    args = ap.parse_args()

    from torch.utils.data import default_collate

    from realityview.data.config import CAMS
    from realityview.engine.config import apply_overrides
    from realityview.engine.trainer import to_device
    from realityview.tasks.stage1_backbone import REPO, read_logs
    from realityview.tasks.stage2_diffusion import load_trained_model, write_video

    chase = args.camera == "chase"
    if not chase and args.camera not in CAMS:
        ap.error(f"--camera must be chase or one of {list(CAMS)}")
    model, task = load_trained_model(args.checkpoint, apply_overrides({}, args.set))
    d = task.d
    split = args.split or d["eval_split"]
    if args.logs:
        one = args.logs[0]
        is_file = len(args.logs) == 1 and (os.path.isfile(one) or os.path.isfile(os.path.join(REPO, one)))
        logs = read_logs(one if is_file else args.logs, args.first)
    else:
        logs = (task.eval_logs() or task.train_logs())[:args.first]
    need = "data.chase_root" if chase else "data.video_root"
    if not d[need.split(".")[1]]:
        ap.error(f"{need} is not set (pass it with --set)")
    dataset = task._dataset(split, logs, start_stride=1, kinds="chase" if chase else "real")
    if dataset.skipped:
        print(f"logs that are not ready, left out: {dataset.skipped}", flush=True)
    if not dataset.starts:
        raise SystemExit("no clips to sample")
    variant = dataset.variants.index(-1 if chase else list(CAMS).index(args.camera))
    if args.starts:
        starts = [s for s in args.starts if 0 <= s < len(dataset.starts)]
    else:
        n = max(1, min(args.clips, len(dataset.starts)))
        starts = np.linspace(0, len(dataset.starts) - 1, n + 2)[1:-1].round().astype(int).tolist()
    s = task.s
    steps = args.steps or int(s["steps"])
    guidance = float(s["guidance"]) if args.guidance is None else args.guidance
    shift = float(s["shift"]) if args.shift is None else args.shift

    device = torch.device(args.device)
    model = model.to(device)
    os.makedirs(args.out, exist_ok=True)
    records = []
    for k, start in enumerate(starts):
        sample = dataset[int(start) * len(dataset.variants) + variant]
        batch = to_device(default_collate([sample]), device)
        target = batch["target"]
        T, hw = target.shape[1], tuple(target.shape[-2:])
        gen = torch.Generator().manual_seed(args.seed + k)
        t0 = time.time()
        latents = model.sample(batch, T, hw, steps=steps, guidance=guidance, shift=shift, generator=gen)
        video = model.decode(latents)[0].permute(0, 2, 3, 1).cpu().numpy()
        seconds = time.time() - t0
        truth = target[0].permute(0, 2, 3, 1).cpu().numpy()
        li, i0 = dataset.starts[int(start)]
        if chase and not args.no_ego:
            overlay = read_overlay(dataset.chase[li]["folder"], (hw[1], hw[0]))
            if overlay is not None:
                video, truth = lay_over(video, overlay), lay_over(truth, overlay)
        frames = video if args.no_target else np.concatenate([truth, video], axis=2)
        name = f"{sample['log_id'][:8]}_{int(sample['moment']):04d}_{args.camera}.mp4"
        path = write_video(os.path.join(args.out, name), frames)
        records.append({"file": os.path.basename(path), "log_id": sample["log_id"], "moment": int(sample["moment"]),
                        "camera": args.camera, "seed": args.seed + k, "seconds": round(seconds, 2)})
        print(f"{os.path.basename(path)}: {T} frames, {seconds:.1f} s", flush=True)
    with open(os.path.join(args.out, f"samples_{args.camera}.json"), "w") as f:
        json.dump({"checkpoint_step": task.step, "split": split, "steps": steps, "guidance": guidance, "shift": shift,
                   "frames": int(d["clip_frames"]), "size": list(d["size"]), "clips": records}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
