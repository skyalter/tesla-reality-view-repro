#!/usr/bin/env python3
"""Render the chase-camera targets of stage 2 from a trained Gaussian head.

For every moment t of a log the model runs once (inputs: the 7 cameras at t and at the earlier input moments)
and gives Gaussians in the reference frame of t plus, per Gaussian, the number of other input views that
confirm its depth (``count_observations``). The target of moment t merges the Gaussians of the moments
t + c for c in ``--context`` (frames, 20 Hz; static Gaussians from all of them, moving objects from t only),
moved into the reference frame of t, and renders them from a camera ``--back`` metres behind and ``--up``
metres above the rear axle, pitched down by ``--pitch`` degrees (default: 15 m, 9 m, 18 degrees, 80 degrees
horizontal field of view).

    python scripts/render_targets.py --checkpoint <head run>/last.pt --weights-root <weights dir> \\
        --data-root <processed root> --split train --logs data_lists/av2_train_stage1.txt --first 2 --out <dir>

Output, per log in <out>/<split>/<log id>/:
    rgb/<ts>.jpg      chase image (ts = timestamp of the front-center image at t, ns)
    conf/<ts>.png     confidence alpha x V x D, x 255 (0 on the ego vehicle)
    alpha/ vis/ dcons/ mover/<ts>.png   the factors and the moving-object share, x 255
    ego_mask.png      where the ego vehicle is (fixed camera: one per log); the confidence is 0 there
    ego_overlay.png   the ego vehicle as an RGBA image to lay over the chase images for display (--ego-model)
    frames.tsv        one line per frame; camera.json   camera and provenance; DONE   when the log is complete
Logs with DONE are skipped.

Debugging without a head checkpoint: ``--untrained --model-name da3-base [--backbone-checkpoint <last.pt>]``
uses an untrained appearance branch with every opacity set to ``--opacity``.
The ego vehicle (``--ego-model``): by default the model in assets/ego_models/tesla_my_2025_bloxbloger
("2025 Tesla Model Y" by BloxBloger, CC BY-NC 4.0, see the README there), drawn in grey-blue; 'sedan' is a plain car built in code, and
any .glb or .obj file can be given with --ego-model-axes, --ego-model-size and --ego-model-color.
Optional clean-up before rendering (all off by default): ``--drop-sky``, ``--drop-depth-jump`` and
``--drop-oversized`` leave out Gaussians that only look right from the input cameras (sky, the borders between
an object and its background, oversized blobs); ``--max-height``, ``--max-forward`` and ``--max-side`` keep
only a fixed region around the vehicle, the same for every frame. See ``cleanup_mask`` and ``region_mask`` in
``realityview.render.targets``.
``--preview`` renders a few moments of the first log from several candidate cameras into one comparison image
and a table (holes and confidence per camera).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# named car models: file, (forward, up) axes of the file, (length, rear overhang) in metres, replaced colours,
# materials that are not drawn
EGO_MODELS = {
    "tesla-my": {
        "file": REPO / "assets/ego_models/tesla_my_2025_bloxbloger/2025_tesla_model_y.glb",
        "axes": ("+z", "+y"), "size": (4.79, 0.97),
        "colors": {"Material_2125765635": (0.42, 0.50, 0.60)},     # the paint: grey-blue
        "skip": ("glass_headlight222",),                           # dark lamp glass that hides the red tail light
    },
}
DEFAULT_CANDIDATES = "15,9,18,80;13,9,18,80;10,6,14,80;7,4,12,70"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", help="last.pt of a stage1_head run")
    p.add_argument("--weights-root", required=True, help="directory holding <model name>/model.safetensors")
    p.add_argument("--data-root", required=True, help="processed root")
    p.add_argument("--split", default="train")
    p.add_argument("--logs", nargs="+", required=True, help="a log list file or log ids")
    p.add_argument("--first", type=int, default=None, help="only the first N logs")
    p.add_argument("--shard", type=int, nargs=2, default=None, metavar=("INDEX", "COUNT"),
                   help="of those logs, only every COUNT-th starting at INDEX (to split the work over GPUs)")
    p.add_argument("--out", required=True)
    p.add_argument("--back", type=float, default=15.0, help="camera distance behind the rear axle, m")
    p.add_argument("--up", type=float, default=9.0, help="camera height above the rear axle, m")
    p.add_argument("--pitch", type=float, default=18.0, help="downward pitch, degrees")
    p.add_argument("--hfov", type=float, default=80.0, help="horizontal field of view, degrees")
    p.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    p.add_argument("--near", type=float, default=7.0,
                   help="Gaussians closer than this to the chase camera (metres along its optical axis) are not "
                        "drawn: the virtual camera would otherwise sit inside tree crowns and other things "
                        "above the road. With the default camera the road at the lower image edge is about "
                        "12 m away, the ego vehicle further.")
    p.add_argument("--ego-model", default="tesla-my",
                   help="the car laid over the chase images for display: 'tesla-my' (the model in assets/), "
                        "'sedan' (built in code, no file), the path of a .glb or .obj file, or 'none'")
    p.add_argument("--ego-model-axes", nargs=2, default=None, metavar=("FORWARD", "UP"),
                   help="the axes of the file that point to the front and the top of the car (a file: +x +z)")
    p.add_argument("--ego-model-size", type=float, nargs=2, default=None, metavar=("LENGTH", "OVERHANG"),
                   help="the model is scaled to LENGTH metres and its rear end put OVERHANG metres behind the "
                        "rear axle (a file: 4.87 1.10)")
    p.add_argument("--ego-model-color", action="append", default=[], metavar="MATERIAL=R,G,B",
                   help="replace the colour of a material of a .glb (0..1); can be given several times")
    p.add_argument("--ego-mask", default=None, choices=("box", "model"),
                   help="the area without supervision (confidence 0): 'model' = the outline of --ego-model grown "
                        "by --ego-mask-grow pixels (default), 'box' = the projected box around the vehicle "
                        "(default with --ego-model none)")
    p.add_argument("--ego-mask-grow", type=int, default=4)
    p.add_argument("--context", type=int, nargs="+", default=(-40, -20, 0, 20, 40),
                   help="moments merged into one target, front-center frames relative to t (20 Hz)")
    p.add_argument("--stride", type=int, default=1, help="render every n-th moment")
    p.add_argument("--max-moments", type=int, default=None, help="debugging: only the first n targets per log "
                                                                   "(the log is then not marked DONE)")
    p.add_argument("--tolerance", type=float, default=0.05, help="relative depth tolerance of an observation")
    p.add_argument("--drop-sky", type=float, default=None, metavar="SCORE",
                   help="leave out the Gaussians of pixels with a sky score >= SCORE (0.5 is the sky mask of the "
                        "model that made the scores; needs the sky scores of scripts/sky_scores.py)")
    p.add_argument("--drop-depth-jump", type=float, default=None, metavar="REL",
                   help="leave out the Gaussians of pixels where the predicted depth steps by more than REL "
                        "(relative, for example 0.2) to a neighbouring pixel")
    p.add_argument("--drop-oversized", type=float, default=None, metavar="FACTOR",
                   help="leave out Gaussians whose longest axis is more than FACTOR times the size of their "
                        "source pixel at their depth (for example 4)")
    p.add_argument("--max-height", type=float, default=None, metavar="M",
                   help="leave out Gaussians more than M metres above the road (the ground plane of the ego "
                        "vehicle at the moment the Gaussian comes from)")
    p.add_argument("--max-forward", type=float, default=None, metavar="M",
                   help="leave out Gaussians more than M metres ahead of the rear axle of the ego vehicle at t")
    p.add_argument("--max-side", type=float, default=None, metavar="M",
                   help="leave out Gaussians more than M metres to the left or right of the ego vehicle at t")
    p.add_argument("--mode", default="antialiased", choices=("antialiased", "classic"))
    p.add_argument("--workers", type=int, default=2, help="data loader processes")
    p.add_argument("--device", default="cuda")
    p.add_argument("--untrained", action="store_true", help="debugging: untrained appearance branch")
    p.add_argument("--model-name", default="da3-base", help="with --untrained")
    p.add_argument("--backbone-checkpoint", default=None, help="with --untrained: last.pt of a backbone run")
    p.add_argument("--opacity", type=float, default=0.95, help="with --untrained: opacity of every Gaussian")
    p.add_argument("--preview", action="store_true", help="compare candidate cameras on a few moments")
    p.add_argument("--preview-moments", type=int, default=6)
    p.add_argument("--candidates", default=DEFAULT_CANDIDATES,
                   help="cameras for --preview: 'back,up,pitch,hfov;...'")
    args = p.parse_args(argv)
    if not args.untrained and not args.checkpoint:
        p.error("--checkpoint is required (or --untrained for debugging)")
    if 0 not in args.context:
        p.error("--context must contain 0")
    if args.ego_model in EGO_MODELS and not EGO_MODELS[args.ego_model]["file"].is_file():
        print(f"ego model {args.ego_model}: {EGO_MODELS[args.ego_model]['file']} is missing, using 'sedan'",
              flush=True)
        args.ego_model = "sedan"
    if args.ego_mask is None:
        args.ego_mask = "box" if args.ego_model == "none" else "model"
    if args.ego_mask == "model" and args.ego_model == "none":
        p.error("--ego-mask model needs --ego-model")
    return args


# ------------------------------------------------------------------------------------------------ setup
def build_model(args):
    """(model in evaluation mode on the device, task, provenance dict)."""
    from realityview.tasks.stage1_head import Stage1HeadTask, load_trained_model

    if args.untrained:
        cfg = {"model": {"name": args.model_name, "weights_root": args.weights_root,
                         "backbone_checkpoint": args.backbone_checkpoint, "exposure": False},
               "data": {"root": args.data_root}}
        task = Stage1HeadTask(cfg)
        model = task.build_model()
        info = {"untrained": True, "model_name": args.model_name, "opacity": args.opacity,
                "backbone_checkpoint": os.path.abspath(args.backbone_checkpoint) if args.backbone_checkpoint else None}
    else:
        model, task = load_trained_model(args.checkpoint, args.weights_root, args.data_root)
        info = {"untrained": False, "checkpoint": os.path.abspath(args.checkpoint), "step": task.step,
                "model_name": task.m["name"]}
    return model.to(args.device).eval(), task, info


def build_dataset(task, args, logs):
    from realityview.data.head_dataset import Stage1HeadDataset

    d = task.d
    return Stage1HeadDataset(args.data_root, args.split, logs, input_offsets=tuple(d["input_offsets"]),
                             front_center_mode=d["front_center_mode"], depth_layer=d["depth_layer"],
                             with_sky=args.drop_sky is not None, target_offsets=[0], min_displacement_m=0.0, seed=task.seed)


# ------------------------------------------------------------------------------------------------ moments
@torch.no_grad()
def moment_scene(model, task, batch, args) -> dict:
    """Run the model on one moment: its Gaussians (CPU), observation counts and frames."""
    from realityview.render.targets import cleanup_mask, count_observations

    keep, mover = task.gaussian_masks(batch)
    out = model(batch["images"], batch["ref_T_cam"], batch["K"], batch["dt"], keep, mover)
    g = out["gaussians"][0].detach()
    if args.untrained:
        g = g.replace(opacities=torch.full_like(g.opacities, args.opacity))
    n_obs = count_observations(g, out["depth"][0], batch["K"][0], batch["ref_T_cam"][0], args.tolerance)
    if any(x is not None for x in (args.drop_sky, args.drop_depth_jump, args.drop_oversized, args.max_height)):
        ok = cleanup_mask(g, keep[0], out["depth"][0], batch.get("sky", [None])[0], sky_score=args.drop_sky,
                          depth_jump=args.drop_depth_jump, max_size=args.drop_oversized,
                          max_height=args.max_height, ref_T_ego=batch["ref_T_ego"][0])
        g = g.__class__(**{k: v[ok] for k, v in g.__dict__.items()})
        n_obs = n_obs[ok]
    fc = (batch["slot"][0] == task.t_slot) & (batch["cam_id"][0] == 0)
    return {"g": g.__class__(**{k: v.cpu() for k, v in g.__dict__.items()}), "n_obs": n_obs.cpu(),
            "city_T_ref": batch["city_T_ref"][0].double().cpu(), "ref_T_ego": batch["ref_T_ego"][0].float().cpu(),
            "timestamp_ns": int(batch["timestamp_ns"][0][fc][0]), "background": out["background"].detach().cpu()}


def scenes(model, task, ds, log_index, targets, args):
    """Yield (t, context moments, cache) for every target moment t, in order.

    Every moment runs through the model once; its result stays in the cache (CPU) until no later target
    needs it."""
    from torch.utils.data import DataLoader, Subset

    from realityview.engine.trainer import to_device

    index = {i: idx for idx, (li, i) in enumerate(ds.samples) if li == log_index}
    need = {t: [t + c for c in sorted(args.context) if t + c in index] for t in targets}
    order = sorted(set().union(*need.values())) if need else []
    loader = DataLoader(Subset(ds, [index[m] for m in order]), batch_size=1, num_workers=args.workers,
                        multiprocessing_context="spawn" if args.workers > 0 else None)
    pending = list(targets)
    # earliest moment still needed by this target or any later one (a later target can need an earlier
    # moment than the next one does: t + 20 uses moment t again)
    earliest = {}
    low = math.inf
    for t in reversed(pending):
        low = min(low, min(need[t]))
        earliest[t] = low
    cache = {}
    for m, batch in zip(order, loader):
        cache[m] = moment_scene(model, task, to_device(batch, args.device), args)
        while pending and max(need[pending[0]]) <= m:
            t = pending.pop(0)
            yield t, need[t], cache
            low = earliest[pending[0]] if pending else math.inf
            for k in [k for k in cache if k < low]:
                del cache[k]


def merged(t, context, cache, device, args=None):
    """The Gaussians of the context moments in the reference frame of t (float64 transforms); with ``args``,
    only those inside the region --max-forward / --max-side around the ego vehicle at t."""
    from realityview.render.targets import merge_moments, region_mask

    ref_T_city = torch.linalg.inv(cache[t]["city_T_ref"])
    parts = []
    for j in context:
        c = cache[j]
        g = c["g"].__class__(**{k: v.to(device) for k, v in c["g"].__dict__.items()})
        parts.append((g, c["n_obs"].to(device), None if j == t else ref_T_city @ c["city_T_ref"], j == t))
    g, n_obs, current = merge_moments(parts)
    if args is not None and (args.max_forward is not None or args.max_side is not None):
        ok = region_mask(g, cache[t]["ref_T_ego"].to(device), args.max_forward, args.max_side)
        g, n_obs, current = g.__class__(**{k: v[ok] for k, v in g.__dict__.items()}), n_obs[ok], current[ok]
    return g, n_obs, current


def ego_layers(args, ref_T_ego, ref_T_cam, K, H, W):
    """(ego mask (H, W) bool on the device of K, overlay (H, W, 4) float RGBA in RGB order or None)."""
    from realityview.render.ego_overlay import load_mesh, place_mesh, render_mesh, sedan_mesh
    from realityview.render.targets import ego_box_mask, overlay_mask

    overlay = None
    mask = ego_box_mask(ref_T_ego, ref_T_cam, K, H, W)
    if args.ego_model != "none":
        if args.ego_model == "sedan":
            mesh = sedan_mesh()
        else:
            preset = EGO_MODELS.get(args.ego_model, {"file": args.ego_model, "axes": ("+x", "+z"),
                                                     "size": (4.87, 1.10), "colors": {}})
            colors = dict(preset["colors"])
            colors.update({k: tuple(float(x) for x in v.split(","))
                           for k, v in (item.split("=", 1) for item in args.ego_model_color)})
            path = str(preset["file"])
            extra = {"colors": colors, "skip": preset.get("skip", ())} if path.lower().endswith(".glb") else {}
            length, overhang = args.ego_model_size or preset["size"]
            mesh = place_mesh(load_mesh(path, **extra), *(args.ego_model_axes or preset["axes"]),
                              length=length, rear_overhang=overhang)
        ego_T_cam = (torch.linalg.inv(ref_T_ego.double()) @ ref_T_cam.double()).cpu().numpy()
        overlay = render_mesh(mesh, ego_T_cam, K.double().cpu().numpy(), H, W)
        if args.ego_mask == "model":
            body = render_mesh(mesh, ego_T_cam, K.double().cpu().numpy(), H, W, shadow=0.0, supersample=2)
            mask = overlay_mask(body, args.ego_mask_grow).to(K.device)
    return mask, overlay


def to_u8(x: torch.Tensor) -> np.ndarray:
    return (x.clamp(0, 1) * 255.0).round().byte().cpu().numpy()


def rgb_u8(rgb: torch.Tensor) -> np.ndarray:
    """(3, H, W) 0..1 -> (H, W, 3) uint8 BGR for cv2."""
    return np.ascontiguousarray(to_u8(rgb).transpose(1, 2, 0)[:, :, ::-1])


# ------------------------------------------------------------------------------------------------ targets
def render_log(model, task, ds, log_index, args, info):
    from realityview.engine.checkpoint import code_version
    from realityview.render.camera import chase_camera
    from realityview.render.ego_overlay import save_overlay
    from realityview.render.targets import EGO_BOX, render_chase

    log_id = ds.logs[log_index][0]
    folder = os.path.join(args.out, args.split, log_id)
    if os.path.isfile(os.path.join(folder, "DONE")):
        print(f"{log_id}: done, skipped", flush=True)
        return
    for sub in ("rgb", "conf", "alpha", "vis", "dcons", "mover"):
        os.makedirs(os.path.join(folder, sub), exist_ok=True)
    moments = sorted(i for li, i in ds.samples if li == log_index)
    targets = moments[::max(1, args.stride)]
    if args.max_moments:
        targets = targets[:args.max_moments]
    W, H = args.size
    dev = args.device
    ego_mask = None
    rows = []
    t0 = time.time()
    for t, context, cache in scenes(model, task, ds, log_index, targets, args):
        start = time.time()
        g, n_obs, _ = merged(t, context, cache, dev, args)
        ref_T_ego = cache[t]["ref_T_ego"].to(dev)
        ref_T_cam, K = chase_camera(ref_T_ego, args.back, args.up, args.pitch, args.hfov, W, H)
        if ego_mask is None:
            ego_mask, overlay = ego_layers(args, ref_T_ego, ref_T_cam, K, H, W)
            cv2.imwrite(os.path.join(folder, "ego_mask.png"), ego_mask.byte().cpu().numpy() * 255)
            if overlay is not None:
                save_overlay(os.path.join(folder, "ego_overlay.png"), overlay)
            ego_T_cam = torch.linalg.inv(ref_T_ego) @ ref_T_cam
            camera = {"back_m": args.back, "up_m": args.up, "pitch_down_deg": args.pitch, "hfov_deg": args.hfov,
                      "width": W, "height": H, "K": K.cpu().tolist(), "K_convention": "integer pixel centres",
                      "ego_T_cam": ego_T_cam.cpu().tolist(), "camera_axes": "OpenCV (x right, y down, z forward)",
                      "ego_box": EGO_BOX, "ego_mask": args.ego_mask, "ego_mask_grow_px": args.ego_mask_grow,
                      "ego_model": args.ego_model if args.ego_model in ("sedan", "none", *EGO_MODELS)
                      else os.path.basename(args.ego_model), "context_frames": list(args.context), "stride": args.stride,
                      "tolerance": args.tolerance, "mode": args.mode, "near_m": args.near,
                      "cleanup": {"drop_sky": args.drop_sky, "drop_depth_jump": args.drop_depth_jump,
                                  "drop_oversized": args.drop_oversized, "max_height_m": args.max_height,
                                  "max_forward_m": args.max_forward, "max_side_m": args.max_side},
                      "model": info,
                      "code_version": code_version()}
            with open(os.path.join(folder, "camera.json"), "w") as f:
                json.dump(camera, f, indent=2)
        out = render_chase(g, n_obs, ref_T_cam, K, H, W, background=cache[t]["background"].to(dev), mode=args.mode,
                           near=args.near)
        conf = out["confidence"].masked_fill(ego_mask, 0.0)
        ts = cache[t]["timestamp_ns"]
        cv2.imwrite(os.path.join(folder, "rgb", f"{ts}.jpg"), rgb_u8(out["rgb"]), [cv2.IMWRITE_JPEG_QUALITY, 95])
        for sub, x in (("conf", conf), ("alpha", out["alpha"]), ("vis", out["visibility"]),
                       ("dcons", out["depth_consistency"]), ("mover", out["mover"])):
            cv2.imwrite(os.path.join(folder, sub, f"{ts}.png"), to_u8(x))
        outside = ~ego_mask
        rows.append([t, ts, ",".join(str(j) for j in context), f"{float(conf[outside].mean()):.4f}",
                     f"{float((conf[outside] > 0.5).float().mean()):.4f}",
                     f"{float((out['alpha'][outside] > 0.5).float().mean()):.4f}", len(g)])
        print(f"{log_id[:8]} moment {t}: {len(g) / 1e6:.2f} M Gaussians, conf>0.5 {rows[-1][4]}, "
              f"alpha>0.5 {rows[-1][5]}, render {time.time() - start:.2f} s, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        del g, n_obs, out
    with open(os.path.join(folder, "frames.tsv"), "w") as f:
        f.write("moment\ttimestamp_ns\tcontext\tmean_conf\tconf_gt_0.5\talpha_gt_0.5\tgaussians\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"{log_id}: {len(rows)} frames in {time.time() - t0:.1f} s "
          f"({(time.time() - t0) / max(1, len(rows)):.2f} s per frame)", flush=True)
    if not args.max_moments:
        open(os.path.join(folder, "DONE"), "w").close()


def preview(model, task, ds, args):
    """One image: a row per candidate camera, a column per moment, for the first log; and a table with, per
    camera, the share of pixels outside the ego mask with opacity > 0.5 and their mean confidence."""
    from realityview.render.camera import chase_camera
    from realityview.render.ego_overlay import composite
    from realityview.render.targets import render_chase

    cams = [tuple(float(x) for x in c.split(",")) for c in args.candidates.split(";") if c.strip()]
    log_id = ds.logs[0][0]
    moments = sorted(i for li, i in ds.samples if li == 0)
    n = min(args.preview_moments, len(moments))
    picks = sorted({moments[round(k * (len(moments) - 1) / max(1, n - 1))] for k in range(n)})
    W, H = args.size
    tw, th = W // 2, H // 2
    tiles = {}
    layers = {}
    stats = {r: [] for r in range(len(cams))}
    for t, context, cache in scenes(model, task, ds, 0, picks, args):
        g, n_obs, _ = merged(t, context, cache, args.device, args)
        ref_T_ego = cache[t]["ref_T_ego"].to(args.device)
        for r, (back, up, pitch, hfov) in enumerate(cams):
            ref_T_cam, K = chase_camera(ref_T_ego, back, up, pitch, hfov, W, H)
            out = render_chase(g, n_obs, ref_T_cam, K, H, W, background=cache[t]["background"].to(args.device),
                               mode=args.mode, near=args.near)
            image = rgb_u8(out["rgb"])
            if r not in layers:                             # the camera is fixed in the ego frame
                layers[r] = ego_layers(args, ref_T_ego, ref_T_cam, K, H, W)
            ego_mask, overlay = layers[r]
            if overlay is not None:
                image = composite(image, np.concatenate([overlay[..., 2::-1], overlay[..., 3:]], -1))
            edge = cv2.morphologyEx(ego_mask.byte().cpu().numpy(), cv2.MORPH_GRADIENT,
                                    np.ones((5, 5), np.uint8)) > 0
            image[edge] = (0, 255, 255)                     # outline of the ego mask
            outside = ~ego_mask
            conf = out["confidence"][outside]
            stats[r].append((float((out["alpha"][outside] > 0.5).float().mean()), float(conf.mean()),
                             float((conf > 0.5).float().mean())))
            tiles[(r, t)] = (cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA), stats[r][-1][2])
        print(f"preview moment {t}: {len(g) / 1e6:.2f} M Gaussians", flush=True)
        del g, n_obs
    grid = np.zeros((len(cams) * th, len(picks) * tw, 3), np.uint8)
    for r, (back, up, pitch, hfov) in enumerate(cams):
        for c, t in enumerate(picks):
            tile, share = tiles[(r, t)]
            tile = tile.copy()
            labels = (f"back {back:g} m, up {up:g} m, pitch {pitch:g} deg, hfov {hfov:g} deg",
                      f"moment {t}, conf>0.5: {share:.2f}")
            tile[:52, :460] //= 3                           # darker band behind the labels
            for k, label in enumerate(labels):
                cv2.putText(tile, label, (8, 20 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                            cv2.LINE_AA)
            grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"preview_{args.split}_{log_id}.jpg")
    cv2.imwrite(path, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    table = os.path.join(args.out, f"preview_{args.split}_{log_id}.tsv")
    with open(table, "w") as f:
        f.write("back_m\tup_m\tpitch_down_deg\thfov_deg\talpha_gt_0.5\tmean_conf\tconf_gt_0.5\n")
        for r, cam in enumerate(cams):
            mean = np.mean(np.asarray(stats[r]), axis=0)
            f.write("\t".join(f"{x:g}" for x in cam) + "\t" + "\t".join(f"{x:.4f}" for x in mean) + "\n")
    print(f"preview: {path} and {table}", flush=True)


def main(argv=None) -> int:
    from realityview.tasks.stage1_backbone import read_logs

    args = parse_args(argv)
    one = args.logs[0]
    spec = one if len(args.logs) == 1 and (os.path.isfile(one) or one.endswith(".txt")) else list(args.logs)
    logs = read_logs(spec, args.first)
    if args.shard:
        index, count = args.shard
        if not 0 <= index < count:
            raise SystemExit("--shard INDEX COUNT needs 0 <= INDEX < COUNT")
        logs = logs[index::count]
    torch.backends.cuda.matmul.allow_tf32 = False
    model, task, info = build_model(args)
    if args.preview:
        logs = logs[:1]
    ds = build_dataset(task, args, logs)
    if ds.skipped:
        print(f"logs that are not ready, skipped: {ds.skipped}", flush=True)
    if args.preview:
        preview(model, task, ds, args)
        return 0
    for li in range(len(ds.logs)):
        render_log(model, task, ds, li, args, info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
