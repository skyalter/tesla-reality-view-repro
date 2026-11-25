"""Peak GPU memory and time of one stage-1 training step, and DA3METRIC sky-score throughput.

    python -m realityview.models.benchmark step --model da3-large-1.1 --views 14 [--ckpt] [--head-ckpt]
    python -m realityview.models.benchmark maxviews --model da3-large-1.1 [--ckpt]
    python -m realityview.models.benchmark sky --images 64 --chunk 16

A step is: forward (with the frozen reference unless --no-ref), a stand-in loss on the depth, confidence and reference depth,
backward, AdamW step (fused). Inputs are random 518 x 392 images with plausible poses; one sample per step.
The first step (allocations, kernel selection) is not timed; peak memory is measured over the timed steps,
after the optimizer state exists. Weights: ``--weights`` or ``$RV_WEIGHTS_ROOT``. Prints one JSON line.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from .sky import SkyScorer
from .stage1 import Stage1Model


def _inputs(n, dev, H=392, W=518):
    g = torch.Generator().manual_seed(0)
    imgs = torch.randint(0, 256, (1, n, 3, H, W), generator=g, dtype=torch.uint8)
    T = torch.eye(4).repeat(1, n, 1, 1)
    T[0, :, 0, 3] = torch.linspace(-0.3, 0.3, n)
    T[0, n // 2:, 2, 3] = -5.0
    K = torch.tensor([[260.0, 0, W / 2], [0, 260.0, H / 2], [0, 0, 1]]).repeat(1, n, 1, 1)
    dt = torch.zeros(1, n)
    dt[0, n // 2:] = -0.5
    return imgs.to(dev), T.to(dev), K.to(dev), dt.to(dev)


def _step(m, opt, inp):
    out = m(*inp)
    target = out["ref_log_depth"] + 0.1 if "ref_log_depth" in out else out["log_depth"].detach() + 0.1
    loss = (out["log_depth"] - target).abs().mean() + 1e-3 * out["depth_conf"].mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


def bench_step(model, views, ckpt, head_ckpt, freeze, steps, weights=None, model_obj=None, reference=True):
    dev = torch.device("cuda")
    m = model_obj or Stage1Model(model, weights, freeze=freeze, grad_checkpoint=ckpt, head_checkpoint=head_ckpt,
                                 frozen_reference=reference).to(dev)
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-6, fused=True)
    inp = _inputs(views, dev)
    res = {"model": model, "views": views, "grad_checkpoint": ckpt, "head_checkpoint": head_ckpt,
           "freeze": freeze, "reference": m.reference is not None, "trainable_M": round(sum(p.numel() for p in m.trainable_parameters()) / 1e6, 1)}
    try:
        _step(m, opt, inp)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        for _ in range(steps):
            _step(m, opt, inp)
        torch.cuda.synchronize()
        res["seconds_per_step"] = round((time.perf_counter() - t) / steps, 3)
        res["peak_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        res["peak_reserved_GiB"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        res["ok"] = True
    except torch.OutOfMemoryError:
        res["ok"] = False
        res["error"] = "out of memory"
    del opt, inp
    return res


def bench_maxviews(model, ckpt, head_ckpt, freeze, weights=None):
    dev = torch.device("cuda")
    m = Stage1Model(model, weights, freeze=freeze, grad_checkpoint=ckpt, head_checkpoint=head_ckpt).to(dev)
    best, rows = 0, []
    for n in range(1, 15):
        torch.cuda.empty_cache()
        r = bench_step(model, n, ckpt, head_ckpt, freeze, 1, model_obj=m)
        rows.append({k: r.get(k) for k in ("views", "ok", "peak_GiB", "seconds_per_step")})
        if not r["ok"]:
            break
        best = n
    return {"model": model, "grad_checkpoint": ckpt, "head_checkpoint": head_ckpt, "freeze": freeze,
            "max_views": best, "rows": rows}


def bench_sky(n_images, chunk, weights=None):
    dev = torch.device("cuda")
    s = SkyScorer(weights).to(dev)
    imgs = torch.randint(0, 256, (n_images, 3, 392, 518), dtype=torch.uint8, device=dev)
    s(imgs[:chunk], chunk_size=chunk)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    s(imgs, chunk_size=chunk)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    return {"sky_images": n_images, "chunk": chunk, "images_per_second": round(n_images / dt, 1),
            "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["step", "maxviews", "sky"])
    ap.add_argument("--model", default="da3-large-1.1")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--views", type=int, default=14)
    ap.add_argument("--ckpt", action="store_true", help="checkpoint the backbone blocks")
    ap.add_argument("--head-ckpt", action="store_true", help="checkpoint the depth head per chunk of views")
    ap.add_argument("--freeze", default="default")
    ap.add_argument("--no-ref", action="store_true", help="build the model without the frozen reference")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--images", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=16)
    a = ap.parse_args()
    if a.mode == "step":
        r = bench_step(a.model, a.views, a.ckpt, a.head_ckpt, a.freeze, a.steps, a.weights, reference=not a.no_ref)
    elif a.mode == "maxviews":
        r = bench_maxviews(a.model, a.ckpt, a.head_ckpt, a.freeze, a.weights)
    else:
        r = bench_sky(a.images, a.chunk, a.weights)
    r["gpu"] = torch.cuda.get_device_name()
    r["torch"] = torch.__version__
    print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
