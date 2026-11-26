#!/usr/bin/env python3
"""Check the project environment on a GPU node. Run through lighthouse/test_env.sh.

Prints the GPU and driver, runs a small mixed-precision computation, and with --gsplat compiles gsplat's
CUDA extension (first use only) and renders a few Gaussians forward and backward.
"""
from __future__ import annotations

import subprocess
import sys
import time


def cxx_runtime() -> str:
    """The libstdc++ this process has loaded. PyTorch pulls one in on import, and gsplat's extension,
    built by the environment's compiler, needs one at least as new as that compiler."""
    try:
        with open("/proc/self/maps") as maps:
            paths = {line.split()[-1] for line in maps if "libstdc++" in line}
    except OSError:
        return "unknown"
    return ", ".join(sorted(paths)) or "none loaded"


def main() -> int:
    import torch

    print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
    print("C++ runtime:", cxx_runtime())
    if not torch.cuda.is_available():
        print("FAILED: torch cannot see a GPU (driver too old for this build, or no GPU in this allocation)")
        return 1
    print("gpu:", torch.cuda.get_device_name(0), f"| {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB",
          "| compute capability", ".".join(map(str, torch.cuda.get_device_capability(0))))
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        print("driver:", out.splitlines()[0] if out else "unknown")
    except (OSError, subprocess.SubprocessError):
        print("driver: could not run nvidia-smi")

    x = torch.randn(4096, 4096, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = (x @ x).float().mean()
    torch.cuda.synchronize()
    print("mixed-precision matmul ok:", bool(torch.isfinite(y)))

    if "--gsplat" in sys.argv:
        t0 = time.time()
        from gsplat import rasterization

        n, w, h = 2000, 320, 240
        g = torch.Generator(device="cuda").manual_seed(0)
        means = torch.randn(n, 3, device="cuda", generator=g) * torch.tensor([2.0, 1.5, 1.0], device="cuda")
        means[:, 2] += 6.0
        means.requires_grad_(True)
        quats = torch.nn.functional.normalize(torch.randn(n, 4, device="cuda", generator=g), dim=-1)
        scales = torch.full((n, 3), 0.05, device="cuda")
        opacities = torch.full((n,), 0.5, device="cuda")
        colors = torch.rand(n, 3, device="cuda", generator=g)
        viewmat = torch.eye(4, device="cuda")[None]
        k = torch.tensor([[300.0, 0.0, w / 2], [0.0, 300.0, h / 2], [0.0, 0.0, 1.0]], device="cuda")[None]
        try:
            image, alpha, _ = rasterization(means, quats, scales, opacities, colors, viewmat, k, w, h,
                                            render_mode="RGB+ED")
        except ImportError as error:  # the compiled extension is loaded on first use
            if "CXXABI" not in str(error) and "GLIBCXX" not in str(error):
                raise
            print("FAILED: gsplat's compiled extension needs a newer C++ runtime than the one this process loaded")
            print("  loaded:", cxx_runtime())
            print("  error: ", error)
            print("  the cluster scripts preload the environment's copy: see rv_gsplat_env in lighthouse/common.sh")
            return 1
        image[..., :3].mean().backward()
        torch.cuda.synchronize()
        print(f"gsplat render and backward ok in {time.time() - t0:.0f} s (includes compiling on first use):",
              f"image {tuple(image.shape)}, covered pixels {(alpha > 0.01).float().mean().item():.2f},",
              f"gradient finite {bool(torch.isfinite(means.grad).all())}")
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
