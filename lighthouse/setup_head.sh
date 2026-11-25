#!/usr/bin/env bash
# What the Gaussian-head training needs on top of the base environment: the lpips package and the VGG
# weights it loads through torchvision (about 550 MB, kept under $RV_SCRATCH/cache/torch).
# Run once on a login node:  lighthouse/setup_head.sh
set -euo pipefail
. "$(dirname "$0")/common.sh"
rv_require_scratch
rv_activate
export TORCH_HOME="${TORCH_HOME:-$RV_SCRATCH/cache/torch}"
mkdir -p "$TORCH_HOME"
python -m pip install --no-cache-dir --no-deps lpips==0.1.4
python - <<'PY'
import importlib.metadata as metadata
import os

import lpips
import torch

net = lpips.LPIPS(net="vgg", verbose=False)
a, b = torch.rand(1, 3, 64, 64), torch.rand(1, 3, 64, 64)
print("lpips", metadata.version("lpips"), "| distance of two random images:", round(float(net(a, b, normalize=True)), 4))
print("weights cached under", os.environ["TORCH_HOME"])
PY
echo "HEAD SETUP OK"
