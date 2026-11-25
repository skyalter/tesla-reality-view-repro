#!/usr/bin/env bash
# Create the project's conda environment on the cluster. Run once on a login node (needs internet).
#
# Usage: lighthouse/setup_env.sh <cu130|cu126>
#
# Choose the PyTorch build from the GPU driver of the compute nodes (lighthouse/check_cluster.sh --gpu):
#   driver 580 or newer  -> cu130   CUDA 13.0, the same build as the development workstation
#   driver 525 to 579    -> cu126   CUDA 12.6
#
# The environment is called "realityview", like the one on the workstation (set RV_ENV in
# lighthouse/local.env to change the name). It mirrors the workstation's package versions and holds
# everything for preprocessing and training, plus s5cmd and rclone for moving data.
# Packages are pinned in lighthouse/requirements.txt. Nothing from ~/.local is used.
#
# Safe to re-run: an existing environment is kept and only missing packages are installed.
# Afterwards check it on a GPU node with lighthouse/test_env.sh.
set -euo pipefail

. "$(dirname "$0")/common.sh"

# Not named BUILD or HOST: activating an environment that holds conda's compilers overwrites those.
TORCH_BUILD="${1:-}"
case "$TORCH_BUILD" in
  cu130) CUDA_VERSION=13.0 ;;
  cu126) CUDA_VERSION=12.6 ;;
  *) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
TORCH_VERSION=2.12.1
TORCHVISION_VERSION=0.27.1
export PYTHONNOUSERSITE=1

rv_conda
if conda env list | awk '{print $1}' | grep -qx "$RV_ENV"; then
  echo "== environment $RV_ENV already exists; installing into it"
else
  echo "== creating environment $RV_ENV (Python 3.12, CUDA $CUDA_VERSION compiler for gsplat, s5cmd, rclone)"
  conda create -y -n "$RV_ENV" --override-channels -c conda-forge \
    python=3.12 pip ninja s5cmd rclone "cuda-version=$CUDA_VERSION" cuda-nvcc cuda-cudart-dev cuda-cccl
fi
conda activate "$RV_ENV"
conda env config vars set PYTHONNOUSERSITE=1 >/dev/null

echo "== PyTorch $TORCH_VERSION ($TORCH_BUILD)"
python -m pip install --no-cache-dir "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
  --index-url "https://download.pytorch.org/whl/$TORCH_BUILD"
echo "== other packages"
python -m pip install --no-cache-dir -r "$LH_DIR/requirements.txt"

echo "== installed"
python - <<'PY'
import importlib
import sys

print("python", sys.version.split()[0])
for name in ("torch", "torchvision", "numpy", "pandas", "pyarrow", "cv2", "scipy", "PIL", "matplotlib", "einops",
             "safetensors", "yaml", "imageio", "gsplat", "lpips"):
    module = importlib.import_module(name)
    print(f"{name} {getattr(module, '__version__', '?')}")
import torch

print("torch built for CUDA", torch.version.cuda, "| a login node has no GPU, so cuda available =", torch.cuda.is_available())
PY
echo "s5cmd $(s5cmd version)   rclone $(rclone version | head -n 1)"
echo "environment: $CONDA_PREFIX"
echo "next: lighthouse/test_env.sh <gpu type>   to check it on a GPU node"
