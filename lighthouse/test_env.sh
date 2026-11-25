#!/usr/bin/env bash
# Check the project environment on a GPU node, through one short Slurm allocation (1 GPU, 15 minutes).
# Run from a login node after lighthouse/setup_env.sh.
#
# Usage: lighthouse/test_env.sh <gpu type> [--gsplat]
#
#   <gpu type>  as Slurm names it, e.g. h100 or l40s (see the node list in lighthouse/check_cluster.sh)
#   --gsplat    also compile gsplat's CUDA extension and render a few Gaussians. The first run compiles
#               for a few minutes; the result is cached in ~/.cache/torch_extensions and reused by every
#               later job, so do this once before any multi-GPU training.
#
# The allocation is given up if it does not start within 5 minutes.
set -euo pipefail

. "$(dirname "$0")/common.sh"

GPU="${1:-}"
[ -n "$GPU" ] && [ "${GPU#--}" = "$GPU" ] || { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
PROBE_ARGS="${2:-}"
: "${RV_ACCOUNT:?RV_ACCOUNT is not set in lighthouse/local.env}"
: "${RV_PARTITION:?RV_PARTITION is not set in lighthouse/local.env}"
: "${RV_CONDA_SH:?RV_CONDA_SH is not set in lighthouse/local.env}"

srun -A "$RV_ACCOUNT" -p "$RV_PARTITION" --gpus-per-node="$GPU:1" -c 8 --mem=32G -t 15 --immediate=300 bash -c "
  set -eo pipefail
  . '$RV_CONDA_SH'
  conda activate '$RV_ENV'
  export PYTHONNOUSERSITE=1
  # gsplat compiles with the CUDA toolkit inside the environment; build for both GPU generations.
  export CUDA_HOME=\"\$CONDA_PREFIX\"
  export CPATH=\"\$CONDA_PREFIX/targets/x86_64-linux/include\${CPATH:+:\$CPATH}\"
  export LIBRARY_PATH=\"\$CONDA_PREFIX/targets/x86_64-linux/lib:\$CONDA_PREFIX/lib\${LIBRARY_PATH:+:\$LIBRARY_PATH}\"
  export TORCH_CUDA_ARCH_LIST='8.9;9.0'
  export MAX_JOBS=8
  echo \"node: \$(hostname)\"
  python '$LH_DIR/env_probe.py' $PROBE_ARGS
"
