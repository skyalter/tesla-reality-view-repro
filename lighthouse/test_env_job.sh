#!/usr/bin/env bash
# Body of the environment check; runs on a GPU node. Submitted by lighthouse/submit_test_env.sh.
# sbatch runs a copy of this file, so the repository location comes in through RV_LH_DIR.
set -eo pipefail
. "${RV_LH_DIR:?RV_LH_DIR is not set; submit through lighthouse/submit_test_env.sh}/common.sh"
rv_activate
# gsplat compiles with the CUDA toolkit inside the environment; build for both GPU generations.
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST='8.9;9.0'
export MAX_JOBS=8
echo "node: $(hostname)   job: ${SLURM_JOB_ID:-none}   $(date '+%F %T')"
python "$RV_LH_DIR/env_probe.py" "$@"
