# Shared settings for the cluster scripts. Sourced, never executed.
# Site-specific values come from lighthouse/local.env (see local.env.example).

LH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$LH_DIR/.." && pwd)"

if [ -f "$LH_DIR/local.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$LH_DIR/local.env"
  set +a
fi

# Name of the project's conda environment; the same name as on the development workstation.
RV_ENV="${RV_ENV:-realityview}"

# Call from scripts that need the storage layout.
rv_require_scratch() {
  : "${RV_SCRATCH:?RV_SCRATCH is not set; copy lighthouse/local.env.example to lighthouse/local.env and fill it in}"
  RV_RAW="${RV_RAW:-$RV_SCRATCH/raw/argoverse2_sensor}"
  RV_WEIGHTS="${RV_WEIGHTS:-$RV_SCRATCH/weights}"
}

# Makes the conda command available in this shell.
rv_conda() {
  : "${RV_CONDA_SH:?RV_CONDA_SH is not set in lighthouse/local.env}"
  [ -f "$RV_CONDA_SH" ] || { echo "conda.sh not found: $RV_CONDA_SH" >&2; return 1; }
  # conda's hook reads unset variables; relax "set -u" while it runs.
  set +u
  # shellcheck disable=SC1090
  . "$RV_CONDA_SH"
}

# Activates the project environment. Packages in ~/.local are never mixed in.
rv_activate() {
  rv_conda
  conda activate "$RV_ENV" || { echo "environment $RV_ENV is missing; run lighthouse/setup_env.sh" >&2; return 1; }
  export PYTHONNOUSERSITE=1
}

# Settings for compiling gsplat's CUDA code and for loading the result. Call after rv_activate.
rv_gsplat_env() {
  # gsplat compiles on first use, with the CUDA toolkit and the compiler inside the environment.
  export CUDA_HOME="$CONDA_PREFIX"
  export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
  export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  # A GPU generation the build was not compiled for generally cannot run it, and every job shares one
  # build, so name all of them: compute capability 8.9 is the L40S, 9.0 the H100. Other hardware: set
  # RV_CUDA_ARCH_LIST in local.env.
  export TORCH_CUDA_ARCH_LIST="${RV_CUDA_ARCH_LIST:-8.9;9.0}"
  # Code from that compiler needs the environment's own C++ runtime, but PyTorch loads the system's copy
  # first. Where the system is older than the compiler (RHEL 8, for one), gsplat's extension then fails
  # to import with "version `CXXABI_1.3.15' not found". Preloading the environment's copy makes every
  # process started from here use it.
  local runtime="$CONDA_PREFIX/lib/libstdc++.so.6"
  if [ -e "$runtime" ]; then
    case ":${LD_PRELOAD:-}:" in
      *":$runtime:"*) ;;
      *) export LD_PRELOAD="$runtime${LD_PRELOAD:+:$LD_PRELOAD}" ;;
    esac
  else
    echo "WARNING: $runtime is missing; gsplat's extension may fail to load" >&2
  fi
}
