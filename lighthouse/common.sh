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
