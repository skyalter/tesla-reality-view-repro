# Common start of every job script. Sourced, never executed.
# sbatch runs a copy of the job script, so the code location comes in through RV_CODE (set by submit.sh).
set -eo pipefail
. "${RV_CODE:?RV_CODE is not set; submit through lighthouse/submit.sh}/lighthouse/common.sh"
rv_require_scratch
rv_activate

export RV_RAW RV_WEIGHTS
export RV_PROCESSED="${RV_PROCESSED:-$RV_SCRATCH/processed/av2_v1}"
export RV_RUNS="${RV_RUNS:-$RV_SCRATCH/runs}"
export PYTHONPATH="$RV_CODE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# gsplat: compile settings and the environment's own C++ runtime (see common.sh).
rv_gsplat_env
# Weights that libraries download themselves (torchvision VGG for LPIPS) live on scratch.
export TORCH_HOME="${TORCH_HOME:-$RV_SCRATCH/cache/torch}"
cd "$RV_CODE"
echo "== ${RV_JOB:-job} '${RV_RUN_NAME:-}' on $(hostname), Slurm job ${SLURM_JOB_ID:-none}, code ${RV_CODE_VERSION:-unknown}, $(date '+%F %T')"

# rv_to_nas <local directory> <name under $RV_NAS> [rclone filter options]
# Copies results to the NAS. Without RV_NAS it only says where the results are.
rv_to_nas() {
  local src="$1" name="$2"
  shift 2
  if [ -z "${RV_NAS:-}" ]; then
    echo "RV_NAS is not set in lighthouse/local.env: nothing copied to the NAS; results are in $src"
    return 0
  fi
  if rclone copy "$src" "$RV_NAS/$name" --transfers 4 "$@"; then
    echo "copied to $RV_NAS/$name"
  else
    echo "WARNING: copying $src to $RV_NAS/$name failed"
    return 1
  fi
}

# Copies this job's own log file next to the results on the NAS.
rv_log_to_nas() {
  local log="${RV_LOG_DIR:-}/${RV_JOB:-}_${RV_RUN_NAME:-}_${SLURM_JOB_ID:-}.log"
  [ -n "${RV_NAS:-}" ] && [ -f "$log" ] || return 0
  rclone copy "$log" "$RV_NAS/$1/logs" || echo "WARNING: copying the job log to the NAS failed"
}
