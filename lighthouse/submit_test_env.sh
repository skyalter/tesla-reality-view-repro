#!/usr/bin/env bash
# Queue the environment check as a batch job (1 GPU, 20 minutes). It waits in the queue for as long as
# needed and writes its output to a log file, so no terminal has to stay open.
#
# Usage: lighthouse/submit_test_env.sh <gpu type> [--gsplat]
#
#   <gpu type>  as Slurm names it, e.g. h100 or l40s
#   --gsplat    also compile gsplat's CUDA extension and render a few Gaussians
#
# Log: ~/Reality_View/logs/test_env_<gpu type>_<job id>.log   (the last line is ALL OK on success)
set -euo pipefail

. "$(dirname "$0")/common.sh"

GPU="${1:-}"
[ -n "$GPU" ] && [ "${GPU#--}" = "$GPU" ] || { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
shift
: "${RV_ACCOUNT:?RV_ACCOUNT is not set in lighthouse/local.env}"
: "${RV_PARTITION:?RV_PARTITION is not set in lighthouse/local.env}"

LOG_DIR="$HOME/Reality_View/logs"
mkdir -p "$LOG_DIR"
JOB_ID="$(sbatch --parsable -A "$RV_ACCOUNT" -p "$RV_PARTITION" --gpus-per-node="$GPU:1" -c 8 --mem=32G -t 20 \
  -J "rv-test-env-$GPU" -o "$LOG_DIR/test_env_${GPU}_%j.log" \
  --export=ALL,RV_LH_DIR="$LH_DIR" "$LH_DIR/test_env_job.sh" "$@")"
echo "submitted job $JOB_ID"
echo "queue:  squeue -u $USER"
echo "log:    $LOG_DIR/test_env_${GPU}_${JOB_ID}.log"
