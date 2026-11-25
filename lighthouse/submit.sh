#!/usr/bin/env bash
# Submit one of the stage-1 jobs. A job runs a snapshot of the committed code, so later edits or pulls
# do not change a queued or running job. Output goes to a log file; nothing has to stay open.
#
# Usage: lighthouse/submit.sh <job> [options] [-- KEY=VALUE ...]
#
#   preprocess   images, LiDAR depth, boxes and index of the stage-1 logs           CPU only
#   sky          sky scores, then the data checks and the overlay pictures           1 GPU
#   baseline     depth errors of the original network on the tuning logs             1 GPU
#   train        train the backbone                                                  --gpus N
#
# Options
#   --gpu TYPE     GPU type as Slurm names it     default: l40s for sky and baseline, h100 for train
#   --gpus N       number of GPUs for train       default 8
#   --cpus N       CPU cores                      default: 32 for preprocess, 8 per GPU otherwise
#   --hours H      time limit in hours            default: preprocess 4, sky 6, baseline 1, train 24
#   --name NAME    run name: the directory under $RV_SCRATCH/runs and on the NAS
#                                                 default: baseline_original / stage1_backbone
#   --config NAME  file in configs/ without .yaml (baseline, train)   default stage1_backbone
#   --after JOBID  start only after that job has finished successfully
#   --dry-run      print the sbatch command, submit nothing
#   -- KEY=VALUE   configuration overrides handed to the script (baseline, train)
#
# Logs: ~/Reality_View/logs/<job>_<name>_<job id>.log
# A training run that reaches its time limit stops with a checkpoint; submit the same command again to
# continue it. To stop one earlier without losing work: touch $RV_SCRATCH/runs/<name>/STOP
set -euo pipefail

. "$(dirname "$0")/common.sh"

usage() { sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; }

JOB="${1:-}"
case "$JOB" in
  preprocess|sky|baseline|train) shift ;;
  *) usage; exit 2 ;;
esac

GPU="" GPUS=8 CPUS="" HOURS="" NAME="" CONFIG="stage1_backbone" AFTER="" DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --hours) HOURS="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --after) AFTER="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; break ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

rv_require_scratch
: "${RV_ACCOUNT:?RV_ACCOUNT is not set in lighthouse/local.env}"
: "${RV_PARTITION:?RV_PARTITION is not set in lighthouse/local.env}"
PROCESSED="${RV_PROCESSED:-$RV_SCRATCH/processed/av2_v1}"

need() { [ -e "$1" ] || { echo "missing: $1  ($2)" >&2; exit 1; }; }

RESOURCES=()
case "$JOB" in
  preprocess)
    NAME="${NAME:-data}"; HOURS="${HOURS:-4}"; CPUS="${CPUS:-32}"
    RESOURCES=(-c "$CPUS" --mem=96G)
    need "$RV_RAW/train" "run lighthouse/download_data.sh first"
    need "$RV_RAW/val" "run lighthouse/download_data.sh first"
    ;;
  sky)
    NAME="${NAME:-data}"; HOURS="${HOURS:-6}"; CPUS="${CPUS:-8}"; GPU="${GPU:-l40s}"
    RESOURCES=(--gpus-per-node="$GPU:1" -c "$CPUS" --mem=48G)
    need "$RV_WEIGHTS/da3metric-large/model.safetensors" "run lighthouse/download_data.sh --only weights"
    [ -n "$AFTER" ] || need "$PROCESSED/logs" "submit the preprocess job first, or pass --after <its job id>"
    ;;
  baseline)
    NAME="${NAME:-baseline_original}"; HOURS="${HOURS:-1}"; CPUS="${CPUS:-8}"; GPU="${GPU:-l40s}"
    RESOURCES=(--gpus-per-node="$GPU:1" -c "$CPUS" --mem=48G)
    need "$REPO_DIR/configs/$CONFIG.yaml" "no such configuration"
    [ -n "$AFTER" ] || need "$PROCESSED/logs" "process the data first"
    ;;
  train)
    NAME="${NAME:-stage1_backbone}"; HOURS="${HOURS:-24}"; GPU="${GPU:-h100}"; CPUS="${CPUS:-$((8 * GPUS))}"
    RESOURCES=(--nodes=1 --gpus-per-node="$GPU:$GPUS" -c "$CPUS" --mem="$((40 * GPUS))G")
    need "$REPO_DIR/configs/$CONFIG.yaml" "no such configuration"
    [ -n "$AFTER" ] || need "$PROCESSED/logs" "process the data first"
    ;;
esac
MINUTES="$(awk -v h="$HOURS" 'BEGIN { m = int(h * 60 + 0.5); print (m < 5 ? 5 : m) }')"

COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
if [ -n "$(git -C "$REPO_DIR" status --porcelain --untracked-files=no)" ]; then
  echo "warning: uncommitted changes in $REPO_DIR are NOT part of the job; it runs commit ${COMMIT:0:7}" >&2
fi
if git -C "$REPO_DIR" fetch -q origin 2>/dev/null; then
  if [ "$COMMIT" != "$(git -C "$REPO_DIR" rev-parse origin/main 2>/dev/null)" ]; then
    echo "note: this checkout is not at origin/main; 'git pull --ff-only' gets the latest code" >&2
  fi
fi
CODE="$RV_SCRATCH/code/$COMMIT"
LOG_DIR="$HOME/Reality_View/logs"

SBATCH=(sbatch --parsable -A "$RV_ACCOUNT" -p "$RV_PARTITION" "${RESOURCES[@]}" -t "$MINUTES"
        -J "rv-$JOB-$NAME" -o "$LOG_DIR/${JOB}_${NAME}_%j.log")
[ -z "$AFTER" ] || SBATCH+=(--dependency="afterok:$AFTER")
SBATCH+=(--export="ALL,RV_CODE=$CODE,RV_CODE_VERSION=$COMMIT,RV_JOB=$JOB,RV_RUN_NAME=$NAME,RV_CONFIG=$CONFIG,RV_GPUS=$GPUS,RV_MINUTES=$MINUTES,RV_LOG_DIR=$LOG_DIR"
         "$CODE/lighthouse/jobs/$JOB.sh" "$@")

if [ "$DRY_RUN" = 1 ]; then
  echo "code snapshot: $CODE (commit ${COMMIT:0:7})"
  printf '%q ' "${SBATCH[@]}"; echo
  exit 0
fi

if [ ! -f "$CODE/.complete" ]; then
  mkdir -p "$CODE"
  git -C "$REPO_DIR" archive "$COMMIT" | tar -x -C "$CODE"
  touch "$CODE/.complete"
fi
mkdir -p "$LOG_DIR"
JOB_ID="$("${SBATCH[@]}")"
echo "submitted job $JOB_ID  ($JOB, code ${COMMIT:0:7}, time limit $MINUTES min)"
echo "queue:  squeue -u $USER"
echo "log:    $LOG_DIR/${JOB}_${NAME}_${JOB_ID}.log"
