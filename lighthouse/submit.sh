#!/usr/bin/env bash
# Submit a job. A job runs a snapshot of the committed code, so later edits or pulls do not change a queued
# or running job. Output goes to a log file; nothing has to stay open.
#
# Usage: lighthouse/submit.sh <job> [options] [-- extra arguments]
#
# Stage 1
#   preprocess   images, LiDAR depth, boxes and index of the stage-1 logs           CPU only
#   sky          sky scores, then the data checks and the overlay pictures           1 GPU
#   baseline     depth errors of the original network on the tuning logs             1 GPU
#   train        train the backbone                                                  --gpus N
#   head         train the Gaussian head on the trained backbone                     --gpus N
#   targets      render the chase-camera images of a list of logs with the head      --gpus N
# Stage 2
#   frames       cut the photos of all cameras to the picture size of the diffusion model   CPU only
#   diffusion    train the diffusion model (--config stage2_blank or stage2_fill)    --gpus N
#   sample       sample videos from a trained diffusion model (--run NAME)           1 GPU
#
# Options
#   --gpu TYPE     GPU type as Slurm names it     default: l40s for sky, baseline, targets and sample; h100
#                                                 for train, head and diffusion
#   --gpus N       number of GPUs (train, head, targets, diffusion)   default 8
#   --cpus N       CPU cores                      default: 32 for preprocess, 16 for frames, 8 per GPU otherwise
#   --hours H      time limit in hours            default: preprocess 4, sky 6, baseline 1, train, head and
#                                                 diffusion 24, targets 12, frames 6, sample 4
#   --name NAME    run name: the directory under $RV_SCRATCH/runs (targets: $RV_SCRATCH/targets, frames:
#                  $RV_SCRATCH/video) and on the NAS; for sample, a name other than the run's puts the
#                  videos in <run>/final_samples_<name> instead of <run>/final_samples
#                                                 default: baseline_original / stage1_backbone / stage1_head /
#                                                 targets_v1 / video_v1 / the configuration name (diffusion)
#   --config NAME  file in configs/ without .yaml (baseline, train, head, diffusion)
#                                                 default stage1_backbone; stage1_head for head; none for
#                                                 diffusion (it must be given)
#   --backbone NAME  run name of the backbone training whose last.pt the head starts from (head)
#                                                 default stage1_backbone
#   --head NAME    run name of the head training whose last.pt is used (targets, diffusion, sample)
#                                                 default stage1_head
#   --split SPLIT  train or val (targets, frames)  default: train for targets; both, one after the other, for frames
#   --logs FILE    log list (targets, frames)      default: data_lists/av2_train_stage1.txt for train,
#                                                 data_lists/av2_val_dev10.txt for val
#   --first N      only the first N logs of the list   default: 150 for train, all for val
#                  (frames takes --logs and --first only together with --split)
#   --chase NAME   name of the targets job whose images are the chase targets (diffusion, sample)
#                                                 default targets_v1
#   --video NAME   name of the frames job whose photos are used (diffusion, sample)   default video_v1
#   --run NAME     run name of the diffusion training to sample from (sample)
#   --after JOBID  start only after that job has finished successfully
#   --dry-run      print the sbatch command, submit nothing
#   -- ...         baseline, train, head, diffusion: configuration overrides KEY=VALUE; targets: arguments of
#                  scripts/render_targets.py; sample: arguments of scripts/sample_video.py
#
# Logs: ~/Reality_View/logs/<job>_<name>_<job id>.log
# A training run that reaches its time limit stops with a checkpoint; submit the same command again to
# continue it. To stop one earlier without losing work: touch $RV_SCRATCH/runs/<name>/STOP
set -euo pipefail

. "$(dirname "$0")/common.sh"

usage() { sed -n '2,/^set -euo pipefail/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'; }

JOB="${1:-}"
case "$JOB" in
  preprocess|sky|baseline|train|head|targets|frames|diffusion|sample) shift ;;
  *) usage; exit 2 ;;
esac

GPU="" GPUS=8 CPUS="" HOURS="" NAME="" CONFIG="" BACKBONE="stage1_backbone" HEAD="stage1_head" AFTER="" DRY_RUN=0
SPLIT="" LOGS="" FIRST="" CHASE="targets_v1" VIDEO="video_v1" RUN=""
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --cpus) CPUS="$2"; shift 2 ;;
    --hours) HOURS="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --backbone) BACKBONE="$2"; shift 2 ;;
    --head) HEAD="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --logs) LOGS="$2"; shift 2 ;;
    --first) FIRST="$2"; shift 2 ;;
    --chase) CHASE="$2"; shift 2 ;;
    --video) VIDEO="$2"; shift 2 ;;
    --run) RUN="$2"; shift 2 ;;
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
RUNS="${RV_RUNS:-$RV_SCRATCH/runs}"
if [ -z "$CONFIG" ]; then
  case "$JOB" in
    head) CONFIG=stage1_head ;;
    diffusion) echo "diffusion needs --config: stage2_blank or stage2_fill" >&2; exit 2 ;;
    *) CONFIG=stage1_backbone ;;
  esac
fi
case "$SPLIT" in ""|train|val) ;; *) echo "--split must be train or val" >&2; exit 2 ;; esac

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
  head)
    NAME="${NAME:-stage1_head}"; HOURS="${HOURS:-24}"; GPU="${GPU:-h100}"; CPUS="${CPUS:-$((8 * GPUS))}"
    RESOURCES=(--nodes=1 --gpus-per-node="$GPU:$GPUS" -c "$CPUS" --mem="$((40 * GPUS))G")
    need "$REPO_DIR/configs/$CONFIG.yaml" "no such configuration"
    need "$PROCESSED/logs" "process the data first"
    [ -n "$AFTER" ] || need "$RUNS/$BACKBONE/last.pt" "train the backbone first, or name its run with --backbone"
    ;;
  targets)
    NAME="${NAME:-targets_v1}"; HOURS="${HOURS:-12}"; GPU="${GPU:-l40s}"; CPUS="${CPUS:-$((6 * GPUS))}"
    SPLIT="${SPLIT:-train}"
    RESOURCES=(--nodes=1 --gpus-per-node="$GPU:$GPUS" -c "$CPUS" --mem="$((48 * GPUS))G")
    need "$PROCESSED/logs" "process the data first"
    [ -n "$AFTER" ] || need "$RUNS/$HEAD/last.pt" "train the Gaussian head first, or name its run with --head"
    ;;
  frames)
    NAME="${NAME:-video_v1}"; HOURS="${HOURS:-6}"; CPUS="${CPUS:-16}"
    RESOURCES=(-c "$CPUS" --mem=64G)
    if [ -z "$SPLIT" ] && [ -n "$LOGS$FIRST" ]; then
      echo "frames: --logs and --first need --split train or val" >&2; exit 2
    fi
    need "$RV_RAW/train" "run lighthouse/download_data.sh first"
    need "$PROCESSED/logs" "process the data first"
    ;;
  diffusion)
    NAME="${NAME:-$CONFIG}"; HOURS="${HOURS:-24}"; GPU="${GPU:-h100}"; CPUS="${CPUS:-$((8 * GPUS))}"
    RESOURCES=(--nodes=1 --gpus-per-node="$GPU:$GPUS" -c "$CPUS" --mem="$((48 * GPUS))G")
    need "$REPO_DIR/configs/$CONFIG.yaml" "no such configuration"
    need "$PROCESSED/logs" "process the data first"
    need "$RV_WEIGHTS/wan2.1-t2v-1.3b/diffusion_pytorch_model.safetensors" \
         "download the Wan2.1 weights: python scripts/download_weights.py --dest $RV_WEIGHTS --set stage2"
    need "$RUNS/$HEAD/last.pt" "train the Gaussian head first, or name its run with --head"
    if [ -z "$AFTER" ]; then
      need "$RV_SCRATCH/targets/$CHASE/train" "render the chase targets first (targets job), or name them with --chase"
      need "$RV_SCRATCH/video/$VIDEO/train" "cut the photos first (frames job), or name them with --video"
      case " $* " in
        *" data.eval_logs=null "*) ;;     # no evaluation: the tuning logs are not needed
        *) need "$RV_SCRATCH/targets/$CHASE/val" "the evaluation needs the chase targets of the tuning logs: targets --split val (or pass -- data.eval_logs=null)"
           need "$RV_SCRATCH/video/$VIDEO/val" "the evaluation needs the photos of the tuning logs: frames (or pass -- data.eval_logs=null)" ;;
      esac
    fi
    ;;
  sample)
    [ -n "$RUN" ] || { echo "sample needs --run NAME: the diffusion run to sample from" >&2; exit 2; }
    NAME="${NAME:-$RUN}"; HOURS="${HOURS:-4}"; CPUS="${CPUS:-8}"; GPU="${GPU:-l40s}"; GPUS=1
    RESOURCES=(--gpus-per-node="$GPU:1" -c "$CPUS" --mem=64G)
    [ -n "$AFTER" ] || need "$RUNS/$RUN/last.pt" "train the diffusion model first, or name its run with --run"
    need "$RUNS/$HEAD/last.pt" "the Gaussian head the diffusion model was trained with; name its run with --head"
    ;;
esac
MINUTES="$(awk -v h="$HOURS" 'BEGIN { m = int(h * 60 + 0.5); print (m < 5 ? 5 : m) }')"

# sbatch splits --export at commas.
for value in "$NAME" "$CONFIG" "$BACKBONE" "$HEAD" "$SPLIT" "$LOGS" "$FIRST" "$CHASE" "$VIDEO" "$RUN" "$RV_SCRATCH" "$HOME"; do
  case "$value" in *,*) echo "a name or path given to a job cannot contain a comma: $value" >&2; exit 2 ;; esac
done

COMMIT="$(git -C "$REPO_DIR" rev-parse HEAD)"
# The job runs the commit, not the working tree: what it needs must be committed.
in_commit() {
  git -C "$REPO_DIR" cat-file -e "$COMMIT:$1" 2>/dev/null \
    || { echo "$1 is not part of commit ${COMMIT:0:7} (the job runs committed files only): $2" >&2; exit 1; }
}
in_commit "lighthouse/jobs/$JOB.sh" "pull the latest code, or commit the file"
case "$JOB" in baseline|train|head|diffusion) in_commit "configs/$CONFIG.yaml" "pull the latest code, or commit the file" ;; esac
case "$LOGS" in ""|/*) ;; *) in_commit "$LOGS" "a log list inside the repository must be committed; or give an absolute path" ;; esac
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
SBATCH+=(--export="ALL,RV_CODE=$CODE,RV_CODE_VERSION=$COMMIT,RV_JOB=$JOB,RV_RUN_NAME=$NAME,RV_CONFIG=$CONFIG,RV_GPUS=$GPUS,RV_MINUTES=$MINUTES,RV_LOG_DIR=$LOG_DIR,RV_BACKBONE_RUN=$BACKBONE,RV_HEAD_RUN=$HEAD,RV_SPLIT=$SPLIT,RV_LOGS=$LOGS,RV_FIRST=$FIRST,RV_CHASE_RUN=$CHASE,RV_VIDEO_RUN=$VIDEO,RV_SAMPLE_RUN=$RUN"
         "$CODE/lighthouse/jobs/$JOB.sh" "$@")

if [ "$DRY_RUN" = 1 ]; then
  echo "code snapshot: $CODE (commit ${COMMIT:0:7})"
  printf '%q ' "${SBATCH[@]}"; echo
  exit 0
fi

# Results must be able to reach long-term storage before a job is worth queueing.
if [ -n "${RV_NAS:-}" ]; then
  if ! (rv_activate && rclone mkdir "$RV_NAS" && rclone lsf "$RV_NAS" --max-depth 1 >/dev/null); then
    echo "cannot reach $RV_NAS through rclone; run lighthouse/check_nas.sh, or unset RV_NAS to keep results on scratch" >&2
    exit 1
  fi
else
  echo "note: RV_NAS is not set in lighthouse/local.env; results will stay on scratch" >&2
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
