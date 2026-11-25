#!/usr/bin/env bash
# Download what stage 1 needs onto the cluster: Argoverse 2 Sensor logs and pretrained weights.
# Run on a node with internet access; on a login node, run it inside tmux so it survives a dropped
# connection. Safe to re-run: finished parts are skipped, interrupted parts resume.
#
# Usage: lighthouse/download_data.sh [--dry-run] [--only train|val|weights] [--workers N]
#
#   train    logs listed in data_lists/av2_train_stage1.txt, without the stereo cameras
#   val      the reporting logs (complete) and the tuning logs (without the stereo cameras)
#   weights  pretrained weights, pinned to exact revisions and verified by SHA-256
#
# Where things go (RV_SCRATCH comes from lighthouse/local.env):
#   $RV_SCRATCH/raw/argoverse2_sensor/<split>/<log_id>/...
#   $RV_SCRATCH/weights/<name>/...
set -euo pipefail

. "$(dirname "$0")/common.sh"
rv_require_scratch

ONLY="" WORKERS=16
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) EXTRA+=(--dry-run); shift ;;
    --only) ONLY="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$ONLY" in ""|train|val|weights) ;; *) echo "--only must be train, val or weights" >&2; exit 2 ;; esac

rv_activate_tools
LISTS="$REPO_DIR/data_lists"
AV2="$REPO_DIR/scripts/av2_download.sh"
want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
status=0

echo "raw data -> $RV_RAW"
echo "weights  -> $RV_WEIGHTS"

if want train; then
  echo "== training logs (no stereo)"
  if [ -f "$LISTS/av2_train_stage1.txt" ]; then
    "$AV2" --split train --list "$LISTS/av2_train_stage1.txt" --dest "$RV_RAW" --no-stereo \
      --workers "$WORKERS" ${EXTRA[@]+"${EXTRA[@]}"} || status=1
  else
    echo "data_lists/av2_train_stage1.txt is not in the repository yet: skipped (pull the latest code first)"
    status=1
  fi
fi

if want val; then
  echo "== validation logs for reporting (complete)"
  "$AV2" --split val --list "$LISTS/av2_val_report20.txt" --dest "$RV_RAW" \
    --workers "$WORKERS" --check-out "$RV_RAW/val_report.download_check.tsv" ${EXTRA[@]+"${EXTRA[@]}"} || status=1
  echo "== validation logs for tuning (no stereo)"
  "$AV2" --split val --list "$LISTS/av2_val_dev10.txt" --dest "$RV_RAW" --no-stereo \
    --workers "$WORKERS" --check-out "$RV_RAW/val_dev.download_check.tsv" ${EXTRA[@]+"${EXTRA[@]}"} || status=1
fi

if want weights; then
  echo "== pretrained weights"
  python3 "$REPO_DIR/scripts/download_weights.py" --dest "$RV_WEIGHTS" --set stage1 ${EXTRA[@]+"${EXTRA[@]}"} || status=1
fi

if [ "$status" = 0 ]; then echo "all requested parts are complete"; else echo "something is incomplete; read the messages above and run again" >&2; fi
exit "$status"
