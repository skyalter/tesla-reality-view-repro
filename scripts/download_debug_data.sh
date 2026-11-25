#!/usr/bin/env bash
# Small local set for development and unit tests on a workstation. The full data lives on the cluster
# (see lighthouse/download_data.sh).
#
#   1. the validation logs reserved for reporting, complete (with stereo)        about 23 GB for 20 logs
#   2. poses, calibration and 3D boxes of all 700 training logs                  about 0.5 GB
#      (needed to rank the training logs; no images, no lidar)
#   3. pretrained weights                                                        about 3.5 GB
#
# Usage:
#   scripts/download_debug_data.sh --data-root <dir> --weights-root <dir>
#                                  [--val-logs N] [--skip-val] [--skip-train-meta] [--skip-weights] [--dry-run]
#
#   --data-root     Argoverse 2 Sensor root; logs go to <dir>/<split>/<log_id>/
#   --weights-root  pretrained weights go to <dir>/<name>/
#   --val-logs N    only the first N reporting logs (default: all 20)
#
# Safe to re-run: anything already complete is skipped.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
LISTS="$HERE/../data_lists"
DATA_ROOT="" WEIGHTS_ROOT="" VAL_LOGS=0
SKIP_VAL=0 SKIP_TRAIN_META=0 SKIP_WEIGHTS=0
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --weights-root) WEIGHTS_ROOT="$2"; shift 2 ;;
    --val-logs) VAL_LOGS="$2"; shift 2 ;;
    --skip-val) SKIP_VAL=1; shift ;;
    --skip-train-meta) SKIP_TRAIN_META=1; shift ;;
    --skip-weights) SKIP_WEIGHTS=1; shift ;;
    --dry-run) EXTRA+=(--dry-run); shift ;;
    -h|--help) sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$DATA_ROOT" ] || [ "$SKIP_VAL$SKIP_TRAIN_META" = 11 ] || { echo "--data-root is required" >&2; exit 2; }
[ -n "$WEIGHTS_ROOT" ] || [ "$SKIP_WEIGHTS" = 1 ] || { echo "--weights-root is required" >&2; exit 2; }

if [ "$SKIP_VAL" = 0 ]; then
  echo "== validation logs for reporting (complete)"
  "$HERE/av2_download.sh" --split val --list "$LISTS/av2_val_report20.txt" --dest "$DATA_ROOT" \
    --limit "$VAL_LOGS" ${EXTRA[@]+"${EXTRA[@]}"}
fi

if [ "$SKIP_TRAIN_META" = 0 ]; then
  echo "== poses, calibration and 3D boxes of all training logs"
  "$HERE/av2_download.sh" --split train --list "$LISTS/av2_train_log_city.tsv" --dest "$DATA_ROOT" \
    --meta-only ${EXTRA[@]+"${EXTRA[@]}"}
fi

if [ "$SKIP_WEIGHTS" = 0 ]; then
  echo "== pretrained weights"
  python3 "$HERE/download_weights.py" --dest "$WEIGHTS_ROOT" --set stage1 ${EXTRA[@]+"${EXTRA[@]}"}
fi
echo "done"
