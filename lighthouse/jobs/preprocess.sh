#!/usr/bin/env bash
# Process the raw Argoverse 2 logs of stage 1: resized images, LiDAR depth, boxes, index. CPU only.
# Finished logs are skipped, so the job can simply be submitted again after a failure or a time limit.
. "${RV_CODE:?submit through lighthouse/submit.sh}/lighthouse/jobs/_env.sh"

WORKERS="${SLURM_CPUS_PER_TASK:-8}"
status=0
run() {
  echo "+ $*"
  "$@" || status=1
}
run python scripts/preprocess.py --raw-root "$RV_RAW" --out-root "$RV_PROCESSED" --workers "$WORKERS" \
  --split train --logs data_lists/av2_train_stage1.txt
run python scripts/preprocess.py --raw-root "$RV_RAW" --out-root "$RV_PROCESSED" --workers "$WORKERS" \
  --split val --logs data_lists/av2_val_dev10.txt
run python scripts/preprocess.py --raw-root "$RV_RAW" --out-root "$RV_PROCESSED" --workers "$WORKERS" \
  --split val --logs data_lists/av2_val_report20.txt

rv_to_nas "$RV_PROCESSED" "data_av2_v1" --include "runs/**" --include "logs/*/*/CHECKS.json" --include "CONFIG.*" --include "CODE_COMMIT.txt" \
  --include "splits/**" || status=1
rv_log_to_nas "data_av2_v1"
[ "$status" = 0 ] && echo "PREPROCESS OK" || echo "PREPROCESS FAILED: see the messages above"
exit "$status"
