#!/usr/bin/env bash
# Sky scores for every processed image (one GPU), then the data checks and the overlay pictures.
# Logs that already have their sky scores are skipped, so the job can be submitted again.
. "${RV_CODE:?submit through lighthouse/submit.sh}/lighthouse/jobs/_env.sh"

status=0
run() {
  echo "+ $*"
  "$@" || status=1
}
for split in train val; do
  run python scripts/sky_scores.py --out-root "$RV_PROCESSED" --split "$split" \
    --weights "$RV_WEIGHTS/da3metric-large/model.safetensors" --workers "${SLURM_CPUS_PER_TASK:-8}"
done
run python scripts/check_data.py --out-root "$RV_PROCESSED" --split train --overlays --read-samples 1000
run python scripts/check_data.py --out-root "$RV_PROCESSED" --split val --overlays --read-samples 300

rv_to_nas "$RV_PROCESSED" "data_av2_v1" --include "checks_*.json" --include "overlays/**" --include "runs/**" \
  --include "CONFIG.*" --include "CODE_COMMIT.txt" --include "splits/**" || status=1
rv_log_to_nas "data_av2_v1"
[ "$status" = 0 ] && echo "SKY AND CHECKS OK" || echo "SKY OR CHECKS FAILED: see the messages above"
exit "$status"
