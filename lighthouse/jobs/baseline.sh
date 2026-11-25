#!/usr/bin/env bash
# Depth errors of the original pretrained network on the tuning logs, for several pose scales (one GPU).
# The numbers decide the pose scale and show how large each loss term is before training.
# Extra arguments are configuration overrides (KEY=VALUE). RV_POSE_SCALES changes the list of scales.
. "${RV_CODE:?submit through lighthouse/submit.sh}/lighthouse/jobs/_env.sh"

OUT="$RV_RUNS/$RV_RUN_NAME"
mkdir -p "$OUT"
status=0
for scale in ${RV_POSE_SCALES:-20 14 10 6.7}; do
  echo "+ pose scale $scale"
  python scripts/eval_depth.py --config "configs/$RV_CONFIG.yaml" --workers "${SLURM_CPUS_PER_TASK:-8}" \
    --out "$OUT/original_pose_scale_$scale.json" \
    --set data.root="$RV_PROCESSED" model.weights_root="$RV_WEIGHTS" model.pose_scale="$scale" "$@" \
    | tee "$OUT/original_pose_scale_$scale.txt" || status=1
done
rv_to_nas "$OUT" "$RV_RUN_NAME" || status=1
rv_log_to_nas "$RV_RUN_NAME"
[ "$status" = 0 ] && echo "BASELINE OK" || echo "BASELINE FAILED: see the messages above"
exit "$status"
