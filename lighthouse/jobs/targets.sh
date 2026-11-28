#!/usr/bin/env bash
# Render the chase-camera targets of a list of logs with the trained Gaussian head of the run RV_HEAD_RUN:
# one process per GPU, each takes every RV_GPUS-th log. Logs that are already complete are skipped, so the
# same command continues an interrupted job. Results go to $RV_SCRATCH/targets/<name>/<split>/ and, when the
# job ends, to the NAS.
#
# Which logs: RV_SPLIT (train or val), RV_LOGS and RV_FIRST, set by submit.sh --split/--logs/--first.
# Defaults: the first 150 logs of data_lists/av2_train_stage1.txt for train, all of
# data_lists/av2_val_dev10.txt for val.
#
# How they are rendered: RV_TARGET_ARGS, by default the picture size of the diffusion model and the clean-up
# and fixed region chosen on 2026-11-27 (see below); the chase camera is the default of
# scripts/render_targets.py. Extra arguments of the job are handed to scripts/render_targets.py after these,
# so they win (for example --preview, or --back 10 --up 6 --pitch 14).
. "${RV_CODE:?submit through lighthouse/submit.sh}/lighthouse/jobs/_env.sh"

CHECKPOINT="$RV_RUNS/${RV_HEAD_RUN:-stage1_head}/last.pt"
if [ ! -f "$CHECKPOINT" ]; then
  echo "TARGETS FAILED: no head checkpoint at $CHECKPOINT"
  exit 1
fi
SPLIT="${RV_SPLIT:-train}"
if [ "$SPLIT" = val ]; then
  LOGS="${RV_LOGS:-data_lists/av2_val_dev10.txt}"; FIRST="${RV_FIRST:-}"
else
  LOGS="${RV_LOGS:-data_lists/av2_train_stage1.txt}"; FIRST="${RV_FIRST:-150}"
fi
# Size 832x480; drop Gaussians that are oversized (4 times their pixel footprint), sky (score 0.5 or more) or
# at a depth step (20 %); keep what is below 8 m above the road, within 60 m ahead and 20 m to each side.
TARGET_ARGS="${RV_TARGET_ARGS:---size 832 480 --drop-oversized 4 --drop-sky 0.5 --drop-depth-jump 0.2 --max-height 8 --max-forward 60 --max-side 20}"
OUT="$RV_SCRATCH/targets/$RV_RUN_NAME"
mkdir -p "$OUT"
echo "head: $CHECKPOINT   output: $OUT/$SPLIT"
echo "logs: $SPLIT, $LOGS${FIRST:+, the first $FIRST}   render arguments: $TARGET_ARGS $*"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# One process first: gsplat compiles its CUDA code on first use.
python lighthouse/env_probe.py --gsplat

status=0
pids=()
for i in $(seq 0 $((RV_GPUS - 1))); do
  # shellcheck disable=SC2086  # TARGET_ARGS and FIRST are lists of words
  CUDA_VISIBLE_DEVICES="$i" python scripts/render_targets.py --checkpoint "$CHECKPOINT" \
    --weights-root "$RV_WEIGHTS" --data-root "$RV_PROCESSED" --split "$SPLIT" \
    --logs "$LOGS" ${FIRST:+--first $FIRST} --shard "$i" "$RV_GPUS" --out "$OUT" $TARGET_ARGS "$@" \
    > "$OUT/shard_${SPLIT}_$i.log" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
tail -n 2 "$OUT"/shard_"${SPLIT}"_*.log || true

# A preview makes no per-log folders: a missing "$OUT/$SPLIT" must not stop the job before the copy to the NAS.
done_logs="$({ find "$OUT/$SPLIT" -maxdepth 2 -name DONE 2>/dev/null || true; } | wc -l)"
echo "logs complete ($SPLIT): $done_logs"
rv_to_nas "$OUT" "$RV_RUN_NAME" || status=1
rv_log_to_nas "$RV_RUN_NAME"
if [ "$status" = 0 ]; then
  echo "TARGETS OK: $done_logs logs ($SPLIT)"
else
  echo "TARGETS FAILED OR INCOMPLETE: see $OUT/shard_${SPLIT}_*.log; submit the same command again to continue"
fi
exit "$status"
