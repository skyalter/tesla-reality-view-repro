#!/usr/bin/env bash
# Render the chase-camera targets of the training logs with the trained Gaussian head of the run
# RV_HEAD_RUN: one process per GPU, each takes every RV_GPUS-th log. Logs that are already complete are
# skipped, so the same command continues an interrupted job. Results go to $RV_SCRATCH/targets/<name> and,
# when the job ends, to the NAS. Extra arguments are handed to scripts/render_targets.py (for example
# --back 9 --up 5 --pitch 15 --hfov 90, or --first 8).
. "${RV_CODE:?submit through lighthouse/submit.sh}/lighthouse/jobs/_env.sh"

CHECKPOINT="$RV_RUNS/${RV_HEAD_RUN:-stage1_head}/last.pt"
if [ ! -f "$CHECKPOINT" ]; then
  echo "TARGETS FAILED: no head checkpoint at $CHECKPOINT"
  exit 1
fi
OUT="$RV_SCRATCH/targets/$RV_RUN_NAME"
mkdir -p "$OUT"
echo "head: $CHECKPOINT   output: $OUT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# One process first: gsplat compiles its CUDA code on first use.
python lighthouse/env_probe.py --gsplat

status=0
pids=()
for i in $(seq 0 $((RV_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES="$i" python scripts/render_targets.py --checkpoint "$CHECKPOINT" \
    --weights-root "$RV_WEIGHTS" --data-root "$RV_PROCESSED" --split train \
    --logs data_lists/av2_train_stage1.txt --first 150 --shard "$i" "$RV_GPUS" --out "$OUT" "$@" \
    > "$OUT/shard_$i.log" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
tail -n 2 "$OUT"/shard_*.log || true

done_logs="$(find "$OUT/train" -maxdepth 2 -name DONE 2>/dev/null | wc -l)"
echo "logs complete: $done_logs"
rv_to_nas "$OUT" "$RV_RUN_NAME" || status=1
rv_log_to_nas "$RV_RUN_NAME"
if [ "$status" = 0 ]; then
  echo "TARGETS OK: $done_logs logs"
else
  echo "TARGETS FAILED OR INCOMPLETE: see $OUT/shard_*.log; submit the same command again to continue"
fi
exit "$status"
