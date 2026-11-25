#!/usr/bin/env bash
# Train the backbone on RV_GPUS GPUs of one node, one process per GPU.
# The run stops with a checkpoint shortly before the time limit; submitting the same command again
# continues it. Checkpoints, numbers and logs are copied to the NAS while it runs.
# Extra arguments are configuration overrides (KEY=VALUE).
. "${RV_CODE:?submit through lighthouse/submit.sh}/lighthouse/jobs/_env.sh"

RUN_DIR="$RV_RUNS/$RV_RUN_NAME"
mkdir -p "$RUN_DIR"

# What the training loop runs after every checkpoint and at the end.
if [ -n "${RV_NAS:-}" ]; then
  cat > "$RUN_DIR/sync.sh" <<EOF
#!/usr/bin/env bash
rclone copy "$RUN_DIR" "$RV_NAS/$RV_RUN_NAME" --transfers 2 \\
  --include "*.jsonl" --include "*.json" --include "*.log" --include "last.pt" --include "step_*.pt"
EOF
  SYNC="bash $RUN_DIR/sync.sh"
else
  echo "RV_NAS is not set in lighthouse/local.env: results stay in $RUN_DIR"
  SYNC=""
fi

# Leave 20 minutes of the time limit for the last checkpoint, the evaluation and the copy to the NAS.
MAX_MINUTES="$(( RV_MINUTES > 30 ? RV_MINUTES - 20 : RV_MINUTES / 2 ))"
export OMP_NUM_THREADS=4
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

status=0
python -m torch.distributed.run --nproc_per_node="$RV_GPUS" scripts/train.py --config "configs/$RV_CONFIG.yaml" \
  --set run_dir="$RUN_DIR" data.root="$RV_PROCESSED" model.weights_root="$RV_WEIGHTS" \
        max_minutes="$MAX_MINUTES" sync_command="$SYNC" "$@" || status=1

rv_log_to_nas "$RV_RUN_NAME"
if [ "$status" != 0 ]; then
  echo "TRAINING FAILED: see the messages above"
elif [ -f "$RUN_DIR/done.json" ]; then
  echo "TRAINING FINISHED: $(cat "$RUN_DIR/done.json")"
else
  echo "TRAINING STOPPED BEFORE THE END: submit the same command again to continue from $RUN_DIR/last.pt"
fi
exit "$status"
