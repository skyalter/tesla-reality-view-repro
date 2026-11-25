#!/usr/bin/env bash
# Train the Gaussian head on RV_GPUS GPUs of one node, one process per GPU, starting from the backbone of
# the run RV_BACKBONE_RUN. The run stops with a checkpoint shortly before the time limit; submitting the
# same command again continues it. Checkpoints, numbers, comparison images and logs are copied to the NAS
# while it runs. Extra arguments are configuration overrides (KEY=VALUE).
. "${RV_CODE:?submit through lighthouse/submit.sh}/lighthouse/jobs/_env.sh"

RUN_DIR="$RV_RUNS/$RV_RUN_NAME"
mkdir -p "$RUN_DIR"
export RV_BACKBONE_CHECKPOINT="$RV_RUNS/${RV_BACKBONE_RUN:-stage1_backbone}/last.pt"
if [ ! -f "$RV_BACKBONE_CHECKPOINT" ]; then
  echo "HEAD TRAINING FAILED: no backbone checkpoint at $RV_BACKBONE_CHECKPOINT"
  exit 1
fi
echo "backbone: $RV_BACKBONE_CHECKPOINT"

if [ -n "${RV_NAS:-}" ]; then
  cat > "$RUN_DIR/sync.sh" <<SYNC_EOF
#!/usr/bin/env bash
rclone copy "$RUN_DIR" "$RV_NAS/$RV_RUN_NAME" --transfers 2 \\
  --include "*.jsonl" --include "*.json" --include "*.log" --include "last.pt" --include "step_*.pt" \\
  --include "eval_images/**"
SYNC_EOF
  SYNC="bash $RUN_DIR/sync.sh"
else
  echo "RV_NAS is not set in lighthouse/local.env: results stay in $RUN_DIR"
  SYNC=""
fi

MAX_MINUTES="$(( RV_MINUTES > 30 ? RV_MINUTES - 20 : RV_MINUTES / 2 ))"
export OMP_NUM_THREADS=4
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# One process first: gsplat compiles its CUDA code on first use, and several processes compiling at the same
# time overwrite each other. The same call also fetches the LPIPS weights if they are not cached yet.
python lighthouse/env_probe.py --gsplat
python -c "import lpips; lpips.LPIPS(net='vgg', verbose=False); print('lpips weights ready')"

PORT="$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 ))"
status=0
python -m torch.distributed.run --nproc_per_node="$RV_GPUS" --master_port "$PORT" \
  scripts/train.py --config "configs/$RV_CONFIG.yaml" \
  --set run_dir="$RUN_DIR" data.root="$RV_PROCESSED" model.weights_root="$RV_WEIGHTS" \
        max_minutes="$MAX_MINUTES" sync_command="$SYNC" "$@" || status=1

rv_log_to_nas "$RV_RUN_NAME"
if [ "$status" != 0 ]; then
  echo "HEAD TRAINING FAILED: see the messages above"
elif [ -f "$RUN_DIR/done.json" ]; then
  echo "HEAD TRAINING FINISHED: $(cat "$RUN_DIR/done.json")"
else
  echo "HEAD TRAINING STOPPED BEFORE THE END: submit the same command again to continue from $RUN_DIR/last.pt"
fi
exit "$status"
