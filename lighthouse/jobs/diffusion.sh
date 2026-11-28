#!/usr/bin/env bash
# Train the stage-2 diffusion model on RV_GPUS GPUs of one node, one process per GPU. The configuration
# (RV_CONFIG) is configs/stage2_blank.yaml or configs/stage2_fill.yaml. It needs the Gaussian head of the run
# RV_HEAD_RUN, the chase targets of the targets job RV_CHASE_RUN and the photos of the frames job
# RV_VIDEO_RUN, for the training logs and, for the evaluation, the validation logs. The run stops with a
# checkpoint shortly before the time limit; submitting the same command again continues it. Checkpoints,
# numbers, sampled videos and logs are copied to the NAS while it runs. Extra arguments are configuration
# overrides (KEY=VALUE).
. "${RV_CODE:?submit through lighthouse/submit.sh}/lighthouse/jobs/_env.sh"

RUN_DIR="$RV_RUNS/$RV_RUN_NAME"
mkdir -p "$RUN_DIR"
export RV_HEAD_CHECKPOINT="${RV_HEAD_CHECKPOINT:-$RV_RUNS/${RV_HEAD_RUN:-stage1_head}/last.pt}"
export RV_CHASE="${RV_CHASE:-$RV_SCRATCH/targets/${RV_CHASE_RUN:-targets_v1}}"
export RV_VIDEO="${RV_VIDEO:-$RV_SCRATCH/video/${RV_VIDEO_RUN:-video_v1}}"
for path in "$RV_HEAD_CHECKPOINT" "$RV_CHASE/train" "$RV_VIDEO/train" \
            "$RV_WEIGHTS/wan2.1-t2v-1.3b/diffusion_pytorch_model.safetensors" "$RV_WEIGHTS/wan2.1-t2v-1.3b/Wan2.1_VAE.pth"; do
  if [ ! -e "$path" ]; then
    echo "DIFFUSION TRAINING FAILED: missing $path"
    exit 1
  fi
done
case " $* " in
  *" data.eval_logs=null "*) ;;
  *) for path in "$RV_CHASE/val" "$RV_VIDEO/val"; do
       if [ ! -e "$path" ]; then
         echo "DIFFUSION TRAINING FAILED: missing $path (the evaluation needs the tuning logs; or pass data.eval_logs=null)"
         exit 1
       fi
     done ;;
esac
echo "configuration: $RV_CONFIG   head: $RV_HEAD_CHECKPOINT"
echo "chase targets: $RV_CHASE   photos: $RV_VIDEO"

if [ -n "${RV_NAS:-}" ]; then
  # One kept checkpoint is 7 GB: only last.pt goes to the NAS, the kept ones stay on scratch.
  cat > "$RUN_DIR/sync.sh" <<SYNC_EOF
#!/usr/bin/env bash
rclone copy "$RUN_DIR" "$RV_NAS/$RV_RUN_NAME" --transfers 2 \\
  --include "*.jsonl" --include "*.json" --include "*.log" --include "last.pt" --include "samples/**"
SYNC_EOF
  SYNC="bash $RUN_DIR/sync.sh"
else
  echo "RV_NAS is not set in lighthouse/local.env: results stay in $RUN_DIR"
  SYNC=""
fi

MAX_MINUTES="$(( RV_MINUTES > 30 ? RV_MINUTES - 20 : RV_MINUTES / 2 ))"
export OMP_NUM_THREADS=4
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

PORT="$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 ))"
status=0
python -m torch.distributed.run --nproc_per_node="$RV_GPUS" --master_port "$PORT" \
  scripts/train.py --config "configs/$RV_CONFIG.yaml" \
  --set run_dir="$RUN_DIR" data.root="$RV_PROCESSED" model.weights_root="$RV_WEIGHTS" \
        max_minutes="$MAX_MINUTES" sync_command="$SYNC" "$@" || status=1

rv_log_to_nas "$RV_RUN_NAME"
if [ "$status" != 0 ]; then
  echo "DIFFUSION TRAINING FAILED: see the messages above"
elif [ -f "$RUN_DIR/done.json" ]; then
  echo "DIFFUSION TRAINING FINISHED: $(cat "$RUN_DIR/done.json")"
else
  echo "DIFFUSION TRAINING STOPPED BEFORE THE END: submit the same command again to continue from $RUN_DIR/last.pt"
fi
exit "$status"
