#!/usr/bin/env bash
# Sample videos from the trained diffusion model of the run RV_SAMPLE_RUN, on one GPU. For the training logs
# and for the validation logs, and separately for the chase camera and for two real cameras (the control: the
# camera is removed from the inputs and drawn from the others), RV_SAMPLE_CLIPS clips each (default 4).
# Results: $RV_RUNS/<run>/final_samples/<split>/<camera>/ with, per clip, a video and a sheet of four frames
# (front-center photo | target | sample) and samples_<camera>.json with the numbers; all of it is copied to
# the NAS under <run>/final_samples/. With a --name other than the run's the folder is
# final_samples_<name>, so that a second sampling with other settings does not replace the first.
# Extra arguments are handed to scripts/sample_video.py (for example --steps 50, or --set sample.guidance=2).
. "${RV_CODE:?submit through lighthouse/submit.sh}/lighthouse/jobs/_env.sh"

RUN_DIR="$RV_RUNS/${RV_SAMPLE_RUN:?submit with --run NAME}"
CHECKPOINT="$RUN_DIR/last.pt"
HEAD_CHECKPOINT="${RV_HEAD_CHECKPOINT:-$RV_RUNS/${RV_HEAD_RUN:-stage1_head}/last.pt}"
CHASE="${RV_CHASE:-$RV_SCRATCH/targets/${RV_CHASE_RUN:-targets_v1}}"
VIDEO="${RV_VIDEO:-$RV_SCRATCH/video/${RV_VIDEO_RUN:-video_v1}}"
for path in "$CHECKPOINT" "$HEAD_CHECKPOINT"; do
  if [ ! -f "$path" ]; then
    echo "SAMPLES FAILED: missing $path"
    exit 1
  fi
done
FOLDER="final_samples"
[ "${RV_RUN_NAME:-$RV_SAMPLE_RUN}" = "$RV_SAMPLE_RUN" ] || FOLDER="final_samples_$RV_RUN_NAME"
OUT="$RUN_DIR/$FOLDER"
CLIPS="${RV_SAMPLE_CLIPS:-4}"
CAMERAS="${RV_SAMPLE_CAMERAS:-chase ring_front_center ring_side_left}"
echo "checkpoint: $CHECKPOINT   output: $OUT   clips per split and camera: $CLIPS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

status=0
for split in train val; do
  if [ "$split" = train ]; then LOGS=(--logs data_lists/av2_train_stage1.txt --first 150); else LOGS=(--logs data_lists/av2_val_dev10.txt); fi
  if [ ! -d "$CHASE/$split" ] || [ ! -d "$VIDEO/$split" ]; then
    echo "skipped $split: no chase targets in $CHASE/$split or no photos in $VIDEO/$split"
    status=1
    continue
  fi
  for camera in $CAMERAS; do
    echo "+ $split $camera"
    python scripts/sample_video.py --checkpoint "$CHECKPOINT" --out "$OUT/$split/$camera" --split "$split" \
      "${LOGS[@]}" --camera "$camera" --clips "$CLIPS" "$@" \
      --set data.root="$RV_PROCESSED" data.chase_root="$CHASE" data.video_root="$VIDEO" \
            model.weights_root="$RV_WEIGHTS" model.stage1.head_checkpoint="$HEAD_CHECKPOINT" || status=1
  done
done

rv_to_nas "$OUT" "$RV_SAMPLE_RUN/$FOLDER" || status=1
rv_log_to_nas "$RV_SAMPLE_RUN"
[ "$status" = 0 ] && echo "SAMPLES OK: $OUT" || echo "SAMPLES FAILED OR INCOMPLETE: see the messages above"
exit "$status"
