#!/usr/bin/env bash
# Cut the photos of all seven cameras to the picture size of the diffusion model (832x480): the targets of
# the real-camera samples of stage 2 and the photos shown next to the samples. CPU only. Finished logs are
# skipped, so the job can simply be submitted again. Results go to $RV_SCRATCH/video/<name>/<split>/; only a
# small record is copied to the NAS (the pictures can be made again from the raw data).
#
# Which logs: RV_SPLIT, RV_LOGS, RV_FIRST (submit.sh --split/--logs/--first). Without --split: the first 150
# logs of data_lists/av2_train_stage1.txt, then all of data_lists/av2_val_dev10.txt.
. "${RV_CODE:?submit through lighthouse/submit.sh}/lighthouse/jobs/_env.sh"

OUT="$RV_SCRATCH/video/$RV_RUN_NAME"
mkdir -p "$OUT"
WORKERS="${SLURM_CPUS_PER_TASK:-8}"
status=0
cut() {     # cut <split> <list> <first or empty>
  echo "+ $1: $2${3:+, the first $3}"
  # shellcheck disable=SC2086
  python scripts/make_video_frames.py --raw-root "$RV_RAW" --processed-root "$RV_PROCESSED" --split "$1" \
    --logs "$2" ${3:+--first $3} --out "$OUT" --workers "$WORKERS" || status=1
  echo "logs complete ($1): $({ find "$OUT/$1" -maxdepth 2 -name DONE 2>/dev/null || true; } | wc -l)"
}
case "${RV_SPLIT:-}" in
  train) cut train "${RV_LOGS:-data_lists/av2_train_stage1.txt}" "${RV_FIRST:-150}" ;;
  val) cut val "${RV_LOGS:-data_lists/av2_val_dev10.txt}" "${RV_FIRST:-}" ;;
  *) cut train data_lists/av2_train_stage1.txt 150
     cut val data_lists/av2_val_dev10.txt "" ;;
esac
du -sh "$OUT" || true

rv_to_nas "$OUT" "$RV_RUN_NAME" --include "*/*/cameras.json" --include "*/*/DONE" || status=1
rv_log_to_nas "$RV_RUN_NAME"
[ "$status" = 0 ] && echo "FRAMES OK: $OUT" || echo "FRAMES FAILED: see the messages above; submit the same command again to continue"
exit "$status"
