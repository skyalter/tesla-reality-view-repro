#!/usr/bin/env bash
# Download Argoverse 2 Sensor logs from the public S3 bucket (no AWS account needed).
# Shared by the workstation (small debug set) and the cluster (full set).
#
# Usage:
#   av2_download.sh --split <train|val|test> --list <file> --dest <root>
#                   [--no-stereo] [--meta-only] [--limit N] [--workers N]
#                   [--dry-run] [--verify-only] [--check-out <file>]
#
#   --list         text file with one log id per line. Blank lines, comment lines (#) and a header
#                  line are ignored; only the first column is read, so TSV files work too.
#   --no-stereo    skip the two stereo cameras (about 15% of each log)
#   --meta-only    only poses, calibration and 3D box annotations (about 1 MB per log)
#   --limit N      only the first N logs of the list
#   --workers N    parallel transfers (default 32)
#   --dry-run      print what would be copied; download nothing, check nothing
#   --verify-only  download nothing; only check what is on disk against the bucket listing
#   --check-out    where to write the check table (default: <root>/<split>.download_check.tsv)
#
# Layout mirrors the bucket: <root>/<split>/<log_id>/...
# Safe to re-run: files that are already complete are skipped.
# After downloading, every log is compared with the bucket listing (relative path and size of each
# expected file). Exit code is 1 if any log is incomplete.
#
# Requires s5cmd (https://github.com/peak/s5cmd). Set S5CMD to its path if it is not on PATH.
set -euo pipefail

BUCKET="s3://argoverse/datasets/av2/sensor"
SPLIT="" LIST="" DEST="" CHECK_OUT=""
NO_STEREO=0 META_ONLY=0 DRY_RUN=0 VERIFY_ONLY=0 LIMIT=0 WORKERS=32

while [ $# -gt 0 ]; do
  case "$1" in
    --split) SPLIT="$2"; shift 2 ;;
    --list) LIST="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --check-out) CHECK_OUT="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --no-stereo) NO_STEREO=1; shift ;;
    --meta-only) META_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$SPLIT" in train|val|test) ;; *) echo "--split must be train, val or test" >&2; exit 2 ;; esac
[ -f "$LIST" ] || { echo "--list: file not found: $LIST" >&2; exit 2; }
[ -n "$DEST" ] || { echo "--dest is required" >&2; exit 2; }
S5CMD="${S5CMD:-$(command -v s5cmd || true)}"
[ -x "$S5CMD" ] || { echo "s5cmd not found; install it or set S5CMD" >&2; exit 2; }
s5() { "$S5CMD" --no-sign-request "$@"; }

# Log ids are UUIDs; this also drops header and comment lines.
mapfile -t LOGS < <(awk 'NF {print $1}' "$LIST" | grep -E '^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$' || true)
[ "${#LOGS[@]}" -gt 0 ] || { echo "no log ids found in $LIST" >&2; exit 2; }
if [ "$LIMIT" -gt 0 ] && [ "$LIMIT" -lt "${#LOGS[@]}" ]; then LOGS=("${LOGS[@]:0:$LIMIT}"); fi

MODE="full"
[ "$NO_STEREO" = 1 ] && MODE="no-stereo"
[ "$META_ONLY" = 1 ] && MODE="meta-only"
echo "split=$SPLIT logs=${#LOGS[@]} mode=$MODE dest=$DEST"

# Keeps the "<relative path><TAB><size>" lines that this mode is supposed to fetch.
keep_for_mode() {
  if [ "$META_ONLY" = 1 ]; then
    grep -E '^(city_SE3_egovehicle\.feather|annotations\.feather|calibration/[^[:space:]]+)[[:space:]]' || true
  elif [ "$NO_STEREO" = 1 ]; then
    grep -v '^sensors/cameras/stereo_front_' || true
  else
    cat
  fi
}

# Bucket listing of one log as "<relative path><TAB><size>", sorted.
expected() {
  s5 ls "$BUCKET/$SPLIT/$1/*" | awk '{print $NF "\t" $(NF-1)}' | keep_for_mode | LC_ALL=C sort
}

# ---------------------------------------------------------------- download
if [ "$VERIFY_ONLY" = 0 ]; then
  # Real transfers print errors only; a dry run prints every planned copy.
  GLOBAL=(--log error)
  [ "$DRY_RUN" = 1 ] && GLOBAL=(--dry-run)
  [ "$DRY_RUN" = 1 ] || mkdir -p "$DEST/$SPLIT"
  if [ "$META_ONLY" = 1 ]; then
    # A few small files per log: one batch for all logs. "cp -s" skips files whose size already matches.
    CMDS="$(mktemp)"
    trap 'rm -f "$CMDS"' EXIT
    for LOG in "${LOGS[@]}"; do
      SRC="$BUCKET/$SPLIT/$LOG" DST="$DEST/$SPLIT/$LOG"
      echo "cp -s \"$SRC/city_SE3_egovehicle.feather\" \"$DST/city_SE3_egovehicle.feather\""
      echo "cp -s \"$SRC/calibration/*\" \"$DST/calibration/\""
      # The test split has no annotations.
      [ "$SPLIT" = test ] || echo "cp -s \"$SRC/annotations.feather\" \"$DST/annotations.feather\""
    done > "$CMDS"
    s5 --numworkers "$WORKERS" "${GLOBAL[@]}" run "$CMDS"
  else
    EXCLUDE=()
    [ "$NO_STEREO" = 1 ] && EXCLUDE=(--exclude "*stereo_front_*")
    n=0
    for LOG in "${LOGS[@]}"; do
      n=$((n + 1))
      echo "[$n/${#LOGS[@]}] $SPLIT/$LOG"
      s5 --numworkers "$WORKERS" "${GLOBAL[@]}" sync ${EXCLUDE[@]+"${EXCLUDE[@]}"} "$BUCKET/$SPLIT/$LOG/*" "$DEST/$SPLIT/$LOG/"
    done
  fi
fi
[ "$DRY_RUN" = 1 ] && { echo "dry run: nothing downloaded, nothing checked"; exit 0; }

# ---------------------------------------------------------------- check
CHECK_OUT="${CHECK_OUT:-$DEST/$SPLIT.download_check.tsv}"
TMP_CHECK="$(mktemp)"
printf 'split\tlog_id\tmode\texpected_files\texpected_bytes\tmissing_or_wrong_size\tstatus\n' > "$TMP_CHECK"
bad_logs=0 total_bytes=0
for LOG in "${LOGS[@]}"; do
  exp="$(expected "$LOG")"
  if [ -d "$DEST/$SPLIT/$LOG" ]; then
    loc="$(cd "$DEST/$SPLIT/$LOG" && find . -type f -printf '%P\t%s\n' | LC_ALL=C sort)"
  else
    loc=""
  fi
  n_exp="$(printf '%s\n' "$exp" | grep -c . || true)"
  b_exp="$(printf '%s\n' "$exp" | awk -F'\t' '{s += $2} END {printf "%.0f", s}')"
  # Lines of the listing with no identical "<path><TAB><size>" line on disk: missing or wrong size.
  n_bad="$(LC_ALL=C comm -23 <(printf '%s\n' "$exp") <(printf '%s\n' "$loc") | grep -c . || true)"
  status=ok
  if [ "$n_exp" = 0 ]; then status=empty_listing; fi
  if [ "$n_bad" != 0 ]; then status=incomplete; fi
  [ "$status" = ok ] || bad_logs=$((bad_logs + 1))
  total_bytes=$((total_bytes + b_exp))
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$SPLIT" "$LOG" "$MODE" "$n_exp" "$b_exp" "$n_bad" "$status" >> "$TMP_CHECK"
done
mkdir -p "$(dirname "$CHECK_OUT")"
mv "$TMP_CHECK" "$CHECK_OUT"
echo "checked ${#LOGS[@]} logs, expected $(awk -v b="$total_bytes" 'BEGIN {printf "%.2f", b / 1e9}') GB, incomplete: $bad_logs"
echo "check table: $CHECK_OUT"
[ "$bad_logs" = 0 ] || { echo "some logs are incomplete; run the same command again to resume" >&2; exit 1; }
