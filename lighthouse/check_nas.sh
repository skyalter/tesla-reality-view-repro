#!/usr/bin/env bash
# Check that jobs can copy their results to long-term storage through rclone. Run on a login node.
#
# Usage: lighthouse/check_nas.sh [remote:path]
#
# Without an argument it uses RV_NAS from lighthouse/local.env. It lists the top level of the remote,
# creates the directory, writes one small file (_write_check.txt) into it and reads it back. Nothing is
# deleted. Credentials stay in rclone's own configuration; this script never reads or prints them.
set -euo pipefail

. "$(dirname "$0")/common.sh"
rv_activate

TARGET="${1:-${RV_NAS:-}}"
if [ -z "$TARGET" ]; then
  echo "no target: pass remote:path or set RV_NAS in lighthouse/local.env" >&2
  echo "configured remotes:" >&2
  rclone listremotes >&2
  exit 2
fi
REMOTE="${TARGET%%:*}:"

echo "== top level of $REMOTE"
rclone lsd "$REMOTE" --max-depth 1 | awk '{print "   " $NF}' | head -20
echo "== writing to $TARGET"
rclone mkdir "$TARGET"
STAMP="$(date '+%F %T') from $(hostname)"
printf '%s\n' "$STAMP" | rclone rcat "$TARGET/_write_check.txt"
BACK="$(rclone cat "$TARGET/_write_check.txt")"
if [ "$BACK" = "$STAMP" ]; then
  echo "NAS OK: wrote and read back $TARGET/_write_check.txt"
else
  echo "NAS CHECK FAILED: wrote '$STAMP' but read back '$BACK'" >&2
  exit 1
fi
