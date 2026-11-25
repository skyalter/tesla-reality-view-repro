#!/usr/bin/env bash
# A read-only look at the cluster before anything is installed or downloaded. Run on a login node
# and copy the printed report back. Nothing is written or changed.
#
# Usage: lighthouse/check_cluster.sh [--gpu]
#
#   --gpu   also ask Slurm for a 5-minute, 1-GPU allocation per GPU type to read the driver version
#           and to test internet access from a compute node. Skipped if the allocation does not start
#           within 2 minutes. Needs RV_ACCOUNT and RV_PARTITION in lighthouse/local.env.
#
# Works before lighthouse/local.env exists, but then skips the checks that need its values.
set -uo pipefail

. "$(dirname "$0")/common.sh"
WITH_GPU=0
[ "${1:-}" = "--gpu" ] && WITH_GPU=1

S3_PROBE="https://argoverse.s3.amazonaws.com/datasets/av2/sensor/val/02678d04-cc9f-3148-9f95-1ba66347dff9/calibration/intrinsics.feather"
section() { printf '\n== %s\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }
http_code() { curl -sS -o /dev/null -m 15 -I -w '%{http_code}' "$1" 2>/dev/null || echo "failed"; }

section "host"
echo "host: $(hostname)   date: $(date '+%F %T %Z')"
echo "glibc: $(ldd --version 2>/dev/null | head -n 1)"
echo "kernel: $(uname -r)   bash: ${BASH_VERSION}"
echo "python3: $(python3 --version 2>&1)   git: $(git --version 2>&1)"

section "settings from lighthouse/local.env"
for v in RV_SCRATCH RV_RAW RV_ACCOUNT RV_PARTITION RV_CONDA_SH; do
  echo "$v=${!v:-<not set>}"
done

section "tools on PATH"
for t in s5cmd rclone conda module tmux curl; do
  if have "$t" || [ "$(type -t "$t" 2>/dev/null)" = function ]; then echo "$t: yes ($(command -v "$t" 2>/dev/null || echo shell function))"; else echo "$t: no"; fi
done
if [ -n "${RV_CONDA_SH:-}" ]; then
  if [ -f "$RV_CONDA_SH" ]; then
    echo "conda.sh: found"
    ( set +u; . "$RV_CONDA_SH" && conda env list 2>/dev/null | sed 's/^/  /' )
  else
    echo "conda.sh: NOT found at $RV_CONDA_SH"
  fi
fi

section "internet from this node (HTTP status of a HEAD request; 200 or 3xx means reachable)"
echo "Argoverse S3 bucket: $(http_code "$S3_PROBE")"
echo "huggingface.co:      $(http_code https://huggingface.co)"
echo "github.com:          $(http_code https://github.com)"

section "slurm"
if have sinfo; then
  have my_accounts && my_accounts 2>&1 | sed 's/^/  /'
  if [ -n "${RV_PARTITION:-}" ]; then
    echo "nodes in partition $RV_PARTITION (node, GPUs, CPUs, memory MB, state):"
    sinfo -p "$RV_PARTITION" -N -h -o '  %N %G %c %m %t' 2>&1 | sort -u
  else
    echo "RV_PARTITION not set: skipping the node list"
  fi
  echo "my jobs:"
  squeue -u "$USER" -h -o '  %i %P %j %T %M %R' 2>&1
else
  echo "sinfo not found: is this a login node?"
fi

section "storage"
df -h "$HOME" 2>/dev/null | sed 's/^/  /'
if [ -n "${RV_SCRATCH:-}" ]; then
  PARENT="$(dirname "$RV_SCRATCH")"
  df -h "$PARENT" 2>/dev/null | sed 's/^/  /'
  echo "note: df shows the whole file system, not the account's quota"
  echo "top level of $PARENT:"
  ls -1 "$PARENT" 2>&1 | sed 's/^/  /'
  if [ -d "$RV_SCRATCH" ]; then echo "$RV_SCRATCH exists:"; ls -1 "$RV_SCRATCH" | sed 's/^/  /'; else echo "$RV_SCRATCH does not exist yet"; fi
fi

section "existing Argoverse 2 Sensor data"
# A Sensor log looks like <root>/<split>/<log_id>/sensors/cameras/...; the motion-forecasting dataset does not.
probe() {
  local base split first n found=1
  for base in "$1" "$1/sensor"; do
    for split in train val test; do
      [ -d "$base/$split" ] || continue
      first="$(ls -U "$base/$split" 2>/dev/null | head -n 1)"
      if [ -n "$first" ] && [ -d "$base/$split/$first/sensors/cameras" ]; then
        n="$(ls -U "$base/$split" | wc -l)"
        echo "  FOUND Sensor data: $base/$split ($n logs; ring cameras: $(ls -d "$base/$split/$first"/sensors/cameras/ring_* 2>/dev/null | wc -l), lidar dir: $([ -d "$base/$split/$first/sensors/lidar" ] && echo yes || echo no))"
        found=0
      fi
    done
  done
  return $found
}
any=1
ROOTS=()
[ -n "${RV_SCRATCH:-}" ] && ROOTS+=("$(dirname "$RV_SCRATCH")")
# shellcheck disable=SC2206
[ -n "${RV_SEARCH_ROOTS:-}" ] && ROOTS+=($RV_SEARCH_ROOTS)
if [ "${#ROOTS[@]}" = 0 ]; then
  echo "RV_SCRATCH not set: nowhere to look"
else
  for root in "${ROOTS[@]}"; do
    echo "looking under $root"
    while IFS= read -r cand; do
      echo "  candidate: $cand"
      probe "$cand" && any=0
    done < <(find "$root" -maxdepth 3 -type d \( -iname '*argoverse*' -o -iname 'av2*' -o -iname 'sensor' \) 2>/dev/null)
  done
  [ "$any" = 0 ] || echo "no Argoverse 2 Sensor data found (directories that only hold scenario files are the motion-forecasting dataset)"
fi

section "rclone remotes (names only)"
if have rclone; then rclone listremotes 2>&1 | sed 's/^/  /'; else echo "rclone is not on PATH"; fi

if [ "$WITH_GPU" = 1 ]; then
  section "GPU nodes (one short allocation per GPU type)"
  if [ -z "${RV_ACCOUNT:-}" ] || [ -z "${RV_PARTITION:-}" ]; then
    echo "RV_ACCOUNT or RV_PARTITION not set: skipped"
  else
    types="$(sinfo -p "$RV_PARTITION" -h -o '%G' 2>/dev/null | tr ',' '\n' | awk -F: '$1 == "gpu" && NF >= 3 {print $2}' | sort -u)"
    [ -n "$types" ] || echo "could not read GPU types from sinfo"
    for t in $types; do
      echo "-- $t"
      srun -A "$RV_ACCOUNT" -p "$RV_PARTITION" --gpus-per-node="$t:1" -c 1 --mem=2G -t 5 --immediate=120 bash -c "
        echo node: \$(hostname)
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        echo cpus on node: \$(nproc --all)   glibc: \$(ldd --version | head -n 1)
        echo S3 from compute node: \$(curl -sS -o /dev/null -m 15 -I -w '%{http_code}' '$S3_PROBE' 2>/dev/null || echo failed)
        echo huggingface from compute node: \$(curl -sS -o /dev/null -m 15 -I -w '%{http_code}' https://huggingface.co 2>/dev/null || echo failed)
      " 2>&1 | sed 's/^/  /' || echo "  allocation did not start within 2 minutes (node busy?): skipped"
    done
  fi
fi

printf '\n== end of report\n'
