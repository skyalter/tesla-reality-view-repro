#!/usr/bin/env bash
# Body of the environment check; runs on a GPU node. Submitted by lighthouse/submit_test_env.sh.
# sbatch runs a copy of this file, so the repository location comes in through RV_LH_DIR.
set -eo pipefail
. "${RV_LH_DIR:?RV_LH_DIR is not set; submit through lighthouse/submit_test_env.sh}/common.sh"
rv_activate
# gsplat: compile settings and the environment's own C++ runtime (see common.sh).
rv_gsplat_env
export MAX_JOBS=8
echo "node: $(hostname)   job: ${SLURM_JOB_ID:-none}   $(date '+%F %T')"
python "$RV_LH_DIR/env_probe.py" "$@"
