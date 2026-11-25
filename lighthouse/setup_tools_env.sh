#!/usr/bin/env bash
# Create the small conda environment used for downloading (s5cmd and Python). Run once on a login node.
# The training environment is built later by a separate script, once the GPU driver version is known.
#
# Usage: lighthouse/setup_tools_env.sh
set -euo pipefail

. "$(dirname "$0")/common.sh"
rv_conda

if conda env list | awk '{print $1}' | grep -qx "$RV_TOOLS_ENV"; then
  echo "environment $RV_TOOLS_ENV already exists"
else
  conda create -y -n "$RV_TOOLS_ENV" --override-channels -c conda-forge python=3.12 s5cmd
fi
conda activate "$RV_TOOLS_ENV"
echo "s5cmd: $(command -v s5cmd) ($(s5cmd version))"
echo "python: $(python3 --version)"
