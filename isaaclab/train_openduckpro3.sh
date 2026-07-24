#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Edit these defaults for routine runs. Command-line arguments appended below
# still override matching Hydra/Isaac Lab options when needed.
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/data2/wuxin/IsaacLab-2.3.2}"
PYTHON_BIN="${PYTHON_BIN:-/data2/conda/envs/leggedlab-train/bin/python}"
PHYSICAL_GPU="${PHYSICAL_GPU:-1}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
SEED="${SEED:-1}"
RUN_NAME="${RUN_NAME:-openduckpro3_direct_seed${SEED}}"

export ISAACLAB_ROOT
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTHONPATH="${SCRIPT_DIR}/source/isaaclab_openduck:${PYTHONPATH:-}"

cd "${REPO_ROOT}"
echo "OpenDuckPro3 Isaac Lab training"
echo "  gpu=${PHYSICAL_GPU} envs=${NUM_ENVS} iterations=${MAX_ITERATIONS} seed=${SEED}"
echo "  run_name=${RUN_NAME}"

exec "${PYTHON_BIN}" isaaclab/scripts/run_upstream.py train \
  --task Isaac-OpenDuckPro3-Direct-v0 \
  --device cuda:0 \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --seed "${SEED}" \
  --run_name "${RUN_NAME}" \
  --headless \
  "$@"
