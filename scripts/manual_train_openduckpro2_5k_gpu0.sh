#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data2/wuxin/Open_Duck_Mini/unitree_rl_gym}"
CONDA_ENV="${CONDA_ENV:-/data2/conda/envs/openduck-unitree}"
GPU="${GPU:-0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
TASK="${TASK:-openduckpro2}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-openduckpro2}"
RUN_NAME="${RUN_NAME:-manual_openduckpro2_g1_urdf_gpu${GPU}_${MAX_ITERATIONS}it_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/manual_train_runs/${RUN_NAME}}"

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU}}"
export PYTHONUNBUFFERED=1

{
  echo "run_name=${RUN_NAME}"
  echo "repo=${REPO_DIR}"
  echo "gpu=${GPU}"
  echo "num_envs=${NUM_ENVS}"
  echo "max_iterations=${MAX_ITERATIONS}"
  echo "started_at=$(date --iso-8601=seconds)"
} | tee "${LOG_DIR}/metadata.txt"

"${CONDA_ENV}/bin/python" legged_gym/scripts/train.py \
  --task="${TASK}" \
  --experiment_name="${EXPERIMENT_NAME}" \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --num_envs="${NUM_ENVS}" \
  --max_iterations="${MAX_ITERATIONS}" \
  --run_name="${RUN_NAME}" \
  --headless 2>&1 | tee "${LOG_DIR}/train.log"
