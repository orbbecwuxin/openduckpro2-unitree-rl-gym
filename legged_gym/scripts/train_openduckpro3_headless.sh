#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# ======================== EDITABLE CONFIG ========================
# Change these values, then run this script without extra arguments.
CONDA_ENV="/data2/conda/envs/openduck-unitree"
PHYSICAL_GPU=1
NUM_ENVS=4096
MAX_ITERATIONS=10000
SEED=1
RUN_NAME="manual_headless_openduckpro3_flex20_t055_vx015_030"
# ================================================================

PYTHON="${CONDA_ENV}/bin/python"

if (($#)); then
  printf 'This script uses the EDITABLE CONFIG block. Edit the file and run it without arguments.\n' >&2
  exit 2
fi

if [[ ! "${PHYSICAL_GPU}" =~ ^[0-9]+$ ]]; then
  printf 'GPU ID must be a non-negative integer: %s\n' "${PHYSICAL_GPU}" >&2
  exit 2
fi
if [[ ! "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Environment count must be a positive integer: %s\n' "${NUM_ENVS}" >&2
  exit 2
fi
if [[ ! "${MAX_ITERATIONS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Iteration count must be a positive integer: %s\n' "${MAX_ITERATIONS}" >&2
  exit 2
fi
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
  printf 'Seed must be a non-negative integer: %s\n' "${SEED}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  printf 'Python is not executable: %s\n' "${PYTHON}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PATH="${CONDA_ENV}/bin:${PATH}"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"
else
  export PYTHONPATH="${REPO_DIR}"
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="${CONDA_ENV}/lib"
fi

printf 'OpenDuckPro3 headless training\n'
printf '  physical_gpu=%s logical_device=cuda:0\n' "${PHYSICAL_GPU}"
printf '  num_envs=%s iterations=%s seed=%s\n' \
  "${NUM_ENVS}" "${MAX_ITERATIONS}" "${SEED}"
printf '  run_name=%s\n' "${RUN_NAME}"
printf '  pythonpath_head=%s\n' "${REPO_DIR}"

cd "${REPO_DIR}"
exec "${PYTHON}" legged_gym/scripts/train.py \
  --task=openduckpro3 \
  --experiment_name=openduckpro3 \
  --run_name="${RUN_NAME}" \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --num_envs="${NUM_ENVS}" \
  --max_iterations="${MAX_ITERATIONS}" \
  --seed="${SEED}" \
  --headless
