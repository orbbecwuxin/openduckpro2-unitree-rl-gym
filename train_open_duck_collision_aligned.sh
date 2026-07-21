#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Use the Isaac Gym training environment by default.
export CONDA_ENV="${CONDA_ENV:-openduck-unitree}"
export HEADLESS="${HEADLESS:-1}"
export NUM_ENVS="${NUM_ENVS:-4096}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-500}"
export RUN_NAME="${RUN_NAME:-openduck_mujoco_collision_aligned_500}"

exec bash "${SCRIPT_DIR}/train_open_duck_unitree.sh" "$@"
