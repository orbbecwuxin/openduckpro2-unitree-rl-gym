#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export HEADLESS="${HEADLESS:-0}"
export NUM_ENVS="${NUM_ENVS:-4}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-500}"
export RUN_NAME="${RUN_NAME:-openduck_visual}"

if (( NUM_ENVS < 4 )); then
    echo "NUM_ENVS=${NUM_ENVS} is too small for the recurrent PPO default num_mini_batches=4; using NUM_ENVS=4." >&2
    export NUM_ENVS=4
fi

exec bash "${SCRIPT_DIR}/train_open_duck_unitree.sh" "$@"
