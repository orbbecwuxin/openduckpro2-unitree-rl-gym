#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV-openduck-unitree}"
GPU_ID="${GPU_ID:-0}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20}"
RUN_NAME="${RUN_NAME:-openduck_smoke}"
SIM_DEVICE="${SIM_DEVICE:-cuda:${GPU_ID}}"
RL_DEVICE="${RL_DEVICE:-cuda:${GPU_ID}}"
HEADLESS="${HEADLESS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -n "$CONDA_ENV" && ( -z "${CONDA_PREFIX:-}" || "$(basename "$CONDA_PREFIX")" != "$CONDA_ENV" ) ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda was not found. Activate ${CONDA_ENV} first or install conda." >&2
        exit 1
    fi

    CONDA_BASE="$(conda info --base)"
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if [[ "$PYTHON_BIN" == "python" ]] && command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "Python executable not found: ${PYTHON_BIN}" >&2
        exit 1
    fi
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${EXTRA_LD_LIBRARY_PATH:+:${EXTRA_LD_LIBRARY_PATH}}"
fi
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Training OpenDuck with:"
echo "  conda env: ${CONDA_ENV:-current shell}"
echo "  sim device: ${SIM_DEVICE}"
echo "  rl device: ${RL_DEVICE}"
echo "  num envs: ${NUM_ENVS}"
echo "  max iterations: ${MAX_ITERATIONS}"
echo "  run name: ${RUN_NAME}"
echo "  headless: ${HEADLESS}"
echo "  python: ${PYTHON_BIN}"

ARGS=(
    --task=open_duck \
    --sim_device="${SIM_DEVICE}" \
    --rl_device="${RL_DEVICE}" \
    --num_envs="${NUM_ENVS}" \
    --max_iterations="${MAX_ITERATIONS}" \
    --run_name="${RUN_NAME}"
)

if [[ "$HEADLESS" == "1" || "$HEADLESS" == "true" || "$HEADLESS" == "TRUE" ]]; then
    ARGS+=(--headless)
fi

exec "$PYTHON_BIN" legged_gym/scripts/train.py "${ARGS[@]}" "$@"
