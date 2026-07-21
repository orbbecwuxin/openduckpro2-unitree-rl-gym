#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV:-openduck-unitree}"
GPU_ID="${GPU_ID:-0}"
NUM_ENVS="${NUM_ENVS:-1}"
RUN_NAME="${RUN_NAME:-openduck_play}"
SIM_DEVICE="${SIM_DEVICE:-cuda:${GPU_ID}}"
RL_DEVICE="${RL_DEVICE:-cuda:${GPU_ID}}"
LOAD_RUN="${LOAD_RUN:-}"
CHECKPOINT="${CHECKPOINT:--1}"
KEYBOARD_COMMANDS="${KEYBOARD_COMMANDS:-1}"
KEYBOARD_VX="${KEYBOARD_VX:-0.12}"
KEYBOARD_VY="${KEYBOARD_VY:-0.12}"
KEYBOARD_YAW="${KEYBOARD_YAW:-0.6}"
KEYBOARD_VX_STEP="${KEYBOARD_VX_STEP:-0.02}"
KEYBOARD_VY_STEP="${KEYBOARD_VY_STEP:-0.02}"
KEYBOARD_YAW_STEP="${KEYBOARD_YAW_STEP:-0.1}"
KEYBOARD_HEADING_HOLD="${KEYBOARD_HEADING_HOLD:-1}"
KEYBOARD_HEADING_KP="${KEYBOARD_HEADING_KP:-1.5}"

if [[ -z "${CONDA_PREFIX:-}" || "$(basename "$CONDA_PREFIX")" != "$CONDA_ENV" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda was not found. Activate ${CONDA_ENV} first or install conda." >&2
        exit 1
    fi

    CONDA_BASE="$(conda info --base)"
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${EXTRA_LD_LIBRARY_PATH:+:${EXTRA_LD_LIBRARY_PATH}}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=(
    --task=open_duck
    --sim_device="${SIM_DEVICE}"
    --rl_device="${RL_DEVICE}"
    --num_envs="${NUM_ENVS}"
    --run_name="${RUN_NAME}"
    --checkpoint="${CHECKPOINT}"
)

if [[ -n "$LOAD_RUN" ]]; then
    ARGS+=(--load_run="${LOAD_RUN}")
fi

if [[ "$KEYBOARD_COMMANDS" == "1" || "$KEYBOARD_COMMANDS" == "true" || "$KEYBOARD_COMMANDS" == "TRUE" ]]; then
    ARGS+=(
        --keyboard_commands
        --keyboard_vx="${KEYBOARD_VX}"
        --keyboard_vy="${KEYBOARD_VY}"
        --keyboard_yaw="${KEYBOARD_YAW}"
        --keyboard_vx_step="${KEYBOARD_VX_STEP}"
        --keyboard_vy_step="${KEYBOARD_VY_STEP}"
        --keyboard_yaw_step="${KEYBOARD_YAW_STEP}"
        --keyboard_heading_kp="${KEYBOARD_HEADING_KP}"
    )
    if [[ "$KEYBOARD_HEADING_HOLD" == "1" || "$KEYBOARD_HEADING_HOLD" == "true" || "$KEYBOARD_HEADING_HOLD" == "TRUE" ]]; then
        ARGS+=(--keyboard_heading_hold)
    fi
fi

echo "Playing OpenDuck with:"
echo "  conda env: ${CONDA_ENV}"
echo "  sim device: ${SIM_DEVICE}"
echo "  rl device: ${RL_DEVICE}"
echo "  num envs: ${NUM_ENVS}"
echo "  load run: ${LOAD_RUN:-latest}"
echo "  checkpoint: ${CHECKPOINT}"
echo "  keyboard commands: ${KEYBOARD_COMMANDS}"
if [[ "$KEYBOARD_COMMANDS" == "1" || "$KEYBOARD_COMMANDS" == "true" || "$KEYBOARD_COMMANDS" == "TRUE" ]]; then
    echo "  keyboard speeds: vx=${KEYBOARD_VX}, vy=${KEYBOARD_VY}, yaw=${KEYBOARD_YAW}"
    echo "  keyboard speed steps: vx=${KEYBOARD_VX_STEP}, vy=${KEYBOARD_VY_STEP}, yaw=${KEYBOARD_YAW_STEP}"
    echo "  heading hold: ${KEYBOARD_HEADING_HOLD}, kp=${KEYBOARD_HEADING_KP}"
    echo "  controls: W/S=forward/back, A/D=left/right, Q/E=turn, Z=yaw stop, X/Space=stop"
    echo "  command mode: keys latch until another direction key or stop is pressed"
    echo "  speed keys: R/F=vx speed +/-; T/G=vy speed +/-; Y/H=yaw speed +/-"
fi

exec python legged_gym/scripts/play.py "${ARGS[@]}" "$@"
