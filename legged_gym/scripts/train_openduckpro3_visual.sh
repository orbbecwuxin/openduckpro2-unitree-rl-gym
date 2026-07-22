#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# Defaults can be changed here or overridden with command-line options.
CONDA_ENV="${CONDA_ENV:-/data2/conda/envs/openduck-unitree}"
PYTHON="${PYTHON:-${CONDA_ENV}/bin/python}"
PHYSICAL_GPU="${PHYSICAL_GPU:-1}"
DISPLAY_ID="${DISPLAY:-:1}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
SEED="${SEED:-1}"
RUN_NAME="${RUN_NAME:-manual_visual_openduckpro3_$(date +%Y%m%d_%H%M%S)}"

usage() {
  cat <<'EOF'
Usage: train_openduckpro3_visual.sh [options]

Options:
  --gpu ID             Physical GPU exposed to the process (default: 1).
  --display DISPLAY    X11 display used by the Isaac Gym viewer (default: :1).
  --num-envs COUNT     Parallel environments (default: 4096).
  --iterations COUNT   Training iterations (default: 10000).
  --seed SEED          Random seed (default: 1).
  --run-name NAME      TensorBoard/checkpoint run name.
  -h, --help           Show this help.

The selected physical GPU is exposed as logical cuda:0 inside the process.
The script intentionally does not pass --headless.
EOF
}

while (($#)); do
  case "$1" in
    --gpu)
      PHYSICAL_GPU="${2:?--gpu requires an ID}"
      shift 2
      ;;
    --display)
      DISPLAY_ID="${2:?--display requires a value such as :1}"
      shift 2
      ;;
    --num-envs)
      NUM_ENVS="${2:?--num-envs requires a count}"
      shift 2
      ;;
    --iterations)
      MAX_ITERATIONS="${2:?--iterations requires a count}"
      shift 2
      ;;
    --seed)
      SEED="${2:?--seed requires a value}"
      shift 2
      ;;
    --run-name)
      RUN_NAME="${2:?--run-name requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

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
export DISPLAY="${DISPLAY_ID}"
export PATH="${CONDA_ENV}/bin:${PATH}"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="${CONDA_ENV}/lib"
fi

printf 'OpenDuckPro3 visual training\n'
printf '  physical_gpu=%s logical_device=cuda:0 display=%s\n' \
  "${PHYSICAL_GPU}" "${DISPLAY}"
printf '  num_envs=%s iterations=%s seed=%s\n' \
  "${NUM_ENVS}" "${MAX_ITERATIONS}" "${SEED}"
printf '  run_name=%s\n' "${RUN_NAME}"

cd "${REPO_DIR}"
exec "${PYTHON}" legged_gym/scripts/train.py \
  --task=openduckpro3 \
  --experiment_name=openduckpro3 \
  --run_name="${RUN_NAME}" \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --num_envs="${NUM_ENVS}" \
  --max_iterations="${MAX_ITERATIONS}" \
  --seed="${SEED}"
