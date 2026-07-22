#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  launch_openduckpro3_training.sh --config PATH --source-commit SHA [options]

Options:
  --run-id ID         Unique run id; defaults to a timestamped id.
  --foreground        Keep the controller attached to this terminal.
  --validate-only     Validate and print the launch command without creating a run.
  -h, --help          Show this help.

Environment:
  CONDA_ENV           Training environment (default: /data2/conda/envs/openduck-unitree).
  PYTHON              Python executable (default: $CONDA_ENV/bin/python).
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-/data2/conda/envs/openduck-unitree}"
PYTHON="${PYTHON:-${CONDA_ENV}/bin/python}"

CONFIG=""
RUN_ID=""
SOURCE_COMMIT=""
FOREGROUND=0
VALIDATE_ONLY=0

while (($#)); do
  case "$1" in
    --config)
      CONFIG="${2:?--config requires a path}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:?--run-id requires a value}"
      shift 2
      ;;
    --source-commit)
      SOURCE_COMMIT="${2:?--source-commit requires a SHA}"
      shift 2
      ;;
    --foreground)
      FOREGROUND=1
      shift
      ;;
    --validate-only)
      VALIDATE_ONLY=1
      shift
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

if [[ -z "${CONFIG}" || -z "${SOURCE_COMMIT}" ]]; then
  usage >&2
  exit 2
fi
if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${REPO_DIR}/${CONFIG}"
fi
if [[ ! -f "${CONFIG}" ]]; then
  printf 'Config does not exist: %s\n' "${CONFIG}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  printf 'Python is not executable: %s\n' "${PYTHON}" >&2
  exit 2
fi

RUN_ID="${RUN_ID:-codex_openduckpro3_10k_$(date +%Y%m%d_%H%M%S)}"
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'Run id may contain only letters, numbers, dot, underscore, and dash: %s\n' "${RUN_ID}" >&2
  exit 2
fi

CURRENT_COMMIT="$(git -C "${REPO_DIR}" rev-parse HEAD)"
EXPECTED_COMMIT="$(git -C "${REPO_DIR}" rev-parse "${SOURCE_COMMIT}^{commit}")"
if [[ "${CURRENT_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  printf 'Source commit mismatch: expected %s, current %s\n' \
    "${EXPECTED_COMMIT}" "${CURRENT_COMMIT}" >&2
  exit 2
fi
if [[ -n "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=normal)" ]]; then
  printf 'Tracked source worktree is not clean; refusing to launch.\n' >&2
  git -C "${REPO_DIR}" status --short >&2
  exit 2
fi

"${PYTHON}" - "${REPO_DIR}" "${CONFIG}" <<'PY'
import sys
from pathlib import Path

repo = Path(sys.argv[1])
config_path = Path(sys.argv[2])
sys.path.insert(0, str(repo / "legged_gym/scripts"))
import auto_train

config = auto_train.merged_config(config_path)
auto_train.validate_continuous_training_config(config)
print(
    "validated continuous training: "
    f"iterations={config['train']['max_iterations']} "
    f"candidates={len(config['candidates'])} gpus={config['gpus']}"
)
PY

COMMAND=(
  "${PYTHON}"
  "${REPO_DIR}/legged_gym/scripts/auto_train.py"
  --config "${CONFIG}"
  --run-id "${RUN_ID}"
  --cycles 1
  --no-commit
)

printf 'source_commit=%s\nrun_id=%s\nconfig=%s\n' \
  "${CURRENT_COMMIT}" "${RUN_ID}" "${CONFIG}"
printf 'command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

if ((VALIDATE_ONLY)); then
  exit 0
fi

RUN_DIR="${REPO_DIR}/auto_train_runs/${RUN_ID}"
if [[ -e "${RUN_DIR}" ]]; then
  printf 'Run directory already exists: %s\n' "${RUN_DIR}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

if ((FOREGROUND)); then
  exec "${COMMAND[@]}" >>"${RUN_DIR}/launcher.log" 2>&1
fi

nohup "${COMMAND[@]}" >>"${RUN_DIR}/launcher.log" 2>&1 </dev/null &
CONTROLLER_PID=$!
printf '%s\n' "${CONTROLLER_PID}" >"${RUN_DIR}/controller.pid"
printf 'controller_pid=%s\nlauncher_log=%s\n' \
  "${CONTROLLER_PID}" "${RUN_DIR}/launcher.log"
