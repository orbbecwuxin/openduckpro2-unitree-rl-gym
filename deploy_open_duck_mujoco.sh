#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV:-openduck-unitree}"
CONFIG="${1:-open_duck_mini.yaml}"
if [[ $# -gt 0 ]]; then
    shift
fi

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

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${EXTRA_LD_LIBRARY_PATH:+:${EXTRA_LD_LIBRARY_PATH}}"
fi
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if [[ "$PYTHON_BIN" == "python" ]] && command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    else
        echo "Python executable not found: ${PYTHON_BIN}" >&2
        exit 1
    fi
fi

echo "Deploying OpenDuck MuJoCo with:"
echo "  conda env: ${CONDA_ENV:-current shell}"
echo "  python: ${PYTHON_BIN}"
echo "  config: ${CONFIG}"

exec "$PYTHON_BIN" deploy/deploy_mujoco/deploy_mujoco.py "$CONFIG" "$@"
