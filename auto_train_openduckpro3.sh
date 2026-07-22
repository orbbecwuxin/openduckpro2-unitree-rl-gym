#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV:-openduck-unitree}"
CONDA_SH="${CONDA_SH:-/home/orbbec/miniconda3/etc/profile.d/conda.sh}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -n "$CONDA_ENV" && ( -z "${CONDA_PREFIX:-}" || "$(basename "$CONDA_PREFIX")" != "$CONDA_ENV" ) ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

    if ! command -v conda >/dev/null 2>&1; then
        if [[ -f "$CONDA_SH" ]]; then
            # shellcheck source=/dev/null
            source "$CONDA_SH"
        elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
            # shellcheck source=/dev/null
            source "${HOME}/miniconda3/etc/profile.d/conda.sh"
        elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
            # shellcheck source=/dev/null
            source "${HOME}/anaconda3/etc/profile.d/conda.sh"
        fi
    fi

    if ! command -v conda >/dev/null 2>&1; then
        echo "conda was not found. Set CONDA_SH or activate ${CONDA_ENV} first." >&2
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
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${EXTRA_LD_LIBRARY_PATH:+:${EXTRA_LD_LIBRARY_PATH}}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

exec "$PYTHON_BIN" legged_gym/scripts/auto_train.py "$@"
