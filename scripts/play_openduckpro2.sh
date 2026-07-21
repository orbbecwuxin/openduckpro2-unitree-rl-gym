#!/usr/bin/env bash
set -euo pipefail

REPO="/data2/wuxin/Open_Duck_Mini/unitree_rl_gym"
ENV_DIR="/data2/conda/envs/openduck-unitree"
LOAD_RUN="Jul20_17-51-15_codex_spawnfix_pdhalf_noise04_seed1_gpu1_10k_20260720_1744_c000_00_openduckpro2_spawnfix_pdhalf_noise04_seed1_gpu1"

CHECKPOINT="${1:-5500}"
GPU_ID="${2:-0}"

if [[ ! "${CHECKPOINT}" =~ ^[0-9]+$ ]]; then
  printf 'checkpoint must be an integer, got: %s\n' "${CHECKPOINT}" >&2
  exit 2
fi

if [[ -z "${DISPLAY:-}" ]]; then
  printf 'DISPLAY is not set. Run this script from a graphical terminal on the server.\n' >&2
  exit 2
fi

MODEL="${REPO}/logs/openduckpro2/${LOAD_RUN}/model_${CHECKPOINT}.pt"
if [[ ! -f "${MODEL}" ]]; then
  printf 'checkpoint does not exist: %s\n' "${MODEL}" >&2
  exit 2
fi

# Checkpoints retain the CUDA device used during training (cuda:1). Keep all
# physical GPUs visible so torch.load can deserialize them, while simulation
# and inference are explicitly placed on GPU_ID below.
unset CUDA_VISIBLE_DEVICES
export PATH="${ENV_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO}"
printf 'Playing model_%s.pt on physical GPU %s (DISPLAY=%s)\n' \
  "${CHECKPOINT}" "${GPU_ID}" "${DISPLAY}"

exec "${ENV_DIR}/bin/python" legged_gym/scripts/play.py \
  --task=openduckpro2 \
  --experiment_name=openduckpro2 \
  --sim_device="cuda:${GPU_ID}" \
  --rl_device="cuda:${GPU_ID}" \
  --num_envs=1 \
  --load_run="${LOAD_RUN}" \
  --checkpoint="${CHECKPOINT}" \
  --keyboard_commands \
  --keyboard_heading_hold \
  --keyboard_heading_kp=2.0
