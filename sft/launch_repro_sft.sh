#!/bin/bash
# Portable reproduction launcher for Layer1_delta thinking-variant SFT (50K).
#
# Renders configs/sft/layer1_delta_thinking_50k_4o-v2.repro.yaml.tmpl into a
# concrete config (filling absolute paths via env), then runs 8-GPU SFT.
#
# Defaults assume this repo lives at MAIDistillation0623/sft and the persisted
# base model + data splits sit in cosmos (see below). Override any var inline:
#   BASE_MODEL=/path OUTPUT_DIR=/path ./scripts/launch_repro_sft.sh
#
# Use:
#   setsid nohup ./scripts/launch_repro_sft.sh \
#       > logs/repro_sft.log 2>&1 < /dev/null &

set -euo pipefail

# --- Resolve SFT dir (this script + sft_train.py + configs/ all live here) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"          # EasyPosttrain/sft — holds sft_train.py, configs/

# --- Cosmos persistence root (parent of MAIDistillation0623) ---
COSMOS="${COSMOS:-$(cd "${REPO_ROOT}/.." && pwd)}"   # .../MAIDistillation0623

# --- Paths (override via env) ---
RUN_NAME="${RUN_NAME:-layer1_delta_thinking_50k_4o-v2-repro}"
BASE_MODEL="${BASE_MODEL:-${COSMOS}/models/base/Qwen3-4B-Thinking-2507}"
DATA_DIR="${DATA_DIR:-${COSMOS}/../MAIProfileSFT_50k/data/splits/layer1_delta_thinking_50k_4o}"
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_DIR}/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${DATA_DIR}/val.jsonl}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/configs/deepspeed/ds_config_zero3.json}"
ACCEL_CONFIG="${ACCEL_CONFIG:-${REPO_ROOT}/configs/accelerate/accelerate_ds3.yaml}"

# Training writes frequent checkpoints — keep on LOCAL fast disk, rsync after.
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/local_sft_runs/${RUN_NAME}}"
PERSIST_CKPT_DIR="${PERSIST_CKPT_DIR:-${COSMOS}/models/sft/${RUN_NAME}}"

export PRL="${PRL:-/home/aiscuser/.conda/envs/pipeline-rl/bin}"
PYTHON="${PYTHON:-${PRL}/python}"

mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/logs" "${REPO_ROOT}/configs/sft/rendered"

# --- Render the template config ---
RENDERED="${REPO_ROOT}/configs/sft/rendered/${RUN_NAME}.yaml"
export BASE_MODEL TRAIN_JSONL VAL_JSONL OUTPUT_DIR DEEPSPEED_CONFIG
envsubst '${BASE_MODEL} ${TRAIN_JSONL} ${VAL_JSONL} ${OUTPUT_DIR} ${DEEPSPEED_CONFIG}' \
    < "${REPO_ROOT}/configs/sft/layer1_delta_thinking_50k_4o-v2.repro.yaml.tmpl" \
    > "${RENDERED}"

echo "[$(date)] Rendered config -> ${RENDERED}"
echo "  BASE_MODEL=${BASE_MODEL}"
echo "  TRAIN_JSONL=${TRAIN_JSONL}"
echo "  VAL_JSONL=${VAL_JSONL}"
echo "  OUTPUT_DIR=${OUTPUT_DIR}"
echo "  DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG}"

# --- Load secrets (WANDB_API_KEY, AZURE_OPENAI_*) if present ---
if [[ -f /home/aiscuser/.secrets/maiprofile_sft.env ]]; then
    set -a; source /home/aiscuser/.secrets/maiprofile_sft.env; set +a
fi

export WANDB_PROJECT=maiprofile-sft
export WANDB_NAME=${RUN_NAME}
export WANDB_MODE="${WANDB_MODE:-online}"

# NCCL / CUDA stability
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Strip AzureML multi-node MPI envs so accelerate runs single-node DDP
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT \
      LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK || true

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

cd "${REPO_ROOT}"
echo "[$(date)] Starting ${RUN_NAME}  python=$(which "${PYTHON}")"
"${PYTHON}" --version
nvidia-smi -L

"${PRL}/accelerate" launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes 8 \
    sft_train.py \
    --config "${RENDERED}"

EXIT_CODE=$?
echo "[$(date)] Training finished with exit code ${EXIT_CODE}"

# --- rsync local checkpoints -> cosmos (exclude DS sharded global_step*) ---
if [[ -d "${OUTPUT_DIR}" ]]; then
    echo "[$(date)] Syncing ${OUTPUT_DIR} -> ${PERSIST_CKPT_DIR} (excluding global_step*)"
    mkdir -p "${PERSIST_CKPT_DIR}"
    rsync -a --exclude='global_step*' "${OUTPUT_DIR}/" "${PERSIST_CKPT_DIR}/" 2>&1 | tail -20
    echo "[$(date)] Sync done -> ${PERSIST_CKPT_DIR}"
fi

exit ${EXIT_CODE}
