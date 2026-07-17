#!/bin/bash
# Launch script for Layer0_signal thinking-variant SFT on 50K curation data.
# Designed for a single 8-GPU A100 node.
#
# Use:
#   setsid nohup ./scripts/launch_layer0_thinking_50k_sft.sh \
#       > logs/layer0_signal_thinking_50k_4o-v1.log 2>&1 < /dev/null &

set -euo pipefail

PERSISTENT=/scratch/azureml/cr/j/fc096b74f20c46ae94d7fab7e20c1aa4/cap/data-capability/wd/INPUT_msndni/shares/users/yuhangbai/MAIProfileSFT_50k
RUN_NAME=layer0_signal_thinking_50k_4o-v1
LOCAL_RUN_DIR=/scratch/azureml/cr/j/fc096b74f20c46ae94d7fab7e20c1aa4/exe/wd/MAIProfileSFT_local_runs/${RUN_NAME}
LOG_FILE=${PERSISTENT}/logs/${RUN_NAME}.log

mkdir -p "${LOCAL_RUN_DIR}"
mkdir -p "${PERSISTENT}/logs"
mkdir -p "${PERSISTENT}/checkpoints"

source /opt/conda/etc/profile.d/conda.sh
conda activate pipeline-rl

# Load API keys (WANDB_API_KEY, AZURE_OPENAI_*) from local secrets file
if [[ -f /home/aiscuser/.secrets/maiprofile_sft.env ]]; then
    set -a
    source /home/aiscuser/.secrets/maiprofile_sft.env
    set +a
fi

export WANDB_PROJECT=maiprofile-sft
export WANDB_NAME=${RUN_NAME}
export WANDB_TAGS=layer0_signal,sft,qwen3-4b-thinking,gpt4o-teacher,a100,50k,no-mask-think
export WANDB_MODE=online

# NCCL stability
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Strip AzureML multi-node MPI envs so accelerate runs single-node DDP
unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT \
      LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK || true

# Use all 8 GPUs on this node
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd "${PERSISTENT}"

echo "[$(date)] Starting ${RUN_NAME}"
echo "[$(date)] cwd=${PERSISTENT}  python=$(which python)"
echo "[$(date)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
python --version
nvidia-smi -L

accelerate launch \
    --config_file configs/accelerate/accelerate_ds3.yaml \
    --num_processes 8 \
    scripts/sft_train.py \
    --config configs/sft/layer0_signal_thinking_50k_4o-v1.yaml

EXIT_CODE=$?
echo "[$(date)] Training finished with exit code ${EXIT_CODE}"

# Always rsync — we want any saved checkpoints even if training exited non-zero
# (e.g. crash near the end). global_step* (DS sharded states) excluded to keep
# the persistent copy small; HF safetensors come from each checkpoint-*/ root.
if [[ -d "${LOCAL_RUN_DIR}" ]]; then
    DEST="${PERSISTENT}/checkpoints/${RUN_NAME}"
    echo "[$(date)] Syncing ${LOCAL_RUN_DIR} -> ${DEST}  (excluding global_step*)"
    mkdir -p "${DEST}"
    rsync -av --exclude='global_step*' "${LOCAL_RUN_DIR}/" "${DEST}/" 2>&1 | tail -30
    echo "[$(date)] Sync done."
fi

exit ${EXIT_CODE}
