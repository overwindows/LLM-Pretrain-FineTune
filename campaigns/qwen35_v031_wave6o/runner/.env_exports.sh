source /opt/conda/etc/profile.d/conda.sh
conda activate vllm-ablation
export HF_HOME=/scratch/hf_cache
export HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN in shell before sourcing this file}
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
export GEMMA4_MODEL_PATH=/scratch/hf_cache/hub/models--google--gemma-4-26B-A4B-it/snapshots/b2a81a03d25f927590a91d84ba43f96e8ef7349f
export GEMMA4_TEXT_ONLY_MODEL_PATH=/scratch/hf_cache/gemma-4-26B-A4B-it-text-only
export GEMMA4_ASSISTANT_MODEL_PATH=/scratch/hf_cache/hub/models--google--gemma-4-26B-A4B-it-assistant/snapshots/f188f476dc11dd5bb3014dc861529d316bce49d3
