#!/bin/bash
# interactive_debug.sh — Run the SFT/RL pipeline on an interactive AML Singularity node.
#
# This script replaces the AML job submission flow for interactive debugging.
# It handles:
#   1. Clone LLM-Pretrain-FineTune (if not already done)
#   2. Set up Python deps (setup_env.sh)
#   3. Discover Cosmos mount and export AGENTIC_COSMOS_ROOT
#   4. Write /tmp/project_runtime.yaml (with Linux easytrain_root patched in)
#   5. Run run_pipeline.py
#
# Usage:
#   bash /tmp/gpu_code/scripts/interactive_debug.sh [--stages sft] [--force]
#
#   By default runs SFT only. Pass --stages sft data_clean rl eval for the full chain.
#
# Prerequisites (must be set before running this script):
#   export WANDB_API_KEY=...
#   export AZURE_OPENAI_KEY=...  # only needed for eval stage (M2 LLM judge)
#   source ~/.secrets/maiprofile_sft.env  # or set the vars manually

set -e

# ── Parse args ───────────────────────────────────────────────────────────────
STAGES="sft"
FORCE_FLAG=""
GPU_CODE_DIR="/tmp/gpu_code"
PROJECT_YAML_SRC=""   # override: path to project.yaml on this node (optional)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stages)
      shift
      STAGES=""
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        STAGES="$STAGES $1"
        shift
      done
      STAGES="${STAGES# }"
      ;;
    --force)
      FORCE_FLAG="--force"
      shift
      ;;
    --gpu-code-dir)
      GPU_CODE_DIR="$2"
      shift 2
      ;;
    --project)
      PROJECT_YAML_SRC="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

echo "============================================================"
echo "  interactive_debug.sh"
echo "  stages: $STAGES"
echo "  force:  ${FORCE_FLAG:-no}"
echo "============================================================"

# ── Step 1: Clone GPU code if needed ─────────────────────────────────────────
GITHUB_REPO="https://github.com/overwindows/LLM-Pretrain-FineTune"
GITHUB_BRANCH="main"

if [ -d "$GPU_CODE_DIR/.git" ]; then
  echo "[Step 1] $GPU_CODE_DIR already exists — pulling latest..."
  git -C "$GPU_CODE_DIR" pull --ff-only || {
    echo "  WARNING: git pull failed (dirty worktree?). Using existing clone."
  }
else
  echo "[Step 1] Cloning $GITHUB_REPO@$GITHUB_BRANCH → $GPU_CODE_DIR ..."
  git clone --depth=1 --branch "$GITHUB_BRANCH" "$GITHUB_REPO" "$GPU_CODE_DIR"
  echo "  Clone OK."
fi
echo "  gpu_code contents:"
ls "$GPU_CODE_DIR/"

# ── Step 2: Install Python deps ───────────────────────────────────────────────
echo ""
echo "[Step 2] Installing Python deps (setup_env.sh --skip-if-ok)..."
bash "$GPU_CODE_DIR/scripts/setup_env.sh" --skip-if-ok

# Re-apply PYTHONPATH (bash subshell export doesn't propagate)
PTCA_PY=/opt/conda/envs/ptca/bin/python
PTCA_PYVER=$($PTCA_PY -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
export PYTHONUSERBASE=/tmp/ptca_user
export PYTHONPATH=/tmp/ptca_user/lib/python${PTCA_PYVER}/site-packages:${PYTHONPATH:-}
echo "  PYTHONPATH = $PYTHONPATH"

# ── Step 3: Discover Cosmos mount ─────────────────────────────────────────────
echo ""
echo "[Step 3] Discovering Cosmos mount..."

if [ -n "$AGENTIC_COSMOS_ROOT" ] && [ -d "$AGENTIC_COSMOS_ROOT" ]; then
  echo "  AGENTIC_COSMOS_ROOT already set: $AGENTIC_COSMOS_ROOT"
else
  # Use same glob as agent/cosmos.py
  COSMOS_GLOB="/scratch/azureml/cr/j/*/cap/data-capability/wd/INPUT_*"
  COSMOS_CANDIDATES=$(ls -d $COSMOS_GLOB 2>/dev/null | head -10)

  if [ -z "$COSMOS_CANDIDATES" ]; then
    echo "  ERROR: No Cosmos mount found at $COSMOS_GLOB"
    echo "  Please set AGENTIC_COSMOS_ROOT manually:"
    echo "    export AGENTIC_COSMOS_ROOT=/scratch/azureml/cr/j/<JOBID>/cap/data-capability/wd/INPUT_<NAME>"
    exit 1
  fi

  echo "  Found Cosmos mounts:"
  echo "$COSMOS_CANDIDATES" | while read -r m; do echo "    $m"; done

  # Prefer INPUT_cosmos_data; fall back to first match that isn't a small RO file input
  COSMOS_ROOT=$(echo "$COSMOS_CANDIDATES" | grep "INPUT_cosmos_data" | head -1)
  if [ -z "$COSMOS_ROOT" ]; then
    # Take the first match that looks like a large share (has subdirs)
    COSMOS_ROOT=$(echo "$COSMOS_CANDIDATES" | head -1)
  fi

  export AGENTIC_COSMOS_ROOT="$COSMOS_ROOT"
  echo "  AGENTIC_COSMOS_ROOT = $AGENTIC_COSMOS_ROOT"
fi

# Quick sanity check: base model and training data should be reachable
BASE_MODEL_SUBPATH="shares/users/yuhangbai/MAIDistillation0623/models/base/Qwen3-4B-Thinking-2507"
TRAIN_DATA_SUBPATH="shares/users/yuhangbai/MAIProfileSFT_50k/data/splits/layer1_delta_thinking_50k_4o/train.jsonl"

echo ""
echo "  Checking expected data paths under cosmos..."
if [ -d "$AGENTIC_COSMOS_ROOT/$BASE_MODEL_SUBPATH" ]; then
  echo "  ✓ base_model found"
else
  echo "  ✗ base_model NOT found: $AGENTIC_COSMOS_ROOT/$BASE_MODEL_SUBPATH"
  echo "    Listing cosmos root:"
  ls "$AGENTIC_COSMOS_ROOT/" 2>/dev/null | head -20 || echo "    (empty or unreadable)"
fi
if [ -f "$AGENTIC_COSMOS_ROOT/$TRAIN_DATA_SUBPATH" ]; then
  echo "  ✓ train.jsonl found"
else
  echo "  ✗ train.jsonl NOT found: $AGENTIC_COSMOS_ROOT/$TRAIN_DATA_SUBPATH"
fi

# ── Step 4: Write project_runtime.yaml ───────────────────────────────────────
echo ""
echo "[Step 4] Writing /tmp/project_runtime.yaml..."

RUNTIME_YAML=/tmp/project_runtime.yaml

# If user gave a project.yaml on this node, patch just the easytrain_root line.
# Otherwise generate from scratch using known project.yaml values.
if [ -n "$PROJECT_YAML_SRC" ] && [ -f "$PROJECT_YAML_SRC" ]; then
  echo "  Patching $PROJECT_YAML_SRC → $RUNTIME_YAML"
  sed 's|easytrain_root:.*|easytrain_root: "'"$GPU_CODE_DIR"'"|' \
      "$PROJECT_YAML_SRC" > "$RUNTIME_YAML"
else
  echo "  Generating from embedded defaults (project.yaml not mounted — this is normal)"
  cat > "$RUNTIME_YAML" << YAML
# project_runtime.yaml — generated by interactive_debug.sh
# This mirrors project.yaml from agentic_sft_rl/project.yaml with
# easytrain_root corrected for Linux.

task: layer1_delta
run_name: layer1_delta_v1

cosmos_root: null  # auto-discovered by cosmos.py (AGENTIC_COSMOS_ROOT is set)

base_model: "\${cosmos_root}/shares/users/yuhangbai/MAIDistillation0623/models/base/Qwen3-4B-Thinking-2507"

data:
  sft_train_jsonl: "\${cosmos_root}/shares/users/yuhangbai/MAIProfileSFT_50k/data/splits/layer1_delta_thinking_50k_4o/train.jsonl"
  sft_val_jsonl:   "\${cosmos_root}/shares/users/yuhangbai/MAIProfileSFT_50k/data/splits/layer1_delta_thinking_50k_4o/val.jsonl"
  rl_train_jsonl:  null

local_scratch_root: "/scratch/agentic_runs/\${run_name}"
cosmos_persist_root: "\${cosmos_root}/agentic_runs/\${run_name}"

stages:
  - sft
  - data_clean
  - rl
  - eval

sft:
  config_template: sft/configs/sft/layer1_delta_thinking_50k_4o-v2.repro.yaml.tmpl
  num_gpus: 8
  conda_env: ptca
  max_seq_len: 14336
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  num_train_epochs: 2
  learning_rate: 5.0e-6
  warmup_ratio: 0.03
  lr_scheduler_type: cosine
  seed: 42
  mask_think_in_loss: false
  save_steps: 200
  save_total_limit: 3
  attn_implementation: sdpa
  deepspeed_config: sft/configs/deepspeed/ds_config_zero3.json
  accelerate_config: sft/configs/accelerate/accelerate_ds3.yaml
  wandb_project: maiprofile-sft

data_clean:
  conda_env: ptca
  pass_a:
    max_prompt_tokens: 12000
    verifier_dir: null
  pass_b:
    n_rollouts: 16
    temperature: 1.0
    top_p: 0.95
    max_tokens: 4096
    easy_std: 0.05
    sat_mean: 0.95
    sat_std: 0.03
    max_model_len: 40960
    num_shards: 1
    conda_env: ptca

rl:
  conda_env: ptca
  config_name: layer1_rl
  easytrain_rl_dir: null
  num_gpus: 7
  gpu_offset: 1
  actor_fraction: 3
  preprocessor_fraction: 1
  finetune_fraction: 3
  actor_llm_start_port: 8180
  attempts: 8
  train_batch_size: 4
  gradient_accumulation_passes: 16
  seq_length: 20000
  max_train_steps: 500
  save_checkpoint_steps: 50
  kl_coef: 0.02
  entropy_bonus: 0.0001
  learning_rate: 1.0e-6
  max_tokens: 8192
  temperature: 0.8
  wandb_project: layer1-delta-rl
  judge_enabled: false
  judge_endpoint: null
  judge_deployment: gpt-5.1

eval:
  conda_env: ptca
  easytrain_eval_dir: null
  n_test_samples: 1000
  max_model_len: 40960
  metrics:
    - m1_rule
    - m2_llm

secrets:
  azure_openai_key_env: AZURE_OPENAI_KEY
  wandb_api_key_env: WANDB_API_KEY
  secrets_file: "~/.secrets/maiprofile_sft.env"

# Corrected for Linux interactive node (was Q:/LLM-Pretrain-FineTune on Windows)
easytrain_root: "${GPU_CODE_DIR}"
YAML
fi

echo "  Written: $RUNTIME_YAML"

# ── Step 5: Load secrets if available ────────────────────────────────────────
SECRETS_FILE="$HOME/.secrets/maiprofile_sft.env"
if [ -f "$SECRETS_FILE" ]; then
  echo ""
  echo "[Step 5] Loading secrets from $SECRETS_FILE..."
  # shellcheck source=/dev/null
  set -a
  source "$SECRETS_FILE"
  set +a
  echo "  Secrets loaded."
elif [ -z "$WANDB_API_KEY" ]; then
  echo ""
  echo "  WARNING: $SECRETS_FILE not found and WANDB_API_KEY not set."
  echo "  Set it manually: export WANDB_API_KEY=..."
fi

export EASYTRAIN_ROOT="$GPU_CODE_DIR"
echo "  EASYTRAIN_ROOT = $EASYTRAIN_ROOT"

# ── Step 6: Run the pipeline ──────────────────────────────────────────────────
echo ""
echo "[Step 6] Running pipeline — stages: $STAGES"
echo "------------------------------------------------------------"

STAGES_ARGS=""
for s in $STAGES; do
  STAGES_ARGS="$STAGES_ARGS --stages $s"
done
# consolidate: --stages a b c
STAGES_ARG="--stages $STAGES"

$PTCA_PY "$GPU_CODE_DIR/scripts/run_pipeline.py" \
  --project "$RUNTIME_YAML" \
  $STAGES_ARG \
  $FORCE_FLAG

echo ""
echo "============================================================"
echo "  interactive_debug.sh complete."
echo "============================================================"
