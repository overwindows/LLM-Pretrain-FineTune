#!/usr/bin/env bash
# run_experiments.sh — Launch one or more ablation experiments.
#
# Usage:
#   ./run_experiments.sh E001               # single experiment, sc1, 2 reps
#   ./run_experiments.sh E001,E003,E006     # comma-separated list
#   ./run_experiments.sh --all              # all 16 experiments
#   ./run_experiments.sh E011 --scenario sc2 --reps 3
#
# Environment variables (can override before calling):
#   GEMMA4_MODEL_PATH            full model checkpoint path
#   GEMMA4_TEXT_ONLY_MODEL_PATH  text-only checkpoint path
#   GEMMA4_ASSISTANT_MODEL_PATH  MTP assistant model path
#
# The script resolves VLLM_ATTENTION_BACKEND and FlashInfer knobs per
# experiment ID rather than having them hard-coded in bench_experiment.py
# (vllm imports freeze env vars at import time, so they must be set here).

set -euo pipefail
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
cd "$(dirname "$SCRIPT_PATH")"

# Python interpreter used for helper probes and benchmark execution.
: "${PYTHON_BIN:=python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: PYTHON_BIN '$PYTHON_BIN' not found in PATH"
  exit 1
fi

# ---------------------------------------------------------------------------
# CUDA runtime library path (required for precompiled vLLM binaries)
# ---------------------------------------------------------------------------
# Precompiled vLLM binaries need CUDA runtime libs from the active env's
# nvidia-* site-packages. Auto-detect from the current python's site-packages
# so this works under different conda/venv locations.
_NVIDIA_LIB_GLOBS=$($PYTHON_BIN - <<'PY' 2>/dev/null || true
import glob, os, sysconfig
sp = sysconfig.get_paths()["purelib"]
roots = []
for sub in ("nvidia/cu13/lib", "nvidia/cu12/lib"):
    roots.extend(glob.glob(os.path.join(sp, sub)))
# Also include all nvidia/*/lib dirs (cudnn, cublas, etc.)
roots.extend(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
print(":".join(sorted(set(roots))))
PY
)
if [[ -n "${_NVIDIA_LIB_GLOBS}" ]]; then
  export LD_LIBRARY_PATH="${_NVIDIA_LIB_GLOBS}:${LD_LIBRARY_PATH:-}"
fi

# ---------------------------------------------------------------------------
# Default model paths (override via env). Defaults point to HF model IDs so
# weights are pulled from the Hugging Face hub on first use.
# ---------------------------------------------------------------------------
: "${GEMMA4_MODEL_PATH:=google/gemma-4-26B-A4B-it}"
: "${GEMMA4_TEXT_ONLY_MODEL_PATH:=${GEMMA4_MODEL_PATH}}"
: "${GEMMA4_ASSISTANT_MODEL_PATH:=google/gemma-4-26B-A4B-it-assistant}"
export GEMMA4_MODEL_PATH GEMMA4_TEXT_ONLY_MODEL_PATH GEMMA4_ASSISTANT_MODEL_PATH

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
EXP_IDS=()
SCENARIO="sc1"
REPS=2
RUN_ALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)          RUN_ALL=1; shift ;;
    --scenario)     SCENARIO="$2"; shift 2 ;;
    --reps)         REPS="$2"; shift 2 ;;
    --mock-a100-40g)
      # Mock A100 40 GB on an A100 80 GB device by halving the planned
      # gpu_memory_utilization values (KV-cache budget = 40/80 of original).
      # Note: BF16 weights (~48 GB) exceed 40 GB and are expected to OOM.
      export GPU_MEM_SCALE=0.5
      shift
      ;;
    --gpu-mem-scale)
      export GPU_MEM_SCALE="$2"
      shift 2
      ;;
    --list)
      "$PYTHON_BIN" bench_experiment.py --exp E001 --list
      exit 0
      ;;
    -h|--help)
      sed -n '2,22p' "$SCRIPT_PATH"     # print the usage block at top of script
      exit 0
      ;;
    *)              EXP_IDS+=("$1"); shift ;;
  esac
done

if [[ $RUN_ALL -eq 1 ]]; then
  EXP_IDS=("E001,E002,E003,E004,E005,E006,E007,E008,E009,E010,E011,E012,E013,E014,E015,E016")
fi

if [[ ${#EXP_IDS[@]} -eq 0 ]]; then
  echo "ERROR: no experiment ID provided. Use --all or specify e.g. E001"
  exit 1
fi

# Fail fast if the selected Python env cannot import benchmark dependencies.
echo "Python executable: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
echo "Python version   : $($PYTHON_BIN -V 2>&1)"
if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1; then
import vllm
import numpy
import scipy
PY
  echo "ERROR: Python preflight failed. '$PYTHON_BIN' must import vllm, numpy, and scipy."
  echo "Hint: prepend the intended env to PATH, e.g. 'export PATH=/opt/conda/envs/ptca/bin:$PATH'"
  exit 1
fi

if [[ "${GEMMA4_TEXT_ONLY_MODEL_PATH}" == "${GEMMA4_MODEL_PATH}" ]]; then
  echo "WARNING: GEMMA4_TEXT_ONLY_MODEL_PATH equals GEMMA4_MODEL_PATH."
  echo "         Text-only experiments will run against full model weights."
fi

# Normalize: allow both comma-separated and space-separated experiment IDs.
EXP_IDS_CSV=$(IFS=','; echo "${EXP_IDS[*]}")

# ---------------------------------------------------------------------------
# Per-experiment environment variable table.
# These must be set BEFORE vllm is imported, so this shell wrapper is the
# right place.
#
# NOTE: Gemma 4 has heterogeneous attention head dims (256 and 512), so vLLM
# forces TRITON_ATTN regardless of VLLM_ATTENTION_BACKEND.  The env var is
# set here only so it appears in logs for reproducibility — it has no runtime
# effect on this model.
#
# Mapping:
#   VLLM_ATTENTION_BACKEND          : FLASH_ATTN (default, all exps)
#   VLLM_USE_FLASHINFER_MOE_FP8     : 0 | 1  (H100 only; A100 must use 0)
#   VLLM_USE_FLASHINFER_SAMPLER     : 0 | 1  (JIT compile fix for old nvcc)
# ---------------------------------------------------------------------------
set_env_for_exp() {
  local exp="$1"
  # Default: safe values for all experiments
  export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
  export VLLM_USE_FLASHINFER_SAMPLER="0"

  # VLLM_USE_FLASHINFER_MOE_FP8: needed for FP8 MoE on H100 (sm_90+), must be 0 on A100.
  # Only enable for experiments that use FP8 weights (E002–E013, E017–E018).
  # E001, E014, E015, E016 use full-precision weights (BF16), so FP8 MoE is irrelevant.
  case "$exp" in
    E001|E014|E015|E016)
      # BF16 weights — FlashInfer FP8 MoE is irrelevant; keep 0
      export VLLM_USE_FLASHINFER_MOE_FP8="0"
      ;;
    E002|E003|E004|E005|E006|E007|E008|E009|E010|E011|E012|E013|E017|E018)
      # FP8 weights: enable FlashInfer FP8 MoE on H100, keep off on A100
      COMPUTE_CAP=$($PYTHON_BIN -c "import torch; cc=torch.cuda.get_device_capability(); print(cc[0]*10+cc[1])" 2>/dev/null || echo "0")
      if [[ "$COMPUTE_CAP" -ge 90 ]]; then
        export VLLM_USE_FLASHINFER_MOE_FP8="1"
        echo "  → H100 detected (sm_${COMPUTE_CAP}): enabling VLLM_USE_FLASHINFER_MOE_FP8=1"
      else
        export VLLM_USE_FLASHINFER_MOE_FP8="0"
        echo "  → non-H100 (sm_${COMPUTE_CAP}): VLLM_USE_FLASHINFER_MOE_FP8=0 (Marlin FP8 MoE fallback)"
      fi
      ;;
    *)
      export VLLM_USE_FLASHINFER_MOE_FP8="0"
      echo "WARNING: unknown exp '$exp' — using safe defaults"
      ;;
  esac

  echo "  ENV: VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND  VLLM_USE_FLASHINFER_MOE_FP8=$VLLM_USE_FLASHINFER_MOE_FP8  VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
IFS=',' read -ra EXPS <<< "$EXP_IDS_CSV"

echo "=================================================="
echo "  Ablation benchmark"
echo "  Experiments : ${EXP_IDS_CSV}"
echo "  Scenario    : ${SCENARIO}"
echo "  Reps        : ${REPS}"
echo "  GPU_MEM_SCALE: ${GPU_MEM_SCALE:-1.0}"
echo "  BENCH_RESULTS_DIR: ${BENCH_RESULTS_DIR:-results}"
echo "  $(date)"
echo "=================================================="

FAILED=()
for EXP in "${EXPS[@]}"; do
  EXP="$(echo "$EXP" | tr '[:lower:]' '[:upper:]' | tr -d ' ')"
  echo ""
  echo ">>> Setting up environment for ${EXP}"
  set_env_for_exp "$EXP"

  echo ">>> Launching bench_experiment.py --exp ${EXP} --scenario ${SCENARIO} --reps ${REPS}"
  if "$PYTHON_BIN" bench_experiment.py \
      --exp "${EXP}" \
      --scenario "${SCENARIO}" \
      --reps "${REPS}"; then
    echo ">>> ${EXP} COMPLETED"
  else
    echo ">>> ${EXP} FAILED (exit $?)"
    FAILED+=("$EXP")
  fi
done

echo ""
echo "=================================================="
echo "  Ablation run finished"
echo "  Results: ${BENCH_RESULTS_DIR:-results}/all_runs.csv"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "  FAILED experiments: ${FAILED[*]}"
  exit 1
fi
echo "  All experiments PASSED"
echo "=================================================="
