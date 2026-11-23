#!/usr/bin/env bash
set -euo pipefail

runtime_dir=$(mktemp -d)
trap 'rm -rf -- "$runtime_dir"' EXIT

bash scripts/setup_env.sh --profile qwen35 --python "${QWEN35_PYTHON:-python3}" --env-dir "$runtime_dir/venv"
python="$runtime_dir/venv/bin/python"
"$python" -m pip install 'peft==0.18.1' 'bitsandbytes==0.49.2'
"$python" -m pip check
visible_devices="${CUDA_VISIBLE_DEVICES-0}"
export CUDA_VISIBLE_DEVICES="${visible_devices%%,*}"
"$python" campaigns/one_unified_qwen35/run_sft.py "$@"
