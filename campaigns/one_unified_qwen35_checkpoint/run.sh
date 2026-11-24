#!/usr/bin/env bash
set -euo pipefail

checkpoint=$1
cosmos_data=$2
project_yaml=$3
agent_pkg=$4
metrics_output=$5

campaign_dir=$(cd -- "$(dirname -- "$0")" && pwd)
repo=$(cd -- "$campaign_dir/../.." && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/qwen-mai-XXXXXXXX")
cp -R "$repo" "$work/code"

# Use the exact pipeline and task pack inputs of the original AML job.
cp -R "$agent_pkg/agent" "$agent_pkg/tasks" "$work/code/"
cp "$agent_pkg/scripts/run_pipeline.py" "$work/code/scripts/"
if [[ -d "$agent_pkg/configs" ]]; then
    cp -R "$agent_pkg/configs" "$work/code/"
fi

ptca_python=/opt/conda/envs/ptca/bin/python
python_version=$($ptca_python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
# setup_env.sh uses this same user base in the original job environment.
export PYTHONUSERBASE=/tmp/ptca_user
export PYTHONPATH="$PYTHONUSERBASE/lib/python$python_version/site-packages:${PYTHONPATH:-}"
"$ptca_python" -m pip install --user 'pyarrow==24.0.0'
bash "$work/code/scripts/setup_env.sh"
"$ptca_python" -m pip install --user 'transformers==5.2.0' 'accelerate==1.8.1' 'deepspeed==0.15.4'

"$ptca_python" "$campaign_dir/run_scoring.py" "$checkpoint" "$cosmos_data" \
    "$project_yaml" "$metrics_output" "$work"
