#!/usr/bin/env bash
# Run interest-level + topic-level recall for all 6 models on test_1k.
# Stage-1 candidate sets are model-INDEPENDENT and cached (shared across models);
# the first model in each metric populates the 1000-user cache, the rest reuse it.
# Models run SEQUENTIALLY (all hit the same gpt-5.1 judge endpoint).
set -uo pipefail

D=/scratch/azureml/cr/j/65fcf9508e03476381b75ace1f02fb73/exe/wd/l1d_eval
cd "$D"
PRL=/home/aiscuser/.conda/envs/pipeline-rl/bin
CFG=configs/eval/layer1_delta_thinking_50k_4o-v1.local.yaml
TEST=data/splits/layer1_delta_thinking_50k_4o/test_1k.jsonl
# Judge concurrency: 32 matches the M2 judge phase that already ran safely against
# this same gpt-5.1 endpoint; single run (no competing process) so total ≤ 32.
CONC=32

# model_tag : predictions_file : extra_flags
# teacher reuses any predictions file (the `reference` field is the identical
# gpt-4o gold across all of them) with --teacher-mode.
declare -a MODELS=(
  "teacher:eval_results/predictions/gpt5_1k.jsonl:--teacher-mode"
  "sft:eval_results/predictions/sft_1k.jsonl:"
  "rl:eval_results/predictions/rl_1k.jsonl:"
  "gpt5:eval_results/predictions/gpt5_1k.jsonl:"
  "gpt5_2:eval_results/predictions/gpt5_2_1k.jsonl:"
  "gpt51:eval_results/predictions/gpt51_1k.jsonl:"
)

run_metric () {
  local script="$1" outdir="$2" canddir="$3" label="$4"
  echo "############################################################"
  echo "### $label  ($(date '+%H:%M:%S'))"
  echo "############################################################"
  mkdir -p "$outdir" "$canddir"
  for entry in "${MODELS[@]}"; do
    IFS=':' read -r tag preds extra <<< "$entry"
    echo "--- [$label] model=$tag preds=$preds $extra  ($(date '+%H:%M:%S')) ---"
    $PRL/python "$script" \
      --config "$CFG" \
      --predictions "$preds" \
      --test-jsonl "$TEST" \
      --output "$outdir/$tag.json" \
      --model-tag "$tag" \
      --candidates-dir "$canddir" \
      --concurrency "$CONC" \
      $extra 2>&1 | tail -2
    echo "--- [$label] $tag DONE ($(date '+%H:%M:%S')) ---"
  done
}

run_metric scripts/eval_m2_layer1_delta_recall.py \
  eval_results/m2_recall/interest \
  eval_results/m2_recall/interest/candidates \
  "INTEREST recall"

run_metric scripts/eval_m2_layer1_delta_topic_recall.py \
  eval_results/m2_recall/topic \
  eval_results/m2_recall/topic/candidates \
  "TOPIC recall"

echo "ALL RECALL RUNS COMPLETE ($(date '+%H:%M:%S'))"
