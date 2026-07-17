#!/usr/bin/env bash
# Pass B — generate rollouts on node-1 across all 8 GPUs (data-parallel shards).
#
# Runs ON node-1 (after the script + conda env have been staged there). Each GPU
# serves its own vLLM instance of the SFT-50K ckpt and handles a strided subset
# of v2 prompts (prompts[shard::8]); shard outputs are written to node-1 LOCAL
# disk, then concatenated. Caller rsyncs the merged file back to cosmos.
#
# Usage (from node-0):
#   scp pass_b_rollout_difficulty.py run_pass_b_generate_node1.sh node-1:/home/aiscuser/
#   ssh node-1 'bash /home/aiscuser/run_pass_b_generate_node1.sh <V2_JSONL> <SFT_CKPT> [LIMIT]'
set -euo pipefail

V2_JSONL="${1:?need v2 jsonl path}"
SFT_CKPT="${2:?need sft ckpt path}"
LIMIT="${3:-0}"   # 0 = full

PY=/home/aiscuser/.conda/envs/pipeline-rl/bin/python
SCRIPT=/home/aiscuser/pass_b_rollout_difficulty.py
OUTDIR=/home/aiscuser/passb_rollouts
LOGDIR=/home/aiscuser/passb_logs
NUM_SHARDS=8

mkdir -p "$OUTDIR" "$LOGDIR"
echo "[launch] $(date) v2=$V2_JSONL ckpt=$SFT_CKPT limit=$LIMIT shards=$NUM_SHARDS"

pids=()
for s in $(seq 0 $((NUM_SHARDS-1))); do
  CUDA_VISIBLE_DEVICES=$s nohup "$PY" "$SCRIPT" generate \
    --v2-jsonl "$V2_JSONL" \
    --sft-ckpt "$SFT_CKPT" \
    --rollouts-jsonl "$OUTDIR/shard_${s}.jsonl" \
    --num-shards "$NUM_SHARDS" --shard-id "$s" \
    --max-model-len 40960 --gpu-mem-util 0.9 \
    --limit "$LIMIT" \
    > "$LOGDIR/shard_${s}.log" 2>&1 &
  pids+=($!)
  echo "[launch] shard $s -> GPU $s pid ${pids[-1]}"
  sleep 5   # stagger ckpt reads off cosmos
done

echo "[launch] waiting for ${#pids[@]} shards..."
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[done] shard $i OK"
  else
    echo "[done] shard $i FAILED (see $LOGDIR/shard_${i}.log)"
    fail=1
  fi
done

if [[ $fail -ne 0 ]]; then
  echo "[merge] SKIPPED — at least one shard failed"
  exit 1
fi

MERGED=/home/aiscuser/passb_rollouts_merged.jsonl
cat "$OUTDIR"/shard_*.jsonl > "$MERGED"
echo "[merge] $(wc -l < "$MERGED") lines -> $MERGED"
echo "[launch] all shards complete $(date)"
