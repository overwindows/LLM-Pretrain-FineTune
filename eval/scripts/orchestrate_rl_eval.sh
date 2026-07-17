#!/usr/bin/env bash
# Autonomous orchestrator: after RL generation completes, run RL M1 + RL M2.
# Respects the global judge concurrency <=32 by waiting for SFT M2 + teacher M2
# to finish before launching RL M2. Designed to run unattended via nohup.
#
# Required env: AZURE_OPENAI_KEY (exported by caller)
# PIDs passed as env: RL_GEN_PID, SFT_M2_PID, TEACHER_M2_PID
set -uo pipefail

PRL=/home/aiscuser/.conda/envs/pipeline-rl/bin
D=/scratch/azureml/cr/j/65fcf9508e03476381b75ace1f02fb73/exe/wd/l1d_eval
CFG="$D/configs/eval/layer1_delta_thinking_50k_4o-v1.local.yaml"
TEST="$D/data/splits/layer1_delta_thinking_50k_4o/test_1k.jsonl"
RL_PRED="$D/eval_results/predictions/rl_1k.jsonl"
cd "$D" || exit 1

log() { echo "[orchestrate $(date '+%H:%M:%S')] $*"; }

wait_pid() {  # wait_pid <pid> <label>
  local pid="$1" label="$2"
  if [[ -z "$pid" ]]; then log "$label: no pid, skipping wait"; return; fi
  log "waiting for $label (pid $pid)..."
  while kill -0 "$pid" 2>/dev/null; do sleep 15; done
  log "$label (pid $pid) finished"
}

# ---- 1. Wait for RL generation to complete -------------------------------
wait_pid "${RL_GEN_PID:-}" "RL generation"
# give the rename a moment, then verify final file exists
for i in $(seq 1 12); do
  [[ -f "$RL_PRED" ]] && break
  sleep 5
done
if [[ ! -f "$RL_PRED" ]]; then
  log "ERROR: $RL_PRED not found after RL gen; falling back to .partial"
  [[ -f "$RL_PRED.partial" ]] && cp "$RL_PRED.partial" "$RL_PRED"
fi
RL_N=$(wc -l < "$RL_PRED" 2>/dev/null || echo 0)
log "RL predictions ready: $RL_N rows"

# ---- 2. RL M1 (rule-based, no judge key needed) --------------------------
log "running RL M1..."
"$PRL/python" scripts/eval_m1_layer1_delta.py \
  --predictions "$RL_PRED" \
  --test-jsonl "$TEST" \
  --output "$D/eval_results/m1/rl.json" \
  --model-tag rl \
  > "$D/logs/m1_rl_1k.log" 2>&1
log "RL M1 done -> eval_results/m1/rl.json (rc=$?)"

# ---- 3. Wait for SFT M2 + teacher M2 (concurrency <=32) ------------------
wait_pid "${SFT_M2_PID:-}" "SFT M2"
wait_pid "${TEACHER_M2_PID:-}" "teacher M2"
# extra guard: ensure no eval_m2 process is still alive
while pgrep -f eval_m2_layer1_delta_judge >/dev/null 2>&1; do
  log "another M2 judge still running, waiting..."; sleep 15
done

# ---- 4. RL M2 (judge) ----------------------------------------------------
if [[ -z "${AZURE_OPENAI_KEY:-}" ]]; then
  log "ERROR: AZURE_OPENAI_KEY not set; cannot run RL M2"; exit 2
fi
log "running RL M2 judge..."
"$PRL/python" scripts/eval_m2_layer1_delta_judge.py \
  --config "$CFG" \
  --predictions "$RL_PRED" \
  --output "$D/eval_results/m2/rl.json" \
  --model-tag rl \
  > "$D/logs/m2_rl_1k.log" 2>&1
log "RL M2 done -> eval_results/m2/rl.json (rc=$?)"

log "ALL DONE. Summary files:"
ls -la "$D"/eval_results/m1/rl.json "$D"/eval_results/m2/{sft,teacher,rl}.json 2>/dev/null
