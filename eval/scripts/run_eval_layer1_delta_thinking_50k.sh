#!/usr/bin/env bash
# run_eval_layer1_delta_thinking_50k.sh — Post-training eval orchestrator for L1d.
#
# Mirrors validated scripts/run_eval_layer1_delta.sh from the 10K run, with
# the V2 recall pipeline added (per user request).
#
# Pipeline (sequential on this node's local GPUs):
#   1. (optional) wait for SFT training to finish
#   2. pick best checkpoint by min eval_loss; patch eval YAML
#   3. group A — teacher self-prediction (no vLLM):
#         M1 (layer1_delta) + M2 (layer1_delta_judge) + Recall V2 (--teacher-mode)
#         Recall pipeline shares a candidates-dir cache across all 3 subjects.
#   4. groups B (sft) and C (zero_shot):
#         vLLM serve (GPU 0, port 8000) → generate → M1 → M2 → Recall → kill vLLM
#   5. aggregate_eval_report.py → REPORT.md
#
# Run (detached, on node-1 where L1d SFT trained):
#   setsid nohup bash scripts/run_eval_layer1_delta_thinking_50k.sh \
#       > logs/run_eval_layer1_delta_thinking_50k.log 2>&1 < /dev/null &
#
set -uo pipefail

ROOT=/scratch/azureml/cr/j/fc096b74f20c46ae94d7fab7e20c1aa4/cap/data-capability/wd/INPUT_msndni/shares/users/yuhangbai/MAIProfileSFT_50k
RUN_NAME=layer1_delta_thinking_50k_4o-v1
RUN_DIR=/scratch/azureml/cr/j/fc096b74f20c46ae94d7fab7e20c1aa4/exe/wd/MAIProfileSFT_local_runs/${RUN_NAME}
CFG=${ROOT}/configs/eval/${RUN_NAME}.yaml
RES=${ROOT}/eval_results/${RUN_NAME}
TEST=${ROOT}/data/splits/layer1_delta_thinking_50k_4o/test.jsonl
LOG_DIR=${RES}/logs
CAND_DIR=${RES}/m2_recall/candidates    # shared across teacher/sft/zs (per recall.py docs)
PORT=8000
GPU=0
SMOKE_LIMIT=${SMOKE_LIMIT:-0}
WAIT_FOR_TRAINING=${WAIT_FOR_TRAINING:-1}
POLL_S=60
MAX_HOURS=12

mkdir -p "$LOG_DIR" "${RES}/predictions" "${RES}/m1_objective" "${RES}/m2_judge" "${RES}/m2_recall" "$CAND_DIR"
LOG=${ROOT}/logs/run_eval_${RUN_NAME}.log
exec > >(tee -a "$LOG") 2>&1

echo "[$(date +%T)] === run_eval_layer1_delta_thinking_50k starting ==="
echo "  RUN_DIR=$RUN_DIR  CFG=$CFG  PORT=$PORT  GPU=$GPU  SMOKE_LIMIT=$SMOKE_LIMIT"

source /opt/conda/etc/profile.d/conda.sh
conda activate pipeline-rl

if [[ -f /home/aiscuser/.secrets/maiprofile_sft.env ]]; then
    set -a; source /home/aiscuser/.secrets/maiprofile_sft.env; set +a
fi
export AZURE_OPENAI_KEY="${AZURE_OPENAI_KEY:-${AZURE_OPENAI_API_KEY:-}}"

unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK CROSS_RANK || true

cd "$ROOT"

# ---------------------------------------------------------------------------
# Step 1 — wait for training
# ---------------------------------------------------------------------------
if [[ "$WAIT_FOR_TRAINING" == "1" ]]; then
    echo "[$(date +%T)] waiting for $RUN_NAME training to finish (max ${MAX_HOURS}h)"
    deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))
    while true; do
        now=$(date +%s)
        if (( now > deadline )); then
            echo "[$(date +%T)] ERROR: timeout waiting for training"; exit 1
        fi
        if ! pgrep -f "sft_train.py.*${RUN_NAME}" >/dev/null 2>&1; then
            if ls -d ${RUN_DIR}/checkpoint-* >/dev/null 2>&1; then
                echo "[$(date +%T)] training process gone and checkpoints exist — proceeding"
                break
            fi
        fi
        if (( now % 600 < POLL_S )); then
            last=$(tail -1 ${ROOT}/logs/${RUN_NAME}.master.log 2>/dev/null | head -c 200)
            echo "[$(date +%T)] heartbeat: ${last:-no log yet}"
        fi
        sleep "$POLL_S"
    done
    sleep 30
fi

# ---------------------------------------------------------------------------
# Step 2 — pick best checkpoint; patch YAML
# ---------------------------------------------------------------------------
echo "[$(date +%T)] picking best checkpoint by eval_loss..."
BEST_CKPT=$(python3 <<PY
import json, os, glob
ck_dirs = sorted(glob.glob('${RUN_DIR}/checkpoint-*'),
                 key=lambda p: int(p.rsplit('-',1)[-1]))
best = None; best_loss = float('inf')
for d in ck_dirs:
    f = os.path.join(d, 'trainer_state.json')
    if not os.path.exists(f): continue
    s = json.load(open(f))
    step_n = int(d.rsplit('-',1)[-1])
    for e in s.get('log_history', []):
        if 'eval_loss' in e and e.get('step') == step_n:
            if e['eval_loss'] < best_loss:
                best_loss = e['eval_loss']; best = d
            break
if os.path.isfile(os.path.join('${RUN_DIR}', 'model.safetensors')) or \
   os.path.isfile(os.path.join('${RUN_DIR}', 'model.safetensors.index.json')):
    best = '${RUN_DIR}'
if not best and ck_dirs:
    best = ck_dirs[-1]
print(best or '')
PY
)
if [[ -z "$BEST_CKPT" ]]; then
    echo "[$(date +%T)] ERROR: no checkpoint found under $RUN_DIR"; exit 1
fi
echo "[$(date +%T)] best checkpoint = $BEST_CKPT"

python3 <<PY
import yaml
with open('$CFG') as f: cfg = yaml.safe_load(f)
cfg['subject_models']['sft']['model_path'] = '$BEST_CKPT'
with open('$CFG','w') as f: yaml.safe_dump(cfg, f, sort_keys=False)
print('  patched sft.model_path =', cfg['subject_models']['sft']['model_path'])
PY

LIMIT_FLAG=""
if [[ "$SMOKE_LIMIT" != "0" ]]; then
    LIMIT_FLAG="--limit ${SMOKE_LIMIT}"
fi

# ---------------------------------------------------------------------------
# Step 3 — Group A: teacher (gpt-4o self-prediction) M1 + M2 + Recall
# ---------------------------------------------------------------------------
echo "[$(date +%T)] === Group A: teacher (gpt-4o) self-prediction ==="
TEACHER_PRED=${RES}/predictions/teacher.jsonl
python3 <<PY
import json
limit = int('${SMOKE_LIMIT}') if '${SMOKE_LIMIT}' != '0' else None
with open('${TEST}') as fin, open('${TEACHER_PRED}', 'w') as fout:
    for i, line in enumerate(fin):
        if limit is not None and i >= limit: break
        r = json.loads(line)
        m = r.get('metadata', {}) or {}
        ref = r['messages'][-1]['content']
        out = {
            'record_idx': i,
            'user_id': m.get('user_id', ''),
            'date': m.get('date', ''),
            'delta_index': m.get('delta_index', 0),
            'reference': ref,
            'prediction': ref,
            'latency_s': 0.0,
            'input_n_signals': 0,
            'error': None,
        }
        fout.write(json.dumps(out, ensure_ascii=False) + '\n')
print('built teacher.jsonl')
PY

python scripts/eval_m1_layer1_delta.py \
    --predictions "$TEACHER_PRED" \
    --test-jsonl  "$TEST" \
    --output      "${RES}/m1_objective/teacher.json" \
    --model-tag   teacher --teacher-mode \
    2>&1 | tee "${LOG_DIR}/m1_teacher.log"

python scripts/eval_m2_layer1_delta_judge.py \
    --config      "$CFG" \
    --predictions "$TEACHER_PRED" \
    --output      "${RES}/m2_judge/m2_teacher.json" \
    --model-tag   teacher --teacher-mode \
    $LIMIT_FLAG \
    2>&1 | tee "${LOG_DIR}/m2_teacher.log"

# Recall V2 — teacher (builds candidates cache; subsequent runs reuse it).
python scripts/eval_m2_layer1_delta_recall.py \
    --config      "$CFG" \
    --predictions "$TEACHER_PRED" \
    --test-jsonl  "$TEST" \
    --output      "${RES}/m2_recall/recall_teacher.json" \
    --model-tag   teacher \
    --candidates-dir "$CAND_DIR" \
    --teacher-mode \
    $LIMIT_FLAG \
    2>&1 | tee "${LOG_DIR}/recall_teacher.log"

# ---------------------------------------------------------------------------
# Step 4 — Groups B (sft) and C (zero_shot)
# ---------------------------------------------------------------------------
for SUBJECT in sft zero_shot; do
    echo "[$(date +%T)] === Subject: ${SUBJECT} ==="
    PRED=${RES}/predictions/${SUBJECT}.jsonl

    if curl -fs "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "[$(date +%T)] ERROR: port $PORT already in use"; exit 1
    fi

    echo "[$(date +%T)] starting vLLM for ${SUBJECT} on GPU $GPU port $PORT"
    CUDA_VISIBLE_DEVICES=$GPU \
    setsid python scripts/serve_sft.py --config "$CFG" --subject "$SUBJECT" --port "$PORT" \
        > "${LOG_DIR}/vllm_${SUBJECT}.log" 2>&1 &
    VLLM_PID=$!
    echo "$VLLM_PID" > /tmp/vllm_${SUBJECT}.pid
    trap 'kill -9 $VLLM_PID 2>/dev/null; pkill -9 -f "vllm.entrypoints.*--port ${PORT}" 2>/dev/null' EXIT

    READY=0
    for i in $(seq 1 180); do
        sleep 5
        if curl -fs "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
            echo "[$(date +%T)] vLLM ready after $((i*5))s"; READY=1; break
        fi
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM died early"; tail -60 "${LOG_DIR}/vllm_${SUBJECT}.log"; exit 1
        fi
    done
    [[ "$READY" != "1" ]] && { echo "ERROR: vLLM never ready"; tail -60 "${LOG_DIR}/vllm_${SUBJECT}.log"; exit 1; }

    python scripts/generate_outputs.py \
        --config   "$CFG" \
        --subject  "$SUBJECT" \
        --endpoint "http://127.0.0.1:${PORT}/v1" \
        --output   "$PRED" \
        --timeout  1200 \
        $LIMIT_FLAG \
        2>&1 | tee "${LOG_DIR}/generate_${SUBJECT}.log"

    python scripts/eval_m1_layer1_delta.py \
        --predictions "$PRED" \
        --test-jsonl  "$TEST" \
        --output      "${RES}/m1_objective/${SUBJECT}.json" \
        --model-tag   "$SUBJECT" \
        2>&1 | tee "${LOG_DIR}/m1_${SUBJECT}.log"

    python scripts/eval_m2_layer1_delta_judge.py \
        --config      "$CFG" \
        --predictions "$PRED" \
        --output      "${RES}/m2_judge/m2_${SUBJECT}.json" \
        --model-tag   "$SUBJECT" \
        $LIMIT_FLAG \
        2>&1 | tee "${LOG_DIR}/m2_${SUBJECT}.log"

    python scripts/eval_m2_layer1_delta_recall.py \
        --config      "$CFG" \
        --predictions "$PRED" \
        --test-jsonl  "$TEST" \
        --output      "${RES}/m2_recall/recall_${SUBJECT}.json" \
        --model-tag   "$SUBJECT" \
        --candidates-dir "$CAND_DIR" \
        $LIMIT_FLAG \
        2>&1 | tee "${LOG_DIR}/recall_${SUBJECT}.log"

    echo "[$(date +%T)] stopping vLLM (PID $VLLM_PID)"
    kill -TERM "$VLLM_PID" 2>/dev/null || true
    sleep 5
    kill -KILL "$VLLM_PID" 2>/dev/null || true
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    trap - EXIT
    sleep 5
done

# ---------------------------------------------------------------------------
# Step 5 — aggregate
# ---------------------------------------------------------------------------
echo "[$(date +%T)] building REPORT.md"
python scripts/aggregate_eval_report.py --config "$CFG" \
    2>&1 | tee "${LOG_DIR}/aggregate.log" || echo "WARN: aggregate failed"

echo "[$(date +%T)] === DONE ==="
echo "  predictions: ${RES}/predictions/{teacher,sft,zero_shot}.jsonl"
echo "  M1:          ${RES}/m1_objective/{teacher,sft,zero_shot}.json"
echo "  M2 judge:    ${RES}/m2_judge/m2_{teacher,sft,zero_shot}.json"
echo "  M2 recall:   ${RES}/m2_recall/recall_{teacher,sft,zero_shot}.json"
echo "  REPORT:      ${RES}/REPORT.md"
