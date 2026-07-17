# Layer1 Delta RL — Stage 1 scaffolding

This folder is a **staging area**. Drop everything under `verifier_layer1/` and `conf/` into
`QED-Nano/training/` after you `git clone https://github.com/CMU-AIRe/QED-Nano`:

```
QED-Nano/training/
├── pipelinerl/                  # vendored, do NOT edit
├── conf/
│   └── layer1_stage1.yaml       # ← copied from rl_layer1/conf/
├── verifier_layer1/             # ← copied from rl_layer1/verifier_layer1/
│   ├── parser.py
│   ├── server.py
│   ├── reward/
│   └── tests/
```

## What's in here

| File | Purpose |
|---|---|
| `verifier_layer1/parser.py` | Strip `<think>` block, parse JSON, validate Layer1 schema |
| `verifier_layer1/reward/gate.py` | parse_ok × schema_ok hard/soft gate |
| `verifier_layer1/reward/fidelity.py` | Evidence verbatim-grounding + coverage floor |
| `verifier_layer1/reward/count_length.py` | Anti-collapse: count band + length band |
| `verifier_layer1/reward/hallucination.py` | Stage 1: `1 - fidelity` proxy; Stage 2: RM head stub |
| `verifier_layer1/reward/compose.py` | Orchestrator — gate × weighted sum of components |
| `verifier_layer1/server.py` | FastAPI endpoint (`POST /`, `GET /health`) for PipelineRL |
| `verifier_layer1/tests/test_reward.py` | 5 bad-output fixtures + 2 good-output fixtures |
| `conf/layer1_stage1.yaml` | Hydra config: fork of `proof_demo-thinking`, points at sft-50K |

## Running tests locally

```bash
cd verifier_layer1
pytest tests/ -v
```

## Running the verifier standalone (for debugging)

```bash
cd verifier_layer1
LAYER1_STAGE=1 uvicorn server:app --host 0.0.0.0 --port 8001
```

Smoke test:
```bash
curl -X POST http://localhost:8001/ -H 'content-type: application/json' \
  -d '{"prompt": "...", "completion": "<think>...</think>[{...}]",
       "metadata": {"input_signals": [{"raw_record": "..."}]}}'
```

## Request / response contract

PipelineRL's actor posts to `POST /` with:

```json
{
  "prompt": "<full prompt string>",
  "completion": "<model output, may include <think> block>",
  "metadata": {
    "input_signals": [{"raw_record": "..."}, ...],
    "record_id": "user_xxx_delta_yyy",
    "teacher_ref": null
  }
}
```

Response:

```json
{
  "reward": 0.42,
  "components": {
    "parse_ok": true,
    "schema_ok": true,
    "gate": 1.0,
    "fidelity": 0.85,
    "coverage": 0.7,
    "count_gate": 1.0,
    "length_gate": 1.0,
    "anti_collapse": 1.0,
    "anti_hallucination": 0.85,
    "fidelity_score": 0.595,
    "weighted_sum": 0.42
  },
  "metadata": {
    "n_interests": 4,
    "n_topics": 9,
    "n_evidence": 14,
    "total_chars": 1820,
    "had_think_block": true,
    "errors": []
  }
}
```

## Pre-RL validation TODO (Phase 2, runs against this verifier offline)

1. Compute `pct(M2.Precision==0)` baseline on 300-record subset for sft-50K → confirm ≤ 5%
2. Reward monotonicity test: teacher / sft-50K / sft-10K / zero-shot → ZS must be strictly lowest
3. Reward histogram on 1K eval → mean ∈ [0.3, 0.7], std ≥ 0.15
4. Spearman `corr(reward, human_score)` on 30–50 annotated → ≥ 0.4

Driver scripts for these live under `data_prep/` (TODO, not in this scaffold yet).
