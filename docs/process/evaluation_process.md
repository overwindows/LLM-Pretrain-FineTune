# Evaluation — Execution Process (layer1-delta, 260623)

How the 6-model layer1-delta benchmark was actually run on this AML box, end to
end, with the **real hyperparameters** used. This is the "how we did it" companion
to [`report/260623_layer1_delta_model_comparison.md`](../report/260623_layer1_delta_model_comparison.md)
(the "what we found"). Everything referenced below lives under `MAIDistillation0623/eval/`.

> **Models benchmarked:** teacher `gpt-4o` (reference labels), our `SFT` &
> `RL` (Qwen3-4B-Thinking-2507), and SOTA `gpt-5`, `gpt-5-2` (=gpt-5 backend),
> `gpt-5.1`.
> **Test set:** `test_1k.jsonl` — n=1000, sampled seed=42 from the 4531-line test
> split (`eval/data/splits/layer1_delta_thinking_50k_4o/`).
> **All settings come from** `eval/configs/eval/layer1_delta_thinking_50k_4o-v1.local.yaml`.

---

## Pipeline at a glance

```
            SFT / RL (local 4B)                     SOTA (Azure OpenAI)
            ───────────────────                     ───────────────────
1. serve    serve_sft.py → vLLM @ :8000             (no serve; hosted API)
2. generate generate_outputs.py  ───┐               generate_outputs_aoai.py
                                     ▼                        │
                          eval_results/predictions/<tag>_1k.jsonl  ◄───┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            ▼                        ▼                        ▼
3a. M1                    3b. M2 judge              3c. M3 recall
   eval_m1_layer1_delta.py  eval_m2_layer1_delta_   eval_m2_layer1_delta_recall.py
   (rule-based, no LLM)     judge.py (gpt-5.1)      + ..._topic_recall.py (gpt-5.1)
            │                        │                        │
   eval_results/m1/         eval_results/m2/        eval_results/m2_recall/
                                                    {interest,topic}/
```

---

## Step 1 — Serve the local 4B models (SFT / RL)

Script: `eval/scripts/serve_sft.py` — a thin wrapper around `vllm serve`.

Launch (per subject, one GPU):
```bash
PRL=/home/aiscuser/.conda/envs/pipeline-rl/bin
CUDA_VISIBLE_DEVICES=0 $PRL/python scripts/serve_sft.py \
  --config configs/eval/layer1_delta_thinking_50k_4o-v1.local.yaml \
  --subject sft   # or: rl
```

vLLM serving params (from config `vllm:` block):

| param | value |
|---|---|
| port | 8000 |
| max_model_len | 40960 |
| gpu_memory_utilization | 0.9 |
| dtype | bfloat16 |
| prefix caching | **disabled** (`--no-enable-prefix-caching`; observed to occasionally corrupt output on vllm 0.8.5) |
| served name (SFT) | `qwen3-4b-thinking-sft-layer1delta-50k-4o-v1` |
| served name (RL) | `qwen3-4b-thinking-rl-l1d-stage5` |

Checkpoints served:
- **SFT** = `MAIProfileSFT_50k/checkpoints/layer1_delta_thinking_50k_4o-v1`
  (top-level merged best ckpt = checkpoint-2260, best eval_loss 0.0833).
- **RL** = `MAIProfileSFT_runs/qwen3-4b-l1d-rl-stage5`.
- (zero_shot base `Qwen3-4B-Thinking-2507` is wired in config but was not part of
  the final 6-model table.)

## Step 2 — Generate predictions

### 2a. Local SFT / RL — `eval/scripts/generate_outputs.py`
Hits the local vLLM OpenAI-compatible endpoint.
```bash
$PRL/python scripts/generate_outputs.py \
  --config configs/eval/layer1_delta_thinking_50k_4o-v1.local.yaml \
  --subject sft \
  --endpoint http://127.0.0.1:8000/v1 \
  --concurrency 64
```
Generation params (from config `generation:` block):

| param | value |
|---|---|
| temperature | 0.2 |
| top_p | 1.0 |
| max_tokens | 8192 |
| enable_thinking | **true** (thinking traces kept in output) |
| concurrency | 64 (async `Semaphore`) |
| per-request timeout | 120 s (`--timeout` default) |
| retries | up to 4, **only on transient errors** (timeout / conn reset / 429 / 5xx); content-filter etc. fail fast, no retry |

> **RL truncation history (important).** `max_tokens` has been **8192 the whole
> time** — it was *not* the lever. Early RL runs truncated badly (only ~954/1000
> parsed) because **`max_model_len` was 20480**, too small for RL's long thinking +
> JSON (prompt + output exceeded the context window, cutting JSON mid-object). The
> fix was raising **`max_model_len` 20480 → 40960** in the `.local.yaml` (the
> non-local `…-v1.yaml` still has 20480). The **final stored `rl_1k.jsonl` is the
> post-fix run**: 0 errors, only **2/1000** records still hit `finish_reason=length`
> (vs SFT max 5263 completion tokens, well under cap). So the 8192 documented here is
> the value used for the numbers in the report — but if you re-run and see truncation,
> raise `max_model_len` first, then `max_tokens`.

### 2b. SOTA reasoning models — `eval/scripts/generate_outputs_aoai.py`
Separate script because Azure **reasoning** models differ from chat models:
- use `max_completion_tokens` (NOT `max_tokens`),
- do **not** send `temperature` / `top_p` (reasoning models reject non-default),
- optional `reasoning_effort`.

```bash
$PRL/python scripts/generate_outputs_aoai.py \
  --config configs/eval/layer1_delta_thinking_50k_4o-v1.local.yaml \
  --model gpt-5 --model-tag gpt5 \
  --concurrency 16 --resume
```
| param | value |
|---|---|
| max_completion_tokens | 16000 |
| temperature / top_p | not sent |
| reasoning_effort | optional (none used by default) |
| **concurrency** | **16** (per requirement for all SOTA generation) |
| per-request timeout | 300 s |
| retries | up to 4, transient-only; `--resume` re-uses any already-completed records |

### Output of Step 2 (same schema for every model)
`eval/eval_results/predictions/<tag>_1k.jsonl`, one record per line:
```
{record_idx, user_id, date, delta_index, reference, prediction,
 latency_s, input_n_signals, completion_tokens, finish_reason, error}
```
Files (each ~11–12 MB): `sft_1k.jsonl`, `rl_1k.jsonl`, `gpt5_1k.jsonl`,
`gpt5_2_1k.jsonl`, `gpt51_1k.jsonl`. (teacher has **no** prediction file — it is
the `reference` field, identical across all of them, evaluated via `--teacher-mode`.)

> Note: a `gpt52_1k.jsonl` also exists but is a **dead file** (all records errored
> with `DeploymentNotFound`). The valid gpt-5-2 run is `gpt5_2_1k.jsonl`.

## Step 3 — Run the three metric families

### 3a. M1 — objective / rule-based (no LLM) — `eval/scripts/eval_m1_layer1_delta.py`
```bash
$PRL/python scripts/eval_m1_layer1_delta.py \
  --predictions eval_results/predictions/sft_1k.jsonl \
  --test-jsonl  data/splits/layer1_delta_thinking_50k_4o/test_1k.jsonl \
  --output      eval_results/m1/sft.json \
  --model-tag   sft
# teacher: add --teacher-mode and feed any predictions file (uses its reference field)
```
- Computes: parse_rate, schema_rate, n_interests, top/int, interest P/R/F1 (greedy
  name-Jaccard ≥ 0.5), matched topic-IoU, name ROUGE-L, **evidence_fidelity**
  (pred `evidence.action` verbatim in input signals), **length_ratio**.
- **Thinking handling:** `safe_json_loads()` extracts the JSON by brace-matching
  from the first `{`, so the `<think>…</think>` prefix is stripped for all metrics
  **except** `length_ratio`, which uses the raw full text (this is why the report
  also gives a JSON-only length ratio).
- Output: `eval_results/m1/<tag>.json` + `<tag>.per_record.csv`.

### 3b. M2 — LLM judge (gpt-5.1) — `eval/scripts/eval_m2_layer1_delta_judge.py`
```bash
$PRL/python scripts/eval_m2_layer1_delta_judge.py \
  --config configs/eval/layer1_delta_thinking_50k_4o-v1.local.yaml \
  --predictions eval_results/predictions/sft_1k.jsonl \
  --output eval_results/m2/sft.json \
  --model-tag sft
# teacher: --teacher-mode
```
Judge endpoint (from config `judge:` block):

| param | value |
|---|---|
| endpoint | `https://msncompanioneu2.cognitiveservices.azure.com/` |
| api_version | 2024-12-01-preview |
| deployment / model | gpt-5.1 |
| reasoning_effort | none |
| max_completion_tokens | 16000 |
| concurrency | 32 |

- Scores each interest (utility, precision, coherence, granularity_broad) and each
  topic (utility, precision, coherence, granularity) on 1–10.
- Prompts: `eval/prompts/layer1_delta_eval_interest_name(_v2).md`,
  `layer1_delta_eval_topics.md`.
- Output: `eval_results/m2/<tag>.json` + per-item `<tag>.interest_name.jsonl` /
  `<tag>.topics.jsonl`.

### 3c. M3 — signal-grounded recall (gpt-5.1) — driver `eval/scripts/run_all_recall.sh`
Two scripts, both propose→ground(→rescue)→judge:
- interest: `eval_m2_layer1_delta_recall.py` (propose→ground→**rescue**→judge)
- topic: `eval_m2_layer1_delta_topic_recall.py` (propose→ground→judge)

```bash
bash scripts/run_all_recall.sh   # runs interest then topic, all 6 models sequentially
```
| param | value |
|---|---|
| propose / ground / rescue / judge model | **all gpt-5.1** (same endpoint as M2) |
| JUDGE_BATCH_SIZE | 10 proposals/request (official `interest_recall.py` value) |
| concurrency (CONC) | 32 |
| candidate set | built **once per user from raw denoised signals**, model-independent, **cached & shared** across all 6 models (`…/candidates/`) |
| config | single-agent, **no-SBERT** (every model item is a candidate; matches the old single-agent recall table) |

- `recall = covered / grounded`; interest reported as overall / matched / broad,
  topic as recall + grounded sum.
- Models run **sequentially** (they all hit the same gpt-5.1 judge endpoint).
- Output: `eval_results/m2_recall/{interest,topic}/<tag>.json` (+ per-record jsonl).

## Step 4 — Aggregate
- `eval/scripts/aggregate_eval_report.py` rolls M1/M2 JSONs into a single report.
- The curated human-facing comparison is `eval/COMPARISON.md`; the M3 plan/notes are
  `eval/RECALL_PLAN.md`; the goal + full code inventory is `eval/plan.md`.

---

## Reliability / coverage notes (this run)
- M1 errors: 1 per SOTA model (1 content-filtered input); SFT/RL 0.
- M2 judge_fail (Azure content_filter on test data, symmetric & unavoidable):
  teacher 9, sft 7, rl 7, gpt-5 12, gpt-5-2 17, gpt-5.1 20.
- M2 parse_fail ≤1 for all except gpt-5.1 (12/1000 = 1.2%, immaterial to means).
- M3 usable candidate set: interest 997 / topic 996 of 1000.

## Secrets
- Judge/generation Azure key read from env (`AZURE_OPENAI_KEY` / `AOAI_KEY`) —
  **never committed**. Rotate the key after any session where it was exported in a
  shell.

## Environment
- Python env: conda `pipeline-rl` at `/home/aiscuser/.conda/envs/pipeline-rl/bin`
  (invoked by absolute path; `conda activate` is broken on this box).
- vLLM 0.8.5; single GPU (`CUDA_VISIBLE_DEVICES=0`).

---

## Not yet stored here (future process docs)
- SFT training process (data recipe, hyperparams, launch).
- RL training process (verifier, reward, online judge, RL loop).
- RL data-cleaning process (PassA/PassB curation, hard-case construction).
