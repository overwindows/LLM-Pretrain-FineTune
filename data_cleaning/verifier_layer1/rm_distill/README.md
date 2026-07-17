# `rm_distill/`

Offline judging tools for **Phase 2 reward validation** and
**Phase 5 reward-model (RM) distillation**.

## Files

| File | Purpose |
|---|---|
| `llm_judge_client.py` | Minimal async Azure OpenAI client (gpt54-eval / gpt52 / gpt4o), DefaultAzureCredential fallback. No dependency on `maiprofilev3dev/`. |
| `run_judge.py` | Async judge runner with `--concurrency` semaphore, exponential backoff, **per-record streaming write** (resumable). |

## Input record schema (`--input` JSONL)

```json
{
  "record_id": "<user_id>__<YYYYMMDD>__<source>",
  "source": "sft-50K",
  "interests": [
    {"interest_name": "...",
     "topics": [{"topic": "...", "evidence": [{"action": "..."}]}]}
  ]
}
```

The `interests` block is the **model output** to be judged. `source` is
a free-form label (`teacher` / `sft-50K` / `sft-10K` / `zero-shot`) used
later when training the distilled RM.

## Output record schema

```json
{
  "record_id": "...",
  "source": "sft-50K",
  "input_summary": {"n_interests": 4, "n_topics": 9},
  "judge_raw": "<raw judge json text>",
  "judge_parsed": [ {"interest_name": "...", "scores": {"utility": 8, ...}}, ... ],
  "metrics": {
    "n_scored": 4,
    "mean_utility": 7.5, "mean_precision": 8.0,
    "mean_coherence": 7.0, "mean_granularity_broad": 7.5,
    "pct_precision_zero": 0.0
  },
  "usage": {"prompt_tokens": ..., "completion_tokens": ...},
  "elapsed_s": 4.7,
  "judge_model": "gpt54-eval",
  "ts": "2026-05-29T03:21:50+00:00",
  "error": null
}
```

`pct_precision_zero` is the key metric for **Phase 2 #1** (KL-anchor
quality check: must be ≤ 5 % on sft-50K).

## How to run

### Auth setup

Either:

```powershell
$env:GPT54_EVAL_API_KEY = "<key>"   # preferred for batch runs
```

…or rely on `DefaultAzureCredential` (works when `az login` succeeded
or you're on an AML compute with managed identity).

### Phase 2 #1 — pct(Precision==0) on sft-50K subset

```powershell
python -m rl_layer1.rm_distill.run_judge `
  --input  "rl_layer1/data/sft50k_outputs_300.jsonl" `
  --output "rl_layer1/data/sft50k_judge_v2.jsonl" `
  --prompt-file "MaiProfile-main/maiprofilev3dev/evaluation/prompts/layer1_delta_eval_interest_name_v2.md" `
  --judge-model gpt54-eval `
  --concurrency 32 `
  --limit 300
```

Then check `pct_precision_zero` across all rows — should be ≤ 0.05.

### Phase 2 #2 — reward monotonicity (4 sources)

Same command, run once per source (teacher / sft-50K / sft-10K /
zero-shot). Then aggregate offline:

```python
import json
from collections import defaultdict
agg = defaultdict(list)
for src, path in {"teacher":..., "sft50k":..., "sft10k":..., "zs":...}.items():
    for ln in open(path):
        r = json.loads(ln)
        if r["error"]: continue
        agg[src].append(r["metrics"]["mean_utility"])
# Expect: mean(teacher) > mean(sft50k) > mean(sft10k) > mean(zs)
```

### Phase 5 — bulk RM-training data (5–10 K × 4 sources)

Run on each of the 4 sources, then concatenate the JSONLs and use them
as training labels for the distilled RM (`Harrier-oss-v1-270m` + heads).

## Tuning

- **`--concurrency`**: start at 32. If you see ≥1 % 429s drop to 16.
- **`--max-retries`**: default 5, exponential backoff w/ jitter
  (base = 2 s → up to ~32 s per retry).
- **`--limit`**: dev-mode aid; runs only the first N records.
- **`--no-resume`**: force re-judging records that already have a row
  in `--output`. Default behavior is **resume** (skip rows whose
  `record_id` is already in the output file with `error == null`).
