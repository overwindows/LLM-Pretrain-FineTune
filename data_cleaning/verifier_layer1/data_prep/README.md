# `data_prep/`

Offline tools to build the **RL training prompt set** for Layer1-delta.

## Files

| File | Purpose |
|---|---|
| `prepare_hard_cases.py` | Mine "hard" `(user_id, date)` prompts from teacher dogfood data, optionally filtered by SFT-50K M1.F1 |

## Output schema

One JSON per line:

```json
{
  "record_id": "<user_id>__<YYYYMMDD>",
  "user_id": "...",
  "date": "20260103",
  "system_prompt": "<contents of layer1_delta.md>",
  "user_message": "Date: ...\nUser ID: ...\n...",
  "input_signals": [{"Date": "...", "Source": "...", "Action": "...", ...}],
  "n_signals": 17,
  "stratum": "medium",
  "source_run": "gpt52__layer1_delta__v0.8/20260103",
  "sft50k_f1": 0.42
}
```

- `system_prompt` + `user_message` → fed verbatim to the RL actor (Qwen3-4B-Thinking).
- `input_signals` → fed to the verifier service for the fidelity reward.

## How to run

### Fallback mode (no SFT eval yet) — uniform stratified by signal-count

```powershell
python -m rl_layer1.data_prep.prepare_hard_cases `
  --dogfood-root "MaiProfile-main/maiprofilev3dev/dogfood/20260320/gpt52__layer1_delta__v0.8" `
  --prompt-file  "MaiProfile-main/maiprofilev3dev/prompts/layer1_delta.md" `
  --output       "rl_layer1/data/train_hard.jsonl" `
  --n 8000 --holdout-n 200 --seed 42
```

This emits:
- `rl_layer1/data/train_hard.jsonl` — training prompts
- `rl_layer1/data/train_hard.holdout.jsonl` — user-disjoint 200-record holdout (for RL exit eval)

### F1-driven hard-case mining (when SFT-50K eval is ready)

```powershell
python -m rl_layer1.data_prep.prepare_hard_cases `
  --dogfood-root "..." --prompt-file "..." `
  --sft-eval     "path/to/sft50k_m1_per_record.jsonl" `
  --f1-low 0.30 --f1-high 0.60 `
  --output "rl_layer1/data/train_hard_f1.jsonl" --n 8000 --seed 42
```

The SFT eval file is expected to have one line per record. Any of these
field shapes is accepted:

```json
{"record_id": "<uid>__<yyyymmdd>", "m1": {"f1": 0.42}}
{"user_id": "...", "date": "...",  "f1": 0.42}
{"user_id": "...", "date": "...",  "m1_f1": 0.42}
```

## Stratification

Records are bucketed by **# kept signals** (after layer0 filtering):

| Bucket | Range |
|---|---|
| `xs` | 1–4 |
| `small` | 5–14 |
| `medium` | 15–29 |
| `large` | 30–59 |
| `xl` | 60+ |

`--min-signals` (default 5) drops the `xs` bucket — extremely short
deltas don't give RL useful signal.

## Why a holdout?

The Stage 1 exit condition includes:

> `holdout_m1_f1 ≥ 0.595` (= 95 % of sft-50K baseline 0.626)

To measure that during RL training, the verifier server pulls
holdout prompts from `train_hard.holdout.jsonl` every K steps. The
holdout is **user-disjoint** so RL can't memorize per-user shortcuts.
