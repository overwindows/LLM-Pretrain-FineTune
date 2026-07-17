# SFT — Execution Process (layer1-delta, Thinking 50K, GPT-4o teacher)

How the Layer1-delta SFT model (`Qwen3-4B-Thinking-2507` → distilled on GPT-4o
teacher labels) was actually trained, with the **real, recovered hyperparameters**.
This is the "how we trained it" companion to
[`evaluation_process.md`](evaluation_process.md) (how we benchmarked it).
Everything referenced below lives under `MAIDistillation0623/sft/` and
`MAIDistillation0623/models/`.

> **Base model:** `Qwen3-4B-Thinking-2507` (persisted at `models/base/`).
> **Teacher:** `gpt-4o` (reference labels in the training data).
> **Best checkpoint:** `checkpoint-2260`, eval_loss **0.0833**, persisted at
> `models/sft/layer1_delta_thinking_50k_4o-v1/`.
> **Repro run:** `layer1_delta_thinking_50k_4o-v2-repro` (online wandb project
> `maiprofile-sft`).

---

## Pipeline at a glance

```
  data/splits (v1)                  base model (cosmos)
  train/val.jsonl                   models/base/Qwen3-4B-Thinking-2507
        │                                   │
        └───────────────┬───────────────────┘
                        ▼
       scripts/launch_repro_sft.sh   (renders template → concrete config)
                        │
                        ▼
       accelerate launch (8×A100) → scripts/sft_train.py
         · DeepSpeed ZeRO-3 (configs/deepspeed/ds_config_zero3.json)
         · HF Trainer, loss masked to last assistant message
         · logs metrics → online wandb (project maiprofile-sft)
                        │
        ┌───────────────┴────────────────┐
        ▼                                ▼
  OUTPUT_DIR (local /scratch)      checkpoint-N every 200 steps
        │  rsync (exclude global_step*)
        ▼
  models/sft/<run>/  (cosmos persistent: best merged ckpt)
```

---

## Step 0 — Inputs (data + base model)

### Training data (v1 = the SFT split)
`MAIProfileSFT_50k/data/splits/layer1_delta_thinking_50k_4o/`

| file | lines | size | role |
|---|---|---|---|
| `train.jsonl` | 36 252 | 524 MB | training |
| `val.jsonl`   | 4 532  | 64 MB  | eval (every 200 steps) |
| `test.jsonl`  | 4 531  | 66 MB  | held out (used by eval, not training) |
| `manifest.json` | — | — | provenance + split stats |

- **Source curation:** `curation_data_thinking_layer1_delta_50k.jsonl`
  (`source_sha256 = 3e12f51d…e44aee8c`), split **seed=42**, ratios **80/10/10**.
- **Record schema:** `{"messages": [system, user, assistant], "metadata": {...}}`
  - `system` — Layer1 delta task spec (naming rules, schema).
  - `user`   — `Date / User ID / Demographics / Today's Denoised Signals (JSON array of
    {Date, Source, DetailedSource, Action, intent})`.
  - `assistant` — GPT-4o teacher target: `<think>…</think>` + ` ```json {"interests":[…]} ``` `.
  - `metadata` — `{model: gpt-4o, user_id, date, delta_index, prompt_tokens,
    completion_tokens, temperature 0.2, …}`.
- **Teacher assistant length:** mean 4 985 chars, max 82 518 (manifest).

### Base model
`models/base/Qwen3-4B-Thinking-2507/` — re-downloaded + persisted to cosmos
(config + generation_config + 3 safetensors shards + tokenizer). This is the
exact base the SFT/repro run loads.

---

## Step 1 — Render the config + launch (8×A100)

Driver: `scripts/launch_repro_sft.sh` (portable; resolves repo/cosmos roots from
its own location, so it is **job-id agnostic** — the original v1 config hardcoded
the now-dead AML job id `fc096b74…`).

```bash
PRL=/home/aiscuser/.conda/envs/pipeline-rl/bin
cd MAIDistillation0623/sft
export WANDB_API_KEY=...            # online wandb (never commit; rotate after)
export WANDB_MODE=online
# unset any leaked path envs from prior dry-runs first:
unset BASE_MODEL TRAIN_JSONL VAL_JSONL OUTPUT_DIR DEEPSPEED_CONFIG RUN_NAME
setsid nohup bash scripts/launch_repro_sft.sh > logs/repro_sft.log 2>&1 < /dev/null &
```

What the launcher does:
1. `envsubst` the template `configs/sft/layer1_delta_thinking_50k_4o-v2.repro.yaml.tmpl`
   → concrete `configs/sft/rendered/<run>.yaml` (fills `BASE_MODEL / TRAIN_JSONL /
   VAL_JSONL / OUTPUT_DIR / DEEPSPEED_CONFIG`).
2. Defaults: `BASE_MODEL=models/base/Qwen3-4B-Thinking-2507`,
   `DATA_DIR=…/MAIProfileSFT_50k/data/splits/layer1_delta_thinking_50k_4o`,
   `OUTPUT_DIR=/scratch/local_sft_runs/<run>` (local fast disk),
   `PERSIST_CKPT_DIR=models/sft/<run>` (cosmos).
3. Strips AzureML MPI envs (`RANK/WORLD_SIZE/MASTER_ADDR/…`) so accelerate runs
   single-node 8-GPU DDP; sets NCCL/CUDA stability envs;
   `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`.
4. `accelerate launch --config_file configs/accelerate/accelerate_ds3.yaml
   --num_processes 8 scripts/sft_train.py --config <rendered>`.
5. On finish: `rsync -a --exclude='global_step*' OUTPUT_DIR/ PERSIST_CKPT_DIR/`
   (keeps the merged HF checkpoints, drops the bulky DeepSpeed shards).

---

## Step 2 — Training (`scripts/sft_train.py`)

HF `Trainer` + DeepSpeed ZeRO-3, full-parameter (no LoRA/PEFT).

### Hyperparameters (frozen — match the recovered original run)

| param | value | note |
|---|---|---|
| max_seq_len | **14336** | 99.57% coverage on the Qwen tokenizer |
| per_device_train_batch_size | 1 | |
| gradient_accumulation_steps | 4 | 8 GPU × 1 × 4 = **effective batch 32** |
| num_train_epochs | **2** | confirmed via best ckpt `trainer_state.json` (epoch=2.0, step=2260) |
| learning_rate | 5.0e-6 | |
| lr_scheduler_type | cosine | warmup_ratio 0.03 |
| weight_decay | 0.0 | |
| seed | 42 | |
| mask_think_in_loss | **false** | think tokens DO contribute to loss |
| attn_implementation | sdpa | flash-attn ABI mismatch on this box; SDPA works |
| gradient_checkpointing | true | |
| eval / save | every **200** steps | save_total_limit 3 |
| logging_steps | 10 | |
| resume_from_checkpoint | auto | |
| **total steps** | **2260** | ~1130 steps/epoch × 2 |

### Loss masking
`tokenize_record()` renders the full chat with the Qwen chat template, then sets
`labels = -100` everywhere **except the last assistant message span** (the teacher
target). Overlong records are **left-truncated** to `max_seq_len` keeping the
assistant target intact; if the target itself gets cut, the record is skipped.
With `mask_think_in_loss=false` the `<think>…</think>` block trains as part of the
target.

> Build stats (repro run): train kept **36 142** (skip_truncated_target 110,
> overlong 66), val kept **4 522** (skip 10, overlong 8). Tokenization is a
> single-thread python loop redundantly run on all 8 ranks → ~10 min CPU-bound
> with GPUs idle before the first step (expected, not a hang).

### wandb logging (online)
`sft_train.py` enables `report_to=["wandb"]` when `wandb_project` is set (and not
`--no-wandb/--smoke-test/--debug`). It records:
- **metrics** every 10 steps: `loss`, `grad_norm`, `learning_rate`, `epoch`;
  eval `eval_loss` every 200 steps;
- **config**: all hyperparams above (HF Trainer auto-uploads);
- **system metrics**: 8× GPU util/mem/power/temp, CPU, disk, network (wandb auto).
- The **training data** is separately versioned as a wandb **dataset artifact**
  (see Step 4).

---

## Step 3 — Checkpoints & persistence

- Trainer writes `checkpoint-200, 400, …` to `OUTPUT_DIR` (local
  `/scratch/local_sft_runs/<run>`), keeping the latest 3 + the best by eval_loss.
- The **best** checkpoint (original run: `checkpoint-2260`, eval_loss 0.0833) is
  rsynced to cosmos `models/sft/<run>/` as a flat merged HF checkpoint:
  `model.safetensors` (8.0 GB), `config.json`, `generation_config.json`,
  tokenizer files, `chat_template.jinja`, `training_args.bin`, `training_summary.json`.
  The DeepSpeed `global_step*` shards are **excluded** (not needed to serve/eval).

---

## Step 4 — Training data → online wandb artifact (`scripts/log_data_to_wandb.py`)

Stores the exact training data durably + versioned alongside the run:

```bash
$PRL/python scripts/log_data_to_wandb.py \
  --data-dir <…>/layer1_delta_thinking_50k_4o \
  --config   configs/sft/rendered/<run>.yaml \
  --project  maiprofile-sft \
  --artifact layer1_delta_thinking_50k_4o-data
```

- Logs `train/val/test.jsonl` + `manifest.json` + the rendered SFT config as a
  wandb **`dataset`** artifact, each file tagged with its **sha256**
  (train `df821863…`, val `ae384d5e…`, test `1000bdb3…`) for integrity/versioning.
- Result: project `maiprofile-sft`, artifact `layer1_delta_thinking_50k_4o-data`.

---

## Files in this repo (all required to reproduce)

| path | role |
|---|---|
| `scripts/sft_train.py` | HF Trainer + DeepSpeed ZeRO-3 trainer; loss masking; wandb wiring |
| `scripts/launch_repro_sft.sh` | **portable** launcher: render template → 8-GPU accelerate → rsync to cosmos |
| `scripts/launch_layer1_delta_thinking_50k_sft.sh` | original launcher (references the dead job id; kept for provenance) |
| `scripts/log_data_to_wandb.py` | upload train/val/test + manifest + config as a versioned wandb dataset artifact |
| `configs/sft/layer1_delta_thinking_50k_4o-v2.repro.yaml.tmpl` | **portable** config template (paths injected at launch) |
| `configs/sft/layer1_delta_thinking_50k_4o-v1.yaml` | original recovered config (dead-job paths; provenance) |
| `configs/sft/_ref_layer1_delta.yaml` | reference config |
| `configs/sft/rendered/<run>.yaml` | auto-generated concrete config (by launcher) |
| `configs/accelerate/accelerate_ds3.yaml` | accelerate: 8-proc MULTI_GPU bf16, no hardcoded paths |
| `configs/deepspeed/ds_config_zero3.json` | DeepSpeed ZeRO-3, no offload, auto batch/grad |
| `env/pipeline-rl-pip-freeze.txt` | exact training deps (torch 2.6.0, transformers 5.5.4, accelerate 1.8.1, deepspeed 0.15.4, datasets 4.8.4, wandb 0.27.0) |

---

## Environment
- conda `pipeline-rl` at `/home/aiscuser/.conda/envs/pipeline-rl/bin` (invoked by
  absolute path; `conda activate` is broken on this box). Training deps were
  installed to match `env/pipeline-rl-pip-freeze.txt`.
- 8× A100-80GB, single node. bf16. DeepSpeed ZeRO-3, no offload.

## Secrets
- `WANDB_API_KEY` read from env (or `/home/aiscuser/.secrets/maiprofile_sft.env`
  if present). **Never committed. Rotate after any session where it was exported.**

## Gotchas (learned the hard way)
1. `accelerate`/`deepspeed`/`datasets`/`wandb` were **missing** from `pipeline-rl`
   (it was an eval-only env) — install to the freeze versions before launching.
2. `run_in_terminal` shares one persistent bash → dry-run `export OUTPUT_DIR=…`
   leaks into the next launch. **`unset` the path envs before launching.**
3. `du` reports 0 for files on the cosmos mount — use `ls -la` to verify sizes.
4. The original v1 config/launcher reference the dead job id `fc096b74…`; use the
   portable `*.repro.yaml.tmpl` + `launch_repro_sft.sh` instead.
