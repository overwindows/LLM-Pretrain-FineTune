# SFT — Execution Process (layer1-delta, Thinking 50K, GPT-4o teacher)

## Qwen3.5 text-task compatibility (separate from the historical run below)

The dense `Qwen/Qwen3.5-4B` checkpoint uses the full
`Qwen3_5ForConditionalGeneration` class and `qwen3_5` config with nested text
and vision configs. The trainer now selects that concrete native class,
freezes its unused vision tower for text-only DDP, disables caching, and retains
the full model configuration/weights on save. Other existing model families
still use `AutoModelForCausalLM`. Some modern Transformers releases also map
Qwen3.5 through that Auto class; the problem is **not** universal Auto-class
incompatibility. The old Transformers 4.51.3 setup is incompatible with Qwen3.5.
Text SFT accepts string message content only; it does not process images/videos.

Use an isolated Python **3.10+** environment rather than altering the historical
SFT/RL/vLLM installation:

```bash
# PIP_CONSTRAINT may point at an approved exact CUDA/torch stack lock.
# Resolve and validate that GPU lock before any actual training submission.
bash scripts/setup_env.sh --profile qwen35 --python /path/to/python3 \
  --env-dir /new/path/qwen35-sft
```

This profile refuses existing environments, uses
`env/qwen35-sft-requirements.txt`, checks dependency consistency, and records
`resolved-requirements.txt` inside the environment. It never applies Python
version-gate patches. The default `legacy` profile remains unchanged.
The Qwen3.5 constraints pin Transformers **5.19.0** but do **not** claim a
validated CUDA/driver/DeepSpeed stack; inherited `PIP_CONSTRAINT` is respected.
The historical `env/pipeline-rl-pip-freeze.txt` is not a Qwen3.5 runtime lock.

CPU regression evidence: Python 3.12.10, torch 2.14.1 (CUDA unavailable),
Transformers 5.19.0, accelerate 1.15.0, datasets 5.1.0, tokenizers 0.23.2,
safetensors 0.8.0, PyYAML 6.0.3. Tests include a tiny randomly initialized
Qwen3.5 model with both linear and full attention, an actual optimizer update,
full HF save/reload and equal reloaded logits. They do not download model weights.
An additional local-cache-only tokenizer test uses official HF revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`; it skips if metadata is absent.
The existing shared CPU host has unrelated OpenTelemetry dependency conflicts
reported by `pip check`; it is not a clean environment lock. The new Linux venv
bootstrap itself still requires runtime validation.

Render `sft/configs/sft/qwen35_text.yaml.tmpl` to a YAML with real paths. For
an orchestration-approved bounded GPU smoke, use a fresh output directory,
`max_seq_len: 1024`, batch/accumulation 1, no DeepSpeed and no resume:

```bash
unset PYTHONPATH PYTHONUSERBASE
export PYTHONNOUSERSITE=1
/new/path/qwen35-sft/bin/python sft/sft_train.py \
  --config /path/to/rendered-smoke.yaml --max-steps 3 --no-wandb
```

`base_model` should be the fully staged checkpoint directory with immutable
source provenance. For direct Hub loading, optional YAML `model_revision`
is passed to config, tokenizer, and weights. Always set
`enable_thinking: false` for the current nonreasoning gold targets and use the
same chat-template flag for evaluation/inference. Qwen3.5 requires an explicit
boolean setting. A generation prefix that does not exactly match the full
training rendering is an error, not a guessed loss boundary.

Assistant-only labels are computed **before** truncation, then shifted with the
left-truncated input. Targets that cannot fit intact together with a preceding
context token are skipped with a warning. Think masking never falls back to
training a span it was asked to mask. Token accuracy now uses causal shifting.
The warmup ratio is adapted to the installed Trainer API (`warmup_steps` ratios
in modern releases, `warmup_ratio` in legacy releases); modern ratios must be
less than 1 to avoid treating 1 as one absolute step.

Task weighting is a **preparation contract**: the caller materializes positive
integer frequency replication (currently HPF 2, GEM 1, MAI Profile 1) and removes
`sample_weight`. The trainer preserves those repeated rows with uniform
sampling; it does not also apply a weighted loss. Any remaining non-unit,
nonfinite or invalid row weight fails before dataset column projection, even
outside a debug subsample. Quality stays evaluation-only under the caller's
split selection policy. Malformed JSON and unsafe template boundaries fail
explicitly rather than silently reducing the corpus.

Run CPU regressions with:

```bash
python -m unittest discover -s sft/tests -v
```

The final HF model and tokenizer are saved at `output_dir`, including the
tokenizer chat template. Final `trainer_state.json` records step/log history;
`training_summary.json` records finite training loss, concrete model class/type,
thinking policy and source revision. Nonfinite training loss fails before final
export. Load the export using
`Qwen3_5ForConditionalGeneration.from_pretrained(output_dir)` and
`AutoTokenizer.from_pretrained(output_dir)` for **text-only** inference.
Image/video processing is outside this trainer's contract; it does not export
an `AutoProcessor`. Serving also needs an architecture-compatible runtime and
the same nonthinking flag. The historical vLLM 0.8.5 wrapper/shim is **not**
proof of Qwen3.5 support; do not mix it into this training environment or mark
GPU/serving readiness true from these CPU tests.

---

### Opt-in 8192-token ZeRO-3 bounded validation

The next milestone is a **single-rank bounded validation**, not full training
readiness. Retain the tested native-HF stack and install the exact candidate
`deepspeed==0.16.9` after torch is installed, with `DS_BUILD_OPS=0` and
`--no-build-isolation`. Keep the root job's exact `PIP_CONSTRAINT`; run `pip check`
and persist the resolved freeze before model loading. No DeepSpeed package is
added to the default legacy or native-HF environment.

Candidate evidence (not proof of this model's GPU runtime compatibility):
[DeepSpeed 0.16.9's upstream CUDA CI](https://github.com/deepspeedai/DeepSpeed/blob/v0.16.9/.github/workflows/nv-torch-latest-v100.yml)
explicitly tests torch 2.6 / CUDA 12.4. Its
[package requirements](https://github.com/deepspeedai/DeepSpeed/blob/v0.16.9/requirements/requirements.txt)
allow torch 2.6 and pydantic >=2. The PyPI sdist SHA256 is
`88dc08986cd321047c37c2d0f1edd6faac498a1b52eb486d559164362c6b9011`.
The pinned source exposes `DeepSpeedEngine.zero_optimization_stage`,
`set_gradient_accumulation_boundary`, batch-size accessors and `global_steps`,
which are used by Accelerate 1.15 and the runtime proof. Its checkpoint loader
already specifies `weights_only=False`; no global torch-load patch is applied
to this version. This session's CPU/mock tests do not establish Linux dependency
resolution, CUDA kernels or 8192-token memory safety.

`TrainingArguments` is now constructed **before** `from_pretrained`, retaining
the HF DeepSpeed config weak reference used for `deepspeed.zero.Init`.
HF 5.19 also resolves automatic ZeRO buckets from Qwen3.5's nested
`text_config.hidden_size`. The full conditional-generation model is preserved,
with its unused visual tower frozen before the optimizer is constructed.

Use the orchestration-owned stage-3 JSON (BF16 and batch sizes automatic,
gather 16-bit weights on save, no offload/optimizer/scheduler blocks). Add these
keys to the rendered trainer YAML:

```yaml
max_seq_len: 8192
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
enable_thinking: false
deepspeed_config: /path/to/zero3_bf16.json
resume_from_checkpoint: null
validation_fixed_order: true
bounded_validation:
  expected_steps: 8
  warmup_steps: 2
  expected_zero_stage: 3
  expected_world_size: 1
```

Launch with `torchrun --standalone --nproc_per_node=1 sft/sft_train.py
--config <yaml> --max-steps 8 --no-wandb` from the isolated environment.
The exact step cap must match `expected_steps` (at most 10); no debug/smoke flags
or resume are allowed. The outer job owns the GPU count and hard timeout.
`validation_fixed_order` is rejected outside bounded validation and selects
`SequentialSampler`, preserving the caller's repeated task-frequency rows and
planned near-cap coverage. Default training still uses the ordinary Trainer
and sampler.

`bounded_validation.json` is atomically updated at training start and after
each optimizer update, and embedded in the final `training_summary.json`.
It records actual engine class, ZeRO stage, initialized distributed world size,
batch/accumulation settings, device/package versions, and pre-load ZeRO-3
configuration/partitioned-parameter proof. Inactive/mismatched DeepSpeed,
skipped updates, partial accumulation groups, nonfinite microbatch losses or
an incorrect final step count fail explicitly. A failure during training leaves
a failed-status report rather than success-shaped evidence.
Runtime proof also requires `bf16_enabled: true` from the actual engine, records
its optimizer wrapper class and optimizer name, and snapshots
`resolved_hf_deepspeed_config` from the public HF/Accelerate integration after
automatic bucket/batch resolution. That snapshot represents the finalized
configuration passed to DeepSpeed, not all of DeepSpeed's internal defaults.

Each optimizer update has CUDA synchronization at both ends, peak allocated and
reserved bytes reset at its start, and actual microbatch input lengths,
attended/padded tokens, causally supervised tokens and engine update count.
Microbatch losses are the values returned by Trainer (not an independently
normalized per-example metric). Preparation separately counts kept/skipped rows,
skip reasons and valid left-context truncation. Dataset diagnostics do not
include per-example content. Warmup updates are excluded from the aggregate
attended-token, supervised-token and example throughput. These update intervals
exclude dataset preparation, batch prefetch, logging, evaluation and export;
`training_wall_seconds` additionally includes inter-update overhead. Do not
extrapolate a full-run cost as though those excluded stages or sufficient
evaluation had been measured. GPU validation and long-generation evaluation
remain the orchestration session's responsibility.

---

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
