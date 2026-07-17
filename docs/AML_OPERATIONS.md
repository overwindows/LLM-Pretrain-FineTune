# AML Operations Playbook (read this first)

Hard-won operational knowledge for running SFT / RL **inside an AzureML (AML)
interactive job** on this kind of node. It is **task-agnostic** — read it before
you touch any recipe in [`process/`](process/) or
[`REPRODUCE_LAYER1_DELTA.md`](REPRODUCE_LAYER1_DELTA.md).

> **TL;DR for a coding agent dropped into a fresh AML job:**
> 1. The node can be **reclaimed at any time**. Only the **cosmos share** survives.
>    `/scratch` and `/home/aiscuser` are **fast but per-node and ephemeral**.
> 2. The cosmos mount path contains a **per-job id** that is different for every
>    person/job — **never hardcode it**, discover it (§2).
> 3. Train fast on local `/scratch`, then **persist good checkpoints to cosmos**
>    (HF weights only, no optimizer shards) (§3).
> 4. Metrics live in `trainer_state.json` + wandb (SFT) / wandb + stats streams
>    (RL). Know where and how to read them without a browser (§6).
> 5. Secrets come from **env only**, never git. Rotate after exposure (§7).

---

## 1. The machine model (mental model first)

| Fact | Consequence |
|---|---|
| Node = **8×A100-80GB**; a job may span **2 nodes** | Single-node is simplest; multi-node needs env staging on *each* node (§5). |
| `/scratch/...` and `/home/aiscuser` are **local to each node**, fast NVMe | Great for checkpoints/wandb *during* a run; **node-1 cannot see node-0's `/scratch`**. Move data over `ssh`/`rsync`. |
| **cosmos blob share** is the only **shared + durable** storage | Slow (blobfuse). Read inputs from it once, write outputs to local, sync *results* back. |
| AML can **preempt / reclaim** the node with little warning | Anything only on `/scratch` or `/home` at that moment is **gone**. Checkpoint often + persist to cosmos. |

`/scratch` and `/home` being per-node is why RL here runs **single-node** and why
checkpoint evals were done on the *other* node via `ssh`.

---

## 2. The cosmos path trick (the #1 gotcha)

The cosmos share is mounted at a path like:

```
/scratch/azureml/cr/j/<JOBID>/cap/data-capability/wd/INPUT_<DATASTORE>/...
                     ^^^^^^^                          ^^^^^^^^^^^^^^^^
                  per-job, changes                  datastore alias, can change
```

- `<JOBID>` (e.g. `65fcf9508e03476381b75ace1f02fb73`) is **unique per AML job /
  per person** — it is *not* stable across sessions or users.
- `INPUT_<DATASTORE>` (e.g. `INPUT_msndni`) is the input-datastore alias and can
  also differ between people.

**Never hardcode either.** Discover the mount at the top of every script/session:

```bash
# Robust: find the (single) cosmos INPUT mount for THIS job, no matter the id
COSMOS_INPUT=$(ls -d /scratch/azureml/cr/j/*/cap/data-capability/wd/INPUT_* 2>/dev/null | head -1)
[ -n "$COSMOS_INPUT" ] || { echo "cosmos mount not found — is the datastore attached?"; exit 1; }

# The job id, if you need it, is derivable from $PWD (you run inside the same tree)
JOBID=$(pwd | sed -n 's#.*/cr/j/\([0-9a-f]*\)/.*#\1#p')

# Our per-user root (adjust the trailing user path to yours):
export Y="$COSMOS_INPUT/shares/users/yuhangbai"
echo "cosmos user root: $Y"
```

Then use `$Y` / `$COSMOS_INPUT` everywhere. If you see a literal
`.../j/65fcf95.../...` in any script, that is a **bug** — it will break for the
next person. (The `process/` recipe docs quote absolute paths for provenance; treat
those as *examples*, not literals to copy.)

---

## 3. Anti-preemption storage discipline

**Rule: compute-local, persist-durable.**

- **Train to local `/scratch`** (checkpoints, wandb, logs) — it is fast and does
  not thrash blobfuse. e.g. `OUTPUT_DIR=/scratch/local_sft_runs/<run>`.
- **Read inputs from cosmos** once (base model, data splits under `$Y/...`).
- **Persist only what you can't regenerate, to cosmos**, as soon as it's good:
  - HF checkpoint = `*.safetensors` + `config.json` + tokenizer files **only**.
  - **Exclude** DeepSpeed optimizer shards / `global_step*/` (huge, and useless
    for inference/eval). RL's persist step and SFT's `PERSIST_CKPT_DIR` already do
    a filtered `rsync`.
- **Checkpoint frequently** so a reclaim costs you at most a few hundred steps:
  - SFT: `save_steps` + `save_total_limit` (verified: `save_total_limit=3` keeps
    only the newest 3 on the small local disk).
  - RL: `save_checkpoint_steps` (50 in the layer1 runs); keep intermediates you
    might eval, but sync the target one out promptly.
- **Cross-node moves** use `ssh`/`rsync`, e.g.:
  ```bash
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no node-1 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader'
  rsync -a /scratch/<run>/checkpoint-2260/ node-1:/scratch/<run>/checkpoint-2260/
  ```

If a run dies mid-way: the newest local checkpoint under `OUTPUT_DIR` is your
resume point (SFT `--resume_from_checkpoint`, RL `force_restart: false` resumes
from `output_dir`).

---

## 4. Supported input formats

Keep data **simple and explicit** — one JSONL record per line.

**SFT** — chat format, consumed by `sft/sft_train.py`:
```json
{"messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "...(optional few-shot demo)..."},
  {"role": "assistant", "content": "...(demo answer)..."},
  {"role": "user", "content": "<real input>"},
  {"role": "assistant", "content": "<target completion>"}
]}
```
- The **last message must be `role: assistant`** (it's the training target).
- Everything before the final assistant turn is **loss-masked** (system + few-shot
  + real user prompt). `mask_think_in_loss` optionally also masks the
  `<think>…</think>` span inside the target.
- Files: `train.jsonl` / `val.jsonl` / `test.jsonl` under a split dir
  (`data/splits/<task>/`). Rendered via the tokenizer's `apply_chat_template`.

**RL** — one prompt per line + the reference label carried in-record:
- prompt (the user input) + `reference` (e.g. the gpt-4o answer) used by the
  rule reward and the judge. Length-filter to `< max_model_len` **before** the run
  (a single over-length prompt can stall the whole job — see §8).

Starting a new task? The cheapest path is to emit these exact shapes and reuse the
existing trainer/loop unchanged.

---

## 5. Reward / metric design — start simple, add later

Recommended ramp (this is how layer1 actually evolved, and what to imitate):

1. **Cheap objective metrics first (M1).** Deterministic, no API cost, run on every
   checkpoint: parse-rate, schema-valid rate, length ratio, exact/rule matches.
   These alone tell you if the model is *degenerating* — always keep them on.
2. **Rule-based reward before any LLM judge.** A soft-gated composite is enough to
   start:
   ```
   r = gate · ( parse_ok · structural_score )
   gate = 0.1 when parse/schema fails (soft floor, NOT hard 0 — see §8 lesson 1)
   ```
   Rules give a dense, free, deterministic signal and expose parser bugs early.
3. **Add the LLM judge (M2) only when rules plateau**, and make it **opt-in**
   (`LAYER1_JUDGE_ENABLED=1`) with graceful fallback to rule-only if the judge is
   down. Fuse: `r = gate · (w_rule·r_rule + w_llm·r_llm)`.
4. **Change one knob at a time.** The layer1 lineage (stage4 → wrule01 → recall)
   moved exactly one reward weight per run for clean attribution. Do the same.

Design tips baked in from experience:
- **Keep the gate soft.** A hard `0` on parse failure looks like "the model is
  bad" when it's really "the parser is wrong" — you can burn hundreds of steps on
  noise (this literally happened, §8).
- **Reward code should be a small pure function** (parse → gate → score → fuse)
  living next to the task, easy to unit-test on a handful of records offline before
  spending GPU.
- **The eval judge rubric and the RL reward judge can share a rubric** — one source
  of truth for "what good looks like."

---

## 6. Where training signals live & how to check them (no browser needed)

### SFT (`sft/sft_train.py`, HF `Trainer`)
| Signal | Where | How to read |
|---|---|---|
| train loss / lr / grad_norm | stdout every `logging_steps` (5) → your log file | `grep -E "'loss'|'eval_loss'" sft/logs/<run>.log \| tail` |
| **full loss + eval history** | `trainer_state.json` **inside each checkpoint dir** | `python -c "import json;s=json.load(open('$OUTPUT_DIR/checkpoint-2260/trainer_state.json'));print(s['best_metric']); [print(h) for h in s['log_history'][-10:]]"` |
| eval_loss | logged every `eval_steps`; best tracked as `best_metric` | as above |
| wandb (offline default here) | `sft/wandb/offline-run-*/` | `wandb sync sft/wandb/offline-run-*` later, or read `sft/wandb/latest-run/files/wandb-summary.json` |

`trainer_state.json` is the durable, browser-free source of truth — it travels with
the checkpoint to cosmos.

### RL (`pipelinerl`, async GRPO)
| Signal | Where | How to read |
|---|---|---|
| reward, entropy, policy_loss, grad_norm, KL, always/never_success | **wandb** project `pipeline-rl` (config `wandb.use_wandb`) | offline → `<output_dir>/wandb/…/files/wandb-summary.json`; or grep the launcher stdout log |
| per-group verifier / rollout tables | wandb tables (`tables/verifier_last_k`) | inspect in wandb, or the stats stream shards |
| raw stats stream | `<output_dir>/…/*.jsonl` shard streams (`streams.py`) | `tail`/`jq` the shard jsonl |
| checkpoints | `<output_dir>/finetune/intermediate/<step>/`, final `finetune/current/` | eval by serving that dir |

**What healthy looks like (watch together):** reward trending up, `grad_norm` bounded
(clip 0.3), `policy_loss ≈ 0` (wide PPO clip is expected here), `entropy` — a *climb*
is Stage-4-faithful, not a bug (§8 lesson 3). Watch `never_success` + `entropy`
jointly; if reward is flat near a tiny constant, suspect the **parser/gate**, not the
optimizer (§8 lesson 1).

Quick offline pull of a metric from wandb summary:
```bash
python -c "import json,glob;f=sorted(glob.glob('<output_dir>/wandb/*/files/wandb-summary.json'))[-1];d=json.load(open(f));print({k:d[k] for k in d if any(s in k for s in ('reward','entropy','grad_norm','loss','success'))})"
```

---

## 7. Secrets on AML

- **Env only, never git.** Clients read `AZURE_OPENAI_KEY` / `AOAI_KEY` /
  `WANDB_API_KEY` etc. from the environment (or `DefaultAzureCredential`).
- Persist a judge key to a **mode-600** file for reuse within the job, and source it:
  ```bash
  ( umask 077; printf '%s' "$AZURE_OPENAI_KEY" > ~/.azure_judge_key )
  export AZURE_OPENAI_KEY=$(cat ~/.azure_judge_key)
  ```
- **Rotate any key that got echoed** into a terminal, log, or notebook — terminal
  history and job logs are captured. Don't paste keys on a command line that logs.
- Endpoint **hostnames** are override-able env defaults in code, not secrets — swap
  via env for a different tenant.

---

## 8. Hard-won gotchas (the ones that cost real GPU-hours)

1. **A wrong parser silently nullifies the reward.** Qwen3-Thinking ends its JSON
   fence with ```` ``` ````**+`<|im_end|>`**; a naïve `endswith("```")` strip fails
   every rollout → soft-gate floors the reward → the judge never fires → reward sits
   flat near a constant. **190 steps trained on pure noise** before it was caught.
   → Always parse via the task's `parse_completion`, unit-test parse-rate on real
   samples **before** launching, and alarm if parse_ok is near 0.
2. **A single over-length prompt can stall the whole job.** vLLM returns a
   deterministic HTTP 400 for `prompt+completion > max_model_len`; a naïve retry
   loop re-queues it forever, the actor stops, and finetune starves into the NCCL
   watchdog → SIGABRT. → **Length-filter data first**, treat deterministic client
   errors (400/413/"maximum context length") as *skip-this-rollout*, and raise the
   NCCL timeout.
3. **Entropy climbing is not automatically a bug.** For the Stage-4 recipe the
   entropy rises by design; "fixing" it can mean you've stopped reproducing the
   thing you meant to reproduce. Know your recipe's expected shape before you
   "correct" it.
4. **`never_success` is a joint property**, not a model constant — it depends on
   (policy state × temperature × rollouts/group × data subset × success threshold).
   An online rate does not have to match an offline re-score of a converged ckpt.
   Don't chase a "discrepancy" that is just a different measurement condition.
5. **Strip AML's multi-node alloc for single-node RL.** AML sets a 2-node
   `WORLD_SIZE`; launch single-node with `WORLD_SIZE=1 RANK=0 NODE_RANK=0` and pin
   GPUs (`PIPELINERL_GPUS=...`) or the process group hangs.
6. **Don't upgrade the pinned web stack in the RL env** (`fastapi==0.115.12 /
   starlette==0.41.3 / sse-starlette==2.1.3`) — newer versions break the vLLM actor
   `/health` route via the prometheus instrumentator.
7. **Run artifacts don't belong in git.** Rendered configs, checkpoints, wandb
   dirs, and `*.jsonl` data are `.gitignore`'d; if `git status` shows a
   `rendered/` yaml as modified after a run, that's the artifact leaking — ignore
   it, don't commit it.

---

## 9. Fresh-job checklist (copy/paste)

```bash
# 1. Locate durable storage (never hardcode the job id)
COSMOS_INPUT=$(ls -d /scratch/azureml/cr/j/*/cap/data-capability/wd/INPUT_* | head -1)
export Y="$COSMOS_INPUT/shares/users/yuhangbai"          # adjust to your user

# 2. Point compute at fast local disk
export OUTPUT_DIR=/scratch/local_sft_runs/<run>          # ephemeral, fast
export PERSIST_DIR="$Y/<project>/models/<run>"           # durable, sync here

# 3. Secrets from env (rotate if ever echoed)
export AZURE_OPENAI_KEY=$(cat ~/.azure_judge_key)

# 4. Check GPUs (this node and the other)
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
ssh -o BatchMode=yes -o StrictHostKeyChecking=no node-1 nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader

# 5. Train local → 6. persist good ckpts to $PERSIST_DIR (safetensors only) → 7. eval
```

See [`REPRODUCE_LAYER1_DELTA.md`](REPRODUCE_LAYER1_DELTA.md) for the exact per-stage
commands and [`process/`](process/) for the per-stage recipes.
