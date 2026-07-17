# RL Training — Execution Process (layer1-delta, GRPO on QED-Nano / pipelinerl)

How the Layer1-delta **RL** models were actually trained: the async GRPO loop,
the verifier + online judge + fused reward, the framework changes we made, and
the three runs we shipped (stage-4 repro → lowered-`w_rule` → recall reward).
This is the "how we RL'd it" companion to
[`sft_process.md`](sft_process.md) (the RL init) and
[`rl_data_cleaning_process.md`](rl_data_cleaning_process.md) (how the train set
was filtered). The **what-we-found** narrative lives in the experiment reports
`report/260625_rl_repro_experiment.md` (+ wrule01) and
`report/260628_rl_recall_reward.md`; the final numbers live in
`eval/COMPARISON.md`. This doc is the mechanism + recipe + run lineage, and does
**not** repeat those.

> **Framework:** QED-Nano `pipelinerl` (Apache-2.0, `github.com/CMU-AIRe/QED-Nano`
> @ `02a4699`), run **from source in-repo** with our uncommitted changes — frozen
> as a patch + bundle under `rl_repro_bundle/` (see `README_RL_REPRO.md` there).
> **RL init + KL reference:** the SFT-50K checkpoint (`models/sft/layer1_delta_thinking_50k_4o-v1/`),
> tokenizer patched for transformers 4.51.1 (`extra_special_tokens` list→`{}`).
> **Train data:** `rl_data/v2/train.lenfiltered.jsonl` (Pass-A cleaned, 35 615).
> **Judge / teacher:** judge = `gpt-5.1` (Azure `msncompanioneu2`); reference
> labels = `gpt-4o` (carried in the data as `reference`).
> **Shipped checkpoints (cosmos):** `models/rl/layer1_delta_rl_stage4repro_wrule05_ckpt500`,
> `…_wrule01_ckpt500`; recall run `MAIProfileSFT_runs/qwen3-4b-l1d-rl-recall-step{500,900,1000}`.

---

## Pipeline at a glance

```
  RL init = SFT-50K ckpt (patched)          train data = rl_data/v2/train.lenfiltered.jsonl
  models/sft/…-v1  ──────────────┐              (Pass-A cleaned, 35 615)
                                 ▼                        │
                    run.slurm → python -m pipelinerl.launch --config-name=layer1_rl
                                 │  (Hydra config; single node, 7 GPUs)
      ┌──────────────────────────┼───────────────────────────────┐
      ▼                          ▼                                ▼
  3× actor vLLM            1× preprocessor (KL ref vLLM)     3× finetune (DeepSpeed Z3)
  generate 8 rollouts/     computes ref logprobs +           GRPO update (policy_loss=ppo,
  prompt (temp 0.8),       group-relative advantages         group-relative advantage,
  score each in-process:                                     kl_coef 0.02, entropy 1e-4)
   parser → gate →                                                   │
   r_rule (rules) +                                          save every 50 steps →
   r_llm (gpt-5.1 judge)                                     results/<run>/finetune/
      │                                                             │
      └───────── fused reward r = gate·(w_rule·r_rule + w_llm·r_llm) ┘
                                 │
                  best/target ckpt → rsync to cosmos (safetensors, no optimizer shards)
```

GPU0 is reserved (held by another process); the world runs on GPU1–7.

---

## Step 0 — Inputs

| input | value |
|---|---|
| RL init + KL ref | `models/sft/layer1_delta_thinking_50k_4o-v1/` → local patched copy `/home/aiscuser/models/layer1_sft50k_patched` (tokenizer_config `extra_special_tokens` list→`{}` for transformers 4.51.1) |
| train data | `rl_data/v2/train.lenfiltered.jsonl` — Pass-A cleaned 35 625, minus 10 records > 11 808 Qwen-tok (see §"crash") = **35 615** |
| smoke/eval set | `rl/data_cleaning/_smoke/train.lenfiltered.jsonl` (193; eval effectively disabled at run time) |
| judge | `gpt-5.1` @ `https://msncompanioneu2.cognitiveservices.azure.com/` (api 2024-12-01-preview), direct Azure call, `LAYER1_JUDGE_ENABLED=1` |
| recall candidates (recall run only) | pre-generated per-user grounded cache, 35 615 users (see §"recall run") |

Data provenance & the SFT→RL filtering chain (v1→v2→v3) are in
`rl_data_cleaning_process.md`; not repeated here.

---

## Step 1 — The async GRPO loop (three roles)

`pipelinerl` splits the node into three co-running roles (a "world map"); this run
uses `actor_fraction=3 / preprocessor_fraction=1 / finetune_fraction=3` on 7 GPUs:

1. **Actor (3 vLLM)** — samples `attempts=8` rollouts per prompt (temp 0.8,
   max_tokens 8192, `llm_max_rollouts=16` concurrent), then scores each rollout
   **in-process** (no HTTP verifier) via `domains/layer1`: parse → gate → r_rule
   → r_llm → fuse. The rollout policy is `pipelinerl.domains.layer1.generate_layer1_rollout`.
2. **Preprocessor (1 vLLM = the KL reference)** — serves the frozen SFT init to
   compute reference logprobs, then builds group-relative advantages
   `A_i = r_i − μ_group` (`use_advantages=true`, `divide_advantage_by_std=false`).
3. **Finetune (3 GPU, DeepSpeed ZeRO-3 bf16)** — the GRPO update.

Because scoring happens inside the actor and the judge is a direct Azure call,
there is **no separate grader vLLM** (`llm_grader.local=false` short-circuits it).

### GRPO hyperparameters (from the resolved `exp_config.yaml`)

| param | value | note |
|---|---|---|
| algorithm | **GRPO** | `policy_loss=ppo` + group-relative advantage, **no value critic** |
| attempts (group size) | **8** | 8 rollouts/prompt; advantage is within-group |
| sampling | temp **0.8**, max_tokens **8192** | `llm_max_rollouts=16` concurrent |
| kl_coef / final_kl_coef | **0.02** / 0.02 | ref = SFT init; `reward_minus_kl_coef=0.0` |
| entropy_bonus | **1e-4** | positive term |
| epsilon (PPO clip) | 4 | so wide it almost never clips (confirmed: `policy_loss≈0`) |
| clamp_log_ratio_ref_new_value | 5 | |
| learning_rate | **1e-6** | cosine, `num_warmup_steps=10` |
| grad_clip | 0.3 | `weight_decay=0.01` |
| train_batch_size / grad_accum | 4 / **16–18** | (layer1_rl.yaml=16; recall exp_config=18) |
| seq_length | **20000** | `seq_packing=true` |
| max_train_steps | **500** (stage4/wrule01) / **1000** (recall) | `save_checkpoint_steps=50`, keep intermediates |
| attn (finetune) | flash_attention_2 | (SFT used SDPA; the RL env has a working flash-attn) |
| seed | 42 | actor samples with replacement (`random_iter`) → identical sequence across runs at same seed |

---

## Step 2 — The reward (verifier + online judge, fused)

All reward code is the **in-repo** domain plugin
`pipelinerl/domains/layer1/{reward,rollouts,judge,parser}.py` (frozen in
`rl_repro_bundle/`). Config is the `reward:` block of `conf/layer1_rl.yaml`,
consumed by `reward.build_config`.

### Locked formula

```
r      = gate · ( w_rule · r_rule + w_llm · r_llm )

gate   = parse+schema multiplier; soft floor 0.1 when parse/schema fails
r_rule = 0.5 · anti_collapse[2b, teacher-relative, graded]
       + 0.4 · anti_hallucination  (= fidelity proxy)
       + 0.1 · fidelity_score      (coverage-floored, floor 0.3)
r_llm  = ( w_u·utility + w_p·precision [+ w_rec·recall] ) / Σ(active w)
         utility, precision = gpt-5.1 M2 interest judge, normalized /10
         recall              = per-rollout interest coverage (opt-in, see recall run)
```

- **Parser (`parser.py`)** — strips `<think>…</think>` and the ```` ``` ```` code
  fence, `json.loads` with truncation repair, accepts `{"interests":[…]}`,
  normalizes topic/evidence aliases. **This is the single most fragile piece**
  (see B11b below).
- **Gate** — soft: a failed parse/schema does not zero the reward, it multiplies
  by 0.1; but the **judge only fires when `gate>0` and at least one interest
  parsed**, so a broken parser silently starves the judge.
- **r_rule (rules, no LLM)** — anti_collapse graded band (`tol_under/over=2×`,
  `floor=0.4`), anti_hallucination, fidelity (substring grounding of evidence
  actions against input signals, `coverage_floor=0.3`). These are the
  reusable **mechanism**; the task binding is the layer1 signal/interest schema.
- **r_llm (online judge)** — a direct gpt-5.1 Azure call scoring each interest's
  utility/precision (1–10). Gated by `LAYER1_JUDGE_ENABLED=1`; if disabled or the
  judge fails, the reward gracefully falls back to rule-only (`r = gate·r_rule`).

> The graded anti-collapse primitive is vendored from
> `rl/reward/graded_anti_collapse.py`; the standalone verifier used by Pass-A data
> cleaning is `rl/rl_layer1_verifier/` (parser/gate/fidelity/hallucination/compose).

---

## Step 3 — The three runs (single-variable lineage)

All three share seed 42, the full 35 615-sample set, and identical topology. Each
changes exactly one reward knob relative to the previous — clean attribution.

| run | steps | `w_rule/w_llm` | inner `r_llm` weights | one-line result |
|---|---|---|---|---|
| **stage-4 repro** (`layer1_stage4`) | 500 | **0.5 / 0.5** | utility 0.5 / precision 0.5 | faithful Stage-4 reproduction; reward 0.51→0.82, entropy climbs (Stage-4-like) |
| **wrule01** (`layer1_stage4_wrule01`) | 500 | **0.1 / 0.9** | utility 0.5 / precision 0.5 | precision/coherence/conciseness ↑, **recall ↓** — slid *along* the precision↔recall frontier |
| **recall** (`layer1_stage4_recall`) | 1000 | 0.1 / 0.9 | **utility 0.3 / precision 0.4 / recall 0.3** | interest recall **0.43→0.76**, topic recall **0.81→0.94**, precision ~flat — frontier moved *out* |

Numbers and cross-model comparison: `report/260628_rl_recall_reward.md` §8 +
`eval/COMPARISON.md`. Activation of the recall term is **env + Hydra override
only** (base config stays the reproducible wrule01 recipe):

```bash
LAYER1_JUDGE_ENABLED=1 LAYER1_RECALL_ENABLED=1 \
LAYER1_RECALL_CANDIDATES_DIR=<train candidates dir> AZURE_OPENAI_KEY=... \
python -m pipelinerl.launch --config-name layer1_rl \
  reward.llm_weights.utility=0.3 reward.llm_weights.precision=0.4 \
  reward.llm_weights.recall=0.3 \
  output_dir=results/layer1_stage4_recall
```

### Why we reproduced Stage-4, not Stage-5 (the Pass-B decision)

The prior "Stage-5" run relied on **Pass-B (rollout-difficulty) filtering** to cut
the too-hard / saturated tail. We deliberately reproduced **Stage-4** first and
did **not** apply Pass-B, because the Pass-B forensics (`rl_data_cleaning_process.md`
addendum + `260625` §"Pass-B analysis") showed:

1. On a **trained** checkpoint (stage42) the rule reward is near-dead — gate≈0.99,
   anti_collapse/anti_halluc≈0.99, within-group r_rule std median 0.009 — so the
   fraction of "too-hard, whole-group gate=0" prompts is **very small and
   saturation-masked**. Thresholds set from a trained ckpt are wrong.
2. Pass-B difficulty must be measured with the **SFT init** (the actual RL start),
   not a trained ckpt. Until that (GPU-bound) run exists, applying Pass-B would
   just bake in a mis-measured difficulty distribution.

So Stage-4 is the honest, reproducible baseline; Pass-B/Stage-5 is deferred until
`rl_data/v3` is generated from the SFT-init rollouts. The stage-4 run then
*empirically confirmed* the Pass-B motivation: `always_success` saturated
14%→86% within ~80 steps while `sometimes_success` (the only groups with nonzero
advantage) collapsed 71%→8% — i.e. ~86% of compute later spent on zero-gradient
groups. This is a **data-distribution** problem, not a hyperparameter one.

---

## Step 4 — Checkpoints & persistence

- Finetune writes `intermediate/50, 100, …` under `results/<run>/finetune/` (local
  `/scratch`), keeping intermediates; final weights at `finetune/current/`
  (Qwen3-4B safetensors, ~8 GB).
- Target checkpoints are rsynced to cosmos as flat HF checkpoints (safetensors +
  config + tokenizer); DeepSpeed optimizer/`global_step*` shards are **excluded**.
- Shipped: stage4/wrule01 ckpt-500 (`models/rl/`); recall step-500/900/1000
  (`MAIProfileSFT_runs/qwen3-4b-l1d-rl-recall-step*`).

---

## Step 5 — Held-out evaluation

Every RL checkpoint is scored with the **same** M1/M2/M3 pipeline as the 6-model
benchmark — see `evaluation_process.md`. Because `/scratch` + `/home` are per-node
and node-0 GPUs were saturated by live training, checkpoint evals ran on node-1
(rsync `intermediate/N` → `vllm serve` → generate → M1 → M2 judge → interest+topic
recall, reusing the Stage-1 candidate caches). Results: `260628` §8 +
`COMPARISON.md`. Not repeated here.

---

## Framework changes (what we modified in QED-Nano)

Base `02a4699`. Four files patched + one new domain package. Full diff:
`rl_repro_bundle/qed-nano-layer1.patch`; new code:
`rl_repro_bundle/bundle/pipelinerl/domains/layer1/`.

| file | change | why |
|---|---|---|
| `pipelinerl/actor.py` (+44) | `_is_deterministic_bad_request(e)` → treat HTTP 400/413 / "maximum context length" as **skip-this-rollout-group**, keep actor alive | a single oversized prompt was self-stopping the actor → NCCL timeout (see crash below) |
| `pipelinerl/world.py` (+36) | GPU pool via `PIPELINERL_GPUS`; actor base port via `actor_llm_start_port` (8180); preprocessor placement fixes | avoid busy GPU0 / port 8080 collisions; fix Zero/Unbound bugs |
| `pipelinerl/finetune/context.py` (+15) | `InitProcessGroupKwargs(timeout=4h)` on the Accelerator | raise the 1800 s c10d watchdog so a transient actor hiccup can't auto-kill finetune |
| `pipelinerl/launch.py` (+2) | read the actor base port | pair with world.py port change |
| **new** `pipelinerl/domains/layer1/` | reward / rollouts / judge / parser / load_datasets / __init__ | the whole layer1 task binding (reward mechanism + judge + parser + dataset loader) |

---

## Environment
- conda `qed-rl` (py3.11, torch 2.6.0+cu124, vllm 0.8.5.post1, transformers
  4.51.1, deepspeed 0.15.4). Pinned `fastapi==0.115.12 / starlette==0.41.3 /
  sse-starlette==2.1.3` (newer versions break `/health` via
  prometheus-fastapi-instrumentator — **do not upgrade**).
- Launch env: `WORLD_SIZE=1 RANK=0 NODE_RANK=0` (strip AzureML 2-node alloc),
  `PATH` = qed-rl bin first, `PIPELINERL_GPUS=1..7`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Secrets
- `AZURE_OPENAI_KEY` / `AOAI_KEY` (gpt-5.1 judge) read from env — **never
  committed. Rotate after any session where it was exported.**

## Gotchas (learned the hard way — see `report/260625` for full narrative)
1. **Parser silently nullified the reward (B11b, CRITICAL).** Qwen3-Thinking ends
   with ```` ``` ````**+`<|im_end|>`**; the old fence-strip only matched
   `endswith("```")`, so every rollout failed to parse → gate floored 0.1 → judge
   never fired → reward flat ~0.03. **190 steps trained on pure noise.** Fix:
   regex-strip `<\|[^>]*\|>` + cut fence with `rfind("```")`. Validation: parse_ok
   0%→73%. Always parse via `parse_completion`, never bare `json.loads`.
2. **A single oversized prompt killed the whole job (step-122 crash).** vLLM
   returns a deterministic HTTP 400 for `prompt+completion > max_model_len`
   (20106 > 20000). The actor retried it 10×, re-queued 10×, then `stopping
   actor`; finetune then starved into the 1800 s NCCL `ALLGATHER` watchdog →
   SIGABRT. Fix = the 3 layers above (actor 400-tolerance + length-filtered data
   dropping the 10 prompts > 11 808 tok + 4 h NCCL timeout). **Treat deterministic
   client errors as skip-rollout, never stop-actor.**
3. **Entropy climb is Stage-4-faithful, not a bug** — Stage-4 entropy rises to ~1
   (the pathology Stage-5's data changes were meant to fix); reproducing Stage-4
   means reproducing the climb. Watch `never_success` + `entropy` together.
4. **`never_success` is not a model property** — it is joint over (policy state ×
   temp × rollouts/group × data subset × success threshold). The online 11–15% at
   SFT-init/temp 0.8/8-samples does NOT reproduce when scoring a converged ckpt
   offline; that's expected, not a discrepancy.
5. Same per-node `/scratch`+`/home` gotcha as SFT/eval — multi-node needs full env
   staging; single-node was chosen (bottleneck is actor+judge, not finetune).
