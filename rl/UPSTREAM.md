# RL framework provenance & modifications

The RL stage of EasyPosttrain runs on **QED-Nano's `pipelinerl`** (async GRPO).
We do **not** pip-install it — a copy of the upstream `training/` tree is vendored
here at [`qed-nano/`](./qed-nano/) so the repo is runnable as-is.

## Source

| | |
|---|---|
| Upstream | https://github.com/CMU-AIRe/QED-Nano |
| Commit | `02a4699` |
| Subtree vendored | `training/` → `rl/qed-nano/` |
| License | Apache-2.0 (see [`qed-nano/LICENSE`](./qed-nano/LICENSE), [`qed-nano/NOTICE`](./qed-nano/NOTICE)) |

Vendoring, modifying, and redistributing is permitted under Apache-2.0 §4. Our
obligations are met by: keeping `LICENSE`/`NOTICE`, marking changed files with a
notice header, and recording the changes below.

## What we changed (4 files)

Each modified file starts with a `# NOTICE: Modified from QED-Nano ...` header.
Base is commit `02a4699`; a full unified diff is archived at
`rl_repro_bundle/qed-nano-layer1.patch` in the MAIDistillation0623 cosmos share.

| File | Change | Why |
|---|---|---|
| `pipelinerl/actor.py` | add `_is_deterministic_bad_request(e)`; `schedule_rollouts` treats deterministic HTTP 400/413 ("maximum context length" / oversized prompt) as **skip this rollout group**, not retry/abort | a few oversized prompts were killing the whole actor |
| `pipelinerl/world.py` | GPU pool from `PIPELINERL_GPUS`; actor vLLM base port via `actor_llm_start_port`; fix preprocessor-placement edge cases | reserve GPU0, run actors on GPU1–7, avoid port clashes |
| `pipelinerl/launch.py` | `run_actor_llm` reads the actor vLLM base port | pairs with the `world.py` port change |
| `pipelinerl/finetune/context.py` | pass `InitProcessGroupKwargs(timeout=4h)` to `Accelerator` | stop a transient actor hiccup tripping the 1800s NCCL c10d watchdog and killing finetune |

## What we added (new task package)

`pipelinerl/domains/layer1/` — the Layer-1-delta task plug-in (not upstream):

| File | Role |
|---|---|
| `reward.py` | locked reward: `r = w_rule·r_rule + w_llm·r_llm` with a soft parse/format gate |
| `rollouts.py` | rollout generation for the layer1-delta task |
| `judge.py` | LLM-judge client (utility / precision / recall) |
| `parser.py` | extract the structured delta from model output |
| `load_datasets.py` | load the RL train split |
| `__init__.py` | domain registration |

## Config used

- `conf/layer1_rl.yaml` — base RL config incl. the `reward:` block (weights, gate, scales).
- Experiment overlays live under `rl_repro_bundle/exp_configs/` (cosmos):
  `layer1_stage4.exp_config.yaml` (w_rule 0.5 / w_llm 0.5),
  `layer1_stage4_wrule01.exp_config.yaml` (0.1 / 0.9),
  `layer1_stage4_recall.exp_config.yaml` (recall-on).
- Recall run is activated by env
  `LAYER1_JUDGE_ENABLED=1 LAYER1_RECALL_ENABLED=1 LAYER1_RECALL_CANDIDATES_DIR=<dir>`
  plus Hydra override `reward.llm_weights.utility=0.3 precision=0.4 recall=0.3`.

See `../docs/PIPELINE.md` and the cosmos `process_summary/rl_training_process.md`
for the full training procedure, hyperparameters, and the three-run lineage.

## Citation

If you use this framework, please cite QED-Nano (see the upstream repo's README).
