# LLM-Pretrain-FineTune (EasyPosttrain)

GPU-side training code for LLM SFT and RL fine-tuning.
This repo is cloned on AML Singularity GPU nodes by the agentic auto-research brain.

## Architecture

```
Brain (local)                           GPU Node (AML Singularity)
─────────────────────────────────       ──────────────────────────────────────
Q:\LLM-Fine-Tuning\agentic_sft_rl\     git clone this repo → /tmp/gpu_code
  scripts\aml_submit.py  ─────────────►  scripts/run_pipeline.py
  agent\pipeline.py                       → sft/sft_train.py
  agent\sft_stage.py                      → rl/qed-nano/pipelinerl/launch.py
  agent\rl_stage.py                       → data_cleaning/pass_{a,b}.py
  agent\eval_stage.py                     → eval/eval_m{1,2}.py
  project.yaml (via AML input)         Cosmos RW_MOUNT (data in + checkpoints out)
```

## Directory Layout

```
sft/                    SFT training (HF Trainer + DeepSpeed ZeRO-3)
  sft_train.py          Main training script (8 GPUs, bf16, sdpa)
  configs/              YAML config templates
rl/qed-nano/            PipelineRL (GRPO/PPO)
  pipelinerl/           Core framework (Actor, Preprocessor, Finetune roles)
  conf/                 Hydra configs (layer1_rl.yaml, base.yaml)
data_cleaning/          Pass A (quality filter) + Pass B (difficulty filter)
data_prep/              Data conversion and preprocessing utilities
eval/                   M1 (rule-based) + M2 (LLM judge) evaluation
docs/                   AML operations, design docs, reproduction guides
env/                    Conda environment YAMLs (pipeline-rl, qed-rl)
scripts/                Entry point for AML job runs (run_pipeline.py)
```

## Conda Environments

| Env | Used for | Key packages |
|-----|----------|-------------|
| `pipeline-rl` | SFT, data, eval, vLLM | torch 2.6.0+cu124, transformers 5.5.4 |
| `qed-rl` | RL (PipelineRL) only | py3.11, transformers 4.51.1, fastapi==0.115.12 |

## Docs

- `docs/AML_OPERATIONS.md` — AML node setup and Cosmos path guide
- `docs/DESIGN.md` — Task Pack roadmap
- `docs/REPRODUCE_LAYER1_DELTA.md` — Step-by-step reproduction

## Archive

Old Bloom/pretrain content preserved in branch `archive/bloom-pretrain-2023`.
