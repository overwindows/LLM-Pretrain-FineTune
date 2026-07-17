# 复现 Layer1-delta 全链路（当前 repo 的实际命令）

> 本文记录**用现在这个 repo、手动**从头复现 layer1_delta post-training 的每一步：
> 数据准备 → 清洗 → SFT → RL → serve → 生成 → 评测 → 报告。
> 每一步都是可独立运行的脚本；**目前没有统一编排器**（`run.sh --stages` / `project.yaml`
> 属于 [DESIGN.md](DESIGN.md) 的愿景，尚未实现），所以需要人工按序执行、手动接上一步的产物。
>
> 权威细节见 [`docs/process/`](process/) 四篇过程文档；本文是"照着敲就能跑"的命令速查。

---

## 0. 环境与约定

**两个 conda env**（严格区分，装反会报兼容错）：

| env | 用途 | 关键版本 |
|---|---|---|
| `pipeline-rl` | 数据清洗 / SFT / serve / eval | torch 2.6.0+cu124, transformers 4.51.1(或 5.x，SFT 已验证兼容), vllm 0.8.5.post1, deepspeed 0.15.4 |
| `qed-rl` | **仅** RL 训练 | py3.11, torch 2.6.0+cu124, vllm 0.8.5.post1, transformers 4.51.1, deepspeed 0.15.4；`fastapi==0.115.12 / starlette==0.41.3 / sse-starlette==2.1.3` 不可升级 |

一次性准备：

```bash
# RL 框架从源码就地安装（不 pip 装到全局）
bash rl/qed-nano/install.sh

# 密钥走环境变量（AZURE_OPENAI_KEY 等），永不进 git
source /home/aiscuser/.secrets/maiprofile_sft.env
```

**存储约定**：大产物（ckpt、`*.jsonl`、rollout、wandb）不进 git，落在 cosmos
`MAIDistillation0623/`；本地 `/scratch` 只做快盘，跑完 rsync 回 cosmos。
`/scratch` 与 `/home/aiscuser` **per-node 不共享**，cosmos 共享但慢。

> ⚠️ **已知硬编码**（换任务/换机器需改源码）：`data_prep/prepare_50k_data.py`
> 的 `WUC_ROOT` / `DEFAULT_OUT_ROOT`；`rl/qed-nano/conf/layer1_rl.yaml` 的
> `model_path`；`rl/qed-nano/pipelinerl/domains/layer1/load_datasets.py` 的训练集路径；
> eval config 里的绝对路径。

---

## 1. 数据准备 · `data_prep/`（env: 任意 python）

把 teacher 原始 jsonl 按 `user_id` 分层切成 train/val/test（seed=42，可复现）。

```bash
# 50K 蒸馏数据切分 → data/splits/layer1_delta_thinking_50k_4o/{train,val,test}.jsonl + manifest.json
python data_prep/prepare_50k_data.py --step layer1_delta

# （可选）从 test 采样 1k 子集，供快速 eval
python data_prep/sample_test_1k.py \
  --in  <splits>/test.jsonl \
  --out <splits>/test_1k.jsonl \
  --idx <splits>/test_1k.sampled_idx.json \
  --n 1000 --seed 42
```

- **输入**：`curation_data_thinking_layer1_delta_50k.jsonl`（teacher (input,output) 对）
- **输出**：`{train,val,test}.jsonl` + `manifest.json`（含 split 大小、user 数、源 SHA256）

---

## 2. 数据清洗 · `data_cleaning/`（env: `pipeline-rl`）

> 注意：Pass B 依赖 SFT 产物 —— 真实顺序是 **先 SFT(v1) → Pass B → 用 v2/v3 训 RL**，
> 数据清洗与 SFT 是交织的，不是纯线性。

### Pass A — teacher 质量过滤（4 个门：JSON 坏 / schema 空 / 全幻觉 / prompt 超长）

```bash
python data_cleaning/pass_a_teacher_quality.py \
  --train-jsonl <splits>/train.jsonl \
  --base-model  models/base/Qwen3-4B-Thinking-2507 \
  --verifier-dir rl_layer1 \
  --out-jsonl    rl_data/v2/train.jsonl \
  --report-json  rl_data/v2/pass_a_report.json \
  --max-prompt-tokens 12000
```

### Pass B — rollout 难度过滤（4 个子命令；用 SFT ckpt 采 16 rollout 判定太难/太易/饱和）

```bash
# 1) generate (GPU, 可分片)  2) score-rule (CPU)  3) score-llm (Azure)  4) branch (CPU)
python data_cleaning/pass_b_rollout_difficulty.py generate \
  --v2-jsonl rl_data/v2/train.jsonl --sft-ckpt <sft_ckpt> \
  --rollouts-jsonl passb/shard_0.jsonl --num-shards 8 --shard-id 0 \
  --max-model-len 40960 --gpu-mem-util 0.9

python data_cleaning/pass_b_rollout_difficulty.py score-rule \
  --rollouts-jsonl passb/merged.jsonl --scored-jsonl passb/scored_rule.jsonl \
  --verifier-dir rl_layer1

python data_cleaning/pass_b_rollout_difficulty.py score-llm \
  --scored-jsonl passb/scored_rule.jsonl --scored-llm-jsonl passb/scored_llm.jsonl \
  --config eval/configs/eval/layer1_delta_thinking_50k_4o-v1.yaml \
  --eval-scripts-dir eval/scripts --concurrency 32

python data_cleaning/pass_b_rollout_difficulty.py branch \
  --scored-jsonl passb/scored_llm.jsonl \
  --v3-jsonl rl_data/v3/train.jsonl --report-json rl_data/v3/pass_b_report.json

# 8 卡并行跑 generate 的现成脚本（在 node-1 上）：
#   ssh node-1 'bash run_pass_b_generate_node1.sh <V2_JSONL> <SFT_CKPT> [LIMIT]'
```

- **输出**：`rl_data/v2/train.jsonl`（Pass A）、`rl_data/v3/train.jsonl`（Pass B kept）+ 两份 report。

---

## 3. SFT · `sft/`（env: `pipeline-rl`）— ✅ 已实测端到端跑通

```bash
cd sft
# 可用环境变量覆盖 BASE_MODEL / TRAIN_JSONL / VAL_JSONL / OUTPUT_DIR / PERSIST_CKPT_DIR ...
setsid bash launch_repro_sft.sh > logs/sft.log 2>&1 < /dev/null &
```

内部流程：`envsubst` 渲染 `configs/sft/*.tmpl` → `accelerate launch --num_processes 8
sft_train.py --config <rendered>`（DeepSpeed ZeRO-3）→ 产出 `checkpoint-*/`，
结束后自动 rsync 回 cosmos（排除 optimizer 分片）。

- **配方**（frozen）：max_seq 14336，per_device 1 × grad_accum 4 × 8 GPU = eff 32，
  2 epoch，lr 5e-6 cosine warmup 0.03，seed 42，`mask_think_in_loss=false`。
- **验收基线**：best ckpt eval_loss ≈ **0.0833**（与原始复现一致 = L1 通过）。

---

## 4. RL / GRPO · `rl/qed-nano/`（env: `qed-rl`）

用 SFT ckpt 作 init + KL 参考，异步 GRPO（3 actor vLLM + 1 preprocessor + 3 finetune）。

```bash
# 完整训练（judge 开启）
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 LAYER1_JUDGE_ENABLED=1 \
python -m pipelinerl.launch --config-name=layer1_rl output_dir=results/layer1_stage4

# 冒烟（rule-only，无 judge，几步就停）
CUDA_VISIBLE_DEVICES=1,2,3 LAYER1_JUDGE_ENABLED=0 \
python -m pipelinerl.launch --config-name=layer1_rl output_dir=results/layer1_smoke \
  finetune.rl.kl_coef=0 world.preprocessor_fraction=0 \
  world.actor_fraction=2 world.finetune_fraction=1 \
  finetune.interrupt_train_steps=5 finetune.save_checkpoint_steps=5
```

- **配置**：`rl/qed-nano/conf/layer1_rl.yaml`（500 步，kl 0.02，entropy 1e-4，lr 1e-6 cosine）。
- **reward**：`r = gate · (w_rule·r_rule + w_llm·r_llm)`，默认 0.1/0.9 变体；
  `r_llm` = gpt-5.1 M2 judge（utility/precision）。judge 需 `AZURE_OPENAI_KEY`。
- ⚠️ `model_path: /home/aiscuser/models/layer1_sft50k_patched` 写死；SFT ckpt 需先打
  tokenizer patch（`extra_special_tokens` list→`{}` 适配 transformers 4.51.1）。
- **产出**：`results/<run>/finetune/checkpoint-*/model.safetensors`（每 50 步存）。

---

## 5. Serve + 生成 · `eval/scripts/`（env: `pipeline-rl`）

```bash
# 起 vLLM OpenAI 兼容服务（subject: sft | zero_shot | rl | rl_repro）
CUDA_VISIBLE_DEVICES=0 python eval/scripts/serve_sft.py \
  --config eval/configs/eval/layer1_delta_thinking_50k_4o-v1.yaml --subject sft

# 在 test 集上生成预测
python eval/scripts/generate_outputs.py \
  --config eval/configs/eval/layer1_delta_thinking_50k_4o-v1.yaml --subject sft \
  --endpoint http://127.0.0.1:8000/v1 \
  --test-jsonl <splits>/test.jsonl \
  --output eval_results/predictions/sft.jsonl
```

- **输出**：预测 jsonl（含 `prediction`、`reference`、latency、token 数、finish_reason）。

---

## 6. 评测 · `eval/scripts/`（env: `pipeline-rl`；judge 需 `AZURE_OPENAI_KEY`）

```bash
# M1 规则指标（无 LLM）：json_parse/schema rate、interest/topic 匹配、fidelity、length_ratio
python eval/scripts/eval_m1_layer1_delta.py \
  --predictions eval_results/predictions/sft.jsonl \
  --test-jsonl <splits>/test.jsonl \
  --output eval_results/m1/sft.json --model-tag sft

# M2 judge（gpt-5.1）：interest + topic 的 utility/precision/coherence
python eval/scripts/eval_m2_layer1_delta_judge.py \
  --config <cfg> --predictions eval_results/predictions/sft.jsonl \
  --output eval_results/m2/sft.json --model-tag sft

# M2 recall（interest + topic；多 agent：propose→ground→rescue→judge，候选缓存跨模型复用）
python eval/scripts/eval_m2_layer1_delta_recall.py \
  --config <cfg> --predictions eval_results/predictions/sft.jsonl \
  --test-jsonl <splits>/test.jsonl \
  --candidates-dir eval_results/recall/cand \
  --output eval_results/recall/interest/sft.json --model-tag sft
python eval/scripts/eval_m2_layer1_delta_topic_recall.py \
  --config <cfg> --predictions eval_results/predictions/sft.jsonl \
  --test-jsonl <splits>/test.jsonl --model-tag sft \
  --output eval_results/recall/topic/sft.json

# 汇总成 markdown 报告（含成功标准判定）
python eval/scripts/aggregate_eval_report.py --config <cfg>   # → eval_results/REPORT.md
```

### 半自动编排（layer1 专属，路径写死，非通用编排器）

```bash
# 自动：等 SFT 完成 → 挑 best ckpt → teacher/sft/zero_shot 三组全跑 → 出 REPORT.md
setsid nohup bash eval/scripts/run_eval_layer1_delta_thinking_50k.sh \
  > logs/run_eval.log 2>&1 < /dev/null &

# RL 那组的评测编排（尊重 judge 全局并发 ≤32）
bash eval/scripts/orchestrate_rl_eval.sh
# 所有模型的 recall 批量跑：bash eval/scripts/run_all_recall.sh
```

---

## 阶段 → 脚本 → env 速查

| 阶段 | 脚本 | env | 产物 |
|---|---|---|---|
| 数据准备 | `data_prep/prepare_50k_data.py`, `sample_test_1k.py` | 任意 | `splits/{train,val,test}.jsonl` + manifest |
| 清洗 Pass A | `data_cleaning/pass_a_teacher_quality.py` | pipeline-rl | `rl_data/v2/train.jsonl` + report |
| 清洗 Pass B | `data_cleaning/pass_b_rollout_difficulty.py` (4 子命令) | pipeline-rl (+GPU/Azure) | `rl_data/v3/train.jsonl` + report |
| SFT | `sft/launch_repro_sft.sh` → `sft_train.py` | pipeline-rl | `checkpoint-*/` |
| RL | `python -m pipelinerl.launch --config-name=layer1_rl` | **qed-rl** | `results/<run>/finetune/checkpoint-*` |
| Serve | `eval/scripts/serve_sft.py` | pipeline-rl | `http://127.0.0.1:8000/v1` |
| 生成 | `eval/scripts/generate_outputs.py` | pipeline-rl | `predictions/*.jsonl` |
| Eval M1 | `eval/scripts/eval_m1_layer1_delta.py` | pipeline-rl | `m1/*.json` + CSV |
| Eval M2 judge | `eval/scripts/eval_m2_layer1_delta_judge.py` | pipeline-rl + Azure | `m2/*.json` |
| Eval M2 recall | `eval/scripts/eval_m2_layer1_delta_{recall,topic_recall}.py` | pipeline-rl + Azure | `recall/**/*.json` |
| 报告 | `eval/scripts/aggregate_eval_report.py` | pipeline-rl | `REPORT.md` |

---

## 当前限制（相对 [DESIGN.md](DESIGN.md) 愿景）

- ❌ 无统一编排器 `run.sh --stages`（跨阶段要手动接）
- ❌ 无 `project.yaml` 配置收口（路径散落 + 部分写死在源码）
- ❌ 无 `tasks/<name>/` Task Pack 抽象（换任务需改源码）
- ❌ 无全链路 `--smoke` 冒烟
- 🟡 产物归档是各阶段各自 rsync，无统一 `runs/<ts>-<git_sha>/` + provenance
- 🟡 endpoint 仍作默认值散在源码（key 已走 env，安全 OK）

即：**现在能"照命令手动跑通 layer1 全链路"，但还不是"填配置即用的通用 pipeline"。**
