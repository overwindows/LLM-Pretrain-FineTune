# EasyPosttrain — 从"单任务复现"到"通用 Post-Training Pipeline" 设计讨论稿

> 讨论底稿。目的：对齐"我们要把 layer1_delta 的 post-training 经验泛化成一个可复用 pipeline"的**需求、现状、方案**。
> 定位：**不是开发 RL 训练框架**，而是在成熟框架之上做一层 **e2e post-training pipeline**（数据→SFT→RL→评测→归档），把踩过的坑变成默认值。

---

## 1. 需求（我们想要什么）

一个新任务的 owner（我们或其他同事），拿到**一份 teacher 模型的 input/output 数据**后，应当能：

1. **跑通我们做过的事**：数据切分 → 清洗 → SFT →（可选）RL → 评测 → 产物归档，一条主线串起来、尽量自动化。
2. **只做最少的必要事**：以配置为主；只有任务真正独有的部分才需要自己实现（如：输出结构、关心的指标/如何评测、必要时的 reward）。
3. **符合公司内部生态**：明确
   - 训练数据 / 模型**存哪**（cosmos vs 本地 scratch）；
   - 外部 LLM（judge / reward）的 **endpoint 与 key 怎么配**（不硬编码）；
   - 关键文档 / config / 产物如何**持久化到 cosmos**、可复现。
4. **默认保存关键产物**：resulting model、切分后的数据、过滤脚本与统计等，且有**命名/目录规范**，方便日后分析和复现。
5. **不过度开发**：先把"需要一个怎样的 pipeline、价值、设计、怎么用、怎么验证"想清楚，再动手。

**环境假设**：用户和我们现在一样——申请一台 AML 机器，VS Code / shell 连接；有自己的 cosmos folder（如 `/shares/users/<alias>/<project>/`，对应我们的 `.../yuhangbai/MAIDistillation0623/`）。

---

## 2. 现状（repo 现在是什么）

仓库 `EasyPosttrain`（已推 GitHub，Private）目前是 **layer1_delta 这一个任务的可复现集合**：各步骤脚本齐全、可跑，但**没有被一条 pipeline 串起来**，且任务专属逻辑是**硬编码**的。

### 2.1 各组件的"通用度"

| 组件 | 通用度 | 说明 |
|---|---|---|
| `sft/sft_train.py` | 🟢 基本通用 | 吃 chat-format jsonl + last-assistant loss mask，config 驱动 |
| `rl/qed-nano`（vendored pipelinerl 核心） | 🟢 通用 | GRPO 三段式循环与任务无关；我们的 4 个 patch 是通用稳定性修复 |
| judge 客户端（多模型注册） | 🟡 半通用 | 已抽象成注册表，但 endpoint/key 仍写在代码里 |
| `rl/.../domains/layer1/`（reward+parser+rollouts） | 🔴 任务绑定 | 输出 schema、reward 公式写死 |
| `eval/scripts` M1/M2/M3 + `prompts/` | 🔴 任务绑定 | interest/topic/evidence 三级结构 + 专属 judge rubric |
| `data_prep/`、`data_cleaning/verifier_layer1` | 🔴 任务绑定 | teacher 字段、fidelity 校验器专属 |
| config 里的绝对路径 | 🔴 写死旧 job id | 需参数化 |
| 环境坑（fastapi 钉版本 / tokenizer fix / SDPA / NCCL 4h / 长度过滤） | 🟡 通用但散落 | 所有任务都会踩，应沉淀成 setup + 文档 |

### 2.2 一句话结论

- **能**：直接跑 layer1_delta 的复现（改路径 + 从 cosmos 拉大产物即可）。
- **还不能**：直接换个新任务就用——🔴 的部分都得改代码。
- **泛化的本质**：把 🔴 抽成用户填的 **Task Pack**，把 🟡 沉淀成**约定 + 自动化**，🟢 作为不动的通用底座。

---

## 3. 方案（可以做什么、怎么做）

### 3.1 核心抽象：Task Pack（"通用底座" vs "用户提供"）

**Pipeline 提供（用户不碰）**：数据切分/长度过滤/模板化、SFT trainer、RL 循环、serve+generate、judge 客户端、编排器、产物归档与 cosmos 持久化、环境坑的默认修复。

**用户提供（一个 `tasks/<name>/` 目录 + 少量 Python）**：

| 项 | 内容 | layer1 现成样例 |
|---|---|---|
| 必填 · 数据适配器 | `adapt(record) → {prompt, reference}` | `data_prep/prepare_gpt4o_data.py` |
| 必填 · 输出 schema + parser | 声明输出结构 + `parse(completion)` | `domains/layer1/parser.py` |
| 必填 · 评测 | M1 规则函数 + M2 judge rubric（prompt） | `eval/scripts/eval_m*` + `prompts/` |
| 选填（仅 RL）· reward | reward 定义 | `domains/layer1/reward.py` |

**关于 reward（减负关键）**：不强制从零写。Pipeline 给默认模板 `r = gate·(w_rule·r_rule + w_llm·r_llm)`，用户**复用为评测写的 M2 judge rubric 当 `r_llm`**、可选加少量 rule 检查即可。即**评测与 reward 共用一套 judge 定义**。只有高级用户才自定义 reward。

### 3.2 起始数据契约（对输入的明确要求）

- 输入 = teacher 的 **(input, output) 成对样本**，jsonl，每行一条 JSON。
- 用户的 `adapt()` 须能映射到规范单元：`{"prompt": <发给模型的完整输入>, "reference": <teacher 输出>}`。
- 规模建议、字段命名、编码、去重/PII 责任写进 `DATA_CONTRACT.md`。
- 之后的 split / 清洗 / 模板化由 pipeline 负责。

### 3.3 阶段编排（自动化串联）

一条主入口 `run.sh <project.yaml> --stages ...`，每阶段可单跑、可跳过、幂等：

```
raw teacher jsonl (cosmos)
  → adapt + split (train/val/test + manifest)
  → clean (Pass A 质量 / Pass B 难度)
  → SFT
  → [需要 RL?] → RL (pipelinerl + task pack)
  → serve + generate on test
  → eval M1/M2/M3 + report
  → archive artifacts → cosmos
```

理想情况用户只需**填 `project.yaml` + 放好 `tasks/<name>/`** 然后 `run.sh`；想干预用 `--stages sft,eval` 或覆盖 config。

### 3.4 配置收口：一个 `project.yaml`

所有环境差异集中一处，**cosmos 路径都相对 `cosmos_root`**（换用户只改一行）：

```yaml
project: my_new_task
cosmos_root: /shares/users/<alias>/<MyProject>/
base_model:  <path or hf id>
task: tasks/my_new_task
data:
  raw: ${cosmos_root}/raw/teacher.jsonl
  split: {train: 0.8, val: 0.1, test: 0.1, seed: 42}
judge:
  endpoint_env: MY_JUDGE_ENDPOINT     # 读环境变量，不硬编码
  api_key_env:  MY_JUDGE_KEY
  model: gpt-5.1
rl: {enabled: true, max_steps: 500, w_rule: 0.5, w_llm: 0.5}
```

### 3.5 产物与保存规范（复现/分析需求）

默认落到 `cosmos_root/<project>/runs/<run_id>/`（时间戳 + git sha，不可变）：

```
runs/<ts>-<git_sha>/
  config.snapshot.yaml          # 本次完整生效配置
  data/split_manifest.json      # 哪些 id 进了 train/val/test（可复现 split）
  clean/pass_a_report.json ...  # 过滤了什么、为什么
  sft/best_ckpt/ metrics.json
  rl/ckpt-*/ metrics.json
  eval/{m1,m2,m3}.json report.md
  stats/token_len_hist.json reward_curve.csv
  provenance.json               # git sha, base commit, env freeze, 时间, 机器
```

三条铁律：**每 run 一个不可变目录**；**配置快照 + split manifest + env freeze 必存**（复现三要素）；**过滤脚本/报告随数据存**（可追溯"为什么丢了这条"）。

### 3.6 公司生态集成

| 需求 | 方案 |
|---|---|
| 数据/模型存哪 | `cosmos_root` 下规范目录；本地 `/scratch` 只做快盘，跑完 rsync 回 cosmos。**明确 per-node scratch 不共享、cosmos 共享但慢**的坑写进文档 |
| endpoint / key | 一律 **环境变量 + `project.yaml` 的 `*_env` 引用**，代码零硬编码；提供 `endpoints.example.sh` 模板 + `DefaultAzureCredential` 支持；密钥永不进 git |
| 文档/config 持久化 | 编排结束自动把 `config.snapshot` / `report.md` / `provenance.json` 推到 cosmos（现在手动 rsync 的自动化版） |

---

## 4. 怎么验证"开发成功"

- **L0 冒烟**：干净 AML 机 → clone → 装环境 → 指向小样本 cosmos path → `run.sh --smoke`（几十条数据跑通 SFT+eval，~10 分钟）。证明"串起来了"。
- **L1 复现**：用 layer1_delta 数据完整跑一遍，指标落在 `COMPARISON.md` 历史噪声内。**layer1 = 我们自己的 golden test**，证明泛化没跑坏原任务。
- **成功定义**：一个没碰过本项目的同事，照 README，在新机器上用 `project.yaml` 指定 cosmos 数据路径，跑通 SFT/RL/eval 并拿到归档产物。

---

## 5. 明确不做（防过度开发）

- ❌ 不自研 RL 框架 / trainer（pipelinerl、HF Trainer 保持 vendored 不动）
- ❌ 不做 Web UI / 实验管理平台（wandb 已够）
- ❌ 不追求任意模型架构/任意任务范式——**先只保证"teacher 蒸馏 + 结构化输出"这一类**（真正验证过的）
- ❌ 不做通用 DAG 引擎——编排就是带 `--stages` 的脚本
- ✅ 只做：任务契约、编排串联、生态约定、坑位默认化、产物规范

---

## 6. 落地路径（分阶段，可随时停）

| 阶段 | 产出 | 价值 |
|---|---|---|
| P0 | `DESIGN.md`（本稿）+ `DATA_CONTRACT.md` | 对齐认知，零代码成本 |
| P1 | 抽 Task Pack 接口 + 把 layer1 重构为第一个样例 `tasks/layer1_delta/`（只搬位置不改逻辑） | 证明抽象成立，layer1 仍能跑 = 回归通过 |
| P2 | `project.yaml` + 路径参数化 + `run.sh` 编排 + `--smoke` | "一条命令跑通"落地 |
| P3 | 产物归档规范 + cosmos 持久化自动化 + endpoint/key 收口 | 生态闭环 |
| P4 | layer1 跑 L1 复现验收 + "新任务 15 分钟上手"指南 | Definition of Done |

---

## 7. 待决策（讨论用）

1. **Task Pack 形态**：(a) 纯配置 + 约定函数（低门槛）还是 (b) Python 包实现约定接口（灵活，门槛稍高）？— 倾向 **(b)，以 layer1 当 copy-paste 模板**。
2. **reward 默认模板**：认同"评测 judge 与 RL reward 共用定义、只有高级用户自定义 reward"吗？
3. **layer1 定位**：作为 `tasks/layer1_delta/` 样例长期留仓，兼当 golden test + 教程？
4. **范围**：首版是否只支持"teacher 蒸馏 + 结构化输出"，其它任务范式暂不承诺？
