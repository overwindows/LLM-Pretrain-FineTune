# RL Data Cleaning — Layer1-delta (v1 → v2 → v3)

How the SFT training data is filtered into the RL (GRPO) training set. Two passes:

| pass | what it removes | needs | reproducible locally? |
|---|---|---|---|
| **Pass A** (v1→v2) | bad/ungrounded **teacher targets** | teacher labels + tokenizer + verifier | **YES** — `rl/data_cleaning/pass_a_teacher_quality.py` |
| **Pass B** (v2→v3) | prompts that are **too hard / too easy** for the policy | 50K SFT ckpt + rule reward + 16 rollouts/prompt | **YES** — `rl/data_cleaning/pass_b_rollout_difficulty.py` (needs GPUs free for vLLM) |

**Decision (confirmed):** Pass B rollouts are generated with the **50K SFT
checkpoint** (`models/sft/layer1_delta_thinking_50k_4o-v1/`), which is the RL
initialisation (Stage4 ≈ SFT-final here). The rule reward is the binary verifier
gate; the graded anti-collapse term (`rl/reward/graded_anti_collapse.py`,
`tol=2×, floor=0.4`) is used for the RL **training** reward, not the Pass B
difficulty gate.

- **v1** = the SFT `train.jsonl` (36 252 records). Same file we trained SFT on.
- **v2** = after Pass A (teacher quality).
- **v3** = after Pass B (rollout difficulty).

> Remembered node-1 totals: Pass A dropped 660 → v2 = 35 592; Pass B dropped
> 6 650 → v3 = 28 942. These are recollections, not verified code; our local
> Pass A reconstruction prints its own counts (see below) and may differ slightly
> because the exact node-1 grounding target / empty-list policy / token budget
> are node-1 artifacts.

---

## Pass A — teacher quality (LOCAL, reproducible)

Script: [`rl/data_cleaning/pass_a_teacher_quality.py`](../rl/data_cleaning/pass_a_teacher_quality.py)

It **reuses the actual workspace verifier** so the gate semantics match what RL
will use at train time:
- `verifier_layer1/parser.py` → `parse_completion` (strip `<think>`, strip
  ```` ``` ```` fence, `json.loads` **with truncation repair** — tries appending
  `}`, `]`, `]}`, …; accepts `{"interests":[…]}`; normalizes topic/evidence aliases),
- `verifier_layer1/reward/fidelity.py` → `compute_fidelity` (substring grounding
  of evidence actions against the input signals).

### The four filters

| id | name | rule |
|---|---|---|
| **F1** | `gold_none` | teacher completion fails to parse as Layer1 JSON (`parse_ok=False`) — even after truncation repair |
| **F2** | `gold_bad_schema` | parses but yields no schema-valid interests (`schema_ok=False`); **empty `[]` teacher** counts here unless `--keep-empty-teacher` |
| **F3** | `halluc_rate==1` | teacher has interests but **every** evidence action is ungrounded (`compute_fidelity.fidelity == 0` vs the input `Action`/`intent`/`DetailedSource` strings) |
| **F4** | `prompt_overlong` | tokenised prompt (system+user, Qwen tokenizer) > `--max-prompt-tokens` (default 12000) |

### Two bugs found while reconstructing (documented so they don't recur)
1. **Truncation repair matters.** A naïve `json.loads` over-drops F1: GPT-4o
   completions that hit `max_tokens=16000` are truncated mid-JSON but the real
   parser repairs them. Always go through `parse_completion`.
2. **Signal field casing.** The denoised signals use **capitalised** keys
   (`Action`, `intent`, `DetailedSource`), but `fidelity._signal_text` only
   knows lowercase `action`. Passing the raw signal dicts makes the haystack
   empty → F3 flags *everything*. The script's `signal_corpus()` hands fidelity
   the `Action`/`intent`/`DetailedSource` strings directly.
3. **Tokenizer return type.** Under transformers 5.x, `apply_chat_template(
   tokenize=True)` can return a dict-like; `len()` on it silently gives a tiny
   number → F4 never fires. The script's `make_prompt_token_counter` normalizes
   the return to a flat id list.

### Run

```bash
PRL=/home/aiscuser/.conda/envs/pipeline-rl/bin
Y=<cosmos>/shares/users/yuhangbai
W="<workspace>/MaiProfile-main 1/rl_layer1"
$PRL/python $Y/MAIDistillation0623/rl/data_cleaning/pass_a_teacher_quality.py \
  --train-jsonl $Y/MAIProfileSFT_50k/data/splits/layer1_delta_thinking_50k_4o/train.jsonl \
  --base-model  $Y/MAIDistillation0623/models/base/Qwen3-4B-Thinking-2507 \
  --verifier-dir "$W" \
  --out-jsonl    $Y/MAIDistillation0623/rl_data/v2/train.jsonl \
  --report-json  $Y/MAIDistillation0623/rl_data/v2/pass_a_report.json \
  --max-prompt-tokens 12000
```

CPU-only; safe to run alongside GPU training. Output:
`rl_data/v2/train.jsonl` + `rl_data/v2/pass_a_report.json` (per-filter counts).

### Local result (this reconstruction)

<!-- PASS_A_RESULT -->
Full run over 36 252 records → **v2 = 35 625 kept, 627 dropped (1.73%)**:

| filter | local count | remembered node-1 |
|---|---|---|
| F1 `gold_none` | 274 | 134 |
| F2 `gold_bad_schema` (incl. 132 empty `[]` teacher) | 132 | 8 |
| F3 `halluc_rate==1` | 50 | 357 |
| F4 `prompt_overlong` (>12k Qwen tok) | 171 | 161 |
| **total dropped** | **627** | 660 |
| **v2 kept** | **35 625** | 35 592 |

The totals match closely (627 vs 660; 35 625 vs 35 592). The per-filter mix
differs because node-1's exact filter *order* and definitions differ slightly
(F1↔F2: truncation-repairable vs not; F3 grounding strictness; empty-`[]`
attribution). F4=171 ≈ 161 confirms the 12k-token budget. The script is the
faithful local reconstruction; `rl_data/v2/pass_a_report.json` holds the counts.

Outputs: `rl_data/v2/train.jsonl` (35 625 lines, 478 MB) +
`rl_data/v2/pass_a_report.json`.

Token-budget context (gpt-4o metadata, 36 252 records): prompt tokens
p50=1 378, p90=3 526, p99=8 668, max=79 792; **prompt>12k = 143** (gpt-4o) /
≈161 (Qwen) — consistent with the remembered F4=161.

---

## Pass B — rollout difficulty

Script: [`rl/data_cleaning/pass_b_rollout_difficulty.py`](../rl/data_cleaning/pass_b_rollout_difficulty.py)

**Recipe:** for each surviving v2 prompt, generate **16 rollouts**
with the **50K SFT checkpoint** (the RL init), score each with the **rule reward**
(binary `r_rule ∈ {0,1}` = parse/schema/grounding gate passed), then branch:

| branch | condition over the 16 rollouts | action | remembered count (of 35 592) |
|---|---|---|---|
| **too_hard** | `max(r_rule) == 0` (never once valid) | drop | 5 380 (15.12%) |
| **easy_check** | `min == max == 1` **and** `std(r_llm) < 0.05` | drop | 525 (1.48%) |
| **saturated** | kept(mixed) **and** `mean(r_llm) > 0.95` **and** `std(r_llm) < 0.03` | drop | 745 (2.09%) |
| **kept** | everything else | keep | → v3 = 28 942 |

`r_llm` = the M2 LLM-judge reward (interest-level utility + precision).

### Which model generates the rollouts? — **the 50K SFT checkpoint (= RL init)**

Difficulty filtering must reflect **the policy that will actually do RL**. The RL
stage initialises from the SFT model, so Pass B rollouts are generated with the
**50K SFT checkpoint** (`models/sft/layer1_delta_thinking_50k_4o-v1/`). Confirmed
with the owner: Stage4 ≈ SFT-final, so the SFT ckpt is the correct rollout policy.

### Execution (now reproducible locally)
`pass_b_rollout_difficulty.py`: load v2 → serve the SFT ckpt with vLLM → sample
**16 rollouts/prompt** (temp 1.0, top_p 0.95, max_tokens 4096 — matching the GRPO
rollout config) → score each rollout's `r_rule` via the verifier gate
(`parse_completion` + fidelity) → score `r_llm` via the M2 judge → apply the three
branch rules → write `rl_data/v3/train.jsonl` + `pass_b_report.json`.

**Resource note:** rollout generation needs the GPUs for vLLM. While the SFT
repro run is training (all 8 GPUs at 100%), Pass B must wait — run it after SFT
training finishes, or on a second node. The rule-reward scoring and branch logic
are CPU-only and validated independently.

---

## Status
- [x] Pass A script (local, faithful, uses real verifier) — produces `rl_data/v2/` (35 625 lines).
- [x] Pass B script written — `rl/data_cleaning/pass_b_rollout_difficulty.py` (50K SFT ckpt rollouts).
- [ ] Pass B run — waiting for GPUs (SFT repro training in progress, ~4 h remaining).

---

## Addendum (2026-06-26) — stage42 rollout forensics + reward-decomposition method

This addendum records a **diagnostic study** done while preparing the real
Pass B run. It does NOT replace the recipe above (which correctly uses the **50K
SFT checkpoint**). It was run on the **stage42 RL checkpoint** to (a) build and
validate the per-rollout reward-decomposition tooling, and (b) understand what
"low GRPO information quality" looks like in the reward components. Key lesson:
**stage42 is the wrong model to set v3 thresholds with** — but the analysis
method transfers directly to the SFT-checkpoint rollouts.

### Sample
- 517 v2 prompts (stride-69 subsample of train.lenfiltered) × 16 rollouts =
  **8 272 rollouts**, generated with `MAIProfileSFT_runs/qwen3-4b-l1d-rl-stage42`
  (the *trained* RL ckpt, NOT the SFT init).
- Scored with the **training-exact** reward: `pipelinerl/domains/layer1/reward.py`
  recomputes gate + graded r_rule (2b anti-collapse + anti-halluc + fidelity) on
  every rollout text; `r_llm` from the M2 judge (gpt-5.1, 8 200 Azure calls).
- Tooling: `/home/aiscuser/passb_full_analyze.py` (reusable on any rollout set).

### What stage42 looks like (per-rollout global)
| component | mean | std | note |
|---|---|---|---|
| gate | 0.992 | 0.084 | 99% pass |
| anti_collapse | 0.985 | 0.063 | **saturated, ~no penalty** |
| anti_hallucination | 0.989 | 0.095 | **saturated** |
| fidelity_score | 0.795 | 0.264 | only rule term with real spread |
| r_rule (graded) | 0.968 | 0.073 | saturated |
| r_llm (judge) | 0.842 | 0.119 | the live signal |
| **fused** = gate·(0.5·r_rule+0.5·r_llm) | 0.904 | 0.096 | training reward |

### GRPO signal = within-group (16-rollout) std
| within-group std | mean | p50 | p90 |
|---|---|---|---|
| r_rule | 0.027 | **0.009** | 0.073 |
| r_llm | 0.069 | **0.042** | 0.205 |
| fused | 0.048 | **0.022** | 0.119 |

Fraction of prompt-groups with **std(fused) below ε** (≈ no advantage / no grad):
ε=0.02 → 41.2%, **ε=0.03 → 71.6%**, ε=0.05 → 82.0%.

Where the (little) variance lives, per group:
- **both r_rule & r_llm flat (no signal at all): 52.8%**
- variance mainly from r_llm (judge): 42.2%
- variance mainly from r_rule: **only 5.0%**

### Lessons (carry forward)
1. **On a trained ckpt the rule reward is dead.** gate≈0.99, anti-collapse and
   anti-halluc ≈0.99; within-group r_rule std median **0.009**. The only rule
   term with spread is **fidelity** (0.795±0.264). So on stage42 almost all
   learning signal is the **judge (r_llm)**, and 53% of groups have no signal at
   all → heavily saturated.
2. **The correct "GRPO info quality" metric is `std(fused)` within the 16-rollout
   group**, further decomposed into rule-driven vs judge-driven. This is the
   metric to threshold for cleaning, not the binary gate.
3. **Do NOT set v3 thresholds from stage42.** A trained policy makes the data
   look uniformly easy (saturation), which is a property of the *model*, not the
   prompts. Pass B difficulty must be measured with the **50K SFT checkpoint**
   (the RL init), exactly as the recipe above states. On the SFT (weaker) model,
   fidelity / anti-collapse / gate will carry real variance and the
   too_hard / easy / saturated split will look completely different.
4. **Two earlier mis-reads, corrected:** (a) an approximation that set
   `r_rule := binary gate` inflated "easy" to ~69% — fixed by recomputing the
   graded r_rule with the real reward.py; (b) reporting `std(fused)<0.03 = 77.7%`
   conflated the 0.5 fused scaling (`std(fused)=0.5·std(r_llm)` on all-gate-pass
   groups) — always state which quantity the std is over.

### Next step
Generate the Pass B rollouts with the **50K SFT checkpoint** over full v2
(35 625 prompts), score with the same reward decomposition, then set the
too_hard / easy / saturated thresholds from *that* distribution to produce
`rl_data/v3/`. Reuse `passb_full_analyze.py` unchanged.
