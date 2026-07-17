# Layer1-Delta — Final Model Comparison (test_1k, n=1000)

Generated from `eval_results/m1/*.json` and `eval_results/m2/*.json`.
Judge = **gpt-5.1** (M2). Reference labels = **gpt-4o** (teacher).

> **gpt-5-2** is the deployment `gpt-5-2`, whose backend model is **gpt-5** (a
> second-capacity gpt-5). Its numbers are expected to track gpt-5 closely —
> useful as a self-consistency / variance check. **gpt-5.1** is the genuinely
> distinct newer model.

---

## M1 — objective / rule-based (no LLM)

| Model | parse | schema | n_int (pred) | top/int | int F1 | int prec | int rec | topicIoU | nameRouge | evFid | lenRatio |
|-------|------:|-------:|-------------:|--------:|-------:|---------:|--------:|---------:|----------:|------:|---------:|
| teacher (gpt-4o) | 0.990 | 0.985 | 4.69 | 1.60 | 1.000* | 1.000* | 1.000* | 1.000* | — | 0.987 | 1.000 |
| **SFT** (Qwen3-4B) | 0.999 | 0.999 | 4.40 | 1.56 | 0.625 | 0.643 | 0.627 | 0.510 | — | 0.988 | 0.998 |
| **RL** (Qwen3-4B) | 0.998 | 0.998 | 4.79 | 1.37 | 0.618 | 0.618 | 0.639 | 0.398 | — | 0.992 | 1.217 |
| **RL-repro** (Qwen3-4B, ckpt-500, w_rule=0.5) | 1.000 | 1.000 | 5.30 | 1.38 | 0.617 | 0.614 | 0.647 | 0.470 | 0.888 | 0.990 | 1.132 |
| **RL-repro-wrule01** (Qwen3-4B, ckpt-500, w_rule=0.1) | 1.000 | 1.000 | 3.79 | 1.45 | 0.611 | **0.662** | 0.594 | 0.430 | 0.879 | 0.981 | 1.018 |
| gpt-5 | 0.998 | 0.998 | 6.62 | 1.66 | 0.395 | 0.378 | 0.438 | 0.426 | 0.845 | 0.994 | 0.885 |
| gpt-5-2 (=gpt-5) | 0.990 | 0.990 | 6.99 | 1.62 | 0.399 | 0.381 | 0.447 | 0.424 | 0.840 | 0.995 | 0.905 |
| gpt-5.1 | 0.987 | 0.986 | 7.25 | 1.71 | 0.428 | 0.419 | 0.478 | 0.404 | 0.794 | 0.994 | 0.845 |

\* teacher is the reference, so it matches itself by construction (F1/IoU = 1.0).

**Read:** SOTA models emit **~6.6–7.3 interests/record** vs the gpt-4o reference's
~4.7 and our SFT/RL's ~4.4–4.8. They are far more *granular*, so their **interest
set-F1 against the 4o labels is LOW (0.39–0.43)** — not because they are worse, but
because they disagree with 4o's coarser labeling. Evidence-fidelity stays ~0.99 for
all (no hallucinated evidence).

---

## M2 — LLM judge (gpt-5.1), scored 1–10

### Interest level
| Model | n scored | utility | precision | coherence | granularity_broad |
|-------|---------:|--------:|----------:|----------:|------------------:|
| teacher (gpt-4o) | 4688 | 8.511 | 8.659 | 8.936 | 0.966 |
| **SFT** | 4383 | 8.561 | 8.731 | 9.019 | 0.956 |
| **RL** | 4783 | **8.726** | **9.275** | **9.471** | 0.971 |
| **RL-repro** (w_rule=0.5) | 5251 | 8.646 | 9.215 | 9.380 | 0.973 |
| **RL-repro-wrule01** (w_rule=0.1) | 3779 | **8.772** | **9.297** | **9.480** | 0.971 |
| gpt-5 | 6592 | 8.695 | 9.382 | 9.557 | 0.999 |
| gpt-5-2 (=gpt-5) | 6880 | 8.643 | 9.365 | 9.532 | 0.999 |
| gpt-5.1 | 7131 | 8.604 | 9.033 | 9.221 | 0.996 |

### Topic level
| Model | n scored | utility | precision | coherence | granularity |
|-------|---------:|--------:|----------:|----------:|------------:|
| teacher (gpt-4o) | 7418 | 6.874 | 9.275 | 9.511 | 0.988 |
| **SFT** | 6826 | 6.807 | 9.169 | 9.441 | 0.981 |
| **RL** | 6529 | 6.913 | 9.132 | 9.382 | 0.982 |
| **RL-repro** (w_rule=0.5) | 7263 | 6.947 | 9.214 | 9.369 | 0.984 |
| **RL-repro-wrule01** (w_rule=0.1) | 5450 | **7.092** | 9.079 | 9.322 | 0.986 |
| gpt-5 | 10842 | 6.935 | 9.452 | 9.675 | 0.984 |
| gpt-5-2 (=gpt-5) | 10926 | 6.877 | 9.451 | 9.684 | 0.980 |
| gpt-5.1 | 11633 | 6.908 | 9.487 | 9.663 | 0.988 |

---

## Headline findings

1. **Our RL model is competitive with frontier LLMs on interest-level quality.**
   RL interest precision **9.275** and coherence **9.471** essentially match gpt-5
   (9.382 / 9.557) and **beat gpt-5.1** (9.033 / 9.221) — with a 4B model.
2. **RL > SFT** on every interest-level judge axis (utility +0.17, precision +0.54,
   coherence +0.45), confirming the earlier recovery result.
3. **SOTA models are much more granular** (6.6–7.3 interests, 11k+ topics vs our
   ~6.5k). This drives their low M1 interest-F1 vs 4o but their judge scores stay high —
   i.e. they produce more, finer, still-coherent interests.
4. **gpt-5-2 ≈ gpt-5** across all metrics (Δ ≤ 0.06 on judge scores), confirming it is
   the same backend model — good variance/repro sanity check.
5. **Topic-utility is uniformly ~6.8–6.9 for everyone** (incl. gpt-5.1) — the hardest
   axis; not yet solved by scale. Our SFT/RL are within ~0.1 of frontier here.
6. **RL reproduction (`RL-repro`, stage-4 ckpt-500) confirms the RL result.** It
   reproduces the original RL run's interest-level quality within judge noise
   (precision 9.215 vs 9.275, coherence 9.380 vs 9.471, utility 8.646 vs 8.726) and
   **slightly exceeds it on signal-grounded interest recall** (overall 0.428 vs 0.399,
   matched 0.482 vs 0.454, broad 0.270 vs 0.233) while matching topic recall
   (0.808 vs 0.811). M1 stays clean (parse/schema 1.000, evidence-fidelity 0.990). The
   reproduction emits marginally more interests (5.30 vs 4.79/record), explaining the
   higher M1 topicIoU (0.470) and recall. Net: **the RL gains are reproducible.**
7. **Reward re-weighting (`RL-repro-wrule01`, w_rule 0.5→0.1, w_llm 0.5→0.9) trades
   recall for precision & concision.** Same Stage-4 recipe on this machine, only the
   reward mix changed. The model emits **~28% fewer interests** (3.79 vs 5.30/record)
   and **~25% fewer topics** (5450 vs 7263 scored). On judge quality it **improves every
   interest-level axis** — utility 8.772 (vs 8.646), precision 9.297 (vs 9.215),
   coherence 9.480 (vs 9.380) — and lifts topic utility (7.092 vs 6.947), at a small
   topic-precision cost (9.079 vs 9.214). M1 interest precision rises (0.662 vs 0.614)
   while recall drops (0.594 vs 0.647); length_ratio tightens to 1.02 (vs 1.13).
   Signal-grounded recall falls accordingly (interest overall 0.374 vs 0.428, matched
   0.419 vs 0.482, broad 0.224 vs 0.270; topic 0.754 vs 0.808). **Net: lowering w_rule
   produces a more precise, more concise, higher-judged model that covers less of the
   grounded space — a precision↔recall lever, not a strict win.** Training dynamics back
   this: wrule01 ran with higher entropy (late 0.971 vs 0.570) and ~1.5–1.8× more usable
   (non-saturated) advantage groups early/mid, before both converged to the same
   data-difficulty ceiling by step 500.

---

## Coverage / reliability notes
- M1 errors: 1 per SOTA model (1 content-filtered input). SFT/RL 0.
- M2 judge_fail (Azure content_filter on test data, unavoidable & symmetric):
  teacher 9, sft 7, rl 7, rl-repro 13, gpt-5 12, gpt-5-2 17, gpt-5.1 20.
- M2 parse_fail: ≤1 for all except gpt-5.1 (12) — gpt-5.1 occasionally wraps JSON
  differently; 12/1000 = 1.2%, immaterial to means.
- All SOTA generations ran at **concurrency 16** (per requirement). The **RL-repro**
  judge/recall passes also ran at **concurrency 16** (temporary cap; default is 32).

---

## M3 — LLM recall (signal-grounded coverage), test_1k

Measures **how much of the user's signal-grounded interest/topic space each model
covers**: `recall = covered / grounded`, where the **grounded candidate set is built
once from the raw denoised signals (propose → ground → rescue) and shared across all
models** (model-independent). A gpt-5.1 judge then decides, per grounded proposal,
whether the model's output covers it.

- **Single-agent, no-SBERT** configuration (retrieval skipped: all model items passed
  as candidates). This matches the *old* single-agent interest-recall table, so the
  numbers are directly comparable to it.
- Judge / propose / ground / rescue all = **gpt-5.1**; judging batched at 10
  proposals/request (official `JUDGE_BATCH_SIZE`).
- `n` with a usable candidate set: **interest 997**, **topic 996** of 1000.

### Interest-level recall
| Model | overall | matched | broad |
|-------|--------:|--------:|------:|
| teacher (gpt-4o) | 0.373 | 0.419 | 0.238 |
| **SFT** (Qwen3-4B) | 0.376 | 0.425 | 0.230 |
| **RL** (Qwen3-4B) | 0.399 | 0.454 | 0.233 |
| **RL-repro** (Qwen3-4B, ckpt-500, w_rule=0.5) | **0.428** | **0.482** | **0.270** |
| **RL-repro-wrule01** (Qwen3-4B, ckpt-500, w_rule=0.1) | 0.374 | 0.419 | 0.224 |
| gpt-5 | 0.263 | 0.305 | 0.116 |
| gpt-5-2 (=gpt-5) | 0.288 | 0.338 | 0.120 |
| gpt-5.1 | 0.367 | 0.423 | 0.187 |

### Topic-level recall
| Model | recall | grounded (sum) |
|-------|-------:|---------------:|
| teacher (gpt-4o) | 0.787 | 12300 |
| **SFT** (Qwen3-4B) | 0.768 | 12300 |
| **RL** (Qwen3-4B) | 0.811 | 12300 |
| **RL-repro** (Qwen3-4B, ckpt-500, w_rule=0.5) | 0.808 | 12300 |
| **RL-repro-wrule01** (Qwen3-4B, ckpt-500, w_rule=0.1) | 0.754 | 12300 |
| gpt-5 | 0.832 | 12273 |
| gpt-5-2 (=gpt-5) | **0.845** | 12082 |
| gpt-5.1 | 0.836 | 12300 |

### Recall read
1. **Interest recall: our 4B RL family leads (RL-repro 0.428 / 0.482, RL 0.399 / 0.454
   matched) — above all frontier models, including gpt-5.1 (0.367) and gpt-5 (0.263).**
   Because the grounded
   candidate set is fixed and shared, this measures *alignment to the signal-grounded
   interest space*, not raw count. SFT/RL/teacher (the ~4o-style, coarser taggers) map
   onto these grounded interests more cleanly than the ultra-granular SOTA models.
2. **SOTA's granularity hurts interest recall — especially `broad`.** gpt-5/gpt-5-2
   broad recall collapses to ~0.12 (vs teacher 0.238, RL 0.233): they emit many fine
   interests but miss the *umbrella/broad* interests a grounded set contains. gpt-5.1
   recovers somewhat (broad 0.187) but still trails the 4o-family.
3. **Topic recall flips: SOTA leads (gpt-5-2 0.845, gpt-5.1 0.836, gpt-5 0.832) vs RL
   0.811, teacher 0.787, SFT 0.768.** Topics map closely to raw signals, so SOTA's
   breadth pays off at the finer topic granularity. Our **RL still beats teacher and
   SFT** and sits within ~0.02–0.03 of frontier.
4. **RL > SFT on both recall levels** (interest +0.023 overall, topic +0.043),
   consistent with the M2 quality story — RL improves *coverage* as well as quality.
5. Recall is higher at topic (~0.77–0.85) than interest (~0.26–0.40): topics are close
   to raw signals, whereas interests require an abstraction step where granularity
   mismatch costs coverage.

> **Caveats.** (a) gpt-5.1 is both an evaluated subject *and* the propose/ground/judge
> model → mild self-evaluation bias (unavoidable, symmetric across the comparison).
> (b) SBERT retrieval was skipped, so every model item is a candidate (no recall ceiling
> from retrieval miss) — identical to the old single-agent table, hence comparable.
> (c) Per-complexity bucket aggregation was not populated in this run; only pooled
> means are reported. (d) `gpt-5-2` backend ≈ `gpt-5` (variance check).

---

## Recall pipeline / repro
- Interest: `scripts/eval_m2_layer1_delta_recall.py` — propose→ground→rescue→judge.
- Topic: `scripts/eval_m2_layer1_delta_topic_recall.py` — propose→ground→judge (no rescue).
- Driver: `scripts/run_all_recall.sh` (concurrency 32, shared candidate caches under
  `eval_results/m2_recall/{interest,topic}/candidates/`).
- Outputs: `eval_results/m2_recall/{interest,topic}/<model>.json` (+ per-record `.jsonl`).
