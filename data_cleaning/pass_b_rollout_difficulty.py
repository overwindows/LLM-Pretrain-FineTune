#!/usr/bin/env python
"""Pass B — rollout-difficulty cleaning for Layer1-delta RL data (v2 -> v3).

For each surviving v2 prompt, sample 16 rollouts with the **50K SFT checkpoint**
(the RL init), score each rollout's binary rule reward `r_rule` (the verifier
parse+schema gate) and LLM reward `r_llm` (M2 interest judge), then drop prompts
that are too hard / too easy / saturated:

  too_hard    : max(r_rule) == 0            -> policy never produces a valid output
  easy_check  : min==max==1 and std(r_llm) < EASY_STD     -> trivially solved
  saturated   : (mixed r_rule) and mean(r_llm) > SAT_MEAN
                                     and std(r_llm) < SAT_STD  -> no signal left
  kept        : everything else -> v3

Run in stages (so the GPU step can wait until SFT training frees the GPUs):

  1. generate   (GPU)   serve the SFT ckpt with vLLM, 16 rollouts/prompt -> rollouts.jsonl
  2. score-rule (CPU)   r_rule per rollout via verifier gate              -> + r_rule
  3. score-llm  (Azure) r_llm per rollout via M2 interest judge           -> + r_llm
  4. branch     (CPU)   apply the three rules, write v3 + report

Stages 2/4 are CPU-only and validated independently (see --self-test).

Rollout sampling matches the GRPO config: temperature 1.0, top_p 0.95,
max_tokens 4096, n=16.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from typing import Any

# ---- difficulty thresholds (recipe) ----
N_ROLLOUTS = 16
EASY_STD = 0.05    # easy_check: std(r_llm) below this AND all r_rule==1
SAT_MEAN = 0.95    # saturated: mean(r_llm) above this
SAT_STD = 0.03     # saturated: std(r_llm) below this

# ---- rollout sampling (GRPO) ----
GEN_TEMPERATURE = 1.0
GEN_TOP_P = 0.95
GEN_MAX_TOKENS = 4096


# ===========================================================================
# verifier (CPU) — r_rule gate
# ===========================================================================
def load_verifier(verifier_dir: str):
    if verifier_dir not in sys.path:
        sys.path.insert(0, verifier_dir)
    from verifier_layer1.parser import parse_completion  # type: ignore
    from verifier_layer1.reward.gate import compute_gate, GateConfig  # type: ignore
    return parse_completion, compute_gate, GateConfig


def r_rule_for(completion: str, parse_completion, compute_gate, gate_cfg) -> int:
    """Binary rule reward = hard parse+schema gate passed."""
    pr = parse_completion(completion or "")
    g = compute_gate(pr, gate_cfg)
    return 1 if g.value >= 1.0 else 0


# ===========================================================================
# Stage 1 — generate rollouts (GPU / vLLM)
# ===========================================================================
def _patch_tokenizer_compat() -> None:
    """transformers >=5 removed ``all_special_tokens_extended`` (this box runs
    transformers 5.5.4 for SFT, but vLLM 0.8.5's ``get_cached_tokenizer`` still
    reads it). Rebuild the AddedToken list for special tokens from
    ``added_tokens_decoder`` so vLLM can cache it correctly."""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        return

    def _aste(self):
        extended = [
            tok for tok in self.added_tokens_decoder.values()
            if getattr(tok, "special", False)
        ]
        return extended or self.all_special_tokens

    PreTrainedTokenizerBase.all_special_tokens_extended = property(_aste)


def stage_generate(args) -> int:
    _patch_tokenizer_compat()
    from vllm import LLM, SamplingParams

    prompts: list[dict] = []
    with open(args.v2_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    if args.limit:
        prompts = prompts[: args.limit]

    # data-parallel sharding: each shard handles a strided, disjoint subset so
    # the union of all shards covers every prompt exactly once (run one process
    # per GPU with CUDA_VISIBLE_DEVICES set + matching --shard-id).
    if args.num_shards > 1:
        prompts = prompts[args.shard_id :: args.num_shards]
        print(f"shard {args.shard_id}/{args.num_shards}: {len(prompts)} prompts")

    llm = LLM(
        model=args.sft_ckpt,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_prefix_caching=False,
    )
    tok = llm.get_tokenizer()

    def render_prompt(messages: list[dict]) -> str:
        # prompt = system+user (drop the teacher assistant target)
        return tok.apply_chat_template(
            messages[:-1], add_generation_prompt=True, tokenize=False)

    sp = SamplingParams(
        n=N_ROLLOUTS, temperature=GEN_TEMPERATURE, top_p=GEN_TOP_P,
        max_tokens=GEN_MAX_TOKENS)

    os.makedirs(os.path.dirname(args.rollouts_jsonl) or ".", exist_ok=True)

    # resume: count records already written (append-safe; each line is one prompt)
    done = 0
    if os.path.exists(args.rollouts_jsonl):
        with open(args.rollouts_jsonl) as f:
            done = sum(1 for ln in f if ln.strip())
    if done:
        print(f"resume: {done} prompts already done, skipping them")
    remaining = prompts[done:]

    # process in chunks so output is flushed incrementally (crash-safe) while
    # still batching enough prompts to keep the GPU busy.
    chunk = max(1, args.chunk_size)
    written = done
    with open(args.rollouts_jsonl, "a") as fout:
        for i in range(0, len(remaining), chunk):
            batch = remaining[i : i + chunk]
            rendered = [render_prompt(p["messages"]) for p in batch]
            outs = llm.generate(rendered, sp)
            for p, o in zip(batch, outs):
                rec = {
                    "user_id": p.get("metadata", {}).get("user_id"),
                    "date": p.get("metadata", {}).get("date"),
                    "messages": p["messages"],
                    "rollouts": [c.text for c in o.outputs],
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
            written += len(batch)
            print(f"  [progress] {written}/{len(prompts)} prompts written", flush=True)
    print(f"generated {len(prompts)} prompts x {N_ROLLOUTS} rollouts -> "
          f"{args.rollouts_jsonl}")
    return 0


# ===========================================================================
# Stage 2 — score r_rule (CPU)
# ===========================================================================
def stage_score_rule(args) -> int:
    parse_completion, compute_gate, GateConfig = load_verifier(args.verifier_dir)
    gate_cfg = GateConfig()
    n = 0
    with open(args.rollouts_jsonl) as fin, open(args.scored_jsonl, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["r_rule"] = [
                r_rule_for(c, parse_completion, compute_gate, gate_cfg)
                for c in rec["rollouts"]
            ]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"scored r_rule for {n} prompts -> {args.scored_jsonl}")
    return 0


# ===========================================================================
# Stage 3 — score r_llm via M2 interest judge (Azure, gpt-5.1)
# ===========================================================================
#
# Cost discipline: r_llm is only meaningful for a rollout that already passed the
# parse+schema gate (r_rule==1). A gate-failing rollout has no valid interests to
# judge, so it gets r_llm=0.0 *without* an Azure call, and a too_hard prompt
# (every rollout fails the gate) is skipped entirely. This is why the placeholder
# zeros below are excluded from the easy/saturated statistics in ``classify`` (the
# stats run over the gate-passing subset only).
def _load_judge_helpers(eval_scripts_dir: str):
    """Reuse the eval M2 judge so the per-rollout r_llm scoring is byte-for-byte
    the same prompt/parse path as the benchmark judge."""
    if eval_scripts_dir not in sys.path:
        sys.path.insert(0, eval_scripts_dir)
    import importlib
    return importlib.import_module("eval_m2_layer1_delta_judge")


async def stage_score_llm_async(args) -> int:
    import yaml
    from openai import AsyncAzureOpenAI

    mod = _load_judge_helpers(args.eval_scripts_dir)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    jcfg = cfg["judge"]

    api_key = os.environ.get("AZURE_OPENAI_KEY")
    if not api_key:
        print("AZURE_OPENAI_KEY not set in environment", file=sys.stderr)
        return 2
    client = AsyncAzureOpenAI(
        api_key=api_key,
        api_version=jcfg["api_version"],
        azure_endpoint=jcfg["azure_endpoint"],
    )
    deployment = jcfg["deployment"]
    max_tok = int(jcfg.get("max_completion_tokens", 16000))
    re_effort = jcfg.get("reasoning_effort")
    concurrency = args.concurrency or int(jcfg.get("concurrency", 32))
    interest_prompt = mod.INTEREST_NAME_PROMPT_PATH.read_text()
    sem = asyncio.Semaphore(concurrency)

    recs: list[dict] = []
    with open(args.scored_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    if args.limit:
        recs = recs[: args.limit]

    # resume: each output line is one prompt, in input order
    done = 0
    if os.path.exists(args.scored_llm_jsonl):
        with open(args.scored_llm_jsonl) as f:
            done = sum(1 for ln in f if ln.strip())
    if done:
        print(f"resume: {done} prompts already judged, skipping them")
    remaining = recs[done:]

    n_calls = 0

    async def judge_rollout(text: str) -> float:
        """r_llm for one gate-passing rollout = mean normalized interest utility."""
        nonlocal n_calls
        obj = mod.parse_prediction(text or "")
        if not isinstance(obj, dict):
            return 0.0
        interests = obj.get("interests") or []
        if not isinstance(interests, list) or not interests:
            return 0.0
        i_user = mod.build_interest_name_user_prompt(interests)
        n_calls += 1
        i_res, _ = await mod.call_judge(
            client, deployment, interest_prompt, i_user, max_tok, re_effort, sem)
        utils: list[float] = []
        if isinstance(i_res, list):
            for entry in i_res:
                sc = (entry or {}).get("scores") or {}
                u = sc.get("utility")
                if isinstance(u, (int, float)):
                    utils.append((float(u) - 1.0) / 9.0)  # 1-10 -> [0,1]
        return sum(utils) / len(utils) if utils else 0.0

    async def judge_prompt(rec: dict) -> dict:
        r_rule = rec["r_rule"]
        rollouts = rec["rollouts"]
        if max(r_rule) == 0:
            # too_hard: nothing passed the gate -> no Azure calls
            rec["r_llm"] = [0.0] * len(rollouts)
            return rec

        async def one(i: int) -> float:
            if r_rule[i] == 0:
                return 0.0  # gate-fail -> placeholder, excluded from classify stats
            return await judge_rollout(rollouts[i])

        rec["r_llm"] = list(await asyncio.gather(
            *(one(i) for i in range(len(rollouts)))))
        return rec

    chunk = max(1, args.chunk_size)
    written = done
    with open(args.scored_llm_jsonl, "a") as fout:
        for i in range(0, len(remaining), chunk):
            batch = remaining[i : i + chunk]
            out = await asyncio.gather(*(judge_prompt(r) for r in batch))
            for rec in out:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
            written += len(batch)
            print(f"  [progress] {written}/{len(recs)} prompts judged, "
                  f"azure_calls={n_calls}", flush=True)
    print(f"scored r_llm for {len(recs)} prompts -> {args.scored_llm_jsonl} "
          f"(azure_calls={n_calls})")
    return 0


def stage_score_llm(args) -> int:
    return asyncio.run(stage_score_llm_async(args))


# ===========================================================================
# Stage 4 — branch + write v3 (CPU)
# ===========================================================================
def classify(r_rule: list[int], r_llm: list[float] | None) -> str:
    """Return one of: too_hard, easy_check, saturated, kept.

    The easy/saturated statistics run over the **gate-passing** rollouts only
    (r_rule==1); gate-failing rollouts carry an un-judged r_llm placeholder.
    """
    if max(r_rule) == 0:
        return "too_hard"
    # need r_llm for easy/saturated checks; if absent, keep (only too_hard fires)
    if r_llm is None:
        return "kept"
    passing = [r_llm[i] for i in range(len(r_rule)) if r_rule[i] == 1]
    if len(passing) < 2:
        return "kept"
    std_llm = statistics.pstdev(passing)
    mean_llm = statistics.mean(passing)
    all_pass = min(r_rule) == 1  # every rollout passed the gate
    if all_pass and std_llm < EASY_STD:
        return "easy_check"
    # mixed gate outcomes, but the passing rollouts are uniformly near-ceiling
    if (not all_pass) and mean_llm > SAT_MEAN and std_llm < SAT_STD:
        return "saturated"
    return "kept"


def stage_branch(args) -> int:
    counts = {"too_hard": 0, "easy_check": 0, "saturated": 0, "kept": 0, "total": 0}
    os.makedirs(os.path.dirname(args.v3_jsonl) or ".", exist_ok=True)
    with open(args.scored_jsonl) as fin, open(args.v3_jsonl, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            counts["total"] += 1
            label = classify(rec["r_rule"], rec.get("r_llm"))
            counts[label] += 1
            if label == "kept":
                # write the original training record (messages only)
                out = {"messages": rec["messages"]}
                if "metadata" in rec:
                    out["metadata"] = rec["metadata"]
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    counts["dropped"] = counts["total"] - counts["kept"]
    with open(args.report_json, "w") as f:
        json.dump(counts, f, indent=2)
    print("\n=== Pass B report ===")
    print(json.dumps(counts, indent=2))
    print("\nRemembered node-1 (of 35592): too_hard 5380(15.12%) easy 525(1.48%) "
          "saturated 745(2.09%) kept->v3 28942")
    return 0


# ===========================================================================
# self-test — validate branch logic without GPU/Azure
# ===========================================================================
def self_test() -> int:
    cases = [
        ([0] * 16, [0.0] * 16, "too_hard"),
        ([1] * 16, [0.5] * 16, "easy_check"),                 # zero variance
        ([1] * 16, [0.5 + 0.01 * (i % 2) for i in range(16)], "easy_check"),
        ([1, 0] * 8, [0.97] * 14 + [0.98, 0.99], "saturated"),  # mixed, high mean low std
        ([1, 0] * 8, [0.2 + 0.05 * i for i in range(16)], "kept"),  # mixed, spread
        ([1] * 16, [0.1 * i for i in range(16)], "kept"),     # all valid but high variance
    ]
    ok = True
    for rr, rl, exp in cases:
        got = classify(rr, rl)
        flag = "OK " if got == exp else "FAIL"
        if got != exp:
            ok = False
        print(f"  [{flag}] expected={exp:<10} got={got:<10} "
              f"r_rule_sum={sum(rr)} mean_llm={statistics.mean(rl):.3f} "
              f"std_llm={statistics.pstdev(rl):.3f}")
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=False)

    g = sub.add_parser("generate")
    g.add_argument("--v2-jsonl", required=True)
    g.add_argument("--sft-ckpt", required=True)
    g.add_argument("--rollouts-jsonl", required=True)
    g.add_argument("--max-model-len", type=int, default=40960)
    g.add_argument("--gpu-mem-util", type=float, default=0.9)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--num-shards", type=int, default=1)
    g.add_argument("--shard-id", type=int, default=0)
    g.add_argument("--chunk-size", type=int, default=256)

    s = sub.add_parser("score-rule")
    s.add_argument("--rollouts-jsonl", required=True)
    s.add_argument("--scored-jsonl", required=True)
    s.add_argument("--verifier-dir", required=True)

    l = sub.add_parser("score-llm")
    l.add_argument("--scored-jsonl", required=True, help="input (already has r_rule)")
    l.add_argument("--scored-llm-jsonl", required=True, help="output (adds r_llm)")
    l.add_argument("--config", required=True, help="eval yaml with judge: block")
    l.add_argument("--eval-scripts-dir", required=True,
                   help="dir containing eval_m2_layer1_delta_judge.py")
    l.add_argument("--concurrency", type=int, default=0)
    l.add_argument("--chunk-size", type=int, default=64)
    l.add_argument("--limit", type=int, default=0)

    b = sub.add_parser("branch")
    b.add_argument("--scored-jsonl", required=True)
    b.add_argument("--v3-jsonl", required=True)
    b.add_argument("--report-json", required=True)

    sub.add_parser("self-test")

    args = ap.parse_args()
    if args.cmd == "generate":
        return stage_generate(args)
    if args.cmd == "score-rule":
        return stage_score_rule(args)
    if args.cmd == "score-llm":
        return stage_score_llm(args)
    if args.cmd == "branch":
        return stage_branch(args)
    if args.cmd in (None, "self-test"):
        return self_test()
    return 1


if __name__ == "__main__":
    sys.exit(main())
