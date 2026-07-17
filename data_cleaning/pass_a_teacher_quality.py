#!/usr/bin/env python
"""Pass A — teacher-quality cleaning for Layer1-delta RL data (v1 -> v2).

Drops training records whose GPT-4o *teacher target* is itself broken or
ungrounded, so GRPO never chases a bad anchor. Four filters:

  F1 gold_none        teacher completion fails to parse as Layer1 JSON
                      (after <think> strip, fence strip, and the SAME truncation
                       repair the reward gate uses)
  F2 gold_bad_schema  parses, but yields no schema-valid interests
  F3 halluc_rate==1   teacher has interests but EVERY evidence.action is
                      ungrounded (fidelity == 0 vs the input signals)
  F4 prompt_overlong  tokenised prompt (system+user) > --max-prompt-tokens

This script reuses the **actual verifier code** shipped in the workspace so the
gate semantics match exactly what RL will use:
  * verifier_layer1/parser.py    -> parse_completion (truncation repair + schema)
  * verifier_layer1/reward/fidelity.py -> compute_fidelity (grounding)

NOTE on provenance: the remembered node-1 Pass A breakdown was
F1=134 F2=8 F3=357 F4=161 (total 660, v2=35592). Those exact counts depend on
node-1-only choices (grounding target, empty-list handling, the precise token
budget). This script is the faithful *local* reconstruction using the real
verifier; it PRINTS its own counts so any delta is explicit, not hidden.

Usage:
  PRL=/home/aiscuser/.conda/envs/pipeline-rl/bin
  $PRL/python pass_a_teacher_quality.py \
      --train-jsonl  <…>/layer1_delta_thinking_50k_4o/train.jsonl \
      --base-model   <…>/models/base/Qwen3-4B-Thinking-2507 \
      --verifier-dir "<…>/MaiProfile-main 1/rl_layer1" \
      --out-jsonl    <…>/rl_data/v2/train.jsonl \
      --report-json  <…>/rl_data/v2/pass_a_report.json \
      --max-prompt-tokens 12000 \
      --keep-empty-teacher        # keep '[]' teacher targets (default: drop as F2)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

_ARR_RE = re.compile(r"\[\s*\{.*\}\s*\]", re.S)


def load_verifier(verifier_dir: str):
    """Import parse_completion / fidelity helpers from the workspace verifier."""
    if verifier_dir not in sys.path:
        sys.path.insert(0, verifier_dir)
    from verifier_layer1.parser import (  # type: ignore
        parse_completion, flatten_evidence_actions)
    from verifier_layer1.reward.fidelity import (  # type: ignore
        compute_fidelity, FidelityConfig)
    return parse_completion, flatten_evidence_actions, compute_fidelity, FidelityConfig


def input_signals(user_content: str) -> list[Any]:
    """Parse the JSON array of denoised signals embedded in the user message."""
    m = _ARR_RE.search(user_content or "")
    if not m:
        return []
    try:
        sig = json.loads(m.group(0))
        return sig if isinstance(sig, list) else []
    except json.JSONDecodeError:
        return []


def signal_corpus(user_content: str) -> list[str]:
    """Grounding haystack as plain strings.

    The denoised signals use capitalised keys (Date/Source/DetailedSource/
    Action/intent); fidelity._signal_text only knows lowercase field names, so
    we hand it the comparable text directly (Action + intent + DetailedSource).
    Teacher evidence.action is a verbatim copy of the input Action, so substring
    matching against these strings is exactly the intended grounding check.
    """
    out: list[str] = []
    for s in input_signals(user_content):
        if isinstance(s, str):
            out.append(s)
        elif isinstance(s, dict):
            parts = [s.get("Action"), s.get("intent"), s.get("DetailedSource")]
            out.append(" | ".join(p for p in parts if isinstance(p, str) and p))
    return out


def make_prompt_token_counter(base_model: str):
    """Robust prompt-token counter (transformers 5.x apply_chat_template can
    return a list OR a dict-like; handle both)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    def count(messages: list[dict]) -> int:
        prompt_msgs = messages[:-1]  # everything except the assistant target
        try:
            out = tok.apply_chat_template(
                prompt_msgs, add_generation_prompt=True, tokenize=True)
        except Exception:
            text = "\n".join(m.get("content", "") for m in prompt_msgs)
            return len(tok(text)["input_ids"])
        # normalise return type -> flat list of ids
        if isinstance(out, dict) or hasattr(out, "input_ids"):
            ids = out["input_ids"]
        else:
            ids = out
        if ids and isinstance(ids[0], (list, tuple)):  # batched
            ids = ids[0]
        return len(ids)

    return count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--verifier-dir", required=True,
                    help="dir containing verifier_layer1/ (the rl_layer1 root)")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--report-json", required=True)
    ap.add_argument("--max-prompt-tokens", type=int, default=12000)
    ap.add_argument("--keep-empty-teacher", action="store_true",
                    help="keep teacher targets that parse to an empty interest "
                         "list (default: count as F2 and drop)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    parse_completion, flatten_evidence_actions, compute_fidelity, FidelityConfig = \
        load_verifier(args.verifier_dir)
    count_prompt_tokens = make_prompt_token_counter(args.base_model)
    fcfg = FidelityConfig()

    counts = {"total": 0, "F1_gold_none": 0, "F2_gold_bad_schema": 0,
              "F3_halluc_rate_1": 0, "F4_prompt_overlong": 0,
              "empty_teacher": 0, "kept": 0}

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)

    with open(args.train_jsonl) as fin, open(args.out_jsonl, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            counts["total"] += 1
            if args.limit and counts["total"] > args.limit:
                counts["total"] -= 1
                break
            rec = json.loads(line)
            messages = rec.get("messages", [])
            assistant = messages[-1].get("content", "") if messages else ""
            user = messages[1].get("content", "") if len(messages) > 1 else ""

            pr = parse_completion(assistant)

            # F1 — teacher completion does not parse as JSON at all
            if not pr.parse_ok:
                counts["F1_gold_none"] += 1
                continue

            # empty teacher target: parsed OK but no interests
            is_empty = (pr.parsed is None or len(pr.parsed) == 0)
            if is_empty:
                counts["empty_teacher"] += 1
                if not args.keep_empty_teacher:
                    counts["F2_gold_bad_schema"] += 1
                    continue
                # keep-empty path: an empty target cannot hallucinate; only F4 left
                if count_prompt_tokens(messages) > args.max_prompt_tokens:
                    counts["F4_prompt_overlong"] += 1
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts["kept"] += 1
                continue

            # F2 — parsed but schema invalid
            if not pr.schema_ok:
                counts["F2_gold_bad_schema"] += 1
                continue

            # F3 — fully ungrounded teacher (no evidence action traceable)
            actions = flatten_evidence_actions(pr.parsed)
            fid = compute_fidelity(actions, signal_corpus(user), fcfg)
            if fid.fidelity <= 0.0:
                counts["F3_halluc_rate_1"] += 1
                continue

            # F4 — prompt too long for the RL rollout budget
            if count_prompt_tokens(messages) > args.max_prompt_tokens:
                counts["F4_prompt_overlong"] += 1
                continue

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts["kept"] += 1

            if counts["total"] % 4000 == 0:
                print(f"... {counts['total']} processed, kept {counts['kept']}",
                      flush=True)

    counts["dropped"] = counts["total"] - counts["kept"]
    with open(args.report_json, "w") as f:
        json.dump(counts, f, indent=2)
    print("\n=== Pass A report ===")
    print(json.dumps(counts, indent=2))
    print("\nRemembered node-1 breakdown: F1=134 F2=8 F3=357 F4=161 "
          "total=660 kept=35592 (provenance, not a hard target)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
