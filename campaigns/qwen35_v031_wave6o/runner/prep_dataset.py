#!/usr/bin/env python3
"""Convert internal JSONL prompts -> custom-format JSONL for `vllm bench throughput`.

`vllm bench throughput --dataset-name custom --dataset-path X.jsonl` reads
one record per line:

  {"prompt": "<text>"}  [optional: "output_tokens": N]

CustomDataset wraps the prompt as a single user-role chat message and applies
the model's chat template internally. So we:

1. Read `_export_prompt:true` rows from one source file or a directory of files.
2. Fold the original [system, user] messages into a single combined string
   (CustomDataset only supports one role).
3. Tokenize via the Gemma 4 chat template (role=user) and filter by token count.
4. Write one `{"prompt": ...}` JSON per line.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from typing import Iterable

from transformers import AutoTokenizer


def iter_export_prompts(paths: list[str]) -> Iterable[dict]:
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if not d.get("_export_prompt"):
                    continue
                msgs = d.get("messages")
                if not msgs:
                    continue
                yield d


def fold_messages(msgs: list[dict]) -> str:
    parts = []
    for m in msgs:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        if role == "system":
            parts.append(f"[SYSTEM]\n{content}")
        else:
            parts.append(content)
    return "\n\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    help="Single JSONL file OR directory of JSONL files OR glob.")
    ap.add_argument("--dst", required=True, help="Output JSONL")
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--min-tokens", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, required=True,
                    help="Drop prompts whose rendered token count exceeds this.")
    ap.add_argument("--max-keep", type=int, default=5000)
    args = ap.parse_args()

    # Resolve source paths
    if os.path.isdir(args.src):
        paths = sorted(glob.glob(os.path.join(args.src, "*.txt")) +
                       glob.glob(os.path.join(args.src, "*.jsonl")))
    elif any(ch in args.src for ch in "*?["):
        paths = sorted(glob.glob(args.src))
    else:
        paths = [args.src]
    if not paths:
        print(f"[prep] no files matched {args.src}", file=sys.stderr)
        return 1
    print(f"[prep] reading from {len(paths)} file(s):")
    for p in paths:
        print(f"  - {p}")

    print(f"[prep] loading tokenizer for {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    kept = []
    counts = {"total": 0, "too_short": 0, "too_long": 0, "error": 0}
    for d in iter_export_prompts(paths):
        counts["total"] += 1
        try:
            text = fold_messages(d["messages"])
        except Exception:
            counts["error"] += 1
            continue

        try:
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            counts["error"] += 1
            continue

        n = len(tok(rendered, add_special_tokens=False).input_ids)
        if n < args.min_tokens:
            counts["too_short"] += 1
            continue
        if n > args.max_tokens:
            counts["too_long"] += 1
            continue

        kept.append({"prompt": text, "_rendered_tokens": n})
        if len(kept) >= args.max_keep:
            break
        if counts["total"] % 1000 == 0:
            print(f"[prep] scanned={counts['total']} kept={len(kept)}", flush=True)

    print(f"[prep] DONE")
    print(f"[prep]   scanned={counts['total']} kept={len(kept)}")
    print(f"[prep]   filtered: too_short={counts['too_short']} too_long={counts['too_long']} "
          f"err={counts['error']}")

    if kept:
        lens = sorted(x["_rendered_tokens"] for x in kept)
        n = len(lens)
        print(f"[prep]   rendered token lens (n={n}):")
        print(f"    min={lens[0]}  p10={lens[int(n*0.1)]}  p50={lens[n//2]}  "
              f"p90={lens[int(n*0.9)]}  p99={lens[int(n*0.99)]}  max={lens[-1]}")
        print(f"    mean={sum(lens)/n:.0f}")

    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    with open(args.dst, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps({"prompt": rec["prompt"]}, ensure_ascii=False) + "\n")
    print(f"[prep] wrote {len(kept)} records -> {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


