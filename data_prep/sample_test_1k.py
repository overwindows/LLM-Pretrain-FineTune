#!/usr/bin/env python3
"""Sample test.jsonl rows + record keys for filtering downstream artifacts.

Usage:
  python scripts/sample_test_1k.py \
    --in   data/splits/layer0_signal_thinking_50k_4o/test.jsonl \
    --out  data/splits/layer0_signal_thinking_50k_4o/test_1k.jsonl \
    --idx  data/splits/layer0_signal_thinking_50k_4o/test_1k.sampled_idx.json \
    --n 1000 --seed 42
"""
import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--idx", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.inp)
    out = Path(args.out)
    idx = Path(args.idx)

    # NOTE: use newline-only splitting (proper JSONL semantics). The default
    # str.splitlines() also breaks on Unicode line separators (e.g. U+2028)
    # that legitimately occur *inside* JSON string values, which would both
    # inflate the line count and corrupt those records into invalid JSON.
    raw = src.read_text(encoding="utf-8").split("\n")
    if raw and raw[-1] == "":
        raw = raw[:-1]
    lines = raw
    total = len(lines)
    if args.n > total:
        raise SystemExit(f"requested n={args.n} > total={total}")

    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(range(total), args.n))
    chosen_set = set(chosen)

    keys = []
    out_lines = []
    for i, line in enumerate(lines):
        if i not in chosen_set:
            continue
        row = json.loads(line)
        meta = row.get("metadata", {}) or {}
        keys.append({
            "record_idx": i,
            "user_id": meta.get("user_id"),
            "date": meta.get("date"),
            "delta_index": meta.get("delta_index"),
        })
        out_lines.append(line)

    out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    idx.write_text(json.dumps({
        "source": str(src),
        "total": total,
        "n_sampled": len(chosen),
        "seed": args.seed,
        "line_indices": chosen,
        "keys": keys,
    }, indent=2), encoding="utf-8")

    print(f"[sampler] {src.name}: {total} → {len(chosen)} (seed={args.seed})")
    print(f"  out: {out}")
    print(f"  idx: {idx}")


if __name__ == "__main__":
    main()
