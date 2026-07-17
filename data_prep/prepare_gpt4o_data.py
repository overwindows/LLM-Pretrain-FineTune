"""prepare_gpt4o_data.py — Build train/val/test splits for the 2026-05-18
GPT-4o 2x2 experiment (layer0_signal, Thinking vs Instruct base × SFT vs zero-shot).

Source: MAICurationData/curation_data_thinking_layer0_signal.jsonl (10,791 rows,
9,999 unique users, single day 20260512, teacher=gpt-4o).

Outputs (both splits use the SAME user_id partitioning with seed=42):

    data/splits/layer0_signal_gpt4o_thinking/{train,val,test}.jsonl
        - assistant target = `<think>{reasoning}</think>{json}` (unchanged)
        - 159 no-think records wrapped with empty `<think></think>{json}`
          for format consistency; loss contribution negligible.

    data/splits/layer0_signal_gpt4o_instruct/{train,val,test}.jsonl
        - assistant target = `{json}` only (think block stripped)

Both also emit a manifest.json with split sizes, user counts, and the hash of
the source file for reproducibility.

Usage:
    python scripts/prepare_gpt4o_data.py \
        --source /scratch/.../MAICurationData/curation_data_thinking_layer0_signal.jsonl \
        --out-root /home/aiscuser/MAIProfileSFT/data/splits \
        --ratios 0.8 0.1 0.1 \
        --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
HAS_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wrap_thinking(content: str) -> str:
    """Return assistant content for the *thinking* variant.
    If it already has a <think>...</think>, return unchanged.
    Otherwise prepend empty `<think></think>` so the format is uniform."""
    if HAS_THINK_RE.search(content):
        return content
    return "<think></think>\n" + content.lstrip()


def strip_thinking(content: str) -> str:
    """Return assistant content for the *instruct* variant: strip
    `<think>...</think>` (incl. trailing whitespace). If no think tag, return
    unchanged."""
    return THINK_RE.sub("", content, count=1).lstrip()


def make_record(rec: dict[str, Any], strategy: str) -> dict[str, Any]:
    """Return a deep-ish copy with the LAST assistant message rewritten per strategy."""
    msgs = rec["messages"]
    last = msgs[-1]
    assert last["role"] == "assistant", "expected last message to be assistant"
    if strategy == "thinking":
        new_content = wrap_thinking(last["content"])
    elif strategy == "instruct":
        new_content = strip_thinking(last["content"])
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    new_msgs = msgs[:-1] + [{"role": "assistant", "content": new_content}]
    return {"messages": new_msgs, "metadata": rec.get("metadata", {})}


def split_by_user(
    records: list[dict[str, Any]],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Stratified split by user_id with the given (train, val, test) ratios.

    All records of one user go to one split. Approximation: we shuffle users
    then take prefixes. Record counts will track user counts × ~records-per-user.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, f"ratios must sum to 1, got {ratios}"
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        uid = (r.get("metadata") or {}).get("user_id", "")
        by_user[uid].append(r)

    users = sorted(by_user.keys())
    rng = random.Random(seed)
    rng.shuffle(users)

    n = len(users)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    train_u = users[:n_train]
    val_u = users[n_train : n_train + n_val]
    test_u = users[n_train + n_val :]

    out = {
        "train": [r for u in train_u for r in by_user[u]],
        "val":   [r for u in val_u   for r in by_user[u]],
        "test":  [r for u in test_u  for r in by_user[u]],
    }
    return out


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default="/scratch/azureml/cr/j/56b555526f67417a9d16fe377020dfa4/exe/wd/"
                "MAICurationData/curation_data_thinking_layer0_signal.jsonl",
    )
    ap.add_argument("--out-root", default="/home/aiscuser/MAIProfileSFT/data/splits")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.source)
    out_root = Path(args.out_root)
    print(f"[load] {src}")
    records = load_jsonl(src)
    print(f"[load] {len(records)} records")

    # Sanity: how many have <think>?
    n_with_think = sum(
        1 for r in records if HAS_THINK_RE.search(r["messages"][-1]["content"])
    )
    print(f"[sanity] records with <think>: {n_with_think} / {len(records)} "
          f"({100*n_with_think/len(records):.1f}%)")

    # Unique users
    uids = {(r.get("metadata") or {}).get("user_id", "") for r in records}
    print(f"[sanity] unique user_ids: {len(uids)}")

    src_hash = file_sha256(src)
    print(f"[sanity] source sha256: {src_hash}")

    # Split by user
    splits = split_by_user(records, tuple(args.ratios), args.seed)
    for k, v in splits.items():
        print(f"[split] {k}: {len(v)} records "
              f"({len({(r.get('metadata') or {}).get('user_id') for r in v})} users)")

    # Write both variants
    for strategy, tag in [("thinking", "layer0_signal_gpt4o_thinking"),
                          ("instruct", "layer0_signal_gpt4o_instruct")]:
        sub_root = out_root / tag
        for split_name, recs in splits.items():
            transformed = [make_record(r, strategy) for r in recs]
            out_path = sub_root / f"{split_name}.jsonl"
            write_jsonl(out_path, transformed)
            print(f"[write] {out_path}  ({len(transformed)} records)")

        # Quick stats
        train_lens = [len(r["messages"][-1]["content"]) for r in (
            [make_record(rec, strategy) for rec in splits["train"]]
        )]
        manifest = {
            "strategy": strategy,
            "source": str(src),
            "source_sha256": src_hash,
            "seed": args.seed,
            "ratios": list(args.ratios),
            "splits": {
                k: {
                    "n_records": len(v),
                    "n_users": len({(r.get("metadata") or {}).get("user_id") for r in v}),
                } for k, v in splits.items()
            },
            "train_assistant_len_stats": {
                "n": len(train_lens),
                "mean": sum(train_lens) / max(len(train_lens), 1),
                "min": min(train_lens) if train_lens else 0,
                "max": max(train_lens) if train_lens else 0,
            },
        }
        manifest_path = sub_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[write] {manifest_path}")

    # ---- audit: leakage check ----
    print("\n=== user_id leakage audit ===")
    tr_u = {(r.get("metadata") or {}).get("user_id") for r in splits["train"]}
    va_u = {(r.get("metadata") or {}).get("user_id") for r in splits["val"]}
    te_u = {(r.get("metadata") or {}).get("user_id") for r in splits["test"]}
    print(f"train ∩ val  : {len(tr_u & va_u)}")
    print(f"train ∩ test : {len(tr_u & te_u)}")
    print(f"val   ∩ test : {len(va_u & te_u)}")

    # ---- audit: format consistency on thinking variant ----
    print("\n=== thinking variant format audit (train) ===")
    n_check = 0
    n_with_think = 0
    n_with_empty_think = 0
    for r in splits["train"][:200]:
        m = make_record(r, "thinking")
        last = m["messages"][-1]["content"]
        n_check += 1
        if "<think></think>" in last:
            n_with_empty_think += 1
        elif HAS_THINK_RE.search(last):
            n_with_think += 1
    print(f"first {n_check} records: real think={n_with_think}, "
          f"empty-think wrapped={n_with_empty_think}, "
          f"other={n_check - n_with_think - n_with_empty_think}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
