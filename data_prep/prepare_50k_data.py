"""prepare_50k_data.py — Build train/val/test splits for the 2026-05-25
50K curation experiment.

Forked from scripts/prepare_gpt4o_data.py (10K layer0). Generalizes to
arbitrary step keys via --step argument. Only the thinking variant is
emitted (per current plan: thinking-only SFT on 2 nodes).

Source files (wuc/data/curation_data/):
    layer0_signal : curation_data_thinking_layer0_signal_50k.jsonl
    layer1_delta  : curation_data_thinking_layer1_delta_50k.jsonl

Outputs (per step):
    data/splits/{step}_thinking_50k_4o/{train,val,test}.jsonl
    data/splits/{step}_thinking_50k_4o/manifest.json

Splitting: 80/10/10 by user_id (metadata.user_id), seed=42. Same protocol
as prior 10K run for honest comparability.

Usage:
    python scripts/prepare_50k_data.py --step layer0_signal
    python scripts/prepare_50k_data.py --step layer1_delta
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

WUC_ROOT = Path(
    "/scratch/azureml/cr/j/fc096b74f20c46ae94d7fab7e20c1aa4/cap/data-capability/wd/"
    "INPUT_msndni/shares/users/wuc/data/curation_data"
)

SOURCE_BY_STEP = {
    "layer0_signal": WUC_ROOT / "curation_data_thinking_layer0_signal_50k.jsonl",
    "layer1_delta":  WUC_ROOT / "curation_data_thinking_layer1_delta_50k.jsonl",
}

DEFAULT_OUT_ROOT = Path(
    "/scratch/azureml/cr/j/fc096b74f20c46ae94d7fab7e20c1aa4/cap/data-capability/wd/"
    "INPUT_msndni/shares/users/yuhangbai/MAIProfileSFT_50k/data/splits"
)


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
    """Return assistant content for the thinking variant.
    If already has <think>...</think>, return unchanged.
    Otherwise prepend empty <think></think> so format is uniform."""
    if HAS_THINK_RE.search(content):
        return content
    return "<think></think>\n" + content.lstrip()


def make_record_thinking(rec: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with the last assistant message rewritten to the
    thinking-format invariant."""
    msgs = rec["messages"]
    last = msgs[-1]
    assert last["role"] == "assistant", "expected last message to be assistant"
    new_content = wrap_thinking(last["content"])
    new_msgs = msgs[:-1] + [{"role": "assistant", "content": new_content}]
    return {"messages": new_msgs, "metadata": rec.get("metadata", {})}


def split_by_user(
    records: list[dict[str, Any]],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """Stratified split by user_id. All records of one user go to one split."""
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

    return {
        "train": [r for u in train_u for r in by_user[u]],
        "val":   [r for u in val_u   for r in by_user[u]],
        "test":  [r for u in test_u  for r in by_user[u]],
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=sorted(SOURCE_BY_STEP.keys()),
                    help="step_key — selects source file and output sub-dir")
    ap.add_argument("--source", default=None,
                    help="override source path (default = SOURCE_BY_STEP[step])")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.source) if args.source else SOURCE_BY_STEP[args.step]
    out_root = Path(args.out_root)
    tag = f"{args.step}_thinking_50k_4o"
    sub_root = out_root / tag

    print(f"[step] {args.step}")
    print(f"[load] {src}")
    if not src.exists():
        print(f"ERROR: source file not found: {src}", file=sys.stderr)
        return 2
    records = load_jsonl(src)
    print(f"[load] {len(records)} records")

    n_with_think = sum(
        1 for r in records if HAS_THINK_RE.search(r["messages"][-1]["content"])
    )
    print(f"[sanity] records with <think>: {n_with_think} / {len(records)} "
          f"({100*n_with_think/len(records):.1f}%)")

    uids = {(r.get("metadata") or {}).get("user_id", "") for r in records}
    print(f"[sanity] unique user_ids: {len(uids)}")

    teacher_models = {(r.get("metadata") or {}).get("model", "") for r in records}
    print(f"[sanity] teacher models: {teacher_models}")

    src_hash = file_sha256(src)
    print(f"[sanity] source sha256: {src_hash}")

    splits = split_by_user(records, tuple(args.ratios), args.seed)
    for k, v in splits.items():
        n_users_split = len({(r.get('metadata') or {}).get('user_id') for r in v})
        print(f"[split] {k}: {len(v)} records ({n_users_split} users)")

    # Write thinking variant
    for split_name, recs in splits.items():
        transformed = [make_record_thinking(r) for r in recs]
        out_path = sub_root / f"{split_name}.jsonl"
        write_jsonl(out_path, transformed)
        print(f"[write] {out_path}  ({len(transformed)} records)")

    train_lens = [
        len(make_record_thinking(rec)["messages"][-1]["content"])
        for rec in splits["train"]
    ]
    manifest = {
        "step": args.step,
        "tag": tag,
        "strategy": "thinking",
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
        "train_assistant_len_stats_chars": {
            "n": len(train_lens),
            "mean": sum(train_lens) / max(len(train_lens), 1),
            "min": min(train_lens) if train_lens else 0,
            "max": max(train_lens) if train_lens else 0,
        },
        "teacher_models": sorted(teacher_models),
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

    # ---- audit: format consistency ----
    print("\n=== thinking format audit (train, first 200) ===")
    n_real_think = 0
    n_empty_think = 0
    n_other = 0
    for r in splits["train"][:200]:
        last = make_record_thinking(r)["messages"][-1]["content"]
        if "<think></think>" in last:
            n_empty_think += 1
        elif HAS_THINK_RE.search(last):
            n_real_think += 1
        else:
            n_other += 1
    print(f"real think={n_real_think}  empty-think wrapped={n_empty_think}  other={n_other}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
