#!/usr/bin/env python3
"""Filter L1d teacher M2 judge artifacts to a 1k sampled subset.

Inputs:
  --idx     sampled_idx.json (from sample_test_1k.py)
  --src-dir source M2 judge dir (contains m2_teacher.interest_name.jsonl + .topics.jsonl + .json)
  --out-dir output M2 judge dir (m2_teacher.*.jsonl + recomputed m2_teacher.json summary)

Key matching: (user_id, delta_index)
"""
import argparse
import json
import statistics
from pathlib import Path


def load_jsonl(p: Path):
    rows = []
    with open(p, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def filter_rows(rows, keys):
    out = []
    for r in rows:
        k = (r.get("user_id"), int(r.get("delta_index", -1)))
        if k in keys:
            out.append(r)
    return out


def summarize_scores(rows, score_keys, n_label):
    if not rows:
        return {"n_" + n_label: 0, "scores": {}}
    summary = {"n_" + n_label: len(rows), "scores": {}}
    for sk in score_keys:
        vals = []
        for r in rows:
            v = (r.get("scores") or {}).get(sk)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        if not vals:
            summary["scores"][sk] = {"n": 0}
        else:
            summary["scores"][sk] = {
                "n": len(vals),
                "mean": round(statistics.mean(vals), 4),
                "median": statistics.median(vals),
            }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", required=True)
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="teacher")
    args = ap.parse_args()

    idx = json.loads(Path(args.idx).read_text(encoding="utf-8"))
    keys = {(k["user_id"], int(k["delta_index"])) for k in idx["keys"] if k.get("user_id") is not None}

    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # interest
    src_in = src_dir / f"m2_{args.tag}.interest_name.jsonl"
    src_tp = src_dir / f"m2_{args.tag}.topics.jsonl"
    interest_rows = load_jsonl(src_in)
    topics_rows = load_jsonl(src_tp)

    interest_keep = filter_rows(interest_rows, keys)
    topics_keep = filter_rows(topics_rows, keys)

    with open(out_dir / f"m2_{args.tag}.interest_name.jsonl", "w", encoding="utf-8") as f:
        for r in interest_keep:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / f"m2_{args.tag}.topics.jsonl", "w", encoding="utf-8") as f:
        for r in topics_keep:
            f.write(json.dumps(r) + "\n")

    interest_record_keys = {(r.get("user_id"), int(r.get("delta_index", -1))) for r in interest_keep}
    topics_record_keys = {(r.get("user_id"), int(r.get("delta_index", -1))) for r in topics_keep}
    n_records_with_pred = len(interest_record_keys | topics_record_keys)

    summary = {
        "model_tag": args.tag,
        "subset": "1k_seed42",
        "n_records_in_subset": len(keys),
        "n_records_with_pred": n_records_with_pred,
        "interest_name": summarize_scores(
            interest_keep,
            ["utility", "precision", "coherence", "granularity_broad"],
            "interests_scored",
        ),
        "topics": summarize_scores(
            topics_keep,
            ["utility", "precision", "coherence", "granularity"],
            "topics_scored",
        ),
    }
    (out_dir / f"m2_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[filter] interest: {len(interest_rows)} → {len(interest_keep)}")
    print(f"[filter] topics:   {len(topics_rows)} → {len(topics_keep)}")
    print(f"[filter] records covered: {n_records_with_pred}/{len(keys)}")
    print(f"[filter] out: {out_dir}")


if __name__ == "__main__":
    main()
