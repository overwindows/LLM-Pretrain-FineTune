"""eval_m1_layer1_delta.py — Objective metrics for layer1_delta SFT (no LLM).

Reads predictions.jsonl produced by generate_outputs.py (group B/C) — or
teacher.jsonl (group A, --teacher-mode) — and reports structural / matching /
fidelity metrics tailored to layer1_delta's nested JSON output.

Predictions schema (from generate_outputs.py):
    {record_idx, user_id, date, delta_index,
     reference, prediction, latency_s, input_n_signals, error}

Metrics (mirrors the analysis we proposed in the project plan):

  Structural
    json_parse_rate        : fraction of predictions that parse cleanly
    schema_rate            : fraction with valid layer1_delta schema
    finish_truncation_rate : preds whose length suggests they were cut off
                             (not directly available from vLLM here; we infer
                             from "ends with `}]}]}}` etc"). Best-effort.

  Cardinality
    n_interests_pred / _gold    : mean / median per record
    n_topics_per_interest_*     : mean / median

  Content (vs reference / teacher)
    interest_set_iou            : token-set IoU on union of interest names
    interest_match_p/r/f1@T     : Hungarian matching by Jaccard >= T
    matched_topic_iou           : on matched pairs, topic-set IoU
    matched_name_rougeL         : on matched pairs, ROUGE-L on interest_name

  Fidelity (anti-hallucination)
    evidence_fidelity           : fraction of pred evidence.action strings
                                  that appear verbatim in the input signals

  Length
    length_ratio_chars          : len(pred) / len(reference)

Output: a JSON report at --output, plus per-record CSV breakdown next to it.

Usage:
    python scripts/eval_m1_layer1_delta.py \
        --predictions eval_results/.../predictions/sft.jsonl \
        --test-jsonl  data/splits/layer1_delta/test.jsonl \
        --output      eval_results/.../m1_objective/sft.json \
        --model-tag   sft

    # Teacher self-eval (sanity ceiling)
    python scripts/eval_m1_layer1_delta.py \
        --predictions eval_results/.../predictions/teacher.jsonl \
        --test-jsonl  data/splits/layer1_delta/test.jsonl \
        --output      eval_results/.../m1_objective/teacher.json \
        --model-tag   teacher --teacher-mode
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("eval_m1_layer1_delta")


# ---------------------------------------------------------------------------
# Robust JSON parse
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def safe_json_loads(text: str) -> Any:
    if not text:
        return None
    s = text.strip()
    m = _FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(s[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1])
                except Exception:
                    return None
    return None


# ---------------------------------------------------------------------------
# Schema check
# ---------------------------------------------------------------------------
def validate_schema(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    interests = obj.get("interests")
    if not isinstance(interests, list):
        return False
    for it in interests:
        if not isinstance(it, dict):
            return False
        if not isinstance(it.get("interest_name"), str):
            return False
        topics = it.get("topics")
        if not isinstance(topics, list):
            return False
        for t in topics:
            if not isinstance(t, dict):
                return False
            if not isinstance(t.get("topic"), str):
                return False
            if "source" in t and not isinstance(t["source"], list):
                return False
            ev = t.get("evidence")
            if ev is not None and not isinstance(ev, list):
                return False
    return True


# ---------------------------------------------------------------------------
# Matching utilities
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def norm_tokens(s: str) -> set[str]:
    return set(_TOKEN_RE.findall((s or "").lower()))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def greedy_match(
    pred_names: list[str],
    gold_names: list[str],
    threshold: float,
) -> tuple[list[tuple[int, int, float]], float, float, float]:
    if not pred_names and not gold_names:
        return [], 1.0, 1.0, 1.0
    if not pred_names or not gold_names:
        return [], 0.0, 0.0, 0.0
    P = [norm_tokens(x) for x in pred_names]
    G = [norm_tokens(x) for x in gold_names]
    pairs: list[tuple[float, int, int]] = []
    for i, p in enumerate(P):
        for j, g in enumerate(G):
            s = jaccard(p, g)
            if s > 0:
                pairs.append((s, i, j))
    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for s, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append((i, j, s))
    tp = sum(1 for _, _, s in matches if s >= threshold)
    precision = tp / len(pred_names)
    recall = tp / len(gold_names)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return matches, precision, recall, f1


def rouge_l(pred: str, gold: str) -> float:
    p = _TOKEN_RE.findall((pred or "").lower())
    g = _TOKEN_RE.findall((gold or "").lower())
    if not p or not g:
        return 0.0
    m, n = len(p), len(g)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if p[i] == g[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[m][n]
    prec = lcs / m
    rec = lcs / n
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


# ---------------------------------------------------------------------------
# Evidence grounding from test.jsonl
# ---------------------------------------------------------------------------
def extract_input_actions(messages: list[dict[str, str]]) -> set[str]:
    actions: set[str] = set()
    for m in messages:
        if m.get("role") != "user":
            continue
        text = m.get("content", "")
        idx = text.find("[{")
        if idx < 0:
            continue
        depth = 0
        end = -1
        for k, ch in enumerate(text[idx:], idx):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = k + 1
                    break
        if end < 0:
            continue
        try:
            arr = json.loads(text[idx:end])
            for ev in arr:
                if isinstance(ev, dict) and isinstance(ev.get("Action"), str):
                    actions.add(ev["Action"])
        except Exception:
            pass
    return actions


def extract_pred_actions(obj: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(obj, dict):
        return out
    for it in obj.get("interests", []) or []:
        if not isinstance(it, dict):
            continue
        for t in it.get("topics", []) or []:
            if not isinstance(t, dict):
                continue
            for ev in t.get("evidence", []) or []:
                if isinstance(ev, dict) and isinstance(ev.get("action"), str):
                    out.append(ev["action"])
    return out


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> list[dict[str, Any]]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def stats(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0}
    xs_sorted = sorted(xs)
    return {
        "n": len(xs),
        "mean": round(statistics.mean(xs), 4),
        "median": round(statistics.median(xs), 4),
        "p10": round(xs_sorted[max(0, int(0.1 * len(xs)) - 1)], 4),
        "p90": round(xs_sorted[min(len(xs) - 1, int(0.9 * len(xs)))], 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--test-jsonl", required=True,
                    help="original test.jsonl (for evidence grounding)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--teacher-mode", action="store_true",
                    help="when set, treat predictions identically to reference "
                         "(use this when feeding teacher.jsonl)")
    ap.add_argument("--name-match-threshold", type=float, default=0.5)
    args = ap.parse_args()

    preds = load_jsonl(args.predictions)
    test_records = load_jsonl(args.test_jsonl)

    # index test records by (user_id, delta_index)
    test_index: dict[tuple[str, int], dict[str, Any]] = {}
    for r in test_records:
        md = r.get("metadata", {}) or {}
        key = (str(md.get("user_id", "")), int(md.get("delta_index", -1)))
        test_index[key] = r

    n = len(preds)
    n_err = 0
    n_parse_ok = 0
    n_schema_ok = 0
    n_interests_pred: list[float] = []
    n_interests_gold: list[float] = []
    n_topics_pred: list[float] = []
    n_topics_gold: list[float] = []
    name_iou: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    matched_topic_ious: list[float] = []
    matched_name_rouges: list[float] = []
    evidence_fid_rates: list[float] = []
    len_ratios: list[float] = []
    per_record_rows: list[dict[str, Any]] = []

    for p in preds:
        if p.get("error"):
            n_err += 1
            per_record_rows.append({
                "user_id": p.get("user_id", ""),
                "delta_index": p.get("delta_index", -1),
                "error": p["error"],
            })
            continue
        pred_text = p["prediction"] if not args.teacher_mode else p["reference"]
        gold_text = p["reference"]
        pred_obj = safe_json_loads(pred_text)
        gold_obj = safe_json_loads(gold_text)
        parse_ok = pred_obj is not None
        if parse_ok:
            n_parse_ok += 1
        schema_ok = parse_ok and validate_schema(pred_obj)
        if schema_ok:
            n_schema_ok += 1

        # length ratio (always)
        len_ratios.append(len(pred_text) / max(1, len(gold_text)))

        if not parse_ok or gold_obj is None:
            per_record_rows.append({
                "user_id": p.get("user_id", ""),
                "delta_index": p.get("delta_index", -1),
                "parse_ok": parse_ok,
                "schema_ok": schema_ok,
                "n_interests_pred": None,
                "n_interests_gold": None,
                "precision": None, "recall": None, "f1": None,
                "evidence_fidelity": None,
            })
            continue

        pred_interests = pred_obj.get("interests") or []
        gold_interests = gold_obj.get("interests") or []
        if not isinstance(pred_interests, list):
            pred_interests = []
        if not isinstance(gold_interests, list):
            gold_interests = []

        n_interests_pred.append(len(pred_interests))
        n_interests_gold.append(len(gold_interests))
        for it in pred_interests:
            if isinstance(it, dict):
                n_topics_pred.append(len(it.get("topics") or []))
        for it in gold_interests:
            if isinstance(it, dict):
                n_topics_gold.append(len(it.get("topics") or []))

        pred_names = [it.get("interest_name", "") for it in pred_interests if isinstance(it, dict)]
        gold_names = [it.get("interest_name", "") for it in gold_interests if isinstance(it, dict)]

        # aggregate set IoU on union token sets
        pset, gset = set(), set()
        for nm in pred_names: pset |= norm_tokens(nm)
        for nm in gold_names: gset |= norm_tokens(nm)
        rec_iou = jaccard(pset, gset)
        name_iou.append(rec_iou)

        matches, prec, rec, f1 = greedy_match(
            pred_names, gold_names, threshold=args.name_match_threshold,
        )
        precisions.append(prec); recalls.append(rec); f1s.append(f1)

        for pi, gi, s in matches:
            if s < args.name_match_threshold:
                continue
            pi_topics = set()
            for t in pred_interests[pi].get("topics") or []:
                if isinstance(t, dict) and isinstance(t.get("topic"), str):
                    pi_topics |= norm_tokens(t["topic"])
            gi_topics = set()
            for t in gold_interests[gi].get("topics") or []:
                if isinstance(t, dict) and isinstance(t.get("topic"), str):
                    gi_topics |= norm_tokens(t["topic"])
            matched_topic_ious.append(jaccard(pi_topics, gi_topics))
            matched_name_rouges.append(
                rouge_l(pred_interests[pi].get("interest_name", ""),
                        gold_interests[gi].get("interest_name", ""))
            )

        # Evidence fidelity
        key = (str(p.get("user_id", "")), int(p.get("delta_index", -1)))
        src = test_index.get(key)
        evid_fid = None
        if src is not None:
            input_actions = extract_input_actions(src.get("messages", []))
            pred_actions = extract_pred_actions(pred_obj)
            if pred_actions:
                hits = sum(1 for a in pred_actions if a in input_actions)
                evid_fid = hits / len(pred_actions)
                evidence_fid_rates.append(evid_fid)

        per_record_rows.append({
            "user_id": p.get("user_id", ""),
            "delta_index": p.get("delta_index", -1),
            "parse_ok": parse_ok,
            "schema_ok": schema_ok,
            "n_interests_pred": len(pred_interests),
            "n_interests_gold": len(gold_interests),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "evidence_fidelity": (round(evid_fid, 4) if evid_fid is not None else None),
            "len_ratio": round(len(pred_text) / max(1, len(gold_text)), 3),
        })

    report = {
        "model_tag": args.model_tag,
        "n_predictions": n,
        "n_errored": n_err,
        "name_match_threshold": args.name_match_threshold,
        "structural": {
            "parse_rate": round(n_parse_ok / max(n, 1), 4),
            "schema_rate": round(n_schema_ok / max(n, 1), 4),
        },
        "counts": {
            "n_interests_pred": stats(n_interests_pred),
            "n_interests_gold": stats(n_interests_gold),
            "n_topics_per_interest_pred": stats(n_topics_pred),
            "n_topics_per_interest_gold": stats(n_topics_gold),
        },
        "interest_matching": {
            "set_iou": stats(name_iou),
            "precision": stats(precisions),
            "recall": stats(recalls),
            "f1": stats(f1s),
        },
        "matched_pairs": {
            "topic_set_iou": stats(matched_topic_ious),
            "name_rougeL": stats(matched_name_rouges),
        },
        "evidence_fidelity": stats(evidence_fid_rates),
        "length_ratio": stats(len_ratios),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    # Per-record CSV next to the JSON
    csv_path = out_path.with_suffix(".per_record.csv")
    fields = ["user_id", "delta_index", "parse_ok", "schema_ok",
              "n_interests_pred", "n_interests_gold",
              "precision", "recall", "f1", "evidence_fidelity",
              "len_ratio", "error"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in per_record_rows:
            w.writerow(r)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    logger.info("Wrote %s and %s", out_path, csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
