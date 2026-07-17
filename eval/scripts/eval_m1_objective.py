"""eval_m1_objective.py — Objective metrics for layer0 SFT (no LLM judge).

Reads predictions.jsonl produced by generate_outputs.py (group B or C) and
compares each prediction against the gpt-5.2 reference (stored in the same
file as `reference`). For group A (teacher ceiling), pass --teacher-mode and
the reference is also used as the "prediction" — this yields the upper bound
(parse rate 100%, agreement 100%) and exposes any malformed teacher rows.

Metrics computed (per LAYER0_SFT_PLAN.md §6.4 M1):
  - json_parse_rate              : fraction of predictions that parse cleanly
  - output_length_match_rate     : len(pred.result) == input_n_signals
  - row_coverage_rate            : set(row in pred.result) == {0..N-1}
  - should_filter_agreement      : matched-rows-only mean agreement
  - cohens_kappa                 : same, controlled for class imbalance
  - macro_per_record_agreement   : per-record agreement averaged over records

Output: m1_<model_tag>.json with the above + a per-record CSV breakdown.

Usage:
    python scripts/eval_m1_objective.py \
        --predictions eval_results/.../predictions/sft.jsonl \
        --output      eval_results/.../m1_objective/sft.json \
        --model-tag   sft
    # Group A (teacher) — reference as pseudo-prediction:
    python scripts/eval_m1_objective.py \
        --predictions eval_results/.../predictions/sft.jsonl \
        --teacher-mode \
        --output      eval_results/.../m1_objective/teacher.json \
        --model-tag   teacher
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# JSON parsing (mirrors maiprofilev3dev.modules.llm_client.safe_json_loads,
# kept self-contained to avoid pulling that module's config dependencies).
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_THINK_CLOSE = "</think>"


def safe_json_loads(text: str) -> Any:
    s = (text or "").strip()
    if not s:
        return None
    # If a thinking model emitted a closed <think>...</think> reasoning block,
    # the answer JSON appears AFTER it. Stripping the reasoning prevents the
    # "first { in text" heuristic below from latching onto draft braces that
    # the model wrote inside its own reasoning.
    close = s.rfind(_THINK_CLOSE)
    if close >= 0:
        tail = s[close + len(_THINK_CLOSE):].strip()
        if tail:
            s = tail
    if s.startswith("```"):
        s = _FENCE_RE.sub("", s).strip()
    # Direct attempt
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Strip everything before first `{` / `[`
    start = min((s.find(c) for c in "[{" if s.find(c) >= 0), default=-1)
    if start > 0:
        s = s[start:]
    # Try to close trailing braces/brackets
    for closer in ("", "]", "}", "]}", "}]"):
        try:
            return json.loads(s + closer)
        except json.JSONDecodeError:
            continue
    return None


def extract_result_list(parsed: Any) -> list[dict[str, Any]] | None:
    """The teacher's schema is {"result":[...]}. Be tolerant of stray keys."""
    if parsed is None:
        return None
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for k in ("result", "results", "signals", "data"):
            v = parsed.get(k)
            if isinstance(v, list):
                return v
    return None


# ---------------------------------------------------------------------------
# Cohen's kappa (binary, two raters). Avoid sklearn dependency.
# ---------------------------------------------------------------------------
def cohens_kappa_binary(refs: list[bool], preds: list[bool]) -> float:
    if not refs:
        return 0.0
    n = len(refs)
    # observed agreement
    po = sum(1 for r, p in zip(refs, preds) if r == p) / n
    # marginals
    pr_true = sum(refs) / n
    pp_true = sum(preds) / n
    pe = pr_true * pp_true + (1 - pr_true) * (1 - pp_true)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


# ---------------------------------------------------------------------------
def evaluate(predictions_path: str, teacher_mode: bool, model_tag: str) -> dict[str, Any]:
    records = []
    with open(predictions_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    parse_ok = 0
    length_match = 0
    coverage_ok = 0
    n_total = 0          # records with parseable, length-matched pred (for agreement)
    n_signals_total = 0  # total signals contributing to agreement
    ref_filters: list[bool] = []
    pred_filters: list[bool] = []
    per_record_agreements: list[float] = []
    csv_rows: list[dict[str, Any]] = []

    for rec in records:
        n_in = int(rec.get("input_n_signals") or 0)
        ref_raw = rec["reference"]
        # In teacher mode we evaluate the reference against itself — this
        # measures the ceiling of M1 (basically tests teacher-side JSON sanity).
        pred_raw = ref_raw if teacher_mode else rec.get("prediction", "")

        ref_parsed = extract_result_list(safe_json_loads(ref_raw))
        pred_parsed = extract_result_list(safe_json_loads(pred_raw))

        row_parse_ok = pred_parsed is not None
        if row_parse_ok:
            parse_ok += 1
        row_len_ok = row_parse_ok and len(pred_parsed) == n_in
        if row_len_ok:
            length_match += 1
        row_cov_ok = False
        if row_len_ok:
            try:
                pred_rows = {int(r.get("row", -1)) for r in pred_parsed}
                row_cov_ok = pred_rows == set(range(n_in))
            except (TypeError, ValueError):
                row_cov_ok = False
            if row_cov_ok:
                coverage_ok += 1

        # Agreement requires BOTH sides parseable AND aligned by row
        row_agreement: float | None = None
        if (ref_parsed is not None and pred_parsed is not None
                and row_len_ok and len(ref_parsed) == n_in):
            ref_by_row = {int(r.get("row", -1)): bool(r.get("should_filter", False))
                          for r in ref_parsed if "row" in r}
            pred_by_row = {int(r.get("row", -1)): bool(r.get("should_filter", False))
                           for r in pred_parsed if "row" in r}
            common = sorted(set(ref_by_row) & set(pred_by_row))
            if common:
                rec_matches = 0
                for row in common:
                    rf = ref_by_row[row]
                    pf = pred_by_row[row]
                    ref_filters.append(rf)
                    pred_filters.append(pf)
                    if rf == pf:
                        rec_matches += 1
                row_agreement = rec_matches / len(common)
                per_record_agreements.append(row_agreement)
                n_total += 1
                n_signals_total += len(common)

        csv_rows.append({
            "record_idx": rec.get("record_idx"),
            "user_id": rec.get("user_id"),
            "n_signals_in": n_in,
            "parse_ok": int(row_parse_ok),
            "length_match": int(row_len_ok),
            "row_coverage": int(row_cov_ok),
            "should_filter_agreement": (round(row_agreement, 4)
                                        if row_agreement is not None else ""),
        })

    n_rec = len(records)
    overall_agreement = (
        sum(1 for r, p in zip(ref_filters, pred_filters) if r == p) / len(ref_filters)
        if ref_filters else 0.0
    )
    kappa = cohens_kappa_binary(ref_filters, pred_filters)
    return {
        "model_tag": model_tag,
        "teacher_mode": teacher_mode,
        "n_records": n_rec,
        "n_signals_compared": len(ref_filters),
        "json_parse_rate": round(parse_ok / n_rec, 4) if n_rec else 0.0,
        "output_length_match_rate": round(length_match / n_rec, 4) if n_rec else 0.0,
        "row_coverage_rate": round(coverage_ok / n_rec, 4) if n_rec else 0.0,
        "should_filter_agreement_micro": round(overall_agreement, 4),
        "should_filter_agreement_macro_per_record":
            round(sum(per_record_agreements) / len(per_record_agreements), 4)
            if per_record_agreements else 0.0,
        "cohens_kappa": round(kappa, 4),
        # Reference class balance — useful sanity / interpretation aid
        "reference_filter_rate": round(sum(ref_filters) / len(ref_filters), 4)
            if ref_filters else 0.0,
        "prediction_filter_rate": round(sum(pred_filters) / len(pred_filters), 4)
            if pred_filters else 0.0,
        "per_record_csv": None,  # filled by caller (relative path)
    }, csv_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", required=True, help="m1 summary JSON path")
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--teacher-mode", action="store_true",
                    help="evaluate the reference column against itself (for group A)")
    args = ap.parse_args()

    summary, csv_rows = evaluate(args.predictions, args.teacher_mode, args.model_tag)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out.with_suffix(".per_record.csv")
    summary["per_record_csv"] = csv_path.name
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        if csv_rows:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)

    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
