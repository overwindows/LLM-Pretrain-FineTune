"""aggregate_eval_report.py — Build the final markdown comparison report.

Reads m1_*.json and m2_*.json for groups (teacher, sft, zero_shot) and writes
REPORT.md with side-by-side tables and pass/fail marks against the success
criteria in LAYER0_SFT_PLAN.md §11.

Usage:
    python scripts/aggregate_eval_report.py --config configs/eval/layer0_signal.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml


GROUP_ORDER = ["teacher", "sft", "zero_shot"]
GROUP_LABEL = {"teacher": "A · Teacher (gpt-5.2)",
               "sft": "B · SFT (Qwen3-4B+SFT)",
               "zero_shot": "C · Zero-shot Qwen3-4B"}


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _row(label: str, values: list[str]) -> str:
    return "| " + " | ".join([label] + values) + " |"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    m1_dir = Path(cfg["m1_dir"])
    m2_dir = Path(cfg["m2_dir"])
    th = cfg["thresholds"]

    m1 = {g: _load(m1_dir / f"{g}.json") for g in GROUP_ORDER}
    m2 = {g: _load(m2_dir / f"m2_{g}.json") for g in GROUP_ORDER}

    def cell(d: dict | None, key: str, fmt: str = "{:.4f}") -> str:
        if d is None or d.get(key) in (None, ""):
            return "—"
        v = d[key]
        try:
            return fmt.format(v) if isinstance(v, (int, float)) else str(v)
        except (ValueError, TypeError):
            return str(v)

    lines: list[str] = []
    lines.append(f"# Layer0 SFT Evaluation Report")
    lines.append(f"")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append(f"")
    lines.append(f"Test set: `{cfg['test_jsonl']}` "
                 f"(n={(m1['teacher'] or m1['sft'] or m1['zero_shot'] or {}).get('n_records', '?')})")
    lines.append(f"")

    # --- M1 table ---
    lines.append("## M1 — Objective Metrics (no LLM judge)")
    lines.append("")
    headers = [GROUP_LABEL[g] for g in GROUP_ORDER]
    lines.append(_row("Metric", headers))
    lines.append("|" + "|".join(["---"] * (len(headers) + 1)) + "|")
    for key, label, fmt in [
        ("n_records", "n records", "{:d}"),
        ("n_signals_compared", "n signals compared", "{:d}"),
        ("json_parse_rate", "JSON parse rate", "{:.4f}"),
        ("output_length_match_rate", "output length match", "{:.4f}"),
        ("row_coverage_rate", "row coverage", "{:.4f}"),
        ("should_filter_agreement_micro", "should_filter agreement (micro)", "{:.4f}"),
        ("should_filter_agreement_macro_per_record", "should_filter agreement (macro)", "{:.4f}"),
        ("cohens_kappa", "Cohen's κ (filter decision)", "{:.4f}"),
        ("reference_filter_rate", "reference filter rate", "{:.4f}"),
        ("prediction_filter_rate", "prediction filter rate", "{:.4f}"),
    ]:
        lines.append(_row(label, [cell(m1[g], key, fmt) for g in GROUP_ORDER]))

    # --- M2 table ---
    lines.append("")
    lines.append("## M2 — LLM Judge (GPT-5.1, scores in [1,10])")
    lines.append("")
    lines.append(_row("Metric", headers))
    lines.append("|" + "|".join(["---"] * (len(headers) + 1)) + "|")
    for key, label, fmt in [
        ("n_judged_records", "n judged records", "{:d}"),
        ("accuracy", "accuracy", "{:.3f}"),
        ("precision", "precision (filter=true mean DQ)", "{:.3f}"),
        ("recall", "recall (filter=false mean DQ)", "{:.3f}"),
        ("intent_accuracy", "intent_accuracy", "{:.3f}"),
        ("consistency", "consistency", "{:.3f}"),
        ("n_unjudgeable_parse_error", "unjudgeable: parse error", "{:d}"),
        ("n_unjudgeable_row_misalignment", "unjudgeable: row mismatch", "{:d}"),
    ]:
        lines.append(_row(label, [cell(m2[g], key, fmt) for g in GROUP_ORDER]))

    # --- Success criteria (LAYER0_SFT_PLAN.md §11) ---
    lines.append("")
    lines.append("## Success Criteria (vs LAYER0_SFT_PLAN.md §11)")
    lines.append("")
    sft_m1 = m1.get("sft") or {}
    sft_m2 = m2.get("sft") or {}
    tch_m2 = m2.get("teacher") or {}
    zs_m1 = m1.get("zero_shot") or {}
    zs_m2 = m2.get("zero_shot") or {}

    def mark(ok: bool) -> str:
        return "✅" if ok else "❌"

    def pct(numer, denom):
        if not isinstance(numer, (int, float)) or not isinstance(denom, (int, float)) or denom == 0:
            return "—"
        return f"{numer / denom:.2%}"

    rows = []
    # Criterion: JSON parse rate ≥ 0.99
    jpr = sft_m1.get("json_parse_rate")
    rows.append((f"SFT JSON parse rate ≥ {th['json_parse_rate_min']}",
                 f"{jpr}", mark(isinstance(jpr, (int, float)) and jpr >= th["json_parse_rate_min"])))
    # Criterion: should_filter agreement ≥ 0.90
    sfa = sft_m1.get("should_filter_agreement_micro")
    rows.append((f"SFT should_filter agreement ≥ {th['should_filter_agreement_min']}",
                 f"{sfa}", mark(isinstance(sfa, (int, float)) and sfa >= th["should_filter_agreement_min"])))
    # Criterion: SFT M2 accuracy ≥ 0.85 × teacher's M2 accuracy
    sa = sft_m2.get("accuracy"); ta = tch_m2.get("accuracy")
    ok_a = (isinstance(sa, (int, float)) and isinstance(ta, (int, float))
            and sa >= 0.85 * ta)
    rows.append(("SFT M2 accuracy ≥ 0.85 × teacher accuracy",
                 f"{sa} vs {ta} (ratio {pct(sa, ta)})", mark(ok_a)))
    # Criterion: SFT beats zero-shot on every M1+M2 metric we compare
    beats = True
    detail = []
    for src, key in (("M1", "should_filter_agreement_micro"),
                     ("M1", "cohens_kappa"),
                     ("M1", "json_parse_rate"),
                     ("M2", "accuracy"),
                     ("M2", "consistency")):
        d_sft = sft_m1 if src == "M1" else sft_m2
        d_zs = zs_m1 if src == "M1" else zs_m2
        s = d_sft.get(key); z = d_zs.get(key)
        if isinstance(s, (int, float)) and isinstance(z, (int, float)):
            if s <= z:
                beats = False
                detail.append(f"{src}.{key}: sft={s} <= zs={z}")
    rows.append(("SFT beats zero-shot on all M1+M2 metrics",
                 ("OK" if beats else "; ".join(detail)), mark(beats)))

    lines.append(_row("Criterion", ["Result", "Pass?"]))
    lines.append("|---|---|---|")
    for label, result, ok in rows:
        lines.append(_row(label, [result, ok]))

    # --- Footer ---
    lines.append("")
    lines.append("## Raw outputs")
    lines.append("")
    for g in GROUP_ORDER:
        lines.append(f"- {GROUP_LABEL[g]}: "
                     f"`m1_objective/{g}.json`, `m2_judge/m2_{g}.json` "
                     f"(per-user: `m2_judge/m2_{g}.per_user.jsonl`)")

    out_path = Path(cfg["report_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[report] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
