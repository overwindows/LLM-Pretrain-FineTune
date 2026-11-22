#!/usr/bin/env python3
"""Summarize results/all_runs.csv and compare to A100 baselines.

Prints a ranked comparison table and writes results/summary.md.

Usage:
    python3 analyze_results.py [--csv results/all_runs.csv]
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

# ---------------------------------------------------------------------------
# A100 reference results — old A100 ablation study (EXPERIMENT_PLAN_ABLATION_STUDY.md)
# OLD: A100 80GB with AsyncLLMEngine (examples/run_inference_configurable.py)
# NEW: A100 80GB with LLM engine (benchmarks/gemma4_moe_benchmarks/bench_experiment.py)
#
# These are provided for comparison context only. The old study used a DIFFERENT
# experiment numbering; the table below maps old results to the closest new
# experiment IDs where the configuration is equivalent.
#
# Old study used: output_len=1024, max_model_len=32768, layer1_delta_1k_test.txt
# New study uses: output_len=8192, max_model_len=24576, sc1_delta_v2.jsonl (1000 prompts)
# Numbers are NOT directly comparable — ratios are more informative than absolutes.
#
# Old IDs → new ID mapping (best-effort):
#   OLD E001 BF16 baseline                → new E001
#   OLD E002 +FP8 weights                 → new E002
#   OLD E006 +MTP k=5 (with CUDA graphs)  → new E005 / E006
#   OLD E007 text-only                    → new E006
#   OLD E011 gpu_mem=0.80 [OLD BEST]      → new E010
# ---------------------------------------------------------------------------
A100_RESULTS: dict[str, dict] = {
    # New ID  : (old output_tps, description of closest old experiment)
    "E001": dict(output_tps=654.9,  label="BF16 baseline — old A100 study E001"),
    "E002": dict(output_tps=692.1,  label="+FP8 weights — old A100 study E002"),
    "E003": dict(output_tps=None,   label="FP8 KV (fp8_e4m3) — FAIL on A100 (no old ref)"),
    "E004": dict(output_tps=768.4,  label="+CUDA graphs — old A100 study E005"),
    "E005": dict(output_tps=974.6,  label="+MTP k=5 — old A100 study E006"),
    "E006": dict(output_tps=957.3,  label="+text-only — old A100 study E007"),
    "E007": dict(output_tps=None,   label="batch sweep mns=64 (no old ref)"),
    "E008": dict(output_tps=968.4,  label="batch sweep mns=192 — old A100 study E008"),
    "E009": dict(output_tps=970.9,  label="batch sweep mns=256 — old A100 study E009"),
    "E010": dict(output_tps=983.7,  label="gpu_mem=0.80 — old A100 study E011 [OLD BEST]"),
    "E011": dict(output_tps=None,   label="gpu_mem=0.95 (no old ref)"),
    "E012": dict(output_tps=781.4,  label="no MTP at optimal — old A100 study E013"),
    "E013": dict(output_tps=None,   label="no CUDA graphs at optimal (no old ref)"),
    "E014": dict(output_tps=None,   label="BF16 weights at optimal (no old ref)"),
    "E015": dict(output_tps=477.1,  label="BF16 ref text-only no opts — old A100 study E015"),
    "E016": dict(output_tps=None,   label="BF16 + CUDA graphs only (no old ref)"),
}


def load_csv(csv_path: Path) -> dict[str, list[dict]]:
    """Load CSV and group rows by (exp_id, scenario)."""
    groups: dict[str, list[dict]] = {}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = f"{row['exp_id']}_{row['scenario']}"
            groups.setdefault(key, []).append(row)
    return groups


def summarize_group(rows: list[dict]) -> dict:
    """Compute mean ± stdev for numeric metrics across reps."""
    def nums(field: str) -> list[float]:
        vals = []
        for r in rows:
            v = r.get(field, "")
            if v not in ("", "None", None):
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
        return vals

    def stat(field: str):
        vals = nums(field)
        if not vals:
            return None, None
        mean = statistics.mean(vals)
        stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return mean, stdev

    first = rows[0]
    return {
        "exp_id": first["exp_id"],
        "label": first["label"],
        "scenario": first["scenario"],
        "reps": len(rows),
        "output_tps_mean": stat("output_tps")[0],
        "output_tps_stdev": stat("output_tps")[1],
        "total_tps_mean": stat("total_tps")[0],
        "requests_per_second_mean": stat("requests_per_second")[0],
        "elapsed_mean": stat("elapsed_time")[0],
        "out_len_mean": stat("out_len_mean")[0],
        "quantization": first["quantization"],
        "attention_backend": first["attention_backend"],
        "enforce_eager": first["enforce_eager"],
        "mtp": first["mtp"],
        "mtp_k": first["mtp_k"],
        "max_num_seqs": first["max_num_seqs"],
        "gpu_memory_utilization": first["gpu_memory_utilization"],
        "model_variant": first["model_variant"],
    }


def render_table(summaries: list[dict], scenario: str) -> str:
    # Sort by exp_id
    rows = [s for s in summaries if s["scenario"] == scenario]
    rows.sort(key=lambda r: r["exp_id"])
    if not rows:
        return f"_No data for scenario {scenario}_\n"

    # Find baseline (E001)
    current_base = next((r["output_tps_mean"] for r in rows if r["exp_id"] == "E001"), None)
    old_a100_base = A100_RESULTS.get("E001", {}).get("output_tps")

    lines: list[str] = []
    lines.append(f"### Scenario: {scenario}\n")
    lines.append(
        f"| Exp | Label | out tok/s (A100 80GB) | ±σ | vs E001 | old A100 ref | vs old A100 |"
        f" Backend | eager | MTP | seqs | mem% |"
    )
    lines.append("|-----|-------|:---:|:---:|:---:|:---:|:---:|---------|-------|-----|------|------|")

    for r in rows:
        eid = r["exp_id"]
        otps = r["output_tps_mean"]
        otps_s = r["output_tps_stdev"] or 0.0
        otps_str = f"{otps:.1f}" if otps else "FAIL"
        otps_sig = f"{otps_s:.1f}" if otps else "—"
        # vs current baseline
        vs_base = f"{otps/current_base:.3f}×" if (otps and current_base) else "—"
        # old A100 reference (from examples/ study)
        old_a100 = A100_RESULTS.get(eid, {})
        old_a100_tps = old_a100.get("output_tps")
        old_a100_str = f"{old_a100_tps:.1f}" if old_a100_tps else "FAIL/NA"
        # current vs old A100
        if otps and old_a100_tps:
            ratio = otps / old_a100_tps
            vs_old_a100 = f"{ratio:.2f}×"
        else:
            vs_old_a100 = "—"

        backend = r["attention_backend"] or "?"
        eager = "✓" if str(r["enforce_eager"]).lower() in ("true", "1") else "✗ (CG)"
        mtp_str = f"✓ k={r['mtp_k']}" if str(r["mtp"]).lower() in ("true", "1") else "✗"
        seqs = r["max_num_seqs"]
        mem = r["gpu_memory_utilization"]

        lines.append(
            f"| {eid} | {r['label'][:55]} | {otps_str} | {otps_sig} | {vs_base} "
            f"| {old_a100_str} | {vs_old_a100} | {backend} | {eager} | {mtp_str} | {seqs} | {mem} |"
        )

    # Best row
    best = max((r for r in rows if r["output_tps_mean"]), key=lambda r: r["output_tps_mean"], default=None)
    if best:
        lines.append(f"\n**Best A100 80GB result**: {best['exp_id']} — {best['output_tps_mean']:.1f} output tok/s")
        if current_base:
            lines.append(f"  Overall A100 80GB speedup vs BF16 baseline: {best['output_tps_mean']/current_base:.3f}×")
        old_a100_best_tps = A100_RESULTS.get("E010", {}).get("output_tps")  # E010=gpu_mem=0.80, old A100 best
        if old_a100_best_tps and best["output_tps_mean"]:
            lines.append(f"  A100 80GB best vs old A100 best (E010={old_a100_best_tps}): "
                         f"{best['output_tps_mean']/old_a100_best_tps:.2f}×")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/all_runs.csv")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run run_experiments.sh first.")
        raise SystemExit(1)

    groups = load_csv(csv_path)
    if not groups:
        print("ERROR: CSV is empty or malformed.")
        raise SystemExit(1)

    summaries = [summarize_group(rows) for rows in groups.values()]
    scenarios = sorted({s["scenario"] for s in summaries})

    md_lines: list[str] = [
        "# Gemma 4 MoE FP8 — Ablation Benchmark Summary",
        "",
        "Generated by `analyze_results.py` from `results/all_runs.csv`.",
        "",
        "## Old A100 80GB reference (AsyncEngine, from examples/)",
        "",
        "_Previous ablation study using `AsyncLLMEngine` (examples/run_inference_configurable.py)_",
        "",
        "| Exp | A100 out tok/s | label |",
        "|-----|---:|-------|",
    ]
    for eid in sorted(A100_RESULTS.keys()):
        a = A100_RESULTS[eid]
        tps = f"{a['output_tps']:.1f}" if a["output_tps"] else "FAIL"
        md_lines.append(f"| {eid} | {tps} | {a['label']} |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## New A100 80GB Results (LLM engine, this study)")
    md_lines.append("")
    md_lines.append("_Current ablation study using `vllm.LLM` (benchmarks/gemma4_moe_benchmarks/bench_experiment.py)_")
    md_lines.append("")

    for sc in scenarios:
        md_lines.append(render_table(summaries, sc))

    # Key deltas summary
    md_lines.append("---")
    md_lines.append("## Key configuration contribution (A100 80GB, mean across reps)")
    md_lines.append("")
    md_lines.append("Estimated contribution of each optimization layer, from ablation pairs:")
    md_lines.append("")

    def delta_str(sc: str, eid_on: str, eid_off: str, label: str):
        def get_tps(eid: str) -> float | None:
            key = f"{eid}_{sc}"
            s = next((x for x in summaries if f"{x['exp_id']}_{x['scenario']}" == key), None)
            return s["output_tps_mean"] if s else None

        on_tps = get_tps(eid_on)
        off_tps = get_tps(eid_off)
        if on_tps and off_tps:
            delta = on_tps - off_tps
            pct = (delta / off_tps) * 100
            return f"  {label:<40} : {delta:+.1f} tok/s ({pct:+.1f}%)"
        return f"  {label:<40} : data missing ({eid_on} or {eid_off})"

    for sc in scenarios:
        md_lines.append(f"**{sc}:**")
        # Group A — reproduce REPRODUCE_PRODSHAPE baseline
        md_lines.append(delta_str(sc, "E002", "E001", "FP8 weights vs BF16 (E002-E001)"))
        # Group B — incremental optimizations
        md_lines.append(delta_str(sc, "E004", "E002", "CUDA graphs vs eager (E004-E002)"))
        md_lines.append(delta_str(sc, "E005", "E004", "MTP k=5 (E005-E004)"))
        md_lines.append(delta_str(sc, "E006", "E005", "text-only model (E006-E005)"))
        # Group C — batch size sweep
        md_lines.append(delta_str(sc, "E007", "E006", "batch mns=64 vs 128 (E007-E006)"))
        md_lines.append(delta_str(sc, "E008", "E006", "batch mns=192 vs 128 (E008-E006)"))
        md_lines.append(delta_str(sc, "E009", "E006", "batch mns=256 vs 128 (E009-E006)"))
        # Group D — gpu_mem sweep
        md_lines.append(delta_str(sc, "E010", "E006", "gpu_mem=0.80 vs 0.90 (E010-E006)"))
        md_lines.append(delta_str(sc, "E011", "E006", "gpu_mem=0.95 vs 0.90 (E011-E006)"))
        # Group E — isolation
        md_lines.append(delta_str(sc, "E012", "E006", "disable MTP at optimal (E012-E006) — negative expected"))
        md_lines.append(delta_str(sc, "E013", "E006", "disable CUDA graphs at optimal (E013-E006) — negative expected"))
        md_lines.append(delta_str(sc, "E014", "E006", "BF16 weights at optimal (E014-E006) — isolates FP8"))
        md_lines.append("")

    md_content = "\n".join(md_lines)
    print(md_content)

    out_md = Path("results/summary.md")
    out_md.parent.mkdir(exist_ok=True)
    out_md.write_text(md_content, encoding="utf-8")
    print(f"\nWrote {out_md}")


if __name__ == "__main__":
    main()
