#!/usr/bin/env python3
"""Custom offline throughput driver using `vllm.LLM` directly.

Same engine as `vllm bench throughput`, but lets us use our real prompts
(any length) and capture per-request stats for stdev / percentiles.

Workflow:
  - For each (scenario, max_num_seqs) config, instantiate a fresh LLM.
  - For each rep in 1..reps, call llm.generate(...) on the prompts (one rep at a time).
  - Time the wall clock around generate() to get the run-level throughput.
  - Aggregate per-request output lengths from the RequestOutput list.
  - Append a CSV row, write a per-run JSON, print mean +/- stdev per config.

Key knobs match the vllm-bench-throughput defaults we discussed.
"""
from __future__ import annotations
import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_OUT_DIR = Path("bench_results")

CSV_FIELDS = [
    "ts", "scenario", "dataset", "quantization", "kv_cache_dtype",
    "num_prompts", "output_len_cap",
    "max_model_len", "max_num_batched_tokens", "gpu_mem_util",
    "max_num_seqs", "rep", "seed",
    "chunk_index", "chunk_total",
    "elapsed_time",
    "requests_per_second",
    "prompt_tokens_total", "output_tokens_total", "total_tokens",
    "prompt_tps", "output_tps", "total_tps",
    "out_len_mean", "out_len_stdev", "out_len_p50", "out_len_p90", "out_len_max",
    "finish_stop", "finish_length", "finish_other",
]

SCENARIOS = {
    "sc1": dict(
        dataset="datasets/sc1_delta_v2.jsonl",
        num_prompts=1000,
        output_len=8192,
        max_model_len=24576,
        max_num_batched_tokens=16384,
        gpu_mem_util=0.95,
    ),
    "sc2": dict(
        dataset="datasets/sc2_personal_v2.jsonl",
        num_prompts=500,
        output_len=8192,
        max_model_len=49152,
        max_num_batched_tokens=16384,
        gpu_mem_util=0.95,
    ),
}


def ensure_csv_header(csv_path: Path):
    if not csv_path.exists():
        with csv_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_csv_row(csv_path: Path, row: dict):
    # Fill in any missing fields with empty string so older/newer schemas don't error.
    full = {k: row.get(k, "") for k in CSV_FIELDS}
    with csv_path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(full)


def load_prompts(dataset_path: str, n: int) -> list[str]:
    """Load up to n prompts from a JSONL dataset. n=0 (or None) means load all."""
    prompts = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompts.append(d["prompt"])
            if n and len(prompts) >= n:
                break
    return prompts


def render_chat(tok, raw_prompts: list[str]) -> list[str]:
    """Wrap each prompt as a single user-message chat and render via Gemma 4 chat template.
    Mirrors what `vllm bench throughput --dataset-name custom` would do internally.
    """
    out = []
    for p in raw_prompts:
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
        )
        out.append(text)
    return out


def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def _aggregate_outputs(outputs):
    """Aggregate per-request stats from a list of RequestOutput."""
    out_lens = []
    prompt_total = 0
    output_total = 0
    finish_counts = {"stop": 0, "length": 0, "other": 0}
    for o in outputs:
        prompt_total += len(o.prompt_token_ids)
        for comp in o.outputs:
            n_out = len(comp.token_ids)
            output_total += n_out
            out_lens.append(n_out)
            fr = (comp.finish_reason or "other").lower()
            if fr in finish_counts:
                finish_counts[fr] += 1
            else:
                finish_counts["other"] += 1
    return out_lens, prompt_total, output_total, finish_counts


def _row_from(*, scenario, cfg, max_num_seqs, rep, seed, chunk_index, chunk_total,
              elapsed, n_prompts, prompt_total, output_total, out_lens, finish_counts,
              quantization, kv_cache_dtype):
    total = prompt_total + output_total
    out_lens_sorted = sorted(out_lens)
    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario,
        "dataset": cfg["dataset"],
        "quantization": quantization or "",
        "kv_cache_dtype": kv_cache_dtype or "",
        "num_prompts": n_prompts,
        "output_len_cap": cfg["output_len"],
        "max_model_len": cfg["max_model_len"],
        "max_num_batched_tokens": cfg["max_num_batched_tokens"],
        "gpu_mem_util": cfg["gpu_mem_util"],
        "max_num_seqs": max_num_seqs,
        "rep": rep,
        "seed": seed,
        "chunk_index": chunk_index,
        "chunk_total": chunk_total,
        "elapsed_time": round(elapsed, 3),
        "requests_per_second": round(n_prompts / elapsed, 4) if elapsed > 0 else 0.0,
        "prompt_tokens_total": prompt_total,
        "output_tokens_total": output_total,
        "total_tokens": total,
        "prompt_tps": round(prompt_total / elapsed, 2) if elapsed > 0 else 0.0,
        "output_tps": round(output_total / elapsed, 2) if elapsed > 0 else 0.0,
        "total_tps": round(total / elapsed, 2) if elapsed > 0 else 0.0,
        "out_len_mean": round(statistics.mean(out_lens), 2) if out_lens else None,
        "out_len_stdev": round(statistics.stdev(out_lens), 2) if len(out_lens) > 1 else 0.0,
        "out_len_p50": int(percentile(out_lens_sorted, 0.5)) if out_lens else None,
        "out_len_p90": int(percentile(out_lens_sorted, 0.9)) if out_lens else None,
        "out_len_max": max(out_lens) if out_lens else None,
        "finish_stop": finish_counts["stop"],
        "finish_length": finish_counts["length"],
        "finish_other": finish_counts["other"],
    }


def run_one_config(*, scenario: str, cfg: dict, max_num_seqs: int, reps: int,
                   model: str, out_dir: Path, csv_path: Path,
                   quantization: str | None = None, kv_cache_dtype: str | None = None,
                   chunk_size: int = 0):
    """Build a fresh LLM with this max_num_seqs, run `reps` generations.

    If chunk_size > 0, each rep's prompt list is split into chunks of that size;
    a CSV row + per-chunk JSON is written after every chunk so partial progress
    survives a crash. The engine is not rebuilt between chunks.

    Returns the list of per-(rep, chunk) dicts.
    """
    # Defer heavy imports until after we know we'll use them.
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\n=== build LLM scenario={scenario} max_num_seqs={max_num_seqs} "
          f"quant={quantization or 'bf16'} kv={kv_cache_dtype or 'auto'} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    raw_prompts = load_prompts(cfg["dataset"], cfg["num_prompts"])
    prompts = render_chat(tok, raw_prompts)
    print(f"loaded {len(prompts)} prompts from {cfg['dataset']}", flush=True)

    llm_kwargs = dict(
        model=model,
        trust_remote_code=True,
        max_model_len=cfg["max_model_len"],
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=cfg["max_num_batched_tokens"],
        gpu_memory_utilization=cfg["gpu_mem_util"],
        seed=0,
    )
    if quantization:
        llm_kwargs["quantization"] = quantization
    if kv_cache_dtype:
        llm_kwargs["kv_cache_dtype"] = kv_cache_dtype

    t_engine = time.time()
    llm = LLM(**llm_kwargs)
    print(f"engine built in {time.time()-t_engine:.1f} s", flush=True)

    rows = []
    for rep in range(1, reps + 1):
        seed = rep  # different seed per rep
        sampling = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=cfg["output_len"],
            seed=seed,
            ignore_eos=False,
        )

        # Split into chunks (or one big chunk if chunk_size <= 0)
        if chunk_size and chunk_size > 0:
            chunks = [prompts[i:i + chunk_size] for i in range(0, len(prompts), chunk_size)]
        else:
            chunks = [prompts]
        chunk_total = len(chunks)

        for ci, chunk in enumerate(chunks, start=1):
            tag = f"{scenario}_mns{max_num_seqs}_rep{rep}"
            if chunk_total > 1:
                tag = f"{tag}_chunk{ci:03d}of{chunk_total:03d}"
            print(f"\n--- RUN {tag}  prompts={len(chunk)}  seed={seed} ---", flush=True)
            t0 = time.time()
            outputs = llm.generate(chunk, sampling, use_tqdm=True)
            elapsed = time.time() - t0

            out_lens, prompt_total, output_total, finish_counts = _aggregate_outputs(outputs)
            row = _row_from(
                scenario=scenario, cfg=cfg, max_num_seqs=max_num_seqs,
                rep=rep, seed=seed,
                chunk_index=ci if chunk_total > 1 else "",
                chunk_total=chunk_total if chunk_total > 1 else "",
                elapsed=elapsed, n_prompts=len(chunk),
                prompt_total=prompt_total, output_total=output_total,
                out_lens=out_lens, finish_counts=finish_counts,
                quantization=quantization, kv_cache_dtype=kv_cache_dtype,
            )
            append_csv_row(csv_path, row)
            rows.append(row)

            # Per-chunk (or per-rep, if not chunking) JSON for crash recovery
            per_run = out_dir / f"{tag}.json"
            with per_run.open("w") as f:
                json.dump({**row, "out_lens": out_lens}, f, indent=2)

            print(
                f"  elapsed={elapsed:.1f}s  req/s={row['requests_per_second']:.3f}  "
                f"out_tok/s={row['output_tps']:.0f}  total_tok/s={row['total_tps']:.0f}  "
                f"out_len(mean+/-sd)={row['out_len_mean']}+/-{row['out_len_stdev']}  "
                f"finish=stop:{finish_counts['stop']}/len:{finish_counts['length']}",
                flush=True,
            )

    # Print config summary (across all reps + chunks)
    summarize(scenario, max_num_seqs, rows)

    # Hand the engine to the GC before the next config
    del llm
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return rows


def summarize(scenario: str, mns: int, rows: list[dict]):
    if not rows:
        print(f"[SUMMARY] {scenario} mns={mns}: no successful runs", flush=True)
        return

    def m(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return None, None
        if len(vals) == 1:
            return vals[0], 0.0
        return statistics.mean(vals), statistics.stdev(vals)

    el_m, el_s = m("elapsed_time")
    r_m, r_s = m("requests_per_second")
    o_m, o_s = m("output_tps")
    t_m, t_s = m("total_tps")
    ol_m, ol_s = m("out_len_mean")
    n = len(rows)
    print(
        f"\n[SUMMARY] scenario={scenario}  max_num_seqs={mns}  reps={n}\n"
        f"  elapsed_time   : {el_m:.2f} +/- {el_s:.2f} s\n"
        f"  requests/sec   : {r_m:.4f} +/- {r_s:.4f}\n"
        f"  output tokens/s: {o_m:.2f} +/- {o_s:.2f}\n"
        f"  total tokens/s : {t_m:.2f} +/- {t_s:.2f}\n"
        f"  mean out_len   : {ol_m:.1f} +/- {ol_s:.1f}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=list(SCENARIOS.keys()))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-num-seqs", required=True,
                    help="Comma list, e.g. 64,128,256,512,1024")
    ap.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--dataset", default=None,
                    help="Override the scenario's dataset path.")
    ap.add_argument("--num-prompts", type=int, default=None,
                    help="Override num_prompts. 0 = load entire dataset.")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR),
                    help="Output directory for per-run JSON + all_runs.csv.")
    ap.add_argument("--quantization", default=None,
                    help="e.g. 'fp8'. Passed through to vllm.LLM.")
    ap.add_argument("--kv-cache-dtype", default=None,
                    help="e.g. 'fp8'. Passed through to vllm.LLM.")
    ap.add_argument("--chunk-size", type=int, default=0,
                    help="Split prompts into chunks of this size; CSV/JSON written "
                         "per chunk for crash recovery. 0 = no chunking.")
    ap.add_argument("--gpu-mem-util", type=float, default=None,
                    help="Override gpu_memory_utilization (default from scenario, 0.95). "
                         "Lower (e.g. 0.90) frees headroom to avoid OOM-killer on worker.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "all_runs.csv"
    ensure_csv_header(csv_path)

    cfg = dict(SCENARIOS[args.scenario])  # copy so we can override
    if args.dataset is not None:
        cfg["dataset"] = args.dataset
    if args.num_prompts is not None:
        cfg["num_prompts"] = args.num_prompts  # 0 -> all
    if args.gpu_mem_util is not None:
        cfg["gpu_mem_util"] = args.gpu_mem_util

    sweep = [int(x) for x in args.max_num_seqs.split(",")]
    failures = []  # list of (mns, exception_repr) — keep going across the sweep
                   # so an OOM at high mns doesn't abort the rest, but exit
                   # non-zero at the end so the caller (e.g. bash set -e) sees it.
    for mns in sweep:
        try:
            run_one_config(scenario=args.scenario, cfg=cfg, max_num_seqs=mns,
                           reps=args.reps, model=args.model,
                           out_dir=out_dir, csv_path=csv_path,
                           quantization=args.quantization,
                           kv_cache_dtype=args.kv_cache_dtype,
                           chunk_size=args.chunk_size)
        except Exception as e:
            print(f"!!! config max_num_seqs={mns} failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            failures.append((mns, repr(e)))

    if failures:
        print(f"\nDone with {len(failures)} failure(s):", flush=True)
        for mns, err in failures:
            print(f"  - max_num_seqs={mns}: {err}", flush=True)
        return 1
    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
