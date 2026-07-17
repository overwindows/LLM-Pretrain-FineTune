"""diag_l0_filter_per_record.py — per-record content_filter diagnosis.

Replicates eval_m2_judge.py phase1 EXACTLY (uses safe_json_loads / extract_result_list
/ build_signal_objects). For each record, sends its full signal list as one batch.
If 400 content_filter, falls back to per-signal to find which signal(s) trigger it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncAzureOpenAI, BadRequestError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_m2_judge import (  # type: ignore
    parse_user_msg_tsv,
    build_signal_objects,
    safe_json_loads,
    extract_result_list,
)

PROMPT_PATH = Path(
    "/scratch/azureml/cr/j/fc096b74f20c46ae94d7fab7e20c1aa4/exe/wd/"
    "MaiProfile-main 1/MaiProfile-main/maiprofilev3dev/evaluation/prompts/"
    "eval_raw_signal_denoising.md"
)


def _err_body(exc: BadRequestError) -> tuple[str, dict]:
    try:
        body = exc.response.json()
    except Exception:
        return "", {}
    code = ((body.get("error") or {}).get("code")) or ""
    inner = ((body.get("error") or {}).get("innererror")) or {}
    return code, inner.get("content_filter_result", {}) or {}


async def _call(
    client: AsyncAzureOpenAI, deployment: str, system_prompt: str,
    batch: list[dict], max_tokens: int,
) -> tuple[str, Any]:
    try:
        resp = await client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(batch, ensure_ascii=False, indent=2)},
            ],
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return "ok", resp.choices[0].message.content or ""
    except BadRequestError as e:
        code, cats = _err_body(e)
        if code == "content_filter":
            return "content_filter", cats
        return "other", f"{code}: {str(e)[:200]}"
    except Exception as e:
        return "other", repr(e)


async def main(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(Path(args.config).read_text())["judge"]
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    preds: list[dict] = []
    with open(args.predictions) as f:
        for line in f:
            preds.append(json.loads(line))
    test_records: list[dict] = []
    with open(args.test_jsonl) as f:
        for line in f:
            test_records.append(json.loads(line))

    api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_KEY")
    client = AsyncAzureOpenAI(
        azure_endpoint=cfg["azure_endpoint"],
        api_version=cfg["api_version"],
        api_key=api_key,
        timeout=300.0,
        max_retries=0,
    )

    # Subsample preds at stride for coverage
    if args.stride > 1:
        sample = preds[:: args.stride][: args.limit]
    else:
        sample = preds[: args.limit]
    print(f"sampling {len(sample)} records (stride={args.stride})", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    record_filtered: list[dict] = []
    signal_filtered: list[dict] = []
    counts = {"ok": 0, "filtered_record": 0, "parse_unjudgeable": 0,
              "align_unjudgeable": 0, "other": 0}

    async def _process(pi: int, pred: dict) -> None:
        async with sem:
            test_rec = test_records[pred["record_idx"]]
            user_msg = test_rec["messages"][3]["content"]
            tsv_rows = parse_user_msg_tsv(user_msg)
            text = pred["reference"] if args.teacher_mode else pred.get("prediction", "")
            pred_rows = extract_result_list(safe_json_loads(text))
            if pred_rows is None:
                counts["parse_unjudgeable"] += 1
                return
            signals = build_signal_objects(tsv_rows, pred_rows)
            if signals is None:
                counts["align_unjudgeable"] += 1
                return

            status, payload = await _call(
                client, cfg["deployment"], system_prompt, signals, cfg["max_completion_tokens"]
            )
            if status == "ok":
                counts["ok"] += 1
                return
            if status == "other":
                counts["other"] += 1
                print(f"  [OTHER pi={pi} rec={pred['record_idx']}] {str(payload)[:200]}", flush=True)
                return

            # content_filter on the whole batch — narrow down per signal
            counts["filtered_record"] += 1
            cats_record = payload  # dict of categories from whole batch
            cats_str = ",".join(
                f"{k}={v.get('severity','?')}"
                for k, v in cats_record.items() if isinstance(v, dict) and v.get("filtered")
            )
            print(f"[BATCH FILTERED] pi={pi} rec_idx={pred['record_idx']} user={pred.get('user_id','')} n_signals={len(signals)} cats={cats_str}", flush=True)
            record_filtered.append({
                "pi": pi, "record_idx": pred["record_idx"],
                "user_id": pred.get("user_id", ""),
                "n_signals": len(signals),
                "batch_categories": cats_record,
                "all_actions": [s.get("action", "")[:300] for s in signals],
            })
            # Per-signal probe to find offenders (sequential, but bounded)
            for s in signals:
                st, pl = await _call(client, cfg["deployment"], system_prompt, [s], 2000)
                if st == "content_filter":
                    tr = ",".join(
                        f"{k}={v.get('severity','?')}"
                        for k, v in pl.items() if isinstance(v, dict) and v.get("filtered")
                    )
                    print(f"    -> signal row={s.get('row')} ACTION={s.get('action','')[:200]!r} cats={tr}", flush=True)
                    signal_filtered.append({
                        "pi": pi, "record_idx": pred["record_idx"],
                        "row": s.get("row"),
                        "action": s.get("action", ""),
                        "source": s.get("source", ""),
                        "detailed_source": s.get("detailed_source", ""),
                        "categories": pl,
                    })

    await asyncio.gather(*[_process(pi, p) for pi, p in enumerate(sample)])

    out = Path(args.out)
    out.write_text(json.dumps({
        "summary": {
            "sampled_records": len(sample),
            **counts,
            "filter_rate_pct": round(100 * counts["filtered_record"] / max(1, len(sample)), 2),
            "n_offending_signals_pinpointed": len(signal_filtered),
        },
        "filtered_records": record_filtered,
        "filtered_signals": signal_filtered,
    }, ensure_ascii=False, indent=2))
    print("", flush=True)
    print(f"=== Summary ===", flush=True)
    for k, v in counts.items():
        print(f"  {k}: {v}", flush=True)
    print(f"  filter_rate: {100*counts['filtered_record']/max(1,len(sample)):.1f}%", flush=True)
    print(f"  offending signals pinpointed: {len(signal_filtered)}", flush=True)
    print(f"  -> dumped to {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--test-jsonl", required=True)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--teacher-mode", action="store_true")
    p.add_argument("--out", default="diag_l0_filter_per_record.json")
    asyncio.run(main(p.parse_args()))
