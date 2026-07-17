"""diag_l0_content_filter_v2.py — emulate production phase1 batches, bisect on filter hit.

For each test record (limited):
  1. Build the exact signal list eval_m2_judge.py builds.
  2. Send full record as one batch to gpt-5.1 with the phase1 prompt.
  3. If content_filter 400: bisect halves recursively until we identify
     the smallest set (single signals if possible) that still triggers.
  4. Dump offending signals + category breakdown.
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
    PHASE1_PROMPT_PATH,
)


async def _call(client, deployment, system_prompt, batch, max_tokens) -> tuple[str, Any]:
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
        return "ok", resp.choices[0].message.content
    except BadRequestError as e:
        body = getattr(e, "response", None)
        try:
            err = body.json() if body is not None else {}
        except Exception:
            err = {}
        err_obj = err.get("error", {}) or {}
        code = err_obj.get("code", "")
        if code == "content_filter":
            inner = err_obj.get("innererror") or {}
            cats = inner.get("content_filter_result", {})
            triggered = {k: v.get("severity") for k, v in cats.items() if isinstance(v, dict) and v.get("filtered")}
            return "content_filter", triggered or cats
        return "other", str(e)[:200]
    except Exception as e:
        return "other", repr(e)[:200]


async def bisect_find(client, deployment, system_prompt, batch, max_tokens) -> list[tuple[dict, dict]]:
    """Return list of (signal, triggered_cats) for individual filter-triggering signals."""
    status, payload = await _call(client, deployment, system_prompt, batch, max_tokens)
    if status != "content_filter":
        return []
    if len(batch) == 1:
        return [(batch[0], payload)]
    mid = len(batch) // 2
    left = await bisect_find(client, deployment, system_prompt, batch[:mid], max_tokens)
    right = await bisect_find(client, deployment, system_prompt, batch[mid:], max_tokens)
    return left + right


async def main(args):
    cfg = yaml.safe_load(Path(args.config).read_text())["judge"]
    sys_prompt = PHASE1_PROMPT_PATH.read_text(encoding="utf-8")

    # Load predictions + test
    preds = {}
    with open(args.predictions) as f:
        for line in f:
            d = json.loads(line)
            preds[d["record_idx"]] = d

    test_records = []
    with open(args.test_jsonl) as f:
        for line in f:
            test_records.append(json.loads(line))

    client = AsyncAzureOpenAI(
        azure_endpoint=cfg["azure_endpoint"],
        api_version=cfg["api_version"],
        api_key=os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_KEY"),
        timeout=300.0,
    )

    total = len(preds)
    if args.stride > 1:
        indices = list(range(0, total, args.stride))[: args.limit]
    else:
        indices = list(range(min(args.limit, total)))
    print(f"diagnosing {len(indices)} records (production-shaped batches), span={indices[0]}..{indices[-1]} stride={args.stride}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    results = []
    stats = {"ok": 0, "content_filter": 0, "other": 0, "no_signals": 0}

    async def _run(idx: int) -> None:
        pred = preds.get(idx)
        if not pred:
            return
        test_rec = test_records[idx]
        try:
            user_msg = test_rec["messages"][3]["content"]
        except (IndexError, KeyError):
            return
        tsv_rows = parse_user_msg_tsv(user_msg)
        text = pred["reference"]  # teacher mode
        pred_parsed = extract_result_list(safe_json_loads(text))
        if pred_parsed is None:
            return
        signals = build_signal_objects(tsv_rows, pred_parsed)
        if signals is None or not signals:
            stats["no_signals"] += 1
            return

        async with sem:
            status, payload = await _call(client, cfg["deployment"], sys_prompt, signals, cfg["max_completion_tokens"])
        stats[status] += 1
        if status == "content_filter":
            cats_summary = ",".join(f"{k}={v}" for k, v in payload.items())
            print(f"[FILTER] record_idx={idx} user_id={pred.get('user_id','?')} n_signals={len(signals)} batch_cats={cats_summary}", flush=True)
            # bisect to find offenders
            async with sem:
                offenders = await bisect_find(client, cfg["deployment"], sys_prompt, signals, cfg["max_completion_tokens"])
            for sig, cats in offenders:
                cats_str = ",".join(f"{k}={v}" for k, v in cats.items()) if isinstance(cats, dict) else str(cats)
                src = sig.get("source", "")
                ds = sig.get("detailed_source", "")
                action = sig.get("action", "")
                print(f"  -> row={sig.get('row')} src={src}/{ds} cats={cats_str}", flush=True)
                print(f"     action: {action[:300]!r}", flush=True)
                results.append({
                    "record_idx": idx,
                    "user_id": pred.get("user_id"),
                    "row": sig.get("row"),
                    "source": src,
                    "detailed_source": ds,
                    "action": action,
                    "should_filter_predicted": sig.get("should_filter"),
                    "categories": cats,
                })
        elif status == "other":
            print(f"[OTHER] record_idx={idx}: {payload}", flush=True)

    await asyncio.gather(*[_run(i) for i in indices])

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("", flush=True)
    print("=== Summary ===", flush=True)
    for k, v in stats.items():
        print(f"  {k}: {v}", flush=True)
    print(f"  offending signals dumped: {len(results)} -> {args.out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--test-jsonl", required=True)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--stride", type=int, default=1, help="sample every Nth record")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--out", default="diag_l0_content_filter_v2.json")
    asyncio.run(main(p.parse_args()))
