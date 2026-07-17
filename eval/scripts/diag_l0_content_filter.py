"""diag_l0_content_filter.py — find which user signals trigger Azure content_filter.

Recreates the exact signal list that eval_m2_judge.py's phase1 sends, then
ships each signal individually (or in tiny chunks) to gpt-5.1 with the same
prompt. Logs every 400 content_filter together with the offending signal
text and the filter category breakdown returned by Azure.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncAzureOpenAI, BadRequestError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_m2_judge import (  # type: ignore
    parse_user_msg_tsv,
    build_signal_objects,
)

PROMPT_PATH = Path(
    "/scratch/azureml/cr/j/fc096b74f20c46ae94d7fab7e20c1aa4/exe/wd/"
    "MaiProfile-main 1/MaiProfile-main/maiprofilev3dev/evaluation/prompts/"
    "eval_raw_signal_denoising.md"
)


def load_signals(test_jsonl: Path, pred_jsonl: Path) -> list[tuple[int, int, dict]]:
    """Return list of (record_idx, signal_idx, signal_dict) replicating phase1 input."""
    # Build mapping from record_idx -> test_user_msg + pred_rows
    preds: dict[int, dict] = {}
    with pred_jsonl.open() as f:
        for line in f:
            d = json.loads(line)
            preds[d["record_idx"]] = d

    out: list[tuple[int, int, dict]] = []
    with test_jsonl.open() as f:
        for ri, line in enumerate(f):
            t = json.loads(line)
            user_msg = next(
                (m["content"] for m in t["messages"] if m["role"] == "user"), ""
            )
            tsv_rows = parse_user_msg_tsv(user_msg)
            pred = preds.get(ri)
            if not pred:
                continue
            # parse prediction rows from prediction string
            pred_text = pred.get("prediction", "")
            # Extract JSON object from prediction (the model output)
            try:
                # Strip <think>...</think>
                if "</think>" in pred_text:
                    pred_text = pred_text.split("</think>", 1)[1]
                # Find JSON array
                start = pred_text.find("[")
                end = pred_text.rfind("]")
                pred_rows = json.loads(pred_text[start : end + 1]) if start >= 0 and end > start else []
            except Exception:
                pred_rows = []
            signals = build_signal_objects(tsv_rows, pred_rows) or []
            for si, sig in enumerate(signals):
                out.append((ri, si, sig))
    return out


async def _judge_one(
    client: AsyncAzureOpenAI,
    deployment: str,
    system_prompt: str,
    batch: list[dict],
    max_tokens: int,
) -> tuple[str, Any]:
    """Return (status, payload). status in {ok, content_filter, other}."""
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
            err_json = body.json() if body is not None else {}
        except Exception:
            err_json = {}
        inner = (err_json.get("error", {}) or {}).get("innererror") or {}
        code = (err_json.get("error", {}) or {}).get("code", "")
        if code == "content_filter":
            cats = inner.get("content_filter_result", {})
            triggered = {k: v for k, v in cats.items() if isinstance(v, dict) and v.get("filtered")}
            return "content_filter", triggered or cats
        return "other", str(e)
    except Exception as e:
        return "other", repr(e)


async def main(args: argparse.Namespace) -> None:
    cfg = yaml.safe_load(Path(args.config).read_text())["judge"]
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    print(f"loading signals from {args.predictions} ...", flush=True)
    sigs = load_signals(Path(args.test_jsonl), Path(args.predictions))
    print(f"total signals: {len(sigs)}", flush=True)

    sample = sigs[: args.limit]
    print(f"diagnosing first {len(sample)} signals individually", flush=True)

    api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_KEY")
    client = AsyncAzureOpenAI(
        azure_endpoint=cfg["azure_endpoint"],
        api_version=cfg["api_version"],
        api_key=api_key,
    )

    sem = asyncio.Semaphore(4)
    triggered: list[dict] = []
    counts = {"ok": 0, "content_filter": 0, "other": 0}

    async def _run(ri: int, si: int, sig: dict) -> None:
        async with sem:
            status, payload = await _judge_one(
                client, cfg["deployment"], system_prompt, [sig], 2000
            )
            counts[status] += 1
            if status == "content_filter":
                triggered.append({
                    "record_idx": ri,
                    "signal_idx": si,
                    "signal": sig,
                    "categories": payload,
                })
                cats = ",".join(f"{k}={v.get('severity','?')}" for k, v in payload.items() if isinstance(v, dict))
                print(f"  [FILTER] rec={ri} sig={si} action={sig.get('action','')[:120]!r}  ->  {cats}", flush=True)
            elif status == "other":
                print(f"  [OTHER ERR] rec={ri} sig={si}: {str(payload)[:200]}", flush=True)

    await asyncio.gather(*[_run(ri, si, sig) for ri, si, sig in sample])

    out = Path(args.out)
    out.write_text(json.dumps(triggered, ensure_ascii=False, indent=2))
    print("", flush=True)
    print(f"=== Summary over {len(sample)} signals ===", flush=True)
    print(f"  ok:             {counts['ok']}", flush=True)
    print(f"  content_filter: {counts['content_filter']}  ({100*counts['content_filter']/max(1,len(sample)):.1f}%)", flush=True)
    print(f"  other_errors:   {counts['other']}", flush=True)
    print(f"  triggered signals dumped to {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--test-jsonl", required=True)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--out", default="diag_l0_content_filter.json")
    asyncio.run(main(p.parse_args()))
