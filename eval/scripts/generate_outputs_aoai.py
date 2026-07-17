"""generate_outputs_aoai.py — Run inference on test.jsonl using an Azure
OpenAI reasoning model (gpt-5 / gpt-5.2) via the v1 / Responses surface.

This is a sibling of generate_outputs.py adapted for Azure *reasoning* models:
  * uses `max_completion_tokens` (NOT `max_tokens`)
  * does NOT send `temperature` / `top_p` (reasoning models reject != default)
  * optional `reasoning_effort`
  * auth: sends the key both as Bearer (OpenAI style) and `api-key` header
    (Azure style) so it works against either surface.

Output schema is IDENTICAL to generate_outputs.py so M1/M2 can consume it.

The prompt is messages[:-1] (drop trailing gold assistant); reference is the
dropped assistant content — exactly like generate_outputs.py.

Usage:
    AOAI_KEY=... python scripts/generate_outputs_aoai.py \
        --config configs/eval/layer1_delta_thinking_50k_4o-v1.local.yaml \
        --endpoint https://msncompanioneu2.services.ai.azure.com/openai/v1/ \
        --model gpt-5 \
        --model-tag gpt5 \
        --output eval_results/predictions/gpt5_1k.jsonl \
        --concurrency 16 --resume
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI


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


_N_EVENTS_RE = re.compile(r"\((\d+)\s+events\)", re.IGNORECASE)


def count_input_signals_layer1_delta(user_msg: str) -> int:
    m = _N_EVENTS_RE.search(user_msg)
    if m:
        return int(m.group(1))
    idx = user_msg.find("[{")
    if idx < 0:
        return 0
    depth = 0
    for k, ch in enumerate(user_msg[idx:], idx):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    arr = json.loads(user_msg[idx:k + 1])
                    return len(arr) if isinstance(arr, list) else 0
                except Exception:
                    return 0
    return 0


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc)
    if "maximum context length" in msg or "content_filter" in msg:
        return False
    if "Error code: 400" in msg:
        return False
    return True


async def call_one(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    gen_cfg: dict[str, Any],
    sem: asyncio.Semaphore,
    max_retries: int = 4,
) -> tuple[str, float, int, str]:
    async with sem:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": int(gen_cfg.get("max_completion_tokens", 16000)),
        }
        re_effort = gen_cfg.get("reasoning_effort")
        if re_effort:
            kwargs["reasoning_effort"] = re_effort
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            t0 = time.perf_counter()
            try:
                resp = await client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if not _is_retryable(exc):
                    raise
                await asyncio.sleep(2 ** attempt)
                continue
            text = resp.choices[0].message.content or ""
            comp_tok = 0
            try:
                comp_tok = int(resp.usage.completion_tokens) if resp.usage else 0
            except Exception:
                pass
            finish_reason = ""
            try:
                finish_reason = resp.choices[0].finish_reason or ""
            except Exception:
                pass
            return text, time.perf_counter() - t0, comp_tok, finish_reason
        raise last_exc if last_exc else RuntimeError("call_one failed")


async def main_async(args) -> int:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Generation config for reasoning models: independent of the vLLM block.
    gen_cfg: dict[str, Any] = {
        "max_completion_tokens": args.max_completion_tokens,
    }
    if args.reasoning_effort:
        gen_cfg["reasoning_effort"] = args.reasoning_effort

    data_cfg = cfg.get("data", {}) or {}
    layer_format = data_cfg.get("layer_format", cfg.get("step_key", ""))
    if layer_format != "layer1_delta":
        print(f"[gen-aoai] WARNING: layer_format={layer_format!r}; "
              f"signal counting assumes layer1_delta", flush=True)

    records = load_jsonl(args.test_jsonl or cfg["test_jsonl"])
    if args.limit:
        records = records[: args.limit]
    print(f"[gen-aoai] {len(records)} records; model={args.model} "
          f"tag={args.model_tag} conc={args.concurrency}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    key = os.environ.get(args.api_key_env)
    if not key:
        print(f"[gen-aoai] ERROR: env {args.api_key_env} not set", flush=True)
        return 2
    # Send the key BOTH ways so it works against OpenAI-style (Bearer) and
    # Azure-style (api-key header) surfaces.
    client = AsyncOpenAI(
        base_url=args.endpoint,
        api_key=key,
        default_headers={"api-key": key},
        timeout=args.timeout,
    )

    sem = asyncio.Semaphore(args.concurrency)

    existing_good: dict[int, dict[str, Any]] = {}
    if args.resume and out_path.exists():
        for line in open(out_path):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("error"):
                continue
            ridx = row.get("record_idx")
            if isinstance(ridx, int):
                existing_good[ridx] = row
        print(f"[gen-aoai] resume: {len(existing_good)} good rows preserved; "
              f"will (re)run {len(records) - len(existing_good)}", flush=True)

    async def _run(idx: int, rec: dict[str, Any]) -> dict[str, Any]:
        msgs = rec["messages"]
        if not msgs or msgs[-1].get("role") != "assistant":
            return {
                "record_idx": idx, "user_id": "", "date": "", "delta_index": 0,
                "reference": "", "prediction": "", "latency_s": 0.0,
                "input_n_signals": 0,
                "error": "record missing trailing assistant message",
            }
        prompt_msgs = msgs[:-1]
        reference = msgs[-1]["content"]
        meta = rec.get("metadata", {}) or {}
        user_text = next((m["content"] for m in reversed(prompt_msgs)
                          if m.get("role") == "user"), "")
        n_in = count_input_signals_layer1_delta(user_text)
        try:
            text, latency, comp_tok, finish_reason = await call_one(
                client, args.model, prompt_msgs, gen_cfg, sem)
            err = None
        except Exception as exc:  # noqa: BLE001
            text, latency, comp_tok, finish_reason, err = "", 0.0, 0, "", repr(exc)
        return {
            "record_idx": idx,
            "user_id": meta.get("user_id", ""),
            "date": meta.get("date", ""),
            "delta_index": meta.get("delta_index", 0),
            "reference": reference,
            "prediction": text,
            "latency_s": round(latency, 3),
            "input_n_signals": n_in,
            "completion_tokens": comp_tok,
            "finish_reason": finish_reason,
            "error": err,
        }

    to_run = [(i, r) for i, r in enumerate(records) if i not in existing_good]
    tasks = [_run(i, r) for i, r in to_run]
    new_rows: list[dict[str, Any]] = []
    done = 0
    total = len(to_run)
    t_start = time.perf_counter()

    partial_path = out_path.with_suffix(out_path.suffix + ".partial")
    with open(partial_path, "w") as fout:
        for coro in asyncio.as_completed(tasks):
            row = await coro
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            new_rows.append(row)
            done += 1
            if done % 25 == 0 or done == total:
                elapsed = time.perf_counter() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                n_err = sum(1 for r in new_rows if r.get("error"))
                print(f"[gen-aoai] {done}/{total}  {rate:.2f} req/s  "
                      f"ETA {eta:.0f}s  errors={n_err}", flush=True)

    merged: dict[int, dict[str, Any]] = dict(existing_good)
    for row in new_rows:
        ridx = row.get("record_idx")
        if isinstance(ridx, int):
            merged[ridx] = row
    final_rows = [merged[k] for k in sorted(merged.keys())]

    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_out, "w") as fout:
        for row in final_rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_out.replace(out_path)
    try:
        partial_path.unlink()
    except FileNotFoundError:
        pass

    n_err = sum(1 for r in final_rows if r.get("error"))
    print(f"[gen-aoai] DONE → {out_path}  ({len(final_rows)} rows, {n_err} errors)",
          flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--endpoint", required=True,
                    help="v1 base URL, e.g. https://res.services.ai.azure.com/openai/v1/")
    ap.add_argument("--model", required=True, help="deployment / model name, e.g. gpt-5")
    ap.add_argument("--model-tag", required=True, help="short tag for logging")
    ap.add_argument("--output", required=True)
    ap.add_argument("--api-key-env", default="AOAI_KEY")
    ap.add_argument("--test-jsonl", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-completion-tokens", type=int, default=16000)
    ap.add_argument("--reasoning-effort", default=None,
                    help="none|minimal|low|medium|high (omit = model default)")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
