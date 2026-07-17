"""generate_outputs.py — Run inference on test.jsonl using a vLLM endpoint.

Layer-format-aware: drops the LAST assistant message from each record and uses
the remaining messages as the prompt (works for layer0's 5-msg format and
layer1_delta's 3-msg format identically). The reference is the dropped
assistant content.

Output schema (one JSON object per line):
    {
      "record_idx":   int,
      "user_id":      str,
      "date":         str,
      "delta_index":  int,
      "reference":    str (raw last assistant),
      "prediction":   str (model under test),
      "latency_s":    float,
      "input_n_signals": int   # interpretation depends on data.layer_format
    }

The eval YAML must include:
    data:
      layer_format: layer0_signal | layer1_delta

Group A (teacher) does NOT need this script — its predictions are already in
test.jsonl as the last assistant message. M1/M2 reads it directly.

Usage:
    python scripts/generate_outputs.py \
        --config configs/eval/layer1_delta.yaml \
        --subject sft \
        --endpoint http://127.0.0.1:8000/v1 \
        --output eval_results/layer1_delta/predictions/sft.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
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


# Regex to pull "N user signals" from msg[3]'s prefix line.
_N_SIGNALS_RE = re.compile(r"^(\d+)\s+user signals", re.IGNORECASE)
# Regex for layer1_delta header: "Today's Denoised Signals (N events):"
_N_EVENTS_RE = re.compile(r"\((\d+)\s+events\)", re.IGNORECASE)


def count_input_signals_layer0(user_msg: str) -> int:
    """Count TSV data rows in layer0 msg[3] ("N user signals in TSV format ...")."""
    m = _N_SIGNALS_RE.match(user_msg.strip())
    if m:
        return int(m.group(1))
    n = 0
    for line in user_msg.splitlines():
        if re.match(r"^\d+\t", line):
            n += 1
    return n


def count_input_signals_layer1_delta(user_msg: str) -> int:
    """Count denoised events in layer1_delta user message ("Today's Denoised Signals (N events):")."""
    m = _N_EVENTS_RE.search(user_msg)
    if m:
        return int(m.group(1))
    # Fallback: try to parse the JSON array after the header
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


def count_input_signals(user_msg: str, layer_format: str) -> int:
    if layer_format == "layer0_signal":
        return count_input_signals_layer0(user_msg)
    if layer_format == "layer1_delta":
        return count_input_signals_layer1_delta(user_msg)
    # Default: 0 — don't fail, just leave unknown
    return 0


def _is_retryable(exc: Exception) -> bool:
    # Deterministic client errors (bad request: context-length overflow,
    # content filter, etc.) won't change on retry — fail fast on those.
    msg = str(exc)
    if "maximum context length" in msg or "content_filter" in msg:
        return False
    if "Error code: 400" in msg:
        return False
    # Transient: timeouts, connection resets, 429/5xx, etc.
    return True


async def call_one(
    client: AsyncOpenAI,
    served_name: str,
    messages: list[dict[str, str]],
    gen_cfg: dict[str, Any],
    sem: asyncio.Semaphore,
    max_retries: int = 4,
) -> tuple[str, float, int, str]:
    async with sem:
        kwargs: dict[str, Any] = {
            "model": served_name,
            "messages": messages,
            "temperature": gen_cfg.get("temperature", 0.2),
            "top_p": gen_cfg.get("top_p", 1.0),
            "max_tokens": gen_cfg.get("max_tokens", 2048),
        }
        # Safety belt: disable thinking on Qwen3 (no-op for Instruct, mandatory
        # if someone ever points us at Qwen3-Thinking by mistake).
        if not gen_cfg.get("enable_thinking", False):
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
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
        # Exhausted retries on a transient error
        raise last_exc if last_exc else RuntimeError("call_one failed")


async def main_async(args) -> int:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    subj = cfg["subject_models"][args.subject]
    gen_cfg = dict(cfg["generation"])
    # Optional per-subject generation overrides (e.g. ZS-Thinking needs
    # larger max_tokens than SFT-Thinking). Shallow merge — subject keys win.
    subj_gen_override = subj.get("generation") or {}
    if subj_gen_override:
        for k, v in subj_gen_override.items():
            gen_cfg[k] = v
        print(f"[generate] applied subject-level generation overrides: "
              f"{list(subj_gen_override.keys())}", flush=True)
    served_name = subj["served_name"]
    sem = asyncio.Semaphore(args.concurrency or gen_cfg.get("concurrency", 16))
    data_cfg = cfg.get("data", {}) or {}
    layer_format = data_cfg.get("layer_format", cfg.get("step_key", ""))

    records = load_jsonl(args.test_jsonl or cfg["test_jsonl"])
    if args.limit:
        records = records[: args.limit]
    print(f"[generate] {len(records)} records; subject={args.subject} served={served_name}",
          flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # vLLM OpenAI-compatible server: api_key can be anything non-empty
    client = AsyncOpenAI(base_url=args.endpoint, api_key="EMPTY", timeout=args.timeout)

    # Optional resume: keep successful rows from a previous run; only re-run
    # records whose record_idx is missing or whose previous attempt errored.
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
        print(f"[generate] resume: {len(existing_good)} successful rows preserved; "
              f"will (re)run {len(records) - len(existing_good)} records", flush=True)

    async def _run(idx: int, rec: dict[str, Any]) -> dict[str, Any]:
        msgs = rec["messages"]
        if not msgs or msgs[-1].get("role") != "assistant":
            return {
                "record_idx": idx, "user_id": "", "date": "", "delta_index": 0,
                "reference": "", "prediction": "", "latency_s": 0.0,
                "input_n_signals": 0,
                "error": "record missing trailing assistant message",
            }
        # Drop the last assistant (the gold/reference); use everything before as the prompt.
        prompt_msgs = msgs[:-1]
        reference = msgs[-1]["content"]
        meta = rec.get("metadata", {}) or {}
        # find the (real) user message — last user turn in the prompt
        user_text = next((m["content"] for m in reversed(prompt_msgs) if m.get("role") == "user"), "")
        n_in = count_input_signals(user_text, layer_format)
        try:
            text, latency, comp_tok, finish_reason = await call_one(
                client, served_name, prompt_msgs, gen_cfg, sem)
            err = None
        except Exception as exc:
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

    # Stream new rows to a sidecar file so partial progress survives crashes
    # even when --resume is set. Final merged file is written atomically at end.
    partial_path = out_path.with_suffix(out_path.suffix + ".partial")
    write_mode = "w"
    with open(partial_path, write_mode) as fout:
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
                print(f"[generate] {done}/{total}  "
                      f"{rate:.2f} req/s  ETA {eta:.0f}s", flush=True)

    # Merge: existing_good (preserved) + new_rows (just generated). New rows
    # win on duplicate record_idx, then sort by record_idx for determinism.
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
    print(f"[generate] DONE → {out_path}  ({len(final_rows)} total rows, {n_err} errors)", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--subject", required=True, choices=["sft", "zero_shot", "rl", "rl_repro"])
    ap.add_argument("--endpoint", required=True,
                    help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1")
    ap.add_argument("--output", required=True, help="predictions.jsonl path")
    ap.add_argument("--test-jsonl", default=None, help="override config.test_jsonl")
    ap.add_argument("--limit", type=int, default=0, help="optional cap for smoke testing")
    ap.add_argument("--concurrency", type=int, default=0, help="override config")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="OpenAI client per-request timeout in seconds (default 120)")
    ap.add_argument("--resume", action="store_true",
                    help="If output exists, preserve successful rows and only "
                         "(re)run records whose record_idx is missing or errored.")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
