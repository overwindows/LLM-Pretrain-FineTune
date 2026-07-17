"""Offline judge runner for RM distillation and pre-RL validation.

Two use cases:

1. **Phase 2 #1**: compute ``pct(M2.Precision==0)`` baseline on a small
   subset for one model (e.g., sft-50K) to confirm KL anchor quality.
2. **Phase 2 #2 / Phase 5**: bulk-judge outputs from multiple model
   sources (teacher / sft-50K / sft-10K / zero-shot) to produce the
   training data for the distilled RM.

Design choices
--------------
- **Async + semaphore-bounded concurrency.** Defaults to 32; bump via
  ``--concurrency`` until you start seeing 429s, then back off.
- **Exponential backoff with jitter** on rate-limit / transient errors.
- **Per-record JSONL streaming.** Each result is written as soon as it
  completes, so a crash mid-batch keeps everything already-done.
- **Resume.** If the output file already exists, records with matching
  ``record_id`` are skipped on the next run.
- **Prompt is loaded from a file** — judges typically live under
  ``maiprofilev3dev/evaluation/prompts/`` (e.g.,
  ``layer1_delta_eval_interest_name_v2.md``).

Output schema
-------------
Each output line is::

    {
      "record_id": "user_xxx__delta_yyyymmdd__source_sft50k",
      "source": "sft-50K",
      "input_summary": {"n_interests": 4, "n_topics": 9},
      "judge_raw": "<raw judge text>",
      "judge_parsed": [{"interest_name": "...", "scores": {...}}, ...],
      "metrics": {
          "n_scored": 4,
          "mean_precision": 7.5, "mean_utility": 8.0, ...
          "pct_precision_zero": 0.0,
      },
      "usage": {...},
      "elapsed_s": 4.7,
      "judge_model": "gpt54-eval",
      "ts": "2026-05-29T03:21:50Z",
      "error": null
    }

Run
---
::

    python -m rl_layer1.rm_distill.run_judge \\
      --input outputs_sft50k.jsonl \\
      --output judge_sft50k_v2.jsonl \\
      --prompt-file ../MaiProfile-main/maiprofilev3dev/evaluation/prompts/layer1_delta_eval_interest_name_v2.md \\
      --judge-model gpt54-eval \\
      --concurrency 32

Input record schema (one JSON per line)::

    {
      "record_id": "<unique id>",
      "source": "sft-50K",      # free-form label
      "interests": [            # the model output to score
        {"interest_name": "...",
         "topics": [{"topic": "...", "evidence": [{"action": "..."}]}, ...]}
      ]
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError

from .llm_judge_client import JUDGE_MODELS, build_judge_client, invoke_judge


logger = logging.getLogger("rl_layer1.rm_distill.run_judge")


# ---------------------------------------------------------------------------
# Retryable error set
# ---------------------------------------------------------------------------
RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError)


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _build_judge_user_message(interests: list[dict]) -> str:
    """Build the user message for the interest_name V2 judge.

    The V2 prompt expects ``interests`` with ``interest_name`` + topic name
    strings only (no evidence). We strip down per the prompt's "Input" spec.
    """
    stripped: list[dict[str, Any]] = []
    for it in interests:
        if not isinstance(it, dict):
            continue
        topics = it.get("topics", [])
        topic_names: list[str] = []
        for t in topics:
            if isinstance(t, dict):
                name = t.get("topic") or t.get("topic_name") or t.get("name")
                if isinstance(name, str) and name.strip():
                    topic_names.append(name)
            elif isinstance(t, str):
                topic_names.append(t)
        stripped.append({
            "interest_name": it.get("interest_name", ""),
            "topics": topic_names,
        })
    return json.dumps({"interests": stripped}, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Robust judge parse + metric aggregation
# ---------------------------------------------------------------------------


def _safe_json_loads(text: str) -> Any:
    s = text.strip()
    if s.startswith("```"):
        s = s.lstrip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    for suffix in ("", "]", "}", "]}", "}]"):
        try:
            return json.loads(s + suffix)
        except json.JSONDecodeError:
            continue
    return None


def _aggregate_v2_scores(parsed: Any) -> dict[str, Any]:
    """Reduce a list of per-interest judge dicts to summary stats."""
    if not isinstance(parsed, list):
        return {"n_scored": 0, "parse_judge_ok": False}
    n = 0
    sums = {"utility": 0.0, "precision": 0.0, "coherence": 0.0, "granularity_broad": 0.0}
    n_precision_zero = 0
    for item in parsed:
        if not isinstance(item, dict):
            continue
        scores = item.get("scores", {})
        if not isinstance(scores, dict):
            continue
        # Each pillar must be present; skip the whole item if not.
        try:
            sums["utility"] += float(scores["utility"])
            prec = float(scores["precision"])
            sums["precision"] += prec
            sums["coherence"] += float(scores["coherence"])
            sums["granularity_broad"] += float(scores["granularity_broad"])
        except (KeyError, TypeError, ValueError):
            continue
        if prec == 0:
            n_precision_zero += 1
        n += 1
    if n == 0:
        return {"n_scored": 0, "parse_judge_ok": True}
    return {
        "n_scored": n,
        "parse_judge_ok": True,
        "mean_utility": sums["utility"] / n,
        "mean_precision": sums["precision"] / n,
        "mean_coherence": sums["coherence"] / n,
        "mean_granularity_broad": sums["granularity_broad"] / n,
        "pct_precision_zero": n_precision_zero / n,
        "n_precision_zero": n_precision_zero,
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


async def _judge_one(
    *,
    client: Any,
    sem: asyncio.Semaphore,
    record: dict[str, Any],
    system_prompt: str,
    judge_model: str,
    max_retries: int,
    base_backoff_s: float,
) -> dict[str, Any]:
    record_id = record.get("record_id") or record.get("id") or "<missing>"
    source = record.get("source", "unknown")
    interests = record.get("interests", [])
    n_interests = len(interests) if isinstance(interests, list) else 0
    n_topics = sum(
        len(it.get("topics", [])) if isinstance(it, dict) else 0
        for it in (interests if isinstance(interests, list) else [])
    )
    user_msg = _build_judge_user_message(interests if isinstance(interests, list) else [])

    last_err: str | None = None
    for attempt in range(max_retries + 1):
        async with sem:
            try:
                resp = await invoke_judge(
                    client,
                    model_key=judge_model,
                    system_prompt=system_prompt,
                    user_prompt=user_msg,
                    response_format="json_object",
                )
                parsed = _safe_json_loads(resp.text)
                # V2 judge prompt returns a top-level list. Some judges wrap.
                if isinstance(parsed, dict) and isinstance(parsed.get("interests"), list):
                    parsed = parsed["interests"]
                metrics = _aggregate_v2_scores(parsed)
                return {
                    "record_id": record_id,
                    "source": source,
                    "input_summary": {"n_interests": n_interests, "n_topics": n_topics},
                    "judge_raw": resp.text,
                    "judge_parsed": parsed if isinstance(parsed, list) else None,
                    "metrics": metrics,
                    "usage": resp.usage,
                    "elapsed_s": resp.elapsed_s,
                    "judge_model": judge_model,
                    "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "error": None,
                }
            except RETRYABLE_ERRORS as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt >= max_retries:
                    break
                # Exponential backoff with jitter.
                sleep_s = base_backoff_s * (2 ** attempt) + random.uniform(0, base_backoff_s)
                logger.warning(
                    "[%s] retryable (%s) attempt %d/%d, sleeping %.1fs",
                    record_id, type(e).__name__, attempt + 1, max_retries, sleep_s,
                )
                await asyncio.sleep(sleep_s)
            except APIError as e:
                last_err = f"APIError: {e}"
                break  # non-retryable
            except Exception as e:  # noqa: BLE001 - capture for log
                last_err = f"{type(e).__name__}: {e}"
                break

    return {
        "record_id": record_id,
        "source": source,
        "input_summary": {"n_interests": n_interests, "n_topics": n_topics},
        "judge_raw": None,
        "judge_parsed": None,
        "metrics": {"n_scored": 0, "parse_judge_ok": False},
        "usage": {},
        "elapsed_s": 0.0,
        "judge_model": judge_model,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "error": last_err or "unknown",
    }


# ---------------------------------------------------------------------------
# Main loop with resume + streaming write
# ---------------------------------------------------------------------------


def _load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = rec.get("record_id")
            if rid is not None and rec.get("error") is None:
                done.add(rid)
    return done


def _load_input(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


async def _run_async(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_path = Path(args.output)
    prompt_path = Path(args.prompt_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.judge_model not in JUDGE_MODELS:
        logger.error("unknown judge model %r; available: %s",
                     args.judge_model, sorted(JUDGE_MODELS.keys()))
        return 2

    system_prompt = prompt_path.read_text(encoding="utf-8")
    records = _load_input(input_path)
    done_ids = _load_done_ids(output_path) if args.resume else set()

    todo = [r for r in records if (r.get("record_id") or r.get("id")) not in done_ids]
    if args.limit is not None:
        todo = todo[: args.limit]

    logger.info(
        "input=%d, already done=%d, will judge=%d, concurrency=%d, judge=%s",
        len(records), len(done_ids), len(todo), args.concurrency, args.judge_model,
    )

    if not todo:
        logger.info("nothing to do.")
        return 0

    client = build_judge_client(args.judge_model)
    sem = asyncio.Semaphore(args.concurrency)
    start_ts = time.perf_counter()

    # Open output file in append mode for streaming writes.
    out_lock = asyncio.Lock()
    n_done = 0
    n_failed = 0

    with output_path.open("a", encoding="utf-8") as out_f:

        async def _wrapped(rec: dict[str, Any]) -> None:
            nonlocal n_done, n_failed
            result = await _judge_one(
                client=client, sem=sem, record=rec,
                system_prompt=system_prompt,
                judge_model=args.judge_model,
                max_retries=args.max_retries,
                base_backoff_s=args.base_backoff_s,
            )
            async with out_lock:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                n_done += 1
                if result["error"]:
                    n_failed += 1
                if n_done % 50 == 0 or n_done == len(todo):
                    elapsed = time.perf_counter() - start_ts
                    rate = n_done / max(elapsed, 1e-6)
                    eta = (len(todo) - n_done) / max(rate, 1e-6)
                    logger.info(
                        "progress %d/%d (failed=%d) rate=%.2f/s eta=%.0fs",
                        n_done, len(todo), n_failed, rate, eta,
                    )

        await asyncio.gather(*(_wrapped(r) for r in todo))

    elapsed = time.perf_counter() - start_ts
    logger.info(
        "done. judged=%d failed=%d elapsed=%.1fs (%.2f/s)",
        n_done, n_failed, elapsed, n_done / max(elapsed, 1e-6),
    )
    try:
        await client._client.aclose()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    return 0 if n_failed == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline async judge runner.")
    p.add_argument("--input", required=True, help="JSONL of records to score.")
    p.add_argument("--output", required=True, help="JSONL of judge results.")
    p.add_argument(
        "--prompt-file", required=True,
        help="Path to the judge system prompt (e.g., layer1_delta_eval_interest_name_v2.md).",
    )
    p.add_argument("--judge-model", default="gpt54-eval",
                   help=f"One of {sorted(JUDGE_MODELS.keys())}.")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--base-backoff-s", type=float, default=2.0)
    p.add_argument("--limit", type=int, default=None,
                   help="Only judge the first N records (debug aid).")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip records with matching record_id already in output (default on).")
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    return asyncio.run(_run_async(args))


if __name__ == "__main__":
    sys.exit(main())
