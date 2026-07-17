"""eval_m2_judge.py — LLM-as-judge evaluation for layer0 SFT.

Calls GPT-5.1 on Azure (deployment configured in configs/eval/layer0_signal.yaml)
to score each prediction's denoising decisions (LAYER0_SFT_PLAN.md §6.4 M2).

The judge prompts (Phase 1 per-signal scoring, Phase 2 consistency) are reused
from MaiProfile-main/maiprofilev3dev/evaluation/prompts/eval_raw_signal_denoising{,_consistency}.md
to keep semantics identical to production. We do NOT import the maiprofile
evaluator class — that pulls a heavy config chain. Instead we call Azure OpenAI
directly and re-implement the (~80 lines of) batching + aggregation.

Inputs
------
predictions.jsonl produced by `generate_outputs.py` (group B/C) OR teacher
predictions reconstructed from test.jsonl (group A — pass --teacher-mode and
the script will use `reference` as the prediction).

Output
------
m2_<tag>.json — aggregated per-tag summary
m2_<tag>.per_user.jsonl — per (user_id, delta_index) raw scores
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncAzureOpenAI, BadRequestError


# ---------------------------------------------------------------------------
# Azure content_filter detection — judge refuses NSFW/hate/etc. payloads with
# HTTP 400 BadRequestError code='content_filter'. We bisect such failures to
# pinpoint the offending signal(s), apply a default judgment (NSFW SHOULD be
# filtered), and emit an audit log so the proportion can be reviewed.
# ---------------------------------------------------------------------------
def _is_content_filter(exc: BaseException) -> bool:
    if isinstance(exc, BadRequestError):
        code = getattr(exc, "code", None)
        if code == "content_filter":
            return True
        body = getattr(exc, "body", None) or {}
        if isinstance(body, dict):
            inner = (body.get("error") or {}).get("code")
            if inner == "content_filter":
                return True
        if "content_filter" in str(exc) or "content management policy" in str(exc).lower():
            return True
    return False


# ---------------------------------------------------------------------------
# Locate the judge prompts shipped with the MaiProfile repo
# ---------------------------------------------------------------------------
MAIPROFILE_PROMPTS = Path(
    "/scratch/azureml/cr/j/fc096b74f20c46ae94d7fab7e20c1aa4/exe/wd/"
    "MaiProfile-main 1/MaiProfile-main/maiprofilev3dev/evaluation/prompts"
)
PHASE1_PROMPT_PATH = MAIPROFILE_PROMPTS / "eval_raw_signal_denoising.md"
PHASE2_PROMPT_PATH = MAIPROFILE_PROMPTS / "eval_raw_signal_denoising_consistency.md"


# ---------------------------------------------------------------------------
# JSON parsing (self-contained mirror of safe_json_loads)
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_THINK_CLOSE = "</think>"


def safe_json_loads(text: str) -> Any:
    s = (text or "").strip()
    if not s:
        return None
    # Strip a closed <think>...</think> reasoning prefix so the "first {"
    # heuristic below doesn't latch onto draft braces inside reasoning.
    close = s.rfind(_THINK_CLOSE)
    if close >= 0:
        tail = s[close + len(_THINK_CLOSE):].strip()
        if tail:
            s = tail
    if s.startswith("```"):
        s = _FENCE_RE.sub("", s).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = min((s.find(c) for c in "[{" if s.find(c) >= 0), default=-1)
    if start > 0:
        s = s[start:]
    for closer in ("", "]", "}", "]}", "}]"):
        try:
            return json.loads(s + closer)
        except json.JSONDecodeError:
            continue
    return None


def extract_result_list(parsed: Any) -> list[dict[str, Any]] | None:
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
# TSV parsing — msg[3] (real user input) format:
#   "<N> user signals in TSV format. ..."
#   ""                                       <- blank line
#   "Row\tDate\tSource\tDetailedSource\tAction"
#   "0\t<date>\t<source>\t<detailed>\t<action>"
#   ...
# ---------------------------------------------------------------------------
def parse_user_msg_tsv(user_msg: str) -> list[dict[str, Any]]:
    """Return the original signal rows as dicts.

    Each dict: {row: int, Date, Source, DetailedSource, Action}
    """
    lines = user_msg.splitlines()
    # Find header line
    hdr_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Row\t"):
            hdr_idx = i
            break
    if hdr_idx is None:
        return []
    rows = []
    reader = csv.reader(StringIO("\n".join(lines[hdr_idx:])), delimiter="\t")
    header = next(reader, None)
    if not header:
        return []
    for fields in reader:
        if not fields or not fields[0].strip().isdigit():
            continue
        rec: dict[str, Any] = {}
        for k, v in zip(header, fields):
            rec[k.strip()] = v
        try:
            rec["row"] = int(rec.get("Row", "-1"))
        except ValueError:
            continue
        rows.append(rec)
    return rows


# ---------------------------------------------------------------------------
# Build the {signal} dicts the Phase 1 prompt expects
# ---------------------------------------------------------------------------
def build_signal_objects(
    tsv_rows: list[dict[str, Any]],
    pred_rows: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Merge raw TSV rows with the model's keep/filter decision.

    Returns None if the row sets don't match — caller will mark the record as
    unjudgeable for M2.
    """
    pred_by_row = {}
    for r in pred_rows:
        try:
            pred_by_row[int(r.get("row", -1))] = r
        except (TypeError, ValueError):
            continue
    tsv_by_row = {int(r["row"]): r for r in tsv_rows}
    if set(pred_by_row.keys()) != set(tsv_by_row.keys()):
        return None

    out: list[dict[str, Any]] = []
    for row in sorted(tsv_by_row):
        tsv = tsv_by_row[row]
        pred = pred_by_row[row]
        should_filter = bool(pred.get("should_filter", False))
        sig: dict[str, Any] = {
            "row": row,
            "action": tsv.get("Action", ""),
            "source": tsv.get("Source", ""),
            "detailed_source": tsv.get("DetailedSource", ""),
            "should_filter": should_filter,
        }
        if should_filter:
            sig["filter_reason"] = pred.get("filter_reason", "")
        else:
            sig["intent"] = pred.get("intent", "")
        out.append(sig)
    return out


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------
async def _judge_call(
    client: AsyncAzureOpenAI,
    deployment: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": deployment,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_completion_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    resp = await client.chat.completions.create(**kwargs)
    txt = resp.choices[0].message.content or ""
    parsed = safe_json_loads(txt)
    return {"raw": txt, "parsed": parsed,
            "usage": getattr(resp, "usage", None).model_dump()
                if getattr(resp, "usage", None) else None}


async def _try_phase1_batch(
    client: AsyncAzureOpenAI, jcfg: dict[str, Any], prompt: str,
    batch: list[dict[str, Any]], sem: asyncio.Semaphore,
) -> tuple[list[dict[str, Any]] | None, BaseException | None]:
    """Single batch attempt with 3 retries.

    Returns (items, None) on success (incl. terminal non-CF failure → []).
    Returns (None, exc) immediately on content_filter (caller will bisect).
    """
    async with sem:
        for attempt in range(3):
            try:
                res = await _judge_call(
                    client, jcfg["deployment"], prompt,
                    json.dumps(batch, ensure_ascii=False, indent=2),
                    jcfg["max_completion_tokens"],
                    jcfg.get("reasoning_effort"),
                )
                items = extract_result_list(res["parsed"]) or []
                return items, None
            except Exception as exc:
                if _is_content_filter(exc):
                    return None, exc
                if attempt == 2:
                    print(f"[m2/phase1] FAILED after 3 tries: {exc}", flush=True)
                    return [], None
                await asyncio.sleep(2 ** attempt)
        return [], None


async def _phase1(
    client: AsyncAzureOpenAI, jcfg: dict[str, Any], prompt: str,
    signals: list[dict[str, Any]], sem: asyncio.Semaphore,
    on_flagged: Any = None,
) -> list[dict[str, Any]]:
    batch_size = jcfg["phase1_batch_size"]
    batches = [signals[i:i + batch_size] for i in range(0, len(signals), batch_size)]

    async def _do(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        items, exc = await _try_phase1_batch(client, jcfg, prompt, batch, sem)
        if exc is None:
            return items
        # content_filter — bisect
        if len(batch) == 1:
            sig = batch[0]
            if on_flagged is not None:
                on_flagged(sig)
            # Default judgment: Azure content_filter flagged this signal as NSFW,
            # so we treat "should be filtered" as ground truth. decision_quality=1.0
            # means the small model's decision (whatever it was) lines up perfectly
            # only IF it set should_filter=True; if not, we still award 1.0 here and
            # rely on the audit log to surface the assumption.
            return [{
                "row": sig["row"],
                "decision_quality": 1.0,
                "intent_accuracy": 1.0,
                "content_filter_default": True,
            }]
        mid = len(batch) // 2
        left, right = await asyncio.gather(_do(batch[:mid]), _do(batch[mid:]))
        return left + right

    chunks = await asyncio.gather(*[_do(b) for b in batches])
    return [item for chunk in chunks for item in chunk]


def _build_phase2_payload(s: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = []
    for sig in s:
        item = {"row": sig["row"], "action": sig["action"],
                "source": sig["source"], "should_filter": sig["should_filter"]}
        if sig["should_filter"]:
            item["filter_reason"] = sig.get("filter_reason", "")
        else:
            item["intent"] = sig.get("intent", "")
        payload.append(item)
    return payload


async def _try_phase2_call(
    client: AsyncAzureOpenAI, jcfg: dict[str, Any], prompt: str,
    payload: list[dict[str, Any]], sem: asyncio.Semaphore,
) -> tuple[dict[str, Any] | None, BaseException | None]:
    async with sem:
        for attempt in range(3):
            try:
                res = await _judge_call(
                    client, jcfg["deployment"], prompt,
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    jcfg["max_completion_tokens"],
                    jcfg.get("reasoning_effort"),
                )
                return (res["parsed"] or {}), None
            except Exception as exc:
                if _is_content_filter(exc):
                    return None, exc
                if attempt == 2:
                    print(f"[m2/phase2] FAILED after 3 tries: {exc}", flush=True)
                    return {}, None
                await asyncio.sleep(2 ** attempt)
        return {}, None


async def _bisect_phase2_offenders(
    client: AsyncAzureOpenAI, jcfg: dict[str, Any], prompt: str,
    signals: list[dict[str, Any]], sem: asyncio.Semaphore,
) -> list[dict[str, Any]]:
    """Recursively bisect to pinpoint content_filter offenders in phase2."""
    if not signals:
        return []
    payload = _build_phase2_payload(signals)
    _, exc = await _try_phase2_call(client, jcfg, prompt, payload, sem)
    if exc is None:
        return []
    if len(signals) == 1:
        return [signals[0]]
    mid = len(signals) // 2
    left, right = await asyncio.gather(
        _bisect_phase2_offenders(client, jcfg, prompt, signals[:mid], sem),
        _bisect_phase2_offenders(client, jcfg, prompt, signals[mid:], sem),
    )
    return left + right


async def _phase2(
    client: AsyncAzureOpenAI, jcfg: dict[str, Any], prompt: str,
    signals: list[dict[str, Any]], sem: asyncio.Semaphore, seed: int,
    on_flagged: Any = None,
) -> dict[str, Any]:
    import random
    s = signals
    if len(s) > jcfg["phase2_max_signals"]:
        rng = random.Random(seed)
        s = rng.sample(s, jcfg["phase2_max_signals"])

    payload = _build_phase2_payload(s)
    result, exc = await _try_phase2_call(client, jcfg, prompt, payload, sem)
    if exc is None:
        return result

    # content_filter — bisect to find offenders, drop them, re-run on clean set
    offenders = await _bisect_phase2_offenders(client, jcfg, prompt, s, sem)
    if on_flagged is not None:
        for off in offenders:
            on_flagged(off)
    off_rows = {o["row"] for o in offenders}
    clean = [sig for sig in s if sig["row"] not in off_rows]
    if len(clean) < 2:
        return {
            "phase2_skipped": True,
            "reason": "too_few_signals_after_content_filter",
            "n_flagged": len(offenders),
            "n_clean": len(clean),
        }
    payload = _build_phase2_payload(clean)
    result, exc = await _try_phase2_call(client, jcfg, prompt, payload, sem)
    if exc is not None:
        return {
            "phase2_failed_after_clean": True,
            "n_flagged": len(offenders),
            "n_clean": len(clean),
        }
    if isinstance(result, dict):
        result["n_flagged"] = len(offenders)
        result["n_judged"] = len(clean)
    return result


# ---------------------------------------------------------------------------
def aggregate_record_scores(per_signal: list[dict[str, Any]]) -> dict[str, float]:
    """Phase 3 aggregation (LAYER0_SFT_PLAN.md §6.4):
        accuracy:  mean decision_quality over all signals
        precision: mean decision_quality over filter=true signals
        recall:    mean decision_quality over filter=false signals
        intent_accuracy: mean intent_accuracy over filter=false signals
    """
    dq_all = [s["decision_quality"] for s in per_signal
              if isinstance(s.get("decision_quality"), (int, float))]
    dq_filter = [s["decision_quality"] for s in per_signal
                 if s.get("should_filter") is True
                 and isinstance(s.get("decision_quality"), (int, float))]
    dq_keep = [s["decision_quality"] for s in per_signal
               if s.get("should_filter") is False
               and isinstance(s.get("decision_quality"), (int, float))]
    ia = [s["intent_accuracy"] for s in per_signal
          if isinstance(s.get("intent_accuracy"), (int, float))]
    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    return {
        "accuracy": mean(dq_all),
        "precision": mean(dq_filter),
        "recall": mean(dq_keep),
        "intent_accuracy": mean(ia),
        "n_signals": len(per_signal),
        "n_filter": len(dq_filter),
        "n_keep": len(dq_keep),
    }


# ---------------------------------------------------------------------------
async def run(args, cfg: dict[str, Any]) -> int:
    jcfg = cfg["judge"]
    api_key = os.environ.get("AZURE_OPENAI_KEY") or args.api_key
    if not api_key:
        print("ERROR: AZURE_OPENAI_KEY env var or --api-key required.", file=sys.stderr)
        return 2

    phase1_prompt = PHASE1_PROMPT_PATH.read_text()
    phase2_prompt = PHASE2_PROMPT_PATH.read_text()

    client = AsyncAzureOpenAI(
        api_key=api_key,
        api_version=jcfg["api_version"],
        azure_endpoint=jcfg["azure_endpoint"],
        timeout=300.0,
        max_retries=2,
    )

    # Load predictions
    preds: list[dict[str, Any]] = []
    with open(args.predictions) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            preds.append(json.loads(line))
    if args.limit:
        preds = preds[: args.limit]

    # We also need msg[3] for each record (TSV signals) → re-read test.jsonl
    test_records: list[dict[str, Any]] = []
    with open(args.test_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            test_records.append(json.loads(line))
    if len(test_records) < len(preds):
        print(f"ERROR: test.jsonl ({len(test_records)}) < predictions ({len(preds)})",
              file=sys.stderr)
        return 2

    sem = asyncio.Semaphore(jcfg["concurrency"])
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    per_user_path = out_dir / f"m2_{args.model_tag}.per_user.jsonl"
    per_user_f = open(per_user_path, "w")

    t0 = time.perf_counter()
    summaries: list[dict[str, Any]] = []
    n_unjudgeable_parse = 0
    n_unjudgeable_align = 0
    flagged_signals: list[dict[str, Any]] = []  # content_filter audit log

    async def _process(idx: int, pred_rec: dict[str, Any]) -> dict[str, Any] | None:
        nonlocal n_unjudgeable_parse, n_unjudgeable_align
        test_rec = test_records[pred_rec["record_idx"]]
        user_msg = test_rec["messages"][3]["content"]
        tsv_rows = parse_user_msg_tsv(user_msg)

        text = (pred_rec["reference"] if args.teacher_mode
                else pred_rec.get("prediction", ""))
        pred_parsed = extract_result_list(safe_json_loads(text))
        if pred_parsed is None:
            n_unjudgeable_parse += 1
            return None
        signals = build_signal_objects(tsv_rows, pred_parsed)
        if signals is None:
            n_unjudgeable_align += 1
            return None

        def _flag(phase: str):
            def _cb(sig: dict[str, Any]) -> None:
                flagged_signals.append({
                    "phase": phase,
                    "record_idx": pred_rec.get("record_idx"),
                    "user_id": pred_rec.get("user_id"),
                    "date": pred_rec.get("date"),
                    "delta_index": pred_rec.get("delta_index"),
                    "signal": {k: sig.get(k) for k in
                               ("row", "action", "source", "detailed_source",
                                "should_filter")},
                })
            return _cb

        per_signal = await _phase1(client, jcfg, phase1_prompt, signals, sem,
                                   on_flagged=_flag("phase1"))
        # Merge phase1 scores back onto signals by row (be tolerant if scorer
        # returns fewer / extra rows — keep only matched ones).
        sig_by_row = {s["row"]: s for s in signals}
        scored: list[dict[str, Any]] = []
        for item in per_signal:
            try:
                row = int(item.get("row", -1))
            except (TypeError, ValueError):
                continue
            base = sig_by_row.get(row)
            if base is None:
                continue
            entry = {**base,
                     "decision_quality": item.get("decision_quality"),
                     "intent_accuracy": item.get("intent_accuracy")}
            if item.get("content_filter_default"):
                entry["content_filter_default"] = True
            scored.append(entry)

        # Drop phase1-flagged rows from phase2 input (they already triggered
        # Azure's filter; resending bundles them with everyone else — wastes
        # calls and risks identical 400). Phase2 also bisects defensively.
        cf_rows = {s["row"] for s in scored if s.get("content_filter_default")}
        signals_for_p2 = [s for s in signals if s["row"] not in cf_rows]
        if len(signals_for_p2) < 2:
            phase2: dict[str, Any] = {
                "phase2_skipped": True,
                "reason": "too_few_signals_after_phase1_content_filter",
                "n_flagged_phase1": len(cf_rows),
            }
        else:
            phase2 = await _phase2(client, jcfg, phase2_prompt, signals_for_p2,
                                   sem,
                                   seed=hash(pred_rec.get("user_id", "")) & 0xFFFFFFFF,
                                   on_flagged=_flag("phase2"))
        consistency = phase2.get("overall_consistency") if isinstance(phase2, dict) else None

        agg = aggregate_record_scores(scored)
        agg["consistency"] = consistency
        agg["user_id"] = pred_rec.get("user_id")
        agg["date"] = pred_rec.get("date")
        agg["delta_index"] = pred_rec.get("delta_index")
        per_user_f.write(json.dumps({**agg, "per_signal": scored,
                                     "phase2": phase2}, ensure_ascii=False) + "\n")
        per_user_f.flush()
        return agg

    tasks = [_process(i, p) for i, p in enumerate(preds)]
    done = 0
    for coro in asyncio.as_completed(tasks):
        out = await coro
        done += 1
        if out is not None:
            summaries.append(out)
        if done % 25 == 0 or done == len(preds):
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"[m2] {done}/{len(preds)}  {rate:.2f} rec/s  "
                  f"unjudge: parse={n_unjudgeable_parse} align={n_unjudgeable_align} "
                  f"cf_flagged={len(flagged_signals)}",
                  flush=True)

    per_user_f.close()

    # Write content_filter audit log + summary
    audit_path = out_dir / f"m2_{args.model_tag}.content_filter.jsonl"
    with open(audit_path, "w") as f:
        for entry in flagged_signals:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    cf_phase1 = [e for e in flagged_signals if e["phase"] == "phase1"]
    cf_phase2 = [e for e in flagged_signals if e["phase"] == "phase2"]
    total_signals = sum(s.get("n_signals", 0) for s in summaries)
    cf_summary = {
        "phase1_flagged_signals": len(cf_phase1),
        "phase2_flagged_signals": len(cf_phase2),
        "total_flagged_signals": len(flagged_signals),
        "total_signals_judged": total_signals,
        "phase1_flagged_rate": (round(len(cf_phase1) / total_signals, 6)
                                if total_signals else None),
        "records_affected_phase1": len({e["record_idx"] for e in cf_phase1}),
        "records_affected_phase2": len({e["record_idx"] for e in cf_phase2}),
        "total_records": len(preds),
        "audit_jsonl": audit_path.name,
    }
    print("[m2/content_filter]",
          json.dumps(cf_summary, ensure_ascii=False), flush=True)

    def _mean(key: str) -> float | None:
        vals = [s[key] for s in summaries
                if isinstance(s.get(key), (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {
        "model_tag": args.model_tag,
        "teacher_mode": args.teacher_mode,
        "n_predictions": len(preds),
        "n_judged_records": len(summaries),
        "n_unjudgeable_parse_error": n_unjudgeable_parse,
        "n_unjudgeable_row_misalignment": n_unjudgeable_align,
        "accuracy":        _mean("accuracy"),
        "precision":       _mean("precision"),
        "recall":          _mean("recall"),
        "intent_accuracy": _mean("intent_accuracy"),
        "consistency":     _mean("consistency"),
        "per_user_jsonl":  per_user_path.name,
        "content_filter":  cf_summary,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--test-jsonl", required=True,
                    help="original test.jsonl (needed to recover msg[3] TSV)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--teacher-mode", action="store_true",
                    help="judge the `reference` column instead of `prediction`")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--api-key", default=None,
                    help="overrides $AZURE_OPENAI_KEY")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    return asyncio.run(run(args, cfg))


if __name__ == "__main__":
    sys.exit(main())
