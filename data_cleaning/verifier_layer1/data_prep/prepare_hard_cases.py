"""Prepare the Layer1-delta RL training prompt set.

Goal
----
Produce a JSONL where each line is one RL prompt::

    {
      "record_id": "<user_id>__<YYYYMMDD>",
      "user_id": "<user_id>",
      "date": "YYYYMMDD",
      "system_prompt": "<contents of prompts/layer1_delta.md>",
      "user_message": "Date: ...\\nUser ID: ...\\n...\\nToday's Denoised Signals (N events):\\n[...]",
      "input_signals": [ {Date, Source, DetailedSource, Action, intent}, ... ],
      "n_signals": 17,
      "stratum": "medium",
      "source_run": "gpt52__layer1_delta__v0.8/20260103",
      "sft50k_f1": 0.42        # optional, only when --sft-eval is provided
    }

The script mirrors the production prompt format from
``maiprofilev3dev/modules/layer1_delta.py``, so the RL actor sees the
exact same input distribution as the SFT model did at training time.

Two modes (mutually exclusive)
------------------------------
1. ``--sft-eval``  → **F1-driven hard-case mining (preferred).** Reads a
   per-record M1 eval JSONL from the SFT-50K experiment, keeps records
   with F1 ∈ ``[--f1-low, --f1-high]``, then stratified-samples down to
   ``--n``.

2. **Fallback (default).** No SFT eval available → stratified sample by
   signal-count bucket from the teacher dogfood dataset. This still
   gives RL a useful prompt distribution; you should re-run with
   ``--sft-eval`` once the SFT model has been evaluated.

In both modes, a separate ``--holdout-n`` records are written to
``<output>.holdout.jsonl`` (user-disjoint from train) for use as the
RL exit-condition eval set (≥0.595 M1.F1 on holdout = pass).

Run
---
::

    python -m rl_layer1.data_prep.prepare_hard_cases \\
      --dogfood-root MaiProfile-main/maiprofilev3dev/dogfood/20260320/gpt52__layer1_delta__v0.8 \\
      --prompt-file  MaiProfile-main/maiprofilev3dev/prompts/layer1_delta.md \\
      --output       rl_layer1/data/train_hard.jsonl \\
      --n 8000 --holdout-n 200 --seed 42

With SFT eval::

    python -m rl_layer1.data_prep.prepare_hard_cases \\
      --dogfood-root ... --prompt-file ... \\
      --sft-eval path/to/sft50k_m1_per_record.jsonl \\
      --f1-low 0.30 --f1-high 0.60 \\
      --output rl_layer1/data/train_hard.jsonl --n 8000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


logger = logging.getLogger("rl_layer1.data_prep.prepare_hard_cases")


# ---------------------------------------------------------------------------
# Stratification buckets (by # signals after layer0 filtering)
# ---------------------------------------------------------------------------
SIGNAL_BUCKETS: list[tuple[str, int, int]] = [
    ("xs",       1,   4),
    ("small",    5,  14),
    ("medium",  15,  29),
    ("large",   30,  59),
    ("xl",      60, 10_000),
]


def _bucket_for(n: int) -> str:
    for name, lo, hi in SIGNAL_BUCKETS:
        if lo <= n <= hi:
            return name
    return "unknown"


# ---------------------------------------------------------------------------
# Prompt reconstruction (mirrors modules/layer1_delta.py)
# ---------------------------------------------------------------------------

_KEPT_SIGNAL_KEYS = ("Date", "Source", "DetailedSource", "Action", "intent")


def _format_facts(_facts: Any) -> str:  # no per-user facts in dogfood today
    return "No facts available."


def _build_user_message(user_id: str, date_str: str, kept_signals: list[dict]) -> str:
    signal_json = json.dumps(kept_signals, ensure_ascii=False, separators=(",", ":"))
    return (
        f"Date: {date_str}\n"
        f"User ID: {user_id}\n"
        f"User Demographics / Facts:\n{_format_facts(None)}\n\n"
        f"Today's Denoised Signals ({len(kept_signals)} events):\n{signal_json}"
    )


def _kept_from_layer0(signals: list[dict]) -> list[dict]:
    out: list[dict] = []
    for s in signals or []:
        if not isinstance(s, dict):
            continue
        if s.get("should_filter", False):
            continue
        out.append({k: s[k] for k in _KEPT_SIGNAL_KEYS if k in s})
    return out


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


@dataclass
class Record:
    record_id: str
    user_id: str
    date: str
    system_prompt: str
    user_message: str
    input_signals: list[dict]
    n_signals: int
    stratum: str
    source_run: str
    sft50k_f1: float | None = None


def _iter_layer0_records(dogfood_root: Path) -> Iterable[tuple[str, dict]]:
    """Yield ``(source_run, record)`` for every layer0_signal record under
    ``dogfood_root``. ``source_run`` is the run-relative subdir
    (e.g., ``20260103``)."""
    if not dogfood_root.is_dir():
        raise NotADirectoryError(dogfood_root)
    for delta_dir in sorted(dogfood_root.iterdir()):
        if not delta_dir.is_dir():
            continue
        f = delta_dir / "layer0_signal.jsonl"
        if not f.exists():
            continue
        run_label = f"{dogfood_root.name}/{delta_dir.name}"
        with f.open("r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError as e:
                    logger.warning("bad json in %s: %s", f, e)
                    continue
                yield run_label, rec


def _load_sft_eval(path: Path) -> dict[str, float]:
    """Load per-record F1 from a SFT eval JSONL.

    The eval file is expected to have one line per ``(user_id, date)``
    record with an ``m1`` block (or a top-level ``f1``). Accepts several
    common schema variants::

        {"record_id": "<uid>__<yyyymmdd>", "m1": {"f1": 0.42}, ...}
        {"user_id": "<uid>", "date": "yyyymmdd", "f1": 0.42, ...}
        {"user_id": "<uid>", "date": "yyyymmdd", "m1_f1": 0.42, ...}
    """
    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            rid = rec.get("record_id")
            if rid is None:
                uid = rec.get("user_id")
                date = rec.get("date") or rec.get("date_str")
                if uid is None or date is None:
                    continue
                rid = f"{uid}__{date}"
            f1: Any = None
            if isinstance(rec.get("m1"), dict):
                f1 = rec["m1"].get("f1") or rec["m1"].get("F1")
            if f1 is None:
                f1 = rec.get("m1_f1") or rec.get("f1") or rec.get("F1")
            if f1 is None:
                continue
            try:
                out[rid] = float(f1)
            except (TypeError, ValueError):
                continue
    logger.info("loaded %d records with f1 from %s", len(out), path)
    return out


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _stratified_sample(
    records: list[Record],
    *,
    n: int,
    rng: random.Random,
) -> list[Record]:
    """Even-by-stratum sample. If a bucket runs out, the remainder
    is filled proportionally from other buckets."""
    if n >= len(records):
        return records[:]
    buckets: dict[str, list[Record]] = {}
    for r in records:
        buckets.setdefault(r.stratum, []).append(r)
    per = max(1, n // max(1, len(buckets)))
    out: list[Record] = []
    for k, rs in buckets.items():
        rng.shuffle(rs)
        out.extend(rs[:per])
    # Top up if rounding short-changed us.
    if len(out) < n:
        remaining = [r for r in records if r not in out]
        rng.shuffle(remaining)
        out.extend(remaining[: n - len(out)])
    # Trim if we overshot.
    rng.shuffle(out)
    return out[:n]


# ---------------------------------------------------------------------------
# Build full record list
# ---------------------------------------------------------------------------


def _build_records(
    *,
    dogfood_root: Path,
    system_prompt: str,
    f1_lookup: dict[str, float] | None,
    f1_low: float,
    f1_high: float,
    min_signals: int,
) -> list[Record]:
    all_records: list[Record] = []
    n_skipped_empty = 0
    n_skipped_no_f1 = 0
    n_skipped_outside_band = 0

    for source_run, raw in _iter_layer0_records(dogfood_root):
        user_id = raw.get("user_id")
        date = raw.get("date_str") or raw.get("date")
        if not user_id or not date:
            continue
        kept = _kept_from_layer0(raw.get("signals", []))
        if len(kept) < min_signals:
            n_skipped_empty += 1
            continue
        rid = f"{user_id}__{date}"
        f1: float | None = None
        if f1_lookup is not None:
            f1 = f1_lookup.get(rid)
            if f1 is None:
                n_skipped_no_f1 += 1
                continue
            if not (f1_low <= f1 <= f1_high):
                n_skipped_outside_band += 1
                continue
        user_message = _build_user_message(user_id, date, kept)
        all_records.append(Record(
            record_id=rid,
            user_id=user_id,
            date=date,
            system_prompt=system_prompt,
            user_message=user_message,
            input_signals=kept,
            n_signals=len(kept),
            stratum=_bucket_for(len(kept)),
            source_run=source_run,
            sft50k_f1=f1,
        ))

    logger.info(
        "scan complete: kept=%d, skipped_empty=%d, skipped_no_f1=%d, "
        "skipped_outside_band=%d",
        len(all_records), n_skipped_empty, n_skipped_no_f1, n_skipped_outside_band,
    )
    return all_records


# ---------------------------------------------------------------------------
# Holdout split (user-disjoint)
# ---------------------------------------------------------------------------


def _split_train_holdout(
    records: list[Record],
    *,
    holdout_n: int,
    rng: random.Random,
) -> tuple[list[Record], list[Record]]:
    """Pick a user-disjoint holdout. Greedy: shuffle users, then pull
    *one* record per user into holdout until we hit ``holdout_n``."""
    by_user: dict[str, list[Record]] = {}
    for r in records:
        by_user.setdefault(r.user_id, []).append(r)
    users = list(by_user.keys())
    rng.shuffle(users)

    holdout: list[Record] = []
    holdout_users: set[str] = set()
    for u in users:
        if len(holdout) >= holdout_n:
            break
        rs = by_user[u]
        rng.shuffle(rs)
        holdout.append(rs[0])
        holdout_users.add(u)
    train = [r for r in records if r.user_id not in holdout_users]
    return train, holdout


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def _summarize(records: list[Record]) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    sig_counts = [r.n_signals for r in records]
    bucket_hist: dict[str, int] = {}
    for r in records:
        bucket_hist[r.stratum] = bucket_hist.get(r.stratum, 0) + 1
    summary: dict[str, Any] = {
        "n": len(records),
        "n_signals_min": min(sig_counts),
        "n_signals_max": max(sig_counts),
        "n_signals_mean": round(sum(sig_counts) / len(sig_counts), 2),
        "buckets": bucket_hist,
        "n_unique_users": len({r.user_id for r in records}),
    }
    f1s = [r.sft50k_f1 for r in records if r.sft50k_f1 is not None]
    if f1s:
        summary["sft50k_f1_mean"] = round(sum(f1s) / len(f1s), 3)
        summary["sft50k_f1_min"] = round(min(f1s), 3)
        summary["sft50k_f1_max"] = round(max(f1s), 3)
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dogfood-root", required=True,
                   help="Path to a dogfood run dir containing {YYYYMMDD}/layer0_signal.jsonl.")
    p.add_argument("--prompt-file", required=True,
                   help="Path to prompts/layer1_delta.md.")
    p.add_argument("--output", required=True,
                   help="Output JSONL path for the RL training prompts.")
    p.add_argument("--n", type=int, default=8000,
                   help="Target # training records (after holdout split).")
    p.add_argument("--holdout-n", type=int, default=200,
                   help="# user-disjoint records reserved for RL exit-condition eval.")
    p.add_argument("--min-signals", type=int, default=5,
                   help="Drop records with fewer than this many kept signals.")
    p.add_argument("--sft-eval", default=None,
                   help="(Optional) per-record SFT-50K M1 eval JSONL for F1 filtering.")
    p.add_argument("--f1-low",  type=float, default=0.30,
                   help="Lower bound for hard-case F1 band (inclusive).")
    p.add_argument("--f1-high", type=float, default=0.60,
                   help="Upper bound for hard-case F1 band (inclusive).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    dogfood_root = Path(args.dogfood_root)
    prompt_path = Path(args.prompt_file)
    output_path = Path(args.output)
    holdout_path = output_path.with_suffix(output_path.suffix + ".holdout.jsonl") \
        if not output_path.name.endswith(".jsonl") else \
        output_path.with_name(output_path.stem + ".holdout.jsonl")

    if not dogfood_root.is_dir():
        logger.error("dogfood root not found: %s", dogfood_root)
        return 2
    if not prompt_path.is_file():
        logger.error("prompt file not found: %s", prompt_path)
        return 2

    system_prompt = prompt_path.read_text(encoding="utf-8")
    f1_lookup: dict[str, float] | None = None
    if args.sft_eval:
        f1_lookup = _load_sft_eval(Path(args.sft_eval))
        if not f1_lookup:
            logger.warning("SFT eval file is empty / has no parseable F1 — "
                           "falling back to pure stratified sampling.")
            f1_lookup = None
    else:
        logger.warning(
            "no --sft-eval provided; running FALLBACK mode (uniform stratified "
            "sample, no F1-driven hard-case mining). Re-run with --sft-eval "
            "once the SFT-50K eval has been produced.")

    all_records = _build_records(
        dogfood_root=dogfood_root,
        system_prompt=system_prompt,
        f1_lookup=f1_lookup,
        f1_low=args.f1_low,
        f1_high=args.f1_high,
        min_signals=args.min_signals,
    )
    if not all_records:
        logger.error("no records survived the filters; nothing to do.")
        return 1

    rng = random.Random(args.seed)

    # Holdout first (user-disjoint), then stratified-sample the train pool.
    train_pool, holdout = _split_train_holdout(
        all_records, holdout_n=args.holdout_n, rng=rng,
    )
    train = _stratified_sample(train_pool, n=args.n, rng=rng)

    _write_jsonl(output_path, train)
    _write_jsonl(holdout_path, holdout)

    logger.info("wrote %d train records → %s", len(train), output_path)
    logger.info("wrote %d holdout records → %s", len(holdout), holdout_path)
    logger.info("train summary: %s", json.dumps(_summarize(train), ensure_ascii=False))
    logger.info("holdout summary: %s", json.dumps(_summarize(holdout), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
