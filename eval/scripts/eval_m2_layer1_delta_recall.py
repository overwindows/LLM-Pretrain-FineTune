"""eval_m2_layer1_delta_recall.py — V2 recall pipeline for layer1_delta.

Implements the official MaiProfile multi-agent recall pipeline using the four
prompts under maiprofilev3dev/evaluation/prompts/:

    interest_recall_propose.md   — propose candidate interests from keywords
    interest_recall_ground.md    — validate proposals against keywords
    interest_recall_rescue.md    — generate matched proposals for orphan keywords
    interest_recall_judge.md     — judge whether a model output covers each proposal

Stage 1 (candidates):  propose -> ground -> rescue   [model-INDEPENDENT, cached]
Stage 2 (judge):       grounded_candidates  vs  model.interests   [per-model]

Per-user recall = covered_candidates / total_grounded_candidates,
also reported separately for `matched` and `broad` levels.

Stage 1 results are cached under <candidates-dir>/<user_id>.json so the same
candidate set is reused across teacher / sft / zero_shot runs.

Usage:
    # Once (shared across models) — produces candidates cache:
    python scripts/eval_m2_layer1_delta_recall.py \
        --config configs/eval/layer1_delta.yaml \
        --predictions eval_results/layer1_delta/predictions/sft.jsonl \
        --test-jsonl  data/splits/layer1_delta/test.jsonl \
        --output      eval_results/layer1_delta/m2_recall/recall_sft.json \
        --model-tag   sft \
        --candidates-dir eval_results/layer1_delta/m2_recall/candidates

    # Subsequent models reuse the cache automatically.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncAzureOpenAI

logger = logging.getLogger("eval_m2_layer1_delta_recall")


# ---------------------------------------------------------------------------
# Prompt paths (point at MaiProfile repo)
# ---------------------------------------------------------------------------
MAIPROFILE_PROMPTS = Path(
    os.environ.get(
        "MAIPROFILE_PROMPTS_DIR",
        "/scratch/azureml/cr/j/65fcf9508e03476381b75ace1f02fb73/exe/wd/"
        "MaiProfile-main 1/MaiProfile-main/maiprofilev3dev/evaluation/prompts",
    )
)
PROPOSE_PROMPT_PATH = MAIPROFILE_PROMPTS / "interest_recall_propose.md"
GROUND_PROMPT_PATH = MAIPROFILE_PROMPTS / "interest_recall_ground.md"
RESCUE_PROMPT_PATH = MAIPROFILE_PROMPTS / "interest_recall_rescue.md"
JUDGE_PROMPT_PATH = MAIPROFILE_PROMPTS / "interest_recall_judge.md"

# Chunk size for the per-model coverage judge (matches official
# interest_recall.py JUDGE_BATCH_SIZE=10); avoids token-budget / malformed
# output on large/complex users.
JUDGE_BATCH_SIZE = 10


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def safe_json_loads(text: str) -> Any:
    if not text:
        return None
    s = text.strip()
    m = _FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        if start < 0:
            continue
        depth = 0
        for i, ch in enumerate(s[start:], start):
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except Exception:
                        break
    return None


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


# ---------------------------------------------------------------------------
# Signal -> keyword extraction
# ---------------------------------------------------------------------------
# Layer1_delta user message contains "Today's Denoised Signals (N events):" then
# a JSON array of {Action, intent, ...}. The proposer expects topic keywords —
# the `intent` field is the per-signal layer0 intent, which IS the keyword.
_EVENTS_RE = re.compile(r"Today's Denoised Signals[^\[]*?(\[.*\])", re.DOTALL)


def extract_keywords_from_signal(user_message: str) -> list[str]:
    """Pull the per-event intent strings as keyword inputs for the proposer."""
    if not user_message:
        return []
    m = _EVENTS_RE.search(user_message)
    if not m:
        return []
    try:
        events = json.loads(m.group(1))
    except Exception:
        return []
    if not isinstance(events, list):
        return []
    keywords: list[str] = []
    seen: set[str] = set()
    for ev in events:
        if not isinstance(ev, dict):
            continue
        intent = (ev.get("intent") or "").strip()
        action = (ev.get("Action") or "").strip()
        kw = intent if intent else action
        if kw and kw not in seen:
            seen.add(kw)
            keywords.append(kw)
    return keywords


# ---------------------------------------------------------------------------
# Judge call (single retry-wrapped completion)
# ---------------------------------------------------------------------------
async def call_judge(
    client: AsyncAzureOpenAI,
    deployment: str,
    system_prompt: str,
    user_prompt: str,
    max_completion_tokens: int,
    reasoning_effort: str | None,
    sem: asyncio.Semaphore,
    max_retries: int = 3,
) -> tuple[Any, float]:
    async with sem:
        for attempt in range(max_retries):
            try:
                t0 = time.perf_counter()
                kwargs: dict[str, Any] = {
                    "model": deployment,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_completion_tokens": max_completion_tokens,
                }
                if reasoning_effort:
                    kwargs["reasoning_effort"] = reasoning_effort
                resp = await client.chat.completions.create(**kwargs)
                dt = time.perf_counter() - t0
                text = resp.choices[0].message.content or ""
                return safe_json_loads(text), dt
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning("judge attempt %d/%d failed: %s; retrying in %ds",
                               attempt + 1, max_retries, exc, wait)
                await asyncio.sleep(wait)
        return None, 0.0


# ---------------------------------------------------------------------------
# Stage 1: build candidate set (cached per user_id)
# ---------------------------------------------------------------------------
async def build_candidates_for_user(
    client: AsyncAzureOpenAI,
    deployment: str,
    keywords: list[str],
    sem: asyncio.Semaphore,
    propose_prompt: str,
    ground_prompt: str,
    rescue_prompt: str,
    max_tok: int,
    re_effort: str | None,
    enable_rescue: bool,
) -> dict[str, Any]:
    """Returns dict with grounded candidate proposals + stage stats."""
    info: dict[str, Any] = {
        "keywords": keywords,
        "n_keywords": len(keywords),
        "stages": {},
    }
    if not keywords:
        info["grounded"] = []
        return info

    # --- Stage 1a: propose ---
    user_propose = json.dumps(keywords, ensure_ascii=False, indent=2)
    propose_res, _ = await call_judge(
        client, deployment, propose_prompt, user_propose,
        max_tok, re_effort, sem,
    )
    proposals: list[dict[str, Any]] = []
    if isinstance(propose_res, dict) and isinstance(propose_res.get("proposals"), list):
        for p in propose_res["proposals"]:
            if not isinstance(p, dict):
                continue
            name = (p.get("interest_name") or "").strip()
            level = p.get("granularity_level")
            kws = p.get("keywords") or []
            if not name or level not in ("matched", "broad"):
                continue
            if not isinstance(kws, list):
                continue
            proposals.append({
                "proposal": name,
                "granularity_level": level,
                "cited_keywords": [k for k in kws if isinstance(k, str)],
            })
    info["stages"]["propose"] = {"n_raw_proposals": len(proposals)}
    if not proposals:
        info["grounded"] = []
        return info

    # --- Stage 1b: ground ---
    ground_payload = {
        "keywords": keywords,
        "proposals": proposals,
    }
    ground_res, _ = await call_judge(
        client, deployment, ground_prompt,
        json.dumps(ground_payload, ensure_ascii=False, indent=2),
        max_tok, re_effort, sem,
    )
    validated_map: dict[tuple[str, str], dict[str, Any]] = {}
    if isinstance(ground_res, dict) and isinstance(ground_res.get("validated"), list):
        for entry in ground_res["validated"]:
            if not isinstance(entry, dict):
                continue
            name = (entry.get("proposal") or "").strip()
            level = entry.get("granularity_level")
            valid = bool(entry.get("valid"))
            if not name or level not in ("matched", "broad"):
                continue
            validated_map[(name, level)] = entry

    grounded: list[dict[str, Any]] = []
    rejected_matched: list[dict[str, Any]] = []
    for p in proposals:
        key = (p["proposal"], p["granularity_level"])
        v = validated_map.get(key)
        if v and v.get("valid"):
            grounded.append({**p, "ground_reason": v.get("reason", "")})
        elif p["granularity_level"] == "matched":
            rejected_matched.append({
                "proposal": p["proposal"],
                "keywords": p.get("cited_keywords", []),
                "reason": (v.get("reason") if v else "") or "rejected or missing in ground output",
            })
    info["stages"]["ground"] = {
        "n_validated": sum(1 for v in validated_map.values() if v.get("valid")),
        "n_grounded_kept": len(grounded),
        "n_rejected_matched": len(rejected_matched),
    }

    # --- Stage 1c: rescue (matched only) ---
    if enable_rescue:
        covered_by_matched: set[str] = set()
        for g in grounded:
            if g["granularity_level"] == "matched":
                for kw in g.get("cited_keywords", []):
                    covered_by_matched.add(kw)
        orphan_keywords = [k for k in keywords if k not in covered_by_matched]
        info["stages"]["rescue"] = {
            "n_orphan_keywords": len(orphan_keywords),
        }
        if orphan_keywords:
            rescue_payload = {
                "orphan_keywords": orphan_keywords,
                "rejected_proposals": rejected_matched[:30],
            }
            rescue_res, _ = await call_judge(
                client, deployment, rescue_prompt,
                json.dumps(rescue_payload, ensure_ascii=False, indent=2),
                max_tok, re_effort, sem,
            )
            n_added = 0
            if isinstance(rescue_res, dict) and isinstance(rescue_res.get("proposals"), list):
                for p in rescue_res["proposals"]:
                    if not isinstance(p, dict):
                        continue
                    name = (p.get("interest_name") or "").strip()
                    kws = p.get("keywords") or []
                    if not name or not isinstance(kws, list):
                        continue
                    # Rescue produces matched proposals only; treat as valid by construction.
                    grounded.append({
                        "proposal": name,
                        "granularity_level": "matched",
                        "cited_keywords": [k for k in kws if isinstance(k, str)],
                        "ground_reason": "rescued",
                    })
                    n_added += 1
            info["stages"]["rescue"]["n_added"] = n_added
    else:
        info["stages"]["rescue"] = {"skipped": True}

    info["grounded"] = grounded
    return info


# ---------------------------------------------------------------------------
# Stage 2: per-model coverage judge
# ---------------------------------------------------------------------------
def collect_model_interests(prediction_text: str) -> list[str]:
    obj = safe_json_loads(prediction_text)
    if not isinstance(obj, dict):
        return []
    interests = obj.get("interests") or []
    if not isinstance(interests, list):
        return []
    names: list[str] = []
    seen: set[str] = set()
    for it in interests:
        if not isinstance(it, dict):
            continue
        name = (it.get("interest_name") or "").strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


async def judge_coverage(
    client: AsyncAzureOpenAI,
    deployment: str,
    judge_prompt: str,
    grounded: list[dict[str, Any]],
    model_interest_names: list[str],
    sem: asyncio.Semaphore,
    max_tok: int,
    re_effort: str | None,
) -> list[dict[str, Any]]:
    """Returns a list of result entries (one per grounded proposal)."""
    if not grounded:
        return []
    # If the model output zero interests, every proposal is uncovered.
    if not model_interest_names:
        return [
            {
                "proposal": g["proposal"],
                "granularity_level": g["granularity_level"],
                "covered": False,
                "matched_interest": None,
                "reason": "model produced no interests",
            }
            for g in grounded
        ]
    # Pass ALL model interests as candidates (per-user n is small, ~10).
    # Batch proposals (JUDGE_BATCH_SIZE) so large/complex users don't blow the
    # token budget or trigger malformed/empty judge output (matches official
    # interest_recall.py JUDGE_BATCH_SIZE=10).
    candidates = [
        {"interest_name": n, "similarity": 1.0} for n in model_interest_names
    ]
    out: list[dict[str, Any]] = []
    for start in range(0, len(grounded), JUDGE_BATCH_SIZE):
        chunk = grounded[start:start + JUDGE_BATCH_SIZE]
        judge_input = [
            {
                "proposal": g["proposal"],
                "granularity_level": g["granularity_level"],
                "candidates": candidates,
            }
            for g in chunk
        ]
        res, _ = await call_judge(
            client, deployment, judge_prompt,
            json.dumps(judge_input, ensure_ascii=False, indent=2),
            max_tok, re_effort, sem,
        )
        if isinstance(res, dict) and isinstance(res.get("results"), list):
            out.extend(r for r in res["results"] if isinstance(r, dict))
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def stats(xs: list[float]) -> dict[str, Any]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": round(statistics.mean(xs), 4),
        "median": round(statistics.median(xs), 4),
    }


def bucket_of(n: int) -> str:
    if n <= 0:
        return "0_empty_gold"
    if n == 1:
        return "1_simple"
    if n <= 3:
        return "2-3_small"
    if n <= 5:
        return "4-5_medium"
    return "6+_complex"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main_async(args) -> int:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    jcfg = cfg["judge"]

    api_key = os.environ.get("AZURE_OPENAI_KEY")
    if not api_key:
        logger.error("AZURE_OPENAI_KEY not set in environment")
        return 2
    client = AsyncAzureOpenAI(
        api_key=api_key,
        api_version=jcfg["api_version"],
        azure_endpoint=jcfg["azure_endpoint"],
    )
    deployment = jcfg["deployment"]
    concurrency = int(args.concurrency or jcfg.get("concurrency", 8))
    max_tok = int(jcfg.get("max_completion_tokens", 16000))
    re_effort = jcfg.get("reasoning_effort")

    propose_prompt = PROPOSE_PROMPT_PATH.read_text()
    ground_prompt = GROUND_PROMPT_PATH.read_text()
    rescue_prompt = RESCUE_PROMPT_PATH.read_text()
    judge_prompt = JUDGE_PROMPT_PATH.read_text()

    # --- Build user_id -> keywords from test_jsonl ---
    test_records = load_jsonl(args.test_jsonl)
    uid2keywords: dict[str, list[str]] = {}
    uid2gold_n: dict[str, int] = {}
    for r in test_records:
        meta = r.get("metadata", {}) or {}
        uid = meta.get("user_id") or r.get("user_id") or ""
        if not uid:
            continue
        msgs = r.get("messages", [])
        user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        uid2keywords[uid] = extract_keywords_from_signal(user_msg)
        # gold interest count (for complexity bucketing)
        asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        try:
            g = json.loads(asst) if asst.strip() else {}
            uid2gold_n[uid] = len(g.get("interests", []) or [])
        except Exception:
            uid2gold_n[uid] = 0
    logger.info("Loaded %d test records, %d with keywords", len(test_records),
                sum(1 for k in uid2keywords.values() if k))

    # --- Load predictions for the model under evaluation ---
    preds = load_jsonl(args.predictions)
    if args.limit:
        preds = preds[: args.limit]
    logger.info("Loaded %d predictions (model_tag=%s)", len(preds), args.model_tag)

    # --- Set up output paths ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_suffix(".jsonl")
    cand_dir = Path(args.candidates_dir)
    cand_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)

    # ------------------------------------------------------------------
    # Stage 1: produce / load candidate set per user (cached across models)
    # ------------------------------------------------------------------
    async def ensure_candidates(uid: str) -> dict[str, Any]:
        cache_path = cand_dir / f"{uid}.json"
        if cache_path.exists() and not args.rebuild_candidates:
            try:
                return json.loads(cache_path.read_text())
            except Exception:
                logger.warning("Cache corrupt for %s, rebuilding", uid)
        keywords = uid2keywords.get(uid, [])
        info = await build_candidates_for_user(
            client, deployment, keywords, sem,
            propose_prompt, ground_prompt, rescue_prompt,
            max_tok, re_effort, enable_rescue=not args.skip_rescue,
        )
        try:
            cache_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.warning("Failed to cache %s: %s", uid, e)
        return info

    # ------------------------------------------------------------------
    # Stage 2 per record: judge coverage
    # ------------------------------------------------------------------
    raw_fh = open(raw_path, "w", encoding="utf-8")
    n_records = 0
    n_with_pred = 0
    n_with_candidates = 0
    per_record: list[dict[str, Any]] = []  # for per-bucket aggregation
    matched_recalls: list[float] = []
    broad_recalls: list[float] = []
    overall_recalls: list[float] = []
    bucket_to_recalls: dict[str, list[float]] = defaultdict(list)
    bucket_to_matched: dict[str, list[float]] = defaultdict(list)
    bucket_to_broad: dict[str, list[float]] = defaultdict(list)

    async def process_one(p: dict[str, Any]) -> None:
        nonlocal n_records, n_with_pred, n_with_candidates
        uid = p.get("user_id", "")
        if not uid:
            return
        n_records += 1
        # Stage 1: candidates (cached)
        cand = await ensure_candidates(uid)
        grounded = cand.get("grounded", [])
        if not grounded:
            return
        n_with_candidates += 1

        # Stage 2: collect model interest names
        text = p["reference"] if args.teacher_mode else p["prediction"]
        if not text:
            return
        n_with_pred += 1
        model_names = collect_model_interests(text)

        results = await judge_coverage(
            client, deployment, judge_prompt, grounded, model_names,
            sem, max_tok, re_effort,
        )
        # Compute recalls
        m_total = sum(1 for g in grounded if g["granularity_level"] == "matched")
        b_total = sum(1 for g in grounded if g["granularity_level"] == "broad")
        m_cov = 0
        b_cov = 0
        # build proposal -> level lookup
        prop2level = {g["proposal"]: g["granularity_level"] for g in grounded}
        for r in results:
            if r.get("covered"):
                lvl = prop2level.get(r.get("proposal"))
                if lvl == "matched":
                    m_cov += 1
                elif lvl == "broad":
                    b_cov += 1
        total = m_total + b_total
        covered = m_cov + b_cov
        overall = covered / total if total else None
        matched_r = m_cov / m_total if m_total else None
        broad_r = b_cov / b_total if b_total else None

        bucket = bucket_of(uid2gold_n.get(uid, 0))
        record_summary = {
            "user_id": uid,
            "delta_index": p.get("delta_index", -1),
            "bucket": bucket,
            "n_keywords": cand.get("n_keywords", 0),
            "n_grounded_total": total,
            "n_grounded_matched": m_total,
            "n_grounded_broad": b_total,
            "n_model_interests": len(model_names),
            "n_covered_matched": m_cov,
            "n_covered_broad": b_cov,
            "recall_matched": matched_r,
            "recall_broad": broad_r,
            "recall_overall": overall,
            "results": results,
        }
        raw_fh.write(json.dumps(record_summary, ensure_ascii=False) + "\n")
        raw_fh.flush()
        per_record.append(record_summary)
        if overall is not None:
            overall_recalls.append(overall)
            bucket_to_recalls[bucket].append(overall)
        if matched_r is not None:
            matched_recalls.append(matched_r)
            bucket_to_matched[bucket].append(matched_r)
        if broad_r is not None:
            broad_recalls.append(broad_r)
            bucket_to_broad[bucket].append(broad_r)

    try:
        await asyncio.gather(*(process_one(p) for p in preds))
    finally:
        raw_fh.close()

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    summary = {
        "model_tag": args.model_tag,
        "deployment": deployment,
        "n_records": n_records,
        "n_with_pred": n_with_pred,
        "n_with_candidates": n_with_candidates,
        "n_grounded_total_sum": sum(r["n_grounded_total"] for r in per_record),
        "n_grounded_matched_sum": sum(r["n_grounded_matched"] for r in per_record),
        "n_grounded_broad_sum": sum(r["n_grounded_broad"] for r in per_record),
        "recall_overall": stats(overall_recalls),
        "recall_matched": stats(matched_recalls),
        "recall_broad": stats(broad_recalls),
        "by_bucket": {
            b: {
                "n_rec": len(bucket_to_recalls.get(b, [])),
                "recall_overall": stats(bucket_to_recalls.get(b, [])),
                "recall_matched": stats(bucket_to_matched.get(b, [])),
                "recall_broad": stats(bucket_to_broad.get(b, [])),
            }
            for b in ["1_simple", "2-3_small", "4-5_medium", "6+_complex"]
        },
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info("Wrote %s and %s", out_path, raw_path)
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--candidates-dir", required=True,
                    help="Cache dir for Stage 1 candidate sets, shared across models")
    ap.add_argument("--teacher-mode", action="store_true",
                    help="Use `reference` field instead of `prediction`")
    ap.add_argument("--skip-rescue", action="store_true",
                    help="Skip the rescue step (faster, fewer matched candidates)")
    ap.add_argument("--rebuild-candidates", action="store_true",
                    help="Force re-running Stage 1 even if cache exists")
    ap.add_argument("--concurrency", type=int, default=0,
                    help="Override judge concurrency (default: config judge.concurrency)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
