"""eval_m2_layer1_delta_topic_recall.py — Topic-level (keyword) recall for layer1_delta.

Sibling of eval_m2_layer1_delta_recall.py (which does INTEREST-level recall).
This one measures whether a model's TOPICS cover the topics that *should* exist,
proposed from the user's RAW signals (Action text) — i.e. it catches signal→topic
loss. Mirrors the official maiprofilev3dev/evaluation/keyword_recall.py.

Pipeline (single-agent gpt-5.1, NO SBERT — pass all model topics as candidates):

  Stage 1 (model-INDEPENDENT, cached per user):
    propose : raw signals (Action text) -> candidate keyword/topics
    ground  : validate proposals against the raw signals (drop hallucinated)
    (NO rescue — keyword_recall has none.)
  Stage 2 (per-model):
    judge   : is each validated proposal covered by the model's topics?

  recall = covered_proposals / validated_proposals     (single granularity)

Prompts (env MAIPROFILE_PROMPTS_DIR, default local MaiProfile repo):
    keyword_recall_propose.md
    keyword_recall_ground.md
    keyword_recall_judge.md

Usage (Stage-1 candidates built once, cached, reused across all models):
    python scripts/eval_m2_layer1_delta_topic_recall.py \
        --config configs/eval/layer1_delta_thinking_50k_4o-v1.local.yaml \
        --predictions eval_results/predictions/gpt5_1k.jsonl \
        --test-jsonl  data/splits/layer1_delta_thinking_50k_4o/test_1k.jsonl \
        --output      eval_results/m2_recall/topic/gpt5.json \
        --model-tag   gpt5 \
        --candidates-dir eval_results/m2_recall/topic/candidates \
        [--teacher-mode] [--limit N]
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

logger = logging.getLogger("eval_m2_layer1_delta_topic_recall")


# ---------------------------------------------------------------------------
# Prompt paths (point at MaiProfile repo; env-overridable)
# ---------------------------------------------------------------------------
MAIPROFILE_PROMPTS = Path(
    os.environ.get(
        "MAIPROFILE_PROMPTS_DIR",
        "/scratch/azureml/cr/j/65fcf9508e03476381b75ace1f02fb73/exe/wd/"
        "MaiProfile-main 1/MaiProfile-main/maiprofilev3dev/evaluation/prompts",
    )
)
PROPOSE_PROMPT_PATH = MAIPROFILE_PROMPTS / "keyword_recall_propose.md"
GROUND_PROMPT_PATH = MAIPROFILE_PROMPTS / "keyword_recall_ground.md"
JUDGE_PROMPT_PATH = MAIPROFILE_PROMPTS / "keyword_recall_judge.md"

# Chunk size for the per-model coverage judge (matches official
# keyword_recall.py JUDGE_BATCH_SIZE=10); avoids token-budget / malformed
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


def parse_prediction(text: str) -> Any:
    """Strip <think>, strip fences (even unclosed), object-first balanced extraction."""
    if not text:
        return None
    s = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    s = re.sub(r"```json|```", "", s)
    i = s.find("{")
    if i < 0:
        return safe_json_loads(text)
    depth = 0
    instr = False
    esc = False
    end = -1
    for j in range(i, len(s)):
        c = s[j]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            instr = not instr
            continue
        if instr:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    if end > 0:
        try:
            return json.loads(s[i:end])
        except Exception:
            pass
    return safe_json_loads(text)


# ---------------------------------------------------------------------------
# Signal -> raw event extraction (Action text, like keyword_recall.py)
# ---------------------------------------------------------------------------
_EVENTS_RE = re.compile(r"Today's Denoised Signals[^\[]*?(\[.*\])", re.DOTALL)


def extract_signals_from_message(user_message: str) -> list[dict[str, str]]:
    """Pull per-event {date, source, action} dicts (the proposer's input)."""
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
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for ev in events:
        if not isinstance(ev, dict):
            continue
        action = (ev.get("Action") or ev.get("action") or "").strip()
        if not action or action in seen:
            continue
        seen.add(action)
        out.append({
            "date": str(ev.get("Date", ev.get("date", ""))),
            "source": str(ev.get("Source", ev.get("source", ""))),
            "action": action,
        })
    return out


def collect_model_topics(prediction_text: str) -> list[str]:
    """Flatten interests[].topics[].topic into a deduped list of topic names."""
    obj = parse_prediction(prediction_text)
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
        topics = it.get("topics") or []
        if not isinstance(topics, list):
            continue
        for tp in topics:
            if isinstance(tp, dict):
                name = (tp.get("topic") or tp.get("topic_name") or "").strip()
            elif isinstance(tp, str):
                name = tp.strip()
            else:
                name = ""
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


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
# Stage 1: build candidate set (cached per user) — propose -> ground
# ---------------------------------------------------------------------------
async def build_candidates_for_user(
    client: AsyncAzureOpenAI,
    deployment: str,
    signals: list[dict[str, str]],
    sem: asyncio.Semaphore,
    propose_prompt: str,
    ground_prompt: str,
    max_tok: int,
    re_effort: str | None,
) -> dict[str, Any]:
    """Returns dict with grounded (validated) keyword/topic proposals + stage stats."""
    info: dict[str, Any] = {
        "n_signals": len(signals),
        "stages": {},
    }
    if not signals:
        info["grounded"] = []
        return info

    # --- Stage 1a: propose ---
    user_propose = json.dumps(signals, ensure_ascii=False)
    propose_res, _ = await call_judge(
        client, deployment, propose_prompt, user_propose,
        max_tok, re_effort, sem,
    )
    proposals: list[dict[str, Any]] = []
    if isinstance(propose_res, dict) and isinstance(propose_res.get("proposals"), list):
        for p in propose_res["proposals"]:
            if not isinstance(p, dict):
                continue
            name = (p.get("keyword") or "").strip()
            sigs = p.get("signals") or []
            if not name:
                continue
            if not isinstance(sigs, list):
                sigs = []
            proposals.append({
                "keyword": name,
                "cited_signals": [s for s in sigs if isinstance(s, str)],
            })
    info["stages"]["propose"] = {"n_raw_proposals": len(proposals)}
    if not proposals:
        info["grounded"] = []
        return info

    # --- Stage 1b: ground ---
    ground_payload = {
        "signals": [s["action"] for s in signals],
        "proposals": [
            {"keyword": p["keyword"], "cited_signals": p["cited_signals"]}
            for p in proposals
        ],
    }
    ground_res, _ = await call_judge(
        client, deployment, ground_prompt,
        json.dumps(ground_payload, ensure_ascii=False),
        max_tok, re_effort, sem,
    )
    valid_names: dict[str, str] = {}
    if isinstance(ground_res, dict) and isinstance(ground_res.get("validated"), list):
        for entry in ground_res["validated"]:
            if not isinstance(entry, dict):
                continue
            name = (entry.get("keyword") or "").strip()
            if name and bool(entry.get("valid")):
                valid_names[name] = entry.get("reason", "")

    grounded: list[dict[str, Any]] = []
    for p in proposals:
        if p["keyword"] in valid_names:
            grounded.append({**p, "ground_reason": valid_names[p["keyword"]]})
    info["stages"]["ground"] = {
        "n_validated": len(valid_names),
        "n_grounded_kept": len(grounded),
    }
    info["grounded"] = grounded
    return info


# ---------------------------------------------------------------------------
# Stage 2: per-model coverage judge
# ---------------------------------------------------------------------------
async def judge_coverage(
    client: AsyncAzureOpenAI,
    deployment: str,
    judge_prompt: str,
    grounded: list[dict[str, Any]],
    model_topic_names: list[str],
    sem: asyncio.Semaphore,
    max_tok: int,
    re_effort: str | None,
) -> list[dict[str, Any]]:
    """Returns a list of result entries (one per grounded proposal)."""
    if not grounded:
        return []
    if not model_topic_names:
        return [
            {
                "proposal": g["keyword"],
                "covered": False,
                "matched_keyword": None,
                "reason": "model produced no topics",
            }
            for g in grounded
        ]
    # NO SBERT: pass ALL model topics as candidates (per-user n small ~7-15).
    # Batch proposals (JUDGE_BATCH_SIZE) so large/complex users don't blow the
    # token budget or trigger malformed/empty judge output.
    candidates = [{"keyword": n, "similarity": 1.0} for n in model_topic_names]
    out: list[dict[str, Any]] = []
    for start in range(0, len(grounded), JUDGE_BATCH_SIZE):
        chunk = grounded[start:start + JUDGE_BATCH_SIZE]
        judge_input = [
            {"proposal": g["keyword"], "candidates": candidates}
            for g in chunk
        ]
        res, _ = await call_judge(
            client, deployment, judge_prompt,
            json.dumps(judge_input, ensure_ascii=False),
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
    judge_prompt = JUDGE_PROMPT_PATH.read_text()

    # --- Build user_id -> signals from test_jsonl ---
    test_records = load_jsonl(args.test_jsonl)
    uid2signals: dict[str, list[dict[str, str]]] = {}
    uid2gold_n: dict[str, int] = {}
    for r in test_records:
        meta = r.get("metadata", {}) or {}
        uid = meta.get("user_id") or r.get("user_id") or ""
        if not uid:
            continue
        msgs = r.get("messages", [])
        user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        uid2signals[uid] = extract_signals_from_message(user_msg)
        asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        try:
            g = json.loads(asst) if asst.strip() else {}
            uid2gold_n[uid] = len(g.get("interests", []) or [])
        except Exception:
            uid2gold_n[uid] = 0
    logger.info("Loaded %d test records, %d with signals", len(test_records),
                sum(1 for s in uid2signals.values() if s))

    # --- Load predictions for the model under evaluation ---
    preds = load_jsonl(args.predictions)
    if args.limit:
        preds = preds[: args.limit]
    logger.info("Loaded %d predictions (model_tag=%s)", len(preds), args.model_tag)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_suffix(".jsonl")
    cand_dir = Path(args.candidates_dir)
    cand_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)

    async def ensure_candidates(uid: str) -> dict[str, Any]:
        cache_path = cand_dir / f"{uid}.json"
        if cache_path.exists() and not args.rebuild_candidates:
            try:
                return json.loads(cache_path.read_text())
            except Exception:
                logger.warning("Cache corrupt for %s, rebuilding", uid)
        signals = uid2signals.get(uid, [])
        info = await build_candidates_for_user(
            client, deployment, signals, sem,
            propose_prompt, ground_prompt, max_tok, re_effort,
        )
        try:
            cache_path.write_text(json.dumps(info, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.warning("Failed to cache %s: %s", uid, e)
        return info

    raw_fh = open(raw_path, "w", encoding="utf-8")
    n_records = 0
    n_with_pred = 0
    n_with_candidates = 0
    per_record: list[dict[str, Any]] = []
    recalls: list[float] = []
    bucket_to_recalls: dict[str, list[float]] = defaultdict(list)

    async def process_one(p: dict[str, Any]) -> None:
        nonlocal n_records, n_with_pred, n_with_candidates
        uid = p.get("user_id", "")
        if not uid:
            return
        n_records += 1
        cand = await ensure_candidates(uid)
        grounded = cand.get("grounded", [])
        if not grounded:
            return
        n_with_candidates += 1

        text = p["reference"] if args.teacher_mode else p["prediction"]
        if not text:
            return
        n_with_pred += 1
        model_topics = collect_model_topics(text)

        results = await judge_coverage(
            client, deployment, judge_prompt, grounded, model_topics,
            sem, max_tok, re_effort,
        )
        total = len(grounded)
        covered = sum(1 for r in results if r.get("covered"))
        recall = covered / total if total else None

        bucket = bucket_of(uid2gold_n.get(uid, 0))
        record_summary = {
            "user_id": uid,
            "delta_index": p.get("delta_index", -1),
            "bucket": bucket,
            "n_signals": cand.get("n_signals", 0),
            "n_grounded_total": total,
            "n_model_topics": len(model_topics),
            "n_covered": covered,
            "recall": recall,
            "results": results,
        }
        raw_fh.write(json.dumps(record_summary, ensure_ascii=False) + "\n")
        raw_fh.flush()
        per_record.append(record_summary)
        if recall is not None:
            recalls.append(recall)
            bucket_to_recalls[bucket].append(recall)

    try:
        await asyncio.gather(*(process_one(p) for p in preds))
    finally:
        raw_fh.close()

    summary = {
        "model_tag": args.model_tag,
        "deployment": deployment,
        "n_records": n_records,
        "n_with_pred": n_with_pred,
        "n_with_candidates": n_with_candidates,
        "n_grounded_total_sum": sum(r["n_grounded_total"] for r in per_record),
        "recall": stats(recalls),
        "by_bucket": {
            b: {
                "n_rec": len(bucket_to_recalls.get(b, [])),
                "recall": stats(bucket_to_recalls.get(b, [])),
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
    ap.add_argument("--rebuild-candidates", action="store_true",
                    help="Force re-running Stage 1 even if cache exists")
    ap.add_argument("--concurrency", type=int, default=0,
                    help="Override judge concurrency (default: config judge.concurrency)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
