"""eval_m2_layer1_delta_judge.py — LLM-as-judge for layer1_delta SFT.

Runs the two official MaiProfile evaluation prompts via gpt-5.1 on Azure:

  1. interest_name (v2): score each predicted interest_name on
       utility, precision, coherence, granularity_broad   (1-10 / 0-10 / 1-10 / 0-1)
  2. topics: flatten all topics with their evidence and score each on
       utility, precision, coherence, granularity         (1-10 / 0-10 / 1-10 / 0-1)

Prompts are loaded VERBATIM from the MaiProfile repo at:
    MaiProfile-main 1/MaiProfile-main/maiprofilev3dev/evaluation/prompts/
to keep semantics identical to production.

NOT replicated here (left as TODO for future fidelity, not needed for SFT
diagnosis):
  - V2 DBSCAN clustering for overall_granularity (we report granularity_broad
    per-interest only)
  - Per-user recall via multi-agent proposals
  - Keyword recall computation in topics evaluator

Inputs:
    predictions.jsonl (from generate_outputs.py)
    layer1_delta config yaml (judge endpoint + deployment + concurrency)

Outputs:
    m2_<tag>.json                  — aggregated summary
    m2_<tag>.interest_name.jsonl   — per-record per-interest raw scores
    m2_<tag>.topics.jsonl          — per-record per-topic raw scores

Usage:
    python scripts/eval_m2_layer1_delta_judge.py \
        --config configs/eval/layer1_delta.yaml \
        --predictions eval_results/layer1_delta/predictions/sft.jsonl \
        --output      eval_results/layer1_delta/m2_judge/m2_sft.json \
        --model-tag   sft

    # Teacher (uses `reference` as the prediction)
    python scripts/eval_m2_layer1_delta_judge.py \
        --config configs/eval/layer1_delta.yaml \
        --predictions eval_results/layer1_delta/predictions/teacher.jsonl \
        --output      eval_results/layer1_delta/m2_judge/m2_teacher.json \
        --model-tag   teacher --teacher-mode
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
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncAzureOpenAI

logger = logging.getLogger("eval_m2_layer1_delta_judge")


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
INTEREST_NAME_PROMPT_PATH = MAIPROFILE_PROMPTS / "layer1_delta_eval_interest_name_v2.md"
TOPICS_PROMPT_PATH = MAIPROFILE_PROMPTS / "layer1_delta_eval_topics.md"


# ---------------------------------------------------------------------------
# JSON helpers
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
    # Extract first JSON array OR object — judges return arrays
    for opener, closer in (("[", "]"), ("{", "}")):
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


_THINK_RE = re.compile(r"</think>", re.IGNORECASE)
_FENCE_OPEN_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE_RE = re.compile(r"\s*```$")


def parse_prediction(text: str) -> Any:
    """Parse a model PREDICTION into the layer1-delta object.

    Robust to thinking models: strips the ``<think>`` block, strips code
    fences even when the closing fence is missing (truncated output), and
    prefers an object ``{...}`` over an array ``[...]`` so the inner
    ``"interests": [...]`` array is not mistaken for the top-level value.
    Returns a dict (``{"interests": [...]}``) or None.
    """
    if not text:
        return None
    s = text.strip()
    s = _THINK_RE.split(s)[-1].strip()
    s = _FENCE_OPEN_RE.sub("", s).strip()
    s = _FENCE_CLOSE_RE.sub("", s).strip()
    obj: Any = None
    try:
        obj = json.loads(s)
    except Exception:
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
                            obj = json.loads(s[start:i + 1])
                        except Exception:
                            obj = None
                        break
            if obj is not None:
                break
    # Accept a top-level array of interests by wrapping it.
    if isinstance(obj, list):
        return {"interests": obj}
    return obj


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
# Build user prompts
# ---------------------------------------------------------------------------
def build_interest_name_user_prompt(interests: list[dict[str, Any]]) -> str:
    """Per the v2 prompt: pass interest_name + topic name strings only."""
    payload = []
    for it in interests:
        if not isinstance(it, dict):
            continue
        topic_names: list[str] = []
        for t in it.get("topics", []) or []:
            if isinstance(t, dict) and isinstance(t.get("topic"), str):
                topic_names.append(t["topic"])
        payload.append({
            "interest_name": it.get("interest_name", ""),
            "topics": topic_names,
        })
    return json.dumps({"interests": payload}, ensure_ascii=False, indent=2)


def build_topics_user_prompt(interests: list[dict[str, Any]]) -> str:
    """Per the topics prompt: flat list of topics with own evidence only."""
    flat = []
    for it in interests:
        if not isinstance(it, dict):
            continue
        interest_name = it.get("interest_name", "")
        for t in it.get("topics", []) or []:
            if not isinstance(t, dict):
                continue
            evidence_actions = []
            for e in t.get("evidence", []) or []:
                if isinstance(e, dict) and isinstance(e.get("action"), str):
                    evidence_actions.append(e["action"])
            flat.append({
                "interest_name": interest_name,
                "topic": t.get("topic", ""),
                "source": t.get("source", []),
                "evidence": evidence_actions,
            })
    return json.dumps({"topics": flat}, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Judge call
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
                logger.warning("judge attempt %d/%d failed: %s. retrying in %ds",
                               attempt + 1, max_retries, exc, wait)
                await asyncio.sleep(wait)
        return None, 0.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": round(statistics.mean(xs), 4),
        "median": round(statistics.median(xs), 4),
    }


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
    concurrency = int(jcfg.get("concurrency", 8))
    max_tok = int(jcfg.get("max_completion_tokens", 16000))
    re_effort = jcfg.get("reasoning_effort")

    interest_prompt = INTEREST_NAME_PROMPT_PATH.read_text()
    topics_prompt = TOPICS_PROMPT_PATH.read_text()

    preds = load_jsonl(args.predictions)
    if args.limit:
        preds = preds[: args.limit]
    logger.info("Loaded %d predictions; judging with deployment=%s, concurrency=%d",
                len(preds), deployment, concurrency)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    interest_raw_path = out_path.with_suffix(".interest_name.jsonl")
    topics_raw_path = out_path.with_suffix(".topics.jsonl")

    sem = asyncio.Semaphore(concurrency)

    interest_scores: dict[str, list[float]] = {k: [] for k in
                                              ("utility", "precision", "coherence", "granularity_broad")}
    topic_scores: dict[str, list[float]] = {k: [] for k in
                                           ("utility", "precision", "coherence", "granularity")}
    n_interest_judged = 0
    n_topic_judged = 0
    n_records_with_pred = 0
    n_parse_fail = 0
    n_judge_fail = 0

    interest_raw_fh = open(interest_raw_path, "w", encoding="utf-8")
    topics_raw_fh = open(topics_raw_path, "w", encoding="utf-8")

    async def judge_one(p: dict[str, Any]) -> None:
        nonlocal n_interest_judged, n_topic_judged, n_parse_fail, n_judge_fail, n_records_with_pred
        text = p["prediction"] if not args.teacher_mode else p["reference"]
        if not text:
            return
        n_records_with_pred += 1
        obj = parse_prediction(text)
        if not isinstance(obj, dict):
            n_parse_fail += 1
            return
        interests = obj.get("interests") or []
        if not isinstance(interests, list) or not interests:
            return

        i_user = build_interest_name_user_prompt(interests)
        t_user = build_topics_user_prompt(interests)

        # Fire both judges concurrently for this record
        (i_res, _), (t_res, _) = await asyncio.gather(
            call_judge(client, deployment, interest_prompt, i_user,
                       max_tok, re_effort, sem),
            call_judge(client, deployment, topics_prompt, t_user,
                       max_tok, re_effort, sem),
        )

        rec_id = {"user_id": p.get("user_id", ""),
                  "delta_index": p.get("delta_index", -1)}

        # interest_name results: expected list of objects
        if isinstance(i_res, list):
            for entry in i_res:
                if not isinstance(entry, dict):
                    continue
                scores = entry.get("scores") or {}
                interest_raw_fh.write(json.dumps({**rec_id, **entry},
                                                ensure_ascii=False) + "\n")
                for k in ("utility", "precision", "coherence", "granularity_broad"):
                    v = scores.get(k)
                    if isinstance(v, (int, float)):
                        interest_scores[k].append(float(v))
                        if k == "utility":
                            n_interest_judged += 1
        else:
            n_judge_fail += 1

        if isinstance(t_res, list):
            for entry in t_res:
                if not isinstance(entry, dict):
                    continue
                scores = entry.get("scores") or {}
                topics_raw_fh.write(json.dumps({**rec_id, **entry},
                                               ensure_ascii=False) + "\n")
                for k in ("utility", "precision", "coherence", "granularity"):
                    v = scores.get(k)
                    if isinstance(v, (int, float)):
                        topic_scores[k].append(float(v))
                        if k == "utility":
                            n_topic_judged += 1
        else:
            n_judge_fail += 1

    try:
        await asyncio.gather(*(judge_one(p) for p in preds))
    finally:
        interest_raw_fh.close()
        topics_raw_fh.close()

    summary = {
        "model_tag": args.model_tag,
        "deployment": deployment,
        "n_records": len(preds),
        "n_records_with_pred": n_records_with_pred,
        "n_parse_fail": n_parse_fail,
        "n_judge_fail": n_judge_fail,
        "interest_name": {
            "n_interests_scored": n_interest_judged,
            "scores": {k: stats(v) for k, v in interest_scores.items()},
        },
        "topics": {
            "n_topics_scored": n_topic_judged,
            "scores": {k: stats(v) for k, v in topic_scores.items()},
        },
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info("Wrote %s, %s, %s", out_path, interest_raw_path, topics_raw_path)
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
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--teacher-mode", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
