"""M2 interest-name judge for Layer1 Stage-4 reward (r_llm).

Calls the same Azure judge + prompt used by offline eval
(``eval_m2_layer1_delta_judge.py`` / ``layer1_delta_eval_interest_name_v2.md``)
so the RL ``r_llm`` is consistent with the M2 interest-utility/precision metric.

Per rollout we send the predicted interests (interest_name + topic names) to the
judge, which returns per-interest ``scores`` with ``utility`` (1-10) and
``precision`` (0-10). We average across the rollout's interests and hand the raw
means back to ``reward.fuse_stage4`` for normalization + fusion.

Design notes:
- Realtime call + in-memory cache keyed by the interests payload (dedupes
  identical rollouts within a run).
- On any failure (timeout, parse error, missing key) we return ``(None, None)``
  so the reward falls back to rule-only — the loop never crashes on judge issues.
- Judge disabled entirely when ``LAYER1_JUDGE_ENABLED`` != "1" (smoke mode).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("pipelinerl.domains.layer1.judge")

# ---------------------------------------------------------------------------
# Config (env-overridable; defaults match SFT eval / PassB)
# ---------------------------------------------------------------------------
JUDGE_ENDPOINT = os.environ.get(
    "LAYER1_JUDGE_ENDPOINT",
    "https://msncompanioneu2.cognitiveservices.azure.com/",
)
JUDGE_DEPLOYMENT = os.environ.get("LAYER1_JUDGE_DEPLOYMENT", "gpt-5.1")
JUDGE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
JUDGE_API_KEY_ENV = os.environ.get("LAYER1_JUDGE_API_KEY_ENV", "AZURE_OPENAI_KEY")
JUDGE_MAX_TOKENS = int(os.environ.get("LAYER1_JUDGE_MAX_TOKENS", "8192"))
JUDGE_REASONING_EFFORT = os.environ.get("LAYER1_JUDGE_REASONING_EFFORT", "") or None
JUDGE_TIMEOUT = float(os.environ.get("LAYER1_JUDGE_TIMEOUT", "120"))
JUDGE_MAX_RETRIES = int(os.environ.get("LAYER1_JUDGE_RETRIES", "2"))

_PROMPTS_DIR = Path(os.environ.get(
    "MAIPROFILE_PROMPTS_DIR",
    "/scratch/azureml/cr/j/65fcf9508e03476381b75ace1f02fb73/exe/wd/"
    "MaiProfile-main 1/MaiProfile-main/maiprofilev3dev/evaluation/prompts",
))
_INTEREST_PROMPT_PATH = _PROMPTS_DIR / "layer1_delta_eval_interest_name_v2.md"
_RECALL_JUDGE_PROMPT_PATH = _PROMPTS_DIR / "interest_recall_judge.md"

# Batch size for the per-rollout coverage judge (matches offline recall eval
# JUDGE_BATCH_SIZE=10): batches the *grounded candidate* set, not the model output.
RECALL_JUDGE_BATCH_SIZE = int(os.environ.get("LAYER1_RECALL_JUDGE_BATCH_SIZE", "10"))


def judge_enabled() -> bool:
    return os.environ.get("LAYER1_JUDGE_ENABLED", "0") == "1"


def recall_judge_enabled() -> bool:
    """Recall reward is opt-in: requires both the judge and the recall flag."""
    return judge_enabled() and os.environ.get("LAYER1_RECALL_ENABLED", "0") == "1"


# Directory of pre-generated model-INDEPENDENT grounded candidates, one
# ``<user_id>.json`` per training user (propose -> ground -> rescue cache).
RECALL_CANDIDATES_DIR = os.environ.get("LAYER1_RECALL_CANDIDATES_DIR", "")

_grounded_cache: dict[str, list[dict[str, Any]]] = {}
_grounded_lock = asyncio.Lock()


async def load_grounded(user_id: str) -> list[dict[str, Any]]:
    """Load (and memo-cache) the grounded candidate list for ``user_id``.

    Returns ``[]`` when the candidate file is missing or unreadable — the
    rollout then simply skips the recall term (graceful degradation).
    """
    if not user_id or not RECALL_CANDIDATES_DIR:
        return []
    async with _grounded_lock:
        if user_id in _grounded_cache:
            return _grounded_cache[user_id]
    path = Path(RECALL_CANDIDATES_DIR) / f"{user_id}.json"
    grounded: list[dict[str, Any]] = []
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            g = data.get("grounded") if isinstance(data, dict) else None
            if isinstance(g, list):
                grounded = [x for x in g if isinstance(x, dict)]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("load_grounded(%s) failed: %s", user_id, exc)
        grounded = []
    async with _grounded_lock:
        _grounded_cache[user_id] = grounded
    return grounded


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
_client = None
_system_prompt: str | None = None
_recall_judge_prompt: str | None = None
_cache: dict[str, tuple[float | None, float | None]] = {}
_recall_cache: dict[str, dict[str, float | int | None]] = {}
_cache_lock = asyncio.Lock()


def _get_system_prompt() -> str:
    global _system_prompt
    if _system_prompt is None:
        _system_prompt = _INTEREST_PROMPT_PATH.read_text(encoding="utf-8")
    return _system_prompt


def _get_recall_judge_prompt() -> str:
    global _recall_judge_prompt
    if _recall_judge_prompt is None:
        _recall_judge_prompt = _RECALL_JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    return _recall_judge_prompt


def _get_client():
    global _client
    if _client is None:
        from openai import AsyncAzureOpenAI
        api_key = os.environ.get(JUDGE_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"Judge enabled but env {JUDGE_API_KEY_ENV} is not set "
                f"(needed for {JUDGE_DEPLOYMENT} @ {JUDGE_ENDPOINT})."
            )
        _client = AsyncAzureOpenAI(
            azure_endpoint=JUDGE_ENDPOINT,
            api_key=api_key,
            api_version=JUDGE_API_VERSION,
            timeout=JUDGE_TIMEOUT,
            # Force HTTP/1.1: the msncompanioneu2 endpoint intermittently stalls
            # HTTP/2 streams, which hangs every call until JUDGE_TIMEOUT (~120s)
            # and then fails. HTTP/1.1 returns in a few seconds. (verified 2026-06-27)
            http_client=httpx.AsyncClient(http2=False, timeout=JUDGE_TIMEOUT),
        )
    return _client


# ---------------------------------------------------------------------------
# Prompt + parsing (mirror eval_m2_layer1_delta_judge.py)
# ---------------------------------------------------------------------------


def build_interest_user_prompt(parsed: list[dict[str, Any]]) -> str:
    """Build the judge user prompt from our normalized parsed structure.

    Parser normalizes topics to ``{"topic_name": ..., "evidence": [...]}``; the
    judge wants ``{"interest_name", "topics": [<topic name strings>]}``.
    """
    payload = []
    for it in parsed:
        if not isinstance(it, dict):
            continue
        topic_names = [
            t.get("topic_name", "")
            for t in (it.get("topics") or [])
            if isinstance(t, dict) and isinstance(t.get("topic_name"), str)
        ]
        payload.append({
            "interest_name": it.get("interest_name", ""),
            "topics": topic_names,
        })
    return json.dumps({"interests": payload}, ensure_ascii=False, indent=2)


def _safe_json_array(text: str) -> Any:
    """Extract the first JSON array from the judge response."""
    s = (text or "").strip()
    if s.startswith("```"):
        import re
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start = s.find("[")
        end = s.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _mean_scores(judge_result: Any) -> tuple[float | None, float | None]:
    """Average utility (1-10) and precision (0-10) across interests."""
    if not isinstance(judge_result, list):
        return None, None
    utils: list[float] = []
    precs: list[float] = []
    for entry in judge_result:
        if not isinstance(entry, dict):
            continue
        scores = entry.get("scores") or {}
        u = scores.get("utility")
        p = scores.get("precision")
        if isinstance(u, (int, float)):
            utils.append(float(u))
        if isinstance(p, (int, float)):
            precs.append(float(p))
    mean_u = sum(utils) / len(utils) if utils else None
    mean_p = sum(precs) / len(precs) if precs else None
    return mean_u, mean_p


def _cache_key(user_prompt: str) -> str:
    return hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()


async def judge_interests(parsed: list[dict[str, Any]] | None) -> tuple[float | None, float | None]:
    """Return ``(mean_utility, mean_precision)`` for the rollout, or ``(None, None)``.

    Never raises — judge failures degrade to rule-only reward.
    """
    if not parsed:
        return None, None
    try:
        user_prompt = build_interest_user_prompt(parsed)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("judge prompt build failed: %s", exc)
        return None, None

    key = _cache_key(user_prompt)
    async with _cache_lock:
        if key in _cache:
            return _cache[key]

    system_prompt = _get_system_prompt()
    client = _get_client()

    result: tuple[float | None, float | None] = (None, None)
    for attempt in range(JUDGE_MAX_RETRIES + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": JUDGE_DEPLOYMENT,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_completion_tokens": JUDGE_MAX_TOKENS,
            }
            if JUDGE_REASONING_EFFORT:
                kwargs["reasoning_effort"] = JUDGE_REASONING_EFFORT
            resp = await client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            result = _mean_scores(_safe_json_array(text))
            break
        except Exception as exc:
            wait = 2 ** attempt
            logger.warning("judge attempt %d failed: %s (retry in %ds)",
                           attempt + 1, exc, wait)
            if attempt < JUDGE_MAX_RETRIES:
                await asyncio.sleep(wait)

    async with _cache_lock:
        _cache[key] = result
    return result


# ---------------------------------------------------------------------------
# Recall judge (per-rollout interest coverage vs. cached grounded ground-truth)
# ---------------------------------------------------------------------------
def _model_interest_names(parsed: list[dict[str, Any]] | None) -> list[str]:
    """De-duplicated interest names from the rollout's parsed output."""
    if not parsed:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for it in parsed:
        if not isinstance(it, dict):
            continue
        name = (it.get("interest_name") or "").strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


async def _recall_judge_call(judge_input: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One coverage-judge call over a chunk of grounded proposals."""
    system_prompt = _get_recall_judge_prompt()
    client = _get_client()
    for attempt in range(JUDGE_MAX_RETRIES + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": JUDGE_DEPLOYMENT,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(
                        judge_input, ensure_ascii=False, indent=2)},
                ],
                "max_completion_tokens": JUDGE_MAX_TOKENS,
            }
            if JUDGE_REASONING_EFFORT:
                kwargs["reasoning_effort"] = JUDGE_REASONING_EFFORT
            resp = await client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            obj = _safe_json_object(text)
            if isinstance(obj, dict) and isinstance(obj.get("results"), list):
                return [r for r in obj["results"] if isinstance(r, dict)]
            return []
        except Exception as exc:
            wait = 2 ** attempt
            logger.warning("recall judge attempt %d failed: %s (retry in %ds)",
                           attempt + 1, exc, wait)
            if attempt < JUDGE_MAX_RETRIES:
                await asyncio.sleep(wait)
    return []


def _safe_json_object(text: str) -> Any:
    """Extract the first JSON object from the judge response."""
    s = (text or "").strip()
    if s.startswith("```"):
        import re
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


async def judge_recall(
    parsed: list[dict[str, Any]] | None,
    grounded: list[dict[str, Any]] | None,
) -> dict[str, float | int | None]:
    """Return per-rollout interest-recall against the cached grounded set.

    ``grounded`` is the model-INDEPENDENT ground-truth candidate list produced
    offline (propose -> ground -> rescue), each item carrying ``proposal`` and
    ``granularity_level`` ("matched" | "broad"). We ask the judge which grounded
    proposals are covered by the rollout's interests and report recall.

    Returns ``{"overall", "matched", "broad", "n_grounded", "n_covered"}``.
    ``overall`` is the value fused into the reward; ``matched``/``broad`` are
    stored for diagnostics only. Never raises — on failure returns all-None so
    the reward falls back to utility+precision only.
    """
    empty = {"overall": None, "matched": None, "broad": None,
             "n_grounded": 0, "n_covered": 0}
    if not grounded:
        return empty

    model_names = _model_interest_names(parsed)
    m_total = sum(1 for g in grounded if g.get("granularity_level") == "matched")
    b_total = sum(1 for g in grounded if g.get("granularity_level") == "broad")
    total = m_total + b_total
    if total == 0:
        return empty

    # If the rollout produced no interests, every proposal is uncovered.
    if not model_names:
        return {"overall": 0.0,
                "matched": 0.0 if m_total else None,
                "broad": 0.0 if b_total else None,
                "n_grounded": total, "n_covered": 0}

    # Cache by (sorted model names, sorted grounded proposals).
    key_src = json.dumps(
        {"m": sorted(model_names),
         "g": sorted(g.get("proposal", "") for g in grounded)},
        ensure_ascii=False,
    )
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
    async with _cache_lock:
        if key in _recall_cache:
            return _recall_cache[key]

    candidates = [{"interest_name": n, "similarity": 1.0} for n in model_names]
    prop2level = {g.get("proposal"): g.get("granularity_level") for g in grounded}

    # Optional unbiased random subsample to cap judge load on huge gold sets.
    cap = int(os.environ.get("LAYER1_RECALL_MAX_GROUNDED", "0"))
    judged = grounded
    if cap > 0 and len(grounded) > cap:
        import random as _rnd
        judged = _rnd.sample(grounded, cap)

    results: list[dict[str, Any]] = []
    try:
        for start in range(0, len(judged), RECALL_JUDGE_BATCH_SIZE):
            chunk = judged[start:start + RECALL_JUDGE_BATCH_SIZE]
            judge_input = [
                {
                    "proposal": g.get("proposal"),
                    "granularity_level": g.get("granularity_level"),
                    "candidates": candidates,
                }
                for g in chunk
            ]
            results.extend(await _recall_judge_call(judge_input))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("judge_recall failed: %s", exc)
        return empty

    if cap > 0 and len(grounded) > cap:
        # Subsampled: recall over the judged subset (level totals recomputed).
        m_total = sum(1 for g in judged if g.get("granularity_level") == "matched")
        b_total = sum(1 for g in judged if g.get("granularity_level") == "broad")
        total = m_total + b_total

    m_cov = b_cov = 0
    for r in results:
        if r.get("covered"):
            lvl = prop2level.get(r.get("proposal"))
            if lvl == "matched":
                m_cov += 1
            elif lvl == "broad":
                b_cov += 1
    covered = m_cov + b_cov
    out = {
        "overall": (covered / total) if total else None,
        "matched": (m_cov / m_total) if m_total else None,
        "broad": (b_cov / b_total) if b_total else None,
        "n_grounded": total,
        "n_covered": covered,
    }
    async with _cache_lock:
        _recall_cache[key] = out
    return out
