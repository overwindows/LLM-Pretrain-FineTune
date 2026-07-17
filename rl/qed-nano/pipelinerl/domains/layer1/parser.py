"""Parser for Layer1 delta model output.

Vendored verbatim from ``rl_layer1/verifier_layer1/parser.py`` so the layer1
domain is self-contained inside the pipelinerl package. Pure stdlib.

Handles:
1. Stripping the ``<think>...</think>`` block from Qwen3-Thinking outputs.
2. Stripping common markdown fences (```json ... ```).
3. ``json.loads`` with mild fallback for truncated outputs.
4. Validating the Layer1 schema:
       [{"interest_name": str, "topics": [{"topic_name": str,
                                            "evidence": [{"action": str, ...}, ...]}]}]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Tolerant aliases the model occasionally emits.
_TOPIC_NAME_ALIASES = ("topic_name", "topic", "name")
_EVIDENCE_KEY_ALIASES = ("evidence", "evidences", "actions")
_ACTION_KEY_ALIASES = ("action", "raw_record", "text")


@dataclass
class ParseResult:
    parsed: list[dict[str, Any]] | None = None
    parse_ok: bool = False
    schema_ok: bool = False
    had_think_block: bool = False
    errors: list[str] = field(default_factory=list)
    raw_post_think: str = ""


def strip_think(text: str) -> tuple[str, bool]:
    """Strip a leading ``<think>...</think>`` block.

    Returns ``(post_think_text, had_block)``.
    """
    if not text:
        return "", False
    close_idx = text.rfind(THINK_CLOSE)
    if close_idx == -1:
        return text, False
    return text[close_idx + len(THINK_CLOSE):].lstrip(), True


# Chat/special tokens some models leave trailing in the raw output
# (e.g. ``<|im_end|>``, ``<|endoftext|>``). These can sit after the closing
# code fence and break ``endswith("```")`` based fence stripping.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]*\|>")


def _strip_markdown_fence(text: str) -> str:
    s = text.strip()
    # Drop special/chat tokens anywhere so they cannot trail the JSON payload.
    s = _SPECIAL_TOKEN_RE.sub("", s).strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        # Cut at the closing fence wherever it is; ignore any trailing prose.
        fence_end = s.rfind("```")
        if fence_end != -1:
            s = s[:fence_end]
        s = s.strip()
    return s


def _try_json_load(text: str) -> tuple[Any, str | None]:
    """Try ``json.loads`` with small repair attempts for truncated outputs."""
    payload = _strip_markdown_fence(text)
    if not payload:
        return None, "empty payload after fence strip"
    last_err: str | None = None
    for suffix in ("", "}", "]", "]}", "}]", "]}]"):
        try:
            return json.loads(payload + suffix), None
        except json.JSONDecodeError as exc:
            last_err = f"json.loads failed (suffix={suffix!r}): {exc.msg} @ line {exc.lineno} col {exc.colno}"
    return None, last_err


def _normalize_interest(raw: Any, idx: int) -> tuple[dict | None, list[str]]:
    """Normalize a single interest dict; return ``(normalized, errors)``."""
    errs: list[str] = []
    if not isinstance(raw, dict):
        return None, [f"interest[{idx}]: not a dict (got {type(raw).__name__})"]

    name = raw.get("interest_name")
    if not isinstance(name, str) or not name.strip():
        errs.append(f"interest[{idx}]: missing/empty 'interest_name'")
        name = ""

    topics_raw = raw.get("topics")
    if not isinstance(topics_raw, list):
        errs.append(f"interest[{idx}]: 'topics' missing or not a list")
        topics_raw = []

    topics_norm: list[dict] = []
    for t_idx, t in enumerate(topics_raw):
        if not isinstance(t, dict):
            errs.append(f"interest[{idx}].topics[{t_idx}]: not a dict")
            continue
        t_name = None
        for k in _TOPIC_NAME_ALIASES:
            v = t.get(k)
            if isinstance(v, str) and v.strip():
                t_name = v
                break
        if t_name is None:
            errs.append(f"interest[{idx}].topics[{t_idx}]: missing topic name")
            continue

        ev_raw = None
        for k in _EVIDENCE_KEY_ALIASES:
            if k in t:
                ev_raw = t[k]
                break
        if not isinstance(ev_raw, list):
            errs.append(f"interest[{idx}].topics[{t_idx}]: 'evidence' missing or not a list")
            ev_raw = []

        ev_norm: list[dict] = []
        for e_idx, e in enumerate(ev_raw):
            if isinstance(e, str):
                ev_norm.append({"action": e})
                continue
            if not isinstance(e, dict):
                errs.append(f"interest[{idx}].topics[{t_idx}].evidence[{e_idx}]: bad type {type(e).__name__}")
                continue
            action = None
            for k in _ACTION_KEY_ALIASES:
                v = e.get(k)
                if isinstance(v, str) and v.strip():
                    action = v
                    break
            if action is None:
                errs.append(f"interest[{idx}].topics[{t_idx}].evidence[{e_idx}]: missing 'action'")
                continue
            ev_norm.append({**e, "action": action})

        topics_norm.append({"topic_name": t_name, "evidence": ev_norm})

    if not name and not topics_norm:
        return None, errs

    return {"interest_name": name, "topics": topics_norm}, errs


def parse_completion(completion: str) -> ParseResult:
    """Parse a raw completion string into a normalized Layer1 structure."""
    result = ParseResult()
    post_think, had_block = strip_think(completion or "")
    result.had_think_block = had_block
    result.raw_post_think = post_think

    parsed, err = _try_json_load(post_think)
    if err is not None:
        result.errors.append(err)
        return result
    result.parse_ok = True

    if isinstance(parsed, dict) and isinstance(parsed.get("interests"), list):
        parsed = parsed["interests"]

    if not isinstance(parsed, list):
        result.errors.append(f"top-level is not a list (got {type(parsed).__name__})")
        return result

    interests_norm: list[dict] = []
    for i, item in enumerate(parsed):
        norm, errs = _normalize_interest(item, i)
        result.errors.extend(errs)
        if norm is not None:
            interests_norm.append(norm)

    if not interests_norm:
        result.errors.append("no valid interests after schema normalization")
        return result

    result.parsed = interests_norm
    result.schema_ok = True
    return result


# ---------------------------------------------------------------------------
# Convenience aggregators used by downstream reward components.
# ---------------------------------------------------------------------------

def count_structure(parsed: list[dict]) -> dict[str, int]:
    """Return basic counts: ``n_interests``, ``n_topics``, ``n_evidence``."""
    n_interests = len(parsed)
    n_topics = 0
    n_evidence = 0
    for interest in parsed:
        topics = interest.get("topics", [])
        n_topics += len(topics)
        for t in topics:
            n_evidence += len(t.get("evidence", []))
    return {
        "n_interests": n_interests,
        "n_topics": n_topics,
        "n_evidence": n_evidence,
    }


def flatten_evidence_actions(parsed: list[dict]) -> list[str]:
    """Return all evidence ``action`` strings, preserving duplicates."""
    out: list[str] = []
    for interest in parsed:
        for topic in interest.get("topics", []):
            for e in topic.get("evidence", []):
                action = e.get("action") if isinstance(e, dict) else None
                if isinstance(action, str) and action.strip():
                    out.append(action)
    return out
