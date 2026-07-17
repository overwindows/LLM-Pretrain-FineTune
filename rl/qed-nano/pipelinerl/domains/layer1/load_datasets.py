"""Dataset loader for Layer1 delta Stage-4 RL.

Reads local JSONL files in the SFT chat format::

    {"messages": [ {role:system}, {role:user}, {role:assistant} ],
     "metadata": {user_id, delta_index, ...}}

and yields pipelinerl ``problem`` dicts. We:
- keep ``system`` + ``user`` as the actor prompt (``messages_prompt``);
- extract ``input_signals`` from the user message ("Today's Denoised Signals");
- derive the teacher reference from the ``assistant`` message
  (``teacher_count`` = #interests, ``teacher_length`` = post-think char length)
  for the 2b graded anti-collapse band.

Config schema (``train_dataset_names`` / ``test_dataset_names``)::

    - path: /abs/path/to/train.jsonl
      name: layer1_delta_v2     # optional dataset tag
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from omegaconf import DictConfig, ListConfig

from .parser import count_structure, parse_completion, strip_think

logger = logging.getLogger("pipelinerl.domains.layer1.load_datasets")

_SIGNALS_RE = re.compile(
    r"Denoised Signals[^\[]*(\[.*\])", re.DOTALL
)


def _extract_input_signals(user_content: str) -> list[Any]:
    """Pull the JSON signal array out of the user message; tolerant of failure."""
    if not user_content:
        return []
    m = _SIGNALS_RE.search(user_content)
    if not m:
        return []
    blob = m.group(1)
    # The array may be followed by trailing prose; trim to balanced brackets.
    depth = 0
    end = -1
    for i, ch in enumerate(blob):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end != -1:
        blob = blob[: end + 1]
    try:
        arr = json.loads(blob)
        return arr if isinstance(arr, list) else []
    except json.JSONDecodeError:
        return []


def _teacher_reference(assistant_content: str) -> tuple[int | None, int | None]:
    """Return ``(teacher_count, teacher_length)`` from the gpt-4o assistant ref."""
    if not assistant_content:
        return None, None
    post_think, _ = strip_think(assistant_content)
    teacher_length = len(post_think.strip()) or None
    res = parse_completion(assistant_content)
    if res.parsed:
        teacher_count = count_structure(res.parsed)["n_interests"]
    else:
        teacher_count = None
    return teacher_count, teacher_length


def _iter_records(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _to_problem(rec: dict[str, Any], dataset_name: str) -> dict[str, Any] | None:
    messages = rec.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return None
    sys_msg = next((m for m in messages if m.get("role") == "system"), None)
    user_msg = next((m for m in messages if m.get("role") == "user"), None)
    asst_msg = next((m for m in messages if m.get("role") == "assistant"), None)
    if user_msg is None:
        return None

    prompt_messages = []
    if sys_msg is not None:
        prompt_messages.append({"role": "system", "content": sys_msg.get("content", "")})
    prompt_messages.append({"role": "user", "content": user_msg.get("content", "")})

    input_signals = _extract_input_signals(user_msg.get("content", ""))
    teacher_count, teacher_length = _teacher_reference(
        asst_msg.get("content", "") if asst_msg else ""
    )

    meta = rec.get("metadata") or {}
    rid = f"{meta.get('user_id', '')}:{meta.get('delta_index', -1)}"

    return {
        "dataset": dataset_name,
        "id": rid,
        "user_id": meta.get("user_id", ""),
        "messages_prompt": prompt_messages,
        "input_signals": input_signals,
        "teacher_count": teacher_count,
        "teacher_length": teacher_length,
    }


def load_datasets(dataset_names, seed: int = 42, **kwargs) -> list[dict[str, Any]]:
    """Load local JSONL datasets into a list of problem dicts."""
    if isinstance(dataset_names, (DictConfig, ListConfig)):
        specs = list(dataset_names)
    else:
        specs = list(dataset_names or [])

    problems: list[dict[str, Any]] = []
    for spec in specs:
        if isinstance(spec, str):
            path, name = spec, "layer1_delta"
        else:
            path = spec.get("path")
            name = spec.get("name", "layer1_delta")
        if not path:
            logger.warning("dataset spec missing 'path': %s", spec)
            continue
        n_before = len(problems)
        for rec in _iter_records(path):
            prob = _to_problem(rec, name)
            if prob is not None:
                problems.append(prob)
        logger.info("Loaded %d problems from %s", len(problems) - n_before, path)

    logger.info("Total layer1 problems: %d", len(problems))
    return problems
