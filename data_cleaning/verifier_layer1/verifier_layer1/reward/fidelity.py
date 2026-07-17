"""Evidence fidelity + coverage floor.

We want two things from this signal:

1. **Fidelity** — each ``evidence.action`` should be traceable back to a
   raw input signal. This prevents the model from inventing actions wholesale.
2. **Coverage floor** — fidelity alone has a hacking path: emit a single
   evidence string copied from the input → ``fidelity = 1.0``. We multiply
   fidelity by ``max(coverage, COVERAGE_FLOOR)``, where coverage is the ratio
   of emitted evidence to input signals (capped at 1.0). The floor keeps
   honest-but-low-coverage outputs from being annihilated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


# Match how the existing pipeline serializes signals so we don't fight
# semantic-equivalent differences in whitespace / case.
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").lower()).strip()


def _signal_text(signal: Any) -> str:
    """Best-effort extraction of comparable text from a heterogenous signal dict.

    Accepts:
    - ``str`` (raw record line)
    - ``dict`` with any of: ``raw_record``, ``text``, ``action``, ``query``,
      ``title``, ``url``, ``page_title``.
    Concatenates everything found, since evidence may quote any sub-field.
    """
    if isinstance(signal, str):
        return signal
    if not isinstance(signal, dict):
        return ""
    fields = (
        "raw_record", "text", "action", "query", "title", "url",
        "page_title", "search_query", "event",
    )
    parts: list[str] = []
    for k in fields:
        v = signal.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v)
    return " | ".join(parts)


@dataclass
class FidelityConfig:
    coverage_floor: float = 0.3
    # Minimum length (chars) of an action string before we score it; very
    # short actions trivially hit substring match and inflate fidelity.
    min_action_chars: int = 3


@dataclass
class FidelityResult:
    fidelity: float          # raw verbatim-hit rate over emitted actions
    coverage: float          # min(n_pred_actions, n_input_signals) / max(1, n_input_signals)
    score: float             # fidelity * max(coverage, floor) — used in reward
    n_pred_actions: int
    n_hit: int
    n_input_signals: int


def compute_fidelity(
    pred_actions: Iterable[str],
    input_signals: Iterable[Any],
    cfg: FidelityConfig | None = None,
) -> FidelityResult:
    """Compute evidence fidelity + coverage-floored score.

    Match rule: each predicted action (normalized) must appear as a
    substring of the concatenated normalized input-signal corpus. We use a
    single big haystack instead of pairwise matching to make matching robust
    to which signal field the model chose to quote.
    """
    cfg = cfg or FidelityConfig()

    actions = list(pred_actions)
    signals = list(input_signals)
    n_pred = len(actions)
    n_signals = len(signals)

    if n_pred == 0:
        return FidelityResult(
            fidelity=0.0, coverage=0.0, score=0.0,
            n_pred_actions=0, n_hit=0, n_input_signals=n_signals,
        )

    haystack = " ||| ".join(_normalize(_signal_text(s)) for s in signals)
    n_hit = 0
    for a in actions:
        a_norm = _normalize(a)
        if len(a_norm) < cfg.min_action_chars:
            continue
        if a_norm in haystack:
            n_hit += 1

    fidelity = n_hit / n_pred
    coverage = min(1.0, n_pred / max(1, n_signals))
    score = fidelity * max(coverage, cfg.coverage_floor)
    return FidelityResult(
        fidelity=fidelity,
        coverage=coverage,
        score=score,
        n_pred_actions=n_pred,
        n_hit=n_hit,
        n_input_signals=n_signals,
    )
