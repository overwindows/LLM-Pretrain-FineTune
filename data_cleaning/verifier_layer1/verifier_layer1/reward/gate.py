"""Parse + schema gate.

Stage 1 supports both **hard** (0/1) and **soft** (low_value / 1.0) modes.
Default at training start is *soft* with ``soft_value=0.1`` to keep gradient
flowing while the model is still learning to emit valid JSON. Once
``schema_rate`` stabilizes above ~0.95, switch to *hard* via config.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..parser import ParseResult


@dataclass
class GateConfig:
    mode: str = "soft"  # "hard" | "soft"
    soft_value: float = 0.1


@dataclass
class GateResult:
    value: float            # in [0, 1]; the multiplier applied to reward
    pass_hard: bool         # parse_ok AND schema_ok
    parse_ok: bool
    schema_ok: bool


def compute_gate(parse_res: ParseResult, cfg: GateConfig | None = None) -> GateResult:
    """Compose the parse/schema gate from a :class:`ParseResult`.

    Hard mode: ``value = 1.0`` iff parse_ok AND schema_ok, else 0.
    Soft mode: ``value = cfg.soft_value`` for failures (so the sample still
    produces a non-zero gradient via the rest of the reward).
    """
    cfg = cfg or GateConfig()
    pass_hard = parse_res.parse_ok and parse_res.schema_ok
    if pass_hard:
        value = 1.0
    elif cfg.mode == "soft":
        value = max(0.0, min(1.0, cfg.soft_value))
    else:
        value = 0.0
    return GateResult(
        value=value,
        pass_hard=pass_hard,
        parse_ok=parse_res.parse_ok,
        schema_ok=parse_res.schema_ok,
    )
