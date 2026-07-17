"""Reward orchestrator for Layer1 delta RL.

Stage 1 composition (this file):

    r = gate * (
          w_anti_collapse      * anti_collapse
        + w_anti_hallucination * anti_hallucination
        + w_fidelity           * fidelity_score    # already coverage-floored
        )

``gate`` is the parse+schema multiplier (hard 0/1 or soft floor while still
warming up). The inner term is a *weighted sum*, not a product — this keeps
the gradient dense in Stage 1 instead of collapsing whenever any single
sub-signal hits 0. Weights default to 0.5 / 0.4 / 0.1 (see RL plan v3).

Stage 2 (future) adds ``recall_score`` and swaps the proxy
``anti_hallucination = fidelity`` for the distilled RM's hallucination head.
That logic is intentionally NOT wired here yet — we want Stage 1 to be a
small, debuggable surface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ..parser import (
    ParseResult,
    count_structure,
    flatten_evidence_actions,
    parse_completion,
)
from .count_length import (
    AntiCollapseResult,
    CountLengthConfig,
    compute_anti_collapse,
)
from .fidelity import FidelityConfig, FidelityResult, compute_fidelity
from .gate import GateConfig, GateResult, compute_gate
from .hallucination import HallucinationResult, compute_hallucination_stage1


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RewardWeights:
    anti_collapse: float = 0.5
    anti_hallucination: float = 0.4
    fidelity: float = 0.1

    def normalized(self) -> "RewardWeights":
        s = self.anti_collapse + self.anti_hallucination + self.fidelity
        if s <= 0:
            raise ValueError("RewardWeights sum must be > 0")
        return RewardWeights(
            anti_collapse=self.anti_collapse / s,
            anti_hallucination=self.anti_hallucination / s,
            fidelity=self.fidelity / s,
        )


@dataclass
class RewardConfig:
    stage: int = 1
    weights: RewardWeights = field(default_factory=RewardWeights)
    gate: GateConfig = field(default_factory=GateConfig)
    count_length: CountLengthConfig = field(default_factory=CountLengthConfig)
    fidelity: FidelityConfig = field(default_factory=FidelityConfig)

    def __post_init__(self) -> None:
        if self.stage != 1:
            # Stage 2/3 should swap in a different compose function; fail
            # loudly rather than silently using Stage 1 logic.
            raise NotImplementedError(
                f"compose.RewardConfig only supports stage=1 (got stage={self.stage}); "
                "Stage 2+ live in a separate compose path that is not implemented yet."
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class RewardOutput:
    reward: float
    components: dict[str, Any]
    metadata: dict[str, Any]


def compute_reward(
    completion: str,
    input_signals: list[Any],
    cfg: RewardConfig,
) -> RewardOutput:
    """Compute Stage 1 reward for a single rollout.

    Parameters
    ----------
    completion:
        Raw model output (may include ``<think>`` block).
    input_signals:
        List of raw signal dicts / strings for this record. Used by
        :func:`fidelity.compute_fidelity` to score evidence grounding.
    cfg:
        :class:`RewardConfig`. ``stage`` must be 1.
    """
    weights = cfg.weights.normalized()
    parse_res = parse_completion(completion)
    gate_res = compute_gate(parse_res, cfg.gate)

    components: dict[str, Any] = {
        "parse_ok": parse_res.parse_ok,
        "schema_ok": parse_res.schema_ok,
        "gate": gate_res.value,
    }
    structure_counts = (
        count_structure(parse_res.parsed) if parse_res.parsed else
        {"n_interests": 0, "n_topics": 0, "n_evidence": 0}
    )
    completion_chars = len((parse_res.raw_post_think or "").strip())

    metadata: dict[str, Any] = {
        **structure_counts,
        "total_chars": completion_chars,
        "had_think_block": parse_res.had_think_block,
        "errors": list(parse_res.errors),
    }

    # If gate fully fails (hard mode, parse/schema bad), short-circuit so
    # we don't run downstream signals on empty `parsed`.
    if gate_res.value == 0.0:
        components.update({
            "anti_collapse": 0.0,
            "anti_hallucination": 0.0,
            "fidelity": 0.0,
            "coverage": 0.0,
            "fidelity_score": 0.0,
            "weighted_sum": 0.0,
        })
        return RewardOutput(reward=0.0, components=components, metadata=metadata)

    # ---- downstream signals (work on best-effort parsed or empty) ----
    parsed = parse_res.parsed or []
    actions = flatten_evidence_actions(parsed)

    fid_res = compute_fidelity(actions, input_signals, cfg.fidelity)
    hallu_res = compute_hallucination_stage1(fid_res.fidelity)
    ac_res = compute_anti_collapse(
        n_interests=structure_counts["n_interests"],
        completion_chars=completion_chars,
        cfg=cfg.count_length,
    )

    weighted_sum = (
        weights.anti_collapse * ac_res.score
        + weights.anti_hallucination * hallu_res.score
        + weights.fidelity * fid_res.score
    )
    reward = gate_res.value * weighted_sum

    components.update({
        "anti_collapse": ac_res.score,
        "count_gate": ac_res.count_gate,
        "length_gate": ac_res.length_gate,
        "anti_hallucination": hallu_res.score,
        "hallu_rate": hallu_res.hallu_rate,
        "fidelity": fid_res.fidelity,
        "coverage": fid_res.coverage,
        "fidelity_score": fid_res.score,
        "weighted_sum": weighted_sum,
    })
    metadata.update({
        "n_pred_actions": fid_res.n_pred_actions,
        "n_evidence_hit": fid_res.n_hit,
        "n_input_signals": fid_res.n_input_signals,
    })
    return RewardOutput(reward=reward, components=components, metadata=metadata)


# ---------------------------------------------------------------------------
# Config helpers (consumed by server.py for env / hydra → dataclass mapping)
# ---------------------------------------------------------------------------


def build_config(d: dict[str, Any] | None) -> RewardConfig:
    """Build a :class:`RewardConfig` from a plain dict (e.g., Hydra/JSON).

    Unknown keys are ignored (forward-compatible with Stage 2 keys).
    """
    d = dict(d or {})
    weights = RewardWeights(**(d.get("weights") or {}))
    gate = GateConfig(**(d.get("gate") or {}))
    cl_raw = dict(d.get("count_length") or {})
    if "count_band" in cl_raw:
        cl_raw["count_band"] = tuple(cl_raw["count_band"])
    if "length_band" in cl_raw:
        cl_raw["length_band"] = tuple(cl_raw["length_band"])
    cl = CountLengthConfig(**cl_raw)
    fid = FidelityConfig(**(d.get("fidelity") or {}))
    return RewardConfig(
        stage=int(d.get("stage", 1)),
        weights=weights,
        gate=gate,
        count_length=cl,
        fidelity=fid,
    )


def reward_output_to_dict(out: RewardOutput) -> dict[str, Any]:
    return {
        "reward": float(out.reward),
        "components": out.components,
        "metadata": out.metadata,
    }


def reward_config_to_dict(cfg: RewardConfig) -> dict[str, Any]:
    return {
        "stage": cfg.stage,
        "weights": asdict(cfg.weights),
        "gate": asdict(cfg.gate),
        "count_length": asdict(cfg.count_length),
        "fidelity": asdict(cfg.fidelity),
    }
