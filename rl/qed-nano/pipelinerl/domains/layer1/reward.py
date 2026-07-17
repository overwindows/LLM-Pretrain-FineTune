"""Stage-4 fused reward for Layer1 delta RL (in-process, pure Python).

Locked formula (user sign-off Jun 25, option A):

    r = gate * ( 0.5 * r_rule + 0.5 * r_llm )

    r_rule = 0.5 * anti_collapse[2b graded]
           + 0.4 * anti_hallucination      (Stage-1 proxy = fidelity)
           + 0.1 * fidelity_score          (already coverage-floored)

    r_llm  = 0.5 * (utility   / 10)         (M2 interest judge)
           + 0.5 * (precision / 10)

* ``gate`` is the parse+schema multiplier (hard 0/1 or soft floor).
* ``anti_collapse`` is the teacher-relative graded band (2b): no penalty inside
  [teacher/2, teacher*2], smooth ramp outside, bounded by a floor.
* ``r_llm`` is supplied by the caller (the rollout fn calls the Azure judge).
  When ``r_llm`` is ``None`` (judge disabled, e.g. smoke test) the reward falls
  back to rule-only: ``r = gate * r_rule``.

This module is self-contained (only stdlib + ``.parser``) so it can run inside
the pipelinerl actor process with no extra deps.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .parser import (
    count_structure,
    flatten_evidence_actions,
    parse_completion,
)


# ===========================================================================
# Gate (parse + schema)
# ===========================================================================


@dataclass
class GateConfig:
    mode: str = "soft"  # "hard" | "soft"
    soft_value: float = 0.1


def compute_gate(parse_ok: bool, schema_ok: bool, cfg: GateConfig) -> tuple[float, bool]:
    """Return ``(gate_value, pass_hard)``."""
    pass_hard = parse_ok and schema_ok
    if pass_hard:
        return 1.0, True
    if cfg.mode == "soft":
        return max(0.0, min(1.0, cfg.soft_value)), False
    return 0.0, False


# ===========================================================================
# Fidelity + coverage floor
# ===========================================================================

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").lower()).strip()


def _signal_text(signal: Any) -> str:
    if isinstance(signal, str):
        return signal
    if not isinstance(signal, dict):
        return ""
    fields = (
        "raw_record", "text", "action", "query", "title", "url",
        "page_title", "search_query", "event", "Action", "intent",
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
    min_action_chars: int = 3


@dataclass
class FidelityResult:
    fidelity: float
    coverage: float
    score: float
    n_pred_actions: int
    n_hit: int
    n_input_signals: int


def compute_fidelity(
    pred_actions: Iterable[str],
    input_signals: Iterable[Any],
    cfg: FidelityConfig,
) -> FidelityResult:
    actions = list(pred_actions)
    signals = list(input_signals)
    n_pred = len(actions)
    n_signals = len(signals)

    if n_pred == 0:
        return FidelityResult(0.0, 0.0, 0.0, 0, 0, n_signals)

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
    return FidelityResult(fidelity, coverage, score, n_pred, n_hit, n_signals)


# ===========================================================================
# Anti-collapse 2b: teacher-relative graded band
# (vendored from $Y/MAIDistillation0623/rl/reward/graded_anti_collapse.py)
# ===========================================================================


@dataclass(frozen=True)
class GradedBandConfig:
    tol_under: float = 2.0      # too short allowed down to teacher/2 for free
    tol_over: float = 2.0       # too long allowed up to teacher*2 for free
    scale_under: float = 1.4
    scale_over: float = 1.0
    floor: float = 0.4


def graded_score(pred: float, teacher: float, cfg: GradedBandConfig) -> float:
    """Graded score in [floor, 1.0] for ``pred`` vs a teacher reference value."""
    if teacher is None or teacher <= 0:
        return 1.0
    p = max(float(pred), 0.0)
    if p <= 0:
        return cfg.floor
    r = p / float(teacher)
    log_r = math.log(r)
    if log_r >= 0:  # over
        excess = log_r - math.log(cfg.tol_over)
        if excess <= 0:
            return 1.0
        return max(cfg.floor, 1.0 - excess / cfg.scale_over)
    else:           # under
        excess = (-log_r) - math.log(cfg.tol_under)
        if excess <= 0:
            return 1.0
        return max(cfg.floor, 1.0 - excess / cfg.scale_under)


@dataclass
class GradedAntiCollapseConfig:
    count: GradedBandConfig = field(default_factory=lambda: GradedBandConfig(
        tol_under=2.0, tol_over=2.0, scale_under=1.4, scale_over=1.2, floor=0.4))
    length: GradedBandConfig = field(default_factory=lambda: GradedBandConfig(
        tol_under=2.0, tol_over=2.0, scale_under=1.4, scale_over=1.0, floor=0.4))
    combine: str = "mean"  # "mean" or "min"


def graded_anti_collapse(
    pred_count: float, teacher_count: float,
    pred_length: float, teacher_length: float,
    cfg: GradedAntiCollapseConfig,
) -> dict:
    s_count = graded_score(pred_count, teacher_count, cfg.count)
    s_length = graded_score(pred_length, teacher_length, cfg.length)
    score = min(s_count, s_length) if cfg.combine == "min" else 0.5 * (s_count + s_length)
    return {"anti_collapse": score, "count_score": s_count, "length_score": s_length}


# ===========================================================================
# Stage-4 reward config + compose
# ===========================================================================


@dataclass
class RuleWeights:
    anti_collapse: float = 0.5
    anti_hallucination: float = 0.4
    fidelity: float = 0.1


@dataclass
class LLMWeights:
    utility: float = 0.5
    precision: float = 0.5
    recall: float = 0.0


@dataclass
class Stage4Config:
    stage: int = 4
    # Outer fuse: r = gate * (w_rule * r_rule + w_llm * r_llm)
    w_rule: float = 0.5
    w_llm: float = 0.5
    rule_weights: RuleWeights = field(default_factory=RuleWeights)
    llm_weights: LLMWeights = field(default_factory=LLMWeights)
    gate: GateConfig = field(default_factory=GateConfig)
    fidelity: FidelityConfig = field(default_factory=FidelityConfig)
    anti_collapse: GradedAntiCollapseConfig = field(default_factory=GradedAntiCollapseConfig)
    # judge score scales (utility 1-10, precision 0-10; recall already 0-1)
    utility_scale: float = 10.0
    precision_scale: float = 10.0


@dataclass
class Stage4Output:
    reward: float
    success: bool          # pass_hard gate AND reward above success threshold
    pass_hard: bool
    r_rule: float
    r_llm: float | None
    components: dict[str, Any]
    metadata: dict[str, Any]
    # Inputs the caller needs to build the judge prompt:
    parsed: list[dict] | None


def compute_rule_reward(
    completion: str,
    input_signals: list[Any],
    teacher_count: float | None,
    teacher_length: float | None,
    cfg: Stage4Config,
) -> Stage4Output:
    """Compute gate + r_rule (no judge). ``r_llm`` is filled in later by the fuse."""
    parse_res = parse_completion(completion)
    gate_value, pass_hard = compute_gate(parse_res.parse_ok, parse_res.schema_ok, cfg.gate)

    structure_counts = (
        count_structure(parse_res.parsed) if parse_res.parsed else
        {"n_interests": 0, "n_topics": 0, "n_evidence": 0}
    )
    completion_chars = len((parse_res.raw_post_think or "").strip())

    components: dict[str, Any] = {
        "parse_ok": parse_res.parse_ok,
        "schema_ok": parse_res.schema_ok,
        "gate": gate_value,
    }
    metadata: dict[str, Any] = {
        **structure_counts,
        "total_chars": completion_chars,
        "had_think_block": parse_res.had_think_block,
        "teacher_count": teacher_count,
        "teacher_length": teacher_length,
    }

    if gate_value == 0.0:
        components.update({
            "anti_collapse": 0.0, "anti_hallucination": 0.0,
            "fidelity": 0.0, "coverage": 0.0, "fidelity_score": 0.0, "r_rule": 0.0,
        })
        return Stage4Output(
            reward=0.0, success=False, pass_hard=False, r_rule=0.0, r_llm=None,
            components=components, metadata=metadata, parsed=parse_res.parsed,
        )

    parsed = parse_res.parsed or []
    actions = flatten_evidence_actions(parsed)
    fid = compute_fidelity(actions, input_signals, cfg.fidelity)
    anti_hallucination = max(0.0, min(1.0, fid.fidelity))  # Stage-1 proxy
    ac = graded_anti_collapse(
        pred_count=structure_counts["n_interests"],
        teacher_count=teacher_count if teacher_count else structure_counts["n_interests"],
        pred_length=completion_chars,
        teacher_length=teacher_length if teacher_length else completion_chars,
        cfg=cfg.anti_collapse,
    )

    rw = cfg.rule_weights
    r_rule = (
        rw.anti_collapse * ac["anti_collapse"]
        + rw.anti_hallucination * anti_hallucination
        + rw.fidelity * fid.score
    )

    components.update({
        "anti_collapse": ac["anti_collapse"],
        "count_score": ac["count_score"],
        "length_score": ac["length_score"],
        "anti_hallucination": anti_hallucination,
        "fidelity": fid.fidelity,
        "coverage": fid.coverage,
        "fidelity_score": fid.score,
        "r_rule": r_rule,
    })
    metadata.update({
        "n_pred_actions": fid.n_pred_actions,
        "n_evidence_hit": fid.n_hit,
        "n_input_signals": fid.n_input_signals,
    })
    # reward filled by fuse_stage4; provisional rule-only reward here.
    return Stage4Output(
        reward=gate_value * r_rule, success=pass_hard, pass_hard=pass_hard,
        r_rule=r_rule, r_llm=None, components=components, metadata=metadata,
        parsed=parse_res.parsed,
    )


def fuse_stage4(
    out: Stage4Output,
    gate_value: float,
    utility: float | None,
    precision: float | None,
    cfg: Stage4Config,
    success_threshold: float = 0.5,
    recall: float | None = None,
) -> Stage4Output:
    """Fuse r_rule with judge r_llm. ``utility``/``precision`` are raw judge
    scores (utility 1-10, precision 0-10); ``recall`` is the per-rollout interest
    recall (already 0-1) or ``None`` when the recall reward is disabled/failed.

    ``r_llm`` is a weight-normalized blend of the available judge components:
    ``r_llm = (w_u*u + w_p*p [+ w_recall*recall]) / sum(active weights)``.
    Recall is only blended when both its weight is > 0 and a value is present, so
    a missing recall degrades gracefully to utility+precision without rescaling
    the reward. If both utility and precision are ``None`` the reward stays
    rule-only.
    """
    if utility is None and precision is None:
        r = gate_value * out.r_rule
        out.reward = r
        out.success = out.pass_hard and (r >= success_threshold)
        out.components["r_llm"] = None
        return out

    u = 0.0 if utility is None else max(0.0, min(1.0, utility / cfg.utility_scale))
    p = 0.0 if precision is None else max(0.0, min(1.0, precision / cfg.precision_scale))

    w_u = cfg.llm_weights.utility
    w_p = cfg.llm_weights.precision
    w_rec = getattr(cfg.llm_weights, "recall", 0.0)
    weighted = [(w_u, u), (w_p, p)]
    rec_norm: float | None = None
    if w_rec > 0.0 and recall is not None:
        rec_norm = max(0.0, min(1.0, recall))
        weighted.append((w_rec, rec_norm))
    denom = sum(w for w, _ in weighted)
    r_llm = (sum(w * v for w, v in weighted) / denom) if denom > 0 else 0.0

    fused = cfg.w_rule * out.r_rule + cfg.w_llm * r_llm
    r = gate_value * fused

    out.r_llm = r_llm
    out.reward = r
    out.success = out.pass_hard and (r >= success_threshold)
    out.components.update({
        "r_llm": r_llm,
        "judge_utility": utility,
        "judge_precision": precision,
        "judge_utility_norm": u,
        "judge_precision_norm": p,
        "judge_recall": recall,
        "judge_recall_norm": rec_norm,
        "fused_inner": fused,
    })
    return out


# ---------------------------------------------------------------------------
# Config builder (Hydra DictConfig / plain dict -> Stage4Config)
# ---------------------------------------------------------------------------


def build_config(d: dict[str, Any] | None) -> Stage4Config:
    d = dict(d or {})

    def _band(raw: dict | None, default: GradedBandConfig) -> GradedBandConfig:
        raw = dict(raw or {})
        return GradedBandConfig(
            tol_under=float(raw.get("tol_under", default.tol_under)),
            tol_over=float(raw.get("tol_over", default.tol_over)),
            scale_under=float(raw.get("scale_under", default.scale_under)),
            scale_over=float(raw.get("scale_over", default.scale_over)),
            floor=float(raw.get("floor", default.floor)),
        )

    ac_raw = dict(d.get("anti_collapse") or {})
    ac_default = GradedAntiCollapseConfig()
    ac = GradedAntiCollapseConfig(
        count=_band(ac_raw.get("count"), ac_default.count),
        length=_band(ac_raw.get("length"), ac_default.length),
        combine=str(ac_raw.get("combine", "mean")),
    )

    return Stage4Config(
        stage=int(d.get("stage", 4)),
        w_rule=float(d.get("w_rule", 0.5)),
        w_llm=float(d.get("w_llm", 0.5)),
        rule_weights=RuleWeights(**(d.get("rule_weights") or {})),
        llm_weights=LLMWeights(**(d.get("llm_weights") or {})),
        gate=GateConfig(**(d.get("gate") or {})),
        fidelity=FidelityConfig(**(d.get("fidelity") or {})),
        anti_collapse=ac,
        utility_scale=float(d.get("utility_scale", 10.0)),
        precision_scale=float(d.get("precision_scale", 10.0)),
    )
