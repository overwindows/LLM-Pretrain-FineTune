"""Hallucination signal.

Stage 1: ``1 - fidelity`` as a cheap proxy (local, deterministic).
Stage 2 (future): replace with the distilled RM's hallucination head.

We isolate this in its own module so the Stage 1 -> Stage 2 swap is a single
import change in :mod:`compose`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HallucinationResult:
    hallu_rate: float       # in [0, 1]; higher = worse
    score: float            # anti-hallucination, in [0, 1]; higher = better


def compute_hallucination_stage1(fidelity: float) -> HallucinationResult:
    """Stage 1 proxy: anti-hallucination = ``fidelity`` itself.

    Rationale: in Stage 1 we have no semantic precision signal, so we lean
    on string-level grounding. The squared form ``(1 - hallu)^2`` from the
    original plan is dropped here — squaring sharpens the gradient but is
    redundant when the same signal is also entering ``fidelity_score``.
    """
    fidelity = max(0.0, min(1.0, fidelity))
    return HallucinationResult(
        hallu_rate=1.0 - fidelity,
        score=fidelity,
    )


def compute_hallucination_stage2(rm_hallu_prob: float) -> HallucinationResult:
    """Stage 2: use the distilled RM's hallucination-head probability.

    ``rm_hallu_prob`` is the model's estimate of ``P(precision_judge == 0)``.
    """
    p = max(0.0, min(1.0, rm_hallu_prob))
    return HallucinationResult(
        hallu_rate=p,
        score=1.0 - p,
    )
