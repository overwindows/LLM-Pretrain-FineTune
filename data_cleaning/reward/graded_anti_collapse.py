"""Graded, teacher-relative rule reward for Layer1 anti-collapse.

Motivation (per design discussion): the teacher's output size is only a *reference*.
We should NOT hard-zero a rollout the moment it steps outside a fixed band. Instead:

  * keep a generous **dead-band** around the teacher value where reward == 1.0
    (small differences are free — "只有特别不一样才罚");
  * outside the dead-band, apply a **graded** penalty that grows smoothly with how
    far off we are ("超出越多 / 越少,罚得越多");
  * **bound** the penalty with a floor so reward never collapses to 0 just for
    length ("但也不用罚过多");
  * allow **asymmetry** (over-long vs too-short can be penalised differently).

This is the teacher-relative replacement for the fixed-band `_band_gate` in
`verifier_layer1/reward/count_length.py` (the Stage-1 scaffold uses absolute
bands count=(2,9), length=(200,2000) with a hard 0/1 gate; this module makes the
band per-record from the teacher and replaces the hard gate with a soft ramp).

Shape (one metric, e.g. interest-count or output-length):

    r = pred / teacher                      (ratio vs teacher reference)
    d = |ln r|                              (symmetric log distance)
    if d <= ln(tol):           score = 1.0                  # dead-band
    else:                      score = max(floor,
                                           1 - (d - ln(tol)) / scale)

`tol` is the multiplicative half-width of the dead-band (tol=2.0 → no penalty
for anything in [0.5x, 2x] of the teacher). `scale` controls how fast the penalty
ramps (in log units). `floor` bounds the worst case (e.g. 0.4 → at most -0.6).
Using the log-ratio makes "2x too long" and "2x too short" symmetric by default,
and the under/over parameters let you break that symmetry on purpose.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GradedBandConfig:
    """Graded, teacher-relative penalty for one scalar metric.

    tol_*   : multiplicative dead-band half-width (>=1.0). Inside [t/tol, t*tol] => 1.0.
    scale_* : log-units over which the penalty ramps from 1.0 down toward the floor.
    floor   : minimum score (penalty is bounded; never below this just for size).
    Separate under/over so too-short and too-long can be weighted differently.
    """
    tol_under: float = 2.0      # too short allowed down to teacher/2 for free
    tol_over: float = 2.0       # too long allowed up to teacher*2 for free
    scale_under: float = 1.4    # ramp width below (ln units; ~ a further 4x to hit floor)
    scale_over: float = 1.0     # ramp width above (penalise over-long a bit faster)
    floor: float = 0.4          # bounded penalty: score never drops below this

    def __post_init__(self) -> None:
        assert self.tol_under >= 1.0 and self.tol_over >= 1.0
        assert self.scale_under > 0 and self.scale_over > 0
        assert 0.0 <= self.floor <= 1.0


def graded_score(pred: float, teacher: float, cfg: GradedBandConfig) -> float:
    """Graded score in [floor, 1.0] for pred vs a teacher reference value."""
    # Degenerate teacher reference: fall back to a permissive constant.
    if teacher is None or teacher <= 0:
        return 1.0
    p = max(float(pred), 0.0)
    if p <= 0:
        # produced nothing while teacher wanted something -> worst (but bounded)
        return cfg.floor
    r = p / float(teacher)
    log_r = math.log(r)
    if log_r >= 0:  # pred >= teacher  -> over
        excess = log_r - math.log(cfg.tol_over)
        if excess <= 0:
            return 1.0
        return max(cfg.floor, 1.0 - excess / cfg.scale_over)
    else:           # pred < teacher  -> under
        excess = (-log_r) - math.log(cfg.tol_under)
        if excess <= 0:
            return 1.0
        return max(cfg.floor, 1.0 - excess / cfg.scale_under)


@dataclass(frozen=True)
class GradedAntiCollapseConfig:
    count: GradedBandConfig = GradedBandConfig(
        tol_under=2.0, tol_over=2.0, scale_under=1.4, scale_over=1.2, floor=0.4)
    length: GradedBandConfig = GradedBandConfig(
        tol_under=2.0, tol_over=2.0, scale_under=1.4, scale_over=1.0, floor=0.4)
    combine: str = "mean"  # "mean" or "min" over the two metrics


def graded_anti_collapse(
    pred_count: float, teacher_count: float,
    pred_length: float, teacher_length: float,
    cfg: GradedAntiCollapseConfig | None = None,
) -> dict:
    """Drop-in graded anti-collapse term.

    Replaces the fixed-band product gate: both count and length are scored
    relative to the teacher, then combined. Returns the components for logging.
    """
    cfg = cfg or GradedAntiCollapseConfig()
    s_count = graded_score(pred_count, teacher_count, cfg.count)
    s_length = graded_score(pred_length, teacher_length, cfg.length)
    if cfg.combine == "min":
        score = min(s_count, s_length)
    else:
        score = 0.5 * (s_count + s_length)
    return {
        "anti_collapse": score,
        "count_score": s_count,
        "length_score": s_length,
        "pred_count": pred_count, "teacher_count": teacher_count,
        "pred_length": pred_length, "teacher_length": teacher_length,
    }


if __name__ == "__main__":
    # quick sanity sweep: teacher wants 5 interests, ~1000 chars
    cfg = GradedAntiCollapseConfig()
    for pc in (0, 1, 2, 3, 5, 8, 10, 20):
        out = graded_anti_collapse(pc, 5, 1000, 1000, cfg)
        print(f"count={pc:>3}  count_score={out['count_score']:.3f}  "
              f"anti_collapse={out['anti_collapse']:.3f}")
    print("---")
    for pl in (100, 250, 500, 1000, 2000, 4000, 8000):
        out = graded_anti_collapse(5, 5, pl, 1000, cfg)
        print(f"len={pl:>5}  length_score={out['length_score']:.3f}  "
              f"anti_collapse={out['anti_collapse']:.3f}")
