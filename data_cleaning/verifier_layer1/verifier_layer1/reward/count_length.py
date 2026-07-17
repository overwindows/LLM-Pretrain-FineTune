"""Anti-collapse: count band + length band.

Both bands are fixed (per training-set distribution, NOT per-record from
teacher) to avoid anchoring back to teacher quantitatively. Defaults assume
typical Layer1 delta records — override via config to match your dataset's
P10/P90.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CountLengthConfig:
    # Per-record count band on number of interests.
    count_band: tuple[int, int] = (2, 9)
    # Per-record total-completion-char band (post-think).
    length_band: tuple[int, int] = (200, 2000)
    # Width of the soft taper outside the hard band, in band-units.
    # 0.0 = hard 0/1 gate. >0 = linear taper to 0 over [edge, edge*taper].
    taper: float = 0.0


@dataclass
class AntiCollapseResult:
    count_gate: float
    length_gate: float
    score: float            # average of the two; what we feed to compose
    n_interests: int
    completion_chars: int


def _band_gate(value: float, lo: float, hi: float, taper: float) -> float:
    """Return 1.0 inside ``[lo, hi]``; optionally taper to 0 outside."""
    if lo <= value <= hi:
        return 1.0
    if taper <= 0:
        return 0.0
    if value < lo:
        edge = lo - taper * lo
        if value <= edge:
            return 0.0
        return (value - edge) / max(1e-6, lo - edge)
    # value > hi
    edge = hi + taper * hi
    if value >= edge:
        return 0.0
    return (edge - value) / max(1e-6, edge - hi)


def compute_anti_collapse(
    n_interests: int,
    completion_chars: int,
    cfg: CountLengthConfig | None = None,
) -> AntiCollapseResult:
    cfg = cfg or CountLengthConfig()
    c_lo, c_hi = cfg.count_band
    l_lo, l_hi = cfg.length_band
    cg = _band_gate(n_interests, c_lo, c_hi, cfg.taper)
    lg = _band_gate(completion_chars, l_lo, l_hi, cfg.taper)
    return AntiCollapseResult(
        count_gate=cg,
        length_gate=lg,
        score=(cg + lg) / 2.0,
        n_interests=n_interests,
        completion_chars=completion_chars,
    )
