"""Stage 1 reward unit tests.

Covers:
- Parser: think strip, JSON repair, schema normalization.
- Gate: hard vs soft, schema failure paths.
- Fidelity: substring match, coverage floor, copy-paste hack mitigation.
- Anti-collapse: count band, length band.
- End-to-end compose: 5 bad-output fixtures + 2 good-output fixtures.

Run from ``rl_layer1/`` with::

    pytest verifier_layer1/tests -v

These tests do NOT require any GPU / network — pure-Python only.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from verifier_layer1.parser import (
    count_structure,
    flatten_evidence_actions,
    parse_completion,
    strip_think,
)
from verifier_layer1.reward.compose import (
    RewardConfig,
    RewardWeights,
    build_config,
    compute_reward,
)
from verifier_layer1.reward.count_length import CountLengthConfig, compute_anti_collapse
from verifier_layer1.reward.fidelity import FidelityConfig, compute_fidelity
from verifier_layer1.reward.gate import GateConfig, compute_gate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SAMPLE_SIGNALS: list[dict[str, Any]] = [
    {"raw_record": "2026-01-14\tBing\tBing Search web\tmsft"},
    {"raw_record": "2026-01-14\tMSN\tMSN View\tWindows 11 latest update fixes"},
    {"raw_record": "2026-01-15\tBing\tBing Search web\tseattle seahawks roster 2026"},
    {"raw_record": "2026-01-15\tEdge\tPage\tnordstromrack.com lululemon align pants"},
    {"raw_record": "2026-01-16\tBing\tBing Search web\tsuper bowl 2026 tickets"},
    {"raw_record": "2026-01-16\tMSN\tMSN View\tNFL playoffs schedule rams seahawks"},
]


GOOD_COMPLETION = """<think>
The user searched for Microsoft stock, looked at Windows updates, and has clear
NFL/Seahawks engagement plus some fashion shopping. Group accordingly.
</think>
[
  {
    "interest_name": "Microsoft Stocks & Windows Ecosystem",
    "topics": [
      {"topic_name": "MSFT stock", "evidence": [{"action": "msft"}]},
      {"topic_name": "Windows updates",
       "evidence": [{"action": "Windows 11 latest update fixes"}]}
    ]
  },
  {
    "interest_name": "Seattle Seahawks Engagement",
    "topics": [
      {"topic_name": "Seahawks news",
       "evidence": [{"action": "seattle seahawks roster 2026"},
                    {"action": "NFL playoffs schedule rams seahawks"}]},
      {"topic_name": "Super Bowl 2026",
       "evidence": [{"action": "super bowl 2026 tickets"}]}
    ]
  },
  {
    "interest_name": "Shopping & Women's Fashion",
    "topics": [
      {"topic_name": "lululemon",
       "evidence": [{"action": "nordstromrack.com lululemon align pants"}]}
    ]
  }
]
"""


# Bad 1: schema invalid (top-level is dict, not list, and uses wrong keys).
BAD_SCHEMA = """<think>...</think>
{"foo": "bar", "interest": "Sports"}
"""

# Bad 2: pure copy-paste — one interest, one topic, one evidence that is just
# the raw signal verbatim. Should fail count_gate.
BAD_COPY_PASTE = """<think>...</think>
[
  {
    "interest_name": "Search",
    "topics": [
      {"topic_name": "Search query",
       "evidence": [{"action": "msft"}]}
    ]
  }
]
"""

# Bad 3: empty list.
BAD_EMPTY = """<think>...</think>
[]
"""

# Bad 4: hallucinated evidence (actions not from input signals).
BAD_HALLUCINATED = """<think>...</think>
[
  {"interest_name": "Crypto Trading",
   "topics": [{"topic_name": "Bitcoin",
               "evidence": [{"action": "buying bitcoin at $80k"},
                            {"action": "ethereum staking yields"}]}]},
  {"interest_name": "Yoga Retreats in Bali",
   "topics": [{"topic_name": "Ubud retreats",
               "evidence": [{"action": "best yoga retreats ubud 2026"},
                            {"action": "bali wellness packages"}]}]},
  {"interest_name": "Astrophotography",
   "topics": [{"topic_name": "Telescope shopping",
               "evidence": [{"action": "celestron 8se review"}]}]}
]
"""

# Bad 5: super long, over-extracted (many interests, mostly junk).
BAD_TOO_LONG = """<think>...</think>
""" + json.dumps(
    [
        {
            "interest_name": f"Interest {i}",
            "topics": [
                {"topic_name": f"Topic {i}", "evidence": [{"action": f"random action {i}"}]}
            ],
        }
        for i in range(40)
    ]
)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------


def test_strip_think_with_block() -> None:
    text = "<think>hidden reasoning</think>\n[1,2,3]"
    out, had = strip_think(text)
    assert had is True
    assert out == "[1,2,3]"


def test_strip_think_no_block() -> None:
    text = "[1,2,3]"
    out, had = strip_think(text)
    assert had is False
    assert out == text


def test_strip_think_only_close_tag() -> None:
    # Some templates start mid-think and just emit the close tag.
    text = "midway thoughts</think>final"
    out, had = strip_think(text)
    assert had is True
    assert out == "final"


def test_parse_good_completion() -> None:
    res = parse_completion(GOOD_COMPLETION)
    assert res.parse_ok is True
    assert res.schema_ok is True
    assert res.had_think_block is True
    assert res.parsed is not None
    counts = count_structure(res.parsed)
    # 3 interests, 2+2+1 = 5 topics, 1+1+2+1+1 = 6 evidence items.
    assert counts == {"n_interests": 3, "n_topics": 5, "n_evidence": 6}
    actions = flatten_evidence_actions(res.parsed)
    assert "msft" in actions


def test_parse_schema_failure_dict_top_level() -> None:
    res = parse_completion(BAD_SCHEMA)
    assert res.parse_ok is True
    assert res.schema_ok is False
    assert res.parsed is None


def test_parse_empty_list_is_schema_failure() -> None:
    res = parse_completion(BAD_EMPTY)
    # Empty top-level list normalizes to "no valid interests" -> schema_ok False.
    assert res.parse_ok is True
    assert res.schema_ok is False


def test_parse_with_markdown_fence() -> None:
    text = "<think>x</think>\n```json\n[{\"interest_name\": \"A\", \"topics\": []}]\n```"
    res = parse_completion(text)
    # Single interest with empty topics is still a valid normalized record.
    assert res.parse_ok is True
    assert res.schema_ok is True
    assert res.parsed is not None and len(res.parsed) == 1


def test_parse_truncated_json_repaired() -> None:
    # Realistic mild truncation — missing the trailing ``]`` only. Heavier
    # truncation (e.g., mid-string cutoff) is intentionally NOT repaired and
    # will fail the gate, which is the desired behavior for collapsed outputs.
    text = (
        '<think>x</think>\n'
        '[{"interest_name": "A", "topics": '
        '[{"topic_name": "t", "evidence": [{"action": "msft"}]}]}'
    )
    res = parse_completion(text)
    assert res.parse_ok is True
    assert res.schema_ok is True


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------


def test_gate_hard_pass() -> None:
    res = parse_completion(GOOD_COMPLETION)
    g = compute_gate(res, GateConfig(mode="hard"))
    assert g.value == 1.0 and g.pass_hard is True


def test_gate_hard_fail() -> None:
    res = parse_completion(BAD_SCHEMA)
    g = compute_gate(res, GateConfig(mode="hard"))
    assert g.value == 0.0 and g.pass_hard is False


def test_gate_soft_fail_gives_floor() -> None:
    res = parse_completion(BAD_SCHEMA)
    g = compute_gate(res, GateConfig(mode="soft", soft_value=0.1))
    assert g.value == pytest.approx(0.1)
    assert g.pass_hard is False


# ---------------------------------------------------------------------------
# Fidelity tests
# ---------------------------------------------------------------------------


def test_fidelity_perfect_match() -> None:
    actions = ["msft", "Windows 11 latest update fixes"]
    res = compute_fidelity(actions, SAMPLE_SIGNALS, FidelityConfig(coverage_floor=0.3))
    assert res.fidelity == pytest.approx(1.0)
    # 2 actions / 6 signals = 0.33 > floor 0.3
    assert res.coverage == pytest.approx(2 / 6)
    assert res.score > 0.3


def test_fidelity_hallucinated() -> None:
    actions = ["buying bitcoin at $80k", "celestron 8se review"]
    res = compute_fidelity(actions, SAMPLE_SIGNALS)
    assert res.fidelity == 0.0
    assert res.score == 0.0


def test_fidelity_copy_paste_hack_blocked_by_coverage_floor() -> None:
    # Single verbatim evidence -> fidelity = 1.0 but coverage = 1/6 = 0.17.
    # With floor=0.3 the score is 1.0 * 0.3 = 0.3 (not 1.0).
    actions = ["msft"]
    res = compute_fidelity(actions, SAMPLE_SIGNALS, FidelityConfig(coverage_floor=0.3))
    assert res.fidelity == 1.0
    assert res.coverage < 0.3
    assert res.score == pytest.approx(0.3)


def test_fidelity_short_action_skipped() -> None:
    # 'a' is below min_action_chars=3 → should not count as hit even if it
    # appears in many signals.
    actions = ["a", "msft"]
    res = compute_fidelity(actions, SAMPLE_SIGNALS, FidelityConfig(min_action_chars=3))
    # 1 hit out of 2 emitted -> 0.5
    assert res.fidelity == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Anti-collapse tests
# ---------------------------------------------------------------------------


def test_anti_collapse_inside_band() -> None:
    res = compute_anti_collapse(
        n_interests=4, completion_chars=800,
        cfg=CountLengthConfig(count_band=(2, 9), length_band=(200, 2000)),
    )
    assert res.count_gate == 1.0 and res.length_gate == 1.0
    assert res.score == 1.0


def test_anti_collapse_too_few_interests() -> None:
    res = compute_anti_collapse(
        n_interests=1, completion_chars=800,
        cfg=CountLengthConfig(count_band=(2, 9)),
    )
    assert res.count_gate == 0.0


def test_anti_collapse_too_long() -> None:
    res = compute_anti_collapse(
        n_interests=40, completion_chars=20000,
        cfg=CountLengthConfig(count_band=(2, 9), length_band=(200, 2000)),
    )
    assert res.count_gate == 0.0 and res.length_gate == 0.0


# ---------------------------------------------------------------------------
# End-to-end compose tests
# ---------------------------------------------------------------------------


def _stage1_cfg() -> RewardConfig:
    # Use hard gate so failure paths return 0 (cleaner assertions). The
    # production default is soft, exercised separately below.
    return build_config({
        "stage": 1,
        "gate": {"mode": "hard"},
        "count_length": {"count_band": [2, 9], "length_band": [200, 2000]},
        "fidelity": {"coverage_floor": 0.3, "min_action_chars": 3},
        "weights": {
            "anti_collapse": 0.5,
            "anti_hallucination": 0.4,
            "fidelity": 0.1,
        },
    })


def test_reward_good_output_high() -> None:
    out = compute_reward(GOOD_COMPLETION, SAMPLE_SIGNALS, _stage1_cfg())
    assert out.components["gate"] == 1.0
    assert out.components["fidelity"] == pytest.approx(1.0)
    assert out.components["count_gate"] == 1.0
    assert out.components["length_gate"] == 1.0
    # 0.5*1 + 0.4*1 + 0.1*score; score = 1.0 * max(6/6=1, 0.3) = 1.0
    assert out.reward >= 0.95


def test_reward_bad_schema_zero() -> None:
    out = compute_reward(BAD_SCHEMA, SAMPLE_SIGNALS, _stage1_cfg())
    assert out.components["schema_ok"] is False
    assert out.reward == 0.0


def test_reward_copy_paste_low() -> None:
    out = compute_reward(BAD_COPY_PASTE, SAMPLE_SIGNALS, _stage1_cfg())
    # Count gate fails (1 interest < lo=2), so anti_collapse contributes 0.
    # Fidelity = 1.0 but coverage_floor caps the score at 0.3 → 0.1*0.3=0.03.
    # Anti-hallu = fidelity = 1.0 → 0.4*1.0 = 0.4.
    # Total ≈ 0.43, well below the good-output band.
    assert out.components["count_gate"] == 0.0
    assert out.reward < 0.5
    assert out.reward > 0.0


def test_reward_empty_list_zero() -> None:
    out = compute_reward(BAD_EMPTY, SAMPLE_SIGNALS, _stage1_cfg())
    assert out.components["schema_ok"] is False
    assert out.reward == 0.0


def test_reward_hallucinated_penalized() -> None:
    out = compute_reward(BAD_HALLUCINATED, SAMPLE_SIGNALS, _stage1_cfg())
    # Schema passes, count_gate passes (3 interests in [2,9]).
    # Fidelity ~ 0 → anti_hallucination ~ 0, fidelity_score ~ 0.
    # So reward ≈ 0.5 * anti_collapse only.
    assert out.components["schema_ok"] is True
    assert out.components["fidelity"] == 0.0
    assert out.components["anti_hallucination"] == 0.0
    assert out.reward < 0.6


def test_reward_too_long_zero() -> None:
    out = compute_reward(BAD_TOO_LONG, SAMPLE_SIGNALS, _stage1_cfg())
    # 40 interests > hi=9 -> count_gate=0; total chars > 2000 -> length_gate=0
    assert out.components["count_gate"] == 0.0
    assert out.components["length_gate"] == 0.0
    # Only the fidelity_score channel might contribute slightly if any
    # hallucinated text happened to overlap; with random actions it's ~0.
    assert out.reward < 0.1


def test_reward_ordering_good_beats_all_bad() -> None:
    cfg = _stage1_cfg()
    good = compute_reward(GOOD_COMPLETION, SAMPLE_SIGNALS, cfg).reward
    bads = [
        compute_reward(c, SAMPLE_SIGNALS, cfg).reward
        for c in (BAD_SCHEMA, BAD_COPY_PASTE, BAD_EMPTY, BAD_HALLUCINATED, BAD_TOO_LONG)
    ]
    assert all(good > b for b in bads), (
        f"good={good:.3f} should exceed all bads={bads}"
    )


def test_reward_soft_gate_keeps_nonzero_on_schema_fail() -> None:
    cfg = build_config({
        "stage": 1,
        "gate": {"mode": "soft", "soft_value": 0.1},
        "count_length": {"count_band": [2, 9], "length_band": [200, 2000]},
        "weights": {"anti_collapse": 0.5, "anti_hallucination": 0.4, "fidelity": 0.1},
    })
    out = compute_reward(BAD_SCHEMA, SAMPLE_SIGNALS, cfg)
    # Soft gate short-circuits to 0 only when gate==0; soft_value=0.1 lets the
    # full inner pipeline run on best-effort parsed=None → all components 0
    # except gate. Sanity: reward should be ≥ 0.
    assert out.reward == 0.0  # because compose short-circuits when parsed is None
    # The gate value itself was non-zero though — useful for diagnostic logs.
    assert out.components["gate"] == pytest.approx(0.1)


def test_reward_weights_normalize() -> None:
    # Use non-normalized weights; verify reward stays bounded in [0, 1].
    cfg = build_config({
        "stage": 1,
        "gate": {"mode": "hard"},
        "weights": {"anti_collapse": 5, "anti_hallucination": 4, "fidelity": 1},
    })
    out = compute_reward(GOOD_COMPLETION, SAMPLE_SIGNALS, cfg)
    assert 0.0 <= out.reward <= 1.0


def test_reward_config_rejects_unsupported_stage() -> None:
    with pytest.raises(NotImplementedError):
        build_config({"stage": 2})


# ---------------------------------------------------------------------------
# Smoke: server-equivalent dict-in / dict-out works.
# ---------------------------------------------------------------------------


def test_compose_output_is_json_serializable() -> None:
    from verifier_layer1.reward.compose import reward_output_to_dict
    out = compute_reward(GOOD_COMPLETION, SAMPLE_SIGNALS, _stage1_cfg())
    d = reward_output_to_dict(out)
    # Round-trip through JSON to catch non-serializable values.
    json.dumps(d)
