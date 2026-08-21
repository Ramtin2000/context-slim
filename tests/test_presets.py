"""Presets are a stance on how much cache damage is acceptable."""

from __future__ import annotations

import pytest

from context_slim import plan
from context_slim._types import Decision
from context_slim.presets import PRESETS, get

LOOP: list[dict[str, object]] = [{"role": "system", "content": "S" * 32_000}]
for _i in range(10):
    LOOP += [
        {"role": "user", "content": f"step {_i}"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": f"c{_i}", "type": "function",
                         "function": {"name": "inspect", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": f"c{_i}", "content": "RESULT " * 400},
    ]


def _counts(preset: str) -> dict[Decision, int]:
    p = plan(LOOP, preset=preset)
    return {d: sum(1 for v in p.verdicts if v.decision is d) for d in Decision}


def test_every_preset_is_tail_first() -> None:
    """Oldest-first was dominated at every horizon measured, so nothing selects it."""
    assert all(p.order == "tail_first" for p in PRESETS.values())


def test_unknown_preset_names_its_options() -> None:
    with pytest.raises(KeyError, match="unknown preset"):
        get("reckless")


def test_conservative_presets_approve_no_more_than_liberal_ones() -> None:
    strict = _counts("cache-preserving")[Decision.PLAN]
    mid = _counts("balanced")[Decision.PLAN]
    loose = _counts("aggressive")[Decision.PLAN]
    assert strict <= mid <= loose


def test_cache_preserving_approves_nothing_on_a_short_horizon() -> None:
    assert _counts("cache-preserving")[Decision.PLAN] == 0


def test_explicit_arguments_override_the_preset() -> None:
    """Passing the preset's own value explicitly must still be honoured."""
    p = plan(LOOP, preset="aggressive", horizon=1)
    assert p.horizon == 1
    assert all(v.decision is not Decision.PLAN for v in p.verdicts)


def test_default_is_balanced() -> None:
    assert plan(LOOP).horizon == PRESETS["balanced"].horizon


def test_law_one_holds_under_every_preset() -> None:
    for name in PRESETS:
        for v in plan(LOOP, preset=name).verdicts:
            if v.decision is Decision.PLAN:
                assert v.math.net_at_horizon.nano >= 0
