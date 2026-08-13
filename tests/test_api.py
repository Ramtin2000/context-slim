"""Public API contract: purity, diagnostics, and the no-float rule."""

from __future__ import annotations

import ast
import copy
import pathlib

import context_slim
from context_slim import apply, doctor, plan, simulate
from context_slim._types import Decision

SRC = pathlib.Path(context_slim.__file__).parent


def loop(n_tools=8, body_chars=4_000):
    msgs = [{"role": "system", "content": "x" * 3_000, "cache_control": {"type": "ephemeral"}}]
    for i in range(n_tools):
        msgs.append({"role": "assistant", "content": f"calling tool {i}"})
        msgs.append({"role": "tool", "content": "y" * body_chars})
    msgs.append({"role": "user", "content": "and then?"})
    return msgs


# --- purity ----------------------------------------------------------------


def test_plan_does_not_mutate_input():
    msgs = loop()
    before = copy.deepcopy(msgs)
    plan(msgs, horizon=40)
    assert msgs == before


def test_apply_returns_a_new_list_and_leaves_the_original_alone():
    msgs = loop()
    before = copy.deepcopy(msgs)
    out, _ = apply(msgs, plan(msgs, horizon=40))
    assert msgs == before
    assert out is not msgs


# --- reporting -------------------------------------------------------------


def test_report_accounts_for_every_verdict():
    msgs = loop()
    p = plan(msgs, horizon=40)
    _, report = apply(msgs, p)
    total = report.edits_applied + report.edits_refused + report.edits_deferred
    assert total == len(p.verdicts)


def test_applied_edits_leave_a_readable_stub():
    msgs = loop()
    out, report = apply(msgs, plan(msgs, horizon=60))
    if report.edits_applied:
        stubs = [m for m in out if "elided" in str(m.get("content", ""))]
        assert stubs, "an applied edit must leave a visible stub"
        assert "tokens" in str(stubs[0]["content"])


def test_short_horizon_prunes_no_more_than_a_long_one():
    msgs = loop()
    short = simulate(msgs, turns=3)
    long = simulate(msgs, turns=200)
    assert short.edits_applied <= long.edits_applied


def test_tail_first_beats_oldest_first_on_cost():
    """The central claim, as a unit test: same candidates, different order,
    measurably different economics."""
    msgs = loop()
    tail = simulate(msgs, turns=30, order="tail_first")
    oldest = simulate(msgs, turns=30, order="oldest_first")
    assert tail.net_at_horizon.nano >= oldest.net_at_horizon.nano


# --- doctor ----------------------------------------------------------------


def test_doctor_flags_a_conversation_too_short_to_cache():
    diags = doctor([{"role": "user", "content": "hi"}])
    assert any(d.code == "below-min-prefix" for d in diags)


def test_doctor_flags_lookback_overrun():
    msgs = [{"role": "system", "content": "x" * 8_000, "cache_control": {"type": "ephemeral"}}]
    msgs += [{"role": "user", "content": f"turn {i} " * 50} for i in range(40)]
    diags = doctor(msgs, model="anthropic/claude-opus-5")
    overrun = [d for d in diags if d.code == "lookback-overrun"]
    assert overrun, "40 blocks past a breakpoint must be flagged"
    assert overrun[0].est_cost_per_turn.nano > 0


def test_doctor_warns_when_anthropic_caching_is_never_enabled():
    msgs = [{"role": "user", "content": "x" * 9_000}]
    diags = doctor(msgs, model="anthropic/claude-opus-5")
    assert any(d.code == "no-breakpoint" for d in diags)


# --- the no-float rule -----------------------------------------------------


def test_money_math_never_touches_float():
    """Law: money is exact. A float literal inside the money-carrying modules
    is a bug even if the tests happen to pass today.

    ``cache/model.py`` is deliberately exempt: it returns ``turns``, a
    dimensionless count that is legitimately a float. No *Money* value is ever
    derived from it.
    """
    offenders = []
    for path in [SRC / "_types.py", SRC / "cache" / "rates.py"]:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                offenders.append(f"{path.name}:{node.lineno} (float literal)")
            # A bare `float` in a type annotation is fine; calling float() is not.
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "float"
            ):
                offenders.append(f"{path.name}:{node.lineno} (float() call)")
    assert not offenders, f"float arithmetic in money code: {offenders}"


def test_money_refuses_float_scaling():
    from context_slim import Money

    try:
        Money.from_usd("1.00") * 0.5  # type: ignore[operator]
    except TypeError:
        return
    raise AssertionError("Money must refuse to be scaled by a float")


def test_decision_enum_is_exhaustive():
    assert {d.value for d in Decision} == {"PLAN", "DEFER", "REFUSE"}
