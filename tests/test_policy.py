"""Law 1: context-slim must never cost more than it saves."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from context_slim._types import Candidate, Decision
from context_slim.cache import rates
from context_slim.policy import plan, verdict

LUNA = rates.GPT_5_6_LUNA


def cand(w, s, index=0):
    return Candidate(kind="tool_result", index=index, w_tokens=w, s_tokens=s)


# --- the invariant ---------------------------------------------------------


@settings(max_examples=2_000)
@given(
    s=st.integers(min_value=0, max_value=50_000),
    extra=st.integers(min_value=0, max_value=200_000),
    horizon=st.integers(min_value=0, max_value=500),
)
def test_law_1_never_plans_a_net_negative_edit(s, extra, horizon):
    v = verdict(cand(w=s + extra, s=s), LUNA, horizon=horizon)
    if v.decision is Decision.PLAN:
        assert v.math.net_at_horizon.nano > 0


@settings(max_examples=500)
@given(
    s=st.integers(min_value=1, max_value=20_000),
    extra=st.integers(min_value=0, max_value=100_000),
    horizon=st.integers(min_value=0, max_value=300),
)
def test_every_verdict_explains_itself(s, extra, horizon):
    v = verdict(cand(w=s + extra, s=s), LUNA, horizon=horizon)
    assert v.reason.strip(), "a verdict with no reason is unactionable"


@settings(max_examples=300)
@given(
    s=st.integers(min_value=1, max_value=10_000),
    extra=st.integers(min_value=0, max_value=50_000),
    horizon=st.integers(min_value=0, max_value=200),
)
def test_longer_horizon_never_makes_a_prune_less_attractive(s, extra, horizon):
    a = verdict(cand(w=s + extra, s=s), LUNA, horizon=horizon)
    b = verdict(cand(w=s + extra, s=s), LUNA, horizon=horizon + 25)
    assert b.math.net_at_horizon.nano >= a.math.net_at_horizon.nano


# --- the three outcomes are all reachable ----------------------------------


def test_tail_prune_is_planned():
    v = verdict(cand(w=5_000, s=4_000), LUNA, horizon=20)
    assert v.decision is Decision.PLAN
    assert "pays back" in v.reason


def test_head_prune_is_refused_with_the_numbers():
    v = verdict(cand(w=50_000, s=1_000), LUNA, horizon=20)
    assert v.decision is Decision.REFUSE
    assert "structurally unprofitable" in v.reason
    assert "W/S" in v.reason


def test_marginal_prune_is_deferred_not_discarded():
    """Not worth it now, but the shape says waiting could pay — that is the
    ledger's job later, so it must not be thrown away.

    W/S = 3 needs 22 turns; a 10-turn horizon does not reach it, but 22 is
    close enough that a cache invalidation would make the edit free."""
    v = verdict(cand(w=3_000, s=1_000), LUNA, horizon=10)
    assert v.decision is Decision.DEFER
    assert v.math.turns == 22.0


def test_zero_saving_is_refused():
    v = verdict(cand(w=1_000, s=0), LUNA, horizon=100)
    assert v.decision is Decision.REFUSE
    assert "nothing to save" in v.reason


# --- plan aggregation ------------------------------------------------------


def test_plan_reports_only_approved_savings():
    p = plan([cand(5_000, 4_000, 0), cand(50_000, 1_000, 1)], LUNA, horizon=20)
    assert len(p.approved) == 1
    assert p.projected_saving.nano > 0
