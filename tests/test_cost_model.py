"""The derivation, checked by hand.

If these fail, every verdict the library produces is wrong.
"""

from __future__ import annotations

from datetime import date
from fractions import Fraction

import pytest

from context_slim.cache import rates
from context_slim.cache.model import break_even, read_cost, write_cost, write_read_ratio

LUNA = rates.GPT_5_6_LUNA
OPUS = rates.CLAUDE_OPUS_5
LEGACY = rates.LEGACY_NO_WRITE_PREMIUM


def n_turns(r, w, s, ttl=None):
    return break_even(r, w, s, horizon=0, ttl=ttl).turns


# --- the write/read ratio --------------------------------------------------


def test_write_read_ratio_is_12_5_on_current_models():
    """The ratio that drives everything: 1.25 / 0.1."""
    assert write_read_ratio(LUNA) == Fraction(25, 2)
    assert write_read_ratio(OPUS, rates.TTL_5M) == Fraction(25, 2)


def test_anthropic_1h_ttl_doubles_the_write_premium():
    assert write_read_ratio(OPUS, rates.TTL_1H) == Fraction(20, 1)


# --- the closed form N = 11.5·(W/S) − 12.5 ---------------------------------


@pytest.mark.parametrize("ratio", [2, 5, 10, 20])
def test_closed_form_matches_derivation(ratio):
    s = 1_000
    w = s * ratio
    expected = 11.5 * ratio - 12.5
    assert n_turns(LUNA, w, s) == pytest.approx(expected)


def test_tail_prune_is_immediately_profitable():
    """W == S: everything after the edit point was removed, so nothing is
    rewritten. There is no payback period at all."""
    assert n_turns(LUNA, 5_000, 5_000) == 0.0


def test_head_prune_never_pays_back_in_a_real_loop():
    """Shaving 1k off the front of a 10k cached prefix takes 102.5 turns."""
    assert n_turns(LUNA, 10_000, 1_000) == pytest.approx(102.5)


def test_head_prune_is_uneconomic_even_without_a_write_premium():
    """The important nuance: removing the 1.25x write premium entirely still
    leaves a head prune needing 80 turns. The premium makes it worse, but the
    real driver is how cheap cache *reads* are — at 0.1x, the savings you are
    buying are tiny relative to the rewrite you are paying for."""
    assert n_turns(LEGACY, 10_000, 1_000) == pytest.approx(80.0)


# --- boundaries ------------------------------------------------------------


def test_removing_nothing_never_pays_back():
    assert n_turns(LUNA, 1_000, 0) is None


def test_cannot_remove_more_than_follows_the_edit_point():
    with pytest.raises(ValueError, match="cannot remove more tokens"):
        break_even(LUNA, w_tokens=100, s_tokens=500, horizon=10)


def test_negative_inputs_rejected():
    with pytest.raises(ValueError):
        break_even(LUNA, w_tokens=-1, s_tokens=0, horizon=10)
    with pytest.raises(ValueError):
        break_even(LUNA, w_tokens=10, s_tokens=1, horizon=-5)


def test_unknown_ttl_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="no rate for ttl"):
        write_cost(OPUS, 1_000, ttl=999)


# --- money exactness -------------------------------------------------------


def test_costs_are_exact_integers():
    """$0.20/Mtok at 0.1x = 20 nanodollars per token, exactly."""
    assert read_cost(LUNA, 1_000).nano == 20_000
    assert write_cost(LUNA, 1_000).nano == 250_000


def test_net_at_horizon_is_linear_in_horizon():
    a = break_even(LUNA, 4_000, 1_000, horizon=10)
    b = break_even(LUNA, 4_000, 1_000, horizon=20)
    delta = b.net_at_horizon.nano - a.net_at_horizon.nano
    assert delta == a.saving_per_turn.nano * 10


# --- rate table hygiene ----------------------------------------------------


def test_rates_are_fresh():
    rates.assert_fresh(date.today())


def test_every_rate_declares_its_provenance():
    for r in rates.RATES.values():
        assert r.source.startswith("https://")
        assert r.price_confidence in ("official", "third-party", "archetype")
        assert r.cache_write_mult, f"{r.key} declares no write multipliers"


def test_unknown_model_raises_with_suggestions():
    with pytest.raises(KeyError, match="known:"):
        rates.get("openai/does-not-exist")
