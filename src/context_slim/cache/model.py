"""The cost model.

This module contains the derivation the whole project rests on. It is short
on purpose: if the arithmetic here is wrong, every downstream verdict is wrong,
so it should be readable in one sitting and checkable by hand.

Setup
-----
A conversation has a cached prefix of ``P`` tokens. You want to delete ``S``
tokens from somewhere inside it. Because a prompt cache is a *prefix* cache,
editing at some point invalidates everything after that point: let ``W`` be the
number of cached tokens sitting at or after the edit point (so ``W >= S``).

The turn on which you prune::

    unpruned:  P·read
    pruned:    (P − W)·read  +  (W − S)·write
    delta_now = (W − S)·write − W·read          # the price of pruning

Every turn after that::

    saving_per_turn = S·read

So the number of turns before the edit pays for itself is::

    N = delta_now / saving_per_turn
      = ((W − S)·write − W·read) / (S·read)

Substituting the multipliers shared by Anthropic Claude and OpenAI GPT-5.6+
(``read = 0.1``, ``write = 1.25``) gives the closed form::

    N = 11.5·(W/S) − 12.5

Two sanity checks, both of which fall out correctly:

* **Tail prune** (``W == S`` — you deleted everything after the edit point, so
  nothing needs rewriting): ``N = −1``. Profitable immediately.
* **Head prune** (``W = 10·S`` — shaving 5k off the front of a 50k prefix):
  ``N = 102.5`` turns. No real agent loop runs that long.

The ``12.5`` in the closed form is ``write / read`` — the ratio between what a
provider charges to fill the cache and what it charges to read it. That ratio,
not the token count, is what decides whether pruning is worth doing.
"""

from __future__ import annotations

from fractions import Fraction

from .._types import BreakEven, Money
from .rates import ModelRates

_PER_MTOK = 1_000_000


def token_cost(rates: ModelRates, tokens: int, mult: Fraction) -> Money:
    """Exact cost of ``tokens`` input tokens at ``mult`` times the base rate.

    Kept in :class:`~fractions.Fraction` until the final conversion so that
    chained multipliers never accumulate rounding error.
    """
    if tokens < 0:
        raise ValueError("tokens must be non-negative")
    exact = Fraction(tokens * rates.input_nano_per_mtok, _PER_MTOK) * mult
    return Money(int(exact))


def read_cost(rates: ModelRates, tokens: int) -> Money:
    """Cost of reading ``tokens`` from a warm cache."""
    return token_cost(rates, tokens, rates.cache_read_mult)


def write_cost(rates: ModelRates, tokens: int, ttl: int | None = None) -> Money:
    """Cost of writing ``tokens`` into the cache."""
    return token_cost(rates, tokens, rates.write_mult(ttl))


def break_even(
    rates: ModelRates,
    w_tokens: int,
    s_tokens: int,
    horizon: int,
    ttl: int | None = None,
) -> BreakEven:
    """Cost a single candidate edit. See the module docstring for the derivation.

    ``horizon`` is how many more turns the loop is expected to run. It is the
    input users most often get wrong, and the one the verdict is most sensitive
    to — a prune that is excellent at 50 turns can be a waste at 10.
    """
    if s_tokens < 0 or w_tokens < 0:
        raise ValueError("token counts must be non-negative")
    if w_tokens < s_tokens:
        raise ValueError(
            f"w_tokens ({w_tokens}) < s_tokens ({s_tokens}): cannot remove more tokens "
            "than sit after the edit point"
        )
    if horizon < 0:
        raise ValueError("horizon must be non-negative")

    cost_now = write_cost(rates, w_tokens - s_tokens, ttl) - read_cost(rates, w_tokens)
    saving_per_turn = read_cost(rates, s_tokens)
    net_at_horizon = Money(saving_per_turn.nano * horizon - cost_now.nano)

    turns: float | None
    if cost_now.nano <= 0:
        # The rewrite is cheaper than the reads it replaces — free money.
        turns = 0.0
    elif saving_per_turn.nano <= 0:
        # Nothing is actually saved per turn, so it never pays back.
        turns = None
    else:
        turns = cost_now.nano / saving_per_turn.nano

    return BreakEven(
        turns=turns,
        cost_now=cost_now,
        saving_per_turn=saving_per_turn,
        net_at_horizon=net_at_horizon,
    )


def write_read_ratio(rates: ModelRates, ttl: int | None = None) -> Fraction:
    """``write / read`` — the single number that characterises a provider's
    pruning economics. 12.5 on Claude (5m TTL) and on GPT-5.6+; 10 on a model
    with no write premium."""
    return rates.write_mult(ttl) / rates.cache_read_mult
