"""The break-even decision engine.

Layer 2. Turns costed candidates into one of three verdicts. The verdict — not
the edit — is what this library sells, which is why ``REFUSE`` is a normal
return value rather than an error, and why every verdict is required to explain
itself in prose a user can act on.
"""

from __future__ import annotations

from collections.abc import Iterable

from ._types import BreakEven, Candidate, Decision, Money, PrunePlan, Verdict
from .cache import model
from .cache.rates import ModelRates


def break_even_turns(
    rates: ModelRates,
    w_tokens: int,
    s_tokens: int,
    ttl: int | None = None,
) -> float | None:
    """Turns until an edit pays for itself, or ``None`` if it never does."""
    return model.break_even(rates, w_tokens, s_tokens, horizon=0, ttl=ttl).turns


def verdict(
    candidate: Candidate,
    rates: ModelRates,
    horizon: int,
    ttl: int | None = None,
    max_payback_turns: float | None = None,
) -> Verdict:
    """Decide whether a single candidate edit is worth making.

    Law 1 lives here: a ``PLAN`` is never returned for an edit whose projected
    value at the given horizon is negative. Everything else in the library
    depends on that invariant holding.
    """
    math_ = model.break_even(rates, candidate.w_tokens, candidate.s_tokens, horizon, ttl)
    decision, reason = _classify(math_, candidate, horizon)

    # A preset may cap payback more tightly than the horizon alone would. This
    # only ever downgrades a verdict, so Law 1 cannot be weakened by it.
    if (
        decision is Decision.PLAN
        and max_payback_turns is not None
        and math_.turns is not None
        and math_.turns > max_payback_turns
    ):
        decision = Decision.DEFER
        reason = (
            f"pays back in {math_.turns:.1f} turns, over this preset's "
            f"{max_payback_turns:.0f}-turn limit — held for a cheaper moment"
        )

    if decision is Decision.PLAN and math_.net_at_horizon.nano < 0:  # pragma: no cover
        raise AssertionError("Law 1 violated: PLAN emitted with negative projected value")

    return Verdict(decision=decision, candidate=candidate, math=math_, reason=reason)


def _classify(math_: BreakEven, candidate: Candidate, horizon: int) -> tuple[Decision, str]:
    ratio = candidate.w_tokens / candidate.s_tokens if candidate.s_tokens else float("inf")

    if candidate.s_tokens == 0:
        return Decision.REFUSE, "removes no tokens, so there is nothing to save"

    if math_.turns is None:
        return (
            Decision.REFUSE,
            f"never pays back: rewriting {candidate.w_tokens:,} tokens costs "
            f"{math_.cost_now} and saves nothing per turn",
        )

    if math_.net_at_horizon.nano > 0:
        return (
            Decision.PLAN,
            f"pays back after {math_.turns:.1f} turns (horizon {horizon}); "
            f"costs {math_.cost_now} now, saves {math_.saving_per_turn}/turn, "
            f"net {math_.net_at_horizon} at horizon",
        )

    if math_.net_at_horizon.nano == 0:
        return Decision.DEFER, f"exactly breaks even at horizon {horizon}; no reason to churn cache"

    # Unprofitable *here*, but the shape tells us whether waiting could help.
    # A tail-ward edit (low W/S) becomes profitable with more turns; a head-ward
    # edit is structurally doomed and should be refused outright rather than
    # left on the ledger forever.
    if math_.turns <= horizon * 3:
        return (
            Decision.DEFER,
            f"needs {math_.turns:.1f} turns to pay back but only {horizon} remain; "
            f"deferring until the cache is invalidated anyway (W/S = {ratio:.1f})",
        )

    return (
        Decision.REFUSE,
        f"structurally unprofitable: W/S = {ratio:.1f} means {math_.turns:.1f} turns to "
        f"pay back {math_.cost_now}, against a horizon of {horizon}. "
        "Prune closer to the tail instead.",
    )


def plan(
    candidates: Iterable[Candidate],
    rates: ModelRates,
    horizon: int,
    ttl: int | None = None,
    max_payback_turns: float | None = None,
) -> PrunePlan:
    """Cost every candidate and return a pure, side-effect-free plan."""
    verdicts: list[Verdict] = [
        verdict(c, rates, horizon, ttl, max_payback_turns) for c in candidates
    ]
    return PrunePlan(model=rates.key, horizon=horizon, verdicts=verdicts)


def estimate_horizon(turns_so_far: int, observed_loop_lengths: list[int] | None = None) -> int:
    """Estimate remaining turns.

    With no history, assumes a loop runs about 20 turns — short enough that the
    default errs toward refusing expensive prunes, which is the safe direction
    to be wrong in.
    """
    if observed_loop_lengths:
        typical = sorted(observed_loop_lengths)[len(observed_loop_lengths) // 2]
    else:
        typical = 20
    return max(0, typical - turns_so_far)


__all__ = [
    "Decision",
    "Money",
    "break_even_turns",
    "estimate_horizon",
    "plan",
    "verdict",
]
