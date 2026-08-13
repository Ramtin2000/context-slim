"""context-slim — prune your LLM agent's context without destroying your prompt cache.

The public surface is deliberately four functions:

``doctor``   find cache pathologies that cost money silently
``plan``     decide what is worth pruning — pure, no I/O, no mutation
``apply``    execute an approved plan
``simulate`` project cost over N future turns without calling any API

``plan`` and ``apply`` are separate so that "don't prune" is an ordinary
outcome you can inspect, rather than an exception or a silent no-op.
"""

from __future__ import annotations

from collections.abc import Sequence

from ._types import (
    BreakEven,
    Candidate,
    CostReport,
    Decision,
    Diagnostic,
    Message,
    Money,
    PrunePlan,
    Verdict,
)
from .cache import prefix as _prefix
from .cache.rates import ModelRates
from .cache.rates import get as get_rates
from .ops import expiry as _expiry
from .policy import break_even_turns, estimate_horizon
from .policy import plan as _plan_candidates

__version__ = "0.1.0"


def doctor(
    messages: Sequence[Message],
    model: str = "openai/gpt-5.6-luna",
    breakpoints: Sequence[int] | None = None,
) -> list[Diagnostic]:
    """Report cache pathologies. Costs nothing and calls nothing."""
    return _prefix.doctor(messages, get_rates(model), breakpoints)


def plan(
    messages: Sequence[Message],
    *,
    model: str = "openai/gpt-5.6-luna",
    horizon: int = 20,
    order: str = "tail_first",
    keep_recent: int = 2,
    ttl: int | None = None,
) -> PrunePlan:
    """Decide what to prune. Pure: never mutates ``messages``, never does I/O."""
    rates = get_rates(model)
    cands = _expiry.candidates(messages, order=order, keep_recent=keep_recent)
    return _plan_candidates(cands, rates, horizon, ttl)


def apply(messages: Sequence[Message], prune_plan: PrunePlan) -> tuple[list[Message], CostReport]:
    """Execute the approved edits in ``prune_plan``, returning a new message list."""
    out = [dict(m) for m in messages]
    approved = prune_plan.approved

    cost_now = Money.zero()
    saving = Money.zero()
    net = Money.zero()
    removed = 0

    for v in approved:
        i = v.candidate.index
        out[i] = _expiry.render_stub(out[i], v.reason)
        removed += v.candidate.s_tokens
        cost_now = cost_now + v.math.cost_now
        saving = saving + v.math.saving_per_turn
        net = net + v.math.net_at_horizon

    counts = {d: 0 for d in Decision}
    for v in prune_plan.verdicts:
        counts[v.decision] += 1

    report = CostReport(
        model=prune_plan.model,
        horizon=prune_plan.horizon,
        edits_applied=len(approved),
        edits_refused=counts[Decision.REFUSE],
        edits_deferred=counts[Decision.DEFER],
        tokens_removed=removed,
        cost_now=cost_now,
        saving_per_turn=saving,
        net_at_horizon=net,
    )
    return out, report


def simulate(
    messages: Sequence[Message],
    *,
    model: str = "openai/gpt-5.6-luna",
    turns: int = 20,
    order: str = "tail_first",
) -> CostReport:
    """Project the outcome of pruning over ``turns`` future turns. No API calls."""
    p = plan(messages, model=model, horizon=turns, order=order)
    _, report = apply(messages, p)
    return report


__all__ = [
    "BreakEven",
    "Candidate",
    "CostReport",
    "Decision",
    "Diagnostic",
    "Message",
    "ModelRates",
    "Money",
    "PrunePlan",
    "Verdict",
    "__version__",
    "apply",
    "break_even_turns",
    "doctor",
    "estimate_horizon",
    "get_rates",
    "plan",
    "simulate",
]
