"""CacheAlignedContext: Anchor pinning & orchestration.

A prompt cache is a *prefix* cache: OpenAI and equivalent engines hash the
static prompt sequentially in 128-token blocks, and caching only activates
once that prefix reaches a minimum length (1,024 tokens on GPT-5.6, 512 on
Claude Opus 5). Mutate byte 0 — pop the oldest turn, reorder a tool schema —
and the whole hash chain invalidates: `cached_tokens` drops to zero and every
token in the prefix is billed and re-written from scratch.

The Anchor Zone is the head of a conversation whose cumulative tokens first
reach that activation minimum: the system prompt, developer instructions,
tool schemas, and Turn 1 task, in the order a real agent loop emits them.
Everything in it is pinned — never a pruning candidate, at any cost, for any
preset. The Compaction Zone is everything after it: the tail and intermediate
turns, where all truncation, payload stripping, and deduplication happen.

``doctor``, ``plan``, ``apply``, and ``simulate`` below are the same
functional engine re-exported at the package root (``context_slim.doctor``,
etc.) — pure functions, easy to test and to compose. :class:`CacheAlignedContext`
wraps them for callers who would rather hold a stateful handle on one
conversation than pass ``messages`` and a hand-computed anchor boundary to
every call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from . import expiry as _expiry
from .cache import prefix as _prefix
from .cache.prefix import anchor_boundary
from .cache.rates import ModelRates
from .cache.rates import get as get_rates
from .checkpoint import Checkpoint
from .ledger import Ledger
from .policy import plan as _plan_candidates
from .presets import Preset
from .presets import get as get_preset
from .schemas import CostReport, Decision, Diagnostic, Message, Money, PrunePlan

__all__ = ["CacheAlignedContext", "apply", "doctor", "plan", "simulate"]


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
    preset: str | Preset | None = None,
    horizon: int | None = None,
    order: str | None = None,
    keep_recent: int | None = None,
    ttl: int | None = None,
    anchor_index: int | None = None,
) -> PrunePlan:
    """Decide what to prune. Pure: never mutates ``messages``, never does I/O.

    A ``preset`` supplies order, keep_recent, horizon and a payback cap; any
    argument passed explicitly overrides it. Every override defaults to ``None``
    rather than to the preset's value, because comparing against a default
    cannot distinguish "not passed" from "passed the same value on purpose".

    ``anchor_index``, if given, is the last index of the immutable Anchor
    Zone (see :func:`context_slim.cache.prefix.anchor_boundary`); no message
    at or before it is ever proposed as a candidate. :class:`CacheAlignedContext`
    computes and passes this automatically — plain callers of this function
    are protected only by the break-even engine's economic refusal unless
    they compute and pass it themselves.

    With no preset, ``BALANCED`` applies: tail-first, refusing anything slower
    than ~12 turns to pay back. Conservative on purpose — the measured result
    was that pruning usually loses money.
    """
    cfg = get_preset(preset) if isinstance(preset, str) else (preset or get_preset("balanced"))
    rates = get_rates(model)
    cands = _expiry.candidates(
        messages,
        order=cfg.order if order is None else order,
        keep_recent=cfg.keep_recent if keep_recent is None else keep_recent,
        anchor_index=anchor_index,
    )
    return _plan_candidates(
        cands,
        rates,
        cfg.horizon if horizon is None else horizon,
        ttl,
        cfg.max_payback_turns,
    )


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
    preset: str | Preset | None = None,
) -> CostReport:
    """Project the outcome of pruning over ``turns`` future turns. No API calls."""
    p = plan(messages, model=model, horizon=turns, order=order, preset=preset)
    _, report = apply(messages, p)
    return report


@dataclass
class CacheAlignedContext:
    """Pins the Anchor Zone and orchestrates pruning against everything after it.

    Holds one conversation's ``messages`` alongside a ``model`` and an
    optional ``preset``, and threads the computed Anchor Zone boundary through
    every ``plan()`` call automatically — the caller never has to compute or
    pass it by hand, and can never accidentally omit it.

    ``messages`` is read, never mutated: :meth:`apply` returns a new list, the
    same contract as the module-level :func:`apply`.
    """

    messages: Sequence[Message]
    model: str = "openai/gpt-5.6-luna"
    preset: str | Preset | None = None

    @property
    def rates(self) -> ModelRates:
        return get_rates(self.model)

    @property
    def anchor_index(self) -> int:
        """Last index inside the immutable Anchor Zone."""
        return anchor_boundary(self.messages, self.rates.min_cacheable_tokens)

    @property
    def anchor(self) -> list[Message]:
        """The immutable prefix: system prompt, tool schemas, Turn 1 task.

        Guaranteed byte-identical across every :meth:`plan` / :meth:`apply`
        call on this context — see ``tests/test_prefix.py``.
        """
        return list(self.messages[: self.anchor_index + 1])

    @property
    def compaction_zone(self) -> list[Message]:
        """Everything eligible for truncation: the tail and intermediate turns."""
        return list(self.messages[self.anchor_index + 1 :])

    def doctor(self) -> list[Diagnostic]:
        """Report cache pathologies for the held conversation."""
        return doctor(self.messages, model=self.model)

    def plan(
        self,
        *,
        horizon: int | None = None,
        order: str | None = None,
        keep_recent: int | None = None,
        ttl: int | None = None,
    ) -> PrunePlan:
        """Decide what to prune, with the Anchor Zone pinned automatically."""
        return plan(
            self.messages,
            model=self.model,
            preset=self.preset,
            horizon=horizon,
            order=order,
            keep_recent=keep_recent,
            ttl=ttl,
            anchor_index=self.anchor_index,
        )

    def apply(self, prune_plan: PrunePlan) -> tuple[list[Message], CostReport]:
        """Execute an approved plan against the held conversation."""
        return apply(self.messages, prune_plan)

    def simulate(self, *, turns: int = 20, order: str = "tail_first") -> CostReport:
        """Project the outcome of pruning over ``turns`` future turns. No API calls."""
        p = self.plan(horizon=turns, order=order)
        _, report = self.apply(p)
        return report

    def snapshot(
        self,
        live: Sequence[Message],
        *,
        turn: int = 0,
        ledger: Ledger | None = None,
    ) -> Checkpoint:
        """Record the compaction of ``live`` so it can be replayed byte-for-byte.

        ``live`` is this conversation *after* compaction — what
        :meth:`apply` returned and what the agent is actually sending. The
        checkpoint stores each replaced message's rendered content verbatim
        rather than the size and reason needed to re-render it, because
        re-rendering a stub measures the stub instead of the original and
        silently changes the bytes. See :mod:`context_slim.checkpoint`.
        """
        if len(live) != len(self.messages):
            raise ValueError(
                f"live conversation has {len(live)} messages, held has "
                f"{len(self.messages)} — snapshot compares them index by index, "
                "so they must be the same conversation"
            )

        anchor_index = self.anchor_index
        stubbed = {
            i: str(live[i]["content"])
            for i in range(anchor_index + 1, len(live))
            if live[i].get("content") != self.messages[i].get("content")
        }
        return Checkpoint(
            anchor_index=anchor_index,
            model=self.model,
            turn=turn,
            stubbed=stubbed,
            ledger=ledger if ledger is not None else Ledger(),
        )
