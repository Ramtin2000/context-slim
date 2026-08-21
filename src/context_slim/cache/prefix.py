"""Where does the cache actually end?

Layer 1's other half. Two of the pathologies detected here cost money with no
pruning involved at all — they are pure diagnostics, and in practice they are
the first thing worth running against a real agent loop.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .._types import Diagnostic, Message, Money
from .model import read_cost
from .rates import ModelRates

LOOKBACK_BLOCKS = 20
"""Anthropic checks at most 20 positions behind a breakpoint when looking for a
prior cache entry. Grow past that and the hit is silently missed — no error,
no warning, just a bill."""

MAX_BREAKPOINTS = 4


# Calibrated against 24 live gpt-5.6-luna requests (bench/results/validation.json).
# The uncalibrated form over-predicted by a factor of 1.2608 with sd 0.0113 - a
# 0.9% relative spread, so the miss was one constant rather than a modelling
# error. Dividing it out left a maximum residual of 1.77% across every request.
#
# Two caveats worth stating plainly:
#
# 1. This constant is fit to gpt-5.6-luna on one validation set. It is a scale
#    factor for one tokenizer family, not a universal correction.
# 2. The estimator sits INSIDE the budget-triggered pruning policy, so changing
#    it changes which messages get stubbed. That means you cannot recalibrate
#    and replay against an old run - the policy would have behaved differently.
#    Verification has to scale the recorded predictions, not recompute them.
#
# It is also why no reported dollar figure in this project comes from this
# function - those all read the provider's own usage counters.
_CALIBRATION = 1.2608


def estimate_tokens(text: str) -> int:
    """Dependency-free token estimate, calibrated against measured usage.

    Roughly chars/5 with a correction for whitespace runs, then divided by a
    measured constant. Accurate to within ~2% on the validation set; still only
    used to *rank* candidates, never to report a dollar figure.
    """
    if not text:
        return 0
    raw = (len(text) + len(text.split())) / 5
    return max(1, int(raw / _CALIBRATION))


def message_tokens(message: Message) -> int:
    # Tool calls are billed but live outside ``content`` — an assistant message
    # that only makes a call has content=None and would otherwise count as ~1
    # token while costing dozens. Undercounting here shows up directly as error
    # in the predicted-vs-actual validation, so it is counted alongside.
    extra = 0
    tool_calls = message.get("tool_calls")
    if tool_calls:
        extra = estimate_tokens(json.dumps(tool_calls))

    content = message.get("content", "")
    if isinstance(content, str):
        return estimate_tokens(content) + extra
    if content is None:
        return extra
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                total += estimate_tokens(text if isinstance(text, str) else json.dumps(block))
            else:
                total += estimate_tokens(str(block))
        return total + extra
    return estimate_tokens(json.dumps(content)) + extra


def total_tokens(messages: Sequence[Message]) -> int:
    return sum(message_tokens(m) for m in messages)


def suffix_tokens(messages: Sequence[Message], index: int) -> int:
    """Tokens from ``index`` to the end — this is ``W`` for an edit at ``index``."""
    return sum(message_tokens(m) for m in messages[index:])


@dataclass(frozen=True)
class PrefixInfo:
    tokens: int
    cacheable: bool
    reason: str


def cacheable_prefix(messages: Sequence[Message], rates: ModelRates) -> PrefixInfo:
    tokens = total_tokens(messages)
    if tokens < rates.min_cacheable_tokens:
        return PrefixInfo(
            tokens=tokens,
            cacheable=False,
            reason=(
                f"{tokens:,} tokens is below {rates.model}'s {rates.min_cacheable_tokens:,}-token "
                "minimum, so nothing is cached and pruning cannot help"
            ),
        )
    return PrefixInfo(tokens=tokens, cacheable=True, reason=f"{tokens:,} tokens cacheable")


def _breakpoint_indices(messages: Sequence[Message]) -> list[int]:
    out = []
    for i, m in enumerate(messages):
        if m.get("cache_control"):
            out.append(i)
            continue
        content: Any = m.get("content")
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("cache_control") for b in content
        ):
            out.append(i)
    return out


def doctor(
    messages: Sequence[Message],
    rates: ModelRates,
    breakpoints: Sequence[int] | None = None,
) -> list[Diagnostic]:
    """Find cache pathologies that cost money silently."""
    found: list[Diagnostic] = []
    bps = list(breakpoints) if breakpoints is not None else _breakpoint_indices(messages)

    info = cacheable_prefix(messages, rates)
    if not info.cacheable:
        found.append(Diagnostic(code="below-min-prefix", severity="info", message=info.reason))
        return found

    if len(bps) > MAX_BREAKPOINTS:
        found.append(
            Diagnostic(
                code="too-many-breakpoints",
                severity="error",
                message=(
                    f"{len(bps)} cache breakpoints declared; the API accepts at most "
                    f"{MAX_BREAKPOINTS}. The extras are ignored."
                ),
            )
        )

    if bps:
        blocks_past = len(messages) - 1 - max(bps)
        if blocks_past > LOOKBACK_BLOCKS:
            missed = suffix_tokens(messages, max(bps))
            found.append(
                Diagnostic(
                    code="lookback-overrun",
                    severity="error",
                    message=(
                        f"{blocks_past} blocks past the last cache breakpoint, but the lookback "
                        f"window is {LOOKBACK_BLOCKS}. You are missing the cache hit entirely. "
                        "Add a breakpoint nearer the end of the conversation."
                    ),
                    est_cost_per_turn=read_cost(rates, missed) * 9,
                )
            )
    elif rates.provider == "anthropic":
        found.append(
            Diagnostic(
                code="no-breakpoint",
                severity="warning",
                message=(
                    "No cache_control breakpoint found. Anthropic caching is opt-in — "
                    "without a breakpoint nothing is cached and every turn pays full price."
                ),
                est_cost_per_turn=read_cost(rates, info.tokens) * 9,
            )
        )

    if not found:
        found.append(
            Diagnostic(code="ok", severity="info", message="no cache pathologies detected")
        )
    return found


__all__ = [
    "LOOKBACK_BLOCKS",
    "MAX_BREAKPOINTS",
    "Money",
    "PrefixInfo",
    "cacheable_prefix",
    "doctor",
    "estimate_tokens",
    "message_tokens",
    "suffix_tokens",
    "total_tokens",
]
