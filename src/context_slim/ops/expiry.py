"""Tool-result expiry — candidate generation only.

Layer 4 plumbing. The interesting part is the *ordering*: every shipping tool
in this space clears oldest-first, which is the cache-pessimal choice because
it maximises W on every fire. Generating tail-first candidates costs nothing
and is the whole reason this module exists.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from re import Pattern

from .._types import Candidate, Message
from ..cache.prefix import message_tokens, suffix_tokens

DEFAULT_PIN = re.compile(r"Traceback|ERROR|Exception|error:", re.IGNORECASE)
"""Errors are pinned by default. A stale error is still the most useful thing
in an agent's context, and re-deriving one is expensive."""


def _is_tool_result(message: Message) -> bool:
    if message.get("role") == "tool":
        return True
    content = message.get("content")
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") in ("tool_result", "function_call_output")
            for b in content
        )
    return False


def _text_of(message: Message) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return str(content)


def candidates(
    messages: Sequence[Message],
    *,
    order: str = "tail_first",
    keep_recent: int = 2,
    pin: Pattern[str] | None = DEFAULT_PIN,
) -> list[Candidate]:
    """Propose tool results to collapse.

    ``order`` is ``"tail_first"`` (default) or ``"oldest_first"``. The latter
    exists so the benchmark can reproduce what every other tool does, and is
    not a recommended setting.
    """
    if order not in ("tail_first", "oldest_first"):
        raise ValueError(f"order must be 'tail_first' or 'oldest_first', got {order!r}")

    n = len(messages)
    eligible = []
    for i, m in enumerate(messages):
        if not _is_tool_result(m):
            continue
        if i >= n - keep_recent:
            continue  # the most recent results are always live
        if pin is not None and pin.search(_text_of(m)):
            continue
        eligible.append(i)

    if order == "tail_first":
        eligible.reverse()

    out: list[Candidate] = []
    for i in eligible:
        saved = message_tokens(messages[i])
        if saved <= 0:
            continue
        out.append(
            Candidate(
                kind="tool_result",
                index=i,
                w_tokens=suffix_tokens(messages, i),
                s_tokens=saved,
                detail=f"tool result at message {i}",
            )
        )
    return out


def render_stub(message: Message, reason: str) -> Message:
    """Collapse a tool result to a stub that preserves its identity.

    The agent keeps the causal trace — that the call happened, and what it
    returned in outline — and can always re-issue the call if it needs the body.
    """
    size = message_tokens(message)
    stub = dict(message)
    stub["content"] = f"[tool result elided: ~{size:,} tokens, {reason}]"
    return stub


__all__ = ["DEFAULT_PIN", "candidates", "render_stub"]
