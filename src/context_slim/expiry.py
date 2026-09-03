"""Tool-result expiry: candidate generation, and turn-atomic removal.

Layer 4 plumbing. The interesting part of candidate generation is the
*ordering*: every shipping tool in this space clears oldest-first, which is
the cache-pessimal choice because it maximises W on every fire. Generating
tail-first candidates costs nothing and is the whole reason ``candidates()``
exists.

The other half of this module is Hard Protocol Invariant 1: OpenAI's Chat
Completions API returns a 400 if an assistant ``tool_calls`` message survives
without every matching ``role: "tool"`` reply, or vice versa. A tool-calling
turn is therefore atomic — it can be edited only as a whole (``ATOMIC_PURGE``,
via :func:`atomic_purge`) or left in place with its content shrunk
(``TOMBSTONE``, via :func:`tombstone`), never split.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from re import Pattern

from .cache.prefix import message_tokens
from .providers import adapter_for
from .schemas import Candidate, Message

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
    anchor_index: int | None = None,
) -> list[Candidate]:
    """Propose tool results to collapse.

    ``order`` is ``"tail_first"`` (default) or ``"oldest_first"``. The latter
    exists so the benchmark can reproduce what every other tool does, and is
    not a recommended setting.

    ``anchor_index``, if given, is the last index of the immutable Anchor Zone
    (see :func:`context_slim.cache.prefix.anchor_boundary`). Every message at
    or before it is excluded unconditionally — a structural floor underneath
    the break-even engine's economic refusal, not a replacement for it.
    """
    if order not in ("tail_first", "oldest_first"):
        raise ValueError(f"order must be 'tail_first' or 'oldest_first', got {order!r}")

    n = len(messages)
    eligible = []
    for i, m in enumerate(messages):
        if anchor_index is not None and i <= anchor_index:
            continue  # inside the Anchor Zone; never a candidate, at any cost
        if not _is_tool_result(m):
            continue
        if i >= n - keep_recent:
            continue  # the most recent results are always live
        if pin is not None and pin.search(_text_of(m)):
            continue
        eligible.append(i)

    if order == "tail_first":
        eligible.reverse()

    # suffix_tokens(messages, i) for every eligible i would recompute overlapping
    # sums from scratch — O(n) per call, O(n^2) overall on an n-message loop.
    # A single backward pass turns that into one O(n) prefix-sum table instead.
    per_message = [message_tokens(m) for m in messages]
    suffix_totals = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_totals[i] = suffix_totals[i + 1] + per_message[i]

    out: list[Candidate] = []
    for i in eligible:
        saved = per_message[i]
        if saved <= 0:
            continue
        out.append(
            Candidate(
                kind="tool_result",
                index=i,
                w_tokens=suffix_totals[i],
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


class PurgeMode(str, Enum):
    """How an unwanted tool-calling turn is shrunk, per Hard Protocol Invariant 1."""

    ATOMIC_PURGE = "atomic_purge"
    """Evict the assistant ``tool_calls`` message and every matching ``tool``
    reply together. Nothing of the turn survives."""

    TOMBSTONE = "tombstone"
    """Keep the turn's shape — same ``role`` and ``tool_call_id`` — and shrink
    only ``content``. Costs more tokens than a purge but keeps the assistant's
    causal trace that the call happened."""


def _tool_call_ids(message: Message) -> list[str]:
    return [str(tc.get("id")) for tc in (message.get("tool_calls") or []) if tc.get("id")]


@dataclass(frozen=True)
class ToolCallBatch:
    """An assistant ``tool_calls`` message and the indices of every
    ``role: "tool"`` message that answers it. The unit ``atomic_purge`` removes."""

    assistant_index: int
    call_ids: tuple[str, ...]
    result_indices: tuple[int, ...]


def tool_call_batches(messages: Sequence[Message]) -> list[ToolCallBatch]:
    """Pair every OpenAI-shaped assistant ``tool_calls`` message with its replies.

    A batch is not necessarily contiguous — nothing stops an agent loop from
    interleaving unrelated turns between a call and its answer — so replies are
    matched by ``tool_call_id`` across the rest of the transcript, not by
    position.
    """
    batches: list[ToolCallBatch] = []
    for i, m in enumerate(messages):
        ids = _tool_call_ids(m)
        if not ids:
            continue
        wanted = set(ids)
        result_indices: list[int] = []
        for j in range(i + 1, len(messages)):
            cid = str(messages[j].get("tool_call_id", ""))
            if messages[j].get("role") == "tool" and cid in wanted:
                result_indices.append(j)
                wanted.discard(cid)
        batches.append(ToolCallBatch(i, tuple(ids), tuple(result_indices)))
    return batches


def atomic_purge(messages: Sequence[Message], assistant_index: int) -> list[Message]:
    """Evict an assistant ``tool_calls`` message and every matching reply (ATOMIC_PURGE).

    OpenAI's Chat Completions API returns a 400 if either half survives without
    the other — an assistant ``tool_calls: [{"id": "call_xyz", ...}]`` with no
    ``role: "tool", tool_call_id: "call_xyz"`` reply, or the reverse. Evicting
    the whole batch in one edit is the only way to shrink a tool-calling turn
    that keeps the remaining transcript valid.

    Raises ``ValueError`` if ``assistant_index`` does not name a message with
    ``tool_calls`` — there is no batch to purge.
    """
    batch = next(
        (b for b in tool_call_batches(messages) if b.assistant_index == assistant_index), None
    )
    if batch is None:
        raise ValueError(f"message {assistant_index} is not an assistant tool_calls message")
    drop = {batch.assistant_index, *batch.result_indices}
    return [m for i, m in enumerate(messages) if i not in drop]


def tombstone(message: Message) -> Message:
    """Shrink a tool result to the minimal schema-valid payload (TOMBSTONE).

    Keeps ``role`` and the call-linking id (``tool_call_id`` on OpenAI,
    ``tool_use_id`` inside an Anthropic ``tool_result`` block) byte-identical —
    only ``content`` shrinks, to ``{"compacted": true}``. Delegates to the
    provider adapter so both wire shapes produce a still-valid message.
    """
    adapter = adapter_for([message])
    return adapter.stub(message, json.dumps({"compacted": True}))


__all__ = [
    "DEFAULT_PIN",
    "PurgeMode",
    "ToolCallBatch",
    "atomic_purge",
    "candidates",
    "render_stub",
    "tombstone",
    "tool_call_batches",
]
