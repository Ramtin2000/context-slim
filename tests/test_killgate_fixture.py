"""The gate costs real money, so its fixture must be valid before it runs.

A malformed tool-call triple 400s on the first request — after the pre-flight
checks have already passed. These tests are the cheap way to catch that.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bench.killgate import build_loop, prune
from context_slim.cache.prefix import message_tokens


def _assert_wire_valid(msgs: list[dict]) -> None:
    """OpenAI rejects a tool message not answering a preceding tool_calls."""
    for i, m in enumerate(msgs):
        if m.get("role") != "tool":
            continue
        assert i > 0, "tool message cannot be first"
        prev = msgs[i - 1]
        assert prev.get("role") == "assistant", f"msg {i} follows {prev.get('role')}"
        ids = {c["id"] for c in prev.get("tool_calls") or []}
        assert m.get("tool_call_id") in ids, f"msg {i} answers no call in {ids}"


def test_loop_is_wire_valid() -> None:
    _assert_wire_valid(build_loop(2_000, 6, 100))


def test_every_prefix_window_is_wire_valid() -> None:
    # The gate sends growing prefixes, so each one must stand alone.
    loop = build_loop(2_000, 6, 100)
    for t in range(1, 7):
        _assert_wire_valid(loop[: 1 + t * 3])


def test_pruning_preserves_tool_call_pairing() -> None:
    # render_stub rewrites content; it must not drop tool_call_id.
    loop = build_loop(2_000, 8, 100)
    for condition in ("oldest_first", "tail_first"):
        _assert_wire_valid(prune(loop, condition))


def test_tool_calls_are_counted() -> None:
    # An assistant message with content=None still costs tokens.
    caller = build_loop(2_000, 1, 100)[2]
    assert caller["content"] is None
    assert message_tokens(caller) > 5, "tool_calls payload must be billed"
