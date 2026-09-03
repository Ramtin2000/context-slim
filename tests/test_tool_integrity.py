"""Tool call atomicity: Hard Protocol Invariant 1.

OpenAI's Chat Completions API returns a 400 if an assistant ``tool_calls``
message survives without every matching ``role: "tool"`` reply, or vice
versa. A tool-calling turn must therefore be edited as a whole — either
evicted completely (``ATOMIC_PURGE``) or kept in shape with its content
shrunk (``TOMBSTONE``) — never split so that one half survives without the
other.
"""

from __future__ import annotations

from context_slim.expiry import atomic_purge, tombstone, tool_call_batches
from context_slim.schemas import Message


def _parallel_call_loop() -> list[Message]:
    """One assistant turn firing three tool calls in parallel."""
    return [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "do three things"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                {"id": "call_b", "type": "function", "function": {"name": "g", "arguments": "{}"}},
                {"id": "call_c", "type": "function", "function": {"name": "h", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "result a"},
        {"role": "tool", "tool_call_id": "call_b", "content": "result b"},
        {"role": "tool", "tool_call_id": "call_c", "content": "result c"},
        {"role": "user", "content": "thanks"},
    ]


def _no_400_would_occur(messages: list[Message]) -> bool:
    """Reproduce the exact rule the live API enforces: every ``tool_calls`` id
    on an assistant message must have a matching ``role: "tool"`` reply
    somewhere in the transcript, and every ``tool`` reply's id must trace back
    to a call that is still present."""
    issued = {
        tc["id"]
        for m in messages
        if m.get("role") == "assistant"
        for tc in (m.get("tool_calls") or [])
    }
    answered = {m["tool_call_id"] for m in messages if m.get("role") == "tool"}
    return issued == answered


def test_parallel_tool_calls_pair_into_one_batch() -> None:
    loop = _parallel_call_loop()
    batches = tool_call_batches(loop)
    assert len(batches) == 1
    batch = batches[0]
    assert batch.assistant_index == 2
    assert set(batch.call_ids) == {"call_a", "call_b", "call_c"}
    assert batch.result_indices == (3, 4, 5)


def test_valid_loop_never_400s_by_construction() -> None:
    assert _no_400_would_occur(_parallel_call_loop())


def test_atomic_purge_removes_the_whole_batch_and_stays_schema_valid() -> None:
    loop = _parallel_call_loop()
    out = atomic_purge(loop, assistant_index=2)

    assert len(out) == len(loop) - 4  # the assistant message + its 3 replies
    assert _no_400_would_occur(out)
    # Nothing that mentions call_a/b/c survives at all — a purge, not a stub.
    assert not any("call_a" in str(m) or "call_b" in str(m) or "call_c" in str(m) for m in out)


def test_atomic_purge_rejects_a_non_tool_calls_index() -> None:
    loop = _parallel_call_loop()
    try:
        atomic_purge(loop, assistant_index=1)  # the plain user message
    except ValueError:
        return
    raise AssertionError("atomic_purge must refuse an index with no tool_calls")


def test_tombstone_keeps_the_pairing_and_shrinks_only_content() -> None:
    loop = _parallel_call_loop()
    tool_index = 3  # answers call_a
    out = list(loop)
    out[tool_index] = tombstone(loop[tool_index])

    assert out[tool_index]["role"] == "tool"
    assert out[tool_index]["tool_call_id"] == "call_a"
    assert out[tool_index]["content"] != loop[tool_index]["content"]
    assert '"compacted": true' in out[tool_index]["content"]
    assert _no_400_would_occur(out)


def test_tombstone_survives_the_anthropic_content_block_shape() -> None:
    message: Message = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "a very long result"},
        ],
    }
    out = tombstone(message)
    block = out["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "toolu_1"
    assert '"compacted": true' in block["content"]


def test_partial_removal_would_have_400d_the_request() -> None:
    """Negative control: proves ``_no_400_would_occur`` actually detects the
    failure mode ATOMIC_PURGE exists to prevent, by building it directly."""
    loop = _parallel_call_loop()
    broken = [m for i, m in enumerate(loop) if i != 3]  # drop only call_a's reply
    assert not _no_400_would_occur(broken)
