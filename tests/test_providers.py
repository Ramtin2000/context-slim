"""Both wire formats must reduce to the same view, and stubs must stay valid."""

from __future__ import annotations

import pytest

from context_slim.providers import adapter_for, detect

OPENAI = [
    {"role": "assistant", "content": None,
     "tool_calls": [{"id": "c1", "type": "function",
                     "function": {"name": "ls", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "a.py b.py"},
]

ANTHROPIC = [
    {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "ls"}]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "a.py b.py"}]},
]


def test_detects_each_wire_format() -> None:
    assert detect(OPENAI) == "openai"
    assert detect(ANTHROPIC) == "anthropic"


def test_plain_chat_defaults_to_openai() -> None:
    """OpenAI's shape has no distinguishing marker; absence of blocks is the evidence."""
    assert detect([{"role": "user", "content": "hi"}]) == "openai"


@pytest.mark.parametrize("msgs", [OPENAI, ANTHROPIC])
def test_both_formats_yield_the_same_view(msgs: list[dict[str, object]]) -> None:
    calls = adapter_for(msgs).tool_calls(msgs)
    assert len(calls) == 1
    assert calls[0].name == "ls"
    assert calls[0].result_index == 1
    assert "a.py" in calls[0].result_text


def test_openai_stub_preserves_tool_call_id() -> None:
    """Drop tool_call_id and the API rejects the whole request."""
    stub = adapter_for(OPENAI).stub(OPENAI[1], "[cleared]")
    assert stub["tool_call_id"] == "c1"
    assert stub["content"] == "[cleared]"


def test_anthropic_stub_preserves_block_structure() -> None:
    stub = adapter_for(ANTHROPIC).stub(ANTHROPIC[1], "[cleared]")
    block = stub["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "t1"
    assert block["content"] == "[cleared]"


def test_stubs_never_mutate_the_input() -> None:
    before = dict(OPENAI[1])
    adapter_for(OPENAI).stub(OPENAI[1], "[cleared]")
    assert OPENAI[1] == before


def test_unknown_provider_is_rejected_loudly() -> None:
    with pytest.raises(KeyError, match="no adapter"):
        adapter_for("mistral/large")
