"""Multi-turn synthetic message fixtures shared across the test suite.

Every loop here follows the same shape a real agent produces: a large system
prompt (so the Anchor Zone crosses a model's cache-activation minimum),
alternating assistant tool calls and tool results, and a trailing user turn.
"""

from __future__ import annotations

import pytest

from context_slim.schemas import Message


def build_openai_loop(
    n_tools: int = 8,
    body_chars: int = 4_000,
    system_chars: int = 3_000,
) -> list[Message]:
    """An OpenAI-shaped agent loop: ``assistant.tool_calls`` / ``role: "tool"``."""
    msgs: list[Message] = [
        {
            "role": "system",
            "content": "x" * system_chars,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    for i in range(n_tools):
        call_id = f"call_{i}"
        msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "lookup", "arguments": f'{{"i": {i}}}'},
                    }
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": call_id, "content": "y" * body_chars})
    msgs.append({"role": "user", "content": "and then?"})
    return msgs


@pytest.fixture
def openai_loop() -> list[Message]:
    """A moderate 8-tool-call loop, ~35k tokens — big enough to exceed every
    modelled cache-activation minimum."""
    return build_openai_loop()


@pytest.fixture
def large_openai_loop() -> list[Message]:
    """A ~24,000-token, 92-message loop, sized for the sub-5ms performance budget."""
    return build_openai_loop(n_tools=45, body_chars=3_200, system_chars=4_000)
