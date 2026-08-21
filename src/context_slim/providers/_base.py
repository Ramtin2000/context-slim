"""One normalised view over two different tool-call wire formats."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .._types import Message

__all__ = ["Adapter", "ToolCall", "adapter_for", "detect"]


@dataclass(frozen=True)
class ToolCall:
    """A tool invocation and its result, located in the message list."""

    call_id: str
    name: str
    result_index: int
    result_text: str


class Adapter(Protocol):
    """What a provider adapter has to be able to do."""

    name: str

    def is_tool_result(self, message: Message) -> bool: ...

    def text_of(self, message: Message) -> str: ...

    def stub(self, message: Message, note: str) -> Message: ...

    def tool_calls(self, messages: Sequence[Message]) -> list[ToolCall]: ...


class OpenAIAdapter:
    """``role: "tool"`` answering ``assistant.tool_calls``."""

    name = "openai"

    def is_tool_result(self, message: Message) -> bool:
        return message.get("role") == "tool"

    def text_of(self, message: Message) -> str:
        content = message.get("content")
        return content if isinstance(content, str) else ""

    def stub(self, message: Message, note: str) -> Message:
        out = dict(message)
        out["content"] = note
        return out

    def tool_calls(self, messages: Sequence[Message]) -> list[ToolCall]:
        names: dict[str, str] = {}
        calls: list[ToolCall] = []
        for i, m in enumerate(messages):
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                names[str(tc.get("id"))] = str(fn.get("name", "?"))
            if m.get("role") == "tool":
                cid = str(m.get("tool_call_id", ""))
                calls.append(
                    ToolCall(cid, names.get(cid, "?"), i, self.text_of(m))
                )
        return calls


class AnthropicAdapter:
    """``tool_result`` content blocks inside a user message."""

    name = "anthropic"

    def _blocks(self, message: Message) -> list[dict[str, Any]]:
        content = message.get("content")
        return [b for b in content if isinstance(b, dict)] if isinstance(content, list) else []

    def is_tool_result(self, message: Message) -> bool:
        return any(b.get("type") == "tool_result" for b in self._blocks(message))

    def text_of(self, message: Message) -> str:
        parts: list[str] = []
        for b in self._blocks(message):
            if b.get("type") != "tool_result":
                continue
            inner = b.get("content")
            if isinstance(inner, str):
                parts.append(inner)
            elif isinstance(inner, list):
                parts.extend(
                    str(x.get("text", "")) for x in inner if isinstance(x, dict)
                )
        return "".join(parts)

    def stub(self, message: Message, note: str) -> Message:
        out = dict(message)
        out["content"] = [
            {**b, "content": note} if b.get("type") == "tool_result" else b
            for b in self._blocks(message)
        ]
        return out

    def tool_calls(self, messages: Sequence[Message]) -> list[ToolCall]:
        names: dict[str, str] = {}
        calls: list[ToolCall] = []
        for i, m in enumerate(messages):
            for b in self._blocks(m):
                if b.get("type") == "tool_use":
                    names[str(b.get("id"))] = str(b.get("name", "?"))
                elif b.get("type") == "tool_result":
                    cid = str(b.get("tool_use_id", ""))
                    calls.append(ToolCall(cid, names.get(cid, "?"), i, self.text_of(m)))
        return calls


_ADAPTERS: dict[str, Adapter] = {"openai": OpenAIAdapter(), "anthropic": AnthropicAdapter()}


def detect(messages: Sequence[Message]) -> str:
    """Infer the wire format from the messages themselves.

    Defaults to OpenAI: it is the shape with no distinguishing marker, so an
    absence of Anthropic content blocks is the evidence.
    """
    for m in messages:
        content = m.get("content")
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") in {"tool_use", "tool_result"}
            for b in content
        ):
            return "anthropic"
    return "openai"


def adapter_for(name_or_messages: str | Sequence[Message]) -> Adapter:
    if isinstance(name_or_messages, str):
        key = name_or_messages.split("/")[0]
        if key not in _ADAPTERS:
            raise KeyError(f"no adapter for {name_or_messages!r}; have {sorted(_ADAPTERS)}")
        return _ADAPTERS[key]
    return _ADAPTERS[detect(name_or_messages)]
