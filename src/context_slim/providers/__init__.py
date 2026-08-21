"""Provider wire-shape adapters.

The cost model is provider-agnostic; the *message shapes* are not. OpenAI puts
tool calls on ``assistant.tool_calls`` and answers them with ``role: "tool"``;
Anthropic uses ``tool_use`` / ``tool_result`` content blocks. Both are reduced
to the same internal view so nothing downstream has to know which is which.
"""

from __future__ import annotations

from ._base import Adapter, ToolCall, adapter_for, detect

__all__ = ["Adapter", "ToolCall", "adapter_for", "detect"]
