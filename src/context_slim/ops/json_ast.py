"""Schema-preserving JSON trimming.

Tool results are overwhelmingly JSON, and the *schema* carries most of the
meaning. A 400-element array of identically-shaped objects teaches the model
nothing after the third element, but deleting the array outright tells it the
field no longer exists.

So every key path in the input survives into the output, types are preserved,
and every elision is a self-describing string. The output is still valid JSON
the model can parse.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any

__all__ = ["detect_json_spans", "trim_json", "trim_json_text"]

_B64ISH = re.compile(r"^[A-Za-z0-9+/=_-]{256,}$")
_OPENERS = {"{": "}", "[": "]"}


def _entropy(s: str) -> float:
    """Shannon entropy per character. Blobs sit high; prose and paths sit low."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _is_blob(value: str) -> bool:
    """A long, high-entropy, base64-shaped run is payload, not information."""
    return len(value) >= 256 and bool(_B64ISH.match(value)) and _entropy(value) > 4.5


def _elide_middle(value: str, budget_chars: int) -> str:
    """Truncate keeping both ends.

    Both ends matter: the head of a path identifies it, the tail of a stack
    trace is where the error actually is. Cutting only the tail loses half of
    what the model needs.
    """
    if len(value) <= budget_chars or budget_chars < 24:
        return value
    keep = (budget_chars - 20) // 2
    return f"{value[:keep]}…({len(value) - 2 * keep} chars elided)…{value[-keep:]}"


def _shape(obj: Any) -> str:
    """A cheap structural signature, used to tell homogeneous arrays apart."""
    if isinstance(obj, dict):
        return "{" + ",".join(sorted(obj)) + "}"
    if isinstance(obj, list):
        return "[]"
    return type(obj).__name__


def trim_json(
    obj: Any,
    *,
    max_chars: int = 2_000,
    max_depth: int = 6,
    array_head: int = 2,
    array_tail: int = 1,
) -> Any:
    """Trim values while preserving every key path, type and shape.

    ``max_chars`` bounds individual string values rather than the document:
    bounding the document would require a global allocator whose behaviour
    depends on traversal order, which makes the output unstable under
    reordering. Per-value bounds are stable and explain themselves.
    """
    return _walk(obj, max_chars, max_depth, array_head, array_tail, 0)


def _walk(obj: Any, max_chars: int, max_depth: int, head: int, tail: int, depth: int) -> Any:
    if depth >= max_depth:
        if isinstance(obj, dict):
            return f"…({len(obj)} keys at depth {depth})"
        if isinstance(obj, list):
            return f"…({len(obj)} items at depth {depth})"
        return obj

    if isinstance(obj, dict):
        return {
            k: _walk(v, max_chars, max_depth, head, tail, depth + 1) for k, v in obj.items()
        }

    if isinstance(obj, list):
        return _walk_list(obj, max_chars, max_depth, head, tail, depth)

    if isinstance(obj, str):
        if _is_blob(obj):
            return f"<blob:{len(obj)} chars>"
        return _elide_middle(obj, max_chars)

    return obj


def _walk_list(
    obj: list[Any], max_chars: int, max_depth: int, head: int, tail: int, depth: int
) -> list[Any]:
    if len(obj) <= head + tail + 1:
        return [_walk(v, max_chars, max_depth, head, tail, depth + 1) for v in obj]

    shapes = {_shape(v) for v in obj}
    kept_head = [_walk(v, max_chars, max_depth, head, tail, depth + 1) for v in obj[:head]]
    kept_tail = (
        [_walk(v, max_chars, max_depth, head, tail, depth + 1) for v in obj[-tail:]]
        if tail
        else []
    )
    elided = len(obj) - head - tail

    if len(shapes) == 1:
        marker: Any = f"…({elided} more items, same shape)"
    else:
        # Heterogeneous: keep one example of every shape not already shown, so
        # the model never sees a variant disappear entirely.
        seen = {_shape(v) for v in obj[:head]} | {_shape(v) for v in obj[-tail:]}
        extras = []
        for v in obj[head : len(obj) - tail]:
            if _shape(v) not in seen:
                seen.add(_shape(v))
                extras.append(_walk(v, max_chars, max_depth, head, tail, depth + 1))
        marker = f"…({elided} more items, {len(shapes)} shapes)"
        return [*kept_head, *extras, marker, *kept_tail]

    return [*kept_head, marker, *kept_tail]


def detect_json_spans(text: str) -> list[tuple[int, int]]:
    """Find balanced JSON spans without paying parse cost on prose.

    A bracket-balance scan is cheap; ``json.loads`` on a 100 KB log that
    contains no JSON is not. Only balanced candidates are handed to the parser.
    """
    spans: list[tuple[int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch not in _OPENERS:
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        for j in range(i, n):
            c = text[j]
            if esc:
                esc = False
                continue
            if c == "\\" and in_str:
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c in _OPENERS:
                depth += 1
            elif c in ("}", "]"):
                depth -= 1
                if depth == 0:
                    spans.append((i, j + 1))
                    i = j
                    break
        i += 1
    return spans


def trim_json_text(text: str, **kwargs: Any) -> str:
    """Trim JSON found inside a larger text, leaving surrounding prose alone."""
    spans = detect_json_spans(text)
    if not spans:
        return text
    out: list[str] = []
    last = 0
    for start, end in spans:
        if start < last:
            continue
        try:
            parsed = json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
        out.append(text[last:start])
        out.append(json.dumps(trim_json(parsed, **kwargs), ensure_ascii=False))
        last = end
    out.append(text[last:])
    return "".join(out)
