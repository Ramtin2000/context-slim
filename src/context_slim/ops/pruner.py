"""Duplicate-block collapse and whitespace normalisation.

Both operations are deliberately built on C-implemented primitives. The
obvious design here is a byte-level rolling hash with content-defined chunk
boundaries, which is what deduplicating storage systems use. In pure Python
that is a per-byte loop - roughly 100k bytecode iterations per 100 KB - and it
costs tens of milliseconds, which does not fit a sub-5ms budget.

``str.split`` and ``hashlib.blake2b`` do the same job at block granularity and
run in C, so the loop in Python is over *blocks* rather than *bytes*. See
``bench/bench_dedupe.py`` for the measured comparison.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

__all__ = ["DedupeStats", "collapse_whitespace", "dedupe_blocks"]

_BLANK_RUN = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)+")
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_FENCE = re.compile(r"```.*?```", re.DOTALL)


@dataclass
class DedupeStats:
    """What a dedupe pass actually removed."""

    blocks_seen: int = 0
    blocks_collapsed: int = 0
    chars_before: int = 0
    chars_after: int = 0
    first_seen: dict[str, int] = field(default_factory=dict)

    @property
    def chars_saved(self) -> int:
        return self.chars_before - self.chars_after


def collapse_whitespace(text: str, *, preserve_code: bool = True) -> str:
    """Normalise whitespace without touching fenced code blocks.

    Indentation inside ``` fences is load-bearing - it is often the code the
    model has to reason about - so those spans are lifted out, the rest is
    normalised, and they are put back byte-identical.
    """
    if not preserve_code:
        return _collapse(text)

    parts: list[str] = []
    last = 0
    fences: list[str] = []
    for m in _FENCE.finditer(text):
        parts.append(_collapse(text[last : m.start()]))
        parts.append(f"\x00{len(fences)}\x00")
        fences.append(m.group(0))
        last = m.end()
    parts.append(_collapse(text[last:]))

    out = "".join(parts)
    for i, fence in enumerate(fences):
        out = out.replace(f"\x00{i}\x00", fence)
    return out


def _collapse(chunk: str) -> str:
    chunk = _ANSI.sub("", chunk)
    chunk = _TRAILING_WS.sub("", chunk)
    return _BLANK_RUN.sub("\n\n", chunk)


def dedupe_blocks(
    texts: list[str], *, min_block_chars: int = 64
) -> tuple[list[str], DedupeStats]:
    """Collapse blocks that have appeared before into a back-reference.

    Blocks are blank-line separated, which is why this is fast: ``str.split``
    does the segmentation in C and the Python loop runs once per block instead
    of once per byte. Blocks shorter than ``min_block_chars`` are left alone -
    collapsing them costs more in reference text than it saves.

    Identity is a truncated blake2b digest. Collisions are possible in
    principle; at 64 bits and the block counts an agent loop produces, they are
    not a practical concern.
    """
    stats = DedupeStats(chars_before=sum(len(t) for t in texts))
    out: list[str] = []
    index = 1

    for text in texts:
        blocks = text.split("\n\n")
        kept: list[str] = []
        for block in blocks:
            stats.blocks_seen += 1
            if len(block) < min_block_chars:
                kept.append(block)
                continue
            digest = hashlib.blake2b(block.encode(), digest_size=8).hexdigest()
            prior = stats.first_seen.get(digest)
            if prior is None:
                stats.first_seen[digest] = index
                index += 1
                kept.append(block)
            else:
                stats.blocks_collapsed += 1
                kept.append(f"[= identical to block #{prior} above]")
        out.append("\n\n".join(kept))

    stats.chars_after = sum(len(t) for t in out)
    return out, stats
