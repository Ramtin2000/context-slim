"""Tests for the dedupe / whitespace ops."""

from __future__ import annotations

from context_slim.ops.pruner import collapse_whitespace, dedupe_blocks

PARA = ("alpha beta gamma delta epsilon zeta eta theta iota kappa " * 3).strip()


def test_repeated_block_collapses_to_backreference() -> None:
    doc = f"{PARA}\n\nsomething else entirely that is also quite long indeed ok\n\n{PARA}"
    (out,), stats = dedupe_blocks([doc])
    assert stats.blocks_collapsed == 1
    assert "identical to block #1 above" in out
    assert out.count(PARA) == 1


def test_dedupe_works_across_texts() -> None:
    _, stats = dedupe_blocks([PARA, PARA, PARA])
    assert stats.blocks_collapsed == 2
    assert stats.chars_saved > 0


def test_short_blocks_are_left_alone() -> None:
    """Collapsing a short block costs more in reference text than it saves."""
    _, stats = dedupe_blocks(["hi", "hi", "hi"], min_block_chars=64)
    assert stats.blocks_collapsed == 0


def test_no_false_dedupe_on_distinct_blocks() -> None:
    docs = ["\n\n".join(f"{PARA} variant {i} {j}" for j in range(20)) for i in range(5)]
    _, stats = dedupe_blocks(docs)
    assert stats.blocks_collapsed == 0


def test_dedupe_never_grows_a_document() -> None:
    _, stats = dedupe_blocks([PARA, PARA])
    assert stats.chars_after <= stats.chars_before


def test_whitespace_collapse_is_idempotent() -> None:
    messy = "a\n\n\n\n\nb   \n\n\n\nc\t\t\n"
    once = collapse_whitespace(messy)
    assert collapse_whitespace(once) == once


def test_fenced_code_survives_byte_identical() -> None:
    """Indentation inside a fence is the code the model has to reason about."""
    code = "```python\ndef f():\n        return 1\n\n\n\n    # deep blanks kept\n```"
    doc = f"prose   \n\n\n\n{code}\n\n\n\nmore prose"
    out = collapse_whitespace(doc)
    assert code in out
    # Substitute a marker rather than "" - deleting the fence would splice the
    # blank runs either side of it into a false 4-newline sequence.
    assert "\n\n\n\n" not in out.replace(code, "FENCE")


def test_ansi_escapes_stripped() -> None:
    assert collapse_whitespace("\x1b[31mred\x1b[0m text") == "red text"
