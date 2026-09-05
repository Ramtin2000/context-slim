"""The README is executable documentation, so execute it.

v0.1.0 shipped with a quickstart that raised ``NameError`` on its first call —
it used ``messages`` without ever defining it. That block is the first thing a
new reader runs and the first thing a PyPI visitor sees, and nothing in the
suite noticed. This module makes the README's own code a test: if the
quickstart stops running, the build fails before anyone pastes it.
"""

from __future__ import annotations

import pathlib
import re

import pytest

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"


def _read() -> str:
    """Always UTF-8.

    ``Path.read_text()`` uses the platform default, which is cp1252 on Windows
    — and the README is full of em dashes, × and →. CI caught this; a
    Mac-only run never would.
    """
    if not README.exists():  # installed without the repo
        pytest.skip("README.md is not present in this checkout")
    return README.read_text(encoding="utf-8")


def _block(heading: str) -> str:
    """Return the first ```python block under ``heading``."""
    match = re.search(
        rf"^## {re.escape(heading)}\n.*?```python\n(.*?)```",
        _read(),
        re.S | re.M,
    )
    assert match, f"no python block found under '## {heading}' — has the README been restructured?"
    return match.group(1)


def test_quickstart_runs(capsys: pytest.CaptureFixture[str]) -> None:
    exec(compile(_block("Use"), "README.md::Use", "exec"), {"__name__": "__readme__"})

    out = capsys.readouterr().out
    # It should do something visible, not just import cleanly.
    assert out.strip(), "the quickstart printed nothing; it is meant to demonstrate output"
    assert "PLAN" in out or "DEFER" in out or "REFUSE" in out, (
        "the quickstart produced no verdicts — its example conversation is too small "
        "to cross the cache-activation minimum, so it demonstrates nothing"
    )


def test_quickstart_defines_everything_it_uses() -> None:
    """The exact 0.1.0 bug: the block referenced `messages` without defining it."""
    src = _block("Use")
    assert re.search(r"^messages\s*=", src, re.M), (
        "the quickstart uses `messages` but never assigns it — this is the "
        "NameError that shipped in 0.1.0"
    )


def test_install_command_names_the_real_distribution() -> None:
    """`pip install context-slim` installs someone else's package."""
    text = _read()
    assert "pip install ctx-slim" in text
    assert "pip install context-slim" not in text, (
        "the distribution is `ctx-slim`; `context-slim` normalises to the unrelated "
        "`contextslim` package on PyPI"
    )
    assert "Not on PyPI yet" not in text
