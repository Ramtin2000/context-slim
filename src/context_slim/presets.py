"""Named configurations for the common cases.

A preset is a stance on one question: how much cache damage is an acceptable
price for a smaller context? The measured answer is usually "very little" — on
a 20-turn loop, pruning cost 33-59% more than leaving the context alone — so
the defaults here lean conservative and ``CACHE_PRESERVING`` exists to make
"almost never prune" a one-word choice.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "AGGRESSIVE",
    "BALANCED",
    "CACHE_PRESERVING",
    "PRESETS",
    "Preset",
    "get",
]


@dataclass(frozen=True)
class Preset:
    """A named bundle of pruning parameters."""

    name: str
    order: str
    """``tail_first`` or ``oldest_first``. Measured at 16.6% apart on identical
    token counts, so this is the highest-leverage field here."""

    keep_recent: int
    """Recent tool results never touched. They are the cheapest to keep cached
    and the most likely to still be needed."""

    horizon: int
    """Expected remaining turns. Shorter horizons refuse more, because there
    are fewer turns left over which a re-write can amortise."""

    max_payback_turns: float
    """Refuse anything slower to pay back than this, regardless of horizon."""

    description: str


BALANCED = Preset(
    name="balanced",
    order="tail_first",
    keep_recent=2,
    horizon=20,
    max_payback_turns=12.0,
    description="Default. Tail-first, refuses prunes that need more than ~12 turns.",
)

AGGRESSIVE = Preset(
    name="aggressive",
    order="tail_first",
    keep_recent=1,
    horizon=40,
    max_payback_turns=40.0,
    description=(
        "For genuinely long loops where a re-write has many turns to amortise. "
        "Still tail-first: oldest-first is dominated at every horizon we measured."
    ),
)

CACHE_PRESERVING = Preset(
    name="cache-preserving",
    order="tail_first",
    keep_recent=4,
    horizon=10,
    max_payback_turns=4.0,
    description=(
        "Prune only when it pays back almost immediately. Closest to the "
        "measured optimum, which was to prune very little."
    ),
)

PRESETS: dict[str, Preset] = {p.name: p for p in (BALANCED, AGGRESSIVE, CACHE_PRESERVING)}


def get(name: str) -> Preset:
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown preset {name!r}; have {sorted(PRESETS)}") from None
