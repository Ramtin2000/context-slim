"""Core value types.

Everything monetary in this library is an exact integer. Floating-point
arithmetic on money is banned outright (see ``tests/test_no_floats.py``):
a 0.1c rounding drift is the difference between a ``PLAN`` and a ``REFUSE``
verdict, and the whole product claim is that the arithmetic is right.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any

NANO_PER_USD = 1_000_000_000
"""Money is stored in nanodollars. The cheapest published rate we model is
$0.02 per million tokens = 20 nanodollars per token, so nanodollars keep every
per-token rate a whole number. Micro-dollars would not."""


@dataclass(frozen=True, order=True)
class Money:
    """An exact quantity of USD, stored in integer nanodollars."""

    nano: int

    @classmethod
    def from_usd(cls, usd: str) -> Money:
        """Build from a decimal string, e.g. ``Money.from_usd("0.20")``.

        Deliberately takes ``str`` and not ``float`` — passing a float here
        would smuggle binary rounding error into the ledger at the door.
        """
        return cls(int(Decimal(usd) * NANO_PER_USD))

    @classmethod
    def zero(cls) -> Money:
        return cls(0)

    def __add__(self, other: Money) -> Money:
        return Money(self.nano + other.nano)

    def __sub__(self, other: Money) -> Money:
        return Money(self.nano - other.nano)

    def __mul__(self, k: int) -> Money:
        if not isinstance(k, int):
            raise TypeError("Money may only be scaled by an int; use Fraction rates instead")
        return Money(self.nano * k)

    def __neg__(self) -> Money:
        return Money(-self.nano)

    @property
    def usd(self) -> Decimal:
        """Exact decimal USD. For display and reporting only."""
        return Decimal(self.nano) / NANO_PER_USD

    def __str__(self) -> str:
        return f"${self.usd:.6f}"


Message = dict[str, Any]
"""A provider-shaped chat message. Kept as a plain dict so we never have to
round-trip a user's payload through a lossy model of our own."""


class Decision(str, Enum):
    """The three outcomes of the break-even engine.

    ``REFUSE`` being a first-class outcome — rather than an exception or a
    silent no-op — is Law 1 made visible in the type system.
    """

    PLAN = "PLAN"
    DEFER = "DEFER"
    REFUSE = "REFUSE"


@dataclass(frozen=True)
class Candidate:
    """A proposed edit to the message history, not yet costed.

    ``w_tokens`` is the number of cached tokens that sit *after* the edit point
    and would therefore have to be re-written. ``s_tokens`` is how many tokens
    the edit actually removes. The ratio between them decides everything.
    """

    kind: str
    index: int
    w_tokens: int
    s_tokens: int
    detail: str = ""


@dataclass(frozen=True)
class BreakEven:
    """The arithmetic behind a verdict, kept so every decision can show its work."""

    turns: float | None
    """Turns until the prune pays for itself. ``None`` means never — the edit is
    net-negative at every horizon."""

    cost_now: Money
    saving_per_turn: Money
    net_at_horizon: Money


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    candidate: Candidate
    math: BreakEven
    reason: str


@dataclass(frozen=True)
class PrunePlan:
    """Output of ``plan()``. Pure data — holding one has no side effects."""

    model: str
    horizon: int
    verdicts: Sequence[Verdict]

    @property
    def approved(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.decision is Decision.PLAN]

    @property
    def projected_saving(self) -> Money:
        total = Money.zero()
        for v in self.approved:
            total = total + v.math.net_at_horizon
        return total


@dataclass(frozen=True)
class CostReport:
    """What ``apply()`` actually did, in money rather than tokens."""

    model: str
    horizon: int
    edits_applied: int
    edits_refused: int
    edits_deferred: int
    tokens_removed: int
    cost_now: Money
    saving_per_turn: Money
    net_at_horizon: Money

    def __str__(self) -> str:
        return (
            f"{self.edits_applied} edit(s), {self.tokens_removed:,} tokens removed | "
            f"costs {self.cost_now} now, saves {self.saving_per_turn}/turn | "
            f"net {self.net_at_horizon} over {self.horizon} turns"
        )


@dataclass(frozen=True)
class Diagnostic:
    """A cache pathology found by ``doctor()`` that costs money silently."""

    code: str
    severity: str
    message: str
    est_cost_per_turn: Money = field(default_factory=Money.zero)
