"""The prune debt ledger.

A prune that is unprofitable *now* is not unprofitable *forever*. Its cost is
almost entirely the cache re-write it forces, and there are moments when that
re-write is already being paid for other reasons: a tool definition changed, a
breakpoint moved, the TTL lapsed, or the last request simply missed.

At those moments W is already sunk, so a deferred prune costs approximately
nothing. The ledger records unprofitable prunes instead of discarding them and
discharges the whole backlog in the first request where the cache was going to
break anyway.

This is what turns the worst case (a head prune needing ~100 turns to amortise)
into the best case (marginal cost near zero) by timing alone.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from ._types import Candidate, Decision, Verdict

__all__ = ["DischargeReason", "Ledger", "LedgerEntry"]


class DischargeReason(str, Enum):
    """Why the prefix was going to be invalidated regardless of us.

    Each of these means the user is already paying the cache-write cost, so a
    deferred edit rides along for free.
    """

    TOOLS_CHANGED = "tools_changed"
    """Tool definitions changed — invalidates the whole prefix by the documented
    tools -> system -> messages hierarchy."""

    SYSTEM_CHANGED = "system_changed"
    BREAKPOINT_MOVED = "breakpoint_moved"
    TTL_EXPIRED = "ttl_expired"
    CACHE_MISS = "cache_miss"
    """The previous response reported no cached tokens, so the prefix is cold."""


@dataclass(frozen=True)
class LedgerEntry:
    """A prune that was correct in principle but not yet worth its re-write."""

    candidate: Candidate
    reason: str
    turn: int

    def to_dict(self) -> dict[str, Any]:
        return {"candidate": asdict(self.candidate), "reason": self.reason, "turn": self.turn}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LedgerEntry:
        return cls(
            candidate=Candidate(**raw["candidate"]),
            reason=raw["reason"],
            turn=int(raw["turn"]),
        )


@dataclass
class Ledger:
    """Accumulates deferred prunes and discharges them when W is already sunk.

    Serialisable, because an agent loop outlives a process and a ledger that
    forgets on restart would defer the same edit forever.
    """

    entries: list[LedgerEntry] = field(default_factory=list)
    discharged: int = 0

    def defer(self, verdict: Verdict, turn: int) -> None:
        """Record a DEFER verdict. PLAN and REFUSE are not the ledger's business.

        REFUSE in particular must never be deferred: it means the edit is
        net-negative at *every* horizon, so no discharge window makes it worth
        doing.
        """
        if verdict.decision is not Decision.DEFER:
            raise ValueError(
                f"only DEFER verdicts belong in the ledger, got {verdict.decision.value}"
            )
        if any(e.candidate.index == verdict.candidate.index for e in self.entries):
            return
        self.entries.append(
            LedgerEntry(candidate=verdict.candidate, reason=verdict.reason, turn=turn)
        )

    def defer_all(self, verdicts: Iterable[Verdict], turn: int) -> None:
        for v in verdicts:
            if v.decision is Decision.DEFER:
                self.defer(v, turn)

    def discharge_if_free(self, reasons: Sequence[DischargeReason]) -> list[Candidate]:
        """Release the backlog if the prefix is being invalidated anyway.

        Returns the candidates to apply and empties the ledger. With no
        discharge reason this returns nothing and keeps the backlog — the whole
        point is to *wait*.
        """
        if not reasons or not self.entries:
            return []
        released = [e.candidate for e in self.entries]
        self.entries = []
        self.discharged += len(released)
        return released

    @property
    def pending_tokens(self) -> int:
        """Tokens the ledger would remove if discharged right now."""
        return sum(e.candidate.s_tokens for e in self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def to_json(self) -> str:
        return json.dumps(
            {"entries": [e.to_dict() for e in self.entries], "discharged": self.discharged},
            indent=2,
        )

    @classmethod
    def from_json(cls, raw: str) -> Ledger:
        data = json.loads(raw)
        return cls(
            entries=[LedgerEntry.from_dict(e) for e in data.get("entries", [])],
            discharged=int(data.get("discharged", 0)),
        )


def detect_discharge_windows(
    *,
    tools_changed: bool = False,
    system_changed: bool = False,
    breakpoint_moved: bool = False,
    seconds_since_last_call: float | None = None,
    ttl_seconds: int | None = None,
    last_cached_tokens: int | None = None,
) -> list[DischargeReason]:
    """Work out whether the prefix is already being invalidated this turn."""
    reasons: list[DischargeReason] = []
    if tools_changed:
        reasons.append(DischargeReason.TOOLS_CHANGED)
    if system_changed:
        reasons.append(DischargeReason.SYSTEM_CHANGED)
    if breakpoint_moved:
        reasons.append(DischargeReason.BREAKPOINT_MOVED)
    if (
        seconds_since_last_call is not None
        and ttl_seconds is not None
        and seconds_since_last_call > ttl_seconds
    ):
        reasons.append(DischargeReason.TTL_EXPIRED)
    if last_cached_tokens == 0:
        reasons.append(DischargeReason.CACHE_MISS)
    return reasons
