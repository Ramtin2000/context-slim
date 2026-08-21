"""The ledger must never lose a prune, and never discharge for free when it isn't."""

from __future__ import annotations

import pytest

from context_slim._types import BreakEven, Candidate, Decision, Money, Verdict
from context_slim.ledger import DischargeReason, Ledger, detect_discharge_windows


def _verdict(decision: Decision, index: int = 3, s: int = 400) -> Verdict:
    c = Candidate(kind="tool_result", index=index, w_tokens=8000, s_tokens=s)
    return Verdict(decision, c, BreakEven(90.0, Money(1000), Money(10), Money(-500)), "held")


def test_only_defer_verdicts_are_accepted() -> None:
    """REFUSE means net-negative at every horizon; no window makes it worth doing."""
    led = Ledger()
    for bad in (Decision.PLAN, Decision.REFUSE):
        with pytest.raises(ValueError, match="only DEFER"):
            led.defer(_verdict(bad), turn=1)


def test_nothing_discharges_without_a_window() -> None:
    led = Ledger()
    led.defer(_verdict(Decision.DEFER), turn=1)
    assert led.discharge_if_free([]) == []
    assert len(led) == 1


def test_discharge_releases_everything_and_empties() -> None:
    led = Ledger()
    for i in (3, 6, 9):
        led.defer(_verdict(Decision.DEFER, index=i), turn=1)
    released = led.discharge_if_free([DischargeReason.TOOLS_CHANGED])
    assert len(released) == 3
    assert len(led) == 0
    assert led.discharged == 3


def test_same_index_is_not_deferred_twice() -> None:
    led = Ledger()
    led.defer(_verdict(Decision.DEFER, index=3), turn=1)
    led.defer(_verdict(Decision.DEFER, index=3), turn=2)
    assert len(led) == 1


def test_pending_tokens_totals_the_backlog() -> None:
    led = Ledger()
    led.defer(_verdict(Decision.DEFER, index=3, s=400), turn=1)
    led.defer(_verdict(Decision.DEFER, index=6, s=600), turn=1)
    assert led.pending_tokens == 1000


def test_survives_a_process_restart() -> None:
    """An agent loop outlives a process; a forgetful ledger defers forever."""
    led = Ledger()
    led.defer(_verdict(Decision.DEFER), turn=7)
    back = Ledger.from_json(led.to_json())
    assert len(back) == 1
    assert back.entries[0].turn == 7


def test_ttl_window_needs_both_elapsed_and_ttl() -> None:
    assert detect_discharge_windows(seconds_since_last_call=9999) == []
    assert detect_discharge_windows(
        seconds_since_last_call=1801, ttl_seconds=1800
    ) == [DischargeReason.TTL_EXPIRED]


def test_zero_cached_tokens_is_a_cache_miss() -> None:
    assert DischargeReason.CACHE_MISS in detect_discharge_windows(last_cached_tokens=0)
    assert detect_discharge_windows(last_cached_tokens=5000) == []
