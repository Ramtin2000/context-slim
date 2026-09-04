"""Snapshot/restore must be byte-exact, or it is worse than not checkpointing.

An agent loop outlives its process. Restoring one is only safe if the restored
prompt is *byte-identical* to the one that was live before the restart —
prompt caches are prefix caches, so a single changed byte at position `k`
re-bills everything from `k` to the end at the cache-write rate. A restore
that "looks the same" but renders one number differently costs the full
prefix.

The specific trap this module pins is that ``render_stub`` embeds the token
count of the message it replaced::

    [tool result elided: ~1,235 tokens, tail-first, pays back in 4 turns]

Re-deriving that stub on restore measures the *stub*, not the original, and
writes ``~12 tokens`` instead. Same code, same inputs, different bytes. So a
checkpoint records what was decided and replays it; it never recomputes it.
That rule is the whole design, and these tests exist to keep it true.
"""

from __future__ import annotations

import pytest

from context_slim.checkpoint import Checkpoint
from context_slim.core import CacheAlignedContext
from context_slim.expiry import render_stub
from context_slim.schemas import Message
from tests.conftest import build_openai_loop


def _compacted(loop: list[Message]) -> tuple[CacheAlignedContext, list[Message]]:
    """Run one real plan/apply cycle and hand back the live, compacted state."""
    ctx = CacheAlignedContext(loop)
    plan = ctx.plan(horizon=30)
    applied, _ = ctx.apply(plan)
    return ctx, applied


def test_restore_reproduces_messages_byte_for_byte(openai_loop: list[Message]) -> None:
    # The load-bearing property. Everything else in this file is a corollary.
    ctx, live = _compacted(openai_loop)
    snap = ctx.snapshot(live)

    restored = Checkpoint.restore(snap, openai_loop)

    assert restored == live


def test_snapshot_restore_snapshot_is_a_fixed_point(openai_loop: list[Message]) -> None:
    ctx, live = _compacted(openai_loop)

    first = ctx.snapshot(live)
    restored = Checkpoint.restore(first, openai_loop)
    second = CacheAlignedContext(openai_loop).snapshot(restored)

    assert first == second


def test_restore_replays_the_recorded_stub_not_a_fresh_one(openai_loop: list[Message]) -> None:
    """The regression this module exists for.

    Recomputing a stub from an already-stubbed message shrinks the token count
    written into its own text. Pinned here so nobody "simplifies" restore into
    a second ``apply`` call.
    """
    ctx, live = _compacted(openai_loop)
    snap = ctx.snapshot(live)
    if not snap.stubbed:
        pytest.skip("this loop produced no approved prunes; nothing to replay")

    index, recorded = next(iter(sorted(snap.stubbed.items())))
    re_derived = render_stub(live[index], "tail-first, pays back in 4 turns")["content"]

    # If these ever coincide the trap is gone and this test is free to go too.
    assert recorded != re_derived
    assert Checkpoint.restore(snap, openai_loop)[index]["content"] == recorded


def test_restore_never_shrinks_the_cleared_set(openai_loop: list[Message]) -> None:
    """Confound 2 from METHODS.md, re-introduced through the restore path.

    A policy that un-clears a previously cleared message invalidated an entire
    benchmark run once already. Restoring must only ever be able to grow it.
    """
    ctx, live = _compacted(openai_loop)
    early = ctx.snapshot(live)

    later_live = list(live)
    extra = max(early.stubbed, default=0) + 2
    later_live[extra] = render_stub(later_live[extra], "second pass")
    later = CacheAlignedContext(openai_loop).snapshot(later_live)

    assert set(early.stubbed) < set(later.stubbed)

    # Folding the older, smaller set into the newer one is legal — nothing is lost.
    assert set(later.merge(early).stubbed) == set(later.stubbed)

    # The reverse would silently unclear `extra`. That is confound 2, and it raises.
    with pytest.raises(ValueError, match="cleared set may only grow"):
        early.merge(later)


def test_anchor_survives_fifty_turns_of_checkpointing() -> None:
    """The 50-turn exit criterion: the Anchor Zone is byte-identical at turn 50."""
    loop = build_openai_loop(n_tools=25, body_chars=1_200, system_chars=4_000)
    ctx = CacheAlignedContext(loop)
    anchor_at_start = ctx.anchor

    live = list(loop)
    for turn in range(50):
        working = CacheAlignedContext(live)
        live = Checkpoint.restore(working.snapshot(live), loop)
        assert CacheAlignedContext(live).anchor == anchor_at_start, f"anchor moved at turn {turn}"

    assert live[: ctx.anchor_index + 1] == anchor_at_start


def test_apply_only_ever_changes_content(openai_loop: list[Message]) -> None:
    """The load-bearing assumption behind ``snapshot``, pinned so it can't rot.

    ``snapshot`` detects compaction by diffing ``content`` alone. That is only
    sufficient while ``apply`` restricts itself to ``render_stub``. Wire
    ``ATOMIC_PURGE`` (which drops whole messages) or any non-content mutation
    into ``apply`` and snapshots would silently under-record, restoring a
    conversation that differs from the one that ran. This test fails first.
    """
    _, live = _compacted(openai_loop)

    assert len(live) == len(openai_loop), "apply dropped a message; snapshot cannot see that"
    for before, after in zip(openai_loop, live):
        assert list(before.keys()) == list(after.keys()), "apply changed key order"
        for key in before:
            if key != "content":
                assert before[key] == after[key], f"apply mutated {key!r}, which snapshot ignores"


def test_checkpoint_round_trips_through_json(openai_loop: list[Message]) -> None:
    ctx, live = _compacted(openai_loop)
    snap = ctx.snapshot(live)

    assert Checkpoint.from_json(snap.to_json()) == snap
