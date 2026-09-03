"""The Anchor Zone must never move — not a byte, not an index.

A prompt cache is a *prefix* cache: mutating byte 0 of the cached prefix (the
system prompt, tool schemas, Turn 1 task) invalidates the whole hash chain
and drops ``cached_tokens`` to zero. These tests assert the Anchor Zone —
everything up to and including :attr:`CacheAlignedContext.anchor_index` — is
byte-identical before and after every ``plan()`` / ``apply()`` call, at every
preset, regardless of how aggressively the Compaction Zone gets pruned.
"""

from __future__ import annotations

import copy

from context_slim import CacheAlignedContext
from context_slim.cache.prefix import anchor_boundary, total_tokens
from context_slim.cache.rates import GPT_5_6_LUNA
from context_slim.expiry import candidates
from context_slim.presets import PRESETS
from context_slim.schemas import Message


def test_anchor_boundary_reaches_the_cache_activation_minimum(
    large_openai_loop: list[Message],
) -> None:
    idx = anchor_boundary(large_openai_loop, GPT_5_6_LUNA.min_cacheable_tokens)
    assert total_tokens(large_openai_loop[: idx + 1]) >= GPT_5_6_LUNA.min_cacheable_tokens
    # The boundary is *minimal*: one message earlier must fall short.
    if idx > 0:
        assert total_tokens(large_openai_loop[:idx]) < GPT_5_6_LUNA.min_cacheable_tokens


def test_anchor_boundary_clamps_to_the_last_message_on_a_short_conversation() -> None:
    short: list[Message] = [{"role": "user", "content": "hi"}]
    assert anchor_boundary(short, GPT_5_6_LUNA.min_cacheable_tokens) == 0


def test_candidates_never_propose_an_index_inside_the_anchor(
    large_openai_loop: list[Message],
) -> None:
    idx = anchor_boundary(large_openai_loop, GPT_5_6_LUNA.min_cacheable_tokens)
    for preset in PRESETS.values():
        cands = candidates(
            large_openai_loop,
            order=preset.order,
            keep_recent=preset.keep_recent,
            anchor_index=idx,
        )
        assert all(c.index > idx for c in cands), preset.name


def test_apply_leaves_the_anchor_zone_byte_identical(large_openai_loop: list[Message]) -> None:
    ctx = CacheAlignedContext(large_openai_loop, model="openai/gpt-5.6-luna", preset="aggressive")
    anchor_before = copy.deepcopy(ctx.anchor)

    out, _report = ctx.apply(ctx.plan(horizon=60))

    assert out[: ctx.anchor_index + 1] == anchor_before
    assert ctx.anchor == anchor_before  # the source messages were never mutated either


def test_apply_never_edits_an_index_inside_the_anchor(large_openai_loop: list[Message]) -> None:
    ctx = CacheAlignedContext(large_openai_loop, preset="aggressive")
    p = ctx.plan(horizon=60)
    assert all(v.candidate.index > ctx.anchor_index for v in p.approved)
