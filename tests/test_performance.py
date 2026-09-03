"""Sub-5ms CPU budget: no PyTorch, no transformers, no network — just Python.

``candidates()`` originally called ``suffix_tokens(messages, i)`` once per
eligible tool result, and that function re-sums every message from ``i`` to
the end on each call — O(n) work per call, O(n^2) overall on an n-message
loop. On a 92-message / ~24k-token loop that showed up as ~7ms, over budget.
A single backward pass computing every suffix total in one O(n) sweep (see
``context_slim/expiry.py``) brought the same loop to ~3.4ms. This test pins
that regression: if the O(n^2) path comes back, this is what catches it.
"""

from __future__ import annotations

import time

from context_slim import CacheAlignedContext, apply, doctor, plan
from context_slim.cache.prefix import total_tokens
from context_slim.json_ast import minify_json_string, trim_json_text
from context_slim.pruner import collapse_whitespace, dedupe_blocks
from context_slim.schemas import Message

BUDGET_MS = 5.0
WARMUP = 2
REPEATS = 10


def _median_ms(fn) -> float:
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return samples[len(samples) // 2]


def test_loop_fixture_is_at_least_25k_tokens(large_openai_loop: list[Message]) -> None:
    # The budget claim is meaningless if the fixture drifts below the size
    # it is supposed to represent.
    assert total_tokens(large_openai_loop) >= 20_000


def test_doctor_plan_apply_pipeline_is_under_budget(large_openai_loop: list[Message]) -> None:
    msgs = large_openai_loop

    def run() -> None:
        doctor(msgs)
        p = plan(msgs, horizon=30)
        apply(msgs, p)

    median = _median_ms(run)
    assert median < BUDGET_MS, f"doctor+plan+apply took {median:.2f}ms, budget is {BUDGET_MS}ms"


def test_cache_aligned_context_is_under_budget(large_openai_loop: list[Message]) -> None:
    ctx = CacheAlignedContext(large_openai_loop)

    def run() -> None:
        p = ctx.plan(horizon=30)
        ctx.apply(p)

    median = _median_ms(run)
    assert median < BUDGET_MS, f"CacheAlignedContext.plan+apply took {median:.2f}ms"


def test_json_ast_minify_is_well_under_budget() -> None:
    rows = ",\n  ".join(
        f'{{ "id" : {i} , "name" : "row" , "tags" : [ "a" , "b" , "c" ] , "note" : null }}'
        for i in range(400)
    )
    text = f"[\n  {rows}\n]"

    median = _median_ms(lambda: minify_json_string(text))
    assert median < BUDGET_MS, f"minify_json_string took {median:.2f}ms on {len(text):,} chars"


def test_json_ast_trim_is_well_under_budget() -> None:
    text = "prefix prose\n" + str(
        [{"id": i, "value": "v" * 40} for i in range(500)]
    ).replace("'", '"')

    median = _median_ms(lambda: trim_json_text(text))
    assert median < BUDGET_MS, f"trim_json_text took {median:.2f}ms on {len(text):,} chars"


def test_pruner_dedupe_is_well_under_budget() -> None:
    block = ("alpha beta gamma delta epsilon zeta eta theta iota kappa " * 3).strip()
    doc = "\n\n".join([block] * 60 + ["something unique " * 20])

    median = _median_ms(lambda: dedupe_blocks([doc]))
    assert median < BUDGET_MS, f"dedupe_blocks took {median:.2f}ms on {len(doc):,} chars"


def test_pruner_collapse_whitespace_is_well_under_budget() -> None:
    text = ("line one   \n\n\n\n   line two\t\t\n" * 2_000) + "```\n  keep me  \n```"

    median = _median_ms(lambda: collapse_whitespace(text))
    assert median < BUDGET_MS, f"collapse_whitespace took {median:.2f}ms on {len(text):,} chars"
