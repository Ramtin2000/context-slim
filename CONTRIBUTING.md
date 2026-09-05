# Contributing

Thanks for looking. This is a small, single-maintainer project, so here is what
is actually useful and what the bar is.

## The most valuable thing you can send

**A case where the cost model is wrong.**

This library's entire claim is that it is right about money, and that claim is
checkable against your provider's own usage counters. A report that the
prediction didn't match the bill is not a complaint — it is validation the
project cannot run on someone else's account, and it gets triaged before
anything else.

Use the [cost-model discrepancy][discrepancy] issue template. You do not need a
minimal reproduction; the predicted numbers and the provider's
`cached_tokens` are enough.

[discrepancy]: https://github.com/Ramtin2000/context-slim/issues/new?template=cost-model-discrepancy.md

## Development

```bash
git clone https://github.com/Ramtin2000/context-slim
cd context-slim
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Before opening a PR, run what CI runs:

```bash
pytest -q --cov=context_slim --cov-report=term-missing
pytest -q --no-cov tests/test_performance.py   # timing tests skip under coverage
ruff check .
mypy --strict src/context_slim
```

The performance suite is run twice on purpose. A coverage tracer fires on every
line of Python executed, which is exactly what a microbenchmark measures, so
the timing assertions skip themselves when one is active and CI enforces them
in a separate uninstrumented step. If you are changing anything in the hot path
(`expiry.py`, `pruner.py`, `json_ast.py`, `cache/prefix.py`), run the second
command locally and say what it printed in the PR.

## Standards that aren't negotiable

These aren't style preferences — each one exists because breaking it has
already cost this project a benchmark run or a retracted claim.

- **Zero runtime dependencies.** `dependencies = []` in `pyproject.toml`, and
  CI fails if anything appears under it. Dev and benchmark extras are fine.
- **`mypy --strict` clean**, no new `# type: ignore` without a comment saying
  why.
- **Money is exact integers.** Never a float. There is a test that fails the
  build if a float literal appears in money-handling code.
- **Every rate is sourced.** Entries in `cache/rates.py` carry a
  `verified_on` date and a `price_confidence`. Prices come from the provider's
  own pricing page, never from an aggregator.
- **No number without a source.** If you add a performance or cost claim to
  the README or a docstring, it needs a reproducible command that produces it.
  `bench/` costs real API spend; `benchmarks/` is offline and free.

## What is out of scope

- **Self-hosted KV-cache optimisation.** Prefix-only invalidation is an
  artifact of how hosted APIs bill, not a property of KV caches. If you control
  the cache, partial reuse (CacheBlend, EPIC, LMCache) applies and the
  economics here don't. See "When NOT to use it" in the README.
- **Semantic compression.** Summarising, embedding, or dropping tokens by
  learned importance. That needs a model; this library never loads one. The two
  approaches compose — they don't compete.
- **Beating a provider's own feature on mechanics.** Anthropic's
  `clear_tool_uses` runs server-side and works. The contribution here is the
  economic gate in front of it, not a reimplementation of it.

## Pull requests

Small and focused. Explain *why* in the commit message — this repository's
history is written to be read, and a diff that changes behaviour without saying
what it costs or saves is hard to review.

If a change would make the cost model less accurate or less verifiable, it
loses, however convenient it is.
