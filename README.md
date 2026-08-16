# context-slim

**Pruning your LLM agent's context can cost more than leaving it alone.**

Measured against a live API, n=5 arms per condition, bootstrap 95% CIs:

| strategy | tokens sent | cache hit | cost/arm | vs no pruning |
|---|---|---|---|---|
| don't prune | 928,400 | **92.2%** | **$0.007053** | — |
| prune oldest-first | 735,790 | 75.5% | $0.011239 | **+59.4%** |
| prune newest-first | 735,770 | 81.0% | $0.009376 | **+32.9%** |

**20.7% fewer tokens. 33–59% more money.** All three differences significant.

Prompt caches are *prefix* caches, so an un-pruned loop is append-only — every
turn extends the last and the whole prompt is reusable. Pruning breaks that, and
the re-write costs more than the tokens saved.

Where you cut still matters: newest-first is **16.6% cheaper** than oldest-first
at identical token counts (a 20-token difference). Anthropic's context-editing
API clears oldest-first by default.

```
cache hit rate by turn
                t2    t5    t8   t11   t14   t17   t20
don't prune    95%   96%   96%   96%   77%   97%   97%
oldest-first   95%   96%   63%   77%   76%   77%   75%
newest-first   95%   96%   93%   91%   76%   77%   60%
```

> **Caveat this properly.** 8k prefixes, 20 turns, synthetic loops, one model
> (`gpt-5.6-luna`), one account. Larger contexts over longer horizons are
> untested and may behave differently. Three earlier revisions of this
> experiment produced confident numbers that were artifacts — see
> [`METHODS.md`](METHODS.md) for what went wrong and how it was caught.

Reproduce: `python -m bench.killgate --repeats 5` (~$0.14). Raw usage blocks in
[`bench/results/`](bench/results/).

## Install

> **Not on PyPI yet.** Install from source until v0.1.0 ships:

```bash
pip install git+https://github.com/Ramtin2000/context-slim
```

Zero runtime dependencies. No model, no GPU, no network. Python 3.9+.

## Use

```python
from context_slim import doctor, plan, apply

# 1. Find cache pathologies that cost money silently.
for d in doctor(messages, model="openai/gpt-5.6-luna"):
    print(d.code, d.message)

# 2. Decide what is worth pruning. Pure — no I/O, no mutation.
p = plan(messages, model="openai/gpt-5.6-luna", horizon=30)
for v in p.verdicts:
    print(v.decision.value, v.reason)

# 3. Execute only the approved edits.
messages, report = apply(messages, p)
print(report)
```

`plan()` and `apply()` are separate so that **"don't prune" is an ordinary
outcome you can inspect**, not an exception or a silent no-op:

```
REFUSE  msg 2   structurally unprofitable: W/S = 41.2 means 461.3 turns to pay
                back $0.000412, against a horizon of 20. Prune closer to the tail.
PLAN    msg 14  pays back after 4.1 turns (horizon 20); costs $0.000082 now,
                saves $0.000020/turn, net $0.000318 at horizon
```

## The `doctor` check

Two pathologies cost money with no pruning involved at all:

- **`lookback-overrun`** — Anthropic checks at most 20 positions behind a cache
  breakpoint. Grow past that and the hit is missed silently. No error. Just a bill.
- **`no-breakpoint`** — Anthropic caching is opt-in. Without a breakpoint,
  nothing is cached and every turn pays full price.

`context-slim doctor conversation.json` exits non-zero on an error-severity
finding, so it can sit in CI.

## What this is NOT

- ❌ Not a summarizer, embedder, or tokenizer. No model is ever loaded.
- ❌ Not a competitor to Anthropic's context editing or LangChain's compaction —
  a **cost-aware controller** that decides whether and where to invoke them.
- ❌ Not "fewer tokens at any cost." Sometimes the answer is *don't*, and this
  is the only tool that will tell you so.

## When NOT to use it

Short loops, uncached workloads, and any prefix below the model's cache minimum
(512 tokens on Claude Opus 5, 1024 on GPT-5.6). `doctor` reports all three.

## Status

Pre-release, built in public over 14 days. The cost model is validated against
providers' own `cached_tokens` counters — see `bench/killgate.py`.

## License

MIT
