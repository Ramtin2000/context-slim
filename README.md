# context-slim

**Prune your agent's context from the end that doesn't shred your prompt cache.**

Measured on a live API: removing **identical token counts** from a 20-turn agent
loop is **18.7% cheaper** when you cut from the newest end instead of the oldest.

| | tokens sent | cache hit | cost |
|---|---|---|---|
| prune oldest-first | 480,033 | 95.1% | $0.015001 |
| prune newest-first | 480,033 | **97.6%** | **$0.012202** |

Same tokens removed. Same work done. The only variable is which end you cut
from — and prefix caches invalidate *forward*, so cutting at the front throws
away everything behind it.

The mechanism is visible as the loop deepens. Cache hit rate under oldest-first:

```
turn  7   80%
turn  9   76%
turn 11   72%
turn 13   69%
turn 15   66%
turn 17   63%
```

All six inside one 60-second window, so this is eviction, not TTL expiry.

Raw usage blocks: [`bench/results/`](bench/results/). Reproduce with
`python -m bench.killgate` (~$0.04).

## The arithmetic

```
N = 11.5·(W/S) − 12.5
```

Turns before a prune pays for itself. `W` is the tokens that must be
re-written because you edited behind them; `S` is the tokens you saved. Prompt
caches are *prefix* caches, so editing at any point invalidates everything after
it. Delete 1k tokens from the front of a 10k cached prefix and you need **102
more turns** to break even. Your agent has twenty.

None of this is a new observation — Anthropic documents the tradeoff,
`clear_at_least` exists to blunt it, and Claude Code already moved its
compaction to tail-first. What was missing is the number, which is why this repo
leads with a measurement rather than an argument.

### What we tested and got wrong

We set out to show pruning costs more than it saves. **It doesn't** — at 8k
prefixes over 20 turns both pruning strategies beat not pruning. That claim is
withdrawn. The result that survived is about *where* you cut, not *whether*.
Larger prefixes over longer horizons may cross over; we haven't tested that and
don't claim it.

## Install

```bash
pip install context-slim
```

Zero runtime dependencies. No model, no GPU, no network.

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
