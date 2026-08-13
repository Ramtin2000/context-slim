# context-slim

**Prune your LLM agent's context without destroying your prompt cache.**

```
N = 11.5·(W/S) − 12.5
```

That is how many turns a prune takes to pay for itself. `W` is the tokens that
must be re-written because you edited behind them; `S` is the tokens you saved.
Delete 1k tokens from the front of a 10k cached prefix and you need **102 more
turns** before you break even. Your agent has twenty.

Every context pruner reports tokens removed. You pay dollars. On a cached loop
those are not the same number, and sometimes they have opposite signs.

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
