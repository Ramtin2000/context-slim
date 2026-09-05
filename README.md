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

## Why the dedupe is built the way it is

The textbook approach to block deduplication is a byte-level rolling hash with
content-defined boundaries. In pure Python that is one interpreter iteration per
byte, and it does not fit a sub-5ms budget. `str.split` plus `hashlib.blake2b`
does the same job at block granularity with both halves running in C.

Measured on ~93 KB (`python -m bench.bench_dedupe`):

| | time |
|---|---|
| rejected: 64-byte rolling hash (per-byte Python loop) | 25.26 ms |
| shipped: `str.split` + `blake2b` (per-block loop) | **0.24 ms** |
| shipped: full `dedupe_blocks` pass | 0.54 ms |
| shipped: `collapse_whitespace` | 1.06 ms |

**104× on the hashing step.** The rejected implementation is kept in
`bench/bench_dedupe.py` so the comparison is measured rather than asserted.

## The cost model is checkable

Most token accounting asks you to trust it. This one predicts how much of a
request the API will report as cached, *before* the call, then diffs against
`usage.prompt_tokens_details`.

Measured over 24 live requests (`python -m bench.validate`, ~$0.01):

| | raw | calibrated |
|---|---|---|
| prompt-token error (median) | 26.11% | **0.64%** |
| cached-token error (median) | 25.80% | **0.84%** |

The raw 26% was a single wrong constant, not a broken model — the
predicted/actual ratio had a spread of 0.9%, so dividing it out left a max
residual of 1.78%. The estimator is calibrated by that constant in
`cache/prefix.py`, with both caveats stated there: it is fit to one tokenizer
family, and it sits inside the pruning policy's own budget, so it cannot be
recalibrated and replayed against an old run.

No dollar figure in this repo comes from the estimator. Those all read the
provider's usage counters.

## Install

```bash
pip install ctx-slim
```

Zero runtime dependencies. No model, no GPU, no network. Python 3.9+.

> The distribution is `ctx-slim`; the import is `context_slim` and the CLI is
> `context-slim`. PyPI normalises `context-slim` to `contextslim`, which is an
> unrelated package that got there first.

## Use

Copy-pasteable as-is. `messages` is whatever you already send the provider —
OpenAI or Anthropic wire format, no conversion:

```python
from context_slim import doctor, plan, apply

# Your real conversation goes here. This stand-in is sized to cross the
# 1,024-token cache minimum so the example actually has something to decide.
messages = [{"role": "system", "content": "You are a coding agent. " + "x" * 4000}]
for i in range(3):
    messages += [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": "read_file", "arguments": '{"path": "main.py"}'}}]},
        {"role": "tool", "tool_call_id": f"call_{i}", "content": "y" * 6000},
    ]
messages.append({"role": "user", "content": "and then?"})

# 1. Find cache pathologies that cost money silently.
for d in doctor(messages, model="openai/gpt-5.6-luna"):
    print(d.code, d.message)

# 2. Decide what is worth pruning. Pure — no I/O, no mutation.
p = plan(messages, model="openai/gpt-5.6-luna", horizon=30)
for v in p.verdicts:
    print(f"{v.decision.value:7} msg {v.candidate.index:<3} {v.reason}")

# 3. Execute only the approved edits.
messages, report = apply(messages, p)
print(report)
```

That prints, verbatim:

```
PLAN    msg 4   pays back after 10.7 turns (horizon 30); costs $0.000204 now,
                saves $0.000019/turn, net $0.000366 at horizon
DEFER   msg 2   pays back in 22.5 turns, over this preset's 12-turn limit —
                held for a cheaper moment
1 edit(s), 951 tokens removed | costs $0.000204 now, saves $0.000019/turn
```

`plan()` and `apply()` are separate so that **"don't prune" is an ordinary
outcome you can inspect**, not an exception or a silent no-op. Drop the same
conversation to `horizon=5` and both edits stop being worth it:

```
DEFER   msg 4   needs 10.7 turns to pay back but only 5 remain; deferring
                until the cache is invalidated anyway (W/S = 2.0)
REFUSE  msg 2   structurally unprofitable: W/S = 3.0 means 22.5 turns to pay
                back $0.000427, against a horizon of 5. Prune closer to the tail.
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

The result above holds under specific conditions. Outside them this library is
either wrong or unnecessary, and it is cheaper for both of us if you find that
out on this page.

- **You aren't hitting a cache at all.** Below the activation minimum — 1,024
  tokens on GPT-5.6, 512 on Claude Opus 5 — or a prefix that changes every
  request. With no cache there is no write premium to eat, so pruning just
  saves money. Prune freely; you don't need this. `doctor` reports it.

- **Your model has no cache-write premium.** Only `gpt-5.6-sol/terra/luna`
  charge one. On gpt-5, 5.1–5.5, 4.1, 4o and o3 the break-even collapses to
  `N = (W−S)/S`, and pruning wins far more often than the headline suggests.

- **Short loops.** A prune pays back over N turns. If the conversation ends
  before N, it is a pure loss. `plan()` returns `REFUSE` here — that is the
  tool working, not failing.

- **You self-host.** Prefix-only invalidation is a *billing* artifact of hosted
  APIs, not physics. Run vLLM or SGLang yourself and partial KV reuse
  (CacheBlend, EPIC, LMCache blending) changes the economics entirely — you are
  buying GPU time, not tokens with a write multiplier. Explicitly out of scope.

- **You are pruning for accuracy, not cost.** Long contexts degrade quality
  independently of price, and Anthropic measured a 29% agent improvement from
  context editing alone. That is a real reason to prune that none of the maths
  here models. If quality is why you are cutting, cut.

- **You are far past the measured regime.** Everything above was measured at
  8k-token prefixes over 20 turns, one model, one account. At 100k tokens over
  100 turns the sign could differ. Nobody has checked, including me.

**"So the conclusion is don't prune — why ship a library?"**

Because *sometimes* is the whole point, and the boundary is a dollar figure
that moves with your prefix size, your edit size, your horizon and your
provider's rates. This tells you which side of it you are on and shows the
arithmetic, so you can disagree with it. If it returns `REFUSE` on every
conversation you have, you got your answer for free and should uninstall it.

## Repository layout

```
src/context_slim/
  core.py       CacheAlignedContext — pins the Anchor Zone, orchestrates plan/apply
  expiry.py     candidate generation, tail-first ordering, ATOMIC_PURGE / TOMBSTONE
  pruner.py     block dedupe and whitespace collapse (Layer 4 text ops)
  json_ast.py   schema-preserving trim, and dependency-free JSON minification
  schemas.py    frozen dataclasses — Money is exact-integer, never float
  audit.py      CLI: `context-slim doctor|plan|apply|simulate`
  policy.py     the break-even decision engine (Law 1 lives here)
  ledger.py     defers unprofitable prunes until the cache breaks anyway
  cache/        the cost model, rate tables, and prefix diagnostics
  providers/    OpenAI / Anthropic wire-shape adapters
bench/          the real benchmark: live API calls, salted cache namespaces,
                bootstrap CIs. Costs real money to run — see METHODS.md for
                what it took to make it trustworthy.
benchmarks/     an offline, zero-cost illustration of the same idea, using
                the same cost primitives against a synthetic trace. Useful to
                see the shape of the argument before spending anything
                confirming it — not a substitute for bench/.
```

`CacheAlignedContext` (`core.py`) is an optional stateful wrapper around the
same four functions exported at the package root — it computes the Anchor
Zone boundary once and threads it through every call, so a candidate can
never be proposed inside the immutable prefix regardless of preset:

```python
from context_slim import CacheAlignedContext

ctx = CacheAlignedContext(messages, model="openai/gpt-5.6-luna", preset="balanced")
ctx.anchor              # the immutable prefix — system prompt, tools, Turn 1
ctx.compaction_zone      # everything eligible for pruning
messages, report = ctx.apply(ctx.plan(horizon=30))
```

## Status

Pre-release, built in public over 14 days. The cost model is validated against
providers' own `cached_tokens` counters — see `bench/killgate.py`.

## License

MIT
