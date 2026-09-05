# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] — 2026-09-05

### Fixed
- The README — which is also the PyPI project page — said **"Not on PyPI yet"**
  and gave a `git+https` install. PyPI renders the description from the
  uploaded package, so 0.1.0's page kept saying this after the package
  existed. Fixed at the only point it can be: a new release.
- The quickstart under `## Use` never defined `messages`, so pasting it raised
  `NameError` on the first call. It now includes a stand-in conversation, is
  runnable end to end, and the sample output in the README is that code's real
  output rather than an illustration.
- Performance assertions no longer run under a coverage tracer, which inflated
  the very thing they measure (~5.5ms against a 5ms budget the same code clears
  at ~1ms uninstrumented). CI enforces the budget in a separate uninstrumented
  step instead.
- CI now triggers on pushes to `master`. It had been watching `main`, a branch
  this repository has never had, so no commit had ever been checked on push —
  only the weekly schedule ran.

### Added
- `checkpoint.Checkpoint` — epoch checkpointing for agent loops that outlive
  their process. Records the rendered stub text per message index and replays
  it, rather than storing the inputs needed to regenerate it, because
  re-deriving a stub measures the stub instead of the original and silently
  changes the bytes. `restore()` refuses to touch the Anchor Zone; `merge()`
  refuses to drop a previously cleared index.
- `CacheAlignedContext.snapshot()` — records a compaction so it can be replayed
  byte-for-byte.
- A "When NOT to use it" section in the README covering the six conditions the
  measured result depends on, three of which mean you should not install this.

## [Unreleased]

## [0.1.0] — 2026-09-04

First release. Published to PyPI as **`ctx-slim`**; the import is
`context_slim` and the CLI is `context-slim`.

### Added
- **Cost model** (`cache/`) — prefix-cache economics in exact integer money.
  Break-even horizon for editing an already-cached conversation, expressed in
  the token masses actually rewritten (`W`) and removed (`S`), with per-model
  cache read/write multipliers stored as `Fraction`s and `verified_on`-dated.
  Predicts the provider's own `cached_tokens` to a **0.64% median error** after
  calibration, measured over 24 live requests.
- **Break-even policy** (`policy.py`) — every candidate edit resolves to
  `PLAN`, `DEFER`, or `REFUSE` against a derived dollar threshold rather than a
  token floor. "Don't prune" is a first-class, inspectable outcome, and on the
  measured workloads it is usually the correct one.
- **Anchor Zone pinning** (`core.py`, `cache/prefix.py`) — the head of a
  conversation whose cumulative tokens first reach the model's cache-activation
  minimum is never a pruning candidate, at any cost, under any preset.
  `CacheAlignedContext` threads that boundary through every call so a caller
  cannot omit it.
- **Tool-call atomicity** (`expiry.py`) — OpenAI rejects an assistant
  `tool_calls` message whose matching `role: "tool"` reply is missing.
  `ATOMIC_PURGE` evicts a whole batch by `tool_call_id`; `TOMBSTONE` keeps the
  turn's shape and shrinks only its content. Replies are matched by id, not
  position.
- **Structural compaction** (`pruner.py`, `json_ast.py`) — block deduplication
  via `str.split` + `blake2b`, whitespace collapsing, JSON minification in a
  single regex pass, and null-field stripping. No object graph is built.
  Measured **104×** faster on the hashing step than the byte-level rolling hash
  it replaced, which is kept in `bench/` so the comparison stays measured
  rather than asserted.
- **Prune debt ledger** (`ledger.py`) — records unprofitable prunes instead of
  discarding them, and discharges the backlog on the first request where the
  prefix was going to be invalidated anyway (tools changed, TTL lapsed, cache
  missed). Serialisable, because an agent loop outlives a process.
- **`doctor()`** — reports cache pathologies (lookback overruns, missing
  breakpoints) without calling anything. Exits non-zero on error severity, so
  it can fail a build.
- **CLI** (`audit.py`) — `context-slim doctor|plan|apply|simulate`.
- **Provider adapters** (`providers/`) — OpenAI and Anthropic wire formats.
- Benchmarks: an offline FIFO-vs-context-slim comparison (`benchmarks/`) and
  the live-API kill-gate harness (`bench/`).

### Measured
Against a live API, n=5 arms per condition, bootstrap 95% CIs, all differences
significant:

| strategy | tokens sent | cache hit | cost/arm | vs no pruning |
|---|---|---|---|---|
| don't prune | 928,400 | 92.2% | $0.007053 | — |
| prune oldest-first | 735,790 | 75.5% | $0.011239 | **+59.4%** |
| prune newest-first | 735,770 | 81.0% | $0.009376 | **+32.9%** |

Pruning sent **20.7% fewer tokens and cost 33–59% more**. Prune order alone is
worth **16.6%** at identical token counts.

Scope: 8k-token prefixes, 20-turn loops, synthetic conversations, one model
(`gpt-5.6-luna`), one account. Larger contexts over longer horizons are
untested and may behave differently. Three earlier revisions of this experiment
produced confident numbers that were artifacts; see `METHODS.md` for what went
wrong and how each was caught.

[Unreleased]: https://github.com/Ramtin2000/context-slim/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/Ramtin2000/context-slim/releases/tag/v0.1.1
[0.1.0]: https://github.com/Ramtin2000/context-slim/releases/tag/v0.1.0
