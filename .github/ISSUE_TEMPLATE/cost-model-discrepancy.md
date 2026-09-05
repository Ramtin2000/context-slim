---
name: Cost-model discrepancy
about: The predicted cost or cached-token count didn't match what your provider billed
title: "Cost-model discrepancy: "
labels: cost-model
---

<!--
This is the most useful issue you can open here.

The whole claim of this library is that it is right about money, and that claim
is checkable against your provider's own usage counters. A report that it isn't
is not a complaint — it is the validation the project can't run on its own
account. It gets triaged first.

You do not need a minimal reproduction. Numbers are enough.
-->

## What was predicted

<!-- Output of `plan()`, `simulate()`, or `context-slim plan`. Paste it raw. -->

```

```

## What the provider actually billed

<!--
The counters, not the dashboard total:
  OpenAI Chat Completions - usage.prompt_tokens_details.cached_tokens
  OpenAI Responses        - usage.input_tokens_details.cached_tokens
  Anthropic               - usage.cache_read_input_tokens / cache_creation_input_tokens
-->

```

```

## Setup

- `ctx-slim` version:
- Model (exact string, e.g. `openai/gpt-5.6-luna`):
- Preset, or the arguments passed to `plan()`:
- Approximate prefix size, and turns into the conversation:

## Anything that might explain it

<!--
Optional, and no need to work it out yourself. But these are the usual causes,
and if you already know one applies it saves a round trip:

- more than the cache TTL between calls, so the prefix was cold anyway
- tool definitions or the system prompt changed between calls
- another process sharing the same account and cache namespace
- a provider tier whose rates are marked third-party in `cache/rates.py`
  rather than confirmed at the source
-->
