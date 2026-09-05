# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | ✅ |

Pre-1.0. Fixes land on the latest release; there are no backports.

## Reporting a vulnerability

Report privately via [GitHub Security Advisories][advisory] — that opens a
channel only you and the maintainer can see. Please don't open a public issue
for anything exploitable.

[advisory]: https://github.com/Ramtin2000/context-slim/security/advisories/new

Single maintainer, so: acknowledgement within a week, and I'll tell you plainly
if I don't have a fix rather than going quiet.

## What the threat model actually is

This library has **no network access, no subprocess execution, no deserialization
of untrusted formats, and no runtime dependencies.** It transforms a list of
dicts in memory. That removes most of the usual surface, so it's worth being
specific about what remains.

**In scope:**

- **Content leaking across the Anchor Zone boundary.** The Anchor Zone is
  supposed to be byte-stable. Anything that mutates it — or that lets a
  checkpoint restore rewrite it — is a real bug and possibly a
  cache-poisoning vector on a shared cache namespace.
- **Tool-call pairing violations.** Producing an assistant `tool_calls`
  message without its matching `role: "tool"` reply makes the provider reject
  the request. A crafted conversation that induces this is a denial-of-service
  on the caller's agent loop.
- **Checkpoint deserialization.** `Checkpoint.from_json` parses JSON you may
  have stored elsewhere. It validates the schema version and refuses indices
  inside the Anchor Zone, but treat checkpoints as trusted input — they encode
  decisions about your own conversation.
- **Catastrophic backtracking.** `json_ast` uses regular expressions on
  arbitrary tool output. A pattern that goes quadratic on hostile input is in
  scope; there is a performance suite pinning the budget.
- **Unbounded memory** on adversarial input.

**Out of scope:**

- The *contents* of your conversations. This library never transmits anything;
  what you send to a provider is between you and them.
- Prompt injection through tool results. Compaction preserves whatever the
  content says. Deciding whether a tool result is trustworthy is the caller's
  job and cannot be done structurally.
- Cost predictions being wrong. That's a correctness bug and genuinely
  important to me, but it isn't a security issue — use the
  [cost-model discrepancy][discrepancy] template.
- Your provider API keys. This library never reads, stores, or sees one.

[discrepancy]: https://github.com/Ramtin2000/context-slim/issues/new?template=cost-model-discrepancy.md
