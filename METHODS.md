# Methods and threats to validity

Four runs. Three produced confident headline numbers that were artifacts, each
pointing the opposite way to the one before it. Run 1 said pruning saves 29.6%;
run 3 said it costs 41.6% more. Same code, same provider, same week.

That instability is itself the most transferable finding here, so it is
documented rather than quietly dropped.

## Confound 1 — cross-arm cache sharing (runs 1-2, invalidated)

Provider prompt caches are content-addressed and scoped to the **account**, not
to the process or the experiment. All arms were built from one identical prefix
and executed in blocks, so whichever arm ran first paid the cache write and
every later arm free-rode on it. Execution order was confounded with treatment.

Caught by a variance check, not by inspection: sigma was roughly half the mean;
every condition declined monotonically across repeats; and different conditions
returned bit-identical costs ($0.003214 in three arms), which only happens when
arms are reading the same cache entries.

**Fix.** A unique high-entropy salt at the head of each arm's system prompt, so
each of the 15 arms occupies its own cache namespace and pays its own way. Arms
shuffled under a fixed seed so position cannot correlate with treatment.

## Confound 2 — non-monotonic pruning policy (run 3, half invalidated)

`tail_first` came out worst, contradicting prefix-cache theory. Replaying the
stub sets showed the policy recomputed "newest half of candidates" from scratch
each turn, so previously-stubbed messages fell out of that half and reverted to
full content - a deep prefix change on every turn.

Real pruners never un-clear. `clear_tool_uses_20250919` and LangChain's
truncation are both monotonic.

**Fix.** A persistent cleared-set that only grows.

## Confound 3 — fraction-based clearing erases the variable (found pre-run-4)

Accumulating a fixed fraction converges on clearing every candidate, at which
point both orderings produce the same set and the independent variable
disappears (8585 vs 8566 tokens - indistinguishable).

**Fix.** Budget-triggered clearing: clear the minimum needed to get back under a
token threshold, which is what `clear_tool_uses_20250919` does.

## Run 4 design (the valid one)

Salted namespaces, shuffled arms, monotonic budget-triggered policies, n=5,
bootstrap 95% CIs on arm-level means. Predictions were pre-registered before the
data landed: tail-first's hit rate would rise from 69.1%, tail-first would be
cheaper than oldest-first, and no-prune would stay cheapest. All three held.

## Known limitations

- 8k prefixes, 20 turns, synthetic loops. Larger regimes untested.
- One model, one provider, one account, one region.
- `no_prune` sends more tokens by construction; that it still wins on cost is
  the point, but it is not a like-for-like token comparison.
- A cache-hit dip at turn 14 appears in all three conditions including
  `no_prune`, so it is not caused by pruning. Unexplained.
- Provider cache internals are opaque. We observe billing, not the cache.
- n=5 arms: adequate for the observed effect sizes, thin for subtle ones.

## A note for anyone benchmarking prompt caches

Prompt-cache experiments are **stateful across arms** in a way ordinary A/B
benchmarks are not. The cache is a hidden channel between conditions. Any
comparison that does not explicitly namespace its cache keys is measuring
execution order as much as treatment.
