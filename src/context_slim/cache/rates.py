"""Provider rate tables.

Every entry carries the date it was checked and the URL it was checked against.
Rates rot; a cost model built on stale rates is worse than no cost model at all,
so ``assert_fresh()`` is wired into CI and fails the build at 90 days.

``price_confidence`` is deliberately explicit:

* ``"official"``   — read off the provider's own documentation.
* ``"third-party"``— read off an aggregator. Usable, but must be confirmed at
  the source before any published number depends on it.
* ``"archetype"``  — not a real product. A modelling stand-in used to study a
  pricing *shape* (see ``LEGACY_NO_WRITE_PREMIUM``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from fractions import Fraction

TTL_5M = 300
TTL_30M = 1800
TTL_1H = 3600

STALE_AFTER_DAYS = 90


@dataclass(frozen=True)
class ModelRates:
    provider: str
    model: str

    input_nano_per_mtok: int
    """Base input price in nanodollars per million tokens."""

    output_nano_per_mtok: int

    cache_read_mult: Fraction
    """Multiplier on base input for a cache *hit*."""

    cache_write_mult: Mapping[int, Fraction]
    """Multiplier on base input for a cache *write*, keyed by TTL in seconds.

    A value of ``1`` means the provider charges no premium — you pay the normal
    uncached input price you would have paid anyway. That is the pricing shape
    under which pruning is nearly always profitable, and it is why this is a
    mapping rather than a constant."""

    default_ttl: int
    min_cacheable_tokens: int
    verified_on: date
    source: str
    price_confidence: str = "official"

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.model}"

    def write_mult(self, ttl: int | None = None) -> Fraction:
        t = self.default_ttl if ttl is None else ttl
        try:
            return self.cache_write_mult[t]
        except KeyError as exc:
            valid = sorted(self.cache_write_mult)
            raise ValueError(f"{self.key} has no rate for ttl={t}s; known TTLs: {valid}") from exc

    def is_stale(self, today: date) -> bool:
        return today - self.verified_on > timedelta(days=STALE_AFTER_DAYS)


_ANTHROPIC_DOCS = "https://platform.claude.com/docs/en/build-with-claude/prompt-caching"
_OPENAI_DOCS = "https://developers.openai.com/api/docs/guides/prompt-caching"
_VERIFIED = date(2026, 8, 13)


CLAUDE_OPUS_5 = ModelRates(
    provider="anthropic",
    model="claude-opus-5",
    input_nano_per_mtok=5_000_000_000,  # $5.00 / Mtok
    output_nano_per_mtok=30_000_000_000,  # $30.00 / Mtok
    cache_read_mult=Fraction(1, 10),  # 0.1x
    cache_write_mult={TTL_5M: Fraction(5, 4), TTL_1H: Fraction(2, 1)},  # 1.25x / 2x
    default_ttl=TTL_5M,
    min_cacheable_tokens=512,
    verified_on=_VERIFIED,
    source=_ANTHROPIC_DOCS,
)

# --- OpenAI GPT-5.6 family -------------------------------------------------
# Multipliers (0.1x read, 1.25x write, 1024-token minimum) are from OpenAI's
# own caching guide and are marked official. The per-tier *prices* below came
# from third-party aggregators and are flagged accordingly: confirm them
# against OpenAI's pricing page before publishing any dollar figure.

# Luna's prices are confirmed against OpenAI's own model page: $0.20 input,
# $0.02 cached input, $1.20 output per Mtok. The $0.02 cached rate is exactly
# 0.1x the $0.20 input rate, which independently corroborates the 90%-off
# multiplier taken from the caching guide.
#
# Caveat not modelled here: requests over 272K input tokens incur a 2x input
# multiplier. Irrelevant at the 8k prefixes the benchmark uses, but it would
# silently break the cost model on genuinely long contexts.

GPT_5_6_LUNA = ModelRates(
    provider="openai",
    model="gpt-5.6-luna",
    input_nano_per_mtok=200_000_000,  # $0.20 / Mtok
    output_nano_per_mtok=1_200_000_000,  # $1.20 / Mtok
    cache_read_mult=Fraction(1, 10),  # $0.02 / Mtok
    cache_write_mult={TTL_30M: Fraction(5, 4)},  # 1.25x, GPT-5.6+ only
    default_ttl=TTL_30M,
    min_cacheable_tokens=1024,
    verified_on=_VERIFIED,
    source="https://developers.openai.com/api/docs/models/gpt-5.6-luna",
    price_confidence="official",
)

GPT_5_6_TERRA = ModelRates(
    provider="openai",
    model="gpt-5.6-terra",
    input_nano_per_mtok=2_000_000_000,  # $2.00 / Mtok
    output_nano_per_mtok=12_000_000_000,
    cache_read_mult=Fraction(1, 10),
    cache_write_mult={TTL_30M: Fraction(5, 4)},
    default_ttl=TTL_30M,
    min_cacheable_tokens=1024,
    verified_on=_VERIFIED,
    source=_OPENAI_DOCS,
    price_confidence="third-party",
)

# --- The null case ---------------------------------------------------------
# Pricing shape of OpenAI models *before* GPT-5.6, where cache writes carried
# no premium. Not a purchasable model: an archetype, used to demonstrate that
# the break-even effect is caused by the write premium specifically and
# vanishes without it. Priced identically to Luna so the contrast isolates
# exactly one variable.

LEGACY_NO_WRITE_PREMIUM = ModelRates(
    provider="openai",
    model="legacy-no-write-premium",
    input_nano_per_mtok=200_000_000,
    output_nano_per_mtok=1_200_000_000,
    cache_read_mult=Fraction(1, 10),
    cache_write_mult={TTL_30M: Fraction(1, 1)},  # no premium
    default_ttl=TTL_30M,
    min_cacheable_tokens=1024,
    verified_on=_VERIFIED,
    source=_OPENAI_DOCS,
    price_confidence="archetype",
)


RATES: dict[str, ModelRates] = {
    r.key: r
    for r in (
        CLAUDE_OPUS_5,
        GPT_5_6_LUNA,
        GPT_5_6_TERRA,
        LEGACY_NO_WRITE_PREMIUM,
    )
}


def get(model_key: str) -> ModelRates:
    try:
        return RATES[model_key]
    except KeyError as exc:
        raise KeyError(f"unknown model {model_key!r}; known: {sorted(RATES)}") from exc


def assert_fresh(today: date) -> None:
    """Raise if any rate entry has gone stale. Called from the test suite."""
    stale = [r.key for r in RATES.values() if r.is_stale(today)]
    if stale:
        raise AssertionError(
            f"rate entries older than {STALE_AFTER_DAYS} days: {stale}. "
            "Re-verify against the provider's pricing page and bump verified_on."
        )
