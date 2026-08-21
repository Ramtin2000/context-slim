"""Measure the dedupe design that shipped against the one that was rejected.

The rejected design is a byte-level rolling hash with content-defined chunk
boundaries. It is the textbook approach and it is what this module exists to
show is unaffordable in pure Python.

    python -m bench.bench_dedupe
"""

from __future__ import annotations

import hashlib
import statistics
import time

from context_slim.ops.pruner import collapse_whitespace, dedupe_blocks

WINDOW = 64
MOD = (1 << 61) - 1
BASE = 257


def rolling_hash_chunks(data: bytes, mask: int = 0x3FF) -> list[int]:
    """Content-defined chunking via a polynomial rolling hash.

    One Python-level iteration per byte. This is the design that was rejected;
    it is kept here so the claim is measured rather than asserted.
    """
    bounds: list[int] = []
    h = 0
    power = pow(BASE, WINDOW - 1, MOD)
    for i, byte in enumerate(data):
        if i >= WINDOW:
            h = (h - data[i - WINDOW] * power) % MOD
        h = (h * BASE + byte) % MOD
        if i >= WINDOW and (h & mask) == 0:
            bounds.append(i)
    return bounds


def blocks_blake2b(text: str) -> list[str]:
    """The design that shipped: C-level split, C-level hash."""
    return [
        hashlib.blake2b(b.encode(), digest_size=8).hexdigest()
        for b in text.split("\n\n")
    ]


def _sample(fn, arg, n: int = 5) -> float:
    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(arg)
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main() -> None:
    para = ("lorem ipsum dolor sit amet consectetur adipiscing elit " * 12).strip()
    text = "\n\n".join(f"[{i}] {para}" for i in range(140))
    text = text[:100_000]
    data = text.encode()
    print(f"input: {len(data):,} bytes, {text.count(chr(10)+chr(10))+1} blocks\n")

    rolling = _sample(rolling_hash_chunks, data)
    blake = _sample(blocks_blake2b, text)
    ws = _sample(collapse_whitespace, text)
    dd = _sample(lambda t: dedupe_blocks([t, t]), text)

    print(f"{'rejected: 64-byte rolling hash (per-byte Python loop)':<52}{rolling:>8.2f} ms")
    print(f"{'shipped:  str.split + blake2b (per-block loop)':<52}{blake:>8.2f} ms")
    print(f"{'shipped:  collapse_whitespace':<52}{ws:>8.2f} ms")
    print(f"{'shipped:  dedupe_blocks (2x100KB)':<52}{dd:>8.2f} ms")
    print(f"\nspeedup on the hashing step: {rolling / blake:.0f}x")


if __name__ == "__main__":
    main()
