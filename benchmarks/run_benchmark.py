"""Standalone comparison runner: naive FIFO truncation vs. context-slim.

**What this is.** An offline, zero-dependency demonstration you can run right
after cloning the repo, with no API key and no network call. It loads
``benchmarks/traces/sample_agent_trace.json``, then projects the cost of two
strategies using the exact same validated cost primitives the rest of the
library ships (:mod:`context_slim.cache.model`):

* ``naive-fifo`` — what ``messages.pop(0)`` in a loop actually does: drop the
  oldest turns once the transcript crosses a token budget, unconditionally,
  with no regard for what that does to the cache. This is the textbook
  sliding-window pattern the whole project exists to argue against.
* ``context-slim`` — :func:`context_slim.plan` / :func:`context_slim.apply`,
  which only prunes an edit that is projected to pay for itself by the given
  horizon, and refuses (or defers) everything else.

**What this is NOT.** A substitute for the real benchmark. These numbers are
*projected* from the cost model, not measured against a live API — no
``cached_tokens`` counter is ever read here. For the actual, live-validated
result this project's headline claim rests on (n=5 arms, bootstrap 95% CIs,
real ``usage.prompt_tokens_details``), see ``bench/killgate.py`` and the
table in the project ``README.md``. That harness costs real API dollars to
run; this script costs nothing, on purpose, so a new contributor can see the
shape of the argument before spending any money confirming it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from context_slim import apply, plan
from context_slim.cache.model import break_even
from context_slim.cache.prefix import total_tokens
from context_slim.cache.rates import ModelRates
from context_slim.cache.rates import get as get_rates
from context_slim.schemas import BreakEven, Message

DEFAULT_TRACE = pathlib.Path(__file__).parent / "traces" / "sample_agent_trace.json"


def naive_fifo_drop(messages: list[Message], n_to_drop: int) -> list[Message]:
    """What ``messages[:1] + messages[-K:]``-style pruning actually does: keep
    the system message pinned — nearly every naive implementation does at
    least that much — then unconditionally drop the oldest ``n_to_drop`` turns
    after it. No per-message cost awareness at all; that is the whole point
    of the comparison.
    """
    if not messages:
        return messages
    head, rest = messages[:1], messages[1:]
    return head + rest[n_to_drop:]


def project_naive_fifo(
    messages: list[Message], rates: ModelRates, n_to_drop: int, horizon: int
) -> BreakEven:
    """Cost of a naive front-pop, computed with context-slim's own cost
    primitives so the comparison is apples-to-apples: same rate table, same
    read/write multipliers, same derivation — just without the refusal gate.

    The edit happens right after the pinned system message, so everything
    from there to the end counts as "at or after the edit point" — the same
    ``w_tokens`` / ``s_tokens`` relationship :func:`context_slim.expiry.candidates`
    computes for its own candidates, just never checked against a break-even.
    """
    kept = naive_fifo_drop(messages, n_to_drop)
    w_tokens = total_tokens(messages[1:])
    s_tokens = w_tokens - total_tokens(kept[1:])
    return break_even(rates, w_tokens=w_tokens, s_tokens=s_tokens, horizon=horizon)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", nargs="?", default=str(DEFAULT_TRACE))
    p.add_argument("--model", default="openai/gpt-5.6-luna")
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument(
        "--naive-drop",
        type=int,
        default=4,
        help="messages a naive sliding window would drop from the front",
    )
    args = p.parse_args(argv)

    raw = json.loads(pathlib.Path(args.trace).read_text())
    messages: list[Message] = raw["messages"] if isinstance(raw, dict) else raw
    rates = get_rates(args.model)

    total_before = total_tokens(messages)

    naive_kept = naive_fifo_drop(messages, args.naive_drop)
    naive_removed = total_before - total_tokens(naive_kept)
    naive = project_naive_fifo(messages, rates, args.naive_drop, args.horizon)

    slim_plan = plan(messages, model=args.model, horizon=args.horizon, order="tail_first")
    _slim_out, slim_report = apply(messages, slim_plan)

    rows = [
        ("don't prune", total_before, 0, "$0.000000", "$0.000000", "—"),
        (
            "naive-fifo",
            total_before - naive_removed,
            naive_removed,
            str(naive.cost_now),
            str(naive.net_at_horizon),
            f"{naive.turns:.1f}" if naive.turns is not None else "never",
        ),
        (
            "context-slim",
            total_before - slim_report.tokens_removed,
            slim_report.tokens_removed,
            str(slim_report.cost_now),
            str(slim_report.net_at_horizon),
            "—",
        ),
    ]

    header = (
        "strategy",
        "tokens sent",
        "tokens removed",
        "cost now",
        f"net @ {args.horizon}t",
        "payback (turns)",
    )
    widths = [max(len(str(r[i])) for r in (header, *rows)) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(*r))

    print(
        f"\n{slim_report}\n"
        "\nThese numbers are projected from the cost model in cache/model.py, "
        "not measured against a live API. Run `python -m bench.killgate` for "
        "the real, live-validated comparison (costs a small amount of real "
        "API spend; see bench/results/ for prior runs)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
