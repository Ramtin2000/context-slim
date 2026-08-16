"""Day 4 kill gate — does oldest-first pruning actually cost money?

The experiment the whole project is contingent on. It replays synthetic
conversations under three pruning strategies and reads the provider's own
cached-token counters back, so the answer comes from OpenAI's accounting rather
than from our model of it.

Two design choices keep this cheap enough to run for pocket change:

* ``max_output_tokens=1`` — we are measuring input-side cache behaviour, so
  generation is pure waste. Output cost is effectively zero.
* an 8k prefix rather than a "realistic" 100k one — cache mechanics only need
  to clear the provider's 1024-token minimum to exhibit the effect, and W/S
  ratio and horizon (the things the model is parameterised on) are preserved.

Spend controls, in order of how much they matter:

1. ``--dry-run`` prices the run and calls nothing. Always run this first.
2. ``--max-spend`` (default $1.00) aborts pre-flight on the projection.
3. a persisted cumulative ledger, checked across runs, so repeated invocations
   cannot creep past the cap. A per-run cap alone would not hold.

Usage::

    python -m bench.killgate --dry-run
    OPENAI_API_KEY=... python -m bench.killgate --max-spend 0.50

Prefer a ``.env`` file (git-ignored, mode 600) over an inline environment
variable — a key on the command line lands in shell history.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from context_slim._types import Message, Money
from context_slim.cache import rates as rate_tables
from context_slim.cache.model import read_cost, token_cost, write_cost
from context_slim.cache.prefix import total_tokens
from context_slim.ops import expiry

RESULTS = pathlib.Path(__file__).parent / "results"
LEDGER = RESULTS / ".spend_ledger.json"
CONDITIONS = ("no_prune", "oldest_first", "tail_first")

# Deterministic filler. Real prose so the tokenizer behaves normally, repeated
# to a target size. Content is irrelevant to cache mechanics; only length is.
_FILLER = (
    "The agent inspected the repository, listed the files in the working tree, "
    "and recorded the results of each tool invocation for later reference. "
)


def load_dotenv() -> None:
    """Read ``.env`` into the environment if present.

    A key passed inline on the command line is written to shell history in
    plaintext; a mode-600 ``.env`` is not. Existing environment variables win,
    so an explicit inline key still overrides the file.
    """
    env = pathlib.Path(__file__).resolve().parents[1] / ".env"
    if not env.is_file():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _filler(target_tokens: int) -> str:
    reps = max(1, target_tokens * 5 // len(_FILLER))
    return (_FILLER * reps)[: target_tokens * 4]


def build_loop(prefix_tokens: int, turns: int, tool_tokens: int) -> list[Message]:
    """A synthetic agent loop: a big stable system prefix, then tool churn.

    The tool-call plumbing is not decoration. OpenAI rejects a ``tool`` message
    that does not answer an immediately preceding ``assistant`` message carrying
    a matching ``tool_calls`` entry, so the triple must be well-formed or the
    whole run 400s. ``render_stub`` copies every key it is given, which is what
    keeps ``tool_call_id`` intact through pruning.
    """
    msgs: list[Message] = [{"role": "system", "content": _filler(prefix_tokens)}]
    for i in range(turns):
        call_id = f"call_{i:04d}"
        msgs.append({"role": "user", "content": f"Step {i}: what next?"})
        msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "inspect_files",
                            "arguments": json.dumps({"step": i}),
                        },
                    }
                ],
            }
        )
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": f"[{i}] " + _filler(tool_tokens),
            }
        )
    return msgs


def prune(messages: list[Message], condition: str, keep_recent: int = 2) -> list[Message]:
    """Apply one strategy's worth of pruning, ignoring economics entirely.

    This is deliberately *not* using ``context_slim.plan()`` — the gate has to
    measure what the incumbent strategies do, not what we think they should do.
    """
    if condition == "no_prune":
        return list(messages)
    order = "oldest_first" if condition == "oldest_first" else "tail_first"
    cands = expiry.candidates(messages, order=order, keep_recent=keep_recent)
    if not cands:
        return list(messages)
    out = [dict(m) for m in messages]
    for c in cands[: max(1, len(cands) // 2)]:
        out[c.index] = expiry.render_stub(out[c.index], f"{order} policy")
    return out


@dataclass
class TurnRecord:
    condition: str
    turn: int
    est_tokens: int
    prompt_tokens: int = 0
    cached_tokens: int = 0
    cost_nano: int = 0


def project_cost(records: list[TurnRecord], model_key: str) -> Money:
    """Price a run before it happens, assuming the worst: nothing cached."""
    r = rate_tables.get(model_key)
    total = Money.zero()
    for rec in records:
        total = total + token_cost(r, rec.est_tokens, Fraction(1))
    return total


def actual_cost(prompt_tokens: int, cached_tokens: int, model_key: str) -> Money:
    """Price one real response from its usage block.

    Uncached tokens are charged as a cache write (they populate the cache);
    cached tokens are charged at the read rate.
    """
    r = rate_tables.get(model_key)
    fresh = max(0, prompt_tokens - cached_tokens)
    return write_cost(r, fresh) + read_cost(r, cached_tokens)


def plan_run(prefix_tokens: int, turns: int, tool_tokens: int, repeats: int) -> list[TurnRecord]:
    records: list[TurnRecord] = []
    for condition in CONDITIONS:
        for rep in range(repeats):
            loop = build_loop(prefix_tokens, turns, tool_tokens)
            for t in range(1, turns + 1):
                window = prune(loop[: 1 + t * 3], condition)
                records.append(
                    TurnRecord(
                        condition=f"{condition}#{rep}",
                        turn=t,
                        est_tokens=total_tokens(window),
                    )
                )
    return records


def load_ledger() -> dict[str, Any]:
    if LEDGER.exists():
        data: dict[str, Any] = json.loads(LEDGER.read_text())
        return data
    return {"total_nano": 0, "runs": []}


def save_ledger(ledger: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(ledger, indent=2))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.killgate", description=__doc__)
    p.add_argument("--model", default="openai/gpt-5.6-luna")
    p.add_argument("--api-model", default="gpt-5.6-luna", help="model id sent to the API")
    p.add_argument("--prefix-tokens", type=int, default=8_000)
    p.add_argument("--turns", type=int, default=20)
    p.add_argument("--tool-tokens", type=int, default=400)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-spend", type=float, default=1.00, help="hard USD cap, cumulative")
    p.add_argument("--dry-run", action="store_true", help="price the run, call nothing")
    args = p.parse_args(argv)

    load_dotenv()
    records = plan_run(args.prefix_tokens, args.turns, args.tool_tokens, args.repeats)
    projected = project_cost(records, args.model)
    ledger = load_ledger()
    spent = Money(int(ledger["total_nano"]))
    cap = Money.from_usd(f"{args.max_spend:.6f}")

    print(f"model      : {args.model}")
    print(f"requests   : {len(records)}")
    print(f"est tokens : {sum(r.est_tokens for r in records):,}")
    print(f"projected  : {projected}  (worst case, nothing cached)")
    print(f"spent      : {spent} lifetime on this project")
    print(f"cap        : {cap}")

    if spent.nano + projected.nano > cap.nano:
        print(
            f"\nABORT: {spent} already spent + {projected} projected exceeds the "
            f"{cap} cap. Raise --max-spend deliberately if you mean to.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        print("\ndry run — no API calls made.")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "\nno OPENAI_API_KEY found. Put it in a git-ignored .env:\n"
            "    printf 'OPENAI_API_KEY=sk-proj-...\\n' > .env && chmod 600 .env",
            file=sys.stderr,
        )
        return 1

    try:
        from openai import OpenAI
    except ImportError:
        print("\ninstall the bench extra first:  pip install -e '.[bench]'", file=sys.stderr)
        return 1

    client = OpenAI()
    run_cost = Money.zero()
    for i, rec in enumerate(records):
        loop = build_loop(args.prefix_tokens, args.turns, args.tool_tokens)
        window = prune(loop[: 1 + rec.turn * 3], rec.condition.split("#")[0])
        resp = client.chat.completions.create(
            model=args.api_model,
            messages=window,  # type: ignore[arg-type]
            max_completion_tokens=1,
        )
        usage = resp.usage
        assert usage is not None
        details = getattr(usage, "prompt_tokens_details", None)
        rec.prompt_tokens = usage.prompt_tokens
        rec.cached_tokens = getattr(details, "cached_tokens", 0) or 0
        cost = actual_cost(rec.prompt_tokens, rec.cached_tokens, args.model)
        rec.cost_nano = cost.nano
        run_cost = run_cost + cost

        if run_cost.nano + spent.nano > cap.nano:
            print(f"\nSTOP at request {i}: cap reached.", file=sys.stderr)
            break

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    out = RESULTS / "killgate.json"
    out.write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "args": vars(args),
                "actual_cost_nano": run_cost.nano,
                "records": [asdict(r) for r in records],
            },
            indent=2,
        )
    )

    ledger["total_nano"] = spent.nano + run_cost.nano
    ledger["runs"].append({"timestamp": stamp, "cost_nano": run_cost.nano})
    save_ledger(ledger)

    by_condition: dict[str, int] = {}
    for rec in records:
        by_condition[rec.condition.split("#")[0]] = (
            by_condition.get(rec.condition.split("#")[0], 0) + rec.cost_nano
        )

    print(f"\nactual cost: {run_cost}")
    for name, nano in sorted(by_condition.items()):
        print(f"  {name:<14} {Money(nano)}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
