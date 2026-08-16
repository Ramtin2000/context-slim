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
import hashlib
import json
import os
import pathlib
import random
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
CHECKPOINT = RESULTS / ".killgate_progress.json"
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


def build_loop(
    prefix_tokens: int, turns: int, tool_tokens: int, salt: str = ""
) -> list[Message]:
    """A synthetic agent loop: a big stable system prefix, then tool churn.

    ``salt`` is the cache-namespace isolator and is not cosmetic. Provider
    prompt caches are content-addressed and scoped to the *account*, not to the
    process or the experiment, so two arms sharing a prefix share cache entries:
    whichever runs first pays the cache write and every later arm free-rides on
    it. Runs 1 and 2 had no salt, and their cost aggregates are worthless
    because of it - conditions converged on bit-identical costs once nothing
    was being written any more. A high-entropy salt at the very front of the
    system message forces divergence at token 0, so each arm pays its own way.

    The tool-call plumbing is not decoration. OpenAI rejects a ``tool`` message
    that does not answer an immediately preceding ``assistant`` message carrying
    a matching ``tool_calls`` entry, so the triple must be well-formed or the
    whole run 400s. ``render_stub`` copies every key it is given, which is what
    keeps ``tool_call_id`` intact through pruning.
    """
    prefix = _filler(prefix_tokens)
    if salt:
        prefix = f"[session {salt}]\n{prefix}"
    msgs: list[Message] = [{"role": "system", "content": prefix}]
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


def prune(
    messages: list[Message],
    condition: str,
    keep_recent: int = 2,
    budget: int = 9500,
) -> list[Message]:
    """Apply one strategy's worth of pruning, ignoring economics entirely.

    This is deliberately *not* using ``context_slim.plan()`` — the gate has to
    measure what the incumbent strategies do, not what we think they should do.
    """
    if condition == "no_prune":
        return list(messages)
    order = "oldest_first" if condition == "oldest_first" else "tail_first"
    out = [dict(m) for m in messages]
    for idx in sorted(cleared_indices(messages, order, keep_recent, budget)):
        if idx < len(out):
            out[idx] = expiry.render_stub(out[idx], f"{order} policy")
    return out


def cleared_indices(
    messages: list[Message], order: str, keep_recent: int, budget: int
) -> set[int]:
    """Indices cleared by a monotonic, budget-triggered policy.

    Two properties, both taken from how real pruners behave and both absent
    from earlier revisions of this harness:

    1. **Monotonic.** Once cleared, an index stays cleared.
       ``clear_tool_uses_20250919`` and LangChain's truncation both work this
       way. Run 3's policy recomputed "newest half of candidates" from scratch
       each turn, so previously-stubbed messages reverted to full content as
       the loop grew - a deep prefix change every turn, which made
       ``tail_first`` look catastrophic (69.1% hit) for reasons that had
       nothing to do with tail-first pruning.

    2. **Budget-triggered.** Clear the *minimum* needed to get back under a
       token threshold, rather than a fixed fraction. Clearing a fraction
       accumulates monotonically until every candidate is cleared, at which
       point both orderings converge on the same set and the independent
       variable disappears - which is what a fraction-based fix produced
       (8585 vs 8566 tokens, indistinguishable).

    Only under both properties do the two orderings stay genuinely different
    while remaining realistic.
    """
    cleared: set[int] = set()
    for end in range(4, len(messages) + 1, 3):
        window = messages[:end]
        cands = expiry.candidates(window, order=order, keep_recent=keep_recent)
        queue = [c.index for c in cands if c.index not in cleared]
        while queue:
            staged = [
                expiry.render_stub(dict(m), "policy") if i in cleared else m
                for i, m in enumerate(window)
            ]
            if total_tokens(staged) <= budget:
                break
            cleared.add(queue.pop(0))
    return cleared


@dataclass
class TurnRecord:
    condition: str
    turn: int
    est_tokens: int
    salt: str = ""
    prompt_tokens: int = 0
    cached_tokens: int = 0
    cost_nano: int = 0
    at: str = ""  # wall clock, so cache-TTL gaps across resumes are detectable

    @property
    def key(self) -> str:
        """Identity includes the salt.

        Without it, a checkpoint from a differently-designed run resumes into
        this one: run 2's ``no_prune#0@1`` matches run 3's, so 180 confounded
        records would have been silently adopted as if they were corrected
        measurements. The salt is the cache namespace, so binding to it makes
        incompatible runs unable to collide.
        """
        return f"{self.condition}@{self.turn}#{self.salt[:8]}"

    @property
    def done(self) -> bool:
        return self.prompt_tokens > 0


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


def plan_run(
    prefix_tokens: int,
    turns: int,
    tool_tokens: int,
    repeats: int,
    seed: int = 0,
) -> list[TurnRecord]:
    """Lay out the run with one cache namespace per arm, arms interleaved.

    Every (condition, repeat) gets its own salt so no arm can read another's
    cache entries, and the arms are shuffled under a fixed seed so position in
    the execution sequence is not confounded with treatment.
    """
    records: list[TurnRecord] = []
    arms = [(c, r) for c in CONDITIONS for r in range(repeats)]
    random.Random(seed).shuffle(arms)
    for condition, rep in arms:
        salt = hashlib.blake2b(
            f"{seed}:{condition}:{rep}".encode(), digest_size=8
        ).hexdigest()
        loop = build_loop(prefix_tokens, turns, tool_tokens, salt)
        for t in range(1, turns + 1):
            window = prune(loop[: 1 + t * 3], condition)
            records.append(
                TurnRecord(
                    condition=f"{condition}#{rep}",
                    turn=t,
                    est_tokens=total_tokens(window),
                    salt=salt,
                )
            )
    return records


def load_checkpoint() -> dict[str, dict[str, Any]]:
    """Completed requests from earlier attempts, keyed by condition@turn."""
    if not CHECKPOINT.exists():
        return {}
    try:
        raw = json.loads(CHECKPOINT.read_text())
        return {k: v for k, v in raw.items() if isinstance(v, dict)}
    except (json.JSONDecodeError, OSError):
        return {}


def save_checkpoint(records: list[TurnRecord]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(
        json.dumps({r.key: asdict(r) for r in records if r.done}, indent=2)
    )


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
    p.add_argument(
        "--reasoning-effort",
        default="none",
        help=(
            "gpt-5.6 is a reasoning model and reasoning tokens are billed against "
            "max_completion_tokens; 'none' keeps the response empty and cheap"
        ),
    )
    p.add_argument("--max-completion-tokens", type=int, default=16)
    p.add_argument(
        "--timeout", type=float, default=90.0, help="per-request timeout, seconds"
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="SDK-level retries with exponential backoff, for flaky links",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="ignore the checkpoint and re-run every request from scratch",
    )
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=0, help="arm-order shuffle seed")
    p.add_argument("--max-spend", type=float, default=1.00, help="hard USD cap, cumulative")
    p.add_argument("--dry-run", action="store_true", help="price the run, call nothing")
    args = p.parse_args(argv)

    load_dotenv()
    if args.fresh:
        CHECKPOINT.unlink(missing_ok=True)

    records = plan_run(
        args.prefix_tokens, args.turns, args.tool_tokens, args.repeats, args.seed
    )
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

    client = OpenAI(timeout=args.timeout, max_retries=args.max_retries)

    # Resume: a flaky link cannot finish 180 requests in one pass, and without
    # this every failure discards the money already spent.
    resumed = 0
    if not args.fresh:
        done = load_checkpoint()
        for rec in records:
            prior = done.get(rec.key)
            if prior:
                rec.prompt_tokens = prior.get("prompt_tokens", 0)
                rec.cached_tokens = prior.get("cached_tokens", 0)
                rec.cost_nano = prior.get("cost_nano", 0)
                rec.at = prior.get("at", "")
                resumed += 1
        if resumed:
            print(f"resuming   : {resumed}/{len(records)} already done, skipping\n")

    run_cost = Money.zero()
    completed = 0
    failure: str | None = None

    # Anything spent must be recorded even if the run dies partway, or the
    # ledger under-reports and the cap silently stops protecting anything.
    try:
        for i, rec in enumerate(records):
            if rec.done:
                completed = i + 1
                continue
            loop = build_loop(
                args.prefix_tokens, args.turns, args.tool_tokens, rec.salt
            )
            window = prune(loop[: 1 + rec.turn * 3], rec.condition.split("#")[0])
            resp = client.chat.completions.create(
                model=args.api_model,
                messages=window,  # type: ignore[arg-type]
                max_completion_tokens=args.max_completion_tokens,
                reasoning_effort=args.reasoning_effort,
            )
            usage = resp.usage
            assert usage is not None
            details = getattr(usage, "prompt_tokens_details", None)
            rec.prompt_tokens = usage.prompt_tokens
            rec.cached_tokens = getattr(details, "cached_tokens", 0) or 0
            cost = actual_cost(rec.prompt_tokens, rec.cached_tokens, args.model)
            rec.cost_nano = cost.nano
            rec.at = datetime.now(timezone.utc).isoformat()
            run_cost = run_cost + cost
            completed = i + 1

            if completed % 10 == 0:
                save_checkpoint(records)
            if completed % 20 == 0:
                pct = 100 * rec.cached_tokens / max(1, rec.prompt_tokens)
                print(
                    f"  {completed:>3}/{len(records)}  {run_cost}  "
                    f"cached {pct:.0f}%",
                    flush=True,
                )

            if run_cost.nano + spent.nano > cap.nano:
                print(f"\nSTOP at request {i}: cap reached.", file=sys.stderr)
                break
    except KeyboardInterrupt:
        failure = "interrupted by user"
    except Exception as exc:  # partial results are still worth keeping
        failure = f"{type(exc).__name__}: {exc}"

    save_checkpoint(records)

    if failure:
        print(
            f"\nRUN INCOMPLETE after {completed}/{len(records)} requests: {failure}\n"
            f"{run_cost} spent so far has been recorded to the ledger.\n"
            f"Re-run the same command to resume; completed requests are skipped.",
            file=sys.stderr,
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat()
    out = RESULTS / "killgate.json"
    out.write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "args": vars(args),
                "completed": completed,
                "total_requests": len(records),
                "failure": failure,
                "actual_cost_nano": run_cost.nano,
                "records": [asdict(r) for r in records],
            },
            indent=2,
        )
    )

    ledger["total_nano"] = spent.nano + run_cost.nano
    ledger["runs"].append(
        {"timestamp": stamp, "cost_nano": run_cost.nano, "completed": completed}
    )
    save_ledger(ledger)

    by_condition: dict[str, int] = {}
    for rec in records:
        by_condition[rec.condition.split("#")[0]] = (
            by_condition.get(rec.condition.split("#")[0], 0) + rec.cost_nano
        )

    outstanding = [r for r in records if not r.done]
    if outstanding:
        print(f"\n{len(outstanding)}/{len(records)} requests still outstanding.")

    # A resume that straddles the 30-minute cache TTL reads as a cache miss
    # that the pruning policy did not cause. Surface it rather than let it
    # quietly contaminate the comparison.
    stamps = sorted(r.at for r in records if r.at)
    if stamps and resumed:
        span_min = (
            datetime.fromisoformat(stamps[-1]) - datetime.fromisoformat(stamps[0])
        ).total_seconds() / 60
        if span_min > 30:
            print(
                f"\n⚠️  data spans {span_min:.0f} min, longer than the 30-min cache "
                f"TTL.\n    Requests after a gap will show false cache misses. "
                f"Re-run with --fresh\n    for a clean single-pass measurement "
                f"before publishing any number.",
                file=sys.stderr,
            )

    print(f"\nactual cost this run: {run_cost}")
    for name, nano in sorted(by_condition.items()):
        print(f"  {name:<14} {Money(nano)}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
