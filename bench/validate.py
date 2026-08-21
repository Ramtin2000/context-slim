"""Check the cost model against the provider's own counters.

Every other pruning tool asks you to trust its token accounting. This one is
falsifiable: we predict how many tokens the API will report as cached *before*
making the call, then compare against ``usage.prompt_tokens_details``.

The prediction is not a re-reading of the response. It comes from the prefix
tracker: given what was sent last turn and what changed this turn, how much of
the prompt should still match? That is the claim worth testing, because every
dollar figure in the project is downstream of it.

    python -m bench.validate --dry-run
    python -m bench.validate --max-spend 0.20
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass

from context_slim import get_rates
from context_slim._types import Message, Money

from .killgate import (
    RESULTS,
    actual_cost,
    build_loop,
    load_dotenv,
    load_ledger,
    prune,
    save_ledger,
    total_tokens,
)

OUT = RESULTS / "validation.json"


@dataclass
class Prediction:
    turn: int
    condition: str
    predicted_prompt: int
    predicted_cached: int
    actual_prompt: int = 0
    actual_cached: int = 0

    @property
    def cached_abs_pct_error(self) -> float:
        if not self.actual_prompt:
            return 0.0
        # Error as a share of the prompt, not of cached tokens: a 100-token miss
        # on a 10-token cache is not a 1000% modelling failure, it is a 1%
        # misprediction of the prompt. Normalising by the smaller quantity would
        # make the metric explode on cold turns and flatter us on warm ones.
        return 100 * abs(self.predicted_cached - self.actual_cached) / self.actual_prompt


def predict(window: list[Message], previous: list[Message] | None) -> tuple[int, int]:
    """Predict (prompt_tokens, cached_tokens) for this request.

    Prefix caches match from token 0 until the first byte that differs, so the
    cached portion is the tokens of the longest common message prefix with what
    we sent last time. Anything from the first changed message onward is a miss.
    """
    prompt = total_tokens(window)
    if not previous:
        return prompt, 0

    shared: list[Message] = []
    for old, new in zip(previous, window):
        if old != new:
            break
        shared.append(new)
    return prompt, total_tokens(shared) if shared else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="openai/gpt-5.6-luna")
    ap.add_argument("--api-model", default="gpt-5.6-luna")
    ap.add_argument("--prefix-tokens", type=int, default=8_000)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--tool-tokens", type=int, default=400)
    ap.add_argument("--conditions", default="no_prune,tail_first")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-spend", type=float, default=1.0)
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    preds: list[Prediction] = []
    windows: dict[str, list[list[Message]]] = {}

    for cond in conditions:
        loop = build_loop(args.prefix_tokens, args.turns, args.tool_tokens, f"validate-{cond}")
        seq: list[list[Message]] = []
        prev: list[Message] | None = None
        for t in range(1, args.turns + 1):
            window = prune(loop[: 1 + t * 3], cond)
            p, c = predict(window, prev)
            preds.append(Prediction(t, cond, p, c))
            seq.append(window)
            prev = window
        windows[cond] = seq

    est = sum(p.predicted_prompt for p in preds)
    rates = get_rates(args.model)
    worst = Money(est * rates.input_nano_per_mtok // 1_000_000)
    spent = Money(int(load_ledger().get("total_nano", 0)))

    print(f"model      : {args.model}")
    print(f"requests   : {len(preds)}")
    print(f"est tokens : {est:,}")
    print(f"projected  : {worst}  (worst case, nothing cached)")
    print(f"spent      : {spent} lifetime on this project")

    if args.dry_run:
        print("\ndry run — no API calls made.")
        return 0

    cap = Money.from_usd(str(args.max_spend))
    if spent.nano + worst.nano > cap.nano:
        print(f"\nABORT: would exceed the {cap} cap.", file=sys.stderr)
        return 2

    # Same .env handling as the kill gate: a key on the command line lands in
    # shell history, a mode-600 file does not.
    load_dotenv()

    from openai import OpenAI

    client = OpenAI(timeout=20.0, max_retries=4)
    run_cost = Money.zero()
    try:
        for pred in preds:
            window = windows[pred.condition][pred.turn - 1]
            resp = client.chat.completions.create(
                model=args.api_model,
                messages=window,  # type: ignore[arg-type]
                max_completion_tokens=16,
                reasoning_effort="none",
            )
            usage = resp.usage
            assert usage is not None
            details = getattr(usage, "prompt_tokens_details", None)
            pred.actual_prompt = usage.prompt_tokens
            pred.actual_cached = getattr(details, "cached_tokens", 0) or 0
            run_cost = run_cost + actual_cost(
                pred.actual_prompt, pred.actual_cached, args.model
            )
    except Exception as exc:  # partial results still tell us something
        print(f"\nincomplete: {type(exc).__name__}: {exc}", file=sys.stderr)

    done = [p for p in preds if p.actual_prompt]
    ledger = load_ledger()
    ledger["total_nano"] = spent.nano + run_cost.nano
    ledger.setdefault("runs", []).append({"kind": "validate", "cost_nano": run_cost.nano})
    save_ledger(ledger)

    if not done:
        print("no completed requests", file=sys.stderr)
        return 1

    prompt_err = [
        100 * abs(p.predicted_prompt - p.actual_prompt) / p.actual_prompt for p in done
    ]
    cached_err = [p.cached_abs_pct_error for p in done]

    OUT.write_text(json.dumps([asdict(p) for p in done], indent=2))
    print(f"\ncompleted {len(done)}/{len(preds)}, cost {run_cost}")
    print(f"prompt-token error : median {statistics.median(prompt_err):.2f}%  "
          f"max {max(prompt_err):.2f}%")
    print(f"cached-token error : median {statistics.median(cached_err):.2f}%  "
          f"max {max(cached_err):.2f}%")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
