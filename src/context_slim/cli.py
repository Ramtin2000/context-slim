"""Command line interface.

``doctor`` is the wedge: it exits non-zero on a detected cache pathology, so it
can sit in CI and fail a build when an agent's prompt cache is silently broken.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import __version__, apply, doctor, plan, simulate
from ._types import Message
from .presets import PRESETS

_SEVERITY_EXIT = {"error": 1, "warning": 0, "info": 0}


def _load(path: str) -> list[Message]:
    if path == "-":
        raw = sys.stdin.read()
    else:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
    data = json.loads(raw)
    if isinstance(data, dict) and "messages" in data:
        data = data["messages"]
    if not isinstance(data, list):
        raise SystemExit("expected a JSON list of messages, or an object with a 'messages' key")
    return data


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="context-slim", description=__doc__)
    p.add_argument("--version", action="version", version=f"context-slim {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("doctor", "plan", "simulate", "apply"):
        s = sub.add_parser(name)
        s.add_argument("file", help="JSON file of messages, or '-' for stdin")
        s.add_argument("--model", default="openai/gpt-5.6-luna")
        if name != "doctor":
            s.add_argument("--horizon", type=int, default=None)
            s.add_argument("--order", default=None, choices=["tail_first", "oldest_first"])
            s.add_argument(
                "--preset",
                default="balanced",
                choices=sorted(PRESETS),
                help="named stance on how much cache damage is acceptable",
            )

    args = p.parse_args(argv)
    messages = _load(args.file)

    if args.cmd == "doctor":
        worst = 0
        for d in doctor(messages, model=args.model):
            mark = {"error": "✗", "warning": "⚠", "info": "·"}[d.severity]
            cost = f"  (~{d.est_cost_per_turn}/turn)" if d.est_cost_per_turn.nano else ""
            print(f"{mark} [{d.code}] {d.message}{cost}")
            worst = max(worst, _SEVERITY_EXIT[d.severity])
        return worst

    if args.cmd == "simulate":
        turns = args.horizon if args.horizon is not None else PRESETS[args.preset].horizon
        print(simulate(messages, model=args.model, turns=turns, order=args.order))
        return 0

    pl = plan(
        messages,
        model=args.model,
        preset=args.preset,
        horizon=args.horizon,
        order=args.order,
    )
    if args.cmd == "plan":
        for v in pl.verdicts:
            print(f"{v.decision.value:<7} msg {v.candidate.index:<4} {v.reason}")
        print(f"\nprojected: {pl.projected_saving} over {pl.horizon} turns")
        return 0

    out, report = apply(messages, pl)
    json.dump(out, sys.stdout, indent=2)
    print(f"\n{report}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
