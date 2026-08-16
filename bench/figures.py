"""Generate figures from a completed kill-gate run.

Reads committed results only — never re-runs the experiment, so figures are
reproducible without spending money.

    python -m bench.figures

Writes PDF (vector, for the paper) and PNG (for posts) to bench/figures/.
"""

from __future__ import annotations

import collections
import json
import pathlib
import statistics
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results" / "killgate.json"
OUT = pathlib.Path(__file__).parent / "figures"

# Greyscale-safe: every series is separable by marker and linestyle alone, so
# the figures survive a black-and-white print and colour-blind readers.
STYLE = {
    "no_prune": ("#1F4E79", "o", "-", "no pruning"),
    "oldest_first": ("#8C2F39", "s", "--", "prune oldest-first"),
    "tail_first": ("#3B7EA1", "^", "-.", "prune newest-first"),
}
ORDER = ["no_prune", "oldest_first", "tail_first"]

mpl.rcParams.update(
    {
        "figure.figsize": (5.2, 3.2),
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,  # embed TrueType, not Type-3
    }
)


def load() -> list[dict]:
    if not RESULTS.exists():
        sys.exit(f"no results at {RESULTS} — run `python -m bench.killgate` first")
    recs = [r for r in json.loads(RESULTS.read_text())["records"] if r["prompt_tokens"]]
    if not recs:
        sys.exit("results file has no completed records")
    return recs


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def fig_decay(recs: list[dict]) -> None:
    """Cache hit rate against turn index — the mechanism figure.

    Plots the **median** of the 5 arms, not the mean. At most turns the arms
    agree exactly (sd = 0), but a handful of turns contain a single-arm total
    cache miss - one arm returning ~0% while the other four are normal, giving
    sd > 30. Those are transient provider-side evictions, not treatment
    effects; they appear in ``no_prune`` too, which does no pruning at all. The
    mean lets one dropout drag a point down 20 points and invent a "decay" that
    is not there. The median is robust to them and shows the real shape: a step
    function, not a gradual bleed.
    """
    fig, ax = plt.subplots()
    for cond in ORDER:
        colour, marker, ls, label = STYLE[cond]
        by: dict[int, list[float]] = collections.defaultdict(list)
        for r in recs:
            if r["condition"].startswith(cond + "#"):
                by[r["turn"]].append(100 * r["cached_tokens"] / r["prompt_tokens"])
        turns = sorted(by)
        ax.plot(
            turns,
            [statistics.median(by[t]) for t in turns],
            color=colour, marker=marker, ls=ls, ms=3.5, lw=1.3, label=label,
        )
    ax.set_xlabel("Turn")
    ax.set_ylabel("Cache hit rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(
        "Pruning steps cache hits down; where you cut decides when",
        fontsize=9,
    )
    ax.annotate(
        "pruning starts", xy=(6, 78), xytext=(7.6, 45), fontsize=7,
        arrowprops={"arrowstyle": "->", "lw": 0.7},
    )
    ax.annotate(
        "newest-first holds a\nstable prefix 7 turns longer",
        xy=(11, 91), xytext=(11.4, 30), fontsize=7,
        arrowprops={"arrowstyle": "->", "lw": 0.7},
    )
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "fig2_cache_decay")


def fig_divergence(recs: list[dict]) -> None:
    """Tokens sent vs dollars paid — the two axes order the conditions oppositely."""
    tok: dict[str, int] = collections.defaultdict(int)
    cost: dict[str, int] = collections.defaultdict(int)
    arms: dict[str, set[str]] = collections.defaultdict(set)
    for r in recs:
        c = r["condition"].split("#")[0]
        tok[c] += r["prompt_tokens"]
        cost[c] += r["cost_nano"]
        arms[c].add(r["condition"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 3.0))
    labels = [STYLE[c][3].replace("prune ", "") for c in ORDER]
    colours = [STYLE[c][0] for c in ORDER]

    a1.bar(labels, [tok[c] / 1000 for c in ORDER], color=colours, width=0.55)
    a1.set_ylabel("Tokens sent (thousands)")
    a1.set_title("Tokens: pruning wins", fontsize=9)

    a2.bar(labels, [cost[c] / 1e9 / len(arms[c]) for c in ORDER], color=colours, width=0.55)
    a2.set_ylabel("Cost per arm (USD)")
    a2.set_title("Dollars: pruning loses", fontsize=9)

    for ax in (a1, a2):
        ax.tick_params(axis="x", labelsize=8, rotation=12)
    fig.suptitle("20.7% fewer tokens, 33–59% more money", fontsize=10, y=1.02)
    save(fig, "fig1_divergence")


def fig_price_per_token(recs: list[dict]) -> None:
    """Why the divergence happens: pruning converts cheap tokens into dear ones.

    A cached token bills at $0.02/Mtok and an uncached one at $0.25/Mtok - a
    12.5x spread - so the mix matters more than the count.
    """
    tok: dict[str, int] = collections.defaultdict(int)
    cac: dict[str, int] = collections.defaultdict(int)
    cost: dict[str, int] = collections.defaultdict(int)
    for r in recs:
        c = r["condition"].split("#")[0]
        tok[c] += r["prompt_tokens"]
        cac[c] += r["cached_tokens"]
        cost[c] += r["cost_nano"]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.0))
    labels = [STYLE[c][3].replace("prune ", "") for c in ORDER]
    cached = [cac[c] / 1000 for c in ORDER]
    uncached = [(tok[c] - cac[c]) / 1000 for c in ORDER]

    a1.bar(labels, cached, color="#C6D9E8", width=0.55, label="cached  @ $0.02/Mtok")
    a1.bar(labels, uncached, bottom=cached, color="#8C2F39", width=0.55,
           label="uncached @ $0.25/Mtok")
    a1.set_ylabel("Tokens (thousands)")
    a1.set_title("Pruning shrinks the bar but grows the dear part", fontsize=9)
    a1.legend(fontsize=7, loc="upper right")

    a2.bar(labels, [cost[c] / tok[c] * 1000 for c in ORDER], color=[STYLE[c][0] for c in ORDER],
           width=0.55)
    a2.set_ylabel("Effective price ($/Mtok)")
    a2.set_title("Pruning doubles the price of every token", fontsize=9)

    for ax in (a1, a2):
        ax.tick_params(axis="x", labelsize=8, rotation=12)
    save(fig, "fig3_price_per_token")


if __name__ == "__main__":
    data = load()
    print(f"figures from {len(data)} records:")
    fig_divergence(data)
    fig_decay(data)
    fig_price_per_token(data)
