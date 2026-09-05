"""context-slim — prune your LLM agent's context without destroying your prompt cache.

The public surface is deliberately four functions, implemented in :mod:`context_slim.core`:

``doctor``   find cache pathologies that cost money silently
``plan``     decide what is worth pruning — pure, no I/O, no mutation
``apply``    execute an approved plan
``simulate`` project cost over N future turns without calling any API

``plan`` and ``apply`` are separate so that "don't prune" is an ordinary
outcome you can inspect, rather than an exception or a silent no-op.
:class:`~context_slim.core.CacheAlignedContext` wraps the same four functions
for callers who want the Anchor Zone boundary computed and threaded through
automatically instead of passed by hand.
"""

from __future__ import annotations

from .cache.rates import ModelRates
from .cache.rates import get as get_rates
from .core import CacheAlignedContext, apply, doctor, plan, simulate
from .ledger import DischargeReason, Ledger, detect_discharge_windows
from .policy import break_even_turns, estimate_horizon
from .presets import PRESETS, Preset
from .presets import get as get_preset
from .providers import adapter_for, detect
from .schemas import (
    BreakEven,
    Candidate,
    CostReport,
    Decision,
    Diagnostic,
    Message,
    Money,
    PrunePlan,
    Verdict,
)

__version__ = "0.1.1"

__all__ = [
    "PRESETS",
    "BreakEven",
    "CacheAlignedContext",
    "Candidate",
    "CostReport",
    "Decision",
    "Diagnostic",
    "DischargeReason",
    "Ledger",
    "Message",
    "ModelRates",
    "Money",
    "Preset",
    "PrunePlan",
    "Verdict",
    "__version__",
    "adapter_for",
    "apply",
    "break_even_turns",
    "detect",
    "detect_discharge_windows",
    "doctor",
    "estimate_horizon",
    "get_preset",
    "get_rates",
    "plan",
    "simulate",
]
