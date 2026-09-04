"""Epoch checkpointing: serialise a live agent loop without moving a byte.

An agent loop outlives the process running it. Restoring one is only safe if
the restored prompt is *byte-identical* to the prompt that was live before the
restart, because a prompt cache is a prefix cache: change one byte at position
`k` and everything from `k` onwards is re-billed at the cache-write rate. A
restore that renders the same information slightly differently costs the
entire prefix — the exact failure :mod:`context_slim.core` exists to prevent.

The rule that falls out of that, and the one this module is built around:

    **A checkpoint replays recorded decisions. It never recomputes them.**

That is not a stylistic preference. ``expiry.render_stub`` writes the token
count of the message it replaces into the stub's own text::

    [tool result elided: ~1,235 tokens, tail-first, pays back in 4 turns]

Re-deriving that stub during a restore measures the stub rather than the
original and writes ``~12 tokens`` instead — same code, same inputs, 57 bytes
of difference, and a cold prefix behind it. So :class:`Checkpoint` stores the
rendered stub *text* per index rather than the size and reason needed to
regenerate it. Replay cannot drift; regeneration can.

The same lesson has now cost this project three benchmark runs in two
different disguises (confounds 2 and 4 in ``METHODS.md``): state that is
recomputed inside a control loop does not reproduce the state that loop
actually ran with.

Checkpoints deliberately do **not** contain the conversation. The caller owns
its messages; a checkpoint owns the *decisions* taken against them, which is
what makes it small enough to write every epoch and safe to keep alongside a
conversation that is itself persisted elsewhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .ledger import Ledger
from .schemas import Message

__all__ = ["Checkpoint"]

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Checkpoint:
    """The decisions taken against one conversation, replayable byte-for-byte.

    ``stubbed`` maps a message index to the *exact rendered content* that
    replaced it. Storing the rendered text, rather than the inputs needed to
    re-render it, is what makes :meth:`restore` incapable of drifting.
    """

    anchor_index: int
    model: str
    turn: int
    stubbed: dict[int, str] = field(default_factory=dict)
    ledger: Ledger = field(default_factory=Ledger)
    version: int = SCHEMA_VERSION

    @staticmethod
    def restore(checkpoint: Checkpoint, messages: list[Message]) -> list[Message]:
        """Replay ``checkpoint`` onto the original ``messages``.

        ``messages`` is the conversation as it was *before* compaction — the
        caller's own copy, unmutated here. Every recorded stub is written back
        verbatim, so the result is byte-identical to the state that was live
        when the checkpoint was taken.
        """
        if checkpoint.version != SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint schema v{checkpoint.version} cannot be read by v{SCHEMA_VERSION}"
            )

        out = [dict(m) for m in messages]
        for index, content in checkpoint.stubbed.items():
            if not 0 <= index < len(out):
                raise ValueError(
                    f"checkpoint refers to message {index}, but the conversation "
                    f"has {len(out)} — it is not the one this checkpoint was taken against"
                )
            if index <= checkpoint.anchor_index:
                raise ValueError(
                    f"checkpoint would rewrite message {index}, inside the Anchor Zone "
                    f"(ends at {checkpoint.anchor_index}). Refusing: this is the prefix "
                    "the whole library exists to keep byte-stable"
                )
            out[index]["content"] = content
        return out

    def merge(self, other: Checkpoint) -> Checkpoint:
        """Fold an older checkpoint into this one, enforcing monotonicity.

        A cleared message never becomes uncleared. A non-monotonic policy that
        un-cleared previously cleared indices invalidated an entire benchmark
        run once (confound 2, ``METHODS.md``); allowing a restore to do the
        same thing quietly would reintroduce it through the back door.
        """
        if not set(other.stubbed) <= set(self.stubbed):
            missing = sorted(set(other.stubbed) - set(self.stubbed))
            raise ValueError(
                f"cleared set may only grow: merging would drop {missing}. "
                "A restore that unclears a message is confound 2 wearing a different hat"
            )
        merged = dict(other.stubbed)
        merged.update(self.stubbed)
        return Checkpoint(
            anchor_index=self.anchor_index,
            model=self.model,
            turn=max(self.turn, other.turn),
            stubbed=merged,
            ledger=self.ledger,
            version=self.version,
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "anchor_index": self.anchor_index,
                "model": self.model,
                "turn": self.turn,
                # JSON object keys are strings; indices are restored to int below.
                "stubbed": {str(k): v for k, v in sorted(self.stubbed.items())},
                "ledger": json.loads(self.ledger.to_json()),
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, raw: str) -> Checkpoint:
        data: dict[str, Any] = json.loads(raw)
        version = int(data.get("version", 0))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint schema v{version} cannot be read by v{SCHEMA_VERSION}"
            )
        return cls(
            anchor_index=int(data["anchor_index"]),
            model=str(data["model"]),
            turn=int(data["turn"]),
            stubbed={int(k): str(v) for k, v in data.get("stubbed", {}).items()},
            ledger=Ledger.from_json(json.dumps(data.get("ledger", {}))),
            version=version,
        )
