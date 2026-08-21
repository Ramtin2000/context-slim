"""The CLI is the wedge: `doctor` is meant to run in CI, so exit codes matter."""

from __future__ import annotations

import json
import pathlib

import pytest

from context_slim.cli import main

LOOP: list[dict[str, object]] = [{"role": "system", "content": "S" * 32_000}]
for _i in range(10):
    LOOP += [
        {"role": "user", "content": f"step {_i}"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": f"c{_i}", "type": "function",
                         "function": {"name": "inspect", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": f"c{_i}", "content": "RESULT " * 400},
    ]


@pytest.fixture
def loop_file(tmp_path: pathlib.Path) -> str:
    p = tmp_path / "loop.json"
    p.write_text(json.dumps(LOOP))
    return str(p)


def test_plan_prints_a_verdict_per_candidate(
    loop_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["plan", loop_file]) == 0
    out = capsys.readouterr().out
    assert any(word in out for word in ("PLAN", "DEFER", "REFUSE"))


@pytest.mark.parametrize("preset", ["cache-preserving", "balanced", "aggressive"])
def test_every_preset_runs(loop_file: str, preset: str) -> None:
    assert main(["plan", loop_file, "--preset", preset]) == 0


def test_refusal_explains_itself_in_dollars(
    loop_file: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A refusal the user cannot act on is just a silent no-op."""
    main(["plan", loop_file, "--preset", "cache-preserving"])
    out = capsys.readouterr().out
    assert "REFUSE" in out
    assert "W/S" in out and "$" in out


def test_simulate_reports_a_cost(loop_file: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["simulate", loop_file]) == 0
    assert "$" in capsys.readouterr().out


def test_doctor_runs_clean_on_a_healthy_loop(loop_file: str) -> None:
    assert main(["doctor", loop_file]) == 0


def test_reads_from_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`cat log | context-slim plan -` should compose like any other filter."""
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(LOOP)))
    assert main(["plan", "-"]) == 0
    assert capsys.readouterr().out


def test_unknown_preset_is_rejected_by_argparse(loop_file: str) -> None:
    with pytest.raises(SystemExit):
        main(["plan", loop_file, "--preset", "reckless"])


ANTHROPIC_OVERRUN: list[dict[str, object]] = [
    {"role": "system", "content": "S" * 40_000, "cache_control": {"type": "ephemeral"}},
    *[{"role": "user", "content": f"turn {i} " + "x" * 400} for i in range(30)],
]


def test_doctor_exits_nonzero_on_a_pathology(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """This is the CI wedge: a silent cache miss has to fail a build."""
    p = tmp_path / "overrun.json"
    p.write_text(json.dumps(ANTHROPIC_OVERRUN))
    code = main(["doctor", str(p), "--model", "anthropic/claude-opus-5"])
    out = capsys.readouterr().out
    assert code == 1
    assert "lookback-overrun" in out
    assert "/turn" in out  # the cost of ignoring it, not just the fact of it
