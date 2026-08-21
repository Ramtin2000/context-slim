"""JSON trimming must never lose a key path or emit invalid JSON."""

from __future__ import annotations

import base64
import json
import os

from context_slim.ops.json_ast import detect_json_spans, trim_json, trim_json_text


def _paths(obj: object, prefix: str = "") -> set[str]:
    if isinstance(obj, dict):
        out: set[str] = set()
        for k, v in obj.items():
            out |= {f"{prefix}.{k}"} | _paths(v, f"{prefix}.{k}")
        return out
    return set()


def test_every_key_path_survives() -> None:
    src = {"a": {"b": {"c": 1}}, "d": [{"e": 2}], "f": "x" * 9000}
    assert _paths(src) <= _paths(trim_json(src))


def test_output_is_always_valid_json() -> None:
    src = {"xs": [{"k": i, "v": "y" * 500} for i in range(300)], "n": None, "t": True}
    json.loads(json.dumps(trim_json(src)))


def test_homogeneous_array_is_sampled_not_deleted() -> None:
    out = trim_json({"xs": [{"a": 1} for _ in range(200)]})["xs"]
    assert len(out) < 200
    assert any("more items" in x for x in out if isinstance(x, str))


def test_heterogeneous_array_keeps_one_of_each_shape() -> None:
    """A variant vanishing entirely would tell the model the shape is impossible."""
    src = {"xs": [{"a": 1}] * 5 + [{"b": 2}] * 5 + [{"c": 3}] * 5}
    out = trim_json(src, array_head=1, array_tail=1)["xs"]
    shapes = {tuple(sorted(x)) for x in out if isinstance(x, dict)}
    assert shapes == {("a",), ("b",), ("c",)}


def test_deep_nesting_terminates() -> None:
    obj: object = "leaf"
    for _ in range(500):
        obj = {"n": obj}
    assert isinstance(trim_json(obj, max_depth=6), dict)


def test_base64_blob_is_stubbed() -> None:
    blob = base64.b64encode(os.urandom(400)).decode()
    assert trim_json({"img": blob})["img"].startswith("<blob:")


def test_repeated_char_run_is_not_a_blob() -> None:
    """Zero entropy means it compresses; it is filler, not payload."""
    assert not trim_json({"x": "A" * 400})["x"].startswith("<blob:")


def test_long_string_keeps_both_ends() -> None:
    """The head identifies a path; the tail of a trace is where the error is."""
    src = "/start/" + "z" * 5000 + "/end.py"
    out = trim_json({"p": src}, max_chars=80)["p"]
    assert out.startswith("/start/")
    assert out.endswith("/end.py")


def test_prose_is_untouched_and_costs_no_parse() -> None:
    assert detect_json_spans("no json in here at all") == []
    text = "plain prose, nothing structured"
    assert trim_json_text(text) == text


def test_json_embedded_in_prose_is_trimmed_in_place() -> None:
    out = trim_json_text('before {"xs": [1,2,3,4,5,6,7,8,9]} after')
    assert out.startswith("before ") and out.endswith(" after")
    json.loads(out[len("before ") : -len(" after")])
