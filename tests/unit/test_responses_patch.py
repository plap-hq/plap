from __future__ import annotations

import jsonpatch

from plap.responses.ingest.patch import diff, expand_moves


def test_diff_expands_forward_list_move() -> None:
    source = {"items": ["a", "b", "c"]}
    target = {"items": ["b", "c", "a"]}

    patch = diff(source, target)

    assert patch == [
        {"op": "remove", "path": "/items/0"},
        {"op": "add", "path": "/items/2", "value": "a"},
    ]
    assert jsonpatch.apply_patch(source, patch, in_place=False) == target


def test_diff_expands_backward_list_move_and_preserves_other_ops() -> None:
    source = {"items": ["a", "b", "c"], "flag": False, "meta": {"x": 1}}
    target = {"items": ["c", "a", "b"], "flag": True, "meta": {"x": 1, "y": 2}}

    patch = diff(source, target)

    assert all(operation["op"] != "move" for operation in patch)
    assert {"op": "remove", "path": "/items/2"} in patch
    assert {"op": "add", "path": "/items/0", "value": "c"} in patch
    assert {"op": "replace", "path": "/flag", "value": True} in patch
    assert {"op": "add", "path": "/meta/y", "value": 2} in patch
    assert jsonpatch.apply_patch(source, patch, in_place=False) == target


def test_diff_returns_empty_patch_when_values_match() -> None:
    source = {"items": ["a", "b"], "flag": True}

    assert diff(source, source) == []


def test_expand_moves_handles_nested_object_move_directly() -> None:
    source = {"foo": {"bar": "baz", "waldo": "fred"}, "qux": {"corge": "grault"}}
    patch = [{"op": "move", "from": "/foo/waldo", "path": "/qux/thud"}]

    expanded = expand_moves(patch, source)

    assert expanded == [
        {"op": "remove", "path": "/foo/waldo"},
        {"op": "add", "path": "/qux/thud", "value": "fred"},
    ]
    assert jsonpatch.apply_patch(source, expanded, in_place=False) == {
        "foo": {"bar": "baz"},
        "qux": {"corge": "grault", "thud": "fred"},
    }
