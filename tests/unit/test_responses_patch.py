from __future__ import annotations

import jsonpatch
import pytest

from plap.responses.ingest.ingest import _apply_thread_patch
from plap.responses.ingest.models import Message, ToolCall
from plap.responses.ingest.patch import JSONPatch, diff


def test_diff_preserves_forward_list_move() -> None:
    source = {"items": ["a", "b", "c"]}
    target = {"items": ["b", "c", "a"]}

    patch = diff(source, target)

    assert patch == [{"op": "move", "from": "/items/0", "path": "/items/2"}]
    assert jsonpatch.apply_patch(source, patch, in_place=False) == target


def test_diff_preserves_backward_list_move_and_other_ops() -> None:
    source = {"items": ["a", "b", "c"], "flag": False, "meta": {"x": 1}}
    target = {"items": ["c", "a", "b"], "flag": True, "meta": {"x": 1, "y": 2}}

    patch = diff(source, target)

    assert {"op": "move", "from": "/items/2", "path": "/items/0"} in patch
    assert {"op": "replace", "path": "/flag", "value": True} in patch
    assert {"op": "add", "path": "/meta/y", "value": 2} in patch
    assert jsonpatch.apply_patch(source, patch, in_place=False) == target


def test_diff_uses_one_move_when_reordering_duplicate_values() -> None:
    source = {"items": [0, 0, 1]}
    target = {"items": [1, 0, 0]}

    patch = diff(source, target)

    assert patch == [{"op": "move", "from": "/items/2", "path": "/items/0"}]
    assert jsonpatch.apply_patch(source, patch, in_place=False) == target


def test_diff_returns_empty_patch_when_values_match() -> None:
    source = {"items": ["a", "b"], "flag": True}

    assert diff(source, source) == []


def test_diff_snapshots_equal_value_moved_between_object_members() -> None:
    source = {"foo": {"bar": "baz", "waldo": "fred"}, "qux": {"corge": "grault"}}
    target = {"foo": {"bar": "baz"}, "qux": {"corge": "grault", "thud": "fred"}}

    patch = diff(source, target)

    assert patch == [
        {"op": "remove", "path": "/foo/waldo"},
        {"op": "add", "path": "/qux/thud", "value": "fred"},
    ]
    assert jsonpatch.apply_patch(source, patch, in_place=False) == target


def test_diff_removes_temporary_array_element_without_rewriting_shifted_elements() -> None:
    source = [
        {"kind": "main"},
        {"kind": "phase"},
        {"kind": "call"},
        {"kind": "output"},
    ]
    target = [
        {"kind": "main"},
        {"kind": "call"},
        {"kind": "output"},
    ]

    patch = diff(source, target)

    assert patch == [{"op": "remove", "path": "/1"}]
    assert jsonpatch.apply_patch(source, patch, in_place=False) == target


def test_diff_copies_content_between_tool_outputs_as_a_snapshot() -> None:
    source = [
        {"role": "tool", "tool_call_id": "a", "content": "x"},
        {"role": "tool", "tool_call_id": "b", "content": "y"},
    ]
    target = [{"role": "tool", "tool_call_id": "b", "content": "x"}]

    patch = diff(source, target)

    assert patch == [
        {"op": "remove", "path": "/0"},
        {"op": "replace", "path": "/0/content", "value": "x"},
    ]
    replay_source = [
        {"role": "tool", "tool_call_id": "a", "content": "changed before replay"},
        {"role": "tool", "tool_call_id": "b", "content": "y"},
    ]
    assert jsonpatch.apply_patch(replay_source, patch, in_place=False) == target


def test_diff_reorders_and_edits_nested_memory_array() -> None:
    source = {
        "tasks": [
            {"name": "draft", "state": "open"},
            {"name": "review", "state": "open"},
        ]
    }
    target = {
        "tasks": [
            {"name": "review", "state": "open"},
            {"name": "draft", "state": "done"},
        ]
    }

    patch = diff(source, target)

    assert patch == [
        {"op": "move", "from": "/tasks/1", "path": "/tasks/0"},
        {"op": "replace", "path": "/tasks/1/state", "value": "done"},
    ]
    assert jsonpatch.apply_patch(source, patch, in_place=False) == target


def test_diff_combines_array_removal_move_and_snapshot_addition() -> None:
    source = {"items": ["a", "b", "c", "d"]}
    target = {"items": ["c", 1, "a", "d"]}

    patch = diff(source, target)

    assert patch == [
        {"op": "remove", "path": "/items/1"},
        {"op": "move", "from": "/items/1", "path": "/items/0"},
        {"op": "add", "path": "/items/1", "value": 1},
    ]
    assert jsonpatch.apply_patch(source, patch, in_place=False) == target


def _phase_thread() -> list[Message]:
    return [
        Message(role="user", content="main transcript"),
        Message(
            role="user",
            content="before_return",
            memory={"advisor": {"phase": "before_return"}},
        ),
        Message(
            role="assistant",
            tool_calls=[ToolCall(id="client_1", name="read_file", arguments="{}")],
        ),
        Message(
            role="tool",
            tool_call_id="client_1",
            content="old public output",
            memory={"seen": False},
        ),
    ]


def _completed_thread(
    *,
    tool_call_id: str = "client_1",
    content: str = "old public output",
    seen: bool | str = False,
) -> list[Message]:
    return [
        Message(role="user", content="main transcript"),
        Message(
            role="assistant",
            tool_calls=[ToolCall(id=tool_call_id, name="read_file", arguments="{}")],
        ),
        Message(
            role="tool",
            tool_call_id=tool_call_id,
            content=content,
            memory={"seen": seen},
        ),
        Message(
            role="assistant",
            tool_calls=[ToolCall(id="advise_1", name="advise", arguments='{"advice":""}')],
        ),
        Message(role="tool", tool_call_id="advise_1", content="0"),
    ]


def _thread_patch(source: list[Message], target: list[Message]) -> JSONPatch:
    return diff(
        [message.to_primitive() for message in source],
        [message.to_primitive() for message in target],
    )


@pytest.mark.parametrize(
    ("target", "expected_call_id", "expected_content", "expected_seen"),
    [
        (_completed_thread(), "client_1", "changed public output", "changed before replay"),
        (_completed_thread(seen=True), "client_1", "changed public output", True),
        (_completed_thread(tool_call_id="client_2"), "client_2", "changed public output", "changed before replay"),
        (_completed_thread(content="plugin rewrite"), "client_1", "plugin rewrite", "changed before replay"),
    ],
    ids=("structural-only", "memory-changed", "call-id-changed", "content-changed"),
)
def test_thread_patch_preserves_replay_values_unless_target_changes_them(
    target: list[Message],
    expected_call_id: str,
    expected_content: str,
    expected_seen: object,
) -> None:
    source = _phase_thread()
    patch = _thread_patch(source, target)
    assert _apply_thread_patch(source, patch, thread="advisor") == target

    replay_source = _phase_thread()
    replay_source[-1] = Message(
        role="tool",
        tool_call_id="client_1",
        content="changed public output",
        memory={"seen": "changed before replay"},
    )

    replayed = _apply_thread_patch(replay_source, patch, thread="advisor")
    tool_output = next(message for message in replayed if message.role == "tool" and message.tool_call_id.startswith("client_"))

    assert tool_output.tool_call_id == expected_call_id
    assert tool_output.content == expected_content
    assert tool_output.memory == {"seen": expected_seen}
