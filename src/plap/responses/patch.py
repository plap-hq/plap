from __future__ import annotations

from copy import deepcopy

import jsonpatch
from jsonpointer import resolve_pointer

type JSONValue = object
type JSONPatchOperation = dict[str, JSONValue]
type JSONPatch = list[JSONPatchOperation]


def _apply_single_operation(document: object, operation: JSONPatchOperation) -> object:
    return jsonpatch.apply_patch(document, [operation], in_place=False)


def _expand_move(operation: JSONPatchOperation, document: object) -> list[JSONPatchOperation]:
    from_path = operation.get("from")
    path = operation.get("path")
    if not isinstance(from_path, str):
        raise TypeError("move operation 'from' must be a string")
    if not isinstance(path, str):
        raise TypeError("move operation path must be a string")
    return [
        {"op": "remove", "path": from_path},
        {"op": "add", "path": path, "value": deepcopy(resolve_pointer(document, from_path))},
    ]


def expand_moves(patch: JSONPatch, source: object) -> JSONPatch:
    current = deepcopy(source)
    expanded: JSONPatch = []
    for raw_operation in patch:
        operation = deepcopy(raw_operation)
        if operation.get("op") == "move":
            expanded.extend(_expand_move(operation, current))
        else:
            expanded.append(operation)
        current = _apply_single_operation(current, operation)
    return expanded


def diff(source: object, target: object) -> JSONPatch:
    return expand_moves(jsonpatch.make_patch(source, target).patch, source)
