from __future__ import annotations

from typing import Any

import msgspec
from json_repair import repair_json


class ToolArgumentsInvalidJSONError(ValueError):
    pass


class ToolArgumentsNotObjectError(ValueError):
    pass


def stringify_tool_arguments_value(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return msgspec.json.encode(value).decode()


def _decoded_tool_arguments_value(arguments: str) -> object:
    try:
        return msgspec.json.decode(arguments.encode())
    except msgspec.DecodeError as strict_exc:
        try:
            repaired = repair_json(arguments, skip_json_loads=True)
        except Exception as exc:
            raise ToolArgumentsInvalidJSONError("tool arguments must be valid JSON") from exc
        try:
            return msgspec.json.decode(repaired.encode())
        except msgspec.DecodeError:
            raise ToolArgumentsInvalidJSONError("tool arguments must be valid JSON") from strict_exc


def parse_tool_arguments_value(arguments: str) -> object:
    return _decoded_tool_arguments_value(arguments)


def normalize_tool_arguments_text(arguments: str) -> str:
    return msgspec.json.encode(parse_tool_arguments_value(arguments), order="deterministic").decode()


def normalize_tool_arguments_text_or_original(arguments: str) -> str:
    try:
        return normalize_tool_arguments_text(arguments)
    except ToolArgumentsInvalidJSONError:
        return arguments


def parse_tool_arguments_object(arguments: str) -> dict[str, Any]:
    value = parse_tool_arguments_value(arguments)
    if not isinstance(value, dict):
        raise ToolArgumentsNotObjectError("tool arguments must be a JSON object")
    return value
