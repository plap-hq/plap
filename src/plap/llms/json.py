from __future__ import annotations

from typing import Any

import msgspec


def decode_json_value(arguments: str) -> Any:
    return msgspec.json.decode(arguments.encode())


def decode_json_value_with_error(arguments: str) -> tuple[Any | None, msgspec.DecodeError | None]:
    try:
        return decode_json_value(arguments), None
    except msgspec.DecodeError as exc:
        return None, exc


def decode_json_object_or_none(arguments: str) -> dict[str, Any] | None:
    value, _ = decode_json_value_with_error(arguments)
    if not isinstance(value, dict):
        return None
    return value


def decode_json_object_with_error(arguments: str) -> tuple[dict[str, Any] | None, msgspec.DecodeError | None]:
    value, error = decode_json_value_with_error(arguments)
    if error is not None or not isinstance(value, dict):
        return None, error
    return value, None


def schema_property_keys(schema: dict[str, Any] | None) -> list[str]:
    if not isinstance(schema, dict):
        return []
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return []
    return sorted(str(key) for key in properties)


__all__ = [
    "decode_json_object_or_none",
    "decode_json_object_with_error",
    "decode_json_value",
    "decode_json_value_with_error",
    "schema_property_keys",
]
