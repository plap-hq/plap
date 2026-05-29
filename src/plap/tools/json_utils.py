from __future__ import annotations

from typing import Any

import msgspec
from json_repair import repair_json


class JSONInvalidError(ValueError):
    pass


class JSONNotObjectError(ValueError):
    pass


def parse_json_value_with_repair(value: str) -> object:
    try:
        return msgspec.json.decode(value.encode())
    except msgspec.DecodeError as strict_exc:
        try:
            repaired = repair_json(value, skip_json_loads=True)
        except Exception as exc:
            raise JSONInvalidError("value must be valid JSON") from exc
        try:
            return msgspec.json.decode(repaired.encode())
        except msgspec.DecodeError:
            raise JSONInvalidError("value must be valid JSON") from strict_exc


def normalize_json_text_with_repair(value: str) -> str:
    return msgspec.json.encode(parse_json_value_with_repair(value), order="deterministic").decode()


def normalize_json_text_with_repair_or_original(value: str) -> str:
    try:
        return normalize_json_text_with_repair(value)
    except JSONInvalidError:
        return value


def parse_json_object_with_repair(value: str) -> dict[str, Any]:
    decoded = parse_json_value_with_repair(value)
    if not isinstance(decoded, dict):
        raise JSONNotObjectError("value must be a JSON object")
    return decoded
