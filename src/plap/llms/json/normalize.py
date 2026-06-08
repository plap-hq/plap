from __future__ import annotations

import re
from typing import Any

from plap.llms.json.schema import _schema_for_path, compile_validator

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d*)?$")


def _validated(schema: dict[str, Any] | None, value: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    try:
        compile_validator(schema).validate(value)
    except Exception:
        return False
    return True


def _schema_types(schema: dict[str, Any]) -> tuple[str, ...]:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return (schema_type,)
    if isinstance(schema_type, list):
        return tuple(item for item in schema_type if isinstance(item, str))
    return ()


def _normalize_hex_number(token: str) -> int | None:
    sign = 1
    digits = token
    if digits.startswith("+"):
        digits = digits[1:]
    elif digits.startswith("-"):
        sign = -1
        digits = digits[1:]
    if not digits.lower().startswith("0x"):
        return None
    digits = digits[2:]
    if not digits or any(char not in _HEX_DIGITS for char in digits):
        return None
    return sign * int(digits, 16)


def _normalize_number(token: str) -> int | float | None:
    if _NUMBER_RE.fullmatch(token) is None:
        return None
    normalized = token
    if normalized.startswith("+."):
        normalized = f"0{normalized[1:]}"
    elif normalized.startswith("-."):
        normalized = f"-0{normalized[1:]}"
    elif normalized.startswith("."):
        normalized = f"0{normalized}"
    if normalized.endswith("."):
        normalized = f"{normalized}0"
    if normalized.endswith(("e", "E", "e+", "e-", "E+", "E-")):
        normalized = f"{normalized}0"
    if any(char in normalized for char in ".eE"):
        try:
            return float(normalized)
        except ValueError:
            return None
    try:
        return int(normalized)
    except ValueError:
        return None


def _normalized_enum_candidates(value: str, schema: dict[str, Any]) -> list[str]:
    enum_values = schema.get("enum")
    if not isinstance(enum_values, list) or not all(isinstance(item, str) for item in enum_values):
        return []
    stripped = value.strip()
    exact = [item for item in enum_values if item == stripped]
    if len(exact) == 1:
        return exact
    folded = [item for item in enum_values if item.casefold() == stripped.casefold()]
    if len(folded) == 1:
        return folded
    return []


def _normalized_scalar_candidates(value: str, schema: dict[str, Any]) -> list[Any]:
    stripped = value.strip()
    if not stripped:
        return _normalized_enum_candidates(value, schema)

    candidates: list[Any] = []
    candidates.extend(_normalized_enum_candidates(value, schema))
    hex_number = _normalize_hex_number(stripped)
    relaxed_number = _normalize_number(stripped)
    for schema_type in _schema_types(schema):
        if schema_type == "integer":
            if hex_number is not None:
                candidates.append(hex_number)
            if isinstance(relaxed_number, int):
                candidates.append(relaxed_number)
            elif isinstance(relaxed_number, float) and relaxed_number.is_integer():
                candidates.append(int(relaxed_number))
        elif schema_type == "number":
            if hex_number is not None:
                candidates.append(hex_number)
            if relaxed_number is not None:
                candidates.append(relaxed_number)
        elif schema_type == "boolean" and stripped.casefold() in {"true", "false"}:
            candidates.append(stripped.casefold() == "true")
        elif schema_type == "null" and stripped.casefold() in {"null", "none"}:
            candidates.append(None)

    unique: list[Any] = []
    for candidate in candidates:
        if candidate == value or candidate in unique:
            continue
        unique.append(candidate)
    return unique


def _normalize_scalar(value: Any, schema: dict[str, Any] | None) -> Any:
    if not isinstance(value, str) or not isinstance(schema, dict):
        return value
    validated = [candidate for candidate in _normalized_scalar_candidates(value, schema) if _validated(schema, candidate)]
    if len(validated) == 1:
        return validated[0]
    return value


def _normalize_object(
    value: dict[str, Any],
    *,
    root_schema: dict[str, Any],
    path: tuple[str | int, ...],
    contexts: dict[tuple[str | int, ...], Any],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    contexts[path] = normalized
    try:
        for key, item in value.items():
            normalized[key] = _normalize_node(
                item,
                root_schema=root_schema,
                path=(*path, key),
                contexts=contexts,
            )
    finally:
        contexts.pop(path, None)
    return normalized


def _normalize_array(
    value: list[Any],
    *,
    root_schema: dict[str, Any],
    path: tuple[str | int, ...],
    contexts: dict[tuple[str | int, ...], Any],
) -> list[Any]:
    normalized: list[Any] = []
    contexts[path] = normalized
    try:
        for index, item in enumerate(value):
            normalized.append(
                _normalize_node(
                    item,
                    root_schema=root_schema,
                    path=(*path, index),
                    contexts=contexts,
                )
            )
    finally:
        contexts.pop(path, None)
    return normalized


def _normalize_node(
    value: Any,
    *,
    root_schema: dict[str, Any],
    path: tuple[str | int, ...],
    contexts: dict[tuple[str | int, ...], Any],
) -> Any:
    schema = _schema_for_path(root_schema, path, contexts)
    if not isinstance(schema, dict):
        return value
    if _validated(schema, value):
        return value
    if isinstance(value, dict):
        return _normalize_object(value, root_schema=root_schema, path=path, contexts=contexts)
    if isinstance(value, list):
        return _normalize_array(value, root_schema=root_schema, path=path, contexts=contexts)
    return _normalize_scalar(value, schema)


def normalize(value: Any, *, schema: dict[str, Any] | None) -> Any:
    if not isinstance(schema, dict):
        return value
    if _validated(schema, value):
        return value
    return _normalize_node(value, root_schema=schema, path=(), contexts={})


__all__ = ["normalize"]
