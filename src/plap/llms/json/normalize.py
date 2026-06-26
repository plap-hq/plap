from __future__ import annotations

import re
from typing import Any

import blake3
import msgspec

from plap.llms.json.schema import _schema_for_path, compile_validator

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_ARRAY_HINT_KEYS = frozenset(("items", "prefixItems", "additionalItems", "contains", "unevaluatedItems"))
_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d*)?$")
_MISSING = object()


def _validated(schema: dict[str, Any] | None, value: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    try:
        compile_validator(schema).validate(value)
    except Exception:
        return False
    return True


def _digest(value: Any) -> bytes:
    try:
        encoded = msgspec.json.encode(value, order="deterministic")
    except Exception:
        encoded = repr(value).encode("utf-8", "backslashreplace")
    return blake3.blake3(encoded).digest()


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


def _json_scalar_key(value: Any) -> tuple[str, Any] | None:
    if value is None:
        return "null", None
    if isinstance(value, bool):
        return "boolean", value
    if isinstance(value, int | float):
        return "number", value
    if isinstance(value, str):
        return "string", value
    return None


def _append_unique_scalar_candidate(candidates: list[Any], candidate: Any) -> None:
    key = _json_scalar_key(candidate)
    if key is None:
        return
    if any(_json_scalar_key(existing) == key for existing in candidates):
        return
    candidates.append(candidate)


def _schema_string_targets(schema: dict[str, Any]) -> list[str]:
    targets: list[str] = []
    const_value = schema.get("const", _MISSING)
    if isinstance(const_value, str) and const_value not in targets:
        targets.append(const_value)

    enum_values = schema.get("enum")
    if not isinstance(enum_values, list):
        return targets
    for item in enum_values:
        if isinstance(item, str) and item not in targets:
            targets.append(item)
    return targets


def _schema_non_string_scalar_targets(schema: dict[str, Any]) -> list[Any]:
    targets: list[Any] = []
    const_value = schema.get("const", _MISSING)
    if _json_scalar_key(const_value) is not None and not isinstance(const_value, str):
        _append_unique_scalar_candidate(targets, const_value)

    enum_values = schema.get("enum")
    if not isinstance(enum_values, list):
        return targets
    for item in enum_values:
        if _json_scalar_key(item) is None or isinstance(item, str):
            continue
        _append_unique_scalar_candidate(targets, item)
    return targets


def _normalized_string_candidates(value: str, targets: list[str]) -> list[str]:
    if not targets:
        return []
    stripped = value.strip()
    exact = [item for item in targets if item == stripped]
    if len(exact) == 1:
        return exact
    folded = [item for item in targets if item.casefold() == stripped.casefold()]
    if len(folded) == 1:
        return folded
    return []


def _normalized_non_string_scalar_target_candidate(value: str, target: Any) -> Any:
    stripped = value.strip()
    if not stripped:
        return _MISSING
    if isinstance(target, bool):
        lowered = stripped.casefold()
        if lowered in {"true", "false"} and (lowered == "true") == target:
            return target
        return _MISSING
    if target is None:
        if stripped.casefold() in {"null", "none"}:
            return None
        return _MISSING
    if isinstance(target, int):
        hex_number = _normalize_hex_number(stripped)
        if hex_number == target:
            return target
        relaxed_number = _normalize_number(stripped)
        if isinstance(relaxed_number, int) and relaxed_number == target:
            return target
        if isinstance(relaxed_number, float) and relaxed_number.is_integer() and int(relaxed_number) == target:
            return target
        return _MISSING
    if isinstance(target, float):
        hex_number = _normalize_hex_number(stripped)
        if hex_number is not None and float(hex_number) == target:
            return target
        relaxed_number = _normalize_number(stripped)
        if relaxed_number is not None and float(relaxed_number) == target:
            return target
    return _MISSING


def _normalized_schema_target_candidates(value: str, schema: dict[str, Any]) -> list[Any]:
    candidates: list[Any] = []
    for candidate in _normalized_string_candidates(value, _schema_string_targets(schema)):
        _append_unique_scalar_candidate(candidates, candidate)
    for target in _schema_non_string_scalar_targets(schema):
        candidate = _normalized_non_string_scalar_target_candidate(value, target)
        if candidate is _MISSING:
            continue
        _append_unique_scalar_candidate(candidates, candidate)
    return candidates


def _normalized_scalar_candidates(value: str, schema: dict[str, Any]) -> list[Any]:
    candidates: list[Any] = []
    for candidate in _normalized_schema_target_candidates(value, schema):
        _append_unique_scalar_candidate(candidates, candidate)

    stripped = value.strip()
    if not stripped:
        return candidates

    hex_number = _normalize_hex_number(stripped)
    relaxed_number = _normalize_number(stripped)
    for schema_type in _schema_types(schema):
        if schema_type == "integer":
            if hex_number is not None:
                _append_unique_scalar_candidate(candidates, hex_number)
            if isinstance(relaxed_number, int):
                _append_unique_scalar_candidate(candidates, relaxed_number)
            elif isinstance(relaxed_number, float) and relaxed_number.is_integer():
                _append_unique_scalar_candidate(candidates, int(relaxed_number))
        elif schema_type == "number":
            if hex_number is not None:
                _append_unique_scalar_candidate(candidates, hex_number)
            if relaxed_number is not None:
                _append_unique_scalar_candidate(candidates, relaxed_number)
        elif schema_type == "boolean" and stripped.casefold() in {"true", "false"}:
            _append_unique_scalar_candidate(candidates, stripped.casefold() == "true")
        elif schema_type == "null" and stripped.casefold() in {"null", "none"}:
            _append_unique_scalar_candidate(candidates, None)
    return candidates


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
    wraps: frozenset[tuple[bytes, bytes]],
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
                wraps=wraps,
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
    wraps: frozenset[tuple[bytes, bytes]],
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
                    wraps=wraps,
                )
            )
    finally:
        contexts.pop(path, None)
    return normalized


def _schema_targets_array(schema: dict[str, Any]) -> bool:
    if "array" in _schema_types(schema):
        return True
    if any(key in schema for key in _ARRAY_HINT_KEYS):
        return True
    const_value = schema.get("const", _MISSING)
    if isinstance(const_value, list):
        return True
    enum_values = schema.get("enum")
    return isinstance(enum_values, list) and any(isinstance(item, list) for item in enum_values)


def _try_array_wrap(
    value: Any,
    *,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    path: tuple[str | int, ...],
    contexts: dict[tuple[str | int, ...], Any],
    wraps: frozenset[tuple[bytes, bytes]],
) -> Any:
    if not _schema_targets_array(schema):
        return _MISSING
    key = (_digest(schema), _digest(value))
    if key in wraps:
        return _MISSING
    normalized = _normalize_array(
        [value],
        root_schema=root_schema,
        path=path,
        contexts=contexts,
        wraps=wraps | {key},
    )
    if _validated(schema, normalized):
        return normalized
    return _MISSING


def _normalize_node(
    value: Any,
    *,
    root_schema: dict[str, Any],
    path: tuple[str | int, ...],
    contexts: dict[tuple[str | int, ...], Any],
    wraps: frozenset[tuple[bytes, bytes]],
) -> Any:
    schema = _schema_for_path(root_schema, path, contexts)
    if not isinstance(schema, dict):
        return value
    if _validated(schema, value):
        return value
    wrapped = _try_array_wrap(
        value,
        schema=schema,
        root_schema=root_schema,
        path=path,
        contexts=contexts,
        wraps=wraps,
    )
    if wrapped is not _MISSING:
        return wrapped
    if isinstance(value, list):
        return _normalize_array(
            value,
            root_schema=root_schema,
            path=path,
            contexts=contexts,
            wraps=wraps,
        )
    if isinstance(value, dict):
        return _normalize_object(
            value,
            root_schema=root_schema,
            path=path,
            contexts=contexts,
            wraps=wraps,
        )
    return _normalize_scalar(value, schema)


def normalize(value: Any, *, schema: dict[str, Any] | None) -> Any:
    if not isinstance(schema, dict):
        return value
    if _validated(schema, value):
        return value
    return _normalize_node(value, root_schema=schema, path=(), contexts={}, wraps=frozenset())


__all__ = ["normalize"]
