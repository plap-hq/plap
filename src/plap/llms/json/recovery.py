from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from plap.llms.json.schema import _validation_error_rule, compile_validator
from plap.llms.json.serde import decode_json_value_with_error

_IDENTIFIER_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_.$-]*")
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_VALUE_START_CHARS = frozenset("{[\"'+-.0123456789")
_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d*)?")
_SIMPLE_TOKEN_RE = re.compile(r"[A-Za-z0-9_.$-]+")


class _Match(StrEnum):
    MATCH = "match"
    UNKNOWN = "unknown"
    MISMATCH = "mismatch"


def _is_word_apostrophe(text: str, index: int) -> bool:
    if index <= 0 or index + 1 >= len(text):
        return False
    return text[index - 1].isalnum() and text[index + 1].isalnum()


def _skip_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _skip_ignorable(text: str, index: int) -> int:
    while True:
        index = _skip_whitespace(text, index)
        if text.startswith("//", index):
            index += 2
            while index < len(text) and text[index] != "\n":
                index += 1
            continue
        if text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end == -1:
                return len(text)
            index = end + 2
            continue
        return index


def _trim_to_first_container(text: str) -> str:
    index = _skip_whitespace(text, 0)
    if index >= len(text):
        return text

    quote: str | None = None
    while index < len(text):
        char = text[index]
        if quote is not None:
            if char == "\\":
                index = min(index + 2, len(text))
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {'"', "'"}:
            if char == "'" and _is_word_apostrophe(text, index):
                index += 1
                continue
            quote = char
            index += 1
            continue
        if char in "{[":
            return text[index:]
        index += 1
    return text


def _has_structural_container(text: str) -> bool:
    trimmed = _trim_to_first_container(text)
    stripped = text.lstrip()
    return (bool(trimmed) and bool(stripped) and trimmed != text) or (bool(stripped) and stripped[0] in "{[")


def _is_simple_top_level_token(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[0] in {'"', "'"}:
        return True
    if stripped[0] in "+-.0123456789":
        return True
    lowered = stripped.lower()
    if lowered in {"true", "false", "null", "none"}:
        return True
    return not any(char.isspace() for char in stripped)


def _schema_has_specific_root_keys(schema: dict[str, Any] | None) -> bool:
    return _schema_has_specific_keys(schema, None, ())


def _starts_with_schema_key(text: str, schema: dict[str, Any] | None, *, specific_only: bool = False) -> bool:
    index = _skip_ignorable(text, 0)
    if index >= len(text):
        return False
    char = text[index]
    if char in {'"', "'"}:
        quote = char
        index += 1
        start = index
        while index < len(text):
            current = text[index]
            if current == "\\":
                index = min(index + 2, len(text))
                continue
            if current == quote:
                key = text[start:index]
                index = _skip_ignorable(text, index + 1)
                return (
                        index < len(text)
                        and text[index] == ":"
                        and bool(key)
                        and (
                            _schema_accepts_specific_key(schema, None, (), key)
                            if specific_only
                            else _schema_accepts_key(schema, None, (), key)
                        )
                )
            index += 1
        return False
    match = _IDENTIFIER_RE.match(text, index)
    if match is None:
        return False
    key = match.group(0)
    index = _skip_ignorable(text, match.end())
    return index < len(text) and text[index] == ":" and (
        _schema_accepts_specific_key(schema, None, (), key) if specific_only else _schema_accepts_key(schema, None, (), key)
    )


def _is_safe_schema_key_prefix(prefix: str) -> bool:
    return not any(char in prefix for char in ",{}[]\"'")


def _trim_to_first_schema_key(text: str, schema: dict[str, Any] | None) -> str:
    index = _skip_ignorable(text, 0)
    quote: str | None = None
    while index < len(text):
        char = text[index]
        if quote is not None:
            if char == "\\":
                index = min(index + 2, len(text))
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {'"', "'"}:
            candidate = text[index:]
            if _starts_with_schema_key(candidate, schema, specific_only=True) and _is_safe_schema_key_prefix(text[:index]):
                return candidate
            quote = char
            index += 1
            continue
        if char.isalpha() or char == "_":
            candidate = text[index:]
            if _starts_with_schema_key(candidate, schema, specific_only=True) and _is_safe_schema_key_prefix(text[:index]):
                return candidate
        index += 1
    return text


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


def _decode_relaxed_string(content: str) -> str:
    result: list[str] = []
    index = 0
    while index < len(content):
        char = content[index]
        if char != "\\":
            result.append(char)
            index += 1
            continue
        if index + 1 >= len(content):
            result.append("\\")
            break
        next_char = content[index + 1]
        if next_char == "\n":
            index += 2
            continue
        if next_char == "\r":
            index += 3 if index + 2 < len(content) and content[index + 2] == "\n" else 2
            continue
        if next_char == "u":
            hex_digits = content[index + 2 : index + 6]
            if len(hex_digits) == 4 and all(digit in _HEX_DIGITS for digit in hex_digits):
                result.append(chr(int(hex_digits, 16)))
                index += 6
                continue
            result.append("\\")
            result.append("u")
            index += 2
            continue
        if next_char == "x":
            hex_digits = content[index + 2 : index + 4]
            if len(hex_digits) == 2 and all(digit in _HEX_DIGITS for digit in hex_digits):
                result.append(chr(int(hex_digits, 16)))
                index += 4
                continue
            result.append("\\")
            result.append("x")
            index += 2
            continue
        mapped = {
            '"': '"',
            "'": "'",
            "\\": "\\",
            "/": "/",
            "b": "\b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
        }.get(next_char)
        if mapped is not None:
            result.append(mapped)
            index += 2
            continue
        result.append("\\")
        result.append(next_char)
        index += 2
    return "".join(result)


def _strip_comments(text: str) -> str:
    result: list[str] = []
    index = 0
    quote: str | None = None
    while index < len(text):
        char = text[index]
        if quote is not None:
            result.append(char)
            if char == "\\" and index + 1 < len(text):
                result.append(text[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in {'"', "'"}:
            quote = char
            result.append(char)
            index += 1
            continue
        if char == "/" and index + 1 < len(text) and text[index + 1] == "/":
            index += 2
            while index < len(text) and text[index] != "\n":
                index += 1
            continue
        if char == "/" and index + 1 < len(text) and text[index + 1] == "*":
            index += 2
            while index + 1 < len(text) and not (text[index] == "*" and text[index + 1] == "/"):
                index += 1
            index = min(index + 2, len(text))
            continue
        result.append(char)
        index += 1
    return "".join(result)


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


def _resolve_schema_ref(root: dict[str, Any] | None, ref: str) -> dict[str, Any] | None:
    if root is None:
        return None
    if ref == "#":
        return root
    if not ref.startswith("#/"):
        return None
    current: Any = root
    for part in ref[2:].split("/"):
        resolved_part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or resolved_part not in current:
            return None
        current = current[resolved_part]
    return current if isinstance(current, dict) else None


def _merge_schema_hints(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key in (
        "type",
        "items",
        "prefixItems",
        "additionalItems",
        "additionalProperties",
        "contains",
        "unevaluatedItems",
        "unevaluatedProperties",
        "propertyNames",
    ):
        if key in overlay:
            merged[key] = overlay[key]

    for key in ("properties", "patternProperties"):
        base_mapping = merged.get(key)
        overlay_mapping = overlay.get(key)
        if isinstance(base_mapping, dict) and isinstance(overlay_mapping, dict):
            merged[key] = {**base_mapping, **overlay_mapping}
        elif key in overlay:
            merged[key] = overlay[key]

    for key, value in overlay.items():
        if key == "$ref" or key in merged:
            continue
        merged[key] = value
    return merged


def _merge_alternative_schema_hints(schemas: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    additional_properties = False
    for schema in schemas:
        merged = _merge_schema_hints(merged, schema)
        additional = schema.get("additionalProperties")
        if additional is True or isinstance(additional, dict):
            additional_properties = additional
    if additional_properties is not False:
        merged["additionalProperties"] = additional_properties
    return merged


def _schema_match_state(
    root: dict[str, Any] | None,
    schema: dict[str, Any] | None,
    value: Any,
    seen_refs: frozenset[str] = frozenset(),
    seen_nodes: frozenset[int] = frozenset(),
) -> _Match:
    node = _resolve_schema_node(root, schema, None, (), seen_refs, seen_nodes)
    if not isinstance(node, dict):
        return _Match.UNKNOWN
    try:
        validator = compile_validator(node)
    except Exception:
        return _Match.UNKNOWN
    errors = list(validator.iter_errors(value))
    if not errors:
        return _Match.MATCH
    soft_rules = {"required", "minItems", "minProperties", "contains", "dependentRequired"}
    if all(_validation_error_rule(error) in soft_rules for error in errors):
        return _Match.UNKNOWN
    return _Match.MISMATCH


def _resolve_schema_node(
    root: dict[str, Any] | None,
    schema: dict[str, Any] | None,
    contexts: dict[tuple[str | int, ...], Any] | None = None,
    path: tuple[str | int, ...] = (),
    seen_refs: frozenset[str] = frozenset(),
    seen_nodes: frozenset[int] = frozenset(),
) -> dict[str, Any] | None:
    current = schema
    while isinstance(current, dict):
        ref = current.get("$ref")
        if not isinstance(ref, str):
            break
        if ref in seen_refs:
            return current
        resolved = _resolve_schema_ref(root, ref)
        if resolved is None:
            return current
        siblings = {key: value for key, value in current.items() if key != "$ref"}
        current = _merge_schema_hints(resolved, siblings) if siblings else resolved
        seen_refs = frozenset((*seen_refs, ref))
    if not isinstance(current, dict):
        return None
    current_id = id(current)
    if current_id in seen_nodes:
        return current
    seen_nodes = frozenset((*seen_nodes, current_id))

    siblings = {key: value for key, value in current.items() if key not in {"allOf", "anyOf", "oneOf"}}
    all_of = current.get("allOf")
    if isinstance(all_of, list):
        merged: dict[str, Any] = {}
        for subschema in all_of:
            if not isinstance(subschema, dict):
                continue
            resolved = _resolve_schema_node(root, subschema, contexts, path, seen_refs, seen_nodes)
            if resolved is not None:
                merged = _merge_schema_hints(merged, resolved)
        current = _merge_schema_hints(merged, siblings)
    else:
        for key in ("anyOf", "oneOf"):
            alternatives = current.get(key)
            if not isinstance(alternatives, list):
                continue
            resolved_alternatives: list[dict[str, Any]] = []
            context_value = None if contexts is None else contexts.get(path)
            for subschema in alternatives:
                if not isinstance(subschema, dict):
                    continue
                resolved = _resolve_schema_node(root, subschema, contexts, path, seen_refs, seen_nodes)
                if resolved is None:
                    continue
                if context_value is not None and (
                    _schema_match_state(root, resolved, context_value, seen_refs, seen_nodes) == _Match.MISMATCH
                ):
                    continue
                resolved_alternatives.append(resolved)
            if resolved_alternatives:
                current = _merge_schema_hints(_merge_alternative_schema_hints(resolved_alternatives), siblings)
            break

    context_value = None if contexts is None else contexts.get(path)
    dependent_schemas = current.get("dependentSchemas")
    if isinstance(context_value, dict) and isinstance(dependent_schemas, dict):
        for key, subschema in dependent_schemas.items():
            if not isinstance(key, str) or key not in context_value or not isinstance(subschema, dict):
                continue
            resolved = _resolve_schema_node(root, subschema, contexts, path, seen_refs, seen_nodes)
            if resolved is not None:
                current = _merge_schema_hints(current, resolved)

    if_schema = current.get("if")
    if isinstance(context_value, dict) and isinstance(if_schema, dict):
        match_state = _schema_match_state(root, if_schema, context_value, seen_refs, seen_nodes)
        branch_key = "then" if match_state == _Match.MATCH else "else" if match_state == _Match.MISMATCH else None
        if branch_key is not None:
            branch = current.get(branch_key)
            if isinstance(branch, dict):
                resolved = _resolve_schema_node(root, branch, contexts, path, seen_refs, seen_nodes)
                if resolved is not None:
                    current = _merge_schema_hints(current, resolved)

    return current


def _pattern_matches_key(pattern: str, key: str) -> bool:
    try:
        return re.search(pattern, key) is not None
    except re.error:
        return False


def _property_name_matches(schema: dict[str, Any], key: str) -> bool:
    property_names = schema.get("propertyNames")
    if not isinstance(property_names, dict):
        return False
    pattern = property_names.get("pattern")
    if isinstance(pattern, str):
        return _pattern_matches_key(pattern, key)
    return False


def _schema_accepts_specific_key(
    root: dict[str, Any] | None,
    contexts: dict[tuple[str | int, ...], Any] | None,
    path: tuple[str | int, ...],
    key: str,
) -> bool:
    current = _schema_for_path(root, path, contexts)
    if not isinstance(current, dict):
        return False

    properties = current.get("properties")
    if isinstance(properties, dict) and key in properties:
        return True

    pattern_properties = current.get("patternProperties")
    if isinstance(pattern_properties, dict):
        for pattern in pattern_properties:
            if isinstance(pattern, str) and _pattern_matches_key(pattern, key):
                return True

    return _property_name_matches(current, key)


def _schema_has_specific_keys(
    root: dict[str, Any] | None,
    contexts: dict[tuple[str | int, ...], Any] | None,
    path: tuple[str | int, ...],
) -> bool:
    current = _schema_for_path(root, path, contexts)
    if not isinstance(current, dict):
        return False
    return (
        isinstance(current.get("properties"), dict)
        or isinstance(current.get("patternProperties"), dict)
        or isinstance(current.get("propertyNames"), dict)
    )


def _schema_for_object_key(
    root: dict[str, Any] | None,
    contexts: dict[tuple[str | int, ...], Any] | None,
    path: tuple[str | int, ...],
    schema: dict[str, Any],
    key: str,
) -> dict[str, Any] | None:
    property_names_match = _property_name_matches(schema, key)
    property_names_present = isinstance(schema.get("propertyNames"), dict)

    properties = schema.get("properties")
    if isinstance(properties, dict) and key in properties:
        next_schema = properties[key]
        return _resolve_schema_node(root, next_schema if isinstance(next_schema, dict) else None, contexts, (*path, key))

    pattern_properties = schema.get("patternProperties")
    if isinstance(pattern_properties, dict):
        for pattern, pattern_schema in pattern_properties.items():
            if isinstance(pattern, str) and _pattern_matches_key(pattern, key):
                return _resolve_schema_node(root, pattern_schema if isinstance(pattern_schema, dict) else None, contexts, (*path, key))

    additional = schema.get("additionalProperties")
    if isinstance(additional, dict) and (not property_names_present or property_names_match):
        return _resolve_schema_node(root, additional, contexts, (*path, key))

    unevaluated = schema.get("unevaluatedProperties")
    if isinstance(unevaluated, dict) and (not property_names_present or property_names_match):
        return _resolve_schema_node(root, unevaluated, contexts, (*path, key))

    if property_names_match:
        if isinstance(additional, dict):
            return _resolve_schema_node(root, additional, contexts, (*path, key))
        if isinstance(unevaluated, dict):
            return _resolve_schema_node(root, unevaluated, contexts, (*path, key))
    return None


def _schema_accepts_key(
    root: dict[str, Any] | None,
    contexts: dict[tuple[str | int, ...], Any] | None,
    path: tuple[str | int, ...],
    key: str,
) -> bool:
    if _schema_accepts_specific_key(root, contexts, path, key):
        return True

    current = _schema_for_path(root, path, contexts)
    if not isinstance(current, dict):
        return False

    property_names_match = _property_name_matches(current, key)
    property_names_present = isinstance(current.get("propertyNames"), dict)

    additional = current.get("additionalProperties")
    if additional is False:
        unevaluated = current.get("unevaluatedProperties")
        return isinstance(unevaluated, dict) and property_names_match
    if (additional is True or isinstance(additional, dict)) and (not property_names_present or property_names_match):
        return True

    unevaluated = current.get("unevaluatedProperties")
    if (unevaluated is True or isinstance(unevaluated, dict)) and (not property_names_present or property_names_match):
        return True

    if property_names_match:
        return True

    return (
        isinstance(current.get("properties"), dict)
        or isinstance(current.get("patternProperties"), dict)
    )


def _schema_for_path(
    schema: dict[str, Any] | None,
    path: tuple[str | int, ...],
    contexts: dict[tuple[str | int, ...], Any] | None = None,
) -> dict[str, Any] | None:
    current = _resolve_schema_node(schema, schema, contexts, ())
    current_path: tuple[str | int, ...] = ()
    for segment in path:
        if current is None:
            return None
        if isinstance(segment, str):
            current = _schema_for_object_key(schema, contexts, current_path, current, segment)
            current_path = (*current_path, segment)
            continue
        prefix_items = current.get("prefixItems")
        if isinstance(prefix_items, list) and 0 <= segment < len(prefix_items):
            next_schema = prefix_items[segment]
            current = _resolve_schema_node(
                schema,
                next_schema if isinstance(next_schema, dict) else None,
                contexts,
                (*current_path, segment),
            )
            current_path = (*current_path, segment)
            continue
        items = current.get("items")
        if isinstance(items, dict):
            current = _resolve_schema_node(schema, items, contexts, (*current_path, segment))
            current_path = (*current_path, segment)
            continue
        if isinstance(items, list) and 0 <= segment < len(items):
            next_schema = items[segment]
            current = _resolve_schema_node(
                schema,
                next_schema if isinstance(next_schema, dict) else None,
                contexts,
                (*current_path, segment),
            )
            current_path = (*current_path, segment)
            continue
        additional_items = current.get("additionalItems")
        if isinstance(additional_items, dict):
            current = _resolve_schema_node(schema, additional_items, contexts, (*current_path, segment))
            current_path = (*current_path, segment)
            continue
        unevaluated_items = current.get("unevaluatedItems")
        if isinstance(unevaluated_items, dict):
            current = _resolve_schema_node(schema, unevaluated_items, contexts, (*current_path, segment))
            current_path = (*current_path, segment)
            continue
        contains = current.get("contains")
        if isinstance(contains, dict):
            current = _resolve_schema_node(schema, contains, contexts, (*current_path, segment))
            current_path = (*current_path, segment)
            continue
        current = None
    return current if isinstance(current, dict) else None


def _schema_expects_object(
    schema: dict[str, Any] | None,
    path: tuple[str | int, ...],
    contexts: dict[tuple[str | int, ...], Any] | None = None,
) -> bool:
    current = _schema_for_path(schema, path, contexts)
    if not isinstance(current, dict):
        return False
    schema_type = current.get("type")
    if schema_type == "object":
        return True
    if isinstance(schema_type, list) and "object" in schema_type:
        return True
    return (
        isinstance(current.get("properties"), dict)
        or isinstance(current.get("patternProperties"), dict)
        or isinstance(current.get("propertyNames"), dict)
    )


def _parity_normalize_quoted_expression(value: str) -> str:
    if len(value) < 2:
        return _decode_relaxed_string(value)
    first = value[0]
    last = value[-1]
    if first not in {'"', "'"} or first != last:
        return _decode_relaxed_string(value)
    middle = value[1:-1]
    internal_quote_count = 0
    for index, char in enumerate(middle):
        if char != first:
            continue
        if index > 0 and middle[index - 1] == "\\":
            continue
        internal_quote_count += 1
    if internal_quote_count % 2 == 1 or internal_quote_count == 0:
        return _decode_relaxed_string(middle)
    return _decode_relaxed_string(value)


class Outcome(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Result:
    outcome: Outcome
    value: Any | None


class _Parser:
    def __init__(self, text: str, *, partial: bool, schema: dict[str, Any] | None) -> None:
        self._text = text
        self._index = 0
        self._partial = partial
        self._schema = schema
        self._contexts: dict[tuple[str | int, ...], Any] = {}
        self._incomplete = False
        self._rejected = False

    def _peek(self) -> str | None:
        if self._index >= len(self._text):
            return None
        return self._text[self._index]

    def _advance(self) -> str | None:
        char = self._peek()
        if char is not None:
            self._index += 1
        return char

    def _skip_whitespace(self) -> None:
        original = self._index
        self._index = _skip_ignorable(self._text, self._index)
        if self._index == len(self._text) and original < len(self._text) and self._text.startswith("/*", original):
            self._incomplete = True

    def _skip_to_value_start(self) -> None:
        self._skip_whitespace()
        while self._index < len(self._text):
            char = self._text[self._index]
            if char in _VALUE_START_CHARS or char.isalpha() or char == "_":
                return
            self._index += 1
            self._skip_whitespace()

    def _looks_like_next_key(self, path: tuple[str | int, ...], index: int) -> bool:
        index = _skip_ignorable(self._text, index)
        if index >= len(self._text):
            return False
        char = self._text[index]
        if char in {'"', "'"}:
            quote = char
            index += 1
            start = index
            while index < len(self._text):
                current = self._text[index]
                if current == "\\":
                    index = min(index + 2, len(self._text))
                    continue
                if current == quote:
                    key = self._text[start:index]
                    index = _skip_ignorable(self._text, index + 1)
                    return (
                        index < len(self._text)
                        and self._text[index] == ":"
                        and bool(key)
                        and _schema_accepts_key(self._schema, self._contexts, path, key)
                    )
                index += 1
            return False
        match = _IDENTIFIER_RE.match(self._text, index)
        if match is None:
            return False
        key = match.group(0)
        if not _schema_accepts_key(self._schema, self._contexts, path, key):
            return False
        index = _skip_ignorable(self._text, match.end())
        return index < len(self._text) and self._text[index] == ":"

    def _quote_closes_value(self, path: tuple[str | int, ...], next_index: int, *, object_value: bool) -> bool:
        next_index = _skip_ignorable(self._text, next_index)
        if next_index >= len(self._text):
            return True
        next_char = self._text[next_index]
        if next_char in ",}]":
            return True
        return object_value and self._looks_like_next_key(path[:-1], next_index)

    def _find_next_unescaped_quote(self, quote: str, index: int) -> int | None:
        while index < len(self._text):
            char = self._text[index]
            if char == "\\":
                index += 2
                continue
            if char == quote:
                return index
            index += 1
        return None

    def _should_capture_string_expression(self, quote: str, segment_start: int) -> bool:
        next_quote = self._find_next_unescaped_quote(quote, self._index)
        if next_quote is None:
            return False
        between = self._text[self._index:next_quote]
        if not between.strip():
            return False
        if between != between.strip():
            return True
        return _SIMPLE_TOKEN_RE.fullmatch(between.strip()) is None

    def _parse_key_string(self) -> str:
        quote = self._advance()
        if quote is None:
            self._incomplete = True
            return ""
        parts: list[str] = []
        while True:
            char = self._advance()
            if char is None:
                self._incomplete = True
                if self._partial:
                    return "".join(parts)
                return _decode_relaxed_string("".join(parts))
            if char == "\\":
                next_char = self._advance()
                if next_char is None:
                    self._incomplete = True
                    parts.append("\\")
                    if self._partial:
                        return "".join(parts)
                    return _decode_relaxed_string("".join(parts))
                parts.append(char)
                parts.append(next_char)
                continue
            if char == quote:
                return _decode_relaxed_string("".join(parts))
            parts.append(char)

    def _parse_string_value(self, path: tuple[str | int, ...], *, object_value: bool) -> str:
        start_index = self._index
        quote = self._advance()
        if quote is None:
            self._incomplete = True
            return ""
        parts: list[str] = []
        saw_nonclosing_quote = False
        while True:
            char = self._advance()
            if char is None:
                self._incomplete = True
                if self._partial:
                    return "".join(parts)
                return _decode_relaxed_string("".join(parts))
            if char == "\\":
                next_char = self._advance()
                if next_char is None:
                    self._incomplete = True
                    parts.append("\\")
                    if self._partial:
                        return "".join(parts)
                    return _decode_relaxed_string("".join(parts))
                parts.append(char)
                parts.append(next_char)
                continue
            if char == quote:
                if self._quote_closes_value(path, self._index, object_value=object_value):
                    return _decode_relaxed_string("".join(parts))
                if not saw_nonclosing_quote and self._should_capture_string_expression(quote, start_index):
                    self._index = start_index
                    return _parity_normalize_quoted_expression(
                        self._capture_greedy_value(path, object_value=object_value)
                    )
                saw_nonclosing_quote = True
                parts.append(char)
                continue
            parts.append(char)

    def _parse_bare_key(self) -> str | None:
        match = _IDENTIFIER_RE.match(self._text, self._index)
        if match is None:
            return None
        self._index = match.end()
        return match.group(0)

    def _parse_numeric_key(self) -> str | None:
        start = self._index
        if self._peek() == "-":
            self._index += 1
        saw_digit = False
        while self._index < len(self._text):
            char = self._text[self._index]
            if char.isdigit() or char in ".eE+-":
                saw_digit = saw_digit or char.isdigit()
                self._index += 1
                continue
            break
        if not saw_digit:
            self._index = start
            return None
        return self._text[start:self._index]

    def _capture_greedy_value(self, path: tuple[str | int, ...], *, object_value: bool) -> str:
        start = self._index
        stack: list[str] = []
        quote: str | None = None
        while self._index < len(self._text):
            char = self._text[self._index]
            if quote is not None:
                if char == "\\":
                    self._index = min(self._index + 2, len(self._text))
                    continue
                if char == quote:
                    quote = None
                self._index += 1
                continue
            if not stack:
                if char in ",}]":
                    break
                if object_value and self._looks_like_next_key(path[:-1], self._index):
                    break
            if char in {'"', "'"}:
                quote = char
                self._index += 1
                continue
            if char in "([{":
                stack.append(char)
                self._index += 1
                continue
            if char in ")]}":
                if stack:
                    top = stack[-1]
                    if (top, char) in {("(", ")"), ("[", "]"), ("{", "}")}:
                        stack.pop()
                        self._index += 1
                        continue
                if char in "]}":
                    break
            self._index += 1
        return self._text[start:self._index].strip()

    def _capture_number_token(self) -> str:
        start = self._index
        while self._index < len(self._text):
            char = self._text[self._index]
            if char.isspace() or char in ',]}"\'[{()':
                break
            self._index += 1
        return self._text[start:self._index]

    def _parse_number_or_token(self, path: tuple[str | int, ...], *, object_value: bool) -> Any:
        char = self._peek()
        if char is not None and char in "+-.0123456789":
            token = self._capture_number_token()
            if token:
                lowered = token.lower()
                if lowered == "true":
                    return True
                if lowered == "false":
                    return False
                if lowered in {"null", "none"}:
                    return None
                hex_number = _normalize_hex_number(token)
                if hex_number is not None:
                    return hex_number
                number = _normalize_number(token)
                if number is not None:
                    return number
                return token
        token = self._capture_greedy_value(path, object_value=object_value)
        lowered = token.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"null", "none"}:
            return None
        number = _normalize_number(token)
        if number is not None:
            return number
        return token

    def _parse_key(self) -> str | None:
        self._skip_whitespace()
        char = self._peek()
        if char is None:
            self._incomplete = True
            return None
        if char in {'"', "'"}:
            return self._parse_key_string()
        if char.isdigit() or char == "-":
            return self._parse_numeric_key()
        return self._parse_bare_key()

    def _parse_array(self, path: tuple[str | int, ...]) -> list[Any]:
        items: list[Any] = []
        self._advance()
        item_index = 0
        self._contexts[path] = items
        try:
            while True:
                self._skip_whitespace()
                char = self._peek()
                if char is None:
                    self._incomplete = True
                    return items
                if char == "]":
                    self._advance()
                    return items
                if char == ",":
                    self._advance()
                    continue
                if char == "}":
                    self._incomplete = True
                    return items
                value_start = self._index
                items.append(self._parse_value((*path, item_index), object_value=False))
                if self._index == value_start:
                    self._rejected = True
                    return items
                item_index += 1
                self._skip_whitespace()
                if self._peek() == ",":
                    self._advance()
                    continue
                if self._peek() == "]":
                    self._advance()
                    return items
                if self._peek() is None:
                    self._incomplete = True
                    return items
                if self._peek() == "}":
                    self._incomplete = True
                    return items
        finally:
            self._contexts.pop(path, None)

    def _parse_object(self, path: tuple[str | int, ...], *, implicit: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {}
        if not implicit:
            self._advance()
        self._contexts[path] = value
        try:
            while True:
                self._skip_whitespace()
                char = self._peek()
                if char is None:
                    self._incomplete = True
                    return value
                if char == "}":
                    self._advance()
                    return value
                if implicit and char == "]":
                    return value
                if char == ",":
                    self._advance()
                    continue
                key_start = self._index
                key = self._parse_key()
                if key is None:
                    if self._peek() is None:
                        self._incomplete = True
                        return value
                    self._advance()
                    continue
                self._skip_whitespace()
                if self._peek() != ":":
                    value[key] = None
                    if self._peek() is None:
                        self._incomplete = True
                        return value
                    if self._peek() == "}":
                        continue
                    if self._index == key_start:
                        self._advance()
                    continue
                if implicit and key in value:
                    self._index = key_start
                    return value
                self._advance()
                self._skip_whitespace()
                if self._peek() is None:
                    self._incomplete = True
                    value[key] = None
                    return value
                if self._peek() in ",}":
                    value[key] = None
                    continue
                value_start = self._index
                value[key] = self._parse_value((*path, key), object_value=True)
                if self._index == value_start:
                    self._rejected = True
                    return value
                self._skip_whitespace()
                if self._peek() == ",":
                    self._advance()
                    continue
                if self._peek() == "}":
                    self._advance()
                    return value
                if implicit and self._looks_like_next_key(path, self._index):
                    continue
                if implicit and self._peek() == "]":
                    return value
                if implicit and self._peek() is None:
                    self._incomplete = True
                    return value
                if self._peek() is None:
                    self._incomplete = True
                    return value
                if implicit:
                    return value
        finally:
            self._contexts.pop(path, None)

    def _parse_value(self, path: tuple[str | int, ...], *, object_value: bool) -> Any:
        self._skip_whitespace()
        char = self._peek()
        if char is None:
            self._incomplete = True
            return None
        if char == "{":
            return self._parse_object(path)
        if _schema_expects_object(self._schema, path, self._contexts) and self._looks_like_next_key(path, self._index):
            return self._parse_object(path, implicit=True)
        if char == "[":
            return self._parse_array(path)
        if char in {'"', "'"}:
            return self._parse_string_value(path, object_value=object_value)
        if char.isalpha() or char in "+-.0123456789_@$":
            return self._parse_number_or_token(path, object_value=object_value)
        return self._capture_greedy_value(path, object_value=object_value)

    def parse(self) -> Any:
        self._skip_to_value_start()
        if self._peek() is None:
            self._rejected = True
            return None
        return self._parse_value((), object_value=False)


def recover(text: str, *, partial: bool, schema: dict[str, Any] | None = None) -> Result:
    value, error = decode_json_value_with_error(text)
    if error is None:
        return Result(outcome=Outcome.COMPLETE, value=value)

    prepared = text
    if not partial:
        has_container = _has_structural_container(prepared)
        prepared = _trim_to_first_container(prepared)
        prepared = _strip_comments(prepared)
        if not has_container and not _schema_expects_object(schema, ()) and not _is_simple_top_level_token(prepared):
            return Result(outcome=Outcome.REJECTED, value=None)
        specific_root_keys = _schema_has_specific_root_keys(schema)
        if not has_container and _schema_expects_object(schema, ()) and not _starts_with_schema_key(
            prepared,
            schema,
            specific_only=specific_root_keys,
        ):
            prepared = _trim_to_first_schema_key(prepared, schema)
            if not _starts_with_schema_key(prepared, schema, specific_only=specific_root_keys):
                return Result(outcome=Outcome.REJECTED, value=None)

    parser = _Parser(prepared, partial=partial, schema=schema)
    value = parser.parse()
    if parser._rejected:
        return Result(outcome=Outcome.REJECTED, value=None)
    if partial and parser._incomplete:
        return Result(outcome=Outcome.INCOMPLETE, value=value)
    return Result(outcome=Outcome.COMPLETE, value=value)


__all__ = ["Outcome", "Result", "recover"]
