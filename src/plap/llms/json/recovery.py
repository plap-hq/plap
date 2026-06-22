# NOTE: - Infinity / NaN still stay strings

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from plap.llms.json.schema import (
    _schema_accepts_key,
    _schema_accepts_specific_key,
    _schema_expects_object,
    _schema_has_specific_keys,
)
from plap.llms.json.serde import decode_json_value_with_error

_IDENTIFIER_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_.$-]*")
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_VALUE_START_CHARS = frozenset("{[\"'+-.0123456789")
_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d*)?")
_SIMPLE_TOKEN_RE = re.compile(r"[A-Za-z0-9_.$-]+")


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
                        _schema_accepts_specific_key(schema, None, (), key) if specific_only else _schema_accepts_key(schema, None, (), key)
                    )
                )
            index += 1
        return False
    match = _IDENTIFIER_RE.match(text, index)
    if match is None:
        return False
    key = match.group(0)
    index = _skip_ignorable(text, match.end())
    return (
        index < len(text)
        and text[index] == ":"
        and (_schema_accepts_specific_key(schema, None, (), key) if specific_only else _schema_accepts_key(schema, None, (), key))
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
        between = self._text[self._index : next_quote]
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
                    return _parity_normalize_quoted_expression(self._capture_greedy_value(path, object_value=object_value))
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
        return self._text[start : self._index]

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
        return self._text[start : self._index].strip()

    def _capture_number_token(self) -> str:
        start = self._index
        while self._index < len(self._text):
            char = self._text[self._index]
            if char.isspace() or char in ",]}\"'[{()":
                break
            self._index += 1
        return self._text[start : self._index]

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
        if (
            not has_container
            and _schema_expects_object(schema, ())
            and not _starts_with_schema_key(
                prepared,
                schema,
                specific_only=specific_root_keys,
            )
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
