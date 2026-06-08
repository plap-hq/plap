from __future__ import annotations

from plap.llms.json import normalize


def test_normalize_exact_integer_string_under_integer_schema() -> None:
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}

    assert normalize({"n": "5"}, schema=schema) == {"n": 5}


def test_normalize_exact_number_string_under_number_schema() -> None:
    schema = {"type": "object", "properties": {"n": {"type": "number"}}}

    assert normalize({"n": "3.14"}, schema=schema) == {"n": 3.14}


def test_normalize_relaxed_numeric_strings_under_numeric_schemas() -> None:
    int_schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    number_schema = {"type": "object", "properties": {"n": {"type": "number"}}}

    assert normalize({"n": "01"}, schema=int_schema) == {"n": 1}
    assert normalize({"n": "1e3"}, schema=int_schema) == {"n": 1000}
    assert normalize({"n": "1e"}, schema=number_schema) == {"n": 1.0}
    assert normalize({"n": "+.5"}, schema=number_schema) == {"n": 0.5}
    assert normalize({"n": "0x1f"}, schema=int_schema) == {"n": 31}


def test_normalize_exact_boolean_and_null_strings() -> None:
    schema = {
        "type": "object",
        "properties": {
            "enabled": {"type": "boolean"},
            "value": {"type": "null"},
        },
    }

    assert normalize({"enabled": "TRUE", "value": "null"}, schema=schema) == {"enabled": True, "value": None}


def test_normalize_conservative_string_enum_match() -> None:
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["active", "inactive"]},
        },
    }

    assert normalize({"status": " ACTIVE "}, schema=schema) == {"status": "active"}


def test_normalize_does_not_extract_noisy_numbers_or_boolean_synonyms() -> None:
    schema = {
        "type": "object",
        "properties": {
            "count": {"type": "integer"},
            "enabled": {"type": "boolean"},
        },
    }

    assert normalize({"count": "5 items", "enabled": "ON"}, schema=schema) == {"count": "5 items", "enabled": "ON"}


def test_normalize_does_not_normalize_invalid_numeric_like_strings() -> None:
    schema = {"type": "object", "properties": {"n": {"type": "number"}}}

    assert normalize({"n": "1..2"}, schema=schema) == {"n": "1..2"}


def test_normalize_uses_dynamic_schema_branching() -> None:
    schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
        },
        "if": {
            "properties": {
                "kind": {"const": "user"},
            },
            "required": ["kind"],
        },
        "then": {
            "properties": {
                "age": {"type": "integer"},
            },
        },
        "else": {
            "properties": {
                "title": {"type": "string"},
            },
        },
    }

    assert normalize({"kind": "user", "age": "5"}, schema=schema) == {"kind": "user", "age": 5}
    assert normalize({"kind": "org", "title": "Acme"}, schema=schema) == {"kind": "org", "title": "Acme"}


def test_normalize_preserves_already_valid_values() -> None:
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": "integer"},
            "status": {"type": "string", "enum": ["active", "inactive"]},
        },
        "required": ["n", "status"],
    }
    value = {"n": 5, "status": "active"}

    assert normalize(value, schema=schema) == value
