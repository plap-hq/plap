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


def test_normalize_const_scalar_strings() -> None:
    schema = {
        "type": "object",
        "properties": {
            "answer": {"const": 31},
            "enabled": {"const": True},
            "value": {"const": None},
        },
    }

    assert normalize({"answer": "0x1f", "enabled": "TRUE", "value": "none"}, schema=schema) == {
        "answer": 31,
        "enabled": True,
        "value": None,
    }


def test_normalize_const_string_canonicalization() -> None:
    schema = {
        "type": "object",
        "properties": {
            "status": {"const": "active"},
        },
    }

    assert normalize({"status": " ACTIVE "}, schema=schema) == {"status": "active"}


def test_normalize_conservative_string_enum_match() -> None:
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["active", "inactive"]},
        },
    }

    assert normalize({"status": " ACTIVE "}, schema=schema) == {"status": "active"}


def test_normalize_non_string_enum_values_from_strings() -> None:
    schema = {
        "type": "object",
        "properties": {
            "count": {"enum": [1, 2, 3]},
            "enabled": {"enum": [True, False]},
            "ratio": {"enum": [1.5]},
        },
    }

    assert normalize({"count": "01", "enabled": "false", "ratio": "1.5"}, schema=schema) == {
        "count": 1,
        "enabled": False,
        "ratio": 1.5,
    }


def test_normalize_treats_json_equivalent_numeric_enum_values_as_one_candidate() -> None:
    schema = {
        "type": "object",
        "properties": {
            "count": {"enum": [1, 1.0]},
        },
    }

    assert normalize({"count": "1"}, schema=schema) == {"count": 1}


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


def test_normalize_does_not_parse_stringified_container_targets() -> None:
    schema = {
        "type": "object",
        "properties": {
            "payload": {"enum": [{"a": 1}]},
        },
    }

    assert normalize({"payload": '{"a":1}'}, schema=schema) == {"payload": '{"a":1}'}


def test_normalize_wraps_scalar_under_array_schema() -> None:
    schema = {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}}}}

    assert normalize({"ids": "image-AAAA-BBBB-CCCC-DDDD"}, schema=schema) == {"ids": ["image-AAAA-BBBB-CCCC-DDDD"]}


def test_normalize_wraps_normalized_scalar_under_array_item_schema() -> None:
    schema = {"type": "object", "properties": {"numbers": {"type": "array", "items": {"type": "integer"}}}}

    assert normalize({"numbers": "0x1f"}, schema=schema) == {"numbers": [31]}


def test_normalize_wraps_object_under_array_schema() -> None:
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "required": ["n"],
                    "additionalProperties": False,
                },
            }
        },
    }

    assert normalize({"items": {"n": "5"}}, schema=schema) == {"items": [{"n": 5}]}


def test_normalize_wraps_list_under_nested_array_schema() -> None:
    schema = {
        "type": "object",
        "properties": {
            "matrix": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            }
        },
    }

    assert normalize({"matrix": ["a", "b"]}, schema=schema) == {"matrix": [["a", "b"]]}


def test_normalize_wraps_normalized_list_under_nested_array_schema() -> None:
    schema = {
        "type": "object",
        "properties": {
            "matrix": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
            }
        },
    }

    assert normalize({"matrix": ["1", "2"]}, schema=schema) == {"matrix": [[1, 2]]}


def test_normalize_wraps_scalar_through_distinct_nested_array_schemas() -> None:
    schema = {
        "type": "object",
        "properties": {
            "matrix": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            }
        },
    }

    assert normalize({"matrix": "a"}, schema=schema) == {"matrix": [["a"]]}


def test_normalize_does_not_rewrap_recursive_array_schema_forever() -> None:
    schema = {"type": "array", "items": {"$ref": "#"}}

    assert normalize("a", schema=schema) == "a"


def test_normalize_preserves_already_valid_object_branch_before_singleton_array_branch() -> None:
    item_schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
        "additionalProperties": False,
    }
    schema = {
        "type": "object",
        "properties": {
            "payload": {
                "anyOf": [
                    {"type": "array", "items": item_schema},
                    item_schema,
                ]
            }
        },
    }

    assert normalize({"payload": {"n": 5}}, schema=schema) == {"payload": {"n": 5}}


def test_normalize_does_not_wrap_scalar_when_singleton_array_is_invalid() -> None:
    schema = {
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
            }
        },
    }

    assert normalize({"ids": "image-AAAA-BBBB-CCCC-DDDD"}, schema=schema) == {"ids": "image-AAAA-BBBB-CCCC-DDDD"}


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


def test_normalize_preserves_valid_union_string_values() -> None:
    schema = {
        "type": "object",
        "properties": {
            "n": {"type": ["string", "integer"]},
        },
    }

    assert normalize({"n": "01"}, schema=schema) == {"n": "01"}
