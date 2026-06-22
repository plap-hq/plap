from __future__ import annotations

import json

import pytest

from plap.llms.json import Outcome, decode_json_object_or_none, recover

_COMMAND_SCHEMA = {
    "type": "object",
    "properties": {"command": {"type": "string"}},
    "required": ["command"],
}


def _valid_multiline_command() -> str:
    return json.dumps({"command": "x\nrollback: (registry: Registry, snapshot: EntitySnapshot) => void;"})


def _valid_multiline_import_command() -> str:
    return json.dumps(
        {
            "command": (
                "x\nimport type { Registry, EntitySnapshot } from y\nrollback: (registry: Registry, snapshot: EntitySnapshot) => void;"
            )
        }
    )


def _valid_heredoc_command() -> str:
    return json.dumps(
        {
            "command": (
                "cat <<'EOF' > x\nimport type { Registry, EntitySnapshot } from y\n"
                "rollback: (registry: Registry, snapshot: EntitySnapshot) => void;\nEOF"
            )
        }
    )


def _invalid_multiline_command() -> str:
    return '{"command":"x\nrollback: (registry: Registry, snapshot: EntitySnapshot) => void;"'


def _invalid_multiline_import_command() -> str:
    return (
        '{"command":"x\nimport type { Registry, EntitySnapshot } from y\nrollback: (registry: Registry, snapshot: EntitySnapshot) => void;"'
    )


def _invalid_heredoc_command() -> str:
    return (
        '{"command":"cat <<\'EOF\' > x\nimport type { Registry, EntitySnapshot } from y\n'
        'rollback: (registry: Registry, snapshot: EntitySnapshot) => void;\nEOF"'
    )


def _partial_simple_command() -> str:
    return '{"command":"printf'


def _partial_multiline_command() -> str:
    return '{"command":"x\nrollback: (registry: Registry'


def _prose_wrapped_command() -> str:
    return 'Sure, here is the payload: {"command":"printf ok"} Let me know if you want more.'


def _markdown_fenced_command() -> str:
    return '```json\n{"command":"printf ok"}\n```'


def _prose_fenced_multiline_command() -> str:
    return 'Here you go:\n```json\n{"command":"x\\nrollback: (registry: Registry, snapshot: EntitySnapshot) => void;"}\n```'


def _inline_comment_command() -> str:
    return '{"command":"printf ok" // keep this\n}'


def _block_comment_command() -> str:
    return '{/*a*/"command":"printf ok"}'


def _partial_comment_command() -> str:
    return '{/*a*/command:"printf ok"'


def _single_quoted_command() -> str:
    return "{'command': 'printf ok'}"


def _missing_comma_command() -> str:
    return '{"command":"a" "other":"b"}'


def _unquoted_value_command() -> str:
    return "{command: printf_ok}"


def _broken_inner_quotes_command() -> str:
    return '{"command": "say "hello" now"}'


def test_llms_json_package_reexports_strict_and_recovery_helpers() -> None:
    assert decode_json_object_or_none('{"command":"printf ok"}') == {"command": "printf ok"}
    result = recover('{"command":"printf ok"}', partial=False)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == {"command": "printf ok"}


@pytest.mark.parametrize(
    ("raw", "expected_command"),
    [
        (_valid_multiline_command(), "x\nrollback: (registry: Registry, snapshot: EntitySnapshot) => void;"),
        (
            _valid_multiline_import_command(),
            ("x\nimport type { Registry, EntitySnapshot } from y\nrollback: (registry: Registry, snapshot: EntitySnapshot) => void;"),
        ),
        (
            _valid_heredoc_command(),
            (
                "cat <<'EOF' > x\nimport type { Registry, EntitySnapshot } from y\n"
                "rollback: (registry: Registry, snapshot: EntitySnapshot) => void;\nEOF"
            ),
        ),
        (_invalid_multiline_command(), "x\nrollback: (registry: Registry, snapshot: EntitySnapshot) => void;"),
        (
            _invalid_multiline_import_command(),
            ("x\nimport type { Registry, EntitySnapshot } from y\nrollback: (registry: Registry, snapshot: EntitySnapshot) => void;"),
        ),
        (
            _invalid_heredoc_command(),
            (
                "cat <<'EOF' > x\nimport type { Registry, EntitySnapshot } from y\n"
                "rollback: (registry: Registry, snapshot: EntitySnapshot) => void;\nEOF"
            ),
        ),
    ],
)
def test_recover_final_preserves_multiline_command_payloads(raw: str, expected_command: str) -> None:
    result = recover(raw, partial=False, schema=_COMMAND_SCHEMA)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == {"command": expected_command}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (_prose_wrapped_command(), {"command": "printf ok"}),
        (_markdown_fenced_command(), {"command": "printf ok"}),
        (_prose_fenced_multiline_command(), {"command": "x\nrollback: (registry: Registry, snapshot: EntitySnapshot) => void;"}),
        (_inline_comment_command(), {"command": "printf ok"}),
        (_block_comment_command(), {"command": "printf ok"}),
        (_single_quoted_command(), {"command": "printf ok"}),
        (_missing_comma_command(), {"command": "a", "other": "b"}),
        (_unquoted_value_command(), {"command": "printf_ok"}),
        (_broken_inner_quotes_command(), {"command": 'say "hello" now'}),
        ('{0: "x", -1: "y"}', {"0": "x", "-1": "y"}),
    ],
)
def test_recover_final_handles_noisy_and_syntax_cases(raw: str, expected: dict[str, object]) -> None:
    result = recover(raw, partial=False, schema=_COMMAND_SCHEMA)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == expected


@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        (_partial_simple_command(), "printf"),
        (_partial_multiline_command(), "rollback: (registry: Registry"),
    ],
)
def test_recover_partial_keeps_partial_command_fragments(raw: str, fragment: str) -> None:
    result = recover(raw, partial=True, schema=_COMMAND_SCHEMA)

    assert result.outcome == Outcome.INCOMPLETE
    assert isinstance(result.value, dict)
    assert fragment in result.value["command"]


def test_recover_partial_supports_json5_style_comments() -> None:
    result = recover(_partial_comment_command(), partial=True, schema=_COMMAND_SCHEMA)

    assert result.outcome == Outcome.INCOMPLETE
    assert result.value == {"command": "printf ok"}


def test_recover_partial_does_not_swallow_trailing_comment_into_completed_string() -> None:
    result = recover('{"command":"printf ok" // keep this', partial=True, schema=_COMMAND_SCHEMA)

    assert result.outcome == Outcome.INCOMPLETE
    assert result.value == {"command": "printf ok"}


def test_recover_does_not_coerce_types_or_drop_extra_keys() -> None:
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
        "additionalProperties": False,
    }

    result = recover("{'n':'4','x':1}", partial=False, schema=schema)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == {"n": "4", "x": 1}


def test_recover_handles_schema_guided_or_expression_capture() -> None:
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "intent": {"type": "string"},
        },
    }

    result = recover(
        '{ query: "python quicksort" OR "Python 快速排序", intent: 用户在寻找python语言的快速排序 }',
        partial=False,
        schema=schema,
    )

    assert result.outcome == Outcome.COMPLETE
    assert result.value == {
        "query": '"python quicksort" OR "Python 快速排序"',
        "intent": "用户在寻找python语言的快速排序",
    }


def test_recover_handles_schema_guided_implicit_root_object() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
        },
    }

    result = recover("name: John, age: 30", partial=False, schema=schema)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == {"name": "John", "age": 30}


def test_recover_handles_schema_guided_implicit_objects_inside_array() -> None:
    schema = {
        "type": "object",
        "properties": {
            "users": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "id": {"type": "number"},
                    },
                },
            }
        },
    }

    result = recover('{ "users": [ name: John, id: 1 name: Alice, id: 2 ] }', partial=False, schema=schema)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == {"users": [{"name": "John", "id": 1}, {"name": "Alice", "id": 2}]}


def test_recover_handles_schema_guided_root_ref_object() -> None:
    schema = {
        "$ref": "#/$defs/root",
        "$defs": {
            "root": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"},
                },
            }
        },
    }

    result = recover("name: John, age: 30", partial=False, schema=schema)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == {"name": "John", "age": 30}


def test_recover_handles_schema_guided_root_hash_ref_object() -> None:
    schema = {
        "$ref": "#",
        "type": "object",
        "properties": {
            "name": {"type": "string"},
        },
    }

    result = recover("name: John", partial=False, schema=schema)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == {"name": "John"}


def test_recover_handles_schema_guided_prose_before_implicit_root_object() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
        },
    }

    result = recover("Here is the object: name: John, age: 30", partial=False, schema=schema)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == {"name": "John", "age": 30}


def test_recover_handles_schema_guided_label_before_implicit_root_object() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
        },
    }

    result = recover("Response:\nname: John, age: 30", partial=False, schema=schema)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == {"name": "John", "age": 30}


def test_recover_handles_schema_guided_root_all_of_object() -> None:
    schema = {
        "allOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
            },
            {
                "type": "object",
                "properties": {
                    "age": {"type": "number"},
                },
            },
        ]
    }

    result = recover("name: John, age: 30", partial=False, schema=schema)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == {"name": "John", "age": 30}


def test_recover_handles_schema_guided_root_one_of_object_hints() -> None:
    schema = {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
            },
            {
                "type": "object",
                "properties": {
                    "age": {"type": "number"},
                },
            },
        ]
    }

    result = recover("name: John, age: 30", partial=False, schema=schema)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == {"name": "John", "age": 30}


def test_recover_handles_schema_guided_nested_ref_and_pattern_properties() -> None:
    schema = {
        "type": "object",
        "properties": {
            "env": {
                "$ref": "#/$defs/envMap",
            }
        },
        "$defs": {
            "envMap": {
                "type": "object",
                "patternProperties": {
                    "^env_[A-Z]+$": {"type": "string"},
                },
                "additionalProperties": False,
            }
        },
    }

    result = recover("{ env: { env_FOO: one, env_BAR: two } }", partial=False, schema=schema)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == {"env": {"env_FOO": "one", "env_BAR": "two"}}


def test_recover_handles_schema_guided_property_names_with_unevaluated_properties() -> None:
    schema = {
        "type": "object",
        "propertyNames": {"pattern": "^cfg_[a-z]+$"},
        "unevaluatedProperties": {"type": "string"},
    }

    result = recover("cfg_name: one, cfg_mode: fast", partial=False, schema=schema)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == {"cfg_name": "one", "cfg_mode": "fast"}


def test_recover_does_not_start_implicit_object_for_property_names_mismatch() -> None:
    schema = {
        "type": "object",
        "propertyNames": {"pattern": "^cfg_[a-z]+$"},
        "unevaluatedProperties": {"type": "string"},
    }

    result = recover("bad_name: one, cfg_mode: fast", partial=False, schema=schema)

    assert result.outcome == Outcome.REJECTED
    assert result.value is None


def test_recover_handles_schema_guided_if_then_else_object() -> None:
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
                "name": {"type": "string"},
            },
        },
        "else": {
            "properties": {
                "title": {"type": "string"},
            },
        },
    }

    user = recover("kind: user, name: John", partial=False, schema=schema)
    org = recover("kind: org, title: Acme", partial=False, schema=schema)

    assert user.outcome == Outcome.COMPLETE
    assert user.value == {"kind": "user", "name": "John"}
    assert org.outcome == Outcome.COMPLETE
    assert org.value == {"kind": "org", "title": "Acme"}


def test_recover_handles_schema_guided_dependent_schemas_object() -> None:
    schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
        },
        "dependentSchemas": {
            "kind": {
                "properties": {
                    "name": {"type": "string"},
                },
            }
        },
    }

    result = recover("kind: user, name: John", partial=False, schema=schema)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == {"kind": "user", "name": "John"}


def test_recover_handles_schema_guided_ref_with_sibling_properties() -> None:
    schema = {
        "$ref": "#/$defs/base",
        "properties": {
            "extra": {"type": "string"},
        },
        "$defs": {
            "base": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
            }
        },
    }

    result = recover("name: John, extra: note", partial=False, schema=schema)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == {"name": "John", "extra": "note"}


def test_recover_handles_schema_guided_prefix_items_object() -> None:
    schema = {
        "type": "array",
        "prefixItems": [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "id": {"type": "number"},
                },
            }
        ],
    }

    result = recover("[ name: John, id: 1 ]", partial=False, schema=schema)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == [{"name": "John", "id": 1}]


def test_recover_handles_schema_guided_additional_items_object() -> None:
    schema = {
        "type": "array",
        "items": [
            {"type": "number"},
        ],
        "additionalItems": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        },
    }

    result = recover("[1, name: John]", partial=False, schema=schema)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == [1, {"name": "John"}]


def test_recover_handles_schema_guided_unevaluated_items_object() -> None:
    schema = {
        "type": "array",
        "prefixItems": [
            {"type": "number"},
        ],
        "unevaluatedItems": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        },
    }

    result = recover("[1, name: John]", partial=False, schema=schema)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == [1, {"name": "John"}]


def test_recover_handles_schema_guided_contains_object() -> None:
    schema = {
        "type": "array",
        "contains": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        },
    }

    result = recover("[1, name: John]", partial=False, schema=schema)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == [1, {"name": "John"}]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            'Sure! Here is the JSON you requested:\n\n```json\n{\n    "name": "Alice",\n    "age": 30\n}\n```\n\nI hope this helps!',
            {"name": "Alice", "age": 30},
        ),
        (
            'I\'ll create a JSON object with the user\'s information.\n\n{"name": "Bob", "age": 25}',
            {"name": "Bob", "age": 25},
        ),
        ('{"name": "Charlie", "age": 35}\n\nAs you can see, Charlie is 35 years old.', {"name": "Charlie", "age": 35}),
        (
            '```json\n{"name": "First", "age": 1}\n```\n\n```json\n{"name": "Second", "age": 2}\n```',
            {"name": "First", "age": 1},
        ),
    ],
)
def test_recover_matches_useful_outputguard_commentary_extraction_cases(raw: str, expected: dict[str, object]) -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
        },
    }

    result = recover(raw, partial=False, schema=schema)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('Best response:\n\n{"answer": "yes"}\n\nbecause reasons.', {"answer": "yes"}),
        ('Para 1.\n\nPara 2.\n\n{"deep": "value"}\n\nPara 3.', {"deep": "value"}),
        ('OK here goes: {"x": 99}', {"x": 99}),
        ('{"solo": true}\n---\nfooter', {"solo": True}),
        ("The output is: {name: 'Alice', age: 30} and that's it.", {"name": "Alice", "age": 30}),
        (
            "```json\n{name: 'Alice', age: 30,}\n```\nLet me know!",
            {"name": "Alice", "age": 30},
        ),
    ],
)
def test_recover_matches_more_useful_upstream_commentary_and_combined_cases(raw: str, expected: dict[str, object]) -> None:
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "deep": {"type": "string"},
            "x": {"type": "number"},
            "solo": {"type": "boolean"},
            "name": {"type": "string"},
            "age": {"type": "number"},
        },
    }

    result = recover(raw, partial=False, schema=schema)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == expected


@pytest.mark.parametrize(
    ("raw", "partial", "expected_outcome", "expected"),
    [
        ("[1,2,3", True, Outcome.INCOMPLETE, [1, 2, 3]),
        ("[1,2,", True, Outcome.INCOMPLETE, [1, 2]),
        ('"I am text', True, Outcome.INCOMPLETE, "I am text"),
        ('{// comment\n"a": 1}', True, Outcome.COMPLETE, {"a": 1}),
        ("{/* incomplete comment", True, Outcome.INCOMPLETE, {}),
        ("'hello'", True, Outcome.COMPLETE, "hello"),
    ],
)
def test_recover_matches_useful_partialjson_style_cases(
    raw: str,
    partial: bool,
    expected_outcome: Outcome,
    expected: object,
) -> None:
    result = recover(raw, partial=partial)

    assert result.outcome == expected_outcome
    assert result.value == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0x1f", 31),
        ("-0x10", -16),
        ("0XFF", 255),
        ("'line1\\\nline2'", "line1line2"),
        ('"line1\\\r\nline2"', "line1line2"),
    ],
)
def test_recover_matches_useful_partialjson_json5_cases(raw: str, expected: object) -> None:
    result = recover(raw, partial=False)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("42", 42),
        ("-42", -42),
        ("12.34", 12.34),
        ("-12.34", -12.34),
        ("12.", 12.0),
        ("-12.", -12.0),
        ("true", True),
        ("false", False),
        ("null", None),
        ('"I am text"', "I am text"),
        ('"I\\"m text"', 'I"m text'),
        ('{"a":"\\u0041"}', {"a": "A"}),
        ('{"a":"\\u00E9"}', {"a": "é"}),
        ('{"a":"\\u20AC"}', {"a": "€"}),
    ],
)
def test_recover_matches_useful_partialjson_core_cases(raw: str, expected: object) -> None:
    result = recover(raw, partial=False)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"url": "https://example.com/path?a=1&b=2#hash"}', {"url": "https://example.com/path?a=1&b=2#hash"}),
        ('{"msg": "a, b, c,", "x": 1,}', {"msg": "a, b, c,", "x": 1}),
        ('{"regex": "\\\\{.*\\\\}", "data": [1, 2', {"regex": "\\{.*\\}", "data": [1, 2]}),
        ('{"text": "line1\nline2\nline3"}', {"text": "line1\nline2\nline3"}),
    ],
)
def test_recover_matches_useful_outputguard_style_cases(raw: str, expected: dict[str, object]) -> None:
    result = recover(raw, partial=False)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('1. {"step": "one"} is the first step', {"step": "one"}),
        ('> {"quoted": true}', {"quoted": True}),
        ('{"a":1} and {"b":2} are both valid', {"a": 1}),
        ('Response:\n{"items": [{"id": 1}]}\nEnd.', {"items": [{"id": 1}]}),
        ("Here: {a: 'value', b: 42,}\nDone", {"a": "value", "b": 42}),
        ("Sure!\n{items: [1, 2, 3,], total: 3}", {"items": [1, 2, 3], "total": 3}),
        ("{name: 'A', age: 1, // person\n}", {"name": "A", "age": 1}),
    ],
)
def test_recover_matches_more_outputguard_stress_cases(raw: str, expected: dict[str, object]) -> None:
    result = recover(raw, partial=False)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"template": "Hello {{name}}, welcome!"}', {"template": "Hello {{name}}, welcome!"}),
        ('{"html": "<div class=\\"test\\">Hello</div>"}', {"html": '<div class="test">Hello</div>'}),
        ('{"nulls": {"a": null, "b": null}}', {"nulls": {"a": None, "b": None}}),
        ('{"bools": {"a": true, "b": false}}', {"bools": {"a": True, "b": False}}),
        (
            '{"numbers": {"int": 0, "neg": -1, "float": 3.14, "exp": 1.5e10}}',
            {"numbers": {"int": 0, "neg": -1, "float": 3.14, "exp": 1.5e10}},
        ),
        ('[1, "two", true, null, {"five": 5}]', [1, "two", True, None, {"five": 5}]),
    ],
)
def test_recover_preserves_useful_valid_outputguard_style_json(raw: str, expected: object) -> None:
    result = recover(raw, partial=False)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == expected


def test_recover_includes_bakeoff_syntax_cases_where_repairjson_beat_outputguard() -> None:
    cases = [
        ('{"command":"a" "other":"b"}', {"command": "a", "other": "b"}),
        ("{command: printf_ok}", {"command": "printf_ok"}),
    ]

    for raw, expected in cases:
        result = recover(raw, partial=False, schema=_COMMAND_SCHEMA)
        assert result.outcome == Outcome.COMPLETE
        assert result.value == expected


@pytest.mark.parametrize(
    ("raw", "expected_outcome", "expected"),
    [
        ("", Outcome.REJECTED, None),
        ("   \n\t  ", Outcome.REJECTED, None),
        ("not json at all", Outcome.REJECTED, None),
        ("null", Outcome.COMPLETE, None),
        ("true", Outcome.COMPLETE, True),
        ("42", Outcome.COMPLETE, 42),
        ('"just a string"', Outcome.COMPLETE, "just a string"),
    ],
)
def test_recover_handles_empty_garbage_and_primitives_usefully(
    raw: str,
    expected_outcome: Outcome,
    expected: object,
) -> None:
    result = recover(raw, partial=False)

    assert result.outcome == expected_outcome
    assert result.value == expected


def test_recover_is_stable_on_repeated_tricky_inputs() -> None:
    cases = [
        ('{"a": [1, 2, 3}', False, None, {"a": [1, 2, 3]}),
        (
            '{ query: "python quicksort" OR "Python 快速排序", intent: 用户在寻找python语言的快速排序 }',
            False,
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "intent": {"type": "string"},
                },
            },
            {
                "query": '"python quicksort" OR "Python 快速排序"',
                "intent": "用户在寻找python语言的快速排序",
            },
        ),
        (
            '{ "users": [ name: John, id: 1 name: Alice, id: 2 ] }',
            False,
            {
                "type": "object",
                "properties": {
                    "users": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "id": {"type": "number"},
                            },
                        },
                    }
                },
            },
            {"users": [{"name": "John", "id": 1}, {"name": "Alice", "id": 2}]},
        ),
        ('{"command": "say "hello" now"}', False, _COMMAND_SCHEMA, {"command": 'say "hello" now'}),
        ('{"command":"val\\u00', True, _COMMAND_SCHEMA, {"command": "val\\u00"}),
        ('{/*a*/command:"printf ok"', True, _COMMAND_SCHEMA, {"command": "printf ok"}),
    ]

    for _ in range(100):
        for raw, partial, schema, expected in cases:
            result = recover(raw, partial=partial, schema=schema)
            assert result.value == expected


def test_recover_is_stable_on_repeated_recursive_schema_inputs() -> None:
    recursive_object_schema = {
        "type": "object",
        "properties": {
            "root": {"$ref": "#/$defs/node"},
        },
        "$defs": {
            "node": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "child": {"$ref": "#/$defs/node"},
                },
            }
        },
    }
    recursive_array_schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"$ref": "#/$defs/item"},
            },
        },
        "$defs": {
            "item": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "children": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/item"},
                    },
                },
            }
        },
    }
    cases = [
        (
            "{ root: { value: one, child: { value: two, child: { value: three } } }",
            recursive_object_schema,
            {"root": {"value": "one", "child": {"value": "two", "child": {"value": "three"}}}},
        ),
        (
            "{ items: [ name: a, children: [ name: b, children: [ name: c ] ] ] }",
            recursive_array_schema,
            {"items": [{"name": "a", "children": [{"name": "b", "children": [{"name": "c"}]}]}]},
        ),
    ]

    for _ in range(100):
        for raw, schema, expected in cases:
            result = recover(raw, partial=False, schema=schema)
            assert result.outcome == Outcome.COMPLETE
            assert result.value == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("{'a': True, 'b': False, 'c': None}", {"a": True, "b": False, "c": None}),
        ('{"a": [1, 2, 3}', {"a": [1, 2, 3]}),
        ("[1{a:2}]", [1, {"a": 2}]),
        ("{a:1{b:2}}", {"a": 1, "b": 2}),
        ("{'a': .5}", {"a": 0.5}),
        ("{'a': +.5}", {"a": 0.5}),
        ("{'a': -.5}", {"a": -0.5}),
        ("{'a': +5}", {"a": 5}),
        ("{'a': 1e}", {"a": 1.0}),
        ("{'a': 1e+}", {"a": 1.0}),
        ("{'a': 01}", {"a": 1}),
        ("{'a': 1..2}", {"a": "1..2"}),
        ("result = {a:1}", {"a": 1}),
        ("Note: '{not the payload}' {a:1}", {"a": 1}),
        ("Items follow: [1,2,3]", [1, 2, 3]),
    ],
)
def test_recover_matches_useful_repairjson_style_cases(raw: str, expected: object) -> None:
    result = recover(raw, partial=False)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == expected


@pytest.mark.parametrize(
    ("raw", "expected_command"),
    [
        ('{"command":"val\\', "val\\"),
        ('{"command":"val\\n', "val\\n"),
        ('{"command":"val\\u00', "val\\u00"),
        ('{"command":"val\\"', 'val\\"'),
    ],
)
def test_recover_partial_preserves_incomplete_escape_progression(raw: str, expected_command: str) -> None:
    result = recover(raw, partial=True, schema=_COMMAND_SCHEMA)

    assert result.outcome == Outcome.INCOMPLETE
    assert result.value == {"command": expected_command}
