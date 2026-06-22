from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from plap.llms.json import Outcome, decode_json_value, encode_json_value, normalize, recover
from plap.llms.json.schema import compile_validator

_SAFE_CHARS = st.characters(blacklist_categories=("Cs",))
_JSON_SCALARS = st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text(_SAFE_CHARS)
_JSON_VALUES = st.recursive(
    _JSON_SCALARS,
    lambda children: st.lists(children, max_size=5) | st.dictionaries(st.text(_SAFE_CHARS, max_size=12), children, max_size=5),
    max_leaves=20,
)
_SCHEMAS = st.sampled_from(
    [
        None,
        {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "number"}}},
        {
            "type": "object",
            "properties": {
                "env": {
                    "type": "object",
                    "patternProperties": {
                        "^env_[A-Z]+$": {"type": "string"},
                    },
                    "additionalProperties": False,
                }
            },
        },
        {
            "$ref": "#/$defs/root",
            "$defs": {
                "root": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                }
            },
        },
    ]
)


@settings(max_examples=150, deadline=200)
@given(value=_JSON_VALUES)
def test_recover_hypothesis_valid_json_round_trips(value: object) -> None:
    text = json.dumps(value, ensure_ascii=False)

    result = recover(text, partial=False)

    assert result.outcome == Outcome.COMPLETE
    assert result.value == value


@settings(max_examples=250, deadline=200)
@given(text=st.text(_SAFE_CHARS, max_size=160), partial=st.booleans(), schema=_SCHEMAS)
def test_recover_hypothesis_never_crashes_and_complete_values_round_trip(
    text: str,
    partial: bool,
    schema: dict[str, object] | None,
) -> None:
    result = recover(text, partial=partial, schema=schema)

    assert result.outcome in {Outcome.COMPLETE, Outcome.INCOMPLETE, Outcome.REJECTED}
    if result.outcome == Outcome.REJECTED:
        assert result.value is None
        return

    encoded = encode_json_value(result.value)
    assert decode_json_value(encoded) == result.value


@settings(max_examples=150, deadline=200)
@given(value=_JSON_VALUES, schema=_SCHEMAS)
def test_normalize_hypothesis_never_crashes(value: object, schema: dict[str, object] | None) -> None:
    normalize(value, schema=schema)


@settings(max_examples=100, deadline=200)
@given(value=_JSON_VALUES, schema=_SCHEMAS.filter(lambda item: item is not None))
def test_normalize_hypothesis_is_noop_for_valid_values(value: object, schema: dict[str, object]) -> None:
    validator = compile_validator(schema)
    if validator.is_valid(value):
        assert normalize(value, schema=schema) == value
