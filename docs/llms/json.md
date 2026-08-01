# Model JSON

Models often produce JSON-like text rather than strict JSON. A tool call may contain single quotes, comments, a missing
closing brace, or a string where the schema expects a number. `plap.llms.json` separates four operations so application code
can decide how much repair to allow.

Recovery and normalization do not replace validation. Recovery does not remove unknown keys or coerce types. Normalization
changes a value only when a schema provides an unambiguous valid replacement, but the complete result may still violate
another schema rule.

## Decode strict JSON

Use the decode helpers when malformed JSON must fail:

| Function | Result |
| --- | --- |
| `decode_json_value(text)` | Any decoded JSON value; raises `msgspec.DecodeError` |
| `decode_json_value_with_error(text)` | `(value, error)` without raising a decode error |
| `decode_json_object_or_none(text)` | A dictionary, or `None` |
| `decode_json_object_with_error(text)` | `(dictionary, error)` |
| `encode_json_value(value)` | Deterministic JSON text |
| `encode_json_object(value)` | Deterministic JSON text for a dictionary |

## Recover model output

`recover` accepts common model syntax errors and reports whether the value is complete:

```python
from plap.llms.json import Outcome, recover

result = recover(
    "{'command': 'printf ok'}",
    partial=False,
    schema={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
)

if result.outcome == Outcome.REJECTED:
    raise ValueError("model output did not contain recoverable JSON")

value = result.value
```

The outcomes are:

| Outcome | Meaning |
| --- | --- |
| `COMPLETE` | Recovery found a complete value |
| `INCOMPLETE` | Partial mode recovered the value available so far |
| `REJECTED` | The text did not contain a safe recoverable value |

Set `partial=True` while arguments are still streaming. The returned value may then contain unfinished strings or objects.

## Normalize and validate

Recovery can feed normalization and validation directly:

```python
from plap.llms.json import (
    Outcome,
    ValidationError,
    compile_validator,
    normalize,
    recover,
    validation_error_message,
)

SCHEMA = {
    "type": "object",
    "properties": {"count": {"type": "integer"}},
    "required": ["count"],
    "additionalProperties": False,
}

recovered = recover("{'count': '5'}", partial=False, schema=SCHEMA)
if recovered.outcome == Outcome.REJECTED:
    raise ValueError("model output did not contain recoverable JSON")

value = normalize(recovered.value, schema=SCHEMA)
validator = compile_validator(SCHEMA)

try:
    validator.validate(value)
except ValidationError as exc:
    raise ValueError(validation_error_message(exc)) from exc
```

`normalize` can convert exact numeric, boolean, null, `const`, and `enum` strings when the schema confirms one result. It can
also wrap one value in an array when that produces a valid array. It does not extract values from arbitrary prose or remove
additional properties.

`compile_validator` caches validators by schema content. `validation_error_message` adds the failing data path to the
underlying validation message when one is available.

## JSON in completion streams

`Accumulator` uses recovery and normalization while assembling streamed tool calls. Partial snapshots expose the recovered
arguments available so far; the terminal snapshot performs final normalization.

The built-in `retry_on_unusable_tool_calls` validator then checks that final arguments are an object and satisfy the tool's
schema. [Completion retries](retries.md) can ask the model to replace arguments that remain invalid.
