from __future__ import annotations

from typing import Any

import blake3
import jsonschema_rs
import msgspec

Validator = jsonschema_rs.Validator
ValidationError = jsonschema_rs.ValidationError

_VALIDATORS: dict[bytes, Validator] = {}


def _schema_cache_key(schema: dict[str, Any]) -> bytes:
    encoded = msgspec.json.encode(schema, order="deterministic")
    return blake3.blake3(encoded).digest()


def compile_validator(schema: dict[str, Any]) -> Validator:
    cache_key = _schema_cache_key(schema)
    validator = _VALIDATORS.get(cache_key)
    if validator is not None:
        return validator
    validator = jsonschema_rs.validator_for(schema)
    _VALIDATORS[cache_key] = validator
    return validator


def _validation_error_rule(error: ValidationError) -> str | None:
    schema_path = getattr(error, "schema_path", None)
    if not schema_path:
        return None
    rule = schema_path[-1]
    return str(rule)


def _instance_path_label(error: ValidationError) -> str | None:
    instance_path = getattr(error, "instance_path", None) or []
    if not instance_path:
        return None
    path = "data"
    for segment in instance_path:
        path += f"[{segment}]" if isinstance(segment, int) else f".{segment}"
    return path


def validation_error_message(error: ValidationError) -> str:
    message = getattr(error, "message", str(error)).strip()
    path = _instance_path_label(error)
    if path is None:
        return message
    return f"{path}: {message}"


__all__ = [
    "ValidationError",
    "Validator",
    "compile_validator",
    "validation_error_message",
]
