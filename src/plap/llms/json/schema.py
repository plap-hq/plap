from __future__ import annotations

import re
from enum import StrEnum
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


class _Match(StrEnum):
    MATCH = "match"
    UNKNOWN = "unknown"
    MISMATCH = "mismatch"


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

    return isinstance(current.get("properties"), dict) or isinstance(current.get("patternProperties"), dict)


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


__all__ = [
    "ValidationError",
    "Validator",
    "_schema_accepts_key",
    "_schema_accepts_specific_key",
    "_schema_expects_object",
    "_schema_for_path",
    "_schema_has_specific_keys",
    "compile_validator",
    "validation_error_message",
]
