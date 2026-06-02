from __future__ import annotations

from collections.abc import Mapping

from plap.llms.completions.chat import ChatMessage as Message
from plap.responses.patch import JSONValue

_TEXT = ""
_NUMBER = 0
_BOOL = False


def _shape_scalar(value: JSONValue) -> JSONValue:
    if value is None:
        return None
    if isinstance(value, bool):
        return _BOOL
    if isinstance(value, int | float):
        return _NUMBER
    if isinstance(value, str):
        return _TEXT
    raise TypeError(f"unsupported shape scalar: {type(value).__name__}")


def _shape_tool_call(value: Mapping[str, JSONValue]) -> JSONValue:
    shaped: dict[str, JSONValue] = {"id": value["id"]}
    if "name" in value:
        shaped["name"] = _shape_scalar(value["name"])
    if "arguments" in value:
        shaped["arguments"] = _shape_scalar(value["arguments"])
    return shaped


def _shape_message(value: Mapping[str, JSONValue]) -> JSONValue:
    shaped: dict[str, JSONValue] = {"role": value["role"]}
    if "content" in value:
        shaped["content"] = _shape_scalar(value["content"])
    if "name" in value:
        shaped["name"] = _shape_scalar(value["name"])
    if "refusal" in value:
        shaped["refusal"] = _shape_scalar(value["refusal"])
    if "tool_call_id" in value:
        shaped["tool_call_id"] = value["tool_call_id"]
    if "tool_calls" in value:
        tool_calls = value["tool_calls"]
        if not isinstance(tool_calls, list):
            raise TypeError("message tool_calls must be an array")
        shaped["tool_calls"] = [
            _shape_tool_call(call) if isinstance(call, Mapping) else _shape_scalar(call)
            for call in tool_calls
        ]
    if "reasoning_content" in value:
        shaped["reasoning_content"] = _shape_scalar(value["reasoning_content"])
    if "reasoning_details" in value:
        shaped["reasoning_details"] = _shape_value(value["reasoning_details"])
    return shaped


def _shape_mapping(value: Mapping[str, JSONValue]) -> JSONValue:
    if "role" in value:
        return _shape_message(value)
    if "id" in value and "arguments" in value:
        return _shape_tool_call(value)
    return {str(key): _shape_value(item) for key, item in value.items()}


def _shape_value(value: JSONValue) -> JSONValue:
    if isinstance(value, Mapping):
        return _shape_mapping(value)
    if isinstance(value, list):
        return [_shape_value(item) for item in value]
    return _shape_scalar(value)


def shape(messages: list[Message]) -> JSONValue:
    return [_shape_message(message.to_primitive()) for message in messages]


__all__ = ["shape"]
