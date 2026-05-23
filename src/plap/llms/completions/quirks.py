from __future__ import annotations

from typing import Any

from plap.llms.completions.chat import (
    ChatResponseFormatType,
    ChatToolChoiceFunction,
)
from plap.llms.completions.client import Call, Quirk
from plap.llms.completions.errors import ChatCompletionUnsupportedRequestError


def _merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(current, value)
            continue
        merged[key] = value
    return merged


def _first_choice(raw: dict[str, Any]) -> dict[str, Any] | None:
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    return choice if isinstance(choice, dict) else None


def _response_message(raw: dict[str, Any]) -> dict[str, Any] | None:
    choice = _first_choice(raw)
    if choice is None:
        return None
    message = choice.get("message")
    return message if isinstance(message, dict) else None


def _response_delta(raw: dict[str, Any]) -> dict[str, Any] | None:
    choice = _first_choice(raw)
    if choice is None:
        return None
    delta = choice.get("delta")
    return delta if isinstance(delta, dict) else None


def _rename_message_field(raw: dict[str, Any], old: str, new: str) -> None:
    message = _response_message(raw)
    if message is None or old not in message:
        return
    if message.get(new) is not None:
        return
    message[new] = message[old]


def _rename_delta_field(raw: dict[str, Any], old: str, new: str) -> None:
    delta = _response_delta(raw)
    if delta is None or old not in delta:
        return
    if delta.get(new) is not None:
        return
    delta[new] = delta[old]


class Only(Quirk):
    def __init__(self, *names: str) -> None:
        self._names = frozenset(names)

    def request(self, call: Call) -> None:
        call.body = {
            key: value
            for key, value in call.body.items()
            if key in self._names
        }


class Rename(Quirk):
    def __init__(self, old: str, new: str) -> None:
        self._old = old
        self._new = new

    def request(self, call: Call) -> None:
        if self._old not in call.body or self._new in call.body:
            return
        call.body[self._new] = call.body.pop(self._old)


class Set(Quirk):
    def __init__(self, name: str, value: Any) -> None:
        self._name = name
        self._value = value

    def request(self, call: Call) -> None:
        call.body[self._name] = self._value


class SystemRole(Quirk):
    def request(self, call: Call) -> None:
        messages = call.body.get("messages")
        if not isinstance(messages, list):
            return
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "developer":
                message["role"] = "system"


class ExtraBody(Quirk):
    def __init__(self, value: dict[str, Any]) -> None:
        self._value = value

    def request(self, call: Call) -> None:
        current = call.body.get("extra_body")
        if isinstance(current, dict):
            call.body["extra_body"] = _merge_dicts(current, self._value)
            return
        call.body["extra_body"] = dict(self._value)


class DropIf(Quirk):
    def __init__(self, name: str, value: Any) -> None:
        self._name = name
        self._value = value

    def request(self, call: Call) -> None:
        if call.body.get(self._name) == self._value:
            call.body.pop(self._name, None)


class EnsureAssistantReasoningContent(Quirk):
    def request(self, call: Call) -> None:
        messages = call.body.get("messages")
        if not isinstance(messages, list):
            return
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "assistant":
                continue
            if message.get("reasoning_content") is None:
                message["reasoning_content"] = ""


class ForceRequiredTool(Quirk):
    def request(self, call: Call) -> None:
        if not isinstance(call.request.tool_choice, ChatToolChoiceFunction):
            return
        if len(call.request.tools) == 1 and call.request.tools[0].function.name == call.request.tool_choice.name:
            call.body["tool_choice"] = "required"
            return
        raise ChatCompletionUnsupportedRequestError(
            "forced function tool_choice objects are unsupported here; use tool_choice='required' or provide exactly one matching tool"
        )


class RenameOutput(Quirk):
    def __init__(self, old: str, new: str) -> None:
        self._old = old
        self._new = new

    async def complete(self, call: Call, next_complete) -> dict[str, Any]:
        raw = await next_complete(None)
        _rename_message_field(raw, self._old, self._new)
        return raw

    async def stream(self, call: Call, next_complete, next_stream):
        async for raw in next_stream(None):
            _rename_delta_field(raw, self._old, self._new)
            yield raw


class RejectResponseFormat(Quirk):
    def __init__(self, *types: str) -> None:
        self._types = frozenset(ChatResponseFormatType(value) for value in types)

    def request(self, call: Call) -> None:
        response_format = call.request.response_format
        if response_format is None:
            return
        if self._types and response_format.type not in self._types:
            return
        if not self._types:
            raise ChatCompletionUnsupportedRequestError(
                f"response_format is not supported for model {call.request.model!r}"
            )
        raise ChatCompletionUnsupportedRequestError(
            f"response_format is not supported for model {call.request.model!r} when type is {response_format.type.value!r}"
        )


__all__ = [
    "DropIf",
    "EnsureAssistantReasoningContent",
    "ExtraBody",
    "ForceRequiredTool",
    "Only",
    "RejectResponseFormat",
    "Rename",
    "RenameOutput",
    "Set",
    "SystemRole",
]
