from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import blake3
import structlog

from plap.llms.completions.chat import (
    ChatCompletionDelta,
    ChatCompletionResult,
    ChatFinishReason,
    ChatMessage,
    ChatTool,
    ChatToolCall,
    ChatUsage,
)
from plap.llms.json import Outcome, decode_json_value_with_error, encode_json_object, normalize, recover, schema_property_keys
from plap.logging import log_payload

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Snapshot:
    messages: tuple[ChatMessage, ...]
    results: tuple[ChatCompletionResult, ...]
    delta: ChatCompletionDelta | None = None


@dataclass(slots=True)
class _ToolCall:
    tool_call_id: str | None = None
    name: str | None = None
    argument_parts: list[str] = field(default_factory=list)

    def apply(self, delta) -> None:
        if delta.id is not None:
            self.tool_call_id = delta.id
        if delta.name is not None:
            self.name = delta.name
        if delta.arguments_delta is not None:
            self.argument_parts.append(delta.arguments_delta)

    def tool_call(self) -> ChatToolCall:
        if self.tool_call_id is None:
            raise RuntimeError("streamed tool call is missing id")
        if self.name is None:
            raise RuntimeError("streamed tool call is missing name")
        return ChatToolCall(
            id=self.tool_call_id,
            name=self.name,
            arguments="".join(self.argument_parts),
        )


def _arguments_hash(arguments: str | None) -> str | None:
    if arguments is None:
        return None
    return blake3.blake3(arguments.encode()).hexdigest()


def _mapping_keys(value: Any) -> list[str] | None:
    if not isinstance(value, dict):
        return None
    return sorted(str(key) for key in value)


def _tool_call_repair_issues(
    *,
    raw_value: Any | None,
    raw_decode_error: str | None,
    raw_keys: list[str] | None,
    repaired: dict[str, Any] | None,
    repaired_keys: list[str] | None,
    repair_error: Exception | None,
    undeclared_keys: list[str],
) -> list[str]:
    issues: list[str] = []
    if raw_decode_error is not None:
        issues.append("raw_invalid_json")
    elif raw_keys is None:
        issues.append("raw_not_object")
    if repair_error is not None:
        issues.append("repair_failed")
    elif repaired is None:
        issues.append("repaired_not_object")
    else:
        if raw_keys is not None and raw_keys != repaired_keys:
            issues.append("key_set_changed")
        if isinstance(raw_value, dict) and raw_value != repaired:
            issues.append("value_changed")
        if undeclared_keys:
            issues.append("undeclared_keys")
    return issues


def _log_tool_call_repair(
    call: ChatToolCall,
    *,
    tool: ChatTool | None,
    partial: bool,
    raw_arguments: str,
    repaired_arguments: str | None,
    repaired: dict[str, Any] | None,
    repair_error: Exception | None,
) -> None:
    if partial:
        return
    raw_value, raw_decode_error = decode_json_value_with_error(raw_arguments)
    raw_decode_error_message = None if raw_decode_error is None else str(raw_decode_error)
    raw_keys = _mapping_keys(raw_value)
    repaired_keys = _mapping_keys(repaired)
    schema_keys = schema_property_keys(None if tool is None else tool.function.parameters)
    undeclared_keys = [] if repaired_keys is None or not schema_keys else [key for key in repaired_keys if key not in schema_keys]
    issues = _tool_call_repair_issues(
        raw_value=raw_value,
        raw_decode_error=raw_decode_error_message,
        raw_keys=raw_keys,
        repaired=repaired,
        repaired_keys=repaired_keys,
        repair_error=repair_error,
        undeclared_keys=undeclared_keys,
    )
    decoded_key_set_changed = None if raw_keys is None or repaired_keys is None else raw_keys != repaired_keys
    decoded_value_changed = None if not isinstance(raw_value, dict) or repaired is None else raw_value != repaired
    repair_outcome = "error" if repair_error is not None else "dict" if repaired is not None else "non_dict"

    log_payload(
        logger,
        "llm.accumulator.tool_call_repair.payload",
        decoded_key_set_changed=decoded_key_set_changed,
        decoded_value_changed=decoded_value_changed,
        issues=issues,
        raw_arguments=raw_arguments,
        raw_arguments_hash=_arguments_hash(raw_arguments),
        raw_arguments_length=len(raw_arguments),
        raw_decode_error=raw_decode_error_message,
        raw_is_object=isinstance(raw_value, dict),
        raw_json_valid=raw_decode_error is None,
        raw_keys=raw_keys,
        repaired_arguments=repaired_arguments,
        repaired_arguments_hash=_arguments_hash(repaired_arguments),
        repaired_arguments_length=None if repaired_arguments is None else len(repaired_arguments),
        repaired_is_object=repaired is not None,
        repaired_keys=repaired_keys,
        repair_changed=repaired_arguments is not None and repaired_arguments != raw_arguments,
        repair_error_message=None if repair_error is None else str(repair_error),
        repair_error_type=None if repair_error is None else type(repair_error).__name__,
        repair_outcome=repair_outcome,
        schema_keys=schema_keys,
        tool_call_id=call.id,
        tool_name=call.name,
        tool_strict=None if tool is None else tool.function.strict,
        undeclared_keys=undeclared_keys,
    )


def _repair_tool_call(
    call: ChatToolCall,
    *,
    tools_by_name: dict[str, ChatTool],
    partial: bool,
) -> ChatToolCall:
    tool = tools_by_name.get(call.name)
    repaired: dict[str, Any] | None = None
    repaired_arguments: str | None = None
    recovery = recover(
        call.arguments,
        partial=partial,
        schema=None if tool is None else tool.function.parameters,
    )
    if recovery.outcome == Outcome.REJECTED or not isinstance(recovery.value, dict):
        _log_tool_call_repair(
            call,
            tool=tool,
            partial=partial,
            raw_arguments=call.arguments,
            repaired_arguments=None,
            repaired=None,
            repair_error=None,
        )
        return call
    repaired = normalize(recovery.value, schema=None if partial or tool is None else tool.function.parameters)
    repaired_arguments = encode_json_object(repaired)
    _log_tool_call_repair(
        call,
        tool=tool,
        partial=partial,
        raw_arguments=call.arguments,
        repaired_arguments=repaired_arguments,
        repaired=repaired,
        repair_error=None,
    )
    return replace(call, arguments=repaired_arguments)


def _repair_message(
    message: ChatMessage,
    *,
    tools_by_name: dict[str, ChatTool],
    partial: bool,
) -> ChatMessage:
    tool_calls = message.tool_calls
    if not tool_calls:
        return message
    repaired = [_repair_tool_call(call, tools_by_name=tools_by_name, partial=partial) for call in tool_calls]
    return replace(message, tool_calls=repaired)


class Accumulator:
    def __init__(self, *, tools: tuple[ChatTool, ...] = ()) -> None:
        self._tools_by_name = {tool.function.name: tool for tool in tools}
        self._id: str | None = None
        self._model: str | None = None
        self._created_at: float | None = None
        self._content: list[str] = []
        self._has_content = False
        self._reasoning: list[str] = []
        self._calls: dict[int, _ToolCall] = {}
        self._finish: ChatFinishReason | None = None
        self._usage: ChatUsage | None = None
        self._fingerprint: str | None = None
        self._tier: str | None = None

    def apply(self, delta: ChatCompletionDelta) -> Snapshot:
        self._apply(delta)
        result = self._result() if delta.finish_reason is not None else None
        message = result.message if result is not None else self._message(partial=True)
        return Snapshot(
            messages=(message,),
            results=(result,) if result is not None else (),
            delta=delta,
        )

    def _apply(self, delta: ChatCompletionDelta) -> None:
        if delta.id is not None:
            self._id = delta.id
        if delta.model is not None:
            self._model = delta.model
        if delta.created_at is not None:
            self._created_at = delta.created_at
        if delta.content_delta is not None:
            self._has_content = True
            self._content.append(delta.content_delta)
        if delta.reasoning_delta is not None:
            self._reasoning.append(delta.reasoning_delta)
        if delta.tool_call_delta is not None:
            call = self._calls.setdefault(delta.tool_call_delta.index, _ToolCall())
            call.apply(delta.tool_call_delta)
        if delta.finish_reason is not None:
            self._finish = delta.finish_reason
        if delta.usage is not None:
            self._usage = delta.usage
        if delta.system_fingerprint is not None:
            self._fingerprint = delta.system_fingerprint
        if delta.service_tier is not None:
            self._tier = delta.service_tier

    def _message(self, *, partial: bool) -> ChatMessage:
        content = "".join(self._content) if self._has_content else None
        reasoning_content = "".join(self._reasoning) or None
        tool_calls = [self._calls[index].tool_call() for index in sorted(self._calls)]
        message = ChatMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )
        return _repair_message(
            message,
            tools_by_name=self._tools_by_name,
            partial=partial,
        )

    def _result(self) -> ChatCompletionResult:
        if self._finish is None:
            raise RuntimeError("streamed completion finish_reason is missing")
        if self._model is None:
            raise RuntimeError("streamed completion model is missing")
        return ChatCompletionResult(
            id=self._id,
            model=self._model,
            created_at=self._created_at,
            message=self._message(partial=False),
            finish_reason=self._finish,
            usage=self._usage,
            system_fingerprint=self._fingerprint,
            service_tier=self._tier,
        )


__all__ = ["Accumulator", "Snapshot"]
