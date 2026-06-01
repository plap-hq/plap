from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import msgspec
from json_repair import repair_json

from plap.llms.completions.chat import (
    ChatCompletionDelta,
    ChatCompletionResult,
    ChatFinishReason,
    ChatMessage,
    ChatTool,
    ChatToolCall,
    ChatUsage,
)


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


def _repair_tool_call(
    call: ChatToolCall,
    *,
    tools_by_name: dict[str, ChatTool],
    partial: bool,
) -> ChatToolCall:
    tool = tools_by_name.get(call.name)
    schema = tool.function.parameters if tool is not None else None
    try:
        value = repair_json(
            call.arguments,
            return_objects=True,
            skip_json_loads=True,
            stream_stable=partial,
            schema=schema,
            schema_repair_mode="standard",
        )
    except Exception:
        return call
    if not isinstance(value, dict):
        return call
    return replace(
        call,
        arguments=msgspec.json.encode(value, order="deterministic").decode(),
    )


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
        self._reasoning_details: list[dict[str, Any]] = []
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
        if delta.reasoning_details_delta:
            self._reasoning_details.extend(delta.reasoning_details_delta)
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
            reasoning_details=list(self._reasoning_details),
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
