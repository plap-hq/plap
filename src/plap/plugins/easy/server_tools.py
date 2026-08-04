from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import dataclass_transform

from plap.bus import bus
from plap.llms.accumulator import Snapshot
from plap.llms.completions.budget import CompletionBudgetExhaustedError
from plap.llms.completions.chat import (
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatFunctionTool,
    ChatMessage,
    ChatRole,
    ChatTool,
    ChatToolCall,
)
from plap.responses.state import State

BUDGET_TOOL_OUTPUT = "The server could not run this tool because the response budget was exhausted."


@dataclass_transform(frozen_default=True, field_specifiers=(field,))
@dataclass(frozen=True)
class ServerTool(ChatFunctionTool):
    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        dataclass(frozen=True)(cls)

    async def __call__(
        self,
        state: State,
        call: ChatToolCall,
    ) -> ChatMessage:
        raise NotImplementedError


_registered: list[ServerTool] = []


def register[T: ServerTool](tool_type: type[T]) -> type[T]:
    if not isinstance(tool_type, type) or not issubclass(tool_type, ServerTool):
        raise TypeError("registered server tool must be a ServerTool class")
    try:
        tool = tool_type()
    except TypeError as exc:
        raise TypeError("registered server tool must be constructible without arguments") from exc
    if any(registered.name == tool.name for registered in _registered):
        raise ValueError(f"server tool is already registered: {tool.name}")
    _registered.append(tool)
    return tool_type


def rename_to_avoid_collisions(function: ChatFunctionTool, tools: Sequence[ChatTool]) -> ChatFunctionTool:
    names = {tool.function.name for tool in tools}
    if function.name not in names:
        return function
    index = 2
    while f"{function.name}_{index}" in names:
        index += 1
    return replace(function, name=f"{function.name}_{index}")


def _bind_tools(tools: Sequence[ChatTool]) -> tuple[list[ChatTool], dict[str, str]]:
    bound = list(tools)
    wire_names: dict[str, str] = {}
    for registered in _registered:
        function = rename_to_avoid_collisions(registered, bound)
        bound.append(ChatTool(function=function))
        wire_names[registered.name] = function.name
    return bound, wire_names


def _server_call_ids(message: ChatMessage) -> set[str]:
    record = message.memory.get("server_tools")
    if record is None:
        return set()
    if not isinstance(record, Mapping):
        raise TypeError("server tool message memory must be an object")
    call_ids = record.get("call_ids")
    if call_ids is None:
        return set()
    if not isinstance(call_ids, list):
        raise TypeError("server tool call ids memory must be an array")
    result: set[str] = set()
    for call_id in call_ids:
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("server tool call id must be a non-empty string")
        if call_id in result:
            raise ValueError("server tool call ids must not contain duplicates")
        result.add(call_id)
    return result


def _rebind_history(messages: Sequence[ChatMessage], wire_names: Mapping[str, str]) -> list[ChatMessage]:
    rebound: list[ChatMessage] = []
    for message in messages:
        if not message.is_assistant() or not message.tool_calls:
            rebound.append(message)
            continue
        server_call_ids = _server_call_ids(message)
        tool_calls: list[ChatToolCall] = []
        changed = False
        for call in message.tool_calls:
            wire_name = None if call.id not in server_call_ids else wire_names.get(call.name)
            if wire_name is None or wire_name == call.name:
                tool_calls.append(call)
                continue
            tool_calls.append(replace(call, name=wire_name))
            changed = True
        rebound.append(replace(message, tool_calls=tool_calls) if changed else message)
    return rebound


def _registered_by_wire(request: ChatCompletionRequest) -> dict[str, ServerTool]:
    bound = [tool.function for tool in request.tools if isinstance(tool.function, ServerTool)]
    if len(bound) != len(_registered):
        raise RuntimeError("finalized request does not contain every registered server tool")
    return {function.name: registered for function, registered in zip(bound, _registered, strict=True)}


def _canonicalize_message(message: ChatMessage, canonical_by_wire: Mapping[str, str]) -> ChatMessage:
    if not message.is_assistant() or not message.tool_calls:
        return message
    existing = _server_call_ids(message)
    server_call_ids: list[str] = []
    tool_calls: list[ChatToolCall] = []
    for call in message.tool_calls:
        canonical_name = canonical_by_wire.get(call.name)
        if canonical_name is None and call.id not in existing:
            tool_calls.append(call)
            continue
        server_call_ids.append(call.id)
        if canonical_name is None or call.name == canonical_name:
            tool_calls.append(call)
            continue
        tool_calls.append(replace(call, name=canonical_name))
    if not server_call_ids:
        return message
    record = message.memory.get("server_tools", {})
    if not isinstance(record, Mapping):  # pragma: no cover - validated by _server_call_ids
        raise TypeError("server tool message memory must be an object")
    return replace(
        message,
        tool_calls=tool_calls,
        memory={
            **message.memory,
            "server_tools": {
                **record,
                "call_ids": server_call_ids,
            },
        },
    )


def _canonicalize_result(result: ChatCompletionResult, canonical_by_wire: Mapping[str, str]) -> ChatCompletionResult:
    message = _canonicalize_message(result.message, canonical_by_wire)
    return result if message is result.message else replace(result, message=message)


def _canonicalize_snapshot(snapshot: Snapshot, canonical_by_wire: Mapping[str, str]) -> Snapshot:
    messages = tuple(_canonicalize_message(message, canonical_by_wire) for message in snapshot.messages)
    results = tuple(_canonicalize_result(result, canonical_by_wire) for result in snapshot.results)
    if messages == snapshot.messages and results == snapshot.results:
        return snapshot
    return replace(snapshot, messages=messages, results=results)


def _execution_record(tool: ServerTool, call: ChatToolCall, *, error: str | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "tool": tool.name,
        "arguments": call.arguments,
    }
    if error is not None:
        record["error"] = error
    return record


def _budget_exhausted_message(tool: ServerTool, call: ChatToolCall) -> ChatMessage:
    return ChatMessage(
        role="tool",
        tool_call_id=call.id,
        content=BUDGET_TOOL_OUTPUT,
        memory={
            "server_tools": _execution_record(tool, call, error="budget_exhausted"),
        },
    )


def _record_server_tool_call(tool: ServerTool, call: ChatToolCall, message: ChatMessage) -> ChatMessage:
    if not isinstance(message, ChatMessage):
        raise TypeError(f"server tool {tool.name!r} must return a ChatMessage")
    if message.role != ChatRole.TOOL:
        raise ValueError(f"server tool {tool.name!r} must return a tool message")
    if message.tool_call_id != call.id:
        raise ValueError(f"server tool {tool.name!r} returned the wrong tool call id")
    return replace(
        message,
        memory={
            **message.memory,
            "server_tools": _execution_record(tool, call),
        },
    )


@bus.listen("response.request")
async def _inject_server_tools(state: State, *, next) -> ChatCompletionRequest:
    request = await next(state=state)
    tools, wire_names = _bind_tools(request.tools)
    return replace(request, messages=_rebind_history(request.messages, wire_names), tools=tools)


@bus.listen("response.snapshot")
async def _canonicalize_server_tool_calls(
    request: ChatCompletionRequest,
    snapshot: Snapshot,
    *,
    next,
) -> Snapshot:
    snapshot = await next(request=request, snapshot=snapshot)
    canonical_by_wire = {wire_name: tool.name for wire_name, tool in _registered_by_wire(request).items()}
    return _canonicalize_snapshot(snapshot, canonical_by_wire)


@bus.listen("response.completion")
async def _execute_server_tools(
    state: State,
    request: ChatCompletionRequest,
    *,
    next,
) -> ChatCompletionResult:
    result = await next(state=state, request=request)
    tools = {tool.name: tool for tool in _registered_by_wire(request).values()}
    server_call_ids = _server_call_ids(result.message)
    for call in result.message.tool_calls:
        if call.id not in server_call_ids:
            continue
        tool = tools.get(call.name)
        if tool is None:  # pragma: no cover - snapshot canonicalization uses the same finalized request
            raise RuntimeError(f"canonical server tool is not registered: {call.name}")
        try:
            message = await tool(state, call)
        except CompletionBudgetExhaustedError:
            message = _budget_exhausted_message(tool, call)
        else:
            message = _record_server_tool_call(tool, call, message)
        state.threads["main"].messages.append(message)
    return result


__all__ = ["BUDGET_TOOL_OUTPUT", "ServerTool", "register", "rename_to_avoid_collisions"]
