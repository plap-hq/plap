from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import msgspec

from plap.llms.completions.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatFinishReason,
    ChatMessage,
    ChatPrediction,
    ChatResponseFormat,
    ChatStreamOptions,
    ChatTool,
    ChatToolCall,
    ChatToolCallDelta,
    ChatToolChoice,
    ChatUsage,
)
from plap.llms.completions.errors import ChatCompletionProviderError


def _set(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


def _get(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _first(value: Any) -> Any:
    if not value:
        return None
    return value[0]


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def _finish_reason(value: Any) -> ChatFinishReason | None:
    return value


def _stringify_json_value(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return msgspec.json.encode(value).decode()


def _message_body(message: ChatMessage) -> dict[str, Any]:
    value: dict[str, Any] = {"role": message.role}
    _set(value, "content", message.content)
    _set(value, "name", message.name)
    _set(value, "tool_call_id", message.tool_call_id)
    _set(
        value,
        "tool_calls",
        [_tool_call_body(tool_call) for tool_call in message.tool_calls or []] or None,
    )
    if message.role == "assistant":
        _set(value, "refusal", message.refusal)
        _set(value, "reasoning_content", message.reasoning_content)
        _set(value, "reasoning_details", message.reasoning_details)
    return value


def _tool_body(tool: ChatTool) -> dict[str, Any]:
    function: dict[str, Any] = {"name": tool.function.name}
    _set(function, "parameters", tool.function.parameters)
    _set(function, "strict", tool.function.strict)
    _set(function, "description", tool.function.description)
    return {"type": tool.type, "function": function}


def _tool_call_body(tool_call: ChatToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {"name": tool_call.name, "arguments": tool_call.arguments},
    }


def _tool_choice_body(
    tool_choice: ChatToolChoice | None,
) -> str | dict[str, Any] | None:
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    return {"type": tool_choice.type, "function": {"name": tool_choice.name}}


def _response_format_body(
    response_format: ChatResponseFormat | None,
) -> dict[str, Any] | None:
    if response_format is None:
        return None
    if response_format.type != "json_schema":
        return {"type": response_format.type}
    json_schema: dict[str, Any] = {
        "name": response_format.name,
        "schema": response_format.schema or {},
    }
    _set(json_schema, "strict", response_format.strict)
    _set(json_schema, "description", response_format.description)
    return {"type": "json_schema", "json_schema": json_schema}


def _stream_options_body(
    stream_options: ChatStreamOptions | None,
) -> dict[str, Any] | None:
    if stream_options is None:
        return None
    value: dict[str, Any] = {}
    _set(value, "include_usage", stream_options.include_usage)
    return value or None


def _prediction_body(prediction: ChatPrediction | None) -> dict[str, Any] | None:
    if prediction is None:
        return None
    return {"type": prediction.type, "content": prediction.content}


def build_chat_body(
    request: ChatCompletionRequest,
    *,
    stream: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": request.model,
        "messages": [_message_body(message) for message in request.messages],
        "stream": stream,
    }
    _set(body, "tools", [_tool_body(tool) for tool in request.tools] if request.tools else None)
    _set(body, "tool_choice", _tool_choice_body(request.tool_choice))
    _set(body, "parallel_tool_calls", request.parallel_tool_calls)
    _set(body, "response_format", _response_format_body(request.response_format))
    _set(body, "max_completion_tokens", request.max_completion_tokens)
    _set(body, "temperature", request.temperature)
    _set(body, "top_p", request.top_p)
    _set(body, "top_k", request.top_k)
    _set(body, "frequency_penalty", request.frequency_penalty)
    _set(body, "presence_penalty", request.presence_penalty)
    _set(body, "logit_bias", request.logit_bias)
    _set(body, "logprobs", request.logprobs)
    _set(body, "top_logprobs", request.top_logprobs)
    _set(body, "stop", request.stop)
    _set(body, "seed", request.seed)
    _set(body, "n", request.n)
    _set(body, "reasoning_effort", request.reasoning_effort)
    _set(body, "user", request.user)
    _set(body, "prompt_cache_key", request.prompt_cache_key)
    _set(body, "metadata", request.metadata)
    _set(body, "service_tier", request.service_tier)
    _set(body, "prediction", _prediction_body(request.prediction))
    if stream:
        _set(body, "stream_options", _stream_options_body(request.stream_options))
    return body


def to_data(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): to_data(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_data(item) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return to_data(model_dump())

    to_dict = getattr(value, "dict", None)
    if callable(to_dict):
        return to_data(to_dict())

    if hasattr(value, "__dict__"):
        return to_data(vars(value))

    return value


def _tool_calls_from_data(tool_calls: Any) -> list[ChatToolCall] | None:
    if not tool_calls:
        return None
    normalized: list[ChatToolCall] = []
    for index, tool_call in enumerate(tool_calls):
        function = _get(tool_call, "function")
        name = _get(function, "name")
        arguments = _stringify_json_value(_get(function, "arguments"))
        if name is None or arguments is None:
            continue
        normalized.append(
            ChatToolCall(
                id=_get(tool_call, "id") or f"tool_call_{index}",
                name=name,
                arguments=arguments,
            )
        )
    return normalized or None


def _usage_from_data(usage: Any) -> ChatUsage | None:
    if usage is None:
        return None
    prompt_details = _get(usage, "prompt_tokens_details")
    completion_details = _get(usage, "completion_tokens_details")
    return ChatUsage(
        input_tokens=_get(usage, "prompt_tokens") or 0,
        output_tokens=_get(usage, "completion_tokens") or 0,
        total_tokens=_get(usage, "total_tokens") or 0,
        cached_tokens=_get(prompt_details, "cached_tokens"),
        reasoning_tokens=_get(completion_details, "reasoning_tokens") or _get(usage, "reasoning_tokens"),
    )


def completion_result_from_data(
    response: dict[str, Any],
    *,
    request: ChatCompletionRequest,
) -> ChatCompletionResult:
    choice = _first(_get(response, "choices"))
    message = _get(choice, "message") or {}
    return ChatCompletionResult(
        id=_get(response, "id"),
        model=_get(response, "model") or request.model,
        created_at=_float_or_none(_get(response, "created")),
        message=ChatMessage(
            role="assistant",
            content=_get(message, "content"),
            refusal=_get(message, "refusal"),
            reasoning_content=_get(message, "reasoning_content"),
            reasoning_details=_get(message, "reasoning_details"),
            tool_calls=_tool_calls_from_data(_get(message, "tool_calls")),
        ),
        finish_reason=_finish_reason(_get(choice, "finish_reason")),
        usage=_usage_from_data(_get(response, "usage")),
        system_fingerprint=_get(response, "system_fingerprint"),
        service_tier=_get(response, "service_tier"),
    )


def delta_from_data(
    chunk: dict[str, Any],
    *,
    request: ChatCompletionRequest,
) -> ChatCompletionDelta:
    choice = _first(_get(chunk, "choices"))
    delta = _get(choice, "delta") or {}
    tool_call_delta = _first(_get(delta, "tool_calls"))
    function = _get(tool_call_delta, "function")
    return ChatCompletionDelta(
        id=_get(chunk, "id"),
        model=_get(chunk, "model") or request.model,
        created_at=_float_or_none(_get(chunk, "created")),
        choice_index=_get(choice, "index") or 0,
        content_delta=_get(delta, "content"),
        refusal_delta=_get(delta, "refusal"),
        reasoning_delta=_get(delta, "reasoning_content"),
        reasoning_details_delta=_get(delta, "reasoning_details"),
        tool_call_delta=ChatToolCallDelta(
            index=_get(tool_call_delta, "index") or 0,
            id=_get(tool_call_delta, "id"),
            name=_get(function, "name"),
            arguments_delta=_stringify_json_value(_get(function, "arguments")),
        )
        if tool_call_delta is not None
        else None,
        finish_reason=_finish_reason(_get(choice, "finish_reason")),
        usage=_usage_from_data(_get(chunk, "usage")),
        system_fingerprint=_get(chunk, "system_fingerprint"),
        service_tier=_get(chunk, "service_tier"),
    )


def response_to_stream_chunks(
    response: dict[str, Any],
    *,
    request: ChatCompletionRequest,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = completion_result_from_data(response, request=request)
    return (
        {
            "id": result.id,
            "model": result.model,
            "created": result.created_at,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "delta": {
                        "content": result.message.content,
                        "refusal": result.message.refusal,
                        "reasoning_content": result.message.reasoning_content,
                        "reasoning_details": result.message.reasoning_details,
                    },
                }
            ],
        },
        {
            "id": result.id,
            "model": result.model,
            "created": result.created_at,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": result.finish_reason,
                    "delta": {},
                }
            ],
            "usage": {
                "prompt_tokens": result.usage.input_tokens,
                "completion_tokens": result.usage.output_tokens,
                "total_tokens": result.usage.total_tokens,
                "prompt_tokens_details": {
                    "cached_tokens": result.usage.cached_tokens,
                }
                if result.usage and result.usage.cached_tokens is not None
                else None,
                "completion_tokens_details": {
                    "reasoning_tokens": result.usage.reasoning_tokens,
                }
                if result.usage and result.usage.reasoning_tokens is not None
                else None,
                "reasoning_tokens": result.usage.reasoning_tokens if result.usage is not None else None,
            }
            if result.usage is not None
            else None,
            "system_fingerprint": result.system_fingerprint,
            "service_tier": result.service_tier,
        },
    )


@dataclass(slots=True)
class StreamState:
    last_id: str | None = None
    last_model: str | None = None
    last_created_at: float | None = None
    last_choice_index: int = 0
    last_system_fingerprint: str | None = None
    last_service_tier: str | None = None
    saw_content: bool = False
    saw_tool_calls: bool = False
    saw_finish_reason: bool = False

    def apply(self, delta: ChatCompletionDelta) -> None:
        if delta.id is not None:
            self.last_id = delta.id
        if delta.model is not None:
            self.last_model = delta.model
        if delta.created_at is not None:
            self.last_created_at = delta.created_at
        self.last_choice_index = delta.choice_index
        if delta.system_fingerprint is not None:
            self.last_system_fingerprint = delta.system_fingerprint
        if delta.service_tier is not None:
            self.last_service_tier = delta.service_tier
        if delta.content_delta is not None:
            self.saw_content = True
        if delta.tool_call_delta is not None:
            self.saw_tool_calls = True
        if delta.finish_reason is not None:
            self.saw_finish_reason = True

    def inferred_terminal_delta(self) -> ChatCompletionDelta | None:
        if self.saw_finish_reason:
            return None
        if self.saw_tool_calls:
            finish_reason = ChatFinishReason.TOOL_CALLS
        elif self.saw_content:
            finish_reason = ChatFinishReason.STOP
        else:
            return None
        return ChatCompletionDelta(
            id=self.last_id,
            model=self.last_model,
            created_at=self.last_created_at,
            choice_index=self.last_choice_index,
            finish_reason=finish_reason,
            system_fingerprint=self.last_system_fingerprint,
            service_tier=self.last_service_tier,
        )


def raise_incomplete_stream_error() -> None:
    raise ChatCompletionProviderError(
        "stream ended without finish_reason and without inferable content or tool calls"
    )


async def close_stream_object(stream: Any) -> None:
    aclose = getattr(stream, "aclose", None)
    if callable(aclose):
        try:
            result = aclose()
            if inspect.isawaitable(result):
                await result
        except Exception:
            return
        return

    close = getattr(stream, "close", None)
    if callable(close):
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception:
            return


__all__ = [
    "StreamState",
    "build_chat_body",
    "close_stream_object",
    "completion_result_from_data",
    "delta_from_data",
    "raise_incomplete_stream_error",
    "response_to_stream_chunks",
    "to_data",
]
