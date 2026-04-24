from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from openai import (
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from plap.llms.chat import (
    ChatAssistantMessage,
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
from plap.llms.errors import (
    ChatCompletionAuthenticationError,
    ChatCompletionInvalidRequestError,
    ChatCompletionProviderError,
    ChatCompletionRateLimitError,
)


class OpenAICompatibleChatCompletionClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        try:
            response = await self._client.chat.completions.create(
                **to_openai_chat_params(request, stream=False)
            )
        except Exception as exc:
            raise _normalize_openai_error(exc) from exc
        return completion_result_from_provider(response)

    async def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionDelta]:
        try:
            stream = await self._client.chat.completions.create(
                **to_openai_chat_params(request, stream=True)
            )
        except Exception as exc:
            raise _normalize_openai_error(exc) from exc
        async for chunk in stream:
            yield from_chat_completion_chunk(chunk)


def to_openai_chat_params(
    request: ChatCompletionRequest, *, stream: bool
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": request.model,
        "messages": [_message_to_param(message) for message in request.messages],
        "stream": stream,
    }
    _set(
        params,
        "tools",
        [_tool_to_param(tool) for tool in request.tools] if request.tools else None,
    )
    _set(params, "tool_choice", _tool_choice_to_param(request.tool_choice))
    _set(params, "parallel_tool_calls", request.parallel_tool_calls)
    _set(params, "response_format", _response_format_to_param(request.response_format))
    _set(params, "max_completion_tokens", request.max_completion_tokens)
    _set(params, "temperature", request.temperature)
    _set(params, "top_p", request.top_p)
    _set(params, "frequency_penalty", request.frequency_penalty)
    _set(params, "presence_penalty", request.presence_penalty)
    _set(params, "logit_bias", request.logit_bias)
    _set(params, "logprobs", request.logprobs)
    _set(params, "top_logprobs", request.top_logprobs)
    _set(params, "stop", request.stop)
    _set(params, "seed", request.seed)
    _set(params, "n", request.n)
    _set(params, "reasoning_effort", request.reasoning_effort)
    _set(params, "verbosity", request.verbosity)
    _set(params, "stream_options", _stream_options_to_param(request.stream_options))
    _set(params, "user", request.user)
    _set(params, "safety_identifier", request.safety_identifier)
    _set(params, "prompt_cache_key", request.prompt_cache_key)
    _set(params, "prompt_cache_retention", request.prompt_cache_retention)
    _set(params, "metadata", request.metadata)
    _set(params, "service_tier", request.service_tier)
    _set(params, "prediction", _prediction_to_param(request.prediction))
    _set(params, "store", request.store)
    return params


def _message_to_param(message: ChatMessage) -> dict[str, Any]:
    value: dict[str, Any] = {"role": message.role}
    _set(value, "content", message.content)
    _set(value, "name", message.name)
    _set(value, "tool_call_id", message.tool_call_id)
    _set(
        value,
        "tool_calls",
        [_tool_call_to_param(tool_call) for tool_call in message.tool_calls or []]
        or None,
    )
    return value


def _tool_to_param(tool: ChatTool) -> dict[str, Any]:
    function: dict[str, Any] = {"name": tool.function.name}
    _set(function, "parameters", tool.function.parameters)
    _set(function, "strict", tool.function.strict)
    _set(function, "description", tool.function.description)
    return {"type": tool.type, "function": function}


def _tool_call_to_param(tool_call: ChatToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {"name": tool_call.name, "arguments": tool_call.arguments},
    }


def _tool_choice_to_param(
    tool_choice: ChatToolChoice | None,
) -> str | dict[str, Any] | None:
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    return {"type": tool_choice.type, "function": {"name": tool_choice.name}}


def _response_format_to_param(
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


def _stream_options_to_param(
    stream_options: ChatStreamOptions | None,
) -> dict[str, Any] | None:
    if stream_options is None:
        return None
    value: dict[str, Any] = {}
    _set(value, "include_usage", stream_options.include_usage)
    _set(value, "include_obfuscation", stream_options.include_obfuscation)
    return value or None


def _prediction_to_param(prediction: ChatPrediction | None) -> dict[str, Any] | None:
    if prediction is None:
        return None
    return {"type": prediction.type, "content": prediction.content}


def completion_result_from_provider(response: Any) -> ChatCompletionResult:
    choice = _first(_get(response, "choices"))
    message = _get(choice, "message")
    return ChatCompletionResult(
        id=_get(response, "id"),
        model=_get(response, "model"),
        created_at=_float_or_none(_get(response, "created")),
        message=ChatAssistantMessage(
            content=_get(message, "content"),
            refusal=_get(message, "refusal"),
            reasoning_content=_get(message, "reasoning_content"),
            tool_calls=_tool_calls_from_provider(_get(message, "tool_calls")),
        ),
        finish_reason=_finish_reason(_get(choice, "finish_reason")),
        usage=_usage_from_provider(_get(response, "usage")),
        system_fingerprint=_get(response, "system_fingerprint"),
        service_tier=_get(response, "service_tier"),
    )


def from_chat_completion_chunk(chunk: Any) -> ChatCompletionDelta:
    choice = _first(_get(chunk, "choices"))
    delta = _get(choice, "delta")
    tool_call_delta = _first(_get(delta, "tool_calls"))
    function = _get(tool_call_delta, "function")
    return ChatCompletionDelta(
        id=_get(chunk, "id"),
        model=_get(chunk, "model"),
        created_at=_float_or_none(_get(chunk, "created")),
        choice_index=_get(choice, "index") or 0,
        content_delta=_get(delta, "content"),
        refusal_delta=_get(delta, "refusal"),
        reasoning_delta=_get(delta, "reasoning_content"),
        tool_call_delta=ChatToolCallDelta(
            index=_get(tool_call_delta, "index") or 0,
            id=_get(tool_call_delta, "id"),
            name=_get(function, "name"),
            arguments_delta=_stringify_arguments(_get(function, "arguments")),
        )
        if tool_call_delta is not None
        else None,
        finish_reason=_finish_reason(_get(choice, "finish_reason")),
        usage=_usage_from_provider(_get(chunk, "usage")),
        system_fingerprint=_get(chunk, "system_fingerprint"),
        service_tier=_get(chunk, "service_tier"),
    )


def _tool_calls_from_provider(tool_calls: Any) -> list[ChatToolCall] | None:
    if not tool_calls:
        return None
    normalized: list[ChatToolCall] = []
    for index, tool_call in enumerate(tool_calls):
        function = _get(tool_call, "function")
        name = _get(function, "name")
        arguments = _stringify_arguments(_get(function, "arguments"))
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


def _usage_from_provider(usage: Any) -> ChatUsage | None:
    if usage is None:
        return None
    prompt_details = _get(usage, "prompt_tokens_details")
    completion_details = _get(usage, "completion_tokens_details")
    return ChatUsage(
        input_tokens=_get(usage, "prompt_tokens") or 0,
        output_tokens=_get(usage, "completion_tokens") or 0,
        total_tokens=_get(usage, "total_tokens") or 0,
        cached_tokens=_get(prompt_details, "cached_tokens"),
        reasoning_tokens=_get(completion_details, "reasoning_tokens"),
    )


def _normalize_openai_error(exc: Exception) -> ChatCompletionProviderError:
    if isinstance(exc, AuthenticationError):
        return ChatCompletionAuthenticationError(str(exc))
    if isinstance(exc, RateLimitError):
        return ChatCompletionRateLimitError(str(exc))
    if isinstance(exc, BadRequestError):
        return ChatCompletionInvalidRequestError(str(exc))
    if isinstance(exc, APIStatusError):
        return ChatCompletionProviderError(str(exc))
    return ChatCompletionProviderError(str(exc))


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


def _stringify_arguments(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"))
