from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import msgspec
from openai import (
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from plap.llms.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatFinishReason,
    ChatMessage,
    ChatPrediction,
    ChatResponseFormat,
    ChatRole,
    ChatStreamOptions,
    ChatTool,
    ChatToolCall,
    ChatToolCallDelta,
    ChatToolChoice,
    ChatUsage,
    IChatCompletionClient,
)
from plap.llms.errors import (
    ChatCompletionAuthenticationError,
    ChatCompletionContextLengthExceededError,
    ChatCompletionInvalidRequestError,
    ChatCompletionProviderError,
    ChatCompletionRateLimitError,
    is_context_length_exceeded_code,
    is_context_length_exceeded_error,
)

COMMON_CHAT_FIELDS = (
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "stop",
    "seed",
    "n",
)

OPENAI_CHAT_FIELDS = (
    *COMMON_CHAT_FIELDS,
    "logprobs",
    "top_logprobs",
    "reasoning_effort",
    "user",
    "prompt_cache_key",
    "metadata",
    "service_tier",
    "prediction",
)

OPENAI_REASONING_TEXT_FIELDS = ("reasoning_content",)


@dataclass(frozen=True)
class ChatProviderProfile:
    developer_role: ChatRole
    passthrough_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "developer_role", ChatRole(self.developer_role))
        if self.developer_role not in {ChatRole.DEVELOPER, ChatRole.SYSTEM}:
            raise ValueError("developer_role must be developer or system")


@dataclass(slots=True)
class _OpenAIStreamState:
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

    def inferred_finish_reason(self) -> ChatFinishReason | None:
        if self.saw_finish_reason:
            return None
        if self.saw_tool_calls:
            return ChatFinishReason.TOOL_CALLS
        if self.saw_content:
            return ChatFinishReason.STOP
        return None

    def inferred_terminal_delta(self) -> ChatCompletionDelta | None:
        finish_reason = self.inferred_finish_reason()
        if finish_reason is None:
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


def _raise_incomplete_stream_error() -> None:
    raise ChatCompletionProviderError(
        "stream ended without finish_reason and without inferable content or tool calls"
    )


class OpenAICompatibleChatCompletionClient(IChatCompletionClient):
    _reasoning_text_fields = OPENAI_REASONING_TEXT_FIELDS

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        developer_role: ChatRole = ChatRole.DEVELOPER,
    ) -> None:
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._developer_role = ChatRole(developer_role)
        if self._developer_role not in {ChatRole.DEVELOPER, ChatRole.SYSTEM}:
            raise ValueError("developer_role must be developer or system")

    def _chat_params(
        self,
        request: ChatCompletionRequest,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        return to_openai_chat_params(
            request,
            stream=stream,
            developer_role=self._developer_role,
        )

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        try:
            response = await self._client.chat.completions.create(**self._chat_params(request, stream=False))
        except Exception as exc:
            raise _normalize_openai_error(exc) from exc
        return completion_result_from_provider(response, reasoning_text_fields=self._reasoning_text_fields)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionDelta]:
        stream: Any | None = None
        state = _OpenAIStreamState()
        try:
            stream = await self._client.chat.completions.create(**self._chat_params(request, stream=True))
            async for chunk in stream:
                delta = from_chat_completion_chunk(chunk, reasoning_text_fields=self._reasoning_text_fields)
                state.apply(delta)
                yield delta
            inferred_delta = state.inferred_terminal_delta()
            if inferred_delta is not None:
                yield inferred_delta
            elif not state.saw_finish_reason:
                _raise_incomplete_stream_error()
        except Exception as exc:
            raise _normalize_openai_error(exc) from exc
        finally:
            if stream is not None:
                await _close_stream_object(stream)


def to_openai_chat_params(
    request: ChatCompletionRequest,
    *,
    stream: bool,
    developer_role: ChatRole = ChatRole.DEVELOPER,
) -> dict[str, Any]:
    return build_chat_params(
        request,
        stream=stream,
        profile=ChatProviderProfile(
            developer_role=developer_role,
            passthrough_fields=OPENAI_CHAT_FIELDS,
        ),
    )


def build_chat_params(
    request: ChatCompletionRequest,
    *,
    stream: bool,
    profile: ChatProviderProfile,
) -> dict[str, Any]:
    values = _chat_param_values(
        request,
        stream=stream,
        developer_role=profile.developer_role,
    )
    params: dict[str, Any] = {
        "model": values["model"],
        "messages": values["messages"],
        "stream": stream,
    }
    for field in profile.passthrough_fields:
        _set(params, field, values.get(field))
    _set(params, "max_completion_tokens", values.get("max_completion_tokens"))
    if stream:
        _set(params, "stream_options", values.get("stream_options"))
    return params


def _chat_param_values(
    request: ChatCompletionRequest,
    *,
    stream: bool,
    developer_role: ChatRole,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "model": request.model,
        "messages": [_message_to_param(message, developer_role=developer_role) for message in request.messages],
        "stream": stream,
    }
    values["tools"] = [_tool_to_param(tool) for tool in request.tools] if request.tools else None
    values["tool_choice"] = _tool_choice_to_param(request.tool_choice)
    values["parallel_tool_calls"] = request.parallel_tool_calls
    values["response_format"] = _response_format_to_param(request.response_format)
    values["max_completion_tokens"] = request.max_completion_tokens
    values["temperature"] = request.temperature
    values["top_p"] = request.top_p
    values["top_k"] = request.top_k
    values["frequency_penalty"] = request.frequency_penalty
    values["presence_penalty"] = request.presence_penalty
    values["logit_bias"] = request.logit_bias
    values["logprobs"] = request.logprobs
    values["top_logprobs"] = request.top_logprobs
    values["stop"] = request.stop
    values["seed"] = request.seed
    values["n"] = request.n
    values["reasoning_effort"] = request.reasoning_effort
    values["stream_options"] = _stream_options_to_param(request.stream_options)
    values["user"] = request.user
    values["prompt_cache_key"] = request.prompt_cache_key
    values["metadata"] = request.metadata
    values["service_tier"] = request.service_tier
    values["prediction"] = _prediction_to_param(request.prediction)
    return values


def _message_to_param(
    message: ChatMessage,
    *,
    developer_role: ChatRole,
) -> dict[str, Any]:
    role = developer_role if message.role == ChatRole.DEVELOPER else message.role
    value: dict[str, Any] = {"role": role}
    _set(value, "content", message.content)
    _set(value, "name", message.name)
    _set(value, "tool_call_id", message.tool_call_id)
    _set(
        value,
        "tool_calls",
        [_tool_call_to_param(tool_call) for tool_call in message.tool_calls or []] or None,
    )
    if message.role == ChatRole.ASSISTANT:
        _set(value, "refusal", message.refusal)
        _set(value, "reasoning_content", message.reasoning_content)
        _set(value, "reasoning_details", message.reasoning_details)
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
    return value or None


def _prediction_to_param(prediction: ChatPrediction | None) -> dict[str, Any] | None:
    if prediction is None:
        return None
    return {"type": prediction.type, "content": prediction.content}


def _reasoning_text_from_provider(
    value: Any,
    *,
    reasoning_text_fields: tuple[str, ...],
) -> Any:
    for field in reasoning_text_fields:
        reasoning = _get(value, field)
        if reasoning is not None:
            return reasoning
    return None


def completion_result_from_provider(
    response: Any,
    *,
    reasoning_text_fields: tuple[str, ...] = OPENAI_REASONING_TEXT_FIELDS,
) -> ChatCompletionResult:
    choice = _first(_get(response, "choices"))
    message = _get(choice, "message")
    return ChatCompletionResult(
        id=_get(response, "id"),
        model=_get(response, "model"),
        created_at=_float_or_none(_get(response, "created")),
        message=ChatMessage(
            role="assistant",
            content=_get(message, "content"),
            refusal=_get(message, "refusal"),
            reasoning_content=_reasoning_text_from_provider(message, reasoning_text_fields=reasoning_text_fields),
            reasoning_details=_get(message, "reasoning_details"),
            tool_calls=_tool_calls_from_provider(_get(message, "tool_calls")),
        ),
        finish_reason=_finish_reason(_get(choice, "finish_reason")),
        usage=_usage_from_provider(_get(response, "usage")),
        system_fingerprint=_get(response, "system_fingerprint"),
        service_tier=_get(response, "service_tier"),
    )


def from_chat_completion_chunk(
    chunk: Any,
    *,
    reasoning_text_fields: tuple[str, ...] = OPENAI_REASONING_TEXT_FIELDS,
) -> ChatCompletionDelta:
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
        reasoning_delta=_reasoning_text_from_provider(delta, reasoning_text_fields=reasoning_text_fields),
        reasoning_details_delta=_get(delta, "reasoning_details"),
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
        reasoning_tokens=_get(completion_details, "reasoning_tokens") or _get(usage, "reasoning_tokens"),
    )


def _openai_context_length_exceeded_error(exc: BadRequestError) -> ChatCompletionContextLengthExceededError | None:
    body = exc.body if isinstance(exc.body, dict) else None
    error = body.get("error") if isinstance(body, dict) and isinstance(body.get("error"), dict) else None
    code_candidates = (
        error.get("code") if isinstance(error, dict) else None,
        error.get("type") if isinstance(error, dict) else None,
        body.get("code") if isinstance(body, dict) else None,
        body.get("type") if isinstance(body, dict) else None,
    )
    if any(isinstance(code, str) and is_context_length_exceeded_code(code) for code in code_candidates):
        return ChatCompletionContextLengthExceededError(str(exc))

    if is_context_length_exceeded_error(exc):
        return ChatCompletionContextLengthExceededError(str(exc))
    return None


def _normalize_openai_error(exc: Exception) -> ChatCompletionProviderError:
    if isinstance(exc, ChatCompletionProviderError):
        return exc
    if isinstance(exc, AuthenticationError):
        return ChatCompletionAuthenticationError(str(exc))
    if isinstance(exc, RateLimitError):
        return ChatCompletionRateLimitError(str(exc))
    if isinstance(exc, BadRequestError):
        context_length_error = _openai_context_length_exceeded_error(exc)
        if context_length_error is not None:
            return context_length_error
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
    return msgspec.json.encode(value).decode()


async def _close_stream_object(stream: Any) -> None:
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
