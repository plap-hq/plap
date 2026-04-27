from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fireworks.client import AsyncFireworks
from fireworks.client.error import (
    AuthenticationError,
    FireworksError,
    InvalidRequestError,
    PermissionError,
    RateLimitError,
)

from plap.llms.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResult,
    IChatCompletionClient,
)
from plap.llms.errors import (
    ChatCompletionAuthenticationError,
    ChatCompletionInvalidRequestError,
    ChatCompletionProviderError,
    ChatCompletionRateLimitError,
)
from plap.llms.openai_compatible import (
    _set,
    completion_result_from_provider,
    from_chat_completion_chunk,
    to_openai_chat_params,
)


class FireworksChatCompletionClient(IChatCompletionClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._client = client or AsyncFireworks(api_key=api_key, base_url=base_url)

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        params = to_fireworks_chat_params(request)
        try:
            response = await self._client.chat.completions.acreate(
                **params, stream=False
            )
        except Exception as exc:
            raise _normalize_fireworks_error(exc) from exc
        return completion_result_from_provider(response)

    async def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionDelta]:
        params = to_fireworks_chat_params(request)
        try:
            stream = self._client.chat.completions.acreate(**params, stream=True)
        except Exception as exc:
            raise _normalize_fireworks_error(exc) from exc
        async for chunk in stream:
            yield from_chat_completion_chunk(chunk)


def to_fireworks_chat_params(request: ChatCompletionRequest) -> dict[str, Any]:
    openai_params = to_openai_chat_params(
        request,
        stream=False,
        developer_role="system",
    )
    params: dict[str, Any] = {
        "model": openai_params["model"],
        "messages": openai_params["messages"],
    }

    passthrough_fields = [
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "stop",
        "seed",
        "n",
        "reasoning_effort",
        "user",
        "prompt_cache_key",
        "metadata",
        "service_tier",
        "prediction",
    ]
    for field in passthrough_fields:
        _set(params, field, openai_params.get(field))

    _set(params, "max_completion_tokens", request.max_completion_tokens)
    return params


def _normalize_fireworks_error(exc: Exception) -> ChatCompletionProviderError:
    if isinstance(exc, (AuthenticationError, PermissionError)):
        return ChatCompletionAuthenticationError(str(exc))
    if isinstance(exc, RateLimitError):
        return ChatCompletionRateLimitError(str(exc))
    if isinstance(exc, InvalidRequestError):
        return ChatCompletionInvalidRequestError(str(exc))
    if isinstance(exc, FireworksError):
        return ChatCompletionProviderError(str(exc))
    return ChatCompletionProviderError(str(exc))
