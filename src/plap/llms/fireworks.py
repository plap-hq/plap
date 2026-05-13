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
    ChatCompletionContextLengthExceededError,
    ChatCompletionInvalidRequestError,
    ChatCompletionProviderError,
    ChatCompletionRateLimitError,
    is_context_length_exceeded_error,
)
from plap.llms.openai import (
    COMMON_CHAT_FIELDS,
    ChatProviderProfile,
    build_chat_params,
    completion_result_from_provider,
    from_chat_completion_chunk,
)

FIREWORKS_CHAT_FIELDS = (
    *COMMON_CHAT_FIELDS,
    "top_k",
    "logprobs",
    "top_logprobs",
    "reasoning_effort",
    "user",
    "prompt_cache_key",
    "metadata",
    "service_tier",
    "prediction",
)

FIREWORKS_CHAT_PROVIDER_PROFILE = ChatProviderProfile(
    developer_role="system",
    passthrough_fields=FIREWORKS_CHAT_FIELDS,
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
        params = to_fireworks_chat_params(request, stream=False)
        try:
            response = await self._client.chat.completions.acreate(**params)
        except Exception as exc:
            raise _normalize_fireworks_error(exc) from exc
        return completion_result_from_provider(response)

    async def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionDelta]:
        params = to_fireworks_chat_params(request, stream=True)
        try:
            stream = self._client.chat.completions.acreate(**params)
            async for chunk in stream:
                yield from_chat_completion_chunk(chunk)
        except Exception as exc:
            raise _normalize_fireworks_error(exc) from exc


def to_fireworks_chat_params(
    request: ChatCompletionRequest,
    *,
    stream: bool,
) -> dict[str, Any]:
    return build_chat_params(
        request,
        stream=stream,
        profile=FIREWORKS_CHAT_PROVIDER_PROFILE,
    )


def _fireworks_context_length_exceeded_error(exc: InvalidRequestError) -> ChatCompletionContextLengthExceededError | None:
    if is_context_length_exceeded_error(exc):
        return ChatCompletionContextLengthExceededError(str(exc))
    return None


def _normalize_fireworks_error(exc: Exception) -> ChatCompletionProviderError:
    if isinstance(exc, (AuthenticationError, PermissionError)):
        return ChatCompletionAuthenticationError(str(exc))
    if isinstance(exc, RateLimitError):
        return ChatCompletionRateLimitError(str(exc))
    if isinstance(exc, InvalidRequestError):
        context_length_error = _fireworks_context_length_exceeded_error(exc)
        if context_length_error is not None:
            return context_length_error
        return ChatCompletionInvalidRequestError(str(exc))
    if isinstance(exc, FireworksError):
        return ChatCompletionProviderError(str(exc))
    return ChatCompletionProviderError(str(exc))
