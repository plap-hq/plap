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

from plap.llms.completions.client import Call, Provider, Quirk
from plap.llms.completions.common import to_data
from plap.llms.completions.errors import (
    ChatCompletionAuthenticationError,
    ChatCompletionContextLengthExceededError,
    ChatCompletionInvalidRequestError,
    ChatCompletionProviderError,
    ChatCompletionRateLimitError,
    is_context_length_exceeded_error,
)
from plap.llms.completions.quirks import Only, SystemRole


def _fireworks_context_length_exceeded_error(exc: InvalidRequestError) -> ChatCompletionContextLengthExceededError | None:
    if is_context_length_exceeded_error(exc):
        return ChatCompletionContextLengthExceededError(str(exc))
    return None


def normalize_fireworks_error(exc: Exception) -> ChatCompletionProviderError:
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


class FireworksProvider(Provider):
    def __init__(
        self,
        *,
        name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        quirks: tuple[Quirk, ...] = (),
        models: dict[str, tuple[Quirk, ...]] | None = None,
    ) -> None:
        super().__init__(name=name, quirks=quirks, models=models)
        self._client = client or AsyncFireworks(api_key=api_key, base_url=base_url)

    async def complete(self, call: Call) -> dict[str, Any]:
        try:
            response = await self._client.chat.completions.acreate(**call.body)
        except Exception as exc:
            raise normalize_fireworks_error(exc) from exc
        return to_data(response)

    def stream(self, call: Call) -> AsyncIterator[dict[str, Any]]:
        async def run() -> AsyncIterator[dict[str, Any]]:
            try:
                stream = self._client.chat.completions.acreate(**call.body)
                async for chunk in stream:
                    yield to_data(chunk)
            except Exception as exc:
                raise normalize_fireworks_error(exc) from exc

        return run()


FIREWORKS_FIELDS = (
    "model",
    "messages",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "stop",
    "seed",
    "n",
    "logprobs",
    "top_logprobs",
    "reasoning_effort",
    "user",
    "prompt_cache_key",
    "metadata",
    "service_tier",
    "prediction",
)
FIREWORKS_MODELS: dict[str, tuple[Quirk, ...]] = {
    "accounts/fireworks/models/gpt-oss-20b": (),
}


def build_fireworks_provider(*, api_key: str) -> Provider:
    return FireworksProvider(
        name="fireworks",
        api_key=api_key,
        quirks=(SystemRole(), Only(*FIREWORKS_FIELDS)),
        models=FIREWORKS_MODELS,
    )


__all__ = [
    "FireworksProvider",
    "build_fireworks_provider",
    "normalize_fireworks_error",
]
