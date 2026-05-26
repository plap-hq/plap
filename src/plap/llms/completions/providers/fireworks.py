from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import aiohttp
import httpx
import structlog
from fireworks.client import AsyncFireworks
from fireworks.client.error import (
    APITimeoutError,
    AuthenticationError,
    FireworksError,
    InvalidRequestError,
    PermissionError,
    RateLimitError,
)

from plap.llms.completions.client import Call, Provider, Quirk
from plap.llms.completions.common import close_stream_object, to_data
from plap.llms.completions.errors import (
    ChatCompletionAuthenticationError,
    ChatCompletionContextLengthExceededError,
    ChatCompletionInvalidRequestError,
    ChatCompletionProviderError,
    ChatCompletionRateLimitError,
    ChatCompletionTimeoutError,
    is_context_length_exceeded_error,
)
from plap.llms.completions.providers._transport import (
    log_transport_error,
    timeout_error_message,
)
from plap.llms.completions.quirks import Only, Set, SystemRole
from plap.logging import log_debug

logger = structlog.get_logger(__name__)


def _fireworks_base_url(client: Any) -> str | None:
    client_v1 = getattr(client, "_client_v1", None)
    base_url = getattr(client_v1, "base_url", None)
    if base_url is None:
        base_url = getattr(client, "base_url", None)
    return str(base_url) if base_url is not None else None


def _fireworks_timeout_seconds(client: Any) -> float | None:
    client_v1 = getattr(client, "_client_v1", None)
    timeout = getattr(client_v1, "request_timeout", None)
    if timeout is None:
        timeout = getattr(client, "timeout", None)
    if timeout is None:
        return None
    try:
        return float(timeout)
    except (TypeError, ValueError):
        return None


def _log_fireworks_transport_error(*, provider: str, client: Any, call: Call, exc: Exception, streaming: bool) -> None:
    log_transport_error(
        log_fn=log_debug,
        logger=logger,
        provider=provider,
        base_url=_fireworks_base_url(client),
        client_max_retries=None,
        call=call,
        exc=exc,
        streaming=streaming,
        should_log=isinstance(exc, (APITimeoutError, httpx.RequestError, aiohttp.ClientError, asyncio.TimeoutError)),
        extra_context={"client_timeout_seconds": _fireworks_timeout_seconds(client)},
    )


def _fireworks_context_length_exceeded_error(exc: InvalidRequestError) -> ChatCompletionContextLengthExceededError | None:
    if is_context_length_exceeded_error(exc):
        return ChatCompletionContextLengthExceededError(str(exc))
    return None


def normalize_fireworks_error(exc: Exception) -> ChatCompletionProviderError:
    if isinstance(exc, ChatCompletionProviderError):
        return exc
    if isinstance(exc, (APITimeoutError, httpx.TimeoutException, asyncio.TimeoutError)):
        return ChatCompletionTimeoutError(timeout_error_message(exc))
    if isinstance(exc, (AuthenticationError, PermissionError)):
        return ChatCompletionAuthenticationError(str(exc))
    if isinstance(exc, RateLimitError):
        return ChatCompletionRateLimitError(str(exc))
    if isinstance(exc, InvalidRequestError):
        context_length_error = _fireworks_context_length_exceeded_error(exc)
        if context_length_error is not None:
            return context_length_error
        return ChatCompletionInvalidRequestError(str(exc))
    if isinstance(exc, (httpx.RequestError, aiohttp.ClientError)):
        return ChatCompletionProviderError(str(exc))
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
            _log_fireworks_transport_error(provider=self.name, client=self._client, call=call, exc=exc, streaming=False)
            raise normalize_fireworks_error(exc) from exc
        return to_data(response)

    def stream(self, call: Call) -> AsyncIterator[dict[str, Any]]:
        async def run() -> AsyncIterator[dict[str, Any]]:
            stream: Any | None = None
            try:
                stream = self._client.chat.completions.acreate(**call.body)
                async for chunk in stream:
                    yield to_data(chunk)
            except Exception as exc:
                _log_fireworks_transport_error(provider=self.name, client=self._client, call=call, exc=exc, streaming=True)
                raise normalize_fireworks_error(exc) from exc
            finally:
                if stream is not None:
                    await close_stream_object(stream)

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
    "context_length_exceeded_behavior",
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
        quirks=(SystemRole(), Only(*FIREWORKS_FIELDS), Set("context_length_exceeded_behavior", "error")),
        models=FIREWORKS_MODELS,
    )


__all__ = [
    "FireworksProvider",
    "build_fireworks_provider",
    "normalize_fireworks_error",
]
