from __future__ import annotations

import inspect
import random
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, replace

import anyio
import structlog

from plap.llms.completions.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResult,
    IChatCompletionClient,
)
from plap.llms.completions.common import has_output
from plap.llms.completions.errors import (
    ChatCompletionProviderError,
    ChatCompletionTimeoutError,
    ChatCompletionUnsupportedRequestError,
)
from plap.logging import log_debug

logger = structlog.get_logger(__name__)
DEFAULT_STREAM_FIRST_DELTA_TIMEOUT_SECONDS = 60.0
DEFAULT_STREAM_IDLE_DELTA_TIMEOUT_SECONDS = 60.0
TRANSIENT_RETRIES_BEFORE_FALLBACK = 2
TRANSIENT_RETRY_INITIAL_DELAY_SECONDS = 0.5
TRANSIENT_RETRY_MAX_DELAY_SECONDS = 8.0


def _model_attempts(model: str) -> tuple[str, ...]:
    attempts = tuple(part.strip() for part in model.split(","))
    if not attempts or any(not attempt for attempt in attempts):
        raise ChatCompletionUnsupportedRequestError(f"Model fallback chain {model!r} contains an empty model entry")
    return attempts


def _provider_model(model: str, prefix: str) -> str:
    provider_model = model.removeprefix(prefix)
    if not provider_model:
        raise ChatCompletionUnsupportedRequestError(f"No provider model configured for model {model!r}")
    return provider_model


def _unsupported_model(model: str) -> ChatCompletionUnsupportedRequestError:
    return ChatCompletionUnsupportedRequestError(f"No chat completion provider configured for model {model!r}")


def _transient_retry_delay_seconds(*, retry_index: int) -> float:
    base_delay = min(
        TRANSIENT_RETRY_INITIAL_DELAY_SECONDS * pow(2.0, retry_index - 1),
        TRANSIENT_RETRY_MAX_DELAY_SECONDS,
    )
    return (base_delay / 2.0) + (random.random() * (base_delay / 2.0))


async def _sleep_for_transient_retry(delay_seconds: float) -> None:
    await anyio.sleep(delay_seconds)


def _log_router_attempt_failed(
    *,
    request_model: str,
    attempt_model: str,
    attempt_index: int,
    attempt_count: int,
    next_attempt_model: str,
    exc: ChatCompletionProviderError | ChatCompletionUnsupportedRequestError,
    streaming: bool,
) -> None:
    log_debug(
        logger,
        "llm.router.attempt_failed",
        attempt_count=attempt_count,
        attempt_index=attempt_index,
        attempt_model=attempt_model,
        error_message=str(exc),
        error_type=type(exc).__name__,
        next_attempt_model=next_attempt_model,
        request_model=request_model,
        streaming=streaming,
    )


def _log_router_attempt_retry(
    *,
    request_model: str,
    attempt_model: str,
    attempt_index: int,
    attempt_count: int,
    retry_index: int,
    retry_count: int,
    retry_delay_seconds: float,
    exc: ChatCompletionProviderError | ChatCompletionUnsupportedRequestError,
    streaming: bool,
) -> None:
    log_debug(
        logger,
        "llm.router.attempt_retry",
        attempt_count=attempt_count,
        attempt_index=attempt_index,
        attempt_model=attempt_model,
        error_message=str(exc),
        error_type=type(exc).__name__,
        request_model=request_model,
        retry_count=retry_count,
        retry_delay_seconds=retry_delay_seconds,
        retry_index=retry_index,
        streaming=streaming,
    )


def _log_router_fallback_succeeded(
    *,
    request_model: str,
    winner_model: str,
    winning_attempt_index: int,
    attempt_count: int,
    streaming: bool,
) -> None:
    log_debug(
        logger,
        "llm.router.fallback_succeeded",
        attempt_count=attempt_count,
        request_model=request_model,
        streaming=streaming,
        winner_model=winner_model,
        winning_attempt_index=winning_attempt_index,
    )


def _is_transient_retryable_error(
    exc: ChatCompletionProviderError | ChatCompletionUnsupportedRequestError,
) -> bool:
    if isinstance(exc, ChatCompletionTimeoutError):
        return True
    return type(exc) is ChatCompletionProviderError


def _log_router_fallback_exhausted(
    *,
    request_model: str,
    final_attempt_model: str,
    attempt_count: int,
    exc: ChatCompletionProviderError | ChatCompletionUnsupportedRequestError,
    streaming: bool,
) -> None:
    log_debug(
        logger,
        "llm.router.fallback_exhausted",
        attempt_count=attempt_count,
        error_message=str(exc),
        error_type=type(exc).__name__,
        final_attempt_model=final_attempt_model,
        request_model=request_model,
        streaming=streaming,
    )


async def _close_async_iterator(iterator: AsyncIterator[ChatCompletionDelta]) -> None:
    aclose = getattr(iterator, "aclose", None)
    if callable(aclose):
        try:
            result = aclose()
            if inspect.isawaitable(result):
                await result
        except Exception:
            return
        return

    close = getattr(iterator, "close", None)
    if callable(close):
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception:
            return


async def _next_stream_delta(
    iterator: AsyncIterator[ChatCompletionDelta],
    *,
    timeout_seconds: float | None,
    attempt_model: str,
    timeout_label: str,
) -> ChatCompletionDelta | None:
    try:
        if timeout_seconds is None:
            return await anext(iterator)
        with anyio.fail_after(timeout_seconds):
            return await anext(iterator)
    except StopAsyncIteration:
        return None
    except TimeoutError as exc:
        await _close_async_iterator(iterator)
        raise ChatCompletionTimeoutError(
            f"stream for model {attempt_model!r} produced no {timeout_label} within {timeout_seconds} seconds"
        ) from exc


async def _first_output_delta(
    iterator: AsyncIterator[ChatCompletionDelta],
    *,
    timeout_seconds: float | None,
    attempt_model: str,
) -> ChatCompletionDelta:
    try:
        if timeout_seconds is None:
            while True:
                delta = await anext(iterator)
                if has_output(delta):
                    return delta
        with anyio.fail_after(timeout_seconds):
            while True:
                delta = await anext(iterator)
                if has_output(delta):
                    return delta
    except StopAsyncIteration:
        await _close_async_iterator(iterator)
        raise ChatCompletionProviderError(f"stream for model {attempt_model!r} ended before first output delta") from None
    except TimeoutError as exc:
        await _close_async_iterator(iterator)
        raise ChatCompletionTimeoutError(
            f"stream for model {attempt_model!r} produced no first output delta within {timeout_seconds} seconds"
        ) from exc


@dataclass(frozen=True)
class ModelRoute:
    prefix: str
    client: IChatCompletionClient

    def __post_init__(self) -> None:
        if not self.prefix:
            raise ValueError("model route prefix cannot be empty")


class RoutingChatCompletionClient(IChatCompletionClient):
    def __init__(
        self,
        routes: Sequence[ModelRoute],
        *,
        stream_first_delta_timeout_seconds: float | None = DEFAULT_STREAM_FIRST_DELTA_TIMEOUT_SECONDS,
    ) -> None:
        self._routes = tuple(routes)
        if not self._routes:
            raise ValueError("at least one model route is required")
        if stream_first_delta_timeout_seconds is not None and stream_first_delta_timeout_seconds <= 0:
            raise ValueError("stream_first_delta_timeout_seconds must be positive")
        self._stream_first_delta_timeout_seconds = stream_first_delta_timeout_seconds

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        attempts = _model_attempts(request.model)
        attempt_count = len(attempts)
        attempt_try_count = TRANSIENT_RETRIES_BEFORE_FALLBACK + 1
        last_error: ChatCompletionProviderError | ChatCompletionUnsupportedRequestError | None = None
        last_attempt_model: str | None = None
        for attempt_index, attempt_model in enumerate(attempts, start=1):
            route = self._route_for(attempt_model)
            provider_model = _provider_model(attempt_model, route.prefix)
            attempt_request = replace(request, model=provider_model)
            for attempt_try_index in range(1, attempt_try_count + 1):
                try:
                    result = await route.client.complete(attempt_request)
                except (ChatCompletionProviderError, ChatCompletionUnsupportedRequestError) as exc:
                    last_error = exc
                    last_attempt_model = attempt_model
                    if attempt_try_index < attempt_try_count and _is_transient_retryable_error(exc):
                        retry_index = attempt_try_index
                        retry_delay_seconds = _transient_retry_delay_seconds(retry_index=retry_index)
                        _log_router_attempt_retry(
                            request_model=request.model,
                            attempt_model=attempt_model,
                            attempt_index=attempt_index,
                            attempt_count=attempt_count,
                            retry_index=retry_index,
                            retry_count=TRANSIENT_RETRIES_BEFORE_FALLBACK,
                            retry_delay_seconds=retry_delay_seconds,
                            exc=exc,
                            streaming=False,
                        )
                        await _sleep_for_transient_retry(retry_delay_seconds)
                        continue
                    if attempt_index < attempt_count:
                        _log_router_attempt_failed(
                            request_model=request.model,
                            attempt_model=attempt_model,
                            attempt_index=attempt_index,
                            attempt_count=attempt_count,
                            next_attempt_model=attempts[attempt_index],
                            exc=exc,
                            streaming=False,
                        )
                    break
                if attempt_index > 1:
                    _log_router_fallback_succeeded(
                        request_model=request.model,
                        winner_model=attempt_model,
                        winning_attempt_index=attempt_index,
                        attempt_count=attempt_count,
                        streaming=False,
                    )
                return replace(result, model=attempt_model)

        if last_error is None:
            raise _unsupported_model(request.model)
        if attempt_count > 1 and last_attempt_model is not None:
            _log_router_fallback_exhausted(
                request_model=request.model,
                final_attempt_model=last_attempt_model,
                attempt_count=attempt_count,
                exc=last_error,
                streaming=False,
            )
        raise last_error

    async def stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionDelta]:
        attempts = _model_attempts(request.model)
        attempt_count = len(attempts)
        attempt_try_count = TRANSIENT_RETRIES_BEFORE_FALLBACK + 1
        last_error: ChatCompletionProviderError | ChatCompletionUnsupportedRequestError | None = None
        last_attempt_model: str | None = None
        for attempt_index, attempt_model in enumerate(attempts, start=1):
            route = self._route_for(attempt_model)
            provider_model = _provider_model(attempt_model, route.prefix)
            attempt_request = replace(request, model=provider_model)
            for attempt_try_index in range(1, attempt_try_count + 1):
                yielded = False
                iterator = None
                try:
                    iterator = route.client.stream(attempt_request).__aiter__()
                    first_delta = await _first_output_delta(
                        iterator,
                        timeout_seconds=self._stream_first_delta_timeout_seconds,
                        attempt_model=attempt_model,
                    )
                    yielded = has_output(first_delta)
                    yield replace(first_delta, model=attempt_model)
                    while True:
                        delta = await _next_stream_delta(
                            iterator,
                            timeout_seconds=DEFAULT_STREAM_IDLE_DELTA_TIMEOUT_SECONDS,
                            attempt_model=attempt_model,
                            timeout_label="delta",
                        )
                        if delta is None:
                            break
                        yielded = yielded or has_output(delta)
                        yield replace(delta, model=attempt_model)
                except (ChatCompletionProviderError, ChatCompletionUnsupportedRequestError) as exc:
                    if yielded:
                        raise
                    last_error = exc
                    last_attempt_model = attempt_model
                    if attempt_try_index < attempt_try_count and _is_transient_retryable_error(exc):
                        retry_index = attempt_try_index
                        retry_delay_seconds = _transient_retry_delay_seconds(retry_index=retry_index)
                        _log_router_attempt_retry(
                            request_model=request.model,
                            attempt_model=attempt_model,
                            attempt_index=attempt_index,
                            attempt_count=attempt_count,
                            retry_index=retry_index,
                            retry_count=TRANSIENT_RETRIES_BEFORE_FALLBACK,
                            retry_delay_seconds=retry_delay_seconds,
                            exc=exc,
                            streaming=True,
                        )
                        await _sleep_for_transient_retry(retry_delay_seconds)
                        continue
                    if attempt_index < attempt_count:
                        _log_router_attempt_failed(
                            request_model=request.model,
                            attempt_model=attempt_model,
                            attempt_index=attempt_index,
                            attempt_count=attempt_count,
                            next_attempt_model=attempts[attempt_index],
                            exc=exc,
                            streaming=True,
                        )
                    break
                else:
                    if attempt_index > 1:
                        _log_router_fallback_succeeded(
                            request_model=request.model,
                            winner_model=attempt_model,
                            winning_attempt_index=attempt_index,
                            attempt_count=attempt_count,
                            streaming=True,
                        )
                    return

        if last_error is None:
            raise _unsupported_model(request.model)
        if attempt_count > 1 and last_attempt_model is not None:
            _log_router_fallback_exhausted(
                request_model=request.model,
                final_attempt_model=last_attempt_model,
                attempt_count=attempt_count,
                exc=last_error,
                streaming=True,
            )
        raise last_error

    def _route_for(self, model: str) -> ModelRoute:
        best_route: ModelRoute | None = None
        for route in self._routes:
            if not model.startswith(route.prefix):
                continue
            if best_route is None or len(route.prefix) > len(best_route.prefix):
                best_route = route
        if best_route is not None:
            _provider_model(model, best_route.prefix)
            return best_route

        raise ChatCompletionUnsupportedRequestError(f"No chat completion route configured for model {model!r}")


class UnavailableChatCompletionClient(IChatCompletionClient):
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        raise _unsupported_model(request.model)

    def stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionDelta]:
        async def raise_unsupported() -> AsyncIterator[ChatCompletionDelta]:
            raise _unsupported_model(request.model)
            yield  # pragma: no cover

        return raise_unsupported()
