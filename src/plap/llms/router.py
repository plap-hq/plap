from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, replace

from plap.llms.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResult,
    IChatCompletionClient,
)
from plap.llms.errors import ChatCompletionProviderError, ChatCompletionUnsupportedRequestError


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


@dataclass(frozen=True)
class ModelRoute:
    prefix: str
    client: IChatCompletionClient

    def __post_init__(self) -> None:
        if not self.prefix:
            raise ValueError("model route prefix cannot be empty")


class RoutingChatCompletionClient(IChatCompletionClient):
    def __init__(self, routes: Sequence[ModelRoute]) -> None:
        self._routes = tuple(routes)
        if not self._routes:
            raise ValueError("at least one model route is required")

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        last_error: ChatCompletionProviderError | ChatCompletionUnsupportedRequestError | None = None
        for attempt_model in _model_attempts(request.model):
            route = self._route_for(attempt_model)
            provider_model = _provider_model(attempt_model, route.prefix)
            try:
                result = await route.client.complete(replace(request, model=provider_model))
            except (ChatCompletionProviderError, ChatCompletionUnsupportedRequestError) as exc:
                last_error = exc
                continue
            return replace(result, model=attempt_model)

        if last_error is None:
            raise _unsupported_model(request.model)
        raise last_error

    async def stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionDelta]:
        last_error: ChatCompletionProviderError | ChatCompletionUnsupportedRequestError | None = None
        for attempt_model in _model_attempts(request.model):
            route = self._route_for(attempt_model)
            provider_model = _provider_model(attempt_model, route.prefix)
            yielded = False
            try:
                async for delta in route.client.stream(replace(request, model=provider_model)):
                    yielded = True
                    yield replace(delta, model=attempt_model)
            except (ChatCompletionProviderError, ChatCompletionUnsupportedRequestError) as exc:
                if yielded:
                    raise
                last_error = exc
                continue
            else:
                return

        if last_error is None:
            raise _unsupported_model(request.model)
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
