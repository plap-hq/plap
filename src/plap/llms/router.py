from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, replace

from plap.llms.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResult,
    IChatCompletionClient,
)
from plap.llms.errors import ChatCompletionUnsupportedRequestError


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
        route = self._route_for(request.model)
        provider_model = _provider_model(request.model, route.prefix)
        result = await route.client.complete(replace(request, model=provider_model))
        return replace(result, model=request.model)

    async def stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionDelta]:
        route = self._route_for(request.model)
        provider_model = _provider_model(request.model, route.prefix)
        async for delta in route.client.stream(replace(request, model=provider_model)):
            yield replace(delta, model=request.model)

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


def _unsupported_model(model: str) -> ChatCompletionUnsupportedRequestError:
    return ChatCompletionUnsupportedRequestError(f"No chat completion provider configured for model {model!r}")


def _provider_model(model: str, prefix: str) -> str:
    provider_model = model.removeprefix(prefix)
    if not provider_model:
        raise ChatCompletionUnsupportedRequestError(f"No provider model configured for model {model!r}")
    return provider_model
