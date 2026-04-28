from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

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
        return await self._client_for(request.model).complete(request)

    async def stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionDelta]:
        async for delta in self._client_for(request.model).stream(request):
            yield delta

    def _client_for(self, model: str) -> IChatCompletionClient:
        best_route: ModelRoute | None = None
        for route in self._routes:
            if not model.startswith(route.prefix):
                continue
            if best_route is None or len(route.prefix) > len(best_route.prefix):
                best_route = route
        if best_route is not None:
            return best_route.client

        raise ChatCompletionUnsupportedRequestError(
            f"No chat completion route configured for model {model!r}"
        )
