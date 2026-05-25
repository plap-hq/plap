from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from plap.llms.completions.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResult,
    IChatCompletionClient,
)
from plap.llms.completions.common import (
    StreamState,
    build_chat_body,
    completion_result_from_data,
    delta_from_data,
    raise_incomplete_stream_error,
)
from plap.llms.completions.errors import ChatCompletionUnsupportedRequestError

type NextComplete = Callable[[ChatCompletionRequest | None], Awaitable[dict[str, Any]]]
type NextStream = Callable[[ChatCompletionRequest | None], AsyncIterator[dict[str, Any]]]


@dataclass(slots=True)
class Call:
    request: ChatCompletionRequest
    body: dict[str, Any]


class Quirk:
    def request(self, call: Call) -> None:
        return None

    async def complete(self, call: Call, next_complete: NextComplete) -> dict[str, Any]:
        return await next_complete(None)

    async def stream(
        self,
        call: Call,
        next_complete: NextComplete,
        next_stream: NextStream,
    ) -> AsyncIterator[dict[str, Any]]:
        async for chunk in next_stream(None):
            yield chunk


class Provider:
    def __init__(
        self,
        *,
        name: str,
        quirks: tuple[Quirk, ...] = (),
        models: dict[str, tuple[Quirk, ...]] | None = None,
    ) -> None:
        self.name = name
        self.quirks = deepcopy(quirks)
        self.models = deepcopy(dict(models or {}))

    def lookup(self, name: str) -> tuple[Quirk, ...]:
        quirks = self.models.get(name)
        if quirks is None:
            raise ChatCompletionUnsupportedRequestError(f"unsupported {self.name} model: {name}")
        return quirks

    async def complete(self, call: Call) -> dict[str, Any]:
        raise NotImplementedError

    def stream(self, call: Call) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError


def _build_call(request: ChatCompletionRequest, *, stream: bool) -> Call:
    return Call(request=request, body=build_chat_body(request, stream=stream))


def _apply_request_quirks(call: Call, quirks: tuple[Quirk, ...]) -> None:
    for quirk in quirks:
        quirk.request(call)


class ChatCompletionClient(IChatCompletionClient):
    def __init__(self, provider: Provider) -> None:
        self._provider = provider

    def _quirks(self, model: str) -> tuple[Quirk, ...]:
        return (*self._provider.quirks, *self._provider.lookup(model))

    async def _complete_from_call(
        self,
        call: Call,
        quirks: tuple[Quirk, ...],
        index: int,
    ) -> dict[str, Any]:
        if index >= len(quirks):
            return await self._provider.complete(call)

        async def next_complete(request: ChatCompletionRequest | None) -> dict[str, Any]:
            if request is None:
                return await self._complete_from_call(call, quirks, index + 1)
            return await self._complete_request(request, quirks, index + 1)

        return await quirks[index].complete(call, next_complete)

    async def _complete_request(
        self,
        request: ChatCompletionRequest,
        quirks: tuple[Quirk, ...],
        index: int,
    ) -> dict[str, Any]:
        call = _build_call(request, stream=False)
        _apply_request_quirks(call, quirks)
        return await self._complete_from_call(call, quirks, index)

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        raw = await self._complete_request(request, self._quirks(request.model), 0)
        return completion_result_from_data(raw, request=request)

    async def _stream_from_call(
        self,
        call: Call,
        quirks: tuple[Quirk, ...],
        index: int,
    ) -> AsyncIterator[dict[str, Any]]:
        if index >= len(quirks):
            async for chunk in self._provider.stream(call):
                yield chunk
            return

        async def next_complete(request: ChatCompletionRequest | None) -> dict[str, Any]:
            if request is None:
                return await self._complete_request(call.request, quirks, index + 1)
            return await self._complete_request(request, quirks, index + 1)

        def next_stream(request: ChatCompletionRequest | None) -> AsyncIterator[dict[str, Any]]:
            if request is None:
                return self._stream_from_call(call, quirks, index + 1)
            return self._stream_request(request, quirks, index + 1)

        async for chunk in quirks[index].stream(call, next_complete, next_stream):
            yield chunk

    def _stream_request(
        self,
        request: ChatCompletionRequest,
        quirks: tuple[Quirk, ...],
        index: int,
    ) -> AsyncIterator[dict[str, Any]]:
        async def run() -> AsyncIterator[dict[str, Any]]:
            call = _build_call(request, stream=True)
            _apply_request_quirks(call, quirks)
            async for chunk in self._stream_from_call(call, quirks, index):
                yield chunk

        return run()

    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionDelta]:
        async def run() -> AsyncIterator[ChatCompletionDelta]:
            quirks = self._quirks(request.model)
            state = StreamState()
            async for raw in self._stream_request(request, quirks, 0):
                delta = delta_from_data(raw, request=request)
                state.apply(delta)
                yield state.normalized_terminal_delta(delta)
            inferred_delta = state.inferred_terminal_delta()
            if inferred_delta is not None:
                yield inferred_delta
            elif not state.saw_finish_reason:
                raise_incomplete_stream_error()

        return run()


__all__ = [
    "Call",
    "ChatCompletionClient",
    "Provider",
    "Quirk",
]
