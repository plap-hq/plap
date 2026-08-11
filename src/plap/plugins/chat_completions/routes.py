from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anyio
import msgspec
from anyio.abc import TaskStatus
from litestar import Request, post
from litestar.channels import ChannelsPlugin
from litestar.response import ServerSentEvent

from plap.auth import AuthContext
from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.plugins.chat_completions.contracts import ChatCompletionCreateRequest
from plap.plugins.chat_completions.translation import chat_completion_stream, to_chat_completion, to_response_request
from plap.responses import create as create_response
from plap.responses import stream as stream_response
from plap.responses.contracts import ResponseObject, ResponseStreamEvent


def _failed_response_error(response: ResponseObject) -> PlapError:
    code = "server_error" if response.error is None else response.error.code
    message = "The model failed to generate a response." if response.error is None else response.error.message
    return PlapError(
        public=PublicError(
            status_code=500,
            type="server_error",
            code=code,
            message=message,
        ),
        private=PrivateError(
            event="chat_completion.failed",
            reason="response_failed",
            message=f"response failed during chat completion: {response.id}",
            level=ErrorLevel.ERROR,
            context={"response_id": response.id},
        ),
    )


async def _watch_http_disconnect(
    request: Request[object, object, object],
    cancel_scope: anyio.CancelScope,
    *,
    task_status: TaskStatus[anyio.CancelScope] = anyio.TASK_STATUS_IGNORED,
) -> None:
    with anyio.CancelScope() as watcher_scope:
        task_status.started(watcher_scope)
        while True:
            event = await request.receive()
            if event["type"] == "http.disconnect":
                cancel_scope.cancel()
                return


async def _close_events(events: AsyncIterator[ResponseStreamEvent]) -> None:
    aclose = getattr(events, "aclose", None)
    if aclose is not None:
        await aclose()


async def _chat_sse_payload(
    events: AsyncIterator[ResponseStreamEvent],
    *,
    include_usage: bool,
) -> AsyncIterator[str]:
    payloads = chat_completion_stream(events, include_usage=include_usage)
    try:
        async for payload in payloads:
            yield msgspec.json.encode(payload).decode()
    except anyio.get_cancelled_exc_class():
        raise
    except BaseException:
        await payloads.aclose()
        await _close_events(events)
        raise
    else:
        await payloads.aclose()
        await _close_events(events)
    yield "[DONE]"


def _single_group_exception(exc: BaseExceptionGroup) -> BaseException:
    current: BaseException = exc
    while isinstance(current, BaseExceptionGroup) and len(current.exceptions) == 1:
        current = current.exceptions[0]
    return current


@post("/v1/chat/completions", status_code=200)
async def create_chat_completion(
    request: Request[object, object, object],
    data: ChatCompletionCreateRequest,
    svcs: Any,
    auth_context: AuthContext,
) -> object:
    response_request = to_response_request(data)
    channels = request.app.plugins.get(ChannelsPlugin)
    if data.stream:
        events = await stream_response(
            response_request,
            svcs=svcs,
            auth_context=auth_context,
            channels=channels,
            scope=request.scope,
        )
        include_usage = data.stream_options is not None and data.stream_options.include_usage is True
        return ServerSentEvent(
            _chat_sse_payload(events, include_usage=include_usage),
            headers={"content-type": "text/event-stream; charset=utf-8"},
        )

    try:
        async with anyio.create_task_group() as task_group:
            watcher_scope = await task_group.start(_watch_http_disconnect, request, task_group.cancel_scope)
            try:
                response = await create_response(
                    response_request,
                    svcs=svcs,
                    auth_context=auth_context,
                    channels=channels,
                    scope=request.scope,
                )
            finally:
                watcher_scope.cancel()
    except BaseExceptionGroup as exc:
        unwrapped = _single_group_exception(exc)
        if unwrapped is exc:
            raise
        raise unwrapped from None
    if response.status == "failed":
        raise _failed_response_error(response)
    return to_chat_completion(response)


__all__ = ["create_chat_completion"]
