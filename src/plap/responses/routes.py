from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from litestar import Request, delete, get, post, websocket
from litestar.connection import WebSocket
from litestar.exceptions import ValidationException
from litestar.response import ServerSentEvent
from pydantic import ValidationError

from plap.auth import AuthContext
from plap.responses.contracts import (
    CompactedResponseObject,
    CompactRequest,
    InputItemsPage,
    InputTokenCountResponse,
    InputTokensCountRequest,
    ResponseCreateClientEvent,
    ResponseCreateRequest,
    ResponseDeleted,
    ResponseObject,
)
from plap.responses.dependencies import (
    HTTP_ROUTE_DEPENDENCIES,
    WEBSOCKET_ROUTE_DEPENDENCIES,
)
from plap.responses.stubs import (
    build_compacted_response,
    build_deleted_response,
    build_error_event,
    build_input_items_page,
    build_input_token_count,
    build_stream_events,
    build_stub_response,
)
from plap.responses.tools import ToolPolicyError, ToolPolicyResolver


async def _sse_payload(response: ResponseObject) -> AsyncIterator[str]:
    for event in build_stream_events(response):
        yield event.model_dump_json(exclude_none=True)
    yield "[DONE]"


@post("/v1/responses", status_code=200, dependencies=HTTP_ROUTE_DEPENDENCIES)
async def create_response(
    data: ResponseCreateRequest,
    auth_context: AuthContext,
    request: Request[Any, Any, Any],
) -> object:
    _ = auth_context
    try:
        resolver: ToolPolicyResolver = request.app.state.tool_policy_resolver
        await resolver.resolve(data.tools or [])
    except ToolPolicyError as exc:
        raise ValidationException(str(exc)) from exc
    response = build_stub_response(data)
    if data.stream:
        return ServerSentEvent(
            _sse_payload(response),
            headers={"content-type": "text/event-stream; charset=utf-8"},
        )
    return response


@get("/v1/responses/{response_id:str}", dependencies=HTTP_ROUTE_DEPENDENCIES)
async def retrieve_response(
    response_id: str,
    auth_context: AuthContext,
    include: list[str] | None = None,
    include_obfuscation: bool | None = None,
    starting_after: int | None = None,
    stream: bool | None = None,
) -> object:
    _ = auth_context
    _ = include_obfuscation, starting_after
    request = ResponseCreateRequest(include=include)
    response = build_stub_response(request, response_id=response_id)
    if stream:
        return ServerSentEvent(
            _sse_payload(response),
            headers={"content-type": "text/event-stream; charset=utf-8"},
        )
    return response


@delete(
    "/v1/responses/{response_id:str}",
    status_code=200,
    dependencies=HTTP_ROUTE_DEPENDENCIES,
)
async def delete_response(
    response_id: str,
    auth_context: AuthContext,
) -> ResponseDeleted:
    _ = auth_context
    return build_deleted_response(response_id)


@post(
    "/v1/responses/{response_id:str}/cancel",
    status_code=200,
    dependencies=HTTP_ROUTE_DEPENDENCIES,
)
async def cancel_response(
    response_id: str,
    auth_context: AuthContext,
) -> ResponseObject:
    _ = auth_context
    return build_stub_response(
        ResponseCreateRequest(), response_id=response_id, status="cancelled"
    )


@post("/v1/responses/compact", status_code=200, dependencies=HTTP_ROUTE_DEPENDENCIES)
async def compact_response(
    data: CompactRequest,
    auth_context: AuthContext,
) -> CompactedResponseObject:
    _ = auth_context
    return build_compacted_response(data)


@get(
    "/v1/responses/{response_id:str}/input_items",
    dependencies=HTTP_ROUTE_DEPENDENCIES,
)
async def list_input_items(
    response_id: str,
    auth_context: AuthContext,
    after: str | None = None,
    include: list[str] | None = None,
    limit: int | None = None,
    order: str | None = None,
) -> InputItemsPage:
    _ = auth_context
    _ = after, include, limit, order
    return build_input_items_page(response_id)


@post(
    "/v1/responses/input_tokens",
    status_code=200,
    dependencies=HTTP_ROUTE_DEPENDENCIES,
)
async def count_input_tokens(
    data: InputTokensCountRequest,
    auth_context: AuthContext,
) -> InputTokenCountResponse:
    _ = auth_context
    return build_input_token_count(data)


@websocket("/v1/responses", dependencies=WEBSOCKET_ROUTE_DEPENDENCIES)
async def responses_socket(
    socket: WebSocket,
    auth_context: AuthContext,
) -> None:
    _ = auth_context

    await socket.accept()

    while True:
        try:
            payload = await socket.receive_json()
        except Exception:
            return

        try:
            client_event = ResponseCreateClientEvent.model_validate(payload)
        except ValidationError as exc:
            await socket.send_json(
                build_error_event(exc.errors(include_url=False)[0]["msg"]).model_dump(
                    mode="json",
                    exclude_none=True,
                )
            )
            continue

        response = build_stub_response(client_event.response)
        for event in build_stream_events(response):
            await socket.send_json(event.model_dump(mode="json", exclude_none=True))


RESPONSE_ROUTE_HANDLERS = [
    create_response,
    retrieve_response,
    delete_response,
    cancel_response,
    compact_response,
    list_input_items,
    count_input_tokens,
    responses_socket,
]
