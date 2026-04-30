from __future__ import annotations

from collections.abc import AsyncIterator

from litestar import delete, get, post, websocket
from litestar.connection import WebSocket
from litestar.response import ServerSentEvent
from pydantic import ValidationError

from plap.auth import AuthContext
from plap.keyring import SealingKeyring
from plap.llms.chat import IChatCompletionClient
from plap.responses.contracts import (
    CompactRequest,
    InputTokensCountRequest,
    ResponseCompletedEvent,
    ResponseCreateClientEvent,
    ResponseCreateRequest,
    ResponseErrorEvent,
    ResponseObject,
    ResponseStreamEvent,
)
from plap.responses.dependencies import (
    HTTP_ROUTE_DEPENDENCIES,
    WEBSOCKET_ROUTE_DEPENDENCIES,
)
from plap.responses.errors import ResponseOperationUnsupportedError
from plap.responses.reasoning import IReasoningSummarizer
from plap.responses.runtime import stream_response_events
from plap.responses.tools import IToolCallPolicyResolver, IToolPolicyResolver
from plap.responses.tools.mcp import IMCPToolProvider
from plap.settings import Settings


async def _sse_payload(
    events: AsyncIterator[ResponseStreamEvent],
) -> AsyncIterator[str]:
    async for event in events:
        yield event.model_dump_json(exclude_none=True)
    yield "[DONE]"


def _completed_response_from_events(
    events: list[ResponseStreamEvent],
) -> ResponseObject:
    for event in reversed(events):
        if isinstance(event, ResponseCompletedEvent):
            return event.response
    raise RuntimeError("response stream did not complete")


@post("/v1/responses", status_code=200, dependencies=HTTP_ROUTE_DEPENDENCIES)
async def create_response(
    data: ResponseCreateRequest,
    auth_context: AuthContext,
    settings: Settings,
    sealing_keyring: SealingKeyring,
    chat_completion_client: IChatCompletionClient,
    reasoning_summarizer: IReasoningSummarizer,
    tool_policy_resolver: IToolPolicyResolver,
    tool_call_policy_resolver: IToolCallPolicyResolver,
    mcp_tool_provider: IMCPToolProvider | None,
) -> object:
    _ = auth_context
    events = stream_response_events(
        data,
        settings=settings,
        sealing_keyring=sealing_keyring,
        chat_completion_client=chat_completion_client,
        reasoning_summarizer=reasoning_summarizer,
        tool_policy_resolver=tool_policy_resolver,
        tool_call_policy_resolver=tool_call_policy_resolver,
        mcp_tool_provider=mcp_tool_provider,
    )
    if data.stream:
        return ServerSentEvent(
            _sse_payload(events),
            headers={"content-type": "text/event-stream; charset=utf-8"},
        )
    return _completed_response_from_events([event async for event in events])


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
    _ = response_id, include, include_obfuscation, starting_after, stream
    raise ResponseOperationUnsupportedError(status_code=404)


@delete(
    "/v1/responses/{response_id:str}",
    status_code=200,
    dependencies=HTTP_ROUTE_DEPENDENCIES,
)
async def delete_response(
    response_id: str,
    auth_context: AuthContext,
) -> object:
    _ = auth_context
    _ = response_id
    raise ResponseOperationUnsupportedError(status_code=404)


@post("/v1/responses/compact", status_code=200, dependencies=HTTP_ROUTE_DEPENDENCIES)
async def compact_response(
    data: CompactRequest,
    auth_context: AuthContext,
) -> object:
    _ = auth_context
    _ = data
    raise ResponseOperationUnsupportedError()


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
) -> object:
    _ = auth_context
    _ = response_id, after, include, limit, order
    raise ResponseOperationUnsupportedError(status_code=404)


@post(
    "/v1/responses/input_tokens",
    status_code=200,
    dependencies=HTTP_ROUTE_DEPENDENCIES,
)
async def count_input_tokens(
    data: InputTokensCountRequest,
    auth_context: AuthContext,
) -> object:
    _ = auth_context
    _ = data
    raise ResponseOperationUnsupportedError()


@websocket("/v1/responses", dependencies=WEBSOCKET_ROUTE_DEPENDENCIES)
async def responses_socket(
    socket: WebSocket,
    auth_context: AuthContext,
    settings: Settings,
    sealing_keyring: SealingKeyring,
    chat_completion_client: IChatCompletionClient,
    reasoning_summarizer: IReasoningSummarizer,
    tool_policy_resolver: IToolPolicyResolver,
    tool_call_policy_resolver: IToolCallPolicyResolver,
    mcp_tool_provider: IMCPToolProvider | None,
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
        except ValidationError:
            await socket.send_json(
                build_error_event("Invalid client event.").model_dump(
                    mode="json",
                    exclude_none=True,
                )
            )
            continue

        try:
            events = stream_response_events(
                client_event.response,
                settings=settings,
                sealing_keyring=sealing_keyring,
                chat_completion_client=chat_completion_client,
                reasoning_summarizer=reasoning_summarizer,
                tool_policy_resolver=tool_policy_resolver,
                tool_call_policy_resolver=tool_call_policy_resolver,
                mcp_tool_provider=mcp_tool_provider,
            )
            async for event in events:
                await socket.send_json(event.model_dump(mode="json", exclude_none=True))
        except Exception:
            await socket.send_json(
                build_error_event("Invalid request.").model_dump(
                    mode="json",
                    exclude_none=True,
                )
            )


def build_error_event(message: str) -> ResponseErrorEvent:
    return ResponseErrorEvent(
        code="invalid_request_error",
        message=message,
        sequence_number=1,
        type="error",
    )


RESPONSE_ROUTE_HANDLERS = [
    create_response,
    retrieve_response,
    delete_response,
    # OpenAI exposes POST /v1/responses/{response_id}/cancel, but it only
    # applies to background responses. We do not support background execution
    # yet, so there is no detached response job to cancel.
    compact_response,
    list_input_items,
    count_input_tokens,
    responses_socket,
]
