from __future__ import annotations

from collections.abc import AsyncIterator

import anyio
import msgspec
import structlog
from litestar import delete, get, post, websocket
from litestar.channels import ChannelsPlugin, Subscriber
from litestar.connection import WebSocket
from litestar.response import ServerSentEvent
from pydantic import TypeAdapter, ValidationError

from plap.auth import AuthContext
from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.responses.contracts import (
    CompactedResponseObject,
    CompactRequest,
    InputItemsPage,
    ModelInfoListObject,
    ModelListObject,
    ReasoningEffort,
    ResponseCompletedEvent,
    ResponseCreateClientEvent,
    ResponseCreateRequest,
    ResponseDeleted,
    ResponseErrorEvent,
    ResponseObject,
    ResponseStreamEvent,
    ServiceTier,
)
from plap.responses.dependencies import HTTP_ROUTE_DEPENDENCIES, WEBSOCKET_ROUTE_DEPENDENCIES
from plap.responses.projection import ResponseProjection
from plap.responses.streaming import StreamCoordinator
from plap.settings import RuntimeSelector, Settings

logger = structlog.get_logger(__name__)
_STREAM_EVENT_ADAPTER = TypeAdapter(ResponseStreamEvent)


def _not_implemented_public_error(*, action: str) -> PublicError:
    return PublicError(
        status_code=501,
        type="server_error",
        code="not_implemented",
        message=f"responses {action} is not implemented yet",
    )


def _not_implemented_error(*, action: str) -> PlapError:
    return PlapError(
        public=_not_implemented_public_error(action=action),
        private=PrivateError(
            event="response.not_implemented",
            reason="responses_route_not_implemented",
            message=f"responses {action} route is not implemented yet",
            level=ErrorLevel.WARNING,
            context={"action": action},
        ),
    )


async def _sse_not_implemented_payload(*, action: str) -> AsyncIterator[str]:
    public = _not_implemented_public_error(action=action)
    yield build_error_event(public=public).model_dump_json(exclude_none=True)
    yield "[DONE]"


def _is_terminal_event(event: ResponseStreamEvent) -> bool:
    return isinstance(event, ResponseCompletedEvent | ResponseErrorEvent)


def _decode_stream_event(payload: bytes) -> ResponseStreamEvent:
    return _STREAM_EVENT_ADAPTER.validate_json(payload)


async def _iter_projected_events(
    subscriber: Subscriber,
    *,
    projection: ResponseProjection,
) -> AsyncIterator[str]:
    async for payload in subscriber.iter_events():
        event = _decode_stream_event(payload)
        yield msgspec.json.encode(projection.stream_payload(event)).decode()
        if _is_terminal_event(event):
            break


async def _publish_not_implemented_sequence(
    *,
    coordinator: StreamCoordinator,
    action: str,
) -> None:
    await coordinator.created()
    await coordinator.in_progress()
    await coordinator.fail(_not_implemented_public_error(action=action))


async def _sse_not_implemented_channel_payload(
    *,
    request: ResponseCreateRequest,
    channels: ChannelsPlugin,
    action: str,
) -> AsyncIterator[str]:
    projection = ResponseProjection.from_create_request(request, transport="stream")
    projection.validate_create_request(request)
    coordinator = StreamCoordinator(request=request, channels=channels)
    subscriber = await channels.subscribe(coordinator.channel)
    try:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(_publish_not_implemented_sequence, coordinator=coordinator, action=action)
            async for payload in _iter_projected_events(subscriber, projection=projection):
                yield payload
    finally:
        await channels.unsubscribe(subscriber, coordinator.channel)
    yield "[DONE]"


@post("/v1/responses", status_code=200, dependencies=HTTP_ROUTE_DEPENDENCIES)
async def create_response(
    data: ResponseCreateRequest,
    auth_context: AuthContext,
    channels: ChannelsPlugin,
) -> object:
    _ = auth_context
    if data.stream:
        return ServerSentEvent(
            _sse_not_implemented_channel_payload(request=data, channels=channels, action="create"),
            headers={"content-type": "text/event-stream; charset=utf-8"},
        )
    raise _not_implemented_error(action="create")


@get("/v1/models", dependencies=HTTP_ROUTE_DEPENDENCIES)
async def list_models(
    auth_context: AuthContext,
    settings: Settings,
) -> ModelListObject:
    _ = auth_context
    return ModelListObject(
        data=[profile.to_model_object(model=model) for model, profile in sorted(settings.runtime_model_profiles.items())]
    )


@get("/v1/model/info", dependencies=HTTP_ROUTE_DEPENDENCIES)
async def model_info(
    model: str,
    auth_context: AuthContext,
    settings: Settings,
    service_tier: ServiceTier | None = None,
    reasoning_effort: ReasoningEffort | None = None,
) -> ModelInfoListObject:
    _ = auth_context
    profile = settings.resolve_runtime_model_profile(
        model,
        selector=RuntimeSelector(
            service_tier=service_tier,
            reasoning_effort=reasoning_effort,
        ),
    )
    return ModelInfoListObject(data=[profile.model_info.to_contract(model=model)])


@get("/v1/responses/{response_id:str}", dependencies=HTTP_ROUTE_DEPENDENCIES)
async def retrieve_response(
    response_id: str,
    auth_context: AuthContext,
    include: list[str] | None = None,
    include_obfuscation: bool | None = None,
    starting_after: int | None = None,
    stream: bool | None = None,
) -> ResponseObject:
    _ = response_id, auth_context, include, include_obfuscation, starting_after, stream
    raise _not_implemented_error(action="retrieve")


@delete(
    "/v1/responses/{response_id:str}",
    status_code=200,
    dependencies=HTTP_ROUTE_DEPENDENCIES,
)
async def delete_response(
    response_id: str,
    auth_context: AuthContext,
) -> ResponseDeleted:
    _ = response_id, auth_context
    raise _not_implemented_error(action="delete")


@post("/v1/responses/compact", status_code=200, dependencies=HTTP_ROUTE_DEPENDENCIES)
async def compact_response(
    data: CompactRequest,
    auth_context: AuthContext,
) -> CompactedResponseObject:
    _ = data, auth_context
    raise _not_implemented_error(action="compact")


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
    _ = response_id, auth_context, after, include, limit, order
    raise _not_implemented_error(action="list_input_items")


@websocket("/v1/responses", dependencies=WEBSOCKET_ROUTE_DEPENDENCIES)
async def responses_socket(
    socket: WebSocket,
    auth_context: AuthContext,
    settings: Settings,
    channels: ChannelsPlugin,
) -> None:
    _ = auth_context, settings
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
                build_error_event(
                    public=PublicError(
                        status_code=400,
                        type="invalid_request_error",
                        code="invalid_client_event",
                        message="Invalid client event.",
                    )
                ).model_dump(
                    mode="json",
                    exclude_none=True,
                )
            )
            continue

        projection = ResponseProjection.from_create_request(client_event.response, transport="stream")
        projection.validate_create_request(client_event.response)
        coordinator = StreamCoordinator(request=client_event.response, channels=channels)
        subscriber = await channels.subscribe(coordinator.channel)
        try:
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(_publish_not_implemented_sequence, coordinator=coordinator, action="websocket_create")
                async for payload_json in _iter_projected_events(subscriber, projection=projection):
                    await socket.send_json(msgspec.json.decode(payload_json.encode()))
        finally:
            await channels.unsubscribe(subscriber, coordinator.channel)


def build_error_event(*, public: PublicError) -> ResponseErrorEvent:
    return ResponseErrorEvent(
        code=public.code,
        message=public.message,
        param=public.param,
        sequence_number=1,
        type="error",
    )


RESPONSE_ROUTE_HANDLERS = [
    list_models,
    model_info,
    create_response,
    retrieve_response,
    delete_response,
    # OpenAI exposes POST /v1/responses/{response_id}/cancel, but it only
    # applies to background responses. We do not support background execution
    # yet, so there is no detached response job to cancel.
    # OpenAI also exposes POST /v1/responses/input_tokens. We intentionally do
    # not route it because provider-compatible token counting is not supported.
    compact_response,
    list_input_items,
    responses_socket,
]
