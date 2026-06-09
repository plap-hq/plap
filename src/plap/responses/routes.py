from __future__ import annotations

import time
from collections.abc import AsyncIterator
from functools import partial

import anyio
import msgspec
import structlog
from anyio.abc import TaskStatus
from litestar import Request, delete, get, post, websocket
from litestar.channels import ChannelsPlugin, Subscriber
from litestar.connection import WebSocket
from litestar.response import ServerSentEvent
from opentelemetry import trace
from pydantic import TypeAdapter, ValidationError

from plap.auth import AuthContext
from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.keyring import SealingKeyring
from plap.llms.completions.chat import IChatCompletionClient
from plap.logging import bound_context
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
from plap.responses.ingest.ingest import ingest_response_request
from plap.responses.ingest.models import Ingested
from plap.responses.projection import ResponseProjection
from plap.responses.runtime import run_response
from plap.responses.store import PreparedRequest, ResponseStore
from plap.responses.streaming import StreamCoordinator
from plap.settings import RuntimeSelector, Settings
from plap.telemetry import record_scope_context
from plap.tools import IToolCallPolicyResolver, IToolPolicyResolver
from plap.tools.mcp import IMCPToolProvider

logger = structlog.stdlib.get_logger(__name__)
tracer = trace.get_tracer(__name__)
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


def _response_not_found_error(response_id: str, *, action: str) -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=404,
            type="not_found_error",
            code="response_not_found",
            message=f"Response '{response_id}' not found.",
        ),
        private=PrivateError(
            event="response.not_found",
            reason="response_not_found",
            message=f"response not found for {action}: {response_id}",
            level=ErrorLevel.WARNING,
            context={"action": action, "response_id": response_id},
        ),
    )


async def _prepare_create(
    *,
    auth_context: AuthContext,
    request: ResponseCreateRequest,
    response_store: ResponseStore,
    sealing_keyring: SealingKeyring,
    channels: ChannelsPlugin,
) -> tuple[PreparedRequest, Ingested, StreamCoordinator]:
    with tracer.start_as_current_span("response.prepare") as span:
        prepared = await response_store.prepare_request(auth_context, request)
        ingested = await ingest_response_request(
            prepared.execution_request,
            keyring=sealing_keyring,
        )
        coordinator = StreamCoordinator(
            request=prepared.response_request,
            channels=channels,
            prepared=prepared,
            response_store=response_store,
            sealing_keyring=sealing_keyring,
            last_reasoning_id=ingested.last_reasoning_id,
            current_compaction_id=ingested.current_compaction_id,
        )
        span.set_attribute("plap.response.id", coordinator.response_id)
        span.set_attribute("plap.response.model", prepared.response_request.model)
        if prepared.conversation_id is not None:
            span.set_attribute("plap.response.conversation_id", prepared.conversation_id)
        return prepared, ingested, coordinator


def _response_context(prepared: PreparedRequest, coordinator: StreamCoordinator):
    return bound_context(
        conversation_id=prepared.conversation_id,
        response_id=coordinator.response_id,
    )


def _record_request_scope_context(
    request: Request[object, object, object],
    prepared: PreparedRequest,
    coordinator: StreamCoordinator,
) -> None:
    record_scope_context(
        request.scope,
        conversation_id=prepared.conversation_id,
        response_id=coordinator.response_id,
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


async def _watch_socket_disconnect(
    socket: WebSocket,
    cancel_scope: anyio.CancelScope,
) -> None:
    while True:
        event = await socket.receive()
        if event["type"] == "websocket.disconnect":
            cancel_scope.cancel()
            return
        await socket.close(code=1008, reason="Only one active response is allowed per websocket connection.")
        cancel_scope.cancel()
        return


def _is_terminal_event(event: ResponseStreamEvent) -> bool:
    return isinstance(event, ResponseCompletedEvent | ResponseErrorEvent)


def _decode_stream_event(payload: bytes) -> ResponseStreamEvent:
    return _STREAM_EVENT_ADAPTER.validate_json(payload)


async def _iter_projected_payloads(
    subscriber: Subscriber,
    *,
    projection: ResponseProjection,
) -> AsyncIterator[dict[str, object]]:
    async for payload in subscriber.iter_events():
        event = _decode_stream_event(payload)
        yield projection.stream_payload(event)
        if _is_terminal_event(event):
            break


async def _run_stream(
    *,
    prepared: PreparedRequest,
    ingested: Ingested,
    coordinator: StreamCoordinator,
    sealing_keyring: SealingKeyring,
    settings: Settings,
    chat_completion_client: IChatCompletionClient,
    tool_policy_resolver: IToolPolicyResolver,
    tool_call_policy_resolver: IToolCallPolicyResolver,
    mcp_tool_providers: tuple[IMCPToolProvider, ...],
) -> None:
    try:
        await run_response(
            prepared=prepared,
            ingested=ingested,
            coordinator=coordinator,
            sealing_keyring=sealing_keyring,
            settings=settings,
            chat_completion_client=chat_completion_client,
            tool_policy_resolver=tool_policy_resolver,
            tool_call_policy_resolver=tool_call_policy_resolver,
            mcp_tool_providers=mcp_tool_providers,
        )
    except anyio.get_cancelled_exc_class():
        return
    except Exception:
        return


async def _sse_response_payload(
    *,
    http_request: Request[object, object, object],
    request: ResponseCreateRequest,
    auth_context: AuthContext,
    channels: ChannelsPlugin,
    response_store: ResponseStore,
    sealing_keyring: SealingKeyring,
    settings: Settings,
    chat_completion_client: IChatCompletionClient,
    tool_policy_resolver: IToolPolicyResolver,
    tool_call_policy_resolver: IToolCallPolicyResolver,
    mcp_tool_providers: tuple[IMCPToolProvider, ...],
) -> AsyncIterator[str]:
    projection = ResponseProjection.from_create_request(request, transport="stream")
    projection.validate_create_request(request)
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(_watch_http_disconnect, http_request, task_group.cancel_scope)
        prepared, ingested, coordinator = await _prepare_create(
            auth_context=auth_context,
            request=request,
            response_store=response_store,
            sealing_keyring=sealing_keyring,
            channels=channels,
        )
        _record_request_scope_context(http_request, prepared, coordinator)
        subscriber = await channels.subscribe(coordinator.channel)
        try:
            with _response_context(prepared, coordinator):
                task_group.start_soon(
                    partial(
                        _run_stream,
                        prepared=prepared,
                        ingested=ingested,
                        coordinator=coordinator,
                        sealing_keyring=sealing_keyring,
                        settings=settings,
                        chat_completion_client=chat_completion_client,
                        tool_policy_resolver=tool_policy_resolver,
                        tool_call_policy_resolver=tool_call_policy_resolver,
                        mcp_tool_providers=mcp_tool_providers,
                    ),
                )
                async for payload in _iter_projected_payloads(subscriber, projection=projection):
                    yield msgspec.json.encode(payload).decode()
        finally:
            task_group.cancel_scope.cancel()
            with anyio.CancelScope(shield=True):
                await channels.unsubscribe(subscriber, coordinator.channel)
    yield "[DONE]"


@post("/v1/responses", status_code=200, dependencies=HTTP_ROUTE_DEPENDENCIES)
async def create_response(
    request: Request[object, object, object],
    data: ResponseCreateRequest,
    auth_context: AuthContext,
    channels: ChannelsPlugin,
    response_store: ResponseStore,
    sealing_keyring: SealingKeyring,
    settings: Settings,
    chat_completion_client: IChatCompletionClient,
    tool_policy_resolver: IToolPolicyResolver,
    tool_call_policy_resolver: IToolCallPolicyResolver,
    mcp_tool_providers: tuple[IMCPToolProvider, ...],
) -> object:
    projection = ResponseProjection.from_create_request(data, transport="stream" if data.stream else "snapshot")
    projection.validate_create_request(data)
    if data.stream:
        return ServerSentEvent(
            _sse_response_payload(
                http_request=request,
                request=data,
                auth_context=auth_context,
                channels=channels,
                response_store=response_store,
                sealing_keyring=sealing_keyring,
                settings=settings,
                chat_completion_client=chat_completion_client,
                tool_policy_resolver=tool_policy_resolver,
                tool_call_policy_resolver=tool_call_policy_resolver,
                mcp_tool_providers=mcp_tool_providers,
            ),
            headers={"content-type": "text/event-stream; charset=utf-8"},
        )
    response: object
    async with anyio.create_task_group() as task_group:
        watcher_scope = await task_group.start(_watch_http_disconnect, request, task_group.cancel_scope)
        prepared, ingested, coordinator = await _prepare_create(
            auth_context=auth_context,
            request=data,
            response_store=response_store,
            sealing_keyring=sealing_keyring,
            channels=channels,
        )
        _record_request_scope_context(request, prepared, coordinator)
        with _response_context(prepared, coordinator):
            await run_response(
                prepared=prepared,
                ingested=ingested,
                coordinator=coordinator,
                sealing_keyring=sealing_keyring,
                settings=settings,
                chat_completion_client=chat_completion_client,
                tool_policy_resolver=tool_policy_resolver,
                tool_call_policy_resolver=tool_call_policy_resolver,
                mcp_tool_providers=mcp_tool_providers,
            )
            response = projection.response(coordinator.current_response())
        watcher_scope.cancel()
    return response


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
    response_store: ResponseStore,
    include: list[str] | None = None,
    include_obfuscation: bool | None = None,
    starting_after: int | None = None,
    stream: bool | None = None,
) -> ResponseObject:
    _ = starting_after, stream
    with bound_context(response_id=response_id):
        response = await response_store.get_response(auth_context, response_id)
        if response is None:
            raise _response_not_found_error(response_id, action="retrieve")
        projection = ResponseProjection.from_query(include, include_obfuscation=include_obfuscation)
        return projection.response(response)


@delete(
    "/v1/responses/{response_id:str}",
    status_code=200,
    dependencies=HTTP_ROUTE_DEPENDENCIES,
)
async def delete_response(
    response_id: str,
    auth_context: AuthContext,
    response_store: ResponseStore,
) -> ResponseDeleted:
    with bound_context(response_id=response_id):
        deleted = await response_store.delete_response(auth_context, response_id)
        if not deleted:
            raise _response_not_found_error(response_id, action="delete")
        return ResponseDeleted(deleted=True, id=response_id)


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
    response_store: ResponseStore,
    after: str | None = None,
    include: list[str] | None = None,
    limit: int | None = None,
    order: str | None = None,
) -> InputItemsPage:
    with bound_context(response_id=response_id):
        page = await response_store.list_input_items(
            auth_context,
            response_id,
            after=after,
            limit=limit,
            order=order,
        )
        projection = ResponseProjection.from_query(include)
        return projection.input_items_page(page)


@websocket("/v1/responses", dependencies=WEBSOCKET_ROUTE_DEPENDENCIES)
async def responses_socket(
    socket: WebSocket,
    auth_context: AuthContext,
    settings: Settings,
    channels: ChannelsPlugin,
    response_store: ResponseStore,
    sealing_keyring: SealingKeyring,
    chat_completion_client: IChatCompletionClient,
    tool_policy_resolver: IToolPolicyResolver,
    tool_call_policy_resolver: IToolCallPolicyResolver,
    mcp_tool_providers: tuple[IMCPToolProvider, ...],
) -> None:
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

        started_at = time.perf_counter()
        with tracer.start_as_current_span("websocket.response.create") as span:
            projection = ResponseProjection.from_create_request(client_event.response, transport="stream")
            projection.validate_create_request(client_event.response)
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(_watch_socket_disconnect, socket, task_group.cancel_scope)
                prepared, ingested, coordinator = await _prepare_create(
                    auth_context=auth_context,
                    request=client_event.response,
                    response_store=response_store,
                    sealing_keyring=sealing_keyring,
                    channels=channels,
                )
                span.set_attribute("plap.response.id", coordinator.response_id)
                subscriber = await channels.subscribe(coordinator.channel)
                try:
                    with _response_context(prepared, coordinator):
                        task_group.start_soon(
                            partial(
                                _run_stream,
                                prepared=prepared,
                                ingested=ingested,
                                coordinator=coordinator,
                                sealing_keyring=sealing_keyring,
                                settings=settings,
                                chat_completion_client=chat_completion_client,
                                tool_policy_resolver=tool_policy_resolver,
                                tool_call_policy_resolver=tool_call_policy_resolver,
                                mcp_tool_providers=mcp_tool_providers,
                            ),
                        )
                        async for payload in _iter_projected_payloads(subscriber, projection=projection):
                            await socket.send_json(payload)
                        logger.info(
                            "websocket.response.completed",
                            duration_ms=round((time.perf_counter() - started_at) * 1000.0, 3),
                        )
                finally:
                    task_group.cancel_scope.cancel()
                    with anyio.CancelScope(shield=True):
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
