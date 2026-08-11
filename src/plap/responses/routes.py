from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import anyio
import msgspec
import structlog
import svcs
from anyio.abc import TaskStatus
from litestar import Request, delete, get, post, websocket
from litestar.channels import ChannelsPlugin, Subscriber
from litestar.connection import WebSocket
from litestar.di import Provide
from litestar.response import ServerSentEvent
from opentelemetry import trace
from pydantic import TypeAdapter, ValidationError
from svcs import Container

from plap.auth import AuthContext
from plap.auth.dependencies import provide_socket_auth_context
from plap.bus import bus
from plap.config import CueBox
from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.keyring import SealingKeyring
from plap.llms.completions.budget import CompletionBudget
from plap.logging import bound_context
from plap.responses.contracts import (
    CompactedResponseObject,
    CompactRequest,
    InputItemsPage,
    ModelInfoListObject,
    ModelInfoObject,
    ModelInfoPricingObject,
    ModelListObject,
    ModelObject,
    ReasoningEffort,
    ResponseCompletedEvent,
    ResponseCreateClientEvent,
    ResponseCreateRequest,
    ResponseDeleted,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseObject,
    ResponseStreamEvent,
    ServiceTier,
)
from plap.responses.ingest.ingest import ingest_response_request
from plap.responses.projection import ResponseProjection
from plap.responses.state import State
from plap.responses.store import ResponseStore
from plap.responses.streaming import ResponseFinalizationError, StreamCoordinator
from plap.telemetry import record_scope_context

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


def _missing_model_error() -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code="missing_required_parameter",
            message="Parameter 'model' is required.",
            param="model",
        ),
        private=PrivateError(
            event="response.invalid_request",
            reason="model_required",
            message="response model is required",
            level=ErrorLevel.WARNING,
        ),
    )


def _model_not_found_error(model: str) -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=404,
            type="invalid_request_error",
            code="model_not_found",
            message=f"Model '{model}' does not exist.",
            param="model",
        ),
        private=PrivateError(
            event="response.invalid_request",
            reason="model_not_found",
            message=f"response model is not configured: {model}",
            level=ErrorLevel.WARNING,
            context={"model": model},
        ),
    )


def _model_names(config: CueBox) -> list[str]:
    models = config.overlays.get("model", {})
    if not isinstance(models, dict):
        return []
    return sorted(name for name, branch in models.items() if isinstance(name, str) and isinstance(branch, dict))


def _require_model(config: CueBox, model: str | None) -> str:
    if model is None:
        raise _missing_model_error()
    if model not in _model_names(config):
        raise _model_not_found_error(model)
    return model


def _thread_codes(svcs: svcs.Container) -> dict[str, int]:
    raw = svcs.get(CueBox).threads
    return {str(thread): int(code) for thread, code in raw.items()}


def _resolve_response_config(config: CueBox, request: ResponseCreateRequest) -> CueBox:
    selection: dict[str, object] = {"model": _require_model(config, request.model)}
    if request.reasoning is not None and request.reasoning.effort is not None:
        selection["reasoning_effort"] = request.reasoning.effort
    if request.service_tier is not None:
        selection["service_tier"] = request.service_tier
    return config.resolve(selection)


async def _prepare_create(
    *,
    svcs: svcs.Container,
    auth_context: AuthContext,
    request: ResponseCreateRequest,
    channels: ChannelsPlugin,
) -> State:
    sealing_keyring = svcs.get(SealingKeyring)
    response_store = svcs.get(ResponseStore)
    with tracer.start_as_current_span("response.prepare") as span:
        config = _resolve_response_config(svcs.get(CueBox), request)
        svcs.register_local_value(
            CompletionBudget,
            CompletionBudget(
                request.max_output_tokens,
                reasoning_to_output=float(config.reasoning_to_output),
            ),
        )
        prepared = await response_store.prepare_request(auth_context, request)
        thread_codes = _thread_codes(svcs)
        ingested = await ingest_response_request(
            prepared.execution_request,
            keyring=sealing_keyring,
            thread_codes=thread_codes,
        )
        response_request = prepared.response_request
        coordinator = StreamCoordinator(
            request=response_request,
            channels=channels,
            prepared=prepared,
            response_store=response_store,
            sealing_keyring=sealing_keyring,
            last_reasoning_id=ingested.last_reasoning_id,
            last_compaction_id=ingested.last_compaction_id,
        )
        svcs.register_local_value(StreamCoordinator, coordinator)
        state = State.from_ingested(
            ingested=ingested,
            request=response_request,
            config=config,
            svcs=svcs,
            thread_codes=thread_codes,
        )
        span.set_attribute("plap.response.id", coordinator.response_id)
        span.set_attribute("plap.response.model", state.request.model)
        if state.request.conversation_id is not None:
            span.set_attribute("plap.response.conversation_id", state.request.conversation_id)
        return state


def _response_context(state: State):
    coordinator = state.svcs.get(StreamCoordinator)
    return bound_context(
        conversation_id=state.request.conversation_id,
        response_id=coordinator.response_id,
    )


def _resolved_model_config(
    config: CueBox,
    *,
    model: str,
    service_tier: ServiceTier | None = None,
    reasoning_effort: ReasoningEffort | None = None,
) -> CueBox:
    request: dict[str, object] = {"model": _require_model(config, model)}
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    if service_tier is not None:
        request["service_tier"] = service_tier
    return config.resolve(request)


def _model_object(config: CueBox, *, model: str) -> ModelObject:
    return ModelObject(
        id=model,
        created=int(config.model_info.created),
        owned_by=str(config.model_info.provider),
    )


def _model_info_object(config: CueBox, *, model: str) -> ModelInfoObject:
    info = config.model_info
    return ModelInfoObject(
        id=model,
        display_name=str(info.display_name),
        description=str(info.description),
        mode=str(info.mode),
        input_modalities=[str(value) for value in info.input_modalities],
        output_modalities=[str(value) for value in info.output_modalities],
        max_input_tokens=int(info.max_input_tokens),
        max_output_tokens=int(info.max_output_tokens),
        supported_parameters=[str(value) for value in info.supported_parameters],
        pricing=ModelInfoPricingObject(
            input_per_token=float(info.pricing.input_per_token),
            output_per_token=float(info.pricing.output_per_token),
        ),
        provider=str(info.provider),
        deprecated=bool(info.deprecated),
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
    return isinstance(event, ResponseCompletedEvent | ResponseFailedEvent | ResponseIncompleteEvent | ResponseErrorEvent)


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


async def _accept_response(state: State) -> None:
    coordinator = state.svcs.get(StreamCoordinator)
    await anyio.sleep(0)
    with anyio.CancelScope(shield=True):
        await coordinator.created()


async def _execute_response(state: State) -> None:
    coordinator = state.svcs.get(StreamCoordinator)
    try:
        await coordinator.in_progress()
        await bus.emit("response.start", state=state)
    except anyio.get_cancelled_exc_class():
        if coordinator.current_response().status == "in_progress":
            with anyio.CancelScope(shield=True):
                await state.save_progress()
                await coordinator.cancelled()
        raise
    except ResponseFinalizationError:
        raise
    except Exception:
        if coordinator.current_response().status != "in_progress":
            logger.exception("response.execute.after_failed", response_id=coordinator.response_id)
            return
        logger.exception("response.execute.failed", response_id=coordinator.response_id)
    else:
        if coordinator.current_response().status != "in_progress":
            return
        logger.error("response.execute.unterminated", response_id=coordinator.response_id)

    with anyio.CancelScope(shield=True):
        await state.save_progress()
        try:
            await coordinator.failed()
        except Exception as exc:
            raise ResponseFinalizationError("response failure finalization failed") from exc


async def _response_event_stream(
    state: State,
    subscriber: Subscriber,
    channels: ChannelsPlugin,
) -> AsyncIterator[ResponseStreamEvent]:
    coordinator = state.svcs.get(StreamCoordinator)
    try:
        async with anyio.create_task_group() as task_group:
            try:
                with _response_context(state):
                    task_group.start_soon(_execute_response, state)
                    async for payload in subscriber.iter_events():
                        event = _decode_stream_event(payload)
                        yield event
                        if _is_terminal_event(event):
                            break
            except GeneratorExit:
                return
            finally:
                task_group.cancel_scope.cancel()
    finally:
        with anyio.CancelScope(shield=True):
            await channels.unsubscribe(subscriber, coordinator.channel)


async def create(
    data: ResponseCreateRequest,
    *,
    svcs: Container,
    auth_context: AuthContext,
    channels: ChannelsPlugin,
    scope: dict[str, Any] | None = None,
) -> ResponseObject:
    state = await _prepare_create(
        svcs=svcs,
        auth_context=auth_context,
        request=data,
        channels=channels,
    )
    coordinator = state.svcs.get(StreamCoordinator)
    if scope is not None:
        record_scope_context(
            scope,
            conversation_id=state.request.conversation_id,
            response_id=coordinator.response_id,
        )
    with _response_context(state):
        await _accept_response(state)
        try:
            await _execute_response(state)
        except ResponseFinalizationError:
            if coordinator.current_response().status == "in_progress":
                raise
            logger.exception("response.delivery.failed", response_id=coordinator.response_id)
    return coordinator.current_response()


async def stream(
    data: ResponseCreateRequest,
    *,
    svcs: Container,
    auth_context: AuthContext,
    channels: ChannelsPlugin,
    scope: dict[str, Any] | None = None,
) -> AsyncIterator[ResponseStreamEvent]:
    state = await _prepare_create(
        svcs=svcs,
        auth_context=auth_context,
        request=data,
        channels=channels,
    )
    coordinator = state.svcs.get(StreamCoordinator)
    if scope is not None:
        record_scope_context(
            scope,
            conversation_id=state.request.conversation_id,
            response_id=coordinator.response_id,
        )
    subscriber = await channels.subscribe(coordinator.channel)
    try:
        with _response_context(state):
            await _accept_response(state)
    except BaseException:
        with anyio.CancelScope(shield=True):
            await channels.unsubscribe(subscriber, coordinator.channel)
        raise
    return _response_event_stream(state, subscriber, channels)


def _unexpected_public_error() -> PublicError:
    return PublicError(
        status_code=500,
        type="server_error",
        code="internal_error",
        message="An unexpected error occurred.",
    )


def _websocket_error_event(*, public: PublicError) -> ResponseErrorEvent:
    # Preparation failed before a response coordinator existed, so this error
    # begins its own one-event sequence.
    return ResponseErrorEvent(
        code=public.code,
        message=public.message,
        param=public.param,
        sequence_number=1,
        type="error",
    )


def _preparation_error_event(exc: Exception) -> ResponseErrorEvent:
    if isinstance(exc, PlapError):
        public = exc.public or _unexpected_public_error()
        exc.log(
            logger,
            failure_code=public.code,
            failure_type=public.type,
            status_code=public.status_code,
        )
        return _websocket_error_event(public=public)

    public = _unexpected_public_error()
    logger.exception(
        "response.prepare.failed",
        failure_code=public.code,
        failure_type=public.type,
        status_code=public.status_code,
    )
    return _websocket_error_event(public=public)


def _single_group_exception(exc: BaseExceptionGroup) -> BaseException:
    current: BaseException = exc
    while isinstance(current, BaseExceptionGroup) and len(current.exceptions) == 1:
        current = current.exceptions[0]
    return current


async def _sse_response_payload(
    *,
    events: AsyncIterator[ResponseStreamEvent],
    projection: ResponseProjection,
) -> AsyncIterator[str]:
    async for event in events:
        yield msgspec.json.encode(projection.stream_payload(event)).decode()
    yield "[DONE]"


@post("/v1/responses", status_code=200)
async def create_response(
    request: Request[object, object, object],
    data: ResponseCreateRequest,
    svcs: Any,
    auth_context: AuthContext,
) -> object:
    projection = ResponseProjection.from_create_request(data, transport="stream" if data.stream else "snapshot")
    projection.validate_create_request(data)
    channels = request.app.plugins.get(ChannelsPlugin)
    if data.stream:
        events = await stream(
            data,
            svcs=svcs,
            auth_context=auth_context,
            channels=channels,
            scope=request.scope,
        )
        return ServerSentEvent(
            _sse_response_payload(
                events=events,
                projection=projection,
            ),
            headers={"content-type": "text/event-stream; charset=utf-8"},
        )

    try:
        async with anyio.create_task_group() as task_group:
            watcher_scope = await task_group.start(_watch_http_disconnect, request, task_group.cancel_scope)
            try:
                response = await create(
                    data,
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
    return projection.response(response)


@get("/v1/models")
async def list_models(
    svcs: Any,
) -> ModelListObject:
    config = svcs.get(CueBox)
    return ModelListObject(data=[_model_object(_resolved_model_config(config, model=model), model=model) for model in _model_names(config)])


@get("/v1/model/info")
async def model_info(
    model: str,
    svcs: Any,
    service_tier: ServiceTier | None = None,
    reasoning_effort: ReasoningEffort | None = None,
) -> ModelInfoListObject:
    config = svcs.get(CueBox)
    resolved = _resolved_model_config(
        config,
        model=model,
        service_tier=service_tier,
        reasoning_effort=reasoning_effort,
    )
    return ModelInfoListObject(data=[_model_info_object(resolved, model=model)])


@get("/v1/responses/{response_id:str}")
async def retrieve_response(
    response_id: str,
    svcs: Any,
    auth_context: AuthContext,
    include: list[str] | None = None,
    include_obfuscation: bool | None = None,
    starting_after: int | None = None,
    stream: bool | None = None,
) -> ResponseObject:
    _ = starting_after, stream
    response_store = svcs.get(ResponseStore)
    with bound_context(response_id=response_id):
        response = await response_store.get_response(auth_context, response_id)
        if response is None:
            raise _response_not_found_error(response_id, action="retrieve")
        projection = ResponseProjection.from_query(include, include_obfuscation=include_obfuscation)
        return projection.response(response)


@delete(
    "/v1/responses/{response_id:str}",
    status_code=200,
)
async def delete_response(
    response_id: str,
    svcs: Any,
    auth_context: AuthContext,
) -> ResponseDeleted:
    response_store = svcs.get(ResponseStore)
    with bound_context(response_id=response_id):
        deleted = await response_store.delete_response(auth_context, response_id)
        if not deleted:
            raise _response_not_found_error(response_id, action="delete")
        return ResponseDeleted(deleted=True, id=response_id)


@post("/v1/responses/compact", status_code=200)
async def compact_response(
    data: CompactRequest,
    svcs: Any,
) -> CompactedResponseObject:
    _ = data, svcs
    raise _not_implemented_error(action="compact")


@get(
    "/v1/responses/{response_id:str}/input_items",
)
async def list_input_items(
    response_id: str,
    svcs: Any,
    auth_context: AuthContext,
    after: str | None = None,
    include: list[str] | None = None,
    limit: int | None = None,
    order: str | None = None,
) -> InputItemsPage:
    response_store = svcs.get(ResponseStore)
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


@websocket("/v1/responses", dependencies={"auth_context": Provide(provide_socket_auth_context, sync_to_thread=False)})
async def responses_socket(
    socket: WebSocket,
    auth_context: AuthContext,
) -> None:
    channels = socket.app.plugins.get(ChannelsPlugin)
    registry: svcs.Registry = socket.app.state.svcs_registry
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
                _websocket_error_event(
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

        response_svcs = Container(registry)
        try:
            started_at = time.perf_counter()
            with tracer.start_as_current_span("websocket.response.create") as span:
                try:
                    projection = ResponseProjection.from_create_request(client_event.response, transport="stream")
                    projection.validate_create_request(client_event.response)
                    state = await _prepare_create(
                        svcs=response_svcs,
                        auth_context=auth_context,
                        request=client_event.response,
                        channels=channels,
                    )
                except Exception as exc:
                    await socket.send_json(_preparation_error_event(exc).model_dump(mode="json", exclude_none=True))
                    continue
                coordinator = state.svcs.get(StreamCoordinator)
                span.set_attribute("plap.response.id", coordinator.response_id)
                subscriber = await channels.subscribe(coordinator.channel)
                try:
                    with _response_context(state):
                        await _accept_response(state)
                    async with anyio.create_task_group() as task_group:
                        task_group.start_soon(_watch_socket_disconnect, socket, task_group.cancel_scope)
                        with _response_context(state):
                            task_group.start_soon(_execute_response, state)
                            async for payload in _iter_projected_payloads(subscriber, projection=projection):
                                await socket.send_json(payload)
                        logger.info(
                            "websocket.response.completed",
                            duration_ms=round((time.perf_counter() - started_at) * 1000.0, 3),
                        )
                        task_group.cancel_scope.cancel()
                finally:
                    with anyio.CancelScope(shield=True):
                        await channels.unsubscribe(subscriber, coordinator.channel)
        finally:
            with anyio.CancelScope(shield=True):
                await response_svcs.aclose()


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
