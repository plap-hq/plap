from __future__ import annotations

from collections.abc import AsyncIterator

import structlog
from litestar import delete, get, post, websocket
from litestar.connection import WebSocket
from litestar.response import ServerSentEvent
from pydantic import ValidationError

from plap.auth import AuthContext
from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.keyring import SealingKeyring
from plap.llms.chat import IChatCompletionClient
from plap.logging import log_debug, log_payload
from plap.responses.contracts import (
    CompactRequest,
    InputItemsPage,
    InputTokensCountRequest,
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
from plap.responses.dependencies import (
    HTTP_ROUTE_DEPENDENCIES,
    WEBSOCKET_ROUTE_DEPENDENCIES,
)
from plap.responses.reasoning import IReasoningSummarizer
from plap.responses.runtime import stream_response_events
from plap.responses.store import ResponseStore
from plap.responses.tools import IToolCallPolicyResolver, IToolPolicyResolver
from plap.responses.tools.mcp import IMCPToolProvider
from plap.settings import RuntimeSelector, Settings

logger = structlog.get_logger(__name__)


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


def _unsupported_operation_error(*, code: str, message: str, reason: str, private_message: str) -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code=code,
            message=message,
        ),
        private=PrivateError(
            event="response.unsupported_operation",
            reason=reason,
            message=private_message,
            level=ErrorLevel.WARNING,
        ),
    )


async def _sse_payload(
    events: AsyncIterator[ResponseStreamEvent],
) -> AsyncIterator[str]:
    last_sequence_number = 0
    try:
        async for event in events:
            last_sequence_number = event.sequence_number
            yield event.model_dump_json(exclude_none=True)
    except PlapError as exc:
        public = exc.public or PublicError(
            status_code=500,
            type="server_error",
            code="server_error",
            message="Response generation failed.",
        )
        exc.log(
            logger,
            failure_code=public.code,
            failure_type=public.type,
            path="/v1/responses",
            status_code=public.status_code,
            transport="sse",
        )
        yield build_error_event(public=public).model_copy(update={"sequence_number": last_sequence_number + 1}).model_dump_json(
            exclude_none=True
        )
    except Exception:
        logger.exception(
            "response.sse.unhandled_failed",
            error_type="server_error",
            failure_code="server_error",
            failure_type="server_error",
            path="/v1/responses",
            status_code=500,
            transport="sse",
        )
        yield build_error_event(
            public=PublicError(
                status_code=500,
                type="server_error",
                code="server_error",
                message="Response generation failed.",
            )
        ).model_copy(update={"sequence_number": last_sequence_number + 1}).model_dump_json(exclude_none=True)
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
    response_store: ResponseStore,
    tool_policy_resolver: IToolPolicyResolver,
    tool_call_policy_resolver: IToolCallPolicyResolver,
    mcp_tool_providers: tuple[IMCPToolProvider, ...],
) -> object:
    log_debug(
        logger,
        "response.request.received",
        conversation=data.conversation.id if hasattr(data.conversation, "id") else data.conversation,
        model=data.model,
        previous_response_id=data.previous_response_id,
        reasoning_effort=data.reasoning.effort if data.reasoning else None,
        reasoning_summary=(data.reasoning.summary or data.reasoning.generate_summary) if data.reasoning else None,
        stream=data.stream,
        tool_count=len(data.tools or []),
    )
    log_payload(logger, "response.request.payload", payload=data.model_dump(mode="json", exclude_none=True))
    _ = auth_context
    events = stream_response_events(
        data,
        auth_context=auth_context,
        settings=settings,
        sealing_keyring=sealing_keyring,
        chat_completion_client=chat_completion_client,
        reasoning_summarizer=reasoning_summarizer,
        response_store=response_store,
        tool_policy_resolver=tool_policy_resolver,
        tool_call_policy_resolver=tool_call_policy_resolver,
        mcp_tool_providers=mcp_tool_providers,
    )
    if data.stream:
        return ServerSentEvent(
            _sse_payload(events),
            headers={"content-type": "text/event-stream; charset=utf-8"},
        )
    return _completed_response_from_events([event async for event in events])


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
    _ = include, include_obfuscation, starting_after, stream
    response = await response_store.get_response(auth_context, response_id)
    if response is None:
        raise _response_not_found_error(response_id, action="retrieve")
    return response


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
    deleted = await response_store.delete_response(auth_context, response_id)
    if not deleted:
        raise _response_not_found_error(response_id, action="delete")
    return ResponseDeleted(deleted=True, id=response_id)


@post("/v1/responses/compact", status_code=200, dependencies=HTTP_ROUTE_DEPENDENCIES)
async def compact_response(
    data: CompactRequest,
    auth_context: AuthContext,
) -> object:
    _ = auth_context
    _ = data
    raise _unsupported_operation_error(
        code="unsupported_operation",
        message="Response compaction is not supported.",
        reason="response_compaction_unsupported",
        private_message="response compaction route is not supported",
    )


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
    _ = include
    return await response_store.list_input_items(
        auth_context,
        response_id,
        after=after,
        limit=limit,
        order=order,
    )


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
    raise _unsupported_operation_error(
        code="unsupported_operation",
        message="Response input token counting is not supported.",
        reason="response_input_token_count_unsupported",
        private_message="response input token counting route is not supported",
    )


@websocket("/v1/responses", dependencies=WEBSOCKET_ROUTE_DEPENDENCIES)
async def responses_socket(
    socket: WebSocket,
    auth_context: AuthContext,
    settings: Settings,
    sealing_keyring: SealingKeyring,
    chat_completion_client: IChatCompletionClient,
    reasoning_summarizer: IReasoningSummarizer,
    response_store: ResponseStore,
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

        try:
            log_debug(
                logger,
                "response.socket.request.received",
                model=client_event.response.model,
                stream=client_event.response.stream,
            )
            log_payload(logger, "response.socket.request.payload", payload=client_event.model_dump(mode="json", exclude_none=True))
            events = stream_response_events(
                client_event.response,
                auth_context=auth_context,
                settings=settings,
                sealing_keyring=sealing_keyring,
                chat_completion_client=chat_completion_client,
                reasoning_summarizer=reasoning_summarizer,
                response_store=response_store,
                tool_policy_resolver=tool_policy_resolver,
                tool_call_policy_resolver=tool_call_policy_resolver,
                mcp_tool_providers=mcp_tool_providers,
            )
            async for event in events:
                await socket.send_json(event.model_dump(mode="json", exclude_none=True))
        except PlapError as exc:
            public = exc.public or PublicError(
                status_code=500,
                type="server_error",
                code="server_error",
                message="Response generation failed.",
            )
            exc.log(
                logger,
                failure_code=public.code,
                failure_type=public.type,
                path="/v1/responses",
                status_code=public.status_code,
                transport="websocket",
            )
            await socket.send_json(
                build_error_event(public=public).model_dump(
                    mode="json",
                    exclude_none=True,
                )
            )
        except Exception:
            logger.exception(
                "response.socket.unhandled_failed",
                error_type="server_error",
                failure_code="server_error",
                failure_type="server_error",
                path="/v1/responses",
                status_code=500,
                transport="websocket",
            )
            await socket.send_json(
                build_error_event(
                    public=PublicError(
                        status_code=500,
                        type="server_error",
                        code="server_error",
                        message="Response generation failed.",
                    )
                ).model_dump(
                    mode="json",
                    exclude_none=True,
                )
            )


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
    compact_response,
    list_input_items,
    count_input_tokens,
    responses_socket,
]
