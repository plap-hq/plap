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
from plap.responses.errors import ResponseError
from plap.responses.reasoning import IReasoningSummarizer
from plap.responses.runtime import stream_response_events
from plap.responses.store import ResponseStore
from plap.responses.tools import IToolCallPolicyResolver, IToolPolicyResolver
from plap.responses.tools.mcp import IMCPToolProvider
from plap.settings import RuntimeSelector, Settings


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
    response_store: ResponseStore,
    tool_policy_resolver: IToolPolicyResolver,
    tool_call_policy_resolver: IToolCallPolicyResolver,
    mcp_tool_providers: tuple[IMCPToolProvider, ...],
) -> object:
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
        data=[
            profile.to_model_object(model=model)
            for model, profile in sorted(settings.runtime_model_profiles.items())
        ]
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
    try:
        profile = settings.resolve_runtime_model_profile(
            model,
            selector=RuntimeSelector(
                service_tier=service_tier,
                reasoning_effort=reasoning_effort,
            ),
        )
    except ValueError as exc:
        raise ResponseError.invalid_request(private_message=str(exc), cause=exc) from exc
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
        raise ResponseError.not_found(private_message=f"response not found: {response_id}")
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
        raise ResponseError.not_found(private_message=f"response not found for deletion: {response_id}")
    return ResponseDeleted(deleted=True, id=response_id)


@post("/v1/responses/compact", status_code=200, dependencies=HTTP_ROUTE_DEPENDENCIES)
async def compact_response(
    data: CompactRequest,
    auth_context: AuthContext,
) -> object:
    _ = auth_context
    _ = data
    raise ResponseError.unsupported_operation(private_message="response compaction route is not supported")


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
    raise ResponseError.unsupported_operation(private_message="response input token counting route is not supported")


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
        except ResponseError as exc:
            await socket.send_json(
                build_error_event(exc.public.message).model_dump(
                    mode="json",
                    exclude_none=True,
                )
            )
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
