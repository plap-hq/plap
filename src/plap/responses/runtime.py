from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence

import anyio
from anyio.abc import ObjectSendStream

from plap.keyring import SealingKeyring
from plap.llms.chat import ChatToolCall
from plap.responses.contracts import (
    FunctionTool,
    ResponseCompletedEvent,
    ResponseCreateRequest,
    ResponseObject,
    ResponseStreamEvent,
    WebSearchTool,
)
from plap.responses.events import build_stream_events
from plap.responses.ingest import (
    IngestedQueues,
    IngestionError,
    ingest_response_request,
)
from plap.responses.objects import build_response_object
from plap.responses.tools import (
    IToolCallPolicyResolver,
    IToolPolicyResolver,
    ToolCall,
    ToolPolicy,
    ToolPolicyError,
)
from plap.responses.tools.compress import (
    COMPRESS_TOOL_NAME,
    compress_policy,
    compress_tool,
)
from plap.responses.tools.web_search import (
    IMCPToolProvider,
    web_search_policy,
)
from plap.settings import RuntimeModelProfileConfig, Settings

SERVER_TOOL_NAMES = frozenset({COMPRESS_TOOL_NAME})


async def stream_response_events(
    request: ResponseCreateRequest,
    *,
    settings: Settings,
    sealing_keyring: SealingKeyring,
    tool_policy_resolver: IToolPolicyResolver,
    tool_call_policy_resolver: IToolCallPolicyResolver,
    web_search_tool_provider: IMCPToolProvider | None = None,
) -> AsyncIterator[ResponseStreamEvent]:
    _ = tool_call_policy_resolver
    send, receive = anyio.create_memory_object_stream[ResponseStreamEvent](16)
    producer_error: Exception | None = None

    async def produce() -> None:
        nonlocal producer_error
        async with send:
            try:
                await _produce_response_events(
                    send,
                    request,
                    settings=settings,
                    sealing_keyring=sealing_keyring,
                    tool_policy_resolver=tool_policy_resolver,
                    web_search_tool_provider=web_search_tool_provider,
                )
            except Exception as exc:
                producer_error = exc

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(produce)
        async with receive:
            async for event in receive:
                yield event
    if producer_error is not None:
        raise producer_error


def completed_response_from_events(
    events: Sequence[ResponseStreamEvent],
) -> ResponseObject:
    for event in reversed(events):
        if isinstance(event, ResponseCompletedEvent):
            return event.response
    raise RuntimeError("response stream did not complete")


async def _produce_response_events(
    send: ObjectSendStream[ResponseStreamEvent],
    request: ResponseCreateRequest,
    *,
    settings: Settings,
    sealing_keyring: SealingKeyring,
    tool_policy_resolver: IToolPolicyResolver,
    web_search_tool_provider: IMCPToolProvider | None,
) -> None:
    profile = _runtime_model_profile(settings, request)
    ingested = await ingest_response_request(
        request,
        keyring=sealing_keyring,
        transcript_token_budget=profile.transcript_token_budget,
    )
    await prepare_tools(
        request,
        ingested,
        tool_policy_resolver,
        web_search_tool_provider,
    )
    response = build_response_object(request)
    for event in build_stream_events(response):
        await send.send(event)


def _runtime_model_profile(
    settings: Settings,
    request: ResponseCreateRequest,
) -> RuntimeModelProfileConfig:
    if request.model is None:
        raise IngestionError("model is required")
    profile = settings.runtime_model_profiles.get(request.model)
    if profile is None:
        raise IngestionError("unknown runtime model")
    return profile.for_service_tier(request.service_tier)


async def prepare_tools(
    request: ResponseCreateRequest,
    ingested: IngestedQueues,
    resolver: IToolPolicyResolver,
    web_search_tool_provider: IMCPToolProvider | None = None,
) -> tuple[tuple[FunctionTool, ...], dict[str, ToolPolicy]]:
    client_tools = _client_tools(request.tools or ())

    server_tools: list[FunctionTool] = []
    if _has_web_search(request.tools or ()):
        if web_search_tool_provider is None:
            raise ToolPolicyError("web_search requested but no MCP provider configured")
        server_tools.extend(await web_search_tool_provider.tools())

    if not ingested.in_temp_debate:
        server_tools.append(compress_tool())

    _reject_server_name_collisions(client_tools, server_tools)

    tools = [*client_tools, *server_tools]
    tool_policies = await resolver.resolve(client_tools)

    for tool in server_tools:
        if tool.name == COMPRESS_TOOL_NAME:
            tool_policies[tool.name] = compress_policy()
        else:
            tool_policies[tool.name] = web_search_policy(tool.name)


    return tuple(tools), tool_policies


async def resolve_tool_calls(
    calls: Sequence[ChatToolCall],
    *,
    tools: Mapping[str, FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    resolver: IToolCallPolicyResolver,
) -> tuple[ToolPolicy, ...]:
    if not calls:
        return ()
    _validate_tool_call_batch(calls, tool_policies)

    resolved: list[ToolPolicy | None] = []
    client_calls: list[ToolCall] = []
    client_indexes: list[int] = []
    for call in calls:
        policy = tool_policies[call.name]
        if policy.source == "server":
            resolved.append(policy)
            continue

        tool = tools.get(call.name)
        if tool is None:
            raise ToolPolicyError(f"unknown client tool call: {call.name}")
        client_indexes.append(len(resolved))
        resolved.append(None)
        client_calls.append(
            ToolCall(
                tool=tool,
                policy=policy,
                arguments=call.arguments,
            )
        )

    if client_calls:
        client_policies = await resolver.resolve(client_calls)
        for index, policy in zip(client_indexes, client_policies, strict=True):
            resolved[index] = policy

    if any(policy is None for policy in resolved):
        raise RuntimeError("tool call policy resolution did not produce all outputs")
    return tuple(policy for policy in resolved if policy is not None)


def _client_tools(tools: Sequence[object]) -> list[FunctionTool]:
    return [tool for tool in tools if isinstance(tool, FunctionTool)]


def _has_web_search(tools: Sequence[object]) -> bool:
    return any(isinstance(tool, WebSearchTool) for tool in tools)


def _reject_server_name_collisions(
    client_tools: Sequence[FunctionTool],
    server_tools: Sequence[FunctionTool],
) -> None:
    server_names = [tool.name for tool in server_tools]
    if len(set(server_names)) != len(server_names):
        raise ToolPolicyError("server tool names must be unique")
    server_tool_names = set(server_names) | SERVER_TOOL_NAMES
    for tool in client_tools:
        if tool.name in server_tool_names:
            raise ToolPolicyError(f"function tool name is reserved: {tool.name}")


def _validate_tool_call_batch(
    calls: Sequence[ChatToolCall],
    tool_policies: Mapping[str, ToolPolicy],
) -> None:
    for call in calls:
        if call.name not in tool_policies:
            raise ToolPolicyError(f"unknown tool call: {call.name}")
    if any(call.name == COMPRESS_TOOL_NAME for call in calls) and len(calls) != 1:
        raise ToolPolicyError("compress must be called alone")
