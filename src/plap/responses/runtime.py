from __future__ import annotations

from collections.abc import Mapping, Sequence

from plap.llms.chat import ChatToolCall
from plap.responses.contracts import FunctionTool, ResponseCreateRequest, WebSearchTool
from plap.responses.ingest import IngestedQueues
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
    WEB_SEARCH_TOOL_NAME,
    web_search_policy,
    web_search_tool,
)

SERVER_TOOL_NAMES = frozenset({COMPRESS_TOOL_NAME, WEB_SEARCH_TOOL_NAME})


async def prepare_tools(
    request: ResponseCreateRequest,
    ingested: IngestedQueues,
    resolver: IToolPolicyResolver,
) -> tuple[tuple[FunctionTool, ...], dict[str, ToolPolicy]]:
    client_tools = _client_tools(request.tools or ())
    _reject_server_name_collisions(client_tools)

    tools = list(client_tools)
    tool_policies = await resolver.resolve(client_tools)

    web_search = _web_search(request.tools or ())
    if web_search is not None:
        tools.append(web_search_tool(web_search))
        tool_policies[WEB_SEARCH_TOOL_NAME] = web_search_policy()

    if not ingested.in_temp_debate:
        tools.append(compress_tool())
        tool_policies[COMPRESS_TOOL_NAME] = compress_policy()

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


def _web_search(tools: Sequence[object]) -> WebSearchTool | None:
    for tool in tools:
        if isinstance(tool, WebSearchTool):
            return tool
    return None


def _reject_server_name_collisions(tools: Sequence[FunctionTool]) -> None:
    for tool in tools:
        if tool.name in SERVER_TOOL_NAMES:
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
