from __future__ import annotations

from collections.abc import Sequence

import pytest

from plap.llms.chat import ChatToolCall
from plap.responses.contracts import FunctionTool, ResponseCreateRequest, WebSearchTool
from plap.responses.ingest import IngestedQueues
from plap.responses.runtime import (
    COMPRESS_TOOL_NAME,
    WEB_SEARCH_TOOL_NAME,
    prepare_tools,
    resolve_tool_calls,
)
from plap.responses.tools import (
    IToolCallPolicyResolver,
    IToolPolicyResolver,
    ToolCall,
    ToolPolicy,
    ToolPolicyError,
)


async def test_prepare_tools_injects_compress_and_classifies_client_tools() -> None:
    resolver = _RecordingResolver()

    tools, policies = await prepare_tools(
        ResponseCreateRequest(tools=[_read_file_tool()]),
        _ingested(),
        resolver,
    )

    assert [tool.name for tool in tools] == ["read_file", COMPRESS_TOOL_NAME]
    assert resolver.tool_names == [["read_file"]]
    assert policies["read_file"].source == "client"
    assert policies[COMPRESS_TOOL_NAME].source == "server"
    assert policies[COMPRESS_TOOL_NAME].effect_class == "safe"


async def test_prepare_tools_omits_compress_during_temp_debate() -> None:
    tools, policies = await prepare_tools(
        ResponseCreateRequest(tools=[_read_file_tool()]),
        _ingested(in_temp_debate=True),
        _RecordingResolver(),
    )

    assert [tool.name for tool in tools] == ["read_file"]
    assert COMPRESS_TOOL_NAME not in policies


async def test_prepare_tools_adds_web_search_only_when_requested() -> None:
    tools, policies = await prepare_tools(
        ResponseCreateRequest(
            tools=[_read_file_tool(), WebSearchTool(type="web_search")]
        ),
        _ingested(),
        _RecordingResolver(),
    )

    assert [tool.name for tool in tools] == [
        "read_file",
        WEB_SEARCH_TOOL_NAME,
        COMPRESS_TOOL_NAME,
    ]
    assert policies[WEB_SEARCH_TOOL_NAME].source == "server"
    assert policies[WEB_SEARCH_TOOL_NAME].effect_class == "safe"


async def test_prepare_tools_rejects_client_server_name_collision() -> None:
    with pytest.raises(ToolPolicyError, match="reserved"):
        await prepare_tools(
            ResponseCreateRequest(tools=[_tool(COMPRESS_TOOL_NAME)]),
            _ingested(),
            _RecordingResolver(),
        )

    with pytest.raises(ToolPolicyError, match="reserved"):
        await prepare_tools(
            ResponseCreateRequest(tools=[_tool(WEB_SEARCH_TOOL_NAME)]),
            _ingested(),
            _RecordingResolver(),
        )


async def test_resolve_tool_calls_classifies_client_calls_as_ordered_batch() -> None:
    tool = _read_file_tool()
    tools, policies = await prepare_tools(
        ResponseCreateRequest(
            tools=[tool, WebSearchTool(type="web_search")]
        ),
        _ingested(),
        _RecordingResolver(),
    )
    call_resolver = _RecordingCallResolver()

    resolved = await resolve_tool_calls(
        [
            ChatToolCall(id="call_1", name="web_search", arguments='{"query":"x"}'),
            ChatToolCall(id="call_2", name="read_file", arguments='{"path":"a"}'),
            ChatToolCall(id="call_3", name="read_file", arguments='{"path":"b"}'),
        ],
        tools={tool.name: tool for tool in tools},
        tool_policies=policies,
        resolver=call_resolver,
    )

    assert [policy.name for policy in resolved] == [
        "web_search",
        "read_file",
        "read_file",
    ]
    assert [policy.source for policy in resolved] == ["server", "client", "client"]
    assert call_resolver.calls == [
        [("read_file", '{"path":"a"}'), ("read_file", '{"path":"b"}')]
    ]


async def test_resolve_tool_calls_rejects_compress_mixed_with_other_calls() -> None:
    _, policies = await prepare_tools(
        ResponseCreateRequest(tools=[WebSearchTool(type="web_search")]),
        _ingested(),
        _RecordingResolver(),
    )

    with pytest.raises(ToolPolicyError, match="compress must be called alone"):
        await resolve_tool_calls(
            [
                ChatToolCall(id="call_1", name="compress", arguments="{}"),
                ChatToolCall(id="call_2", name="web_search", arguments='{"query":"x"}'),
            ],
            tools={},
            tool_policies=policies,
            resolver=_RecordingCallResolver(),
        )


class _RecordingResolver(IToolPolicyResolver):
    def __init__(self) -> None:
        self.tool_names: list[list[str]] = []

    async def resolve(
        self, tools: Sequence[FunctionTool]
    ) -> dict[str, ToolPolicy]:
        self.tool_names.append([tool.name for tool in tools])
        return {
            tool.name: ToolPolicy(
                name=tool.name,
                source="client",
                effect_class="safe",
            )
            for tool in tools
        }


class _RecordingCallResolver(IToolCallPolicyResolver):
    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    async def resolve(
        self, calls: Sequence[ToolCall]
    ) -> tuple[ToolPolicy, ...]:
        self.calls.append([(call.tool.name, call.arguments) for call in calls])
        return tuple(
            ToolPolicy(
                name=call.tool.name,
                source="client",
                effect_class="safe",
            )
            for call in calls
        )


def _ingested(*, in_temp_debate: bool = False) -> IngestedQueues:
    return IngestedQueues(
        main_context=(),
        main_transcript=(),
        reviewer=(),
        arbitrator=(),
        continuation_side="main",
        in_temp_debate=in_temp_debate,
        compaction=None,
        cursors={"m": 0, "s": 0},
    )


def _read_file_tool() -> FunctionTool:
    return _tool("read_file")


def _tool(name: str) -> FunctionTool:
    return FunctionTool(
        description="test tool",
        name=name,
        parameters={"type": "object"},
        strict=True,
        type="function",
    )
