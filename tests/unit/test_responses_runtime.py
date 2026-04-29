from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest

from plap.keyring import SealingKeyring
from plap.llms.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatMessage,
    ChatToolCall,
    IChatCompletionClient,
)
from plap.responses.contracts import (
    FunctionTool,
    ReasoningConfig,
    ReasoningItem,
    ResponseCreateRequest,
    WebSearchTool,
)
from plap.responses.ingest import (
    IngestedQueues,
    ReasoningPayload,
    content_hash,
    open_reasoning_payload,
    seal_reasoning_payload,
)
from plap.responses.reasoning import IReasoningSummarizer
from plap.responses.runtime import (
    COMPRESS_TOOL_NAME,
    prepare_tools,
    resolve_tool_calls,
    stream_response_events,
)
from plap.responses.tools import (
    EffectClass,
    IMCPToolProvider,
    IToolCallPolicyResolver,
    IToolPolicyResolver,
    ToolCall,
    ToolPolicy,
    ToolPolicyError,
)
from plap.settings import RuntimeModelProfileConfig, Settings

MCP_SEARCH_TOOL_NAME = "search_web"
MCP_NEWS_TOOL_NAME = "search_news"


async def test_prepare_tools_classifies_client_tools_without_compress() -> None:
    resolver = _RecordingResolver()

    tools, policies = await prepare_tools(
        ResponseCreateRequest(tools=[_read_file_tool()]),
        resolver,
    )

    assert [tool.name for tool in tools] == ["read_file"]
    assert resolver.tool_names == [["read_file"]]
    assert policies["read_file"].source == "client"


async def test_prepare_tools_adds_web_search_only_when_requested() -> None:
    tools, policies = await prepare_tools(
        ResponseCreateRequest(
            tools=[_read_file_tool(), WebSearchTool(type="web_search")]
        ),
        _RecordingResolver(),
        _FakeWebSearchProvider(),
    )

    assert [tool.name for tool in tools] == [
        "read_file",
        MCP_SEARCH_TOOL_NAME,
        MCP_NEWS_TOOL_NAME,
    ]
    assert policies[MCP_SEARCH_TOOL_NAME].source == "server"
    assert policies[MCP_SEARCH_TOOL_NAME].effect_class == "safe"


async def test_prepare_tools_rejects_web_search_when_mcp_is_not_configured() -> None:
    with pytest.raises(ToolPolicyError, match="web_search requested"):
        await prepare_tools(
            ResponseCreateRequest(tools=[WebSearchTool(type="web_search")]),
            _RecordingResolver(),
        )


async def test_prepare_tools_rejects_client_server_name_collision() -> None:
    with pytest.raises(ToolPolicyError, match="reserved"):
        await prepare_tools(
            ResponseCreateRequest(tools=[_tool(COMPRESS_TOOL_NAME)]),
            _RecordingResolver(),
        )

    with pytest.raises(ToolPolicyError, match="reserved"):
        await prepare_tools(
            ResponseCreateRequest(
                tools=[
                    _tool(MCP_SEARCH_TOOL_NAME),
                    WebSearchTool(type="web_search"),
                ]
            ),
            _RecordingResolver(),
            _FakeWebSearchProvider(),
        )


async def test_resolve_tool_calls_classifies_client_calls_as_ordered_batch() -> None:
    tool = _read_file_tool()
    tools, policies = await prepare_tools(
        ResponseCreateRequest(
            tools=[tool, WebSearchTool(type="web_search")]
        ),
        _RecordingResolver(),
        _FakeWebSearchProvider(),
    )
    call_resolver = _RecordingCallResolver()

    resolved = await resolve_tool_calls(
        [
            ChatToolCall(
                id="call_1",
                name=MCP_SEARCH_TOOL_NAME,
                arguments='{"query":"x"}',
            ),
            ChatToolCall(id="call_2", name="read_file", arguments='{"path":"a"}'),
            ChatToolCall(id="call_3", name="read_file", arguments='{"path":"b"}'),
        ],
        tools={tool.name: tool for tool in tools},
        tool_policies=policies,
        resolver=call_resolver,
    )

    assert [policy.name for policy in resolved] == [
        MCP_SEARCH_TOOL_NAME,
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
        _RecordingResolver(),
        _FakeWebSearchProvider(),
    )
    policies[COMPRESS_TOOL_NAME] = ToolPolicy(
        name=COMPRESS_TOOL_NAME,
        source="server",
        effect_class="safe",
    )

    with pytest.raises(ToolPolicyError, match="compress must be called alone"):
        await resolve_tool_calls(
            [
                ChatToolCall(id="call_1", name="compress", arguments="{}"),
                ChatToolCall(
                    id="call_2",
                    name=MCP_SEARCH_TOOL_NAME,
                    arguments='{"query":"x"}',
                ),
            ],
            tools={},
            tool_policies=policies,
            resolver=_RecordingCallResolver(),
        )


async def test_stream_response_events_emits_model_message_output() -> None:
    client = _StaticChatClient(ChatMessage(role="assistant", content="hello back"))
    events = [
        event
        async for event in stream_response_events(
            ResponseCreateRequest(model="plap/test", input="hello"),
            settings=_settings(),
            sealing_keyring=_keyring(),
            tool_policy_resolver=_RecordingResolver(),
            tool_call_policy_resolver=_RecordingCallResolver(),
            chat_completion_client=client,
            reasoning_summarizer=_FakeReasoningSummarizer(),
        )
    ]

    assert [event.type for event in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    completed = events[-1].response
    assert completed.output[0].type == "message"
    assert completed.output[0].content[0].text == "hello back"
    assert [tool.function.name for tool in client.requests[0].tools] == [
        COMPRESS_TOOL_NAME
    ]
    developer_prompt = client.requests[0].messages[0]
    assert developer_prompt.role == "developer"
    assert "You are Test Model" in (developer_prompt.content or "")
    assert "Be accurate, direct, and helpful" in (developer_prompt.content or "")
    assert "The `compress` tool replaces" in (developer_prompt.content or "")


async def test_stream_response_events_appends_user_instructions_to_prompt() -> None:
    client = _StaticChatClient(ChatMessage(role="assistant", content="done"))

    _ = [
        event
        async for event in stream_response_events(
            ResponseCreateRequest(
                instructions="Prefer terse answers.",
                model="plap/test",
                input="hello",
            ),
            settings=_settings(),
            sealing_keyring=_keyring(),
            tool_policy_resolver=_RecordingResolver(),
            tool_call_policy_resolver=_RecordingCallResolver(),
            chat_completion_client=client,
            reasoning_summarizer=_FakeReasoningSummarizer(),
        )
    ]

    prompt = client.requests[0].messages[0].content or ""
    assert "User instructions:\nPrefer terse answers." in prompt


async def test_stream_response_events_sends_stable_and_temp_context() -> None:
    keyring = _keyring()
    client = _StaticChatClient(ChatMessage(role="assistant", content="done"))

    events = [
        event
        async for event in stream_response_events(
            ResponseCreateRequest(
                model="plap/test",
                input=[
                    ReasoningItem(
                        encrypted_content=seal_reasoning_payload(
                            ReasoningPayload(
                                side="main",
                                temp=True,
                                messages=(
                                    {
                                        "role": "assistant",
                                        "content": "temp candidate",
                                    },
                                ),
                            ),
                            keyring=keyring,
                        ),
                        id="rs_temp",
                        summary=[],
                        type="reasoning",
                    )
                ],
            ),
            settings=_settings(),
            sealing_keyring=keyring,
            tool_policy_resolver=_RecordingResolver(),
            tool_call_policy_resolver=_RecordingCallResolver(),
            chat_completion_client=client,
            reasoning_summarizer=_FakeReasoningSummarizer(),
        )
    ]

    assert events[-1].type == "response.completed"
    assert client.requests[0].messages[0].role == "developer"
    developer_prompt = client.requests[0].messages[0].content or ""
    assert "Test Model" in developer_prompt
    assert "compress" not in developer_prompt
    assert "compression" not in developer_prompt
    assert "compaction" not in developer_prompt
    assert [message.content for message in client.requests[0].messages[1:]] == [
        "temp candidate"
    ]
    assert client.requests[0].tools == []


async def test_stream_response_events_debate_exposes_only_safe_tools() -> None:
    keyring = _keyring()
    client = _StaticChatClient(ChatMessage(role="assistant", content="done"))
    resolver = _RecordingResolver(
        {
            "update_plan": "visible",
            "mutate_record": "mutation",
            "contextual_lookup": "contextual",
            "mystery_tool": "unknown",
        }
    )

    events = [
        event
        async for event in stream_response_events(
            ResponseCreateRequest(
                model="plap/test",
                input=[
                    ReasoningItem(
                        encrypted_content=seal_reasoning_payload(
                            ReasoningPayload(
                                side="main",
                                temp=True,
                                messages=(
                                    {
                                        "role": "assistant",
                                        "content": "temp candidate",
                                    },
                                ),
                            ),
                            keyring=keyring,
                        ),
                        id="rs_temp",
                        summary=[],
                        type="reasoning",
                    )
                ],
                tools=[
                    _read_file_tool(),
                    _tool("update_plan"),
                    _tool("mutate_record"),
                    _tool("contextual_lookup"),
                    _tool("mystery_tool"),
                    WebSearchTool(type="web_search"),
                ],
            ),
            settings=_settings(),
            sealing_keyring=keyring,
            tool_policy_resolver=resolver,
            tool_call_policy_resolver=_RecordingCallResolver(),
            chat_completion_client=client,
            reasoning_summarizer=_FakeReasoningSummarizer(),
            web_search_tool_provider=_FakeWebSearchProvider(),
        )
    ]

    assert events[-1].type == "response.completed"
    assert resolver.tool_names == [
        [
            "read_file",
            "update_plan",
            "mutate_record",
            "contextual_lookup",
            "mystery_tool",
        ]
    ]
    assert [tool.function.name for tool in client.requests[0].tools] == [
        "read_file",
        MCP_SEARCH_TOOL_NAME,
        MCP_NEWS_TOOL_NAME,
    ]


async def test_stream_response_events_emits_safe_client_function_call() -> None:
    call_resolver = _RecordingCallResolver()
    events = [
        event
        async for event in stream_response_events(
            ResponseCreateRequest(model="plap/test", tools=[_read_file_tool()]),
            settings=_settings(),
            sealing_keyring=_keyring(),
            tool_policy_resolver=_RecordingResolver(),
            tool_call_policy_resolver=call_resolver,
            chat_completion_client=_StaticChatClient(
                ChatMessage(
                    role="assistant",
                    tool_calls=[
                        ChatToolCall(
                            id="upstream_call_1",
                            name="read_file",
                            arguments='{"path":"README.md"}',
                        )
                    ],
                )
            ),
            reasoning_summarizer=_FakeReasoningSummarizer(),
        )
    ]

    completed = events[-1].response
    assert [item.type for item in completed.output] == ["message", "function_call"]
    assert completed.output[0].content[0].text == ""
    assert completed.output[1].name == "read_file"
    assert completed.output[1].call_id.startswith("call_")
    assert call_resolver.calls == [[("read_file", '{"path":"README.md"}')]]


async def test_stream_response_events_emits_visible_client_function_call() -> None:
    call_resolver = _RecordingCallResolver()
    events = [
        event
        async for event in stream_response_events(
            ResponseCreateRequest(model="plap/test", tools=[_tool("update_plan")]),
            settings=_settings(),
            sealing_keyring=_keyring(),
            tool_policy_resolver=_RecordingResolver({"update_plan": "visible"}),
            tool_call_policy_resolver=call_resolver,
            chat_completion_client=_StaticChatClient(
                ChatMessage(
                    role="assistant",
                    tool_calls=[
                        ChatToolCall(
                            id="upstream_call_1",
                            name="update_plan",
                            arguments='{"step":"test"}',
                        )
                    ],
                )
            ),
            reasoning_summarizer=_FakeReasoningSummarizer(),
        )
    ]

    completed = events[-1].response
    assert [item.type for item in completed.output] == ["message", "function_call"]
    assert completed.output[1].name == "update_plan"
    assert call_resolver.calls == [[("update_plan", '{"step":"test"}')]]


async def test_stream_response_events_patches_reasoning_to_unsealed_message() -> None:
    events = [
        event
        async for event in stream_response_events(
            ResponseCreateRequest(model="plap/test", input="hello"),
            settings=_settings(),
            sealing_keyring=_keyring(),
            tool_policy_resolver=_RecordingResolver(),
            tool_call_policy_resolver=_RecordingCallResolver(),
            chat_completion_client=_StaticChatClient(
                ChatMessage(
                    role="assistant",
                    content="answer",
                    reasoning_content="thinking",
                )
            ),
            reasoning_summarizer=_FakeReasoningSummarizer(),
        )
    ]

    completed = events[-1].response
    assert [item.type for item in completed.output] == ["message", "reasoning"]
    public_message = {"role": "assistant", "content": "answer"}
    payload = open_reasoning_payload(
        completed.output[1].encrypted_content,
        keyring=_keyring(),
    )
    assert payload.messages == (
        {
            "content_hash": content_hash(public_message),
            "reasoning_content": "thinking",
        },
    )


async def test_stream_response_events_streams_requested_reasoning_summary() -> None:
    summarizer = _FakeReasoningSummarizer(("checked ", "the answer"))
    events = [
        event
        async for event in stream_response_events(
            ResponseCreateRequest(
                input="hello",
                model="plap/test",
                reasoning=ReasoningConfig(summary="concise"),
            ),
            settings=_settings(),
            sealing_keyring=_keyring(),
            tool_policy_resolver=_RecordingResolver(),
            tool_call_policy_resolver=_RecordingCallResolver(),
            chat_completion_client=_StaticChatClient(
                ChatMessage(
                    role="assistant",
                    content="answer",
                    reasoning_content="thinking",
                )
            ),
            reasoning_summarizer=summarizer,
        )
    ]

    assert [
        event.type for event in events if "reasoning_summary" in event.type
    ] == [
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_summary_part.done",
    ]
    completed_reasoning = events[-1].response.output[1]
    assert completed_reasoning.summary[0].text == "checked the answer"
    assert summarizer.calls[0][0] == "crof/qwen3.5-9b"
    assert summarizer.calls[0][1] == "concise"
    assert summarizer.calls[0][2] == "main"
    assert summarizer.calls[0][3] == (
        {
            "content_hash": content_hash(
                {"role": "assistant", "content": "answer"}
            ),
            "reasoning_content": "thinking",
        },
    )


async def test_stream_response_events_patches_reasoning_and_emits_tool_call() -> None:
    events = [
        event
        async for event in stream_response_events(
            ResponseCreateRequest(model="plap/test", tools=[_read_file_tool()]),
            settings=_settings(),
            sealing_keyring=_keyring(),
            tool_policy_resolver=_RecordingResolver(),
            tool_call_policy_resolver=_RecordingCallResolver(),
            chat_completion_client=_StaticChatClient(
                ChatMessage(
                    role="assistant",
                    content="",
                    reasoning_content="thinking",
                    tool_calls=[
                        ChatToolCall(
                            id="upstream_call_1",
                            name="read_file",
                            arguments='{"path":"README.md"}',
                        )
                    ],
                )
            ),
            reasoning_summarizer=_FakeReasoningSummarizer(),
        )
    ]

    completed = events[-1].response
    assert [item.type for item in completed.output] == [
        "message",
        "reasoning",
        "function_call",
    ]
    public_message = {"role": "assistant", "content": ""}
    payload = open_reasoning_payload(
        completed.output[1].encrypted_content,
        keyring=_keyring(),
    )
    assert payload.messages == (
        {
            "content_hash": content_hash(public_message),
            "reasoning_content": "thinking",
        },
    )


class _RecordingResolver(IToolPolicyResolver):
    def __init__(self, effects: dict[str, EffectClass] | None = None) -> None:
        self.effects = effects or {}
        self.tool_names: list[list[str]] = []

    async def resolve(
        self, tools: Sequence[FunctionTool]
    ) -> dict[str, ToolPolicy]:
        self.tool_names.append([tool.name for tool in tools])
        return {
            tool.name: ToolPolicy(
                name=tool.name,
                source="client",
                effect_class=self.effects.get(tool.name, "safe"),
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
                effect_class=call.policy.effect_class,
            )
            for call in calls
        )


class _FakeWebSearchProvider(IMCPToolProvider):
    async def tools(self) -> tuple[FunctionTool, ...]:
        return (
            _tool(MCP_SEARCH_TOOL_NAME),
            _tool(MCP_NEWS_TOOL_NAME),
        )

    async def call_tool(self, name: str, arguments: dict[str, object]) -> str:
        _ = name, arguments
        return "search result"


class _FakeReasoningSummarizer(IReasoningSummarizer):
    def __init__(self, deltas: Sequence[str] = ()) -> None:
        self.deltas = tuple(deltas)
        self.calls: list[tuple[str, str, str, tuple[object, ...]]] = []

    async def stream(
        self,
        *,
        model: str,
        mode: str,
        side: str,
        messages: Sequence[object],
    ) -> AsyncIterator[str]:
        self.calls.append((model, mode, side, tuple(messages)))
        for delta in self.deltas:
            yield delta


class _StaticChatClient(IChatCompletionClient):
    def __init__(self, message: ChatMessage) -> None:
        self.message = message
        self.requests: list[ChatCompletionRequest] = []

    async def complete(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResult:
        self.requests.append(request)
        return ChatCompletionResult(
            id="chatcmpl_test",
            model=request.model,
            created_at=None,
            message=self.message,
            finish_reason="stop",
        )

    async def stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionDelta]:
        _ = request
        if False:
            yield ChatCompletionDelta(
                id="chatcmpl_test",
                model=None,
                created_at=None,
                choice_index=0,
            )


def _ingested(*, in_temp_debate: bool = False) -> IngestedQueues:
    return IngestedQueues(
        main_context=(),
        main_context_temp=(),
        main_transcript=(),
        reviewer=(),
        arbitrator=(),
        continuation_side="main",
        in_temp_debate=in_temp_debate,
        compaction=None,
        cursors={"m": 0},
    )


def _settings() -> Settings:
    return Settings(
        api_key_pepper="test-pepper",
        database_url="postgresql+asyncpg://test:test@localhost/test",
        llm_crof_api_key="test-crof-key",
        runtime_model_profiles={
            "plap/test": RuntimeModelProfileConfig(
                display_name="Test Model",
                main_model="crof/qwen3.5-9b",
                main_debate_model="crof/qwen3.5-9b",
                reviewer_model="crof/qwen3.5-9b",
                arbitrator_model="crof/qwen3.5-9b",
                reasoning_summarizer_model="crof/qwen3.5-9b",
            )
        },
        sealing_keys=["a" * 43],
    )


def _keyring() -> SealingKeyring:
    return SealingKeyring.from_encoded(["a" * 43])


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
