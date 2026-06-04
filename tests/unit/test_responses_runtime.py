from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import cast
from uuid import uuid4

import anyio
import pytest
from pydantic import TypeAdapter

from plap.keyring import SealingKeyring
from plap.llms.completions.chat import ChatCompletionDelta, ChatFinishReason, ChatToolCallDelta, ChatUsage
from plap.llms.retry import RETRY_TOOL_PLACEHOLDER
from plap.responses.contracts import ResponseCreateRequest, ResponseStreamEvent
from plap.responses.contracts.items import ResponseCompactionItem, ResponseFunctionCallItem, ResponseMessageItem, ResponseReasoningItem
from plap.responses.ingest.models import MAIN_SIDE, Ingested, Message, MessagePatch, Sides, SidesUpdate, ToolCall
from plap.responses.ingest.sealing import open_compaction_payload, open_reasoning_payload
from plap.responses.runner import State, UsageLedger
from plap.responses.runtime import run_response
from plap.responses.store import PreparedRequest
from plap.responses.streaming import StreamCoordinator
from plap.settings import PublicUsageConfig, Settings
from plap.tools import StaticToolCallPolicyResolver, StaticToolPolicyResolver

_STREAM_EVENT_ADAPTER = TypeAdapter(ResponseStreamEvent)


class _RecordingChannels:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, object]]] = []

    async def wait_published(self, data: dict[str, object], channels: str | Sequence[str]) -> None:
        channel_names = [channels] if isinstance(channels, str) else list(channels)
        for channel_name in channel_names:
            self.published.append((channel_name, data))


class _RecordingStore:
    def __init__(self) -> None:
        self.begin_calls = 0
        self.append_calls = 0
        self.replace_calls = 0
        self.finish_calls = 0
        self.cancel_calls = 0
        self.fail_calls = 0

    async def begin_response(self, prepared: PreparedRequest, response) -> None:
        _ = prepared, response
        self.begin_calls += 1

    async def append_output_item(self, prepared: PreparedRequest, response_id: str, output_index: int, item: object) -> None:
        _ = prepared, response_id, output_index, item
        self.append_calls += 1

    async def replace_output_item(self, prepared: PreparedRequest, response_id: str, output_index: int, item: object) -> None:
        _ = prepared, response_id, output_index, item
        self.replace_calls += 1

    async def finish_response(self, prepared: PreparedRequest, response) -> None:
        _ = prepared, response
        self.finish_calls += 1

    async def cancel_response(self, prepared: PreparedRequest, response) -> bool:
        _ = prepared, response
        self.cancel_calls += 1
        return True

    async def fail_response(self, prepared: PreparedRequest, response_id: str) -> bool:
        _ = prepared, response_id
        self.fail_calls += 1
        return True


def _keyring() -> SealingKeyring:
    return SealingKeyring(roots=(b"i" * 32,))


def _request(**updates: object) -> ResponseCreateRequest:
    return ResponseCreateRequest(model="plap-ai/wisp-mini", input="hello", **updates)


def _prepared(request: ResponseCreateRequest | None = None) -> PreparedRequest:
    actual_request = request or _request()
    return PreparedRequest(
        scope_id=uuid4(),
        response_request=actual_request,
        execution_request=actual_request,
        current_input_items=[],
        stored_input_items=[],
        parent_response_id=None,
        conversation_id=None,
        persist_response=True,
    )


def _ingested() -> Ingested:
    return Ingested(
        machine={},
        sides=Sides(),
        last_side=None,
        last_reasoning_id=None,
        current_compaction_id=None,
    )


def _settings() -> Settings:
    return Settings(api_key_pepper="pepper", database_url="postgres://example", sealing_keys=["key"])


def _coordinator(
    store: _RecordingStore,
    channels: _RecordingChannels,
    request: ResponseCreateRequest | None = None,
) -> StreamCoordinator:
    actual_request = request or _request()
    return StreamCoordinator(
        request=actual_request,
        channels=channels,
        prepared=_prepared(actual_request),
        response_store=store,
        sealing_keyring=_keyring(),
    )


def _published_event_types(channels: _RecordingChannels) -> list[str]:
    return [_STREAM_EVENT_ADAPTER.validate_python(payload).type for _, payload in channels.published]


def _last_output_item(coordinator: StreamCoordinator):
    return coordinator.current_response().output[-1]


def _state(store: _RecordingStore | None = None, channels: _RecordingChannels | None = None) -> State:
    actual_store = store or _RecordingStore()
    actual_channels = channels or _RecordingChannels()
    return State.from_ingested(_coordinator(actual_store, actual_channels), _keyring(), _ingested())


def _usage(
    *,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> ChatUsage:
    return ChatUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _delta(
    *,
    content_delta: str | None = None,
    reasoning_delta: str | None = None,
    tool_call_delta: ChatToolCallDelta | None = None,
    finish_reason: ChatFinishReason | None = None,
    usage: ChatUsage | None = None,
) -> ChatCompletionDelta:
    return ChatCompletionDelta(
        id="cmpl_test",
        model="test-model",
        created_at=None,
        choice_index=0,
        content_delta=content_delta,
        reasoning_delta=reasoning_delta,
        tool_call_delta=tool_call_delta,
        finish_reason=finish_reason,
        usage=usage,
        service_tier="default",
    )


class _StubChatClient:
    def __init__(self, *attempts: list[ChatCompletionDelta]) -> None:
        self._attempts = list(attempts)
        self.requests: list[object] = []

    async def complete(self, request):  # pragma: no cover - unused protocol method
        _ = request
        raise NotImplementedError

    def stream(self, request):
        self.requests.append(request)
        index = len(self.requests) - 1
        if index >= len(self._attempts):  # pragma: no cover
            raise AssertionError("unexpected extra attempt")
        deltas = list(self._attempts[index])

        async def _iterator():
            for delta in deltas:
                yield delta

        return _iterator()


class _FakeReasoningSummarizer:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def stream(self, *, mode: str, prior_summary: str | None, fragment: str):
        assert mode == "concise"
        _ = prior_summary, fragment, self.kwargs
        yield "summary part"


def test_usage_ledger_returns_none_without_input_anchor() -> None:
    ledger = UsageLedger(budget=None, reasoning_to_output=1.0)
    ledger.record_output(PublicUsageConfig(), _usage(input_tokens=10, output_tokens=12, cached_tokens=1, reasoning_tokens=5))

    assert ledger.to_response_usage() is None


def test_usage_ledger_scales_single_visible_usage() -> None:
    ledger = UsageLedger(budget=20, reasoning_to_output=1.5)
    usage = _usage(input_tokens=10, output_tokens=12, cached_tokens=1, reasoning_tokens=5)

    ledger.set_input_anchor(usage)
    ledger.record_output(PublicUsageConfig(), usage)

    response_usage = ledger.to_response_usage()
    assert response_usage is not None
    assert response_usage.input_tokens == 10
    assert response_usage.input_tokens_details.cached_tokens == 1
    assert response_usage.output_tokens == 15
    assert response_usage.output_tokens_details.reasoning_tokens == 8
    assert response_usage.total_tokens == 25
    assert ledger.remaining() == 5


def test_usage_ledger_hidden_usage_is_squashed_into_reasoning_tokens() -> None:
    ledger = UsageLedger(budget=100, reasoning_to_output=1.0)
    hidden_usage = _usage(input_tokens=80, output_tokens=15)
    output_usage = _usage(input_tokens=10, output_tokens=5, cached_tokens=2, reasoning_tokens=1)

    ledger.record_hidden(PublicUsageConfig(), hidden_usage)
    ledger.set_input_anchor(output_usage)
    ledger.record_output(PublicUsageConfig(), output_usage)

    response_usage = ledger.to_response_usage()
    assert response_usage is not None
    assert response_usage.input_tokens == 10
    assert response_usage.input_tokens_details.cached_tokens == 2
    assert response_usage.output_tokens == 40
    assert response_usage.output_tokens_details.reasoning_tokens == 36
    assert response_usage.total_tokens == 50
    assert ledger.remaining() == 60


def test_usage_ledger_promoted_hidden_output_keeps_hidden_debit_and_visible_floor() -> None:
    ledger = UsageLedger(budget=100, reasoning_to_output=1.0)
    hidden_usage = _usage(input_tokens=20, output_tokens=9, reasoning_tokens=3)
    input_anchor = _usage(input_tokens=4, output_tokens=0, cached_tokens=1)

    hidden_index = ledger.record_hidden(PublicUsageConfig(), hidden_usage)
    assert hidden_index == 0
    assert ledger.remaining() == 86

    ledger.set_input_anchor(input_anchor)
    ledger.promote_hidden_to_output(hidden_index)

    response_usage = ledger.to_response_usage()
    assert response_usage is not None
    assert response_usage.input_tokens == 4
    assert response_usage.input_tokens_details.cached_tokens == 1
    assert response_usage.output_tokens == 14
    assert response_usage.output_tokens_details.reasoning_tokens == 8
    assert response_usage.total_tokens == 18
    assert ledger.remaining() == 86


def test_usage_ledger_clamps_output_tokens_to_visible_floor_for_discounted_visible_actor() -> None:
    cheap_output = PublicUsageConfig(output_to_output=0.5)
    usage = _usage(input_tokens=10, output_tokens=20, reasoning_tokens=5)
    ledger = UsageLedger(budget=100, reasoning_to_output=1.0)

    ledger.set_input_anchor(usage)
    ledger.record_output(cheap_output, usage)

    response_usage = ledger.to_response_usage()
    assert response_usage is not None
    assert response_usage.input_tokens == 10
    assert response_usage.output_tokens == 15
    assert response_usage.output_tokens_details.reasoning_tokens == 0
    assert response_usage.total_tokens == 25
    assert ledger.remaining() == 90


def test_usage_ledger_rejects_duplicate_input_anchor() -> None:
    ledger = UsageLedger(budget=None, reasoning_to_output=1.0)
    usage = _usage(input_tokens=10, output_tokens=1)

    ledger.set_input_anchor(usage)

    with pytest.raises(ValueError, match="input usage anchor is already set"):
        ledger.set_input_anchor(usage)


def test_usage_ledger_rejects_duplicate_hidden_output_promotion() -> None:
    ledger = UsageLedger(budget=None, reasoning_to_output=1.0)
    hidden_index = ledger.record_hidden(PublicUsageConfig(), _usage(input_tokens=10, output_tokens=4))
    assert hidden_index == 0

    ledger.promote_hidden_to_output(hidden_index)

    with pytest.raises(ValueError, match="hidden usage is already visible output"):
        ledger.promote_hidden_to_output(hidden_index)


async def test_run_response_completes_simple_turn_without_midstream_flushes() -> None:
    channels = _RecordingChannels()
    store = _RecordingStore()
    request = _request()
    coordinator = _coordinator(store, channels, request)
    client = _StubChatClient(
        [
            _delta(content_delta="hel"),
            _delta(
                content_delta="lo",
                finish_reason=ChatFinishReason.STOP,
                usage=_usage(input_tokens=7, output_tokens=3),
            ),
        ]
    )

    await run_response(
        prepared=_prepared(request),
        ingested=Ingested(
            machine={},
            sides=Sides(messages={MAIN_SIDE: [Message(role="user", content="hello")]}),
            last_side=MAIN_SIDE,
            last_reasoning_id=None,
            current_compaction_id=None,
        ),
        coordinator=coordinator,
        sealing_keyring=_keyring(),
        settings=_settings(),
        chat_completion_client=client,
        tool_policy_resolver=StaticToolPolicyResolver(),
        tool_call_policy_resolver=StaticToolCallPolicyResolver(),
        mcp_tool_providers=(),
    )

    assert store.begin_calls == 1
    assert store.cancel_calls == 0
    assert store.fail_calls == 0
    assert store.replace_calls == 0
    assert store.finish_calls == 1
    response = coordinator.current_response()
    assert response.status == "completed"
    assert len(response.output) == 1
    assert isinstance(response.output[0], ResponseMessageItem)
    assert response.output[0].content[0].text == "hello"
    assert response.usage is not None
    assert response.usage.input_tokens == 7


async def test_run_response_prepends_profile_prompt_before_request_instructions() -> None:
    channels = _RecordingChannels()
    store = _RecordingStore()
    request = _request(instructions="Follow the caller instructions.")
    coordinator = _coordinator(store, channels, request)
    client = _StubChatClient(
        [
            _delta(
                content_delta="hello",
                finish_reason=ChatFinishReason.STOP,
                usage=_usage(input_tokens=7, output_tokens=3),
            ),
        ]
    )

    await run_response(
        prepared=_prepared(request),
        ingested=Ingested(
            machine={},
            sides=Sides(messages={MAIN_SIDE: [Message(role="user", content="hello")]}),
            last_side=MAIN_SIDE,
            last_reasoning_id=None,
            current_compaction_id=None,
        ),
        coordinator=coordinator,
        sealing_keyring=_keyring(),
        settings=_settings(),
        chat_completion_client=client,
        tool_policy_resolver=StaticToolPolicyResolver(),
        tool_call_policy_resolver=StaticToolCallPolicyResolver(),
        mcp_tool_providers=(),
    )

    sent_messages = client.requests[0].messages
    assert sent_messages[0].role == "developer"
    assert sent_messages[0].content == "You are Wisp Mini, an AI assistant."
    assert sent_messages[1].role == "developer"
    assert sent_messages[1].content == "Follow the caller instructions."
    assert sent_messages[2].role == "user"
    assert sent_messages[2].content == "hello"


async def test_run_response_cancellation_before_created_is_noop() -> None:
    channels = _RecordingChannels()
    store = _RecordingStore()
    coordinator = _coordinator(store, channels)

    with anyio.CancelScope() as cancel_scope:
        cancel_scope.cancel()
        await run_response(
            prepared=_prepared(),
            ingested=_ingested(),
            coordinator=coordinator,
            sealing_keyring=_keyring(),
            settings=_settings(),
            chat_completion_client=object(),
            tool_policy_resolver=StaticToolPolicyResolver(),
            tool_call_policy_resolver=StaticToolCallPolicyResolver(),
            mcp_tool_providers=(),
        )

    assert store.begin_calls == 0
    assert store.cancel_calls == 0
    assert store.fail_calls == 0
    assert _published_event_types(channels) == []


async def test_run_response_cancellation_after_created_persists_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    channels = _RecordingChannels()
    store = _RecordingStore()
    coordinator = _coordinator(store, channels)
    body_started = anyio.Event()

    async def _block(**kwargs) -> None:
        _ = kwargs
        body_started.set()
        await anyio.sleep(10)

    monkeypatch.setattr("plap.responses.runtime.execute", _block)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(
            partial(
                run_response,
                prepared=_prepared(),
                ingested=_ingested(),
                coordinator=coordinator,
                sealing_keyring=_keyring(),
                settings=_settings(),
                chat_completion_client=object(),
                tool_policy_resolver=StaticToolPolicyResolver(),
                tool_call_policy_resolver=StaticToolCallPolicyResolver(),
                mcp_tool_providers=(),
            ),
        )
        await body_started.wait()
        task_group.cancel_scope.cancel()

    assert store.begin_calls == 1
    assert store.cancel_calls == 1
    assert store.fail_calls == 0
    assert coordinator.current_response().status == "cancelled"
    assert _published_event_types(channels) == [
        "response.created",
        "response.in_progress",
        "response.completed",
    ]


async def test_run_response_retry_persists_hidden_history_and_anchors_usage_to_first_attempt() -> None:
    channels = _RecordingChannels()
    store = _RecordingStore()
    request = _request(
        max_output_tokens=20,
        tools=[{"type": "function", "name": "good", "parameters": {"type": "object"}}],
    )
    coordinator = _coordinator(store, channels, request)
    client = _StubChatClient(
        [
            _delta(
                tool_call_delta=ChatToolCallDelta(index=0, id="call_bad", name="bad", arguments_delta="{}"),
                finish_reason=ChatFinishReason.TOOL_CALLS,
                usage=_usage(input_tokens=20, output_tokens=9),
            )
        ],
        [
            _delta(
                content_delta="fixed",
                finish_reason=ChatFinishReason.STOP,
                usage=_usage(input_tokens=5, output_tokens=2),
            )
        ],
    )

    await run_response(
        prepared=_prepared(request),
        ingested=_ingested(),
        coordinator=coordinator,
        sealing_keyring=_keyring(),
        settings=_settings(),
        chat_completion_client=client,
        tool_policy_resolver=StaticToolPolicyResolver(),
        tool_call_policy_resolver=StaticToolCallPolicyResolver(),
        mcp_tool_providers=(),
    )

    assert len(client.requests) == 2
    assert client.requests[0].max_completion_tokens == 20
    assert client.requests[1].max_completion_tokens == 6

    response = coordinator.current_response()
    assert response.status == "completed"
    assert response.usage is not None
    assert response.usage.input_tokens == 20
    assert response.usage.total_tokens == 36
    assert isinstance(response.output[0], ResponseReasoningItem)
    payload = open_reasoning_payload(response.output[0].encrypted_content, keyring=_keyring())
    assert payload.sides.main[0].is_assistant()
    assert payload.sides.main[0].tool_calls[0].name == "bad"
    assert payload.sides.main[1].is_tool()
    assert payload.sides.main[1].content == RETRY_TOOL_PLACEHOLDER
    assert payload.sides.main[2].role == "user"
    assert "undeclared tool" in cast(str, payload.sides.main[2].content)
    assert isinstance(response.output[1], ResponseMessageItem)
    assert response.output[1].content[0].text == "fixed"


async def test_run_response_summary_flushes_on_summary_done(monkeypatch: pytest.MonkeyPatch) -> None:
    channels = _RecordingChannels()
    store = _RecordingStore()
    request = _request(reasoning={"summary": "concise"})
    coordinator = _coordinator(store, channels, request)
    client = _StubChatClient(
        [
            _delta(content_delta="he", reasoning_delta="private "),
            _delta(
                content_delta="llo",
                reasoning_delta="reasoning",
                finish_reason=ChatFinishReason.STOP,
                usage=_usage(input_tokens=9, output_tokens=4, reasoning_tokens=2),
            ),
        ]
    )
    monkeypatch.setattr("plap.responses.runner.ChatReasoningSummarizer", _FakeReasoningSummarizer)

    await run_response(
        prepared=_prepared(request),
        ingested=_ingested(),
        coordinator=coordinator,
        sealing_keyring=_keyring(),
        settings=_settings(),
        chat_completion_client=client,
        tool_policy_resolver=StaticToolPolicyResolver(),
        tool_call_policy_resolver=StaticToolCallPolicyResolver(),
        mcp_tool_providers=(),
    )

    assert store.replace_calls == 2
    response = coordinator.current_response()
    assert response.status == "completed"
    assert isinstance(response.output[0], ResponseReasoningItem)
    assert response.output[0].summary[0].text == "summary part"
    event_types = _published_event_types(channels)
    assert event_types.index("response.output_item.added") < event_types.index("response.reasoning_summary_part.added")
    assert "response.reasoning_summary_part.done" in event_types


async def test_run_response_budget_exhaustion_marks_incomplete() -> None:
    channels = _RecordingChannels()
    store = _RecordingStore()
    request = _request(
        max_output_tokens=10,
        tools=[{"type": "function", "name": "good", "parameters": {"type": "object"}}],
    )
    coordinator = _coordinator(store, channels, request)
    client = _StubChatClient(
        [
            _delta(
                tool_call_delta=ChatToolCallDelta(index=0, id="call_bad", name="bad", arguments_delta="{}"),
                finish_reason=ChatFinishReason.TOOL_CALLS,
                usage=_usage(input_tokens=20, output_tokens=9),
            )
        ]
    )

    await run_response(
        prepared=_prepared(request),
        ingested=_ingested(),
        coordinator=coordinator,
        sealing_keyring=_keyring(),
        settings=_settings(),
        chat_completion_client=client,
        tool_policy_resolver=StaticToolPolicyResolver(),
        tool_call_policy_resolver=StaticToolCallPolicyResolver(),
        mcp_tool_providers=(),
    )

    response = coordinator.current_response()
    assert response.status == "incomplete"
    assert response.incomplete_details is not None
    assert response.incomplete_details.reason == "max_output_tokens"
    assert response.usage is not None
    assert response.usage.input_tokens == 20
    assert len(response.output) == 1
    assert isinstance(response.output[0], ResponseReasoningItem)


async def test_state_flush_persists_stubbed_open_tail_calls() -> None:
    store = _RecordingStore()
    channels = _RecordingChannels()
    state = _state(store, channels)

    state.main = [
        Message(
            role="assistant",
            content="thinking",
            tool_calls=[ToolCall(id="call_main", name="search", arguments="{}")],
        )
    ]

    await state.flush()

    item = _last_output_item(state._coordinator)
    assert isinstance(item, ResponseReasoningItem)
    payload = open_reasoning_payload(item.encrypted_content, keyring=_keyring())
    assert payload.sides.main[0].is_assistant()
    assert payload.sides.main[0].tool_calls[0].id == "call_main"
    assert payload.sides.main[1].is_tool()
    assert payload.sides.main[1].tool_call_id == "call_main"
    assert isinstance(payload.sides.main[1].content, str)
    assert payload.sides.main[1].content


async def test_state_flush_preserves_explicit_empty_side() -> None:
    store = _RecordingStore()
    channels = _RecordingChannels()
    state = _state(store, channels)

    state.sides["reviewer"] = []

    await state.flush()

    item = _last_output_item(state._coordinator)
    assert isinstance(item, ResponseReasoningItem)
    payload = open_reasoning_payload(item.encrypted_content, keyring=_keyring())
    assert "reviewer" in payload.sides.patches
    assert payload.sides.patches["reviewer"].shape is None
    assert payload.sides.patches["reviewer"].patch == []


async def test_state_finalize_uses_message_patch_and_emits_visible_items() -> None:
    store = _RecordingStore()
    channels = _RecordingChannels()
    state = _state(store, channels)

    state.main = [
        Message(
            role="assistant",
            content="hello",
            reasoning_content="hidden",
            tool_calls=[ToolCall(id="call_main", name="search", arguments="{}")],
        )
    ]

    await state.finalize()

    response = state._coordinator.current_response()
    assert isinstance(response.output[0], ResponseReasoningItem)
    reasoning_payload = open_reasoning_payload(response.output[0].encrypted_content, keyring=_keyring())
    assert isinstance(reasoning_payload.sides.main[0], MessagePatch)
    assert reasoning_payload.sides.main[0].tool_calls is not None
    assert reasoning_payload.sides.main[0].tool_calls[0].id == "call_main"
    assert reasoning_payload.sides.main[0].reasoning_content == "hidden"
    assert isinstance(response.output[1], ResponseMessageItem)
    assert response.output[1].content[0].text == "hello"
    assert isinstance(response.output[2], ResponseFunctionCallItem)
    assert response.output[2].call_id.startswith("call_")


async def test_state_finalize_keeps_closed_assistant_with_user_tail_hidden() -> None:
    store = _RecordingStore()
    channels = _RecordingChannels()
    state = _state(store, channels)

    state.main = [
        Message(role="assistant", content="hidden assistant"),
        Message(role="user", content="tail"),
    ]

    await state.finalize()

    response = state._coordinator.current_response()
    assert len(response.output) == 1
    assert isinstance(response.output[0], ResponseReasoningItem)
    payload = open_reasoning_payload(response.output[0].encrypted_content, keyring=_keyring())
    assert isinstance(payload.sides.main[0], Message)
    assert payload.sides.main[0].content == "hidden assistant"
    assert payload.sides.main[1].role == "user"
    assert payload.sides.main[1].content == "tail"


async def test_state_compaction_finishes_empty_reasoning_then_rebases() -> None:
    store = _RecordingStore()
    channels = _RecordingChannels()
    state = _state(store, channels)

    state.main = [Message(role="assistant", content="hello")]
    await state.flush()

    state.machine = state.machine.model_copy(update={"active": ["reviewer"]})
    state.sides["reviewer"] = [Message(role="assistant", content="review")]
    state.main = [Message(role="assistant", content="post-compaction")]

    await state.compaction(created_by="runtime")

    response = state._coordinator.current_response()
    assert isinstance(response.output[0], ResponseReasoningItem)
    reasoning_payload = open_reasoning_payload(response.output[0].encrypted_content, keyring=_keyring())
    assert reasoning_payload.machine == []
    assert reasoning_payload.sides == SidesUpdate()
    assert isinstance(response.output[1], ResponseCompactionItem)
    compaction_payload = open_compaction_payload(response.output[1].encrypted_content, keyring=_keyring())
    assert compaction_payload.machine == {"active": ["reviewer"]}
    assert compaction_payload.sides[MAIN_SIDE][-1].content == "post-compaction"
    assert compaction_payload.sides["reviewer"][0].content == "review"
    assert state.machine.to_primitive() == {"active": ["reviewer"]}
    assert state.main == []
