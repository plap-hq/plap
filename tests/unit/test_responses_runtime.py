from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import partial
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import anyio
import pytest
import svcs
from box import Box
from pydantic import TypeAdapter

from plap.bus import bus
from plap.config import CueBox
from plap.errors import PlapError
from plap.keyring import SealingKeyring
from plap.llms.completions.chat import ChatCompletionDelta, ChatFinishReason, ChatToolCallDelta, ChatUsage, IChatCompletionClient
from plap.llms.retry import RETRY_TOOL_PLACEHOLDER
from plap.plugins.core.loop import UsageLedger, response_request, run_response
from plap.responses.contracts import ResponseCreateRequest, ResponseStreamEvent
from plap.responses.contracts.items import ResponseCompactionItem, ResponseFunctionCallItem, ResponseMessageItem, ResponseReasoningItem
from plap.responses.ingest.models import MAIN_SIDE, Ingested, Message, MessagePatch, Sides, SidesUpdate, ToolCall
from plap.responses.ingest.sealing import open_compaction_payload, open_reasoning_payload
from plap.responses.state import State
from plap.responses.store import PreparedRequest
from plap.responses.streaming import StreamCoordinator

_STREAM_EVENT_ADAPTER = TypeAdapter(ResponseStreamEvent)


def _reload_summary_handlers():
    bus.reset()
    core_module = importlib.import_module("plap.plugins.core.loop")
    summary_module = importlib.import_module("plap.plugins.summary")
    core_module = importlib.reload(core_module)
    importlib.reload(summary_module)
    return core_module.run_response


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


def _side_codes() -> dict[str, int]:
    return {MAIN_SIDE: 0}


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


def _ingested(*, last_side: str | None = None) -> Ingested:
    return Ingested(
        machine={},
        sides=Sides(),
        last_side=last_side,
        last_reasoning_id=None,
        current_compaction_id=None,
    )


class _Config(Box):
    def resolve(self, request: dict[str, object] | None = None, /, **kwargs: object) -> _Config:
        _ = request, kwargs
        return self


def _config() -> _Config:
    return _Config(
        {
            "display_name": "Test Model",
            "model_info": {
                "display_name": "Test Model",
                "description": "test",
                "mode": "responses",
                "input_modalities": ["text"],
                "output_modalities": ["text"],
                "max_input_tokens": 8192,
                "max_output_tokens": 2048,
                "supported_parameters": [],
                "pricing": {"input_per_token": 0.0, "output_per_token": 0.0},
                "provider": "plap",
                "deprecated": False,
            },
            "main": {
                "model": "test-model",
                "max_completion_tokens": None,
                "tokenizer_hf_repo": None,
                "tokenizer_revision": None,
                "tokenizer_trust_remote_code": False,
                "reasoning_effort": None,
                "service_tier": None,
                "public_usage": {
                    "uncached_input_to_output": 0.25,
                    "cached_input_to_output": 0.05,
                    "output_to_output": 1.0,
                },
                "sampling": {
                    "temperature": None,
                    "top_p": None,
                    "top_k": None,
                    "frequency_penalty": None,
                    "presence_penalty": None,
                    "top_logprobs": None,
                },
            },
            "summary": {
                "model": "test-summarizer",
                "max_completion_tokens": 768,
                "tokenizer_hf_repo": None,
                "tokenizer_revision": None,
                "tokenizer_trust_remote_code": False,
                "reasoning_effort": None,
                "service_tier": None,
                "public_usage": {
                    "uncached_input_to_output": 0.25,
                    "cached_input_to_output": 0.05,
                    "output_to_output": 1.0,
                },
                "sampling": {
                    "temperature": None,
                    "top_p": None,
                    "top_k": None,
                    "frequency_penalty": None,
                    "presence_penalty": None,
                    "top_logprobs": None,
                },
            },
            "reasoning_to_output": 1.0,
        },
        frozen_box=True,
    )


def _loaded(config: _Config | None = None) -> object:
    return SimpleNamespace(plap=SimpleNamespace(config=config or _config()))


def _public_usage(**updates: object) -> Box:
    return Box(
        {
            "uncached_input_to_output": 0.25,
            "cached_input_to_output": 0.05,
            "output_to_output": 1.0,
            **updates,
        },
        frozen_box=True,
    )


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


def _svcs(*, client: object | None = None, config: _Config | None = None) -> svcs.Container:
    registry = svcs.Registry()
    registry.register_value(SealingKeyring, _keyring())
    registry.register_value(CueBox, _loaded(config))
    registry.register_value(IChatCompletionClient, client if client is not None else object())
    return svcs.Container(registry)


def _state(
    store: _RecordingStore | None = None,
    channels: _RecordingChannels | None = None,
    *,
    request: ResponseCreateRequest | None = None,
    client: object | None = None,
    config: _Config | None = None,
    ingested: Ingested | None = None,
) -> State:
    actual_store = store or _RecordingStore()
    actual_channels = channels or _RecordingChannels()
    actual_request = request or _request()
    return State.from_ingested(
        ingested=ingested or _ingested(),
        prepared=_prepared(actual_request),
        svcs=_svcs(client=client, config=config),
        coordinator=_coordinator(actual_store, actual_channels, actual_request),
        sealing_keyring=_keyring(),
        side_codes=_side_codes(),
    )


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


@pytest.mark.anyio
async def test_response_request_applies_internal_sampling_config() -> None:
    request = _request()
    config_data = _config().to_dict()
    config_data["main"]["sampling"] = {
        "temperature": None,
        "top_p": None,
        "top_k": {
            "disabled": False,
            "fixed": None,
            "default": 17,
            "min_value": None,
            "max_value": None,
        },
        "frequency_penalty": {
            "disabled": False,
            "fixed": None,
            "default": 0.25,
            "scale": 1.0,
            "offset": 0.0,
            "min_value": None,
            "max_value": None,
        },
        "presence_penalty": {
            "disabled": False,
            "fixed": None,
            "default": -0.5,
            "scale": 1.0,
            "offset": 0.0,
            "min_value": None,
            "max_value": None,
        },
        "top_logprobs": None,
    }
    config = _Config(config_data, frozen_box=True)
    state = _state(request=request, config=config)

    built = await response_request(state=state, config=state.svcs.get(CueBox).plap.config)

    assert built.top_k == 17
    assert built.frequency_penalty == 0.25
    assert built.presence_penalty == -0.5


class _FakeReasoningSummarizer:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def stream(self, *, mode: str, prior_summary: str | None, fragment: str):
        assert mode == "concise"
        _ = prior_summary, fragment, self.kwargs
        yield "summary part"


def test_usage_ledger_returns_none_without_charges() -> None:
    ledger = UsageLedger(budget=None, reasoning_to_output=1.0)

    assert ledger.usage() is None


def test_usage_ledger_cap_respects_budget_and_limit() -> None:
    ledger = UsageLedger(budget=20, reasoning_to_output=1.0)

    assert ledger.cap(_public_usage(output_to_output=2.0), None) == 10
    assert ledger.cap(_public_usage(output_to_output=2.0), 7) == 7
    assert ledger.cap(None, 7) == 7


def test_usage_ledger_scales_single_visible_usage() -> None:
    ledger = UsageLedger(budget=20, reasoning_to_output=1.5)
    usage = _usage(input_tokens=10, output_tokens=12, cached_tokens=1, reasoning_tokens=5)

    ledger.show(_public_usage(), usage)

    response_usage = ledger.usage()
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

    ledger.hide(_public_usage(), hidden_usage)
    ledger.show(_public_usage(), output_usage)

    response_usage = ledger.usage()
    assert response_usage is not None
    assert response_usage.input_tokens == 80
    assert response_usage.input_tokens_details.cached_tokens == 0
    assert response_usage.output_tokens == 40
    assert response_usage.output_tokens_details.reasoning_tokens == 36
    assert response_usage.total_tokens == 120
    assert ledger.remaining() == 60


def test_usage_ledger_hidden_then_visible_keeps_hidden_debit() -> None:
    ledger = UsageLedger(budget=100, reasoning_to_output=1.0)
    hidden_usage = _usage(input_tokens=20, output_tokens=9, reasoning_tokens=3)
    output_usage = _usage(input_tokens=4, output_tokens=2, cached_tokens=1)

    ledger.hide(_public_usage(), hidden_usage)
    assert ledger.remaining() == 86

    ledger.show(_public_usage(), output_usage)

    response_usage = ledger.usage()
    assert response_usage is not None
    assert response_usage.input_tokens == 20
    assert response_usage.output_tokens == 16
    assert response_usage.output_tokens_details.reasoning_tokens == 14
    assert response_usage.total_tokens == 36
    assert ledger.remaining() == 84


def test_usage_ledger_clamps_output_tokens_to_visible_floor_for_discounted_visible_actor() -> None:
    cheap_output = _public_usage(output_to_output=0.5)
    usage = _usage(input_tokens=10, output_tokens=20, reasoning_tokens=5)
    ledger = UsageLedger(budget=100, reasoning_to_output=1.0)

    ledger.show(cheap_output, usage)

    response_usage = ledger.usage()
    assert response_usage is not None
    assert response_usage.input_tokens == 10
    assert response_usage.output_tokens == 15
    assert response_usage.output_tokens_details.reasoning_tokens == 0
    assert response_usage.total_tokens == 25
    assert ledger.remaining() == 90


async def test_run_response_completes_simple_turn_without_midstream_flushes() -> None:
    channels = _RecordingChannels()
    store = _RecordingStore()
    request = _request()
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
    state = _state(
        store,
        channels,
        request=request,
        client=client,
        ingested=Ingested(
            machine={},
            sides=Sides(messages={MAIN_SIDE: [Message(role="user", content="hello")]}),
            last_side=MAIN_SIDE,
            last_reasoning_id=None,
            current_compaction_id=None,
        ),
    )

    await run_response(state=state)

    assert store.begin_calls == 1
    assert store.cancel_calls == 0
    assert store.fail_calls == 0
    assert store.replace_calls == 0
    assert store.finish_calls == 1
    response = state.coordinator.current_response()
    assert response.status == "completed"
    assert len(response.output) == 1
    assert isinstance(response.output[0], ResponseMessageItem)
    assert response.output[0].content[0].text == "hello"
    assert response.usage is not None
    assert response.usage.input_tokens == 7


def test_state_from_ingested_preserves_last_side() -> None:
    state = _state(ingested=_ingested(last_side="reviewer"))

    assert state.last_side == "reviewer"


async def test_run_response_cancellation_before_created_is_noop() -> None:
    channels = _RecordingChannels()
    store = _RecordingStore()
    state = _state(store, channels, client=object())

    with anyio.CancelScope() as cancel_scope:
        cancel_scope.cancel()
        await run_response(state=state)

    assert store.begin_calls == 0
    assert store.cancel_calls == 0
    assert store.fail_calls == 0
    assert _published_event_types(channels) == []


async def test_run_response_cancellation_after_created_persists_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    channels = _RecordingChannels()
    store = _RecordingStore()
    state = _state(store, channels, client=object(), ingested=_ingested(last_side=MAIN_SIDE))
    body_started = anyio.Event()

    async def _block(**kwargs):
        _ = kwargs
        body_started.set()
        await anyio.sleep(10)

    monkeypatch.setattr("plap.plugins.core.loop.stream_response", _block)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(
            partial(
                run_response,
                state=state,
            ),
        )
        await body_started.wait()
        task_group.cancel_scope.cancel()

    assert store.begin_calls == 1
    assert store.cancel_calls == 1
    assert store.fail_calls == 0
    assert state.coordinator.current_response().status == "cancelled"
    assert _published_event_types(channels) == [
        "response.created",
        "response.in_progress",
        "response.completed",
    ]


@pytest.mark.parametrize(
    ("last_side", "status_code", "code"),
    [
        (None, 500, "internal_error"),
        ("reviewer", 500, "internal_error"),
    ],
)
async def test_run_response_fails_when_core_has_no_default_handler_for_last_side(
    last_side: str | None,
    status_code: int,
    code: str,
) -> None:
    channels = _RecordingChannels()
    store = _RecordingStore()
    state = _state(store, channels, client=object(), ingested=_ingested(last_side=last_side))

    with pytest.raises(PlapError) as exc_info:
        await run_response(state=state)

    assert exc_info.value.public is not None
    assert exc_info.value.public.status_code == status_code
    assert exc_info.value.public.code == code
    assert store.begin_calls == 1
    assert store.fail_calls == 1
    assert state.coordinator.current_response().status == "failed"
    assert _published_event_types(channels) == [
        "response.created",
        "response.in_progress",
        "error",
    ]


async def test_run_response_retry_persists_hidden_history_and_anchors_usage_to_first_attempt() -> None:
    channels = _RecordingChannels()
    store = _RecordingStore()
    request = _request(
        max_output_tokens=20,
        tools=[{"type": "function", "name": "good", "parameters": {"type": "object"}}],
    )
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
    state = _state(store, channels, request=request, client=client, ingested=_ingested(last_side=MAIN_SIDE))

    await run_response(state=state)

    assert len(client.requests) == 2
    assert client.requests[0].max_completion_tokens == 20
    assert client.requests[1].max_completion_tokens == 6

    response = state.coordinator.current_response()
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
    state = _state(store, channels, request=request, client=client, ingested=_ingested(last_side=MAIN_SIDE))
    run_response_with_summary = _reload_summary_handlers()
    monkeypatch.setattr("plap.plugins.summary.ChatReasoningSummarizer", _FakeReasoningSummarizer)

    await run_response_with_summary(state=state)

    assert store.replace_calls == 2
    response = state.coordinator.current_response()
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
    client = _StubChatClient(
        [
            _delta(
                tool_call_delta=ChatToolCallDelta(index=0, id="call_bad", name="bad", arguments_delta="{}"),
                finish_reason=ChatFinishReason.TOOL_CALLS,
                usage=_usage(input_tokens=20, output_tokens=9),
            )
        ]
    )
    state = _state(store, channels, request=request, client=client, ingested=_ingested(last_side=MAIN_SIDE))

    await run_response(state=state)

    response = state.coordinator.current_response()
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

    item = _last_output_item(state.coordinator)
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

    item = _last_output_item(state.coordinator)
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

    response = state.coordinator.current_response()
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

    response = state.coordinator.current_response()
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

    response = state.coordinator.current_response()
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
