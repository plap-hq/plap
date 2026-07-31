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
from plap.keyring import SealingKeyring
from plap.llms.completions.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatFinishReason,
    ChatToolCallDelta,
    ChatUsage,
    IChatCompletionClient,
)
from plap.llms.retry import RETRY_TOOL_PLACEHOLDER
from plap.plugins.core.budget import ResponseBudget, ResponseBudgetExhaustedError, budgeted
from plap.plugins.core.loop import response_request, run_response
from plap.responses.contracts import ResponseCreateRequest, ResponseStreamEvent
from plap.responses.contracts.items import ResponseFunctionCallItem, ResponseMessageItem, ResponseReasoningItem
from plap.responses.ingest.models import (
    CompactedMainTail,
    HiddenMainTail,
    Ingested,
    Message,
    MessagePatch,
    PublicMainTail,
    ReasoningCheckpoint,
    ReasoningPatch,
    Threads,
    ToolCall,
)
from plap.responses.ingest.sealing import open_call_id, open_reasoning_payload
from plap.responses.state import INTERRUPTED_TOOL_OUTPUT, State
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


def _thread_codes() -> dict[str, int]:
    return {"main": 0, "defender": 1024, "reviewer": 1025}


def _request(**updates: object) -> ResponseCreateRequest:
    return ResponseCreateRequest(model="plap-ai/mote", input="hello", **updates)


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


def _ingested(*, active: set[str] | None = None) -> Ingested:
    return Ingested(
        memory={},
        threads=Threads(active={"main"} if active is None else active),
        main_tail=None,
        last_reasoning_id=None,
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
                "output_equivalence": {
                    "uncached_input_to_output": 0.25,
                    "cached_input_to_output": 0.05,
                    "output_to_output": 1.0,
                },
                "sampling": {
                    "temperature": None,
                    "top_p": None,
                    "min_p": None,
                    "top_k": None,
                    "frequency_penalty": None,
                    "presence_penalty": None,
                    "repetition_penalty": None,
                    "seed": None,
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
                "output_equivalence": {
                    "uncached_input_to_output": 0.25,
                    "cached_input_to_output": 0.05,
                    "output_to_output": 1.0,
                },
                "sampling": {
                    "temperature": None,
                    "top_p": None,
                    "min_p": None,
                    "top_k": None,
                    "frequency_penalty": None,
                    "presence_penalty": None,
                    "repetition_penalty": None,
                    "seed": None,
                    "top_logprobs": None,
                },
            },
            "reasoning_to_output": 1.0,
        },
        frozen_box=True,
    )


def _loaded(config: _Config | None = None) -> object:
    return SimpleNamespace(plap=SimpleNamespace(config=config or _config()))


def _output_equivalence(**updates: object) -> Box:
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
        thread_codes=_thread_codes(),
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
    config_data["main"]["max_completion_tokens"] = 321
    config_data["main"]["sampling"] = {
        "temperature": None,
        "top_p": None,
        "min_p": {
            "disabled": False,
            "fixed": None,
            "default": 0.15,
            "scale": 1.0,
            "offset": 0.0,
            "min_value": None,
            "max_value": None,
        },
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
        "repetition_penalty": {
            "disabled": False,
            "fixed": None,
            "default": 1.1,
            "scale": 1.0,
            "offset": 0.0,
            "min_value": None,
            "max_value": None,
        },
        "seed": {
            "disabled": False,
            "fixed": None,
            "default": 11,
            "min_value": None,
            "max_value": None,
        },
        "top_logprobs": None,
    }
    config = _Config(config_data, frozen_box=True)
    state = _state(request=request, config=config)

    built = await response_request(state=state, config=state.svcs.get(CueBox).plap.config)

    assert built.max_completion_tokens == 321
    assert built.min_p == 0.15
    assert built.top_k == 17
    assert built.frequency_penalty == 0.25
    assert built.presence_penalty == -0.5
    assert built.repetition_penalty == 1.1
    assert built.seed == 11


class _FakeReasoningSummarizer:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def stream(self, *, mode: str, prior_summary: str | None, fragment: str):
        assert mode == "concise"
        _ = prior_summary, fragment, self.kwargs
        yield "summary part"


async def _record_usage(
    budget: ResponseBudget,
    field: object,
    usage: ChatUsage,
    *,
    max_completion_tokens: int | None = None,
) -> ChatCompletionRequest:
    raw_client = _StubChatClient([_delta(finish_reason=ChatFinishReason.STOP, usage=usage)])
    client = budgeted(raw_client, budget, field)
    request = ChatCompletionRequest(model="test-model", messages=[], max_completion_tokens=max_completion_tokens)
    async for _delta_item in client.stream(request):
        pass
    return cast(ChatCompletionRequest, raw_client.requests[0])


def test_response_budget_returns_none_without_charges() -> None:
    budget = ResponseBudget(_config(), None)

    assert budget.finish() is None


async def test_response_budget_caps_completion_requests() -> None:
    config_data = _config().to_dict()
    config_data["main"]["output_equivalence"] = _output_equivalence(output_to_output=2.0).to_dict()
    config = _Config(config_data, frozen_box=True)

    unlimited_request = await _record_usage(
        ResponseBudget(config, 20),
        config.main,
        _usage(input_tokens=1, output_tokens=1),
    )
    limited_request = await _record_usage(
        ResponseBudget(config, 20),
        config.main,
        _usage(input_tokens=1, output_tokens=1),
        max_completion_tokens=7,
    )

    assert unlimited_request.max_completion_tokens == 10
    assert limited_request.max_completion_tokens == 7


async def test_response_budget_scales_single_output_usage() -> None:
    config_data = _config().to_dict()
    config_data["reasoning_to_output"] = 1.5
    config = _Config(config_data, frozen_box=True)
    budget = ResponseBudget(config, 20)
    usage = _usage(input_tokens=10, output_tokens=12, cached_tokens=1, reasoning_tokens=5)

    await _record_usage(budget, config.main, usage)
    response_usage = budget.finish(usage)
    assert response_usage is not None
    assert response_usage.input_tokens == 10
    assert response_usage.input_tokens_details.cached_tokens == 1
    assert response_usage.output_tokens == 15
    assert response_usage.output_tokens_details.reasoning_tokens == 8
    assert response_usage.total_tokens == 25
    assert budget.remaining == 5


async def test_response_budget_internal_usage_is_squashed_into_reasoning_tokens() -> None:
    config = _config()
    budget = ResponseBudget(config, 100)
    hidden_usage = _usage(input_tokens=80, output_tokens=15)
    output_usage = _usage(input_tokens=10, output_tokens=5, cached_tokens=2, reasoning_tokens=1)

    await _record_usage(budget, config.main, hidden_usage)
    await _record_usage(budget, config.main, output_usage)
    response_usage = budget.finish(output_usage)
    assert response_usage is not None
    assert response_usage.input_tokens == 80
    assert response_usage.input_tokens_details.cached_tokens == 0
    assert response_usage.output_tokens == 40
    assert response_usage.output_tokens_details.reasoning_tokens == 36
    assert response_usage.total_tokens == 120
    assert budget.remaining == 60


async def test_response_budget_uses_main_input_when_internal_summary_finishes_first() -> None:
    config = _config()
    budget = ResponseBudget(config, 100)
    summary_usage = _usage(input_tokens=80, output_tokens=2)
    main_usage = _usage(input_tokens=10, output_tokens=5, cached_tokens=3)

    await _record_usage(budget, config.summary, summary_usage)
    await _record_usage(budget, config.main, main_usage)
    response_usage = budget.finish(main_usage)
    assert response_usage is not None
    assert response_usage.input_tokens == 10
    assert response_usage.input_tokens_details.cached_tokens == 3


async def test_response_budget_accepts_a_plugin_defined_config_field() -> None:
    config_data = _config().to_dict()
    config_data["critic"] = config_data["main"]
    config = _Config(config_data, frozen_box=True)
    budget = ResponseBudget(config, 20)
    critic_usage = _usage(input_tokens=4, output_tokens=3)
    main_usage = _usage(input_tokens=2, output_tokens=1)

    await _record_usage(budget, config.critic, critic_usage)
    await _record_usage(budget, config.main, main_usage)

    response_usage = budget.finish(main_usage)
    assert response_usage is not None
    assert response_usage.input_tokens == 2
    assert response_usage.output_tokens == 5


async def test_response_budget_internal_then_output_keeps_internal_debit() -> None:
    config = _config()
    budget = ResponseBudget(config, 100)
    hidden_usage = _usage(input_tokens=20, output_tokens=9, reasoning_tokens=3)
    output_usage = _usage(input_tokens=4, output_tokens=2, cached_tokens=1)

    await _record_usage(budget, config.main, hidden_usage)
    assert budget.remaining == 86

    await _record_usage(budget, config.main, output_usage)
    response_usage = budget.finish(output_usage)
    assert response_usage is not None
    assert response_usage.input_tokens == 20
    assert response_usage.output_tokens == 16
    assert response_usage.output_tokens_details.reasoning_tokens == 14
    assert response_usage.total_tokens == 36
    assert budget.remaining == 84


async def test_response_budget_clamps_output_tokens_to_visible_floor_for_discounted_output() -> None:
    config_data = _config().to_dict()
    config_data["main"]["output_equivalence"] = _output_equivalence(output_to_output=0.5).to_dict()
    config = _Config(config_data, frozen_box=True)
    usage = _usage(input_tokens=10, output_tokens=20, reasoning_tokens=5)
    budget = ResponseBudget(config, 100)

    await _record_usage(budget, config.main, usage)
    response_usage = budget.finish(usage)
    assert response_usage is not None
    assert response_usage.input_tokens == 10
    assert response_usage.output_tokens == 15
    assert response_usage.output_tokens_details.reasoning_tokens == 0
    assert response_usage.total_tokens == 25
    assert budget.remaining == 90


async def test_budgeted_client_reports_last_service_tier_when_budget_is_exhausted() -> None:
    config = _config()
    budget = ResponseBudget(config, 1)
    raw_client = _StubChatClient([_delta(finish_reason=ChatFinishReason.STOP, usage=_usage(input_tokens=0, output_tokens=1))])
    request = ChatCompletionRequest(model="test-model", messages=[])

    async for _delta_item in budgeted(raw_client, budget, config.main).stream(request):
        pass

    with pytest.raises(ResponseBudgetExhaustedError) as exc_info:
        async for _delta_item in budgeted(raw_client, budget, config.main).stream(request):
            pass

    assert exc_info.value.last_service_tier == "default"


async def test_response_budget_rejects_unrecorded_usage_and_duplicate_finish() -> None:
    config = _config()
    budget = ResponseBudget(config, 20)
    usage = _usage(input_tokens=1, output_tokens=1)

    with pytest.raises(RuntimeError, match="was not recorded"):
        budget.finish(usage)

    await _record_usage(budget, config.main, usage)
    budget.finish(usage)

    with pytest.raises(RuntimeError, match="already been finished"):
        budget.finish(usage)


async def test_run_response_completes_simple_turn_without_midstream_staging() -> None:
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
            memory={},
            threads=Threads(messages={"main": [Message(role="user", content="hello")]}),
            main_tail=None,
            last_reasoning_id=None,
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


async def test_response_hooks_follow_the_documented_execution_order() -> None:
    bus.reset()
    core = importlib.reload(importlib.import_module("plap.plugins.core.loop"))
    events: list[str] = []

    @bus.listen("response.request")
    async def wrap_request(state, config, *, next):
        events.append("request.before")
        request = await next(state=state, config=config)
        events.append("request.after")
        return request

    @bus.listen("response.completion")
    async def wrap_completion(state, config, budget, request, validators, *, next):
        events.append("completion.before")
        result = await next(
            state=state,
            config=config,
            budget=budget,
            request=request,
            validators=validators,
        )
        events.append("completion.after")
        return result

    @bus.listen("response.snapshot")
    async def wrap_snapshot(state, config, request, snapshot, *, next):
        events.append("snapshot.before")
        snapshot = await next(state=state, config=config, request=request, snapshot=snapshot)
        events.append("snapshot.after")
        return snapshot

    @bus.listen("response.turn")
    async def wrap_turn(state, config, budget, request, *, next):
        events.append("turn.before")
        result = await next(state=state, config=config, budget=budget, request=request)
        events.append("turn.after")
        return result

    @bus.listen("response.loop")
    async def wrap_loop(state, config, budget, *, next):
        events.append("loop.before")
        result = await next(state=state, config=config, budget=budget)
        events.append("loop.after")
        return result

    @bus.listen("response.commit")
    async def wrap_commit(state, *, next):
        events.append("commit.before")
        await next(state=state)
        events.append("commit.after")

    state = _state(
        _RecordingStore(),
        _RecordingChannels(),
        client=_StubChatClient(
            [
                _delta(
                    content_delta="hello",
                    finish_reason=ChatFinishReason.STOP,
                    usage=_usage(input_tokens=2, output_tokens=1),
                )
            ]
        ),
    )

    try:
        await core.run_response(state=state)
    finally:
        bus.reset()
        importlib.reload(core)

    assert events == [
        "loop.before",
        "request.before",
        "request.after",
        "turn.before",
        "completion.before",
        "snapshot.before",
        "snapshot.after",
        "completion.after",
        "turn.after",
        "loop.after",
        "commit.before",
        "commit.after",
    ]


def test_state_from_ingested_preserves_active_threads() -> None:
    state = _state(ingested=_ingested(active={"reviewer"}))

    assert state.threads.active == {"reviewer"}


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
    state = _state(store, channels, client=object())
    body_started = anyio.Event()

    async def _block(**kwargs):
        _ = kwargs
        body_started.set()
        await anyio.sleep(10)

    monkeypatch.setattr("plap.plugins.core.loop.response_loop", _block)

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


async def test_run_response_completes_without_main_execution_when_main_is_inactive() -> None:
    channels = _RecordingChannels()
    store = _RecordingStore()
    state = _state(store, channels, client=object(), ingested=_ingested(active=set()))

    await run_response(state=state)

    assert store.begin_calls == 1
    assert store.fail_calls == 0
    assert store.finish_calls == 1
    assert state.coordinator.current_response().status == "completed"
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
    state = _state(store, channels, request=request, client=client)

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
    assert payload.main[0].is_assistant()
    assert payload.main[0].tool_calls[0].name == "bad"
    assert payload.main[1].is_tool()
    assert payload.main[1].content == RETRY_TOOL_PLACEHOLDER
    assert payload.main[2].role == "user"
    assert "undeclared tool" in cast(str, payload.main[2].content)
    assert isinstance(response.output[1], ResponseMessageItem)
    assert response.output[1].content[0].text == "fixed"


async def test_run_response_summary_saves_progress_on_summary_done(monkeypatch: pytest.MonkeyPatch) -> None:
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
    state = _state(store, channels, request=request, client=client)
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
    state = _state(store, channels, request=request, client=client)

    await run_response(state=state)

    response = state.coordinator.current_response()
    assert response.status == "incomplete"
    assert response.incomplete_details is not None
    assert response.incomplete_details.reason == "max_output_tokens"
    assert response.usage is not None
    assert response.usage.input_tokens == 20
    assert len(response.output) == 1
    assert isinstance(response.output[0], ResponseReasoningItem)


async def test_state_save_progress_persists_stubbed_open_tail_calls() -> None:
    store = _RecordingStore()
    channels = _RecordingChannels()
    state = _state(store, channels)

    state.threads["main"] = [
        Message(
            role="assistant",
            content="thinking",
            tool_calls=[ToolCall(id="call_main", name="search", arguments="{}")],
        )
    ]

    await state.save_progress()

    item = _last_output_item(state.coordinator)
    assert isinstance(item, ResponseReasoningItem)
    payload = open_reasoning_payload(item.encrypted_content, keyring=_keyring())
    assert payload.main[0].is_assistant()
    assert payload.main[0].tool_calls[0].id == "call_main"
    assert payload.main[1].is_tool()
    assert payload.main[1].tool_call_id == "call_main"
    assert isinstance(payload.main[1].content, str)
    assert payload.main[1].content


async def test_state_save_progress_appends_cross_lane_stub_without_main_patch() -> None:
    assistant = Message(
        role="assistant",
        content="working",
        tool_calls=[
            ToolCall(id="call_done", name="subagent", arguments="{}"),
            ToolCall(id="call_open", name="client_tool", arguments="{}"),
        ],
    )
    state = _state(
        ingested=Ingested(
            memory={},
            threads=Threads(active={"main"}, messages={"main": [assistant]}),
            main_tail=HiddenMainTail(source=assistant),
            last_reasoning_id=None,
        )
    )
    completed = Message(role="tool", tool_call_id="call_done", content="subagent result")
    state.threads["main"].append(completed)

    await state.save_progress()

    item = _last_output_item(state.coordinator)
    assert isinstance(item, ResponseReasoningItem)
    payload = open_reasoning_payload(item.encrypted_content, keyring=_keyring())
    assert payload.main == [
        completed,
        Message(role="tool", tool_call_id="call_open", content=INTERRUPTED_TOOL_OUTPUT),
    ]
    assert isinstance(payload.state, ReasoningPatch)
    assert "main" not in payload.state.threads


async def test_state_save_progress_preserves_inactive_cross_lane_call_without_stub() -> None:
    assistant = Message(
        role="assistant",
        content="working",
        tool_calls=[
            ToolCall(id="call_done", name="subagent", arguments="{}"),
            ToolCall(id="call_parked", name="client_tool", arguments="{}"),
        ],
    )
    state = _state(
        ingested=Ingested(
            memory={},
            threads=Threads(active=set(), messages={"main": [assistant]}),
            main_tail=HiddenMainTail(source=assistant),
            last_reasoning_id=None,
        )
    )
    completed = Message(role="tool", tool_call_id="call_done", content="subagent result")
    state.threads["main"].append(completed)

    await state.save_progress()

    item = _last_output_item(state.coordinator)
    assert isinstance(item, ResponseReasoningItem)
    payload = open_reasoning_payload(item.encrypted_content, keyring=_keyring())
    assert payload.main == [completed]
    assert isinstance(payload.state, ReasoningPatch)
    assert "main" not in payload.state.threads


async def test_state_save_progress_preserves_explicit_empty_thread() -> None:
    store = _RecordingStore()
    channels = _RecordingChannels()
    state = _state(store, channels)

    state.threads["reviewer"] = []

    await state.save_progress()

    item = _last_output_item(state.coordinator)
    assert isinstance(item, ResponseReasoningItem)
    payload = open_reasoning_payload(item.encrypted_content, keyring=_keyring())
    assert isinstance(payload.state, ReasoningPatch)
    assert "reviewer" in payload.state.threads
    assert payload.state.threads["reviewer"] == []


@pytest.mark.parametrize("operation", ["save_progress", "commit"])
@pytest.mark.parametrize("location", ["active", "messages"])
async def test_state_emission_rejects_unconfigured_threads(operation: str, location: str) -> None:
    state = _state()
    if location == "active":
        state.threads.active.add("unknown")
    else:
        state.threads["unknown"] = []

    with pytest.raises(ValueError, match="state contains unconfigured threads: unknown"):
        await getattr(state, operation)()


async def test_state_save_progress_rejects_persisted_main_mutation() -> None:
    state = _state(
        ingested=Ingested(
            memory={},
            threads=Threads(messages={"main": [Message(role="user", content="original")]}),
            main_tail=None,
            last_reasoning_id=None,
        )
    )
    state.threads["main"] = [Message(role="user", content="wrong lane")]

    with pytest.raises(RuntimeError, match="replace only the current response suffix"):
        await state.save_progress()


async def test_state_save_progress_accepts_replaced_current_response_suffix() -> None:
    base = Message(role="developer", content="base")
    first = Message(role="developer", content="first")
    second = Message(role="developer", content="second")
    state = _state(
        ingested=Ingested(
            memory={},
            threads=Threads(messages={"main": [base]}),
            main_tail=None,
            last_reasoning_id=None,
        )
    )
    state.threads["main"] = [base, first, second]
    state.threads["main"] = [base, second, first]

    await state.save_progress()

    item = _last_output_item(state.coordinator)
    assert isinstance(item, ResponseReasoningItem)
    payload = open_reasoning_payload(item.encrypted_content, keyring=_keyring())
    assert payload.main == [second, first]
    assert isinstance(payload.state, ReasoningPatch)
    assert "main" not in payload.state.threads


async def test_state_save_progress_accepts_cleared_current_response_suffix() -> None:
    base = Message(role="developer", content="base")
    state = _state(
        ingested=Ingested(
            memory={},
            threads=Threads(messages={"main": [base]}),
            main_tail=None,
            last_reasoning_id=None,
        )
    )
    state.threads["main"].append(Message(role="developer", content="discarded draft"))
    state.threads["main"] = [base]

    await state.save_progress()

    assert state.coordinator.current_response().output == []


async def test_state_save_progress_rejects_changes_within_persisted_main_prefix() -> None:
    first = Message(role="system", content="first")
    second = Message(role="developer", content="second")
    replacements = [
        [],
        [second, first],
        [Message(role="system", content="changed"), second],
        [first, Message(role="user", content="inserted"), second],
    ]

    for replacement in replacements:
        state = _state(
            ingested=Ingested(
                memory={},
                threads=Threads(messages={"main": [first, second]}),
                main_tail=None,
                last_reasoning_id=None,
            )
        )
        state.threads["main"] = replacement

        with pytest.raises(RuntimeError, match="replace only the current response suffix"):
            await state.save_progress()


async def test_state_save_progress_rejects_nested_persisted_main_mutation() -> None:
    base = Message(role="assistant", content="base")
    state = _state(
        ingested=Ingested(
            memory={},
            threads=Threads(messages={"main": [base]}),
            main_tail=PublicMainTail(source=None),
            last_reasoning_id=None,
        )
    )
    state.threads["main"][0].tool_calls.append(ToolCall(id="changed", name="changed", arguments="{}"))

    with pytest.raises(RuntimeError, match="replace only the current response suffix"):
        await state.save_progress()


async def test_state_commit_uses_message_patch_and_emits_visible_items() -> None:
    store = _RecordingStore()
    channels = _RecordingChannels()
    state = _state(store, channels)

    state.threads["main"] = [
        Message(
            role="assistant",
            content="hello",
            reasoning_content="hidden",
            tool_calls=[ToolCall(id="call_main", name="search", arguments="{}")],
        )
    ]

    await state.commit()

    response = state.coordinator.current_response()
    assert isinstance(response.output[0], ResponseReasoningItem)
    reasoning_payload = open_reasoning_payload(response.output[0].encrypted_content, keyring=_keyring())
    assert reasoning_payload.main[0] == state.threads["main"][0]
    assert isinstance(reasoning_payload.main[1], MessagePatch)
    assert reasoning_payload.main[1].message == state.threads["main"][0]
    assert isinstance(response.output[1], ResponseMessageItem)
    assert response.output[1].content[0].text == "hello"
    assert isinstance(response.output[2], ResponseFunctionCallItem)
    assert response.output[2].call_id.startswith("call_")


async def test_state_commit_uses_direct_hidden_assistant_for_new_call_only_turn() -> None:
    state = _state()
    assistant = Message(
        role="assistant",
        reasoning_content="private",
        tool_calls=[ToolCall(id="call_main", name="search", arguments="{}")],
    )
    state.threads["main"] = [assistant]

    await state.commit()

    response = state.coordinator.current_response()
    assert len(response.output) == 2
    assert isinstance(response.output[0], ResponseReasoningItem)
    reasoning = open_reasoning_payload(response.output[0].encrypted_content, keyring=_keyring())
    assert reasoning.main == [assistant]
    assert isinstance(response.output[1], ResponseFunctionCallItem)


async def test_state_commit_emits_full_non_main_checkpoint_after_user() -> None:
    state = _state(
        ingested=Ingested(
            memory={"base": True, "nullable": None},
            threads=Threads(
                active={"main"},
                messages={"main": [Message(role="user", content="question")]},
            ),
            main_tail=None,
            last_reasoning_id="rs_old",
            checkpoint_required=True,
        )
    )
    state.threads["reviewer"] = [Message(role="assistant", content="review state")]
    state.threads.active.add("reviewer")
    state.threads["main"].append(Message(role="assistant", content="answer", reasoning_content="private"))

    await state.commit()

    reasoning_item = state.coordinator.current_response().output[0]
    assert isinstance(reasoning_item, ResponseReasoningItem)
    payload = open_reasoning_payload(reasoning_item.encrypted_content, keyring=_keyring())
    assert payload.previous_reasoning_id is None
    assert isinstance(payload.state, ReasoningCheckpoint)
    assert payload.state.memory == {"base": True, "nullable": None}
    assert payload.state.active == {"main", "reviewer"}
    assert payload.state.threads == {"reviewer": [Message(role="assistant", content="review state")]}
    assert "main" not in payload.state.threads
    assert payload.main == [state.threads["main"][-1], MessagePatch(state.threads["main"][-1])]


async def test_state_commit_keeps_inactive_main_output_private() -> None:
    state = _state(ingested=_ingested(active=set()))
    state.threads["main"] = [
        Message(
            role="assistant",
            content="private answer",
            tool_calls=[ToolCall(id="call_main", name="search", arguments="{}")],
        )
    ]

    await state.commit()

    response = state.coordinator.current_response()
    assert len(response.output) == 1
    assert isinstance(response.output[0], ResponseReasoningItem)
    payload = open_reasoning_payload(response.output[0].encrypted_content, keyring=_keyring())
    assert payload.main == state.threads["main"]


async def test_run_response_emits_reactivated_persisted_main_call_without_main_completion() -> None:
    store = _RecordingStore()
    channels = _RecordingChannels()
    assistant = Message(
        role="assistant",
        content="private answer",
        tool_calls=[ToolCall(id="call_main", name="search", arguments="{}")],
    )
    state = _state(
        store,
        channels,
        client=object(),
        ingested=Ingested(
            memory={},
            threads=Threads(active=set(), messages={"main": [assistant]}),
            main_tail=HiddenMainTail(source=assistant),
            last_reasoning_id=None,
        ),
    )
    state.threads.active.add("main")

    await run_response(state=state)

    response = state.coordinator.current_response()
    assert response.status == "completed"
    assert len(response.output) == 3
    assert isinstance(response.output[0], ResponseReasoningItem)
    reasoning = open_reasoning_payload(response.output[0].encrypted_content, keyring=_keyring())
    assert isinstance(reasoning.state, ReasoningPatch)
    assert reasoning.state.active == {"main"}
    assert reasoning.main == [MessagePatch(message=assistant)]
    assert isinstance(response.output[1], ResponseMessageItem)
    assert response.output[1].content[0].text == "private answer"
    assert isinstance(response.output[2], ResponseFunctionCallItem)
    assert open_call_id(response.output[2].call_id, keyring=_keyring(), thread_codes=_thread_codes()).thread == "main"


async def test_state_commit_materializes_parked_text_tail_after_activation() -> None:
    assistant = Message(role="assistant", content="delayed answer", reasoning_content="hidden")
    state = _state(
        ingested=Ingested(
            memory={},
            threads=Threads(active=set(), messages={"main": [assistant]}),
            main_tail=HiddenMainTail(source=assistant),
            last_reasoning_id=None,
        )
    )
    state.threads.active.add("main")

    await state.commit()

    response = state.coordinator.current_response()
    assert isinstance(response.output[0], ResponseReasoningItem)
    reasoning = open_reasoning_payload(response.output[0].encrypted_content, keyring=_keyring())
    assert reasoning.main == [MessagePatch(message=assistant)]
    assert isinstance(response.output[1], ResponseMessageItem)
    assert response.output[1].content[0].text == "delayed answer"


async def test_state_commit_does_not_republish_active_public_tail() -> None:
    source = Message(role="assistant", content="sealed answer", reasoning_content="private")
    public = Message(role="assistant", content="edited answer", reasoning_content="private")
    state = _state(
        ingested=Ingested(
            memory={},
            threads=Threads(active={"main"}, messages={"main": [public]}),
            main_tail=PublicMainTail(source=source),
            last_reasoning_id="rs_public",
        )
    )

    await state.commit()

    assert state.coordinator.current_response().output == []


async def test_state_commit_does_not_republish_reactivated_public_tail() -> None:
    source = Message(role="assistant", content="sealed answer", reasoning_content="private")
    public = Message(role="assistant", content="edited answer", reasoning_content="private")
    state = _state(
        ingested=Ingested(
            memory={},
            threads=Threads(active=set(), messages={"main": [public]}),
            main_tail=PublicMainTail(source=source),
            last_reasoning_id="rs_public",
        )
    )
    state.threads.active.add("main")

    await state.commit()

    response = state.coordinator.current_response()
    assert len(response.output) == 1
    assert isinstance(response.output[0], ResponseReasoningItem)
    reasoning = open_reasoning_payload(response.output[0].encrypted_content, keyring=_keyring())
    assert isinstance(reasoning.state, ReasoningPatch)
    assert reasoning.state.active == {"main"}
    assert reasoning.main == []


async def test_state_commit_publishes_new_active_tail_after_public_history() -> None:
    source = Message(role="assistant", content="sealed answer", reasoning_content="private")
    public = Message(role="assistant", content="edited answer", reasoning_content="private")
    current = Message(role="assistant", content="new answer", reasoning_content="new private")
    state = _state(
        ingested=Ingested(
            memory={},
            threads=Threads(active={"main"}, messages={"main": [public]}),
            main_tail=PublicMainTail(source=source),
            last_reasoning_id="rs_public",
        )
    )
    state.threads["main"].append(current)

    await state.commit()

    response = state.coordinator.current_response()
    assert isinstance(response.output[0], ResponseReasoningItem)
    reasoning = open_reasoning_payload(response.output[0].encrypted_content, keyring=_keyring())
    assert reasoning.main == [current, MessagePatch(current)]
    assert isinstance(response.output[1], ResponseMessageItem)
    assert response.output[1].content[0].text == "new answer"


async def test_state_commit_does_not_publish_reactivated_compacted_tail() -> None:
    snapshot = Message(role="assistant", content="historical answer", reasoning_content="compacted private")
    state = _state(
        ingested=Ingested(
            memory={},
            threads=Threads(active=set(), messages={"main": [snapshot]}),
            main_tail=CompactedMainTail(source=snapshot),
            last_reasoning_id=None,
            last_compaction_id="cmp_history",
        )
    )
    state.threads.active.add("main")

    await state.commit()

    response = state.coordinator.current_response()
    assert len(response.output) == 1
    assert isinstance(response.output[0], ResponseReasoningItem)
    reasoning = open_reasoning_payload(response.output[0].encrypted_content, keyring=_keyring())
    assert isinstance(reasoning.state, ReasoningPatch)
    assert reasoning.state.active == {"main"}
    assert reasoning.main == []


async def test_state_commit_repositions_persisted_call_only_tail_without_public_message() -> None:
    assistant = Message(
        role="assistant",
        tool_calls=[ToolCall(id="call_main", name="search", arguments="{}")],
    )
    state = _state(
        ingested=Ingested(
            memory={},
            threads=Threads(active=set(), messages={"main": [assistant]}),
            main_tail=HiddenMainTail(source=assistant),
            last_reasoning_id="rs_parked",
        )
    )
    state.threads.active.add("main")

    await state.commit()

    response = state.coordinator.current_response()
    assert len(response.output) == 2
    assert isinstance(response.output[0], ResponseReasoningItem)
    reasoning = open_reasoning_payload(response.output[0].encrypted_content, keyring=_keyring())
    assert reasoning.main == [MessagePatch(assistant)]
    assert isinstance(response.output[1], ResponseFunctionCallItem)


async def test_state_commit_materializes_partial_cross_lane_settlement() -> None:
    assistant = Message(
        role="assistant",
        content="working",
        tool_calls=[
            ToolCall(id="call_done", name="subagent", arguments="{}"),
            ToolCall(id="call_open", name="client_tool", arguments="{}"),
        ],
    )
    state = _state(
        ingested=Ingested(
            memory={},
            threads=Threads(active={"main"}, messages={"main": [assistant]}),
            main_tail=HiddenMainTail(source=assistant),
            last_reasoning_id=None,
        )
    )
    completed = Message(role="tool", tool_call_id="call_done", content="subagent result")
    state.threads["main"].append(completed)

    await state.commit()

    response = state.coordinator.current_response()
    assert isinstance(response.output[0], ResponseReasoningItem)
    reasoning = open_reasoning_payload(response.output[0].encrypted_content, keyring=_keyring())
    assert reasoning.main == [completed, MessagePatch(message=assistant)]
    assert isinstance(response.output[1], ResponseMessageItem)
    calls = [item for item in response.output if isinstance(item, ResponseFunctionCallItem)]
    assert [item.name for item in calls] == ["client_tool"]


async def test_state_commit_keeps_fully_settled_main_turn_hidden() -> None:
    assistant = Message(
        role="assistant",
        content="working",
        tool_calls=[ToolCall(id="call_done", name="subagent", arguments="{}")],
    )
    state = _state(
        ingested=Ingested(
            memory={},
            threads=Threads(active={"main"}, messages={"main": [assistant]}),
            main_tail=HiddenMainTail(source=assistant),
            last_reasoning_id=None,
        )
    )
    completed = Message(role="tool", tool_call_id="call_done", content="subagent result")
    state.threads["main"].append(completed)

    await state.commit()

    response = state.coordinator.current_response()
    assert len(response.output) == 1
    assert isinstance(response.output[0], ResponseReasoningItem)
    reasoning = open_reasoning_payload(response.output[0].encrypted_content, keyring=_keyring())
    assert reasoning.main == [completed]


async def test_state_commit_emits_active_thread_calls_in_deterministic_order() -> None:
    state = _state(ingested=_ingested(active={"main", "reviewer", "defender"}))
    state.threads["main"] = [
        Message(role="assistant", tool_calls=[ToolCall(id="call_main", name="main_tool", arguments="{}")]),
    ]
    state.threads["reviewer"] = [
        Message(role="assistant", tool_calls=[ToolCall(id="call_reviewer", name="review_tool", arguments="{}")]),
    ]
    state.threads["defender"] = [
        Message(role="assistant", tool_calls=[ToolCall(id="call_defender", name="defend_tool", arguments="{}")]),
    ]

    await state.commit()

    calls = [item for item in state.coordinator.current_response().output if isinstance(item, ResponseFunctionCallItem)]
    assert [open_call_id(item.call_id, keyring=_keyring(), thread_codes=_thread_codes()).thread for item in calls] == [
        "main",
        "defender",
        "reviewer",
    ]


async def test_state_commit_keeps_closed_assistant_with_user_tail_hidden() -> None:
    store = _RecordingStore()
    channels = _RecordingChannels()
    state = _state(store, channels)

    state.threads["main"] = [
        Message(role="assistant", content="hidden assistant"),
        Message(role="user", content="tail"),
    ]

    await state.commit()

    response = state.coordinator.current_response()
    assert len(response.output) == 1
    assert isinstance(response.output[0], ResponseReasoningItem)
    payload = open_reasoning_payload(response.output[0].encrypted_content, keyring=_keyring())
    assert isinstance(payload.main[0], Message)
    assert payload.main[0].content == "hidden assistant"
    assert payload.main[1].role == "user"
    assert payload.main[1].content == "tail"


async def test_run_response_finishes_all_public_calls_when_cancelled_during_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _RecordingStore()
    channels = _RecordingChannels()
    state = _state(
        store,
        channels,
        client=object(),
        ingested=Ingested(
            memory={},
            threads=Threads(
                active=set(),
                messages={
                    "main": [Message(role="assistant", tool_calls=[ToolCall(id="call_main", name="main_tool", arguments="{}")])],
                    "reviewer": [
                        Message(
                            role="assistant",
                            tool_calls=[ToolCall(id="call_reviewer", name="review_tool", arguments="{}")],
                        )
                    ],
                },
            ),
            main_tail=HiddenMainTail(
                source=Message(
                    role="assistant",
                    tool_calls=[ToolCall(id="call_main", name="main_tool", arguments="{}")],
                ),
            ),
            last_reasoning_id=None,
        ),
    )
    state.threads.active.add("main")
    state.threads.active.add("reviewer")
    first_call_emitted = anyio.Event()
    release_commit = anyio.Event()
    original_emit = StreamCoordinator.emit
    emitted_calls = 0

    async def emit_with_pause(coordinator: StreamCoordinator, item) -> None:
        nonlocal emitted_calls
        await original_emit(coordinator, item)
        if not isinstance(item, ResponseFunctionCallItem):
            return
        emitted_calls += 1
        if emitted_calls == 1:
            first_call_emitted.set()
            await release_commit.wait()

    monkeypatch.setattr(StreamCoordinator, "emit", emit_with_pause)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(partial(run_response, state=state))
        await first_call_emitted.wait()
        task_group.cancel_scope.cancel()
        release_commit.set()

    response = state.coordinator.current_response()
    assert response.status == "completed"
    assert emitted_calls == 2
    assert store.cancel_calls == 0
