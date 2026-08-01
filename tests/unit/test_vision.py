from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import field
from uuid import uuid4

import anyio
import msgspec
import pytest
import svcs
from box import Box

from plap.bus import bus
from plap.config import CueBox
from plap.keyring import SealingKeyring
from plap.llms.completions.budget import BudgetedChatCompletionClient, CompletionBudget
from plap.llms.completions.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatContentImage,
    ChatFinishReason,
    ChatImageURL,
    ChatMessage,
    ChatToolCall,
    ChatToolCallDelta,
    ChatUsage,
    IChatCompletionClient,
)
from plap.plugins.vision import VISION_TOOL_NAME, _image_id, _vision_history_messages
from plap.responses.contracts import ResponseCreateRequest
from plap.responses.contracts.items import ResponseFunctionCallItem, ResponseMessageItem, ResponseReasoningItem
from plap.responses.ingest.models import Ingested, Message, Threads
from plap.responses.ingest.sealing import open_reasoning_payload
from plap.responses.routes import _accept_response, _execute_response
from plap.responses.state import State
from plap.responses.store import PreparedRequest
from plap.responses.streaming import StreamCoordinator


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
        self.finish_calls = 0

    async def begin_response(self, prepared: PreparedRequest, response) -> None:
        _ = prepared, response
        self.begin_calls += 1

    async def append_output_item(self, prepared: PreparedRequest, response_id: str, output_index: int, item: object) -> None:
        _ = prepared, response_id, output_index, item

    async def replace_output_item(self, prepared: PreparedRequest, response_id: str, output_index: int, item: object) -> None:
        _ = prepared, response_id, output_index, item

    async def finish_response(self, prepared: PreparedRequest, response) -> None:
        _ = prepared, response
        self.finish_calls += 1

    async def cancel_response(self, prepared: PreparedRequest, response) -> bool:
        _ = prepared, response
        return True

    async def fail_response(self, prepared: PreparedRequest, response) -> bool:
        _ = prepared, response
        return True


class _Client:
    def __init__(
        self,
        *,
        streams: list[list[ChatCompletionDelta]],
        completes: list[ChatCompletionResult],
    ) -> None:
        self._streams = list(streams)
        self._completes = list(completes)
        self.stream_requests: list[ChatCompletionRequest] = []
        self.complete_requests: list[ChatCompletionRequest] = []

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        self.complete_requests.append(request)
        if not self._completes:  # pragma: no cover
            raise AssertionError("unexpected extra completion request")
        return self._completes.pop(0)

    def stream(self, request: ChatCompletionRequest):
        self.stream_requests.append(request)
        if len(self.stream_requests) > len(self._streams):  # pragma: no cover
            raise AssertionError("unexpected extra stream request")
        deltas = list(self._streams[len(self.stream_requests) - 1])

        async def run():
            for delta in deltas:
                yield delta

        return run()

    async def aclose(self) -> None:
        return None


class _BlockingClient:
    def __init__(self, delta: ChatCompletionDelta) -> None:
        self._delta = delta
        self.blocked = anyio.Event()
        self.stream_requests: list[ChatCompletionRequest] = []

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        _ = request
        raise AssertionError("unexpected completion request")

    def stream(self, request: ChatCompletionRequest):
        self.stream_requests.append(request)

        async def run():
            yield self._delta
            self.blocked.set()
            await anyio.sleep_forever()

        return run()

    async def aclose(self) -> None:
        return None


def _reload_handlers():
    bus.reset()
    core_module = importlib.import_module("plap.plugins.core.loop")
    easy_server_tools_module = importlib.import_module("plap.plugins.easy.server_tools")
    easy_module = importlib.import_module("plap.plugins.easy")
    vision_module = importlib.import_module("plap.plugins.vision")
    core_module = importlib.reload(core_module)
    importlib.reload(easy_server_tools_module)
    importlib.reload(easy_module)
    importlib.reload(vision_module)
    return core_module


def _keyring() -> SealingKeyring:
    return SealingKeyring(roots=(b"i" * 32,))


def _request(**updates: object) -> ResponseCreateRequest:
    return ResponseCreateRequest(model="plap-ai/wisp", input="hello", **updates)


def _prepared(request: ResponseCreateRequest | None = None) -> PreparedRequest:
    actual_request = request or _request()
    return PreparedRequest(
        scope_id=uuid4(),
        response_request=actual_request,
        execution_request=actual_request,
        stored_input_items=[],
    )


def _image(url: str, *, detail: str | None = None) -> ChatContentImage:
    return ChatContentImage(image_url=ChatImageURL(url=url, detail=detail))


def _image_message(url: str, *, detail: str | None = None) -> Message:
    return Message(role="user", content=[_image(url, detail=detail)])


def _ingested(url: str = "https://example.com/cat.png") -> Ingested:
    return Ingested(
        memory={},
        threads=Threads(messages={"main": [_image_message(url)]}),
        main_tail=None,
        last_reasoning_id=None,
    )


def _server_history_ingested(*, external_name: str | None = None) -> Ingested:
    image = _image("https://example.com/cat.png")
    arguments = msgspec.json.encode({"ids": [_image_id(image)], "prompt": "describe"}).decode()
    calls = [ChatToolCall(id="call_server", name=VISION_TOOL_NAME, arguments=arguments)]
    if external_name is not None:
        calls.append(ChatToolCall(id="call_external", name=external_name, arguments="{}"))
    messages = [
        Message(role="user", content=[image]),
        Message(
            role="assistant",
            tool_calls=calls,
            memory={"server_tools": {"call_ids": ["call_server"]}},
        ),
        Message(
            role="tool",
            tool_call_id="call_server",
            content="vision output",
            memory={"server_tools": {"tool": VISION_TOOL_NAME, "arguments": arguments}},
        ),
    ]
    if external_name is not None:
        messages.append(Message(role="tool", tool_call_id="call_external", content="external output"))
    return Ingested(memory={}, threads=Threads(messages={"main": messages}), main_tail=None, last_reasoning_id=None)


def _tool_call_name(messages: Sequence[ChatMessage], call_id: str) -> str:
    return next(call.name for message in messages for call in message.tool_calls if call.id == call_id)


class _Config(Box):
    def resolve(self, request: dict[str, object] | None = None, /, **kwargs: object) -> _Config:
        _ = request, kwargs
        return self


def _config() -> _Config:
    return _Config(
        {
            "display_name": "Test Model",
            "main": {
                "model": "test-model",
                "max_completion_tokens": None,
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
            "vision": {
                "model": "vision-model",
                "max_completion_tokens": 8192,
                "reasoning_effort": None,
                "service_tier": None,
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
                "output_equivalence": {
                    "uncached_input_to_output": 0.25,
                    "cached_input_to_output": 0.05,
                    "output_to_output": 1.0,
                },
            },
            "reasoning_to_output": 1.0,
        },
        frozen_box=True,
    )


def _svcs(config: Box | None = None) -> svcs.Container:
    registry = svcs.Registry()
    registry.register_value(SealingKeyring, _keyring())
    registry.register_value(CueBox, config or _config())
    return svcs.Container(registry)


def _coordinator(store: _RecordingStore, channels: _RecordingChannels, request: ResponseCreateRequest) -> StreamCoordinator:
    return StreamCoordinator(
        request=request,
        channels=channels,
        prepared=_prepared(request),
        response_store=store,
        sealing_keyring=_keyring(),
    )


def _state(
    client: IChatCompletionClient,
    *,
    request: ResponseCreateRequest | None = None,
    ingested: Ingested | None = None,
    config: Box | None = None,
) -> tuple[State, _RecordingStore, _RecordingChannels]:
    actual_request = request or _request()
    actual_config = config or _config()
    store = _RecordingStore()
    channels = _RecordingChannels()
    container = _svcs(actual_config)
    budget = CompletionBudget(
        actual_request.max_output_tokens,
        reasoning_to_output=float(actual_config.reasoning_to_output),
    )
    container.register_local_value(CompletionBudget, budget)
    container.register_local_value(BudgetedChatCompletionClient, BudgetedChatCompletionClient(client, budget))
    container.register_local_value(StreamCoordinator, _coordinator(store, channels, actual_request))
    state = State.from_ingested(
        ingested=ingested or _ingested(),
        request=actual_request,
        config=actual_config,
        svcs=container,
        thread_codes={"main": 0},
    )
    return state, store, channels


async def _run(state: State) -> None:
    await _accept_response(state)
    await _execute_response(state)


def _usage(*, input_tokens: int, output_tokens: int, reasoning_tokens: int | None = None) -> ChatUsage:
    return ChatUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _delta(
    *,
    content_delta: str | None = None,
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
        tool_call_delta=tool_call_delta,
        finish_reason=finish_reason,
        usage=usage,
        service_tier="default",
    )


def _tool_step(*calls: tuple[str, str, str], usage: ChatUsage) -> list[ChatCompletionDelta]:
    deltas = [
        _delta(tool_call_delta=ChatToolCallDelta(index=index, id=call_id, name=name, arguments_delta=arguments))
        for index, (call_id, name, arguments) in enumerate(calls)
    ]
    deltas.append(_delta(finish_reason=ChatFinishReason.TOOL_CALLS, usage=usage))
    return deltas


def _text_step(text: str, *, usage: ChatUsage) -> list[ChatCompletionDelta]:
    return [_delta(content_delta=text, finish_reason=ChatFinishReason.STOP, usage=usage)]


def _complete(
    text: str,
    *,
    reasoning_content: str | None = None,
    usage: ChatUsage | None = None,
) -> ChatCompletionResult:
    return ChatCompletionResult(
        id="cmp_vision",
        model="vision-model",
        created_at=None,
        message=ChatMessage(role="assistant", content=text, reasoning_content=reasoning_content),
        finish_reason=ChatFinishReason.STOP,
        usage=usage,
        service_tier="default",
    )


def _tool_names(request: ChatCompletionRequest) -> list[str]:
    return [tool.function.name for tool in request.tools]


def _message_text(item: ResponseMessageItem) -> str:
    return item.content[0].text


def _output_items(state: State) -> list[object]:
    return state.svcs.get(StreamCoordinator).current_response().output


def test_image_id_ignores_detail_for_same_source() -> None:
    assert _image_id(_image("https://example.com/cat.png", detail="low")) == _image_id(_image("https://example.com/cat.png", detail="high"))


def test_vision_history_messages_replay_images_and_prior_turns_in_order_without_dedup() -> None:
    image_a = _image("https://example.com/a.png")
    image_b = _image("https://example.com/b.png")
    image_c = _image("https://example.com/c.png")
    image_a_id = _image_id(image_a)
    image_b_id = _image_id(image_b)
    history = [
        ChatMessage(role="user", content=[image_a, image_b]),
        ChatMessage(
            role="assistant",
            tool_calls=[
                ChatToolCall(
                    id="call_vision_1",
                    name=VISION_TOOL_NAME,
                    arguments=msgspec.json.encode({"ids": [image_a_id, image_b_id], "prompt": "compare them"}).decode(),
                )
            ],
        ),
        ChatMessage(
            role="tool",
            tool_call_id="call_vision_1",
            content="first comparison",
            memory={
                "server_tools": {
                    "tool": VISION_TOOL_NAME,
                    "arguments": msgspec.json.encode({"ids": [image_a_id, image_b_id], "prompt": "compare them"}).decode(),
                }
            },
        ),
        ChatMessage(role="user", content=[image_c]),
        ChatMessage(role="user", content=[image_a]),
    ]

    transcript = _vision_history_messages(history)

    assert len(transcript) == 5
    assert isinstance(transcript[0].content, list)
    assert transcript[0].content[0].text == image_a_id
    assert transcript[0].content[2].text == image_b_id
    assert transcript[1].content == f"Selected image ids: {image_a_id}, {image_b_id}\nQuestion: compare them"
    assert transcript[2].content == "first comparison"
    assert isinstance(transcript[3].content, list)
    assert transcript[3].content[0].text == _image_id(image_c)
    assert isinstance(transcript[4].content, list)
    assert transcript[4].content[0].text == image_a_id


def test_vision_history_messages_replay_hidden_reasoning_from_tool_messages() -> None:
    image = _image("https://example.com/a.png")
    image_id = _image_id(image)
    history = [
        ChatMessage(role="user", content=[image]),
        ChatMessage(
            role="assistant",
            tool_calls=[
                ChatToolCall(
                    id="call_vision_1",
                    name=VISION_TOOL_NAME,
                    arguments=msgspec.json.encode({"ids": [image_id], "prompt": "inspect closely"}).decode(),
                )
            ],
        ),
        ChatMessage(
            role="tool",
            tool_call_id="call_vision_1",
            content="first comparison",
            memory={
                "vision": {"reasoning_content": "I checked the labels before comparing the shapes."},
                "server_tools": {
                    "tool": VISION_TOOL_NAME,
                    "arguments": msgspec.json.encode({"ids": [image_id], "prompt": "inspect closely"}).decode(),
                },
            },
        ),
    ]

    transcript = _vision_history_messages(history)

    assert len(transcript) == 3
    assert transcript[2].role == "assistant"
    assert transcript[2].content == "first comparison"
    assert transcript[2].reasoning_content == "I checked the labels before comparing the shapes."


@pytest.mark.anyio
async def test_request_rewrites_images_and_preserves_none_tool_choice() -> None:
    core = _reload_handlers()
    client = _Client(streams=[], completes=[])
    state, _, _ = _state(client, request=_request(tool_choice="none"))

    request = await core.response_request(state=state)

    assert request.tool_choice == "none"
    assert _tool_names(request) == [VISION_TOOL_NAME]
    assert sum(1 for message in request.messages if message.role == "developer") == 1
    assert isinstance(request.messages[1].content, list)
    assert request.messages[1].content[0].text.startswith("image-")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("client_names", "expected_wire_name"),
    [
        ([], "vision"),
        (["vision_2"], "vision"),
        (["vision", "vision_2"], "vision_3"),
    ],
)
async def test_historical_server_calls_follow_current_wire_binding(
    client_names: list[str],
    expected_wire_name: str,
) -> None:
    core = _reload_handlers()
    client = _Client(streams=[], completes=[])
    request = _request(tools=[{"type": "function", "name": name, "parameters": {"type": "object"}} for name in client_names])
    state, _, _ = _state(client, request=request, ingested=_server_history_ingested())

    finalized = await core.response_request(state=state)

    assert _tool_names(finalized) == [*client_names, expected_wire_name]
    assert _tool_call_name(finalized.messages, "call_server") == expected_wire_name
    assert _tool_call_name(state.threads["main"], "call_server") == VISION_TOOL_NAME


@pytest.mark.anyio
async def test_historical_rebinding_leaves_client_calls_unchanged() -> None:
    core = _reload_handlers()
    client = _Client(streams=[], completes=[])
    request = _request(
        tools=[
            {"type": "function", "name": "vision", "parameters": {"type": "object"}},
            {"type": "function", "name": "vision_2", "parameters": {"type": "object"}},
        ]
    )
    state, _, _ = _state(
        client,
        request=request,
        ingested=_server_history_ingested(external_name="vision"),
    )

    finalized = await core.response_request(state=state)

    assert _tool_call_name(finalized.messages, "call_server") == "vision_3"
    assert _tool_call_name(finalized.messages, "call_external") == "vision"


@pytest.mark.anyio
async def test_internal_images_tool_loops_to_final_answer() -> None:
    _reload_handlers()
    image_id = _image_id(_image("https://example.com/cat.png"))
    client = _Client(
        streams=[
            _tool_step(
                ("call_vision", VISION_TOOL_NAME, f'{{"ids":["{image_id}"],"prompt":"describe"}}'),
                usage=_usage(input_tokens=8, output_tokens=4),
            ),
            _text_step("final answer", usage=_usage(input_tokens=5, output_tokens=3)),
        ],
        completes=[_complete("vision output", usage=_usage(input_tokens=6, output_tokens=2))],
    )
    state, store, _ = _state(client)

    await _run(state)

    assert store.begin_calls == 1
    assert store.finish_calls == 1
    assert len(client.stream_requests) == 2
    assert len(client.complete_requests) == 1
    assert client.complete_requests[0].messages[0].role == "developer"
    assert "labels the image immediately after it" in client.complete_requests[0].messages[0].content
    assert client.complete_requests[0].messages[1].content[0].text == image_id
    assert client.complete_requests[0].messages[1].content[1].image_url.url == "https://example.com/cat.png"
    assert client.complete_requests[0].messages[2].content == f"Selected image ids: {image_id}\nQuestion: describe"
    assert any(message.role == "tool" and message.content == "vision output" for message in client.stream_requests[1].messages)
    output = _output_items(state)
    assert not any(isinstance(item, ResponseFunctionCallItem) for item in output)
    final_message = next(item for item in output if isinstance(item, ResponseMessageItem))
    assert _message_text(final_message) == "final answer"


@pytest.mark.anyio
async def test_internal_images_tool_closes_call_when_response_budget_is_exhausted() -> None:
    _reload_handlers()
    image_id = _image_id(_image("https://example.com/cat.png"))
    client = _Client(
        streams=[
            _tool_step(
                ("call_vision_1", VISION_TOOL_NAME, f'{{"ids":["{image_id}"],"prompt":"describe"}}'),
                ("call_vision_2", VISION_TOOL_NAME, f'{{"ids":["{image_id}"],"prompt":"double check"}}'),
                usage=_usage(input_tokens=8, output_tokens=4),
            )
        ],
        completes=[],
    )
    state, _, _ = _state(client, request=_request(max_output_tokens=4))

    await _run(state)

    assert client.complete_requests == []
    assert state.svcs.get(StreamCoordinator).current_response().status == "incomplete"
    failed = [message for message in state.threads["main"] if message.role == "tool"]
    assert [message.tool_call_id for message in failed] == ["call_vision_1", "call_vision_2"]
    assert all(message.content == "The server could not run this tool because the response budget was exhausted." for message in failed)
    assert all(message.memory["server_tools"]["error"] == "budget_exhausted" for message in failed)
    assert not any(isinstance(item, ResponseFunctionCallItem) for item in _output_items(state))


@pytest.mark.anyio
async def test_internal_images_tool_replays_prior_hidden_vision_reasoning_into_later_vision_turns() -> None:
    _reload_handlers()
    image_id = _image_id(_image("https://example.com/cat.png"))
    client = _Client(
        streams=[
            _tool_step(
                ("call_vision_1", VISION_TOOL_NAME, f'{{"ids":["{image_id}"],"prompt":"describe the image"}}'),
                usage=_usage(input_tokens=8, output_tokens=4),
            ),
            _tool_step(
                ("call_vision_2", VISION_TOOL_NAME, f'{{"ids":["{image_id}"],"prompt":"double check the labels"}}'),
                usage=_usage(input_tokens=7, output_tokens=3),
            ),
            _text_step("final answer", usage=_usage(input_tokens=5, output_tokens=3)),
        ],
        completes=[
            _complete(
                "first vision output",
                reasoning_content="I read the labels before answering.",
                usage=_usage(input_tokens=6, output_tokens=2),
            ),
            _complete(
                "second vision output",
                reasoning_content="I verified the labels again.",
                usage=_usage(input_tokens=6, output_tokens=2),
            ),
        ],
    )
    state, _, _ = _state(client)

    await _run(state)

    assert len(client.complete_requests) == 2
    second_request = client.complete_requests[1]
    assert second_request.messages[3].role == "assistant"
    assert second_request.messages[3].content == "first vision output"
    assert second_request.messages[3].reasoning_content == "I read the labels before answering."
    final_message = next(item for item in _output_items(state) if isinstance(item, ResponseMessageItem))
    assert _message_text(final_message) == "final answer"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "ids_case",
    [
        "prefixless",
        "split_full_jsonish",
        "split_prefixless_jsonish",
        "split_full_backslashes",
    ],
)
async def test_internal_images_tool_accepts_lenient_image_ids(ids_case: str) -> None:
    _reload_handlers()
    image_id = _image_id(_image("https://example.com/cat.png"))
    prefixless = image_id.removeprefix("image-")
    if ids_case == "prefixless":
        ids = [prefixless]
    elif ids_case == "split_full_jsonish":
        ids = list(f'["{image_id}"]')
    elif ids_case == "split_prefixless_jsonish":
        ids = list(f'["{prefixless}"]')
    elif ids_case == "split_full_backslashes":
        ids = list(f"[\\{image_id}\\]")
    else:  # pragma: no cover
        raise AssertionError(ids_case)
    arguments = msgspec.json.encode({"ids": ids, "prompt": "describe"}).decode()
    client = _Client(
        streams=[
            _tool_step(
                ("call_vision", VISION_TOOL_NAME, arguments),
                usage=_usage(input_tokens=8, output_tokens=4),
            ),
            _text_step("final answer", usage=_usage(input_tokens=5, output_tokens=3)),
        ],
        completes=[_complete("vision output", usage=_usage(input_tokens=6, output_tokens=2))],
    )
    state, _, _ = _state(client)

    await _run(state)

    assert len(client.stream_requests) == 2
    assert len(client.complete_requests) == 1
    assert client.complete_requests[0].messages[2].content == f"Selected image ids: {image_id}\nQuestion: describe"
    assert not any(
        message.role == "user" and isinstance(message.content, str) and "unknown image ids" in message.content.lower()
        for request in client.stream_requests
        for message in request.messages
    )
    final_message = next(item for item in _output_items(state) if isinstance(item, ResponseMessageItem))
    assert _message_text(final_message) == "final answer"


@pytest.mark.anyio
async def test_internal_images_tool_applies_vision_sampling_config() -> None:
    _reload_handlers()
    image_id = _image_id(_image("https://example.com/cat.png"))
    request = _request(temperature=0.4, top_p=0.9)
    client = _Client(
        streams=[
            _tool_step(
                ("call_vision", VISION_TOOL_NAME, f'{{"ids":["{image_id}"],"prompt":"describe"}}'),
                usage=_usage(input_tokens=8, output_tokens=4),
            ),
            _text_step("final answer", usage=_usage(input_tokens=5, output_tokens=3)),
        ],
        completes=[_complete("vision output", usage=_usage(input_tokens=6, output_tokens=2))],
    )
    config_data = _config().to_dict()
    config_data["vision"]["sampling"] = {
        "temperature": {
            "disabled": False,
            "fixed": None,
            "default": None,
            "scale": 0.5,
            "offset": 0.1,
            "min_value": None,
            "max_value": None,
        },
        "top_p": {
            "disabled": False,
            "fixed": None,
            "default": None,
            "scale": 1.0,
            "offset": -0.2,
            "min_value": None,
            "max_value": None,
        },
        "min_p": {
            "disabled": False,
            "fixed": None,
            "default": 0.05,
            "scale": 1.0,
            "offset": 0.0,
            "min_value": None,
            "max_value": None,
        },
        "top_k": {
            "disabled": False,
            "fixed": None,
            "default": 23,
            "min_value": None,
            "max_value": None,
        },
        "frequency_penalty": {
            "disabled": False,
            "fixed": None,
            "default": 0.4,
            "scale": 1.0,
            "offset": 0.0,
            "min_value": None,
            "max_value": None,
        },
        "presence_penalty": {
            "disabled": False,
            "fixed": None,
            "default": -0.2,
            "scale": 1.0,
            "offset": 0.0,
            "min_value": None,
            "max_value": None,
        },
        "repetition_penalty": {
            "disabled": False,
            "fixed": None,
            "default": 1.2,
            "scale": 1.0,
            "offset": 0.0,
            "min_value": None,
            "max_value": None,
        },
        "seed": {
            "disabled": False,
            "fixed": None,
            "default": 99,
            "min_value": None,
            "max_value": None,
        },
        "top_logprobs": None,
    }
    state, _, _ = _state(client, request=request, config=_Config(config_data, frozen_box=True))

    await _run(state)

    vision_request = client.complete_requests[0]
    assert vision_request.temperature == pytest.approx(0.3)
    assert vision_request.top_p == pytest.approx(0.7)
    assert vision_request.min_p == pytest.approx(0.05)
    assert vision_request.top_k == 23
    assert vision_request.frequency_penalty == 0.4
    assert vision_request.presence_penalty == -0.2
    assert vision_request.repetition_penalty == pytest.approx(1.2)
    assert vision_request.seed == 99


@pytest.mark.anyio
async def test_internal_images_tool_stays_hidden_when_external_tool_remains_open() -> None:
    _reload_handlers()
    image_id = _image_id(_image("https://example.com/cat.png"))
    client = _Client(
        streams=[
            _tool_step(
                ("call_vision", VISION_TOOL_NAME, f'{{"ids":["{image_id}"],"prompt":"describe"}}'),
                ("call_client", "client_tool", "{}"),
                usage=_usage(input_tokens=8, output_tokens=4),
            )
        ],
        completes=[_complete("vision output", usage=_usage(input_tokens=6, output_tokens=2))],
    )
    state, _, _ = _state(
        client,
        request=_request(tools=[{"type": "function", "name": "client_tool", "parameters": {"type": "object"}}]),
    )

    await _run(state)

    output = _output_items(state)
    visible_calls = [item for item in output if isinstance(item, ResponseFunctionCallItem)]
    assert [item.name for item in visible_calls] == ["client_tool"]
    assert all(item.name != VISION_TOOL_NAME for item in visible_calls)


@pytest.mark.anyio
async def test_vision_server_tool_rebinds_without_renaming_client_tools() -> None:
    _reload_handlers()
    image_id = _image_id(_image("https://example.com/cat.png"))
    client = _Client(
        streams=[
            _tool_step(
                ("call_vision", "vision_3", f'{{"ids":["{image_id}"],"prompt":"describe"}}'),
                usage=_usage(input_tokens=8, output_tokens=4),
            ),
            _text_step("final answer", usage=_usage(input_tokens=5, output_tokens=3)),
        ],
        completes=[_complete("vision output", usage=_usage(input_tokens=6, output_tokens=2))],
    )
    state, _, _ = _state(
        client,
        request=_request(
            tools=[
                {"type": "function", "name": "vision", "parameters": {"type": "object"}},
                {"type": "function", "name": "vision_2", "parameters": {"type": "object"}},
            ]
        ),
    )

    await _run(state)

    assert _tool_names(client.stream_requests[0]) == ["vision", "vision_2", "vision_3"]
    assistant = next(message for message in state.threads["main"] if any(call.id == "call_vision" for call in message.tool_calls))
    assert _tool_call_name([assistant], "call_vision") == VISION_TOOL_NAME
    assert assistant.memory["server_tools"]["call_ids"] == ["call_vision"]
    assert _tool_call_name(client.stream_requests[1].messages, "call_vision") == "vision_3"
    tool_output = next(message for message in state.threads["main"] if message.tool_call_id == "call_vision")
    assert tool_output.memory["server_tools"]["tool"] == VISION_TOOL_NAME
    assert tool_output.memory["server_tools"]["arguments"] == (f'{{"ids":["{image_id}"],"prompt":"describe"}}')


@pytest.mark.anyio
async def test_rejected_server_tool_attempt_is_stored_canonical_and_rebound_on_later_requests() -> None:
    core = _reload_handlers()
    client = _Client(
        streams=[
            _tool_step(
                ("call_vision", "vision_2", "{}"),
                usage=_usage(input_tokens=8, output_tokens=4),
            ),
            _text_step("fixed answer", usage=_usage(input_tokens=5, output_tokens=3)),
        ],
        completes=[],
    )
    state, _, _ = _state(
        client,
        request=_request(tools=[{"type": "function", "name": "vision", "parameters": {"type": "object"}}]),
    )

    await _run(state)

    assistant = next(message for message in state.threads["main"] if any(call.id == "call_vision" for call in message.tool_calls))
    assert assistant.tool_calls[0].name == VISION_TOOL_NAME
    assert assistant.memory["server_tools"]["call_ids"] == ["call_vision"]
    assert _tool_call_name(client.stream_requests[1].messages, "call_vision") == "vision_2"
    later_request = await core.response_request(state=state)
    assert _tool_call_name(later_request.messages, "call_vision") == "vision_2"


@pytest.mark.anyio
async def test_vision_server_tool_is_canonical_before_interrupted_progress_is_saved() -> None:
    _reload_handlers()
    image_id = _image_id(_image("https://example.com/cat.png"))
    client = _BlockingClient(
        _delta(
            tool_call_delta=ChatToolCallDelta(
                index=0,
                id="call_vision",
                name="vision_2",
                arguments_delta=f'{{"ids":["{image_id}"],"prompt":"describe"}}',
            )
        )
    )
    state, _, _ = _state(
        client,
        request=_request(tools=[{"type": "function", "name": "vision", "parameters": {"type": "object"}}]),
    )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(_run, state)
        await client.blocked.wait()
        task_group.cancel_scope.cancel()

    assistant = next(message for message in state.threads["main"] if message.is_assistant())
    assert assistant.tool_calls[0].name == VISION_TOOL_NAME
    assert assistant.memory["server_tools"]["call_ids"] == ["call_vision"]
    reasoning = next(item for item in _output_items(state) if isinstance(item, ResponseReasoningItem))
    payload = open_reasoning_payload(reasoning.encrypted_content, keyring=_keyring())
    persisted_assistant = next(message for message in payload.main if message.is_assistant())
    assert persisted_assistant.tool_calls[0].name == VISION_TOOL_NAME
    assert persisted_assistant.memory["server_tools"]["call_ids"] == ["call_vision"]


@pytest.mark.anyio
async def test_unknown_image_id_returns_failed_tool_output() -> None:
    _reload_handlers()
    image_id = _image_id(_image("https://example.com/cat.png"))
    client = _Client(
        streams=[
            _tool_step(
                ("call_vision_bad", VISION_TOOL_NAME, '{"ids":["image-AAAA-BBBB-CCCC-DDDD"],"prompt":"describe"}'),
                usage=_usage(input_tokens=8, output_tokens=4),
            ),
            _tool_step(
                ("call_vision", VISION_TOOL_NAME, f'{{"ids":["{image_id}"],"prompt":"describe"}}'),
                usage=_usage(input_tokens=8, output_tokens=4),
            ),
            _text_step("final answer", usage=_usage(input_tokens=5, output_tokens=3)),
        ],
        completes=[_complete("vision output", usage=_usage(input_tokens=6, output_tokens=2))],
    )
    state, _, _ = _state(client)

    await _run(state)

    assert len(client.stream_requests) == 3
    failed = next(message for message in client.stream_requests[1].messages if message.tool_call_id == "call_vision_bad")
    assert failed.content == "Unknown image IDs: image-AAAA-BBBB-CCCC-DDDD."
    assert failed.memory["vision"] == {"status": "failed", "reason": "unknown_image_ids"}
    assert failed.memory["server_tools"]["tool"] == VISION_TOOL_NAME
    assert len(client.complete_requests) == 1
    output = _output_items(state)
    assert not any(isinstance(item, ResponseFunctionCallItem) for item in output)
    final_message = next(item for item in output if isinstance(item, ResponseMessageItem))
    assert _message_text(final_message) == "final answer"


@pytest.mark.anyio
async def test_multiple_internal_tool_plugins_loop_without_leaking_calls() -> None:
    _reload_handlers()
    easy_server_tools = importlib.import_module("plap.plugins.easy.server_tools")

    @easy_server_tools.register
    class InternalTool(easy_server_tools.ServerTool):
        name: str = "internal"
        parameters: dict[str, object] | None = field(default_factory=lambda: {"type": "object"})
        strict: bool | None = True

        async def __call__(
            self,
            state: State,
            call: ChatToolCall,
        ) -> ChatMessage:
            _ = state
            return ChatMessage(role="tool", tool_call_id=call.id, content=f"{self.name}:{call.name}")

    image_id = _image_id(_image("https://example.com/cat.png"))
    client = _Client(
        streams=[
            _tool_step(
                ("call_vision", VISION_TOOL_NAME, f'{{"ids":["{image_id}"],"prompt":"describe"}}'),
                ("call_internal", "internal_2", "{}"),
                usage=_usage(input_tokens=8, output_tokens=4),
            ),
            _text_step("all internal done", usage=_usage(input_tokens=5, output_tokens=3)),
        ],
        completes=[_complete("vision output", usage=_usage(input_tokens=6, output_tokens=2))],
    )
    state, _, _ = _state(
        client,
        request=_request(tools=[{"type": "function", "name": "internal", "parameters": {"type": "object"}}]),
    )

    await _run(state)

    assert len(client.stream_requests) == 2
    internal_output = next(message for message in state.threads["main"] if message.tool_call_id == "call_internal")
    assert internal_output.content == "internal:internal"
    assert internal_output.memory["server_tools"] == {"tool": "internal", "arguments": "{}"}
    output = _output_items(state)
    assert not any(isinstance(item, ResponseFunctionCallItem) for item in output)
    final_message = next(item for item in output if isinstance(item, ResponseMessageItem))
    assert _message_text(final_message) == "all internal done"
