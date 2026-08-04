from __future__ import annotations

import importlib
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import field
from uuid import uuid4

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
    ChatContentText,
    ChatFinishReason,
    ChatMessage,
    ChatToolCall,
    ChatToolCallDelta,
    ChatToolChoiceMode,
    ChatUsage,
    IChatCompletionClient,
)
from plap.responses.contracts import ResponseCreateRequest
from plap.responses.contracts.items import (
    RequestFunctionCallItem,
    RequestFunctionCallOutputItem,
    RequestReasoningItem,
    ResponseFunctionCallItem,
    ResponseMessageItem,
    ResponseReasoningItem,
)
from plap.responses.ingest.ingest import ingest_response_request
from plap.responses.ingest.models import HiddenMainTail, Ingested, Message, Threads
from plap.responses.routes import _accept_response, _execute_response
from plap.responses.state import State
from plap.responses.store import PreparedRequest
from plap.responses.streaming import StreamCoordinator

_ADVISOR_THREAD = "advisor"
_ADVISE_TOOL_NAME = "advise"
_ADVISOR_TOOL_OUTPUT = "0"
_ABORTED_TOOL_OUTPUT = "Tool call cancelled by advisor."


def _has_advisor_marker(msg: ChatMessage) -> bool:
    return isinstance(msg.memory.get(_ADVISOR_THREAD), dict)


@pytest.fixture(autouse=True)
def restore_core_bus():
    yield
    bus.reset()
    core_module = importlib.import_module("plap.plugins.core.loop")
    importlib.reload(core_module)


def _advisor_module():
    return importlib.import_module("plap.plugins.advisor")


def _markdown_module():
    return importlib.import_module("plap.plugins.advisor.markdown")


def _summary_texts(state: State) -> list[str]:
    texts: list[str] = []
    for item in state.svcs.get(StreamCoordinator).current_response().output:
        if isinstance(item, ResponseReasoningItem):
            texts.extend(part.text for part in item.summary)
    return texts


class _RecordingChannels:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, object]]] = []

    async def wait_published(self, data: dict[str, object], channels: str | Sequence[str]) -> None:
        channel_names = [channels] if isinstance(channels, str) else list(channels)
        for channel_name in channel_names:
            self.published.append((channel_name, data))


class _RecordingStore:
    async def begin_response(self, prepared: PreparedRequest, response) -> None:
        _ = prepared, response

    async def append_output_item(self, prepared: PreparedRequest, response_id: str, output_index: int, item: object) -> None:
        _ = prepared, response_id, output_index, item

    async def replace_output_item(self, prepared: PreparedRequest, response_id: str, output_index: int, item: object) -> None:
        _ = prepared, response_id, output_index, item

    async def finish_response(self, prepared: PreparedRequest, response) -> None:
        _ = prepared, response

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
        main: list[list[ChatCompletionDelta]],
        advisor: list[list[ChatCompletionDelta]],
    ) -> None:
        self._main = list(main)
        self._advisor = list(advisor)
        self.main_requests: list[ChatCompletionRequest] = []
        self.advisor_requests: list[ChatCompletionRequest] = []

    async def complete(self, request: ChatCompletionRequest):  # pragma: no cover - advisor uses stream retry wrapper
        _ = request
        raise NotImplementedError

    def stream(self, request: ChatCompletionRequest):
        if request.model == "advisor-model":
            self.advisor_requests.append(request)
            if not self._advisor:  # pragma: no cover
                raise AssertionError("unexpected advisor request")
            deltas = self._advisor.pop(0)
        else:
            self.main_requests.append(request)
            if not self._main:  # pragma: no cover
                raise AssertionError("unexpected main request")
            deltas = self._main.pop(0)

        async def run():
            for delta in deltas:
                yield delta

        return run()

    async def aclose(self) -> None:
        return None


def _reload_handlers():
    bus.reset()
    core_module = importlib.import_module("plap.plugins.core.loop")
    server_tools_module = importlib.import_module("plap.plugins.easy.server_tools")
    advisor_module = importlib.import_module("plap.plugins.advisor")
    core_module = importlib.reload(core_module)
    importlib.reload(server_tools_module)
    importlib.reload(advisor_module)
    return core_module


def _keyring() -> SealingKeyring:
    return SealingKeyring(roots=(b"i" * 32,))


def _usage(*, input_tokens: int = 1, output_tokens: int = 1) -> ChatUsage:
    return ChatUsage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=input_tokens + output_tokens)


def _field(model: str) -> dict[str, object]:
    return {
        "model": model,
        "max_completion_tokens": 128,
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
    }


class _Config(Box):
    def resolve(self, request: dict[str, object] | None = None, /, **kwargs: object) -> _Config:
        _ = request, kwargs
        return self


def _config() -> _Config:
    return _Config(
        {
            "display_name": "Test Model",
            "main": _field("main-model"),
            "advisor": _field("advisor-model"),
            "reasoning_to_output": 1.0,
        },
        frozen_box=True,
    )


def _svcs(config: _Config | None = None) -> svcs.Container:
    registry = svcs.Registry()
    registry.register_value(SealingKeyring, _keyring())
    registry.register_value(CueBox, config or _config())
    return svcs.Container(registry)


def _tool() -> dict[str, object]:
    return {
        "type": "function",
        "name": "read_file",
        "description": "Read a file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _request(**updates: object) -> ResponseCreateRequest:
    tools = updates.pop("tools", [_tool()])
    input_ = updates.pop("input", "hello")
    return ResponseCreateRequest(model="plap-ai/test", input=input_, tools=tools, **updates)


def _prepared(request: ResponseCreateRequest | None = None) -> PreparedRequest:
    actual_request = request or _request()
    return PreparedRequest(
        scope_id=uuid4(),
        response_request=actual_request,
        execution_request=actual_request,
        stored_input_items=[],
    )


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
) -> State:
    actual_request = request or _request()
    config = _config()
    store = _RecordingStore()
    channels = _RecordingChannels()
    container = _svcs(config)
    budget = CompletionBudget(
        actual_request.max_output_tokens,
        reasoning_to_output=float(config.reasoning_to_output),
    )
    container.register_local_value(CompletionBudget, budget)
    container.register_local_value(BudgetedChatCompletionClient, BudgetedChatCompletionClient(client, budget))
    container.register_local_value(StreamCoordinator, _coordinator(store, channels, actual_request))
    return State.from_ingested(
        ingested=ingested
        or Ingested(
            memory={},
            threads=Threads(messages={"main": [Message(role="user", content="hello")]}),
            main_tail=None,
            last_reasoning_id=None,
        ),
        request=actual_request,
        config=config,
        svcs=container,
        thread_codes={"main": 0, _ADVISOR_THREAD: 1024},
    )


async def _run(state: State) -> None:
    await _accept_response(state)
    await _execute_response(state)


def _after_tool_ingested() -> Ingested:
    assistant = Message(
        role="assistant",
        tool_calls=[ChatToolCall(id="call_read", name="read_file", arguments='{"path":"src/app.py"}')],
    )
    return Ingested(
        memory={},
        threads=Threads(
            messages={
                "main": [
                    Message(role="user", content="hello"),
                    assistant,
                    Message(role="tool", tool_call_id="call_read", content="file contents"),
                ]
            }
        ),
        main_tail=HiddenMainTail(source=assistant),
        last_reasoning_id=None,
    )


def _delta(
    *,
    model: str,
    content_delta: str | None = None,
    reasoning_delta: str | None = None,
    tool_call_delta: ChatToolCallDelta | None = None,
    finish_reason: ChatFinishReason | None = None,
    usage: ChatUsage | None = None,
) -> ChatCompletionDelta:
    return ChatCompletionDelta(
        id=f"cmpl_{model}",
        model=model,
        created_at=None,
        choice_index=0,
        content_delta=content_delta,
        reasoning_delta=reasoning_delta,
        tool_call_delta=tool_call_delta,
        finish_reason=finish_reason,
        usage=usage,
        service_tier="default",
    )


def _text_step(text: str) -> list[ChatCompletionDelta]:
    return [_delta(model="main-model", content_delta=text, finish_reason=ChatFinishReason.STOP, usage=_usage())]


def _tool_step(call_id: str = "call_read") -> list[ChatCompletionDelta]:
    return [
        _delta(
            model="main-model",
            tool_call_delta=ChatToolCallDelta(index=0, id=call_id, name="read_file", arguments_delta='{"path":"src/app.py"}'),
        ),
        _delta(model="main-model", finish_reason=ChatFinishReason.TOOL_CALLS, usage=_usage()),
    ]


def _advisor_calls_step(
    *calls: tuple[str, str, dict[str, object]],
    reasoning: str | None = None,
) -> list[ChatCompletionDelta]:
    deltas = [] if reasoning is None else [_delta(model="advisor-model", reasoning_delta=reasoning)]
    deltas.extend(
        [
            _delta(
                model="advisor-model",
                tool_call_delta=ChatToolCallDelta(
                    index=index,
                    id=call_id,
                    name=name,
                    arguments_delta=msgspec.json.encode(arguments).decode(),
                ),
            )
            for index, (call_id, name, arguments) in enumerate(calls)
        ]
    )
    deltas.append(_delta(model="advisor-model", finish_reason=ChatFinishReason.TOOL_CALLS, usage=_usage()))
    return deltas


def _advisor_step(
    advice: str | None = "",
    *,
    note: str | None = None,
    tool_name: str = _ADVISE_TOOL_NAME,
) -> list[ChatCompletionDelta]:
    arguments: dict[str, str] = {"advice": advice or ""}
    if note is not None:
        arguments["note"] = note
    return _advisor_calls_step(("call_advise", tool_name, arguments))


def _main_output_ingested(state: State, output: ChatMessage) -> Ingested:
    threads = deepcopy(state.threads)
    threads["main"].append(output)
    main_source = next(message for message in reversed(threads["main"]) if message.is_assistant())
    return Ingested(
        memory=deepcopy(state.memory),
        threads=threads,
        main_tail=HiddenMainTail(source=main_source),
        last_reasoning_id=None,
    )


def _register_inspect_tool() -> None:
    easy_server_tools = importlib.import_module("plap.plugins.easy.server_tools")

    @easy_server_tools.register
    class InspectTool(easy_server_tools.ServerTool):
        name: str = "inspect"
        parameters: dict[str, object] = field(
            default_factory=lambda: {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            }
        )
        strict: bool = True

        async def __call__(self, state: State, call: ChatToolCall) -> ChatMessage:
            _ = state
            return ChatMessage(role="tool", tool_call_id=call.id, content=f"inspected:{call.arguments}")


def _bad_advisor_step() -> list[ChatCompletionDelta]:
    return [_delta(model="advisor-model", content_delta="no tool", finish_reason=ChatFinishReason.STOP, usage=_usage())]


def _expensive_bad_advisor_step() -> list[ChatCompletionDelta]:
    return [_delta(model="advisor-model", content_delta="no tool", finish_reason=ChatFinishReason.STOP, usage=_usage(output_tokens=2))]


def test_assistant_markdown_is_compact_and_strips_tool_call_ids() -> None:
    message = ChatMessage(
        role="assistant",
        content="I'll read it.",
        reasoning_content="Need the file.",
        tool_calls=[ChatToolCall(id="call_secret", name="read_file", arguments='{"path":"src/app.py"}')],
    )

    rendered = _markdown_module().assistant_markdown(message)

    assert rendered.startswith("## assistant\n")
    assert "### reasoning_content\n```text\nNeed the file.\n```" in rendered
    assert "### content\n```text\nI'll read it.\n```" in rendered
    assert '### tool_call read_file\n```json\n{"path":"src/app.py"}\n```' in rendered
    assert "call_secret" not in rendered


def test_tool_outputs_render_in_assistant_call_order_without_ids() -> None:
    assistant = ChatMessage(
        role="assistant",
        tool_calls=[
            ChatToolCall(id="call_a", name="read_file", arguments='{"path":"a"}'),
            ChatToolCall(id="call_b", name="read_file", arguments='{"path":"b"}'),
        ],
    )
    history = [
        assistant,
        ChatMessage(role="tool", tool_call_id="call_b", content="B"),
        ChatMessage(role="tool", tool_call_id="call_a", content="A"),
    ]

    markdown = _markdown_module()
    turn = markdown.latest_closed_tool_output_turn(history)

    assert turn is not None
    rendered = markdown.tool_outputs_markdown(turn)
    assert rendered.startswith("## tool\n")
    assert rendered.count("### tool_output read_file") == 2
    assert "### tool_output read_file\n```text\nA\n```" in rendered
    assert "### tool_output read_file\n```text\nB\n```" in rendered
    assert rendered.index("### tool_output read_file\n```text\nA\n```") < rendered.index("### tool_output read_file\n```text\nB\n```")


def test_advisor_main_updates_are_incremental_and_reanchor_to_latest_turn() -> None:
    _reload_handlers()
    state = _state(_Client(main=[], advisor=[]))
    advisor = _advisor_module()

    advisor._append_main_update(state)
    state.threads["main"].extend(
        [
            ChatMessage(
                role="developer",
                content="old advisor guidance",
                memory={_ADVISOR_THREAD: {"artifact": True}},
            ),
            ChatMessage(role="assistant", content="new answer"),
        ]
    )
    advisor._append_main_update(state)

    updates = [message for message in state.threads[_ADVISOR_THREAD] if "transcript_anchor" in message.memory.get(_ADVISOR_THREAD, {})]
    assert len(updates) == 2
    assert "new answer" in updates[-1].content
    assert "old advisor guidance" not in updates[-1].content

    state.threads["main"] = [
        ChatMessage(role="user", content="replacement context"),
        ChatMessage(role="assistant", content="latest answer"),
    ]
    advisor._append_main_update(state)

    updates = [message for message in state.threads[_ADVISOR_THREAD] if "transcript_anchor" in message.memory.get(_ADVISOR_THREAD, {})]
    assert len(updates) == 3
    assert "latest answer" in updates[-1].content
    assert "replacement context" not in updates[-1].content


def test_incremental_main_tool_output_keeps_its_tool_name_without_repeating_call() -> None:
    _reload_handlers()
    state = _state(_Client(main=[], advisor=[]))
    state.threads["main"].append(
        ChatMessage(
            role="assistant",
            tool_calls=[ChatToolCall(id="call_read", name="read_file", arguments='{"path":"src/app.py"}')],
        )
    )
    advisor = _advisor_module()
    advisor._append_main_update(state)
    state.threads["main"].append(ChatMessage(role="tool", tool_call_id="call_read", content="file contents"))

    advisor._append_main_update(state)

    update = state.threads[_ADVISOR_THREAD][-1]
    assert "### tool_output read_file" in update.content
    assert "file contents" in update.content
    assert "### tool_call read_file" not in update.content


@pytest.mark.anyio
async def test_before_tool_noop_returns_function_call() -> None:
    _reload_handlers()
    client = _Client(main=[_tool_step()], advisor=[_advisor_step("")])
    state = _state(client)

    await _run(state)

    output = state.svcs.get(StreamCoordinator).current_response().output
    assert any(isinstance(item, ResponseFunctionCallItem) for item in output)
    assert len(client.advisor_requests) == 1
    advisor_request = client.advisor_requests[0]
    assert [tool.function.name for tool in advisor_request.tools] == [_ADVISE_TOOL_NAME, "read_file"]
    assert advisor_request.tool_choice == ChatToolChoiceMode.REQUIRED
    assert advisor_request.parallel_tool_calls is True
    assert "### tool_call read_file" in advisor_request.messages[-2].content
    assert state.threads[_ADVISOR_THREAD][-1].role == "tool"
    assert state.threads[_ADVISOR_THREAD][-1].tool_call_id == "call_advise"


@pytest.mark.anyio
async def test_advisor_rebinds_advise_without_renaming_client_tool() -> None:
    _reload_handlers()
    advise_tool = {
        "type": "function",
        "name": _ADVISE_TOOL_NAME,
        "parameters": {"type": "object"},
    }
    request = _request(tools=[advise_tool, _tool()])
    client = _Client(main=[_tool_step()], advisor=[_advisor_step("", tool_name="advise_2")])
    state = _state(client, request=request)

    await _run(state)

    assert [tool.function.name for tool in client.main_requests[0].tools] == [_ADVISE_TOOL_NAME, "read_file"]
    advisor_request = client.advisor_requests[0]
    assert [tool.function.name for tool in advisor_request.tools] == ["advise_2", _ADVISE_TOOL_NAME, "read_file"]
    assert advisor_request.tool_choice == ChatToolChoiceMode.REQUIRED


@pytest.mark.anyio
async def test_advisor_executes_parallel_server_tools_before_advise() -> None:
    _reload_handlers()
    _register_inspect_tool()
    client = _Client(
        main=[_tool_step()],
        advisor=[
            _advisor_calls_step(
                ("call_inspect_a", "inspect", {"value": "a"}),
                ("call_inspect_b", "inspect", {"value": "b"}),
                reasoning="Inspect both independent questions.",
            ),
            _advisor_step(""),
        ],
    )
    state = _state(client)

    await _run(state)

    assert len(client.advisor_requests) == 2
    assert client.advisor_requests[0].parallel_tool_calls is True
    second_request = client.advisor_requests[1]
    assert sum("phase" in message.memory.get(_ADVISOR_THREAD, {}) for message in second_request.messages) == 1
    outputs = [message for message in second_request.messages if message.role == "tool"]
    assert [message.tool_call_id for message in outputs[-2:]] == ["call_inspect_a", "call_inspect_b"]
    assert all(message.memory["server_tools"]["tool"] == "inspect" for message in outputs[-2:])
    assert any(message.reasoning_content == "Inspect both independent questions." for message in second_request.messages)
    assert _ADVISOR_THREAD not in state.threads.active
    assert "main" in state.threads.active
    assert not any("phase" in message.memory.get(_ADVISOR_THREAD, {}) for message in state.threads[_ADVISOR_THREAD])


@pytest.mark.anyio
async def test_advisor_client_tool_continuation_keeps_main_frozen_until_advise() -> None:
    _reload_handlers()
    _register_inspect_tool()
    first_client = _Client(
        main=[_tool_step()],
        advisor=[
            _advisor_calls_step(
                ("call_inspect", "inspect", {"value": "server"}),
                ("call_probe", "read_file", {"path": "src/app.py"}),
            )
        ],
    )
    first_state = _state(first_client)

    await _run(first_state)

    first_output = first_state.svcs.get(StreamCoordinator).current_response().output
    visible_calls = [item for item in first_output if isinstance(item, ResponseFunctionCallItem)]
    assert [item.name for item in visible_calls] == ["read_file"]
    assert first_state.threads.active == {_ADVISOR_THREAD}
    phase_messages = [message for message in first_state.threads[_ADVISOR_THREAD] if "phase" in message.memory.get(_ADVISOR_THREAD, {})]
    assert len(phase_messages) == 1
    assert first_client.main_requests and len(first_client.main_requests) == 1

    reasoning = next(item for item in first_output if isinstance(item, ResponseReasoningItem))
    continuation_request = _request(
        input=[
            RequestReasoningItem.model_validate(reasoning.model_dump()),
            RequestFunctionCallItem.model_validate(visible_calls[0].model_dump()),
            RequestFunctionCallOutputItem(
                call_id=visible_calls[0].call_id,
                output="client inspection",
                type="function_call_output",
            ),
        ]
    )
    ingested = await ingest_response_request(
        continuation_request,
        keyring=_keyring(),
        thread_codes={"main": 0, _ADVISOR_THREAD: 1024},
    )
    second_client = _Client(main=[], advisor=[_advisor_step("")])
    second_state = _state(
        second_client,
        request=continuation_request,
        ingested=ingested,
    )

    await _run(second_state)

    assert second_client.main_requests == []
    assert len(second_client.advisor_requests) == 1
    assert any(message.tool_call_id == "call_probe" for message in second_client.advisor_requests[0].messages)
    assert any(
        message.tool_call_id == "call_probe" and message.content == "client inspection" for message in second_state.threads[_ADVISOR_THREAD]
    )
    assert any(message.tool_call_id == "call_inspect" for message in second_state.threads[_ADVISOR_THREAD])
    assert second_state.threads.active == {"main"}
    assert not any("phase" in message.memory.get(_ADVISOR_THREAD, {}) for message in second_state.threads[_ADVISOR_THREAD])
    second_output = second_state.svcs.get(StreamCoordinator).current_response().output
    assert [item.name for item in second_output if isinstance(item, ResponseFunctionCallItem)] == ["read_file"]


@pytest.mark.anyio
async def test_advisor_parallel_client_tool_outputs_resume_exploration() -> None:
    _reload_handlers()
    first_client = _Client(
        main=[_tool_step()],
        advisor=[
            _advisor_calls_step(
                ("call_probe_a", "read_file", {"path": "src/a.py"}),
                ("call_probe_b", "read_file", {"path": "src/b.py"}),
            )
        ],
    )
    first_state = _state(first_client)

    await _run(first_state)

    first_output = first_state.svcs.get(StreamCoordinator).current_response().output
    reasoning = next(item for item in first_output if isinstance(item, ResponseReasoningItem))
    visible_calls = [item for item in first_output if isinstance(item, ResponseFunctionCallItem)]
    assert [item.name for item in visible_calls] == ["read_file", "read_file"]
    continuation_request = _request(
        input=[
            RequestReasoningItem.model_validate(reasoning.model_dump()),
            *[RequestFunctionCallItem.model_validate(item.model_dump()) for item in visible_calls],
            *[
                RequestFunctionCallOutputItem(
                    call_id=item.call_id,
                    output=f"client inspection {index}",
                    type="function_call_output",
                )
                for index, item in enumerate(visible_calls)
            ],
        ]
    )
    ingested = await ingest_response_request(
        continuation_request,
        keyring=_keyring(),
        thread_codes={"main": 0, _ADVISOR_THREAD: 1024},
    )
    second_client = _Client(main=[], advisor=[_advisor_step("")])
    second_state = _state(
        second_client,
        request=continuation_request,
        ingested=ingested,
    )

    await _run(second_state)

    advisor_outputs = [message for message in second_state.threads[_ADVISOR_THREAD] if message.role == "tool"]
    assert any(message.tool_call_id == "call_probe_a" and message.content == "client inspection 0" for message in advisor_outputs)
    assert any(message.tool_call_id == "call_probe_b" and message.content == "client inspection 1" for message in advisor_outputs)
    assert len(second_client.advisor_requests) == 1
    assert second_state.threads.active == {"main"}


@pytest.mark.anyio
async def test_main_client_output_starts_after_tool_phase_with_named_incremental_output() -> None:
    _reload_handlers()
    first_client = _Client(main=[_tool_step()], advisor=[_advisor_step("")])
    first_state = _state(first_client)
    await _run(first_state)

    second_client = _Client(
        main=[_text_step("final answer")],
        advisor=[_advisor_step(""), _advisor_step("")],
    )
    second_state = _state(
        second_client,
        ingested=_main_output_ingested(
            first_state,
            ChatMessage(role="tool", tool_call_id="call_read", content="file contents"),
        ),
    )

    await _run(second_state)

    after_tool_request = second_client.advisor_requests[0]
    transcript = next(
        message for message in reversed(after_tool_request.messages) if "transcript_anchor" in message.memory.get(_ADVISOR_THREAD, {})
    )
    assert "### tool_output read_file" in transcript.content
    assert "file contents" in transcript.content
    assert "### tool_call read_file" not in transcript.content


@pytest.mark.anyio
async def test_advisor_retries_advise_mixed_with_exploration_calls() -> None:
    _reload_handlers()
    client = _Client(
        main=[_tool_step()],
        advisor=[
            _advisor_calls_step(
                ("call_bad_advise", _ADVISE_TOOL_NAME, {"advice": ""}),
                ("call_probe", "read_file", {"path": "src/app.py"}),
            ),
            _advisor_step(""),
        ],
    )
    state = _state(client)

    await _run(state)

    assert len(client.advisor_requests) == 2
    assert any(
        message.role == "user" and isinstance(message.content, str) and "final `advise` call was mixed" in message.content
        for message in state.threads[_ADVISOR_THREAD]
    )
    output = state.svcs.get(StreamCoordinator).current_response().output
    assert [item.name for item in output if isinstance(item, ResponseFunctionCallItem)] == ["read_file"]


@pytest.mark.anyio
async def test_advisor_retry_limit_skips_current_phase() -> None:
    _reload_handlers()
    client = _Client(main=[_tool_step()], advisor=[_bad_advisor_step(), _bad_advisor_step(), _bad_advisor_step()])
    state = _state(client)

    await _run(state)

    output = state.svcs.get(StreamCoordinator).current_response().output
    assert any(isinstance(item, ResponseFunctionCallItem) for item in output)
    assert len(client.advisor_requests) == 3
    thread = state.threads.get(_ADVISOR_THREAD)
    assert thread is not None
    assert all(msg.role == "user" for msg in thread)
    assert state.threads.active == {"main"}
    assert not any("phase" in message.memory.get(_ADVISOR_THREAD, {}) for message in thread)


@pytest.mark.anyio
async def test_advisor_retry_hidden_usage_caps_next_attempt() -> None:
    _reload_handlers()
    request = _request(max_output_tokens=5)
    client = _Client(main=[_tool_step()], advisor=[_expensive_bad_advisor_step()])
    state = _state(client, request=request)

    await _run(state)

    output = state.svcs.get(StreamCoordinator).current_response().output
    assert any(isinstance(item, ResponseFunctionCallItem) for item in output)
    assert len(client.advisor_requests) == 1
    thread = state.threads.get(_ADVISOR_THREAD)
    assert thread is not None
    assert all(msg.role == "user" for msg in thread)
    assert state.threads.active == {"main"}
    assert not any("phase" in message.memory.get(_ADVISOR_THREAD, {}) for message in thread)


@pytest.mark.anyio
async def test_before_tool_advice_aborts_call_and_loops_to_final_answer() -> None:
    _reload_handlers()
    client = _Client(
        main=[_tool_step(), _text_step("final answer")],
        advisor=[_advisor_step("Do not read that file."), _advisor_step("")],
    )
    state = _state(client)

    await _run(state)

    output = state.svcs.get(StreamCoordinator).current_response().output
    assert not any(isinstance(item, ResponseFunctionCallItem) for item in output)
    message = next(item for item in output if isinstance(item, ResponseMessageItem))
    assert message.content[0].text == "final answer"
    assert len(client.main_requests) == 2
    second_main = client.main_requests[1]
    aborted = next(message for message in second_main.messages if message.role == "tool" and message.content == _ABORTED_TOOL_OUTPUT)
    assert aborted.name is None
    assert aborted.memory == {_ADVISOR_THREAD: {"artifact": True}}
    assert any(
        message.role == "developer" and message.content == "Do not read that file." and _has_advisor_marker(message)
        for message in second_main.messages
    )
    second_advisor = client.advisor_requests[1]
    assert "### tool_output read_file" not in second_advisor.messages[-2].content
    assert '{"advisor":"call_advise"}' not in second_advisor.messages[-2].content


@pytest.mark.anyio
async def test_advisor_retry_history_persists_across_phases() -> None:
    _reload_handlers()
    client = _Client(
        main=[_tool_step(), _text_step("final answer")],
        advisor=[_bad_advisor_step(), _advisor_step("Do not read that file."), _advisor_step("")],
    )
    state = _state(client)

    await _run(state)

    assert len(client.advisor_requests) == 3
    second_phase_request = client.advisor_requests[2]
    assert any(message.role == "assistant" and message.content == "no tool" for message in second_phase_request.messages)
    assert any(
        message.role == "user"
        and isinstance(message.content, str)
        and "Your previous answer could not be used as written." in message.content
        for message in second_phase_request.messages
    )


@pytest.mark.anyio
async def test_advisor_note_is_retained_in_advisor_history_without_global_memory() -> None:
    _reload_handlers()
    client = _Client(
        main=[_tool_step()],
        advisor=[_advisor_step("", note="Watch whether the final answer is actually verified.")],
    )
    state = _state(client)

    await _run(state)

    assert len(client.advisor_requests) == 1
    assert _ADVISOR_THREAD not in state.memory
    advise_call = next(call for message in state.threads[_ADVISOR_THREAD] for call in message.tool_calls if call.name == _ADVISE_TOOL_NAME)
    arguments = msgspec.json.decode(advise_call.arguments)
    assert arguments == {"advice": "", "note": "Watch whether the final answer is actually verified."}
    assert not any("phase" in message.memory.get(_ADVISOR_THREAD, {}) for message in state.threads[_ADVISOR_THREAD])


@pytest.mark.anyio
async def test_after_tool_advice_reaches_next_main_request() -> None:
    _reload_handlers()
    client = _Client(main=[_text_step("final answer")], advisor=[_advisor_step("Use the tool output."), _advisor_step("")])
    state = _state(
        client,
        ingested=_after_tool_ingested(),
    )

    await _run(state)

    assert "### tool_output read_file" in client.advisor_requests[0].messages[-2].content
    assert any(
        message.role == "developer" and message.content == "Use the tool output." and _has_advisor_marker(message)
        for message in client.main_requests[0].messages
    )


@pytest.mark.anyio
async def test_after_tool_advice_emits_summary_annotation_when_not_stealth(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_handlers()
    monkeypatch.setattr("plap.plugins.advisor.STEALTH", False)
    client = _Client(main=[_text_step("final answer")], advisor=[_advisor_step("Use the tool output."), _advisor_step("")])
    state = _state(
        client,
        ingested=_after_tool_ingested(),
    )

    await _run(state)

    assert "[advisor] advice: Use the tool output." in _summary_texts(state)


@pytest.mark.anyio
async def test_after_tool_note_emits_summary_annotation_when_not_stealth(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_handlers()
    monkeypatch.setattr("plap.plugins.advisor.STEALTH", False)
    client = _Client(
        main=[_text_step("final answer")],
        advisor=[_advisor_step("", note="Watch whether the tools/list call appears next."), _advisor_step("")],
    )
    state = _state(
        client,
        ingested=_after_tool_ingested(),
    )

    await _run(state)

    assert "[advisor] note: Watch whether the tools/list call appears next." in _summary_texts(state)


@pytest.mark.anyio
async def test_before_return_advice_loops_and_hides_first_answer() -> None:
    _reload_handlers()
    client = _Client(
        main=[_text_step("first answer"), _text_step("revised answer")],
        advisor=[_advisor_step("Revise before returning."), _advisor_step("")],
    )
    state = _state(client)

    await _run(state)

    output = state.svcs.get(StreamCoordinator).current_response().output
    message = next(item for item in output if isinstance(item, ResponseMessageItem))
    assert message.content[0].text == "revised answer"
    assert len(client.main_requests) == 2
    assert any(
        message.role == "developer" and message.content == "Revise before returning." and _has_advisor_marker(message)
        for message in client.main_requests[1].messages
    )
    assert "first answer" in client.advisor_requests[0].messages[-2].content
    assert "Revise before returning." not in client.advisor_requests[1].messages[-2].content


@pytest.mark.anyio
async def test_before_return_advice_emits_summary_annotation_when_not_stealth(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_handlers()
    monkeypatch.setattr("plap.plugins.advisor.STEALTH", False)
    client = _Client(
        main=[_text_step("first answer"), _text_step("revised answer")],
        advisor=[_advisor_step("Revise before returning."), _advisor_step("")],
    )
    state = _state(client)

    await _run(state)

    assert "[advisor] blocked return. advice: Revise before returning." in _summary_texts(state)


@pytest.mark.anyio
async def test_before_return_note_only_emits_neutral_summary_annotation_when_not_stealth(monkeypatch: pytest.MonkeyPatch) -> None:
    _reload_handlers()
    monkeypatch.setattr("plap.plugins.advisor.STEALTH", False)
    client = _Client(
        main=[_text_step("first answer")],
        advisor=[_advisor_step("", note="All good. Agent read the file and compile passed.")],
    )
    state = _state(client)

    await _run(state)

    texts = _summary_texts(state)
    assert "[advisor] note: All good. Agent read the file and compile passed." in texts
    assert not any(text.startswith("[advisor] blocked return.") for text in texts)


def test_content_part_serialization_uses_json_fence() -> None:
    rendered = _markdown_module().assistant_markdown(ChatMessage(role="assistant", content=[ChatContentText(text="part")]))

    assert rendered.startswith("## assistant\n### content\n```json\n")
    assert '[{"text":"part","type":"text"}]' in rendered
    assert rendered.endswith("\n```")


def test_render_main_messages_includes_all_roles() -> None:
    messages = [
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="assistant", content="ok", tool_calls=[ChatToolCall(id="c1", name="t1", arguments="{}")]),
        ChatMessage(role="tool", tool_call_id="c1", content="output"),
        ChatMessage(role="user", content="more"),
    ]

    lines = _markdown_module().render_main_messages(messages)
    rendered = "\n".join(lines)

    assert "## user\n### content\n```text\nhello\n```" in rendered
    assert "## assistant\n### content\n```text\nok\n```" in rendered
    assert "### tool_call t1\n```json\n{}" in rendered
    assert "## tool\n### tool_output t1\n```text\noutput\n```" in rendered
    assert "## user\n### content\n```text\nmore\n```" in rendered


def test_render_main_messages_does_not_emit_message_memory() -> None:
    messages = [
        ChatMessage(
            role="tool",
            name="read_file",
            tool_call_id="call_1",
            content="output",
            memory={"advisor": {"artifact": True}},
        ),
        ChatMessage(role="developer", content="note", memory={"advisor": {"transcript_anchor": "hash"}}),
    ]

    rendered = "\n".join(_markdown_module().render_main_messages(messages))

    assert "artifact" not in rendered
    assert "transcript_anchor" not in rendered


def test_requirements_instruction_renders_effective_defaults() -> None:
    request = ChatCompletionRequest(model="main-model", messages=[])

    rendered = _markdown_module().requirements_instruction(request)

    assert rendered.startswith("# requirements\n```json\n")
    assert '"tool_choice":"auto"' in rendered
    assert '"parallel_tool_calls":true' in rendered
    assert '"response_format":null' in rendered
    assert rendered.endswith("\n```")
