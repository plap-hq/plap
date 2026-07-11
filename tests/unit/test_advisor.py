from __future__ import annotations

import importlib
from collections.abc import Sequence
from types import SimpleNamespace
from uuid import uuid4

import msgspec
import pytest
import svcs
from box import Box

from plap.bus import bus
from plap.config import CueBox
from plap.keyring import SealingKeyring
from plap.llms.completions.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatContentText,
    ChatFinishReason,
    ChatMessage,
    ChatToolCall,
    ChatToolCallDelta,
    ChatUsage,
    IChatCompletionClient,
)
from plap.plugins.advisor import advise_response
from plap.plugins.core.ledger import UsageLedger
from plap.responses.contracts import ResponseCreateRequest
from plap.responses.contracts.items import ResponseFunctionCallItem, ResponseMessageItem, ResponseReasoningItem
from plap.responses.ingest.models import MAIN_SIDE, Ingested, Message, Sides
from plap.responses.state import State
from plap.responses.store import PreparedRequest
from plap.responses.streaming import StreamCoordinator

_ADVISOR_SIDE = "advisor"
_ADVISE_TOOL_NAME = "advise"
_ADVISOR_TOOL_OUTPUT = "0"
_ABORTED_TOOL_OUTPUT = "Tool call cancelled by advisor."


def _has_advisor_marker(msg: ChatMessage) -> bool:
    if msg.reasoning_content is None:
        return False
    try:
        data = msgspec.json.decode(msg.reasoning_content)
        return isinstance(data, dict) and "advisor" in data
    except Exception:
        return False


def _advisor_sentinel(value: str | bool) -> str:
    return msgspec.json.encode({"advisor": value}).decode()


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
    for item in state.coordinator.current_response().output:
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

    async def fail_response(self, prepared: PreparedRequest, response_id: str) -> bool:
        _ = prepared, response_id
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
    advisor_module = importlib.import_module("plap.plugins.advisor")
    core_module = importlib.reload(core_module)
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
        "public_usage": {
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


def _loaded(config: _Config | None = None) -> object:
    return SimpleNamespace(plap=SimpleNamespace(config=config or _config()))


def _svcs(client: IChatCompletionClient, config: _Config | None = None) -> svcs.Container:
    registry = svcs.Registry()
    registry.register_value(SealingKeyring, _keyring())
    registry.register_value(CueBox, _loaded(config))
    registry.register_value(IChatCompletionClient, client)
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
    return ResponseCreateRequest(model="plap-ai/test", input="hello", tools=[_tool()], **updates)


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
    store = _RecordingStore()
    channels = _RecordingChannels()
    return State.from_ingested(
        ingested=ingested
        or Ingested(
            machine={},
            sides=Sides(messages={MAIN_SIDE: [Message(role="user", content="hello")]}),
            last_reasoning_id=None,
            current_compaction_id=None,
        ),
        prepared=_prepared(actual_request),
        svcs=_svcs(client),
        coordinator=_coordinator(store, channels, actual_request),
        sealing_keyring=_keyring(),
        side_codes={MAIN_SIDE: 0, _ADVISOR_SIDE: 1},
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


@pytest.mark.anyio
async def test_advisor_delegates_without_work_when_main_is_inactive() -> None:
    client = _Client(main=[], advisor=[])
    state = _state(client)
    state.deactivate(MAIN_SIDE)
    delegated = 0

    async def next_handler(**kwargs):
        nonlocal delegated
        _ = kwargs
        delegated += 1

    result = await advise_response(
        state=state,
        config=state.svcs.get(CueBox).plap.config,
        ledger=UsageLedger(budget=None, reasoning_to_output=1.0),
        next=next_handler,
    )

    assert result is None
    assert delegated == 1
    assert client.advisor_requests == []


def _advisor_step(advice: str | None = "", *, note: str | None = None) -> list[ChatCompletionDelta]:
    arguments: dict[str, str] = {}
    if advice is not None:
        arguments["advice"] = advice
    if note is not None:
        arguments["note"] = note
    return [
        _delta(
            model="advisor-model",
            tool_call_delta=ChatToolCallDelta(
                index=0,
                id="call_advise",
                name=_ADVISE_TOOL_NAME,
                arguments_delta=msgspec.json.encode(arguments).decode(),
            ),
        ),
        _delta(model="advisor-model", finish_reason=ChatFinishReason.TOOL_CALLS, usage=_usage()),
    ]


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


@pytest.mark.anyio
async def test_before_tool_noop_returns_function_call() -> None:
    core = _reload_handlers()
    client = _Client(main=[_tool_step()], advisor=[_advisor_step("")])
    state = _state(client)

    await core.run_response(state=state)

    output = state.coordinator.current_response().output
    assert any(isinstance(item, ResponseFunctionCallItem) for item in output)
    assert len(client.advisor_requests) == 1
    advisor_request = client.advisor_requests[0]
    assert [tool.function.name for tool in advisor_request.tools] == [_ADVISE_TOOL_NAME, "read_file"]
    assert advisor_request.tool_choice.name == _ADVISE_TOOL_NAME
    assert "### tool_call read_file" in advisor_request.messages[-2].content
    assert state.sides[_ADVISOR_SIDE][-1].role == "tool"
    assert state.sides[_ADVISOR_SIDE][-1].tool_call_id == "call_advise"


@pytest.mark.anyio
async def test_advisor_retry_limit_skips_current_phase() -> None:
    core = _reload_handlers()
    client = _Client(main=[_tool_step()], advisor=[_bad_advisor_step(), _bad_advisor_step(), _bad_advisor_step()])
    state = _state(client)

    await core.run_response(state=state)

    output = state.coordinator.current_response().output
    assert any(isinstance(item, ResponseFunctionCallItem) for item in output)
    assert len(client.advisor_requests) == 3
    side = state.sides.get(_ADVISOR_SIDE)
    assert side is not None
    assert all(msg.role == "user" for msg in side)


@pytest.mark.anyio
async def test_advisor_retry_hidden_usage_caps_next_attempt() -> None:
    core = _reload_handlers()
    request = _request(max_output_tokens=1)
    client = _Client(main=[_tool_step()], advisor=[_expensive_bad_advisor_step()])
    state = _state(client, request=request)

    await core.run_response(state=state)

    output = state.coordinator.current_response().output
    assert any(isinstance(item, ResponseFunctionCallItem) for item in output)
    assert len(client.advisor_requests) == 1
    side = state.sides.get(_ADVISOR_SIDE)
    assert side is not None
    assert all(msg.role == "user" for msg in side)


@pytest.mark.anyio
async def test_before_tool_advice_aborts_call_and_loops_to_final_answer() -> None:
    core = _reload_handlers()
    client = _Client(
        main=[_tool_step(), _text_step("final answer")],
        advisor=[_advisor_step("Do not read that file."), _advisor_step("")],
    )
    state = _state(client)

    await core.run_response(state=state)

    output = state.coordinator.current_response().output
    assert not any(isinstance(item, ResponseFunctionCallItem) for item in output)
    message = next(item for item in output if isinstance(item, ResponseMessageItem))
    assert message.content[0].text == "final answer"
    assert len(client.main_requests) == 2
    second_main = client.main_requests[1]
    assert any(
        message.role == "tool" and message.content == _ABORTED_TOOL_OUTPUT and message.name == "read_file" and _has_advisor_marker(message)
        for message in second_main.messages
    )
    assert any(
        message.role == "developer" and message.content == "Do not read that file." and _has_advisor_marker(message)
        for message in second_main.messages
    )
    second_advisor = client.advisor_requests[1]
    assert '{"advisor":"call_advise"}' not in second_advisor.messages[-2].content


@pytest.mark.anyio
async def test_advisor_retry_history_block_survives_rebuild() -> None:
    core = _reload_handlers()
    client = _Client(
        main=[_tool_step(), _text_step("final answer")],
        advisor=[_bad_advisor_step(), _advisor_step("Do not read that file."), _advisor_step("")],
    )
    state = _state(client)

    await core.run_response(state=state)

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
async def test_advisor_note_is_sent_to_next_turn_scrubbed_and_cleared() -> None:
    core = _reload_handlers()
    client = _Client(
        main=[_tool_step(), _text_step("final answer")],
        advisor=[_advisor_step(None, note="Watch whether the final answer is actually verified."), _advisor_step("")],
    )
    state = _state(client)

    await core.run_response(state=state)

    assert len(client.advisor_requests) == 1
    machine = state.machine.to_primitive().get(_ADVISOR_SIDE, {})
    assert isinstance(machine, dict)
    assert machine.get("note") == "Watch whether the final answer is actually verified."
    main_request = await core.response_request(state=state, config=state.svcs.get(CueBox).plap.config)
    phase_instruction = _advisor_module()._phase_instruction(state, "before_return", main_request)
    assert "# note from previous phase (may be stale)" in phase_instruction
    assert "Watch whether the final answer is actually verified." in phase_instruction
    assert not any(
        message.role == "assistant" and message.tool_calls and "note" in message.tool_calls[0].arguments
        for message in state.sides[_ADVISOR_SIDE]
    )
    _advisor_module()._set_advisor_note(state, None)
    machine = state.machine.to_primitive().get(_ADVISOR_SIDE, {})
    assert not isinstance(machine, dict) or "note" not in machine


@pytest.mark.anyio
async def test_after_tool_advice_reaches_next_main_request() -> None:
    core = _reload_handlers()
    client = _Client(main=[_text_step("final answer")], advisor=[_advisor_step("Use the tool output."), _advisor_step("")])
    state = _state(
        client,
        ingested=Ingested(
            machine={},
            sides=Sides(
                messages={
                    MAIN_SIDE: [
                        Message(role="user", content="hello"),
                        Message(
                            role="assistant",
                            tool_calls=[ChatToolCall(id="call_read", name="read_file", arguments='{"path":"src/app.py"}')],
                        ),
                        Message(role="tool", tool_call_id="call_read", content="file contents"),
                    ]
                }
            ),
            last_reasoning_id=None,
            current_compaction_id=None,
        ),
    )

    await core.run_response(state=state)

    assert "### tool_output read_file" in client.advisor_requests[0].messages[-2].content
    assert any(
        message.role == "developer" and message.content == "Use the tool output." and _has_advisor_marker(message)
        for message in client.main_requests[0].messages
    )


@pytest.mark.anyio
async def test_after_tool_advice_emits_summary_annotation_when_not_stealth(monkeypatch: pytest.MonkeyPatch) -> None:
    core = _reload_handlers()
    monkeypatch.setattr("plap.plugins.advisor.STEALTH", False)
    client = _Client(main=[_text_step("final answer")], advisor=[_advisor_step("Use the tool output."), _advisor_step("")])
    state = _state(
        client,
        ingested=Ingested(
            machine={},
            sides=Sides(
                messages={
                    MAIN_SIDE: [
                        Message(role="user", content="hello"),
                        Message(
                            role="assistant",
                            tool_calls=[ChatToolCall(id="call_read", name="read_file", arguments='{"path":"src/app.py"}')],
                        ),
                        Message(role="tool", tool_call_id="call_read", content="file contents"),
                    ]
                }
            ),
            last_reasoning_id=None,
            current_compaction_id=None,
        ),
    )

    await core.run_response(state=state)

    assert "[advisor] advice: Use the tool output." in _summary_texts(state)


@pytest.mark.anyio
async def test_after_tool_note_emits_summary_annotation_when_not_stealth(monkeypatch: pytest.MonkeyPatch) -> None:
    core = _reload_handlers()
    monkeypatch.setattr("plap.plugins.advisor.STEALTH", False)
    client = _Client(
        main=[_text_step("final answer")],
        advisor=[_advisor_step("", note="Watch whether the tools/list call appears next."), _advisor_step("")],
    )
    state = _state(
        client,
        ingested=Ingested(
            machine={},
            sides=Sides(
                messages={
                    MAIN_SIDE: [
                        Message(role="user", content="hello"),
                        Message(
                            role="assistant",
                            tool_calls=[ChatToolCall(id="call_read", name="read_file", arguments='{"path":"src/app.py"}')],
                        ),
                        Message(role="tool", tool_call_id="call_read", content="file contents"),
                    ]
                }
            ),
            last_reasoning_id=None,
            current_compaction_id=None,
        ),
    )

    await core.run_response(state=state)

    assert "[advisor] note: Watch whether the tools/list call appears next." in _summary_texts(state)


@pytest.mark.anyio
async def test_before_return_advice_loops_and_hides_first_answer() -> None:
    core = _reload_handlers()
    client = _Client(
        main=[_text_step("first answer"), _text_step("revised answer")],
        advisor=[_advisor_step("Revise before returning."), _advisor_step("")],
    )
    state = _state(client)

    await core.run_response(state=state)

    output = state.coordinator.current_response().output
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
    core = _reload_handlers()
    monkeypatch.setattr("plap.plugins.advisor.STEALTH", False)
    client = _Client(
        main=[_text_step("first answer"), _text_step("revised answer")],
        advisor=[_advisor_step("Revise before returning."), _advisor_step("")],
    )
    state = _state(client)

    await core.run_response(state=state)

    assert "[advisor] blocked return. advice: Revise before returning." in _summary_texts(state)


@pytest.mark.anyio
async def test_before_return_note_only_emits_neutral_summary_annotation_when_not_stealth(monkeypatch: pytest.MonkeyPatch) -> None:
    core = _reload_handlers()
    monkeypatch.setattr("plap.plugins.advisor.STEALTH", False)
    client = _Client(
        main=[_text_step("first answer")],
        advisor=[_advisor_step("", note="All good. Agent read the file and compile passed.")],
    )
    state = _state(client)

    await core.run_response(state=state)

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


def test_render_main_messages_does_not_emit_non_assistant_reasoning_content() -> None:
    messages = [
        ChatMessage(role="tool", name="read_file", tool_call_id="call_1", content="output", reasoning_content='{"advisor":"call_x"}'),
        ChatMessage(role="developer", content="note", reasoning_content="hidden"),
    ]

    rendered = "\n".join(_markdown_module().render_main_messages(messages))

    assert "### reasoning_content" not in rendered
    assert '{"advisor":"call_x"}' not in rendered
    assert "hidden" not in rendered


def test_requirements_instruction_renders_effective_defaults() -> None:
    request = ChatCompletionRequest(model="main-model", messages=[])

    rendered = _markdown_module().requirements_instruction(request)

    assert rendered.startswith("# requirements\n```json\n")
    assert '"tool_choice":"auto"' in rendered
    assert '"parallel_tool_calls":true' in rendered
    assert '"response_format":null' in rendered
    assert rendered.endswith("\n```")
