# TODO: should probably add a sticky rules system, complementary to notes

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import msgspec
import structlog

from plap.bus import bus
from plap.llms import RetryLimitExceededError
from plap.llms.completions.budget import (
    BudgetedChatCompletionClient,
    CompletionBudget,
    CompletionBudgetExhaustedError,
)
from plap.llms.completions.chat import (
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatFinishReason,
    ChatFunctionTool,
    ChatMessage,
    ChatStreamOptions,
    ChatTool,
    ChatToolCall,
    ChatToolChoiceFunction,
)
from plap.llms.retry import complete as retry_complete
from plap.llms.retry import retry_on_tool_choice_mismatch, retry_on_unusable_tool_calls
from plap.plugins.advisor.markdown import (
    note_instruction,
    render_main_messages,
    requirements_instruction,
)
from plap.plugins.core.request import build_chat_request
from plap.plugins.easy import bootstrap, server_tools
from plap.responses.state import State
from plap.responses.streaming import StreamCoordinator
from plap.responses.summary import SummaryDelta, SummaryDone

bootstrap.config(__file__)

logger = structlog.stdlib.get_logger(__name__)

ADVISOR_THREAD = "advisor"
ADVISE_TOOL_NAME = "advise"
ADVISOR_TOOL_OUTPUT = "0"
ABORTED_TOOL_OUTPUT = "Tool call cancelled by advisor."
STEALTH = False
ADVISOR_PROMPT = """You are a peer reviewer / watchdog for the main agent.
You receive the agent transcript incrementally, including thoughts.
"""
BEFORE_TOOL_CALL_PHASE = """phase: before_tool_call

Call advise exactly once.
Use {"advice":"",  "note": ...} when no guidance is needed.
IMPORTANT: A non-empty advice will abort the pending tool calls.
DO NOT provide advice at this time UNLESS you (intentionally) want to BLOCK those tool calls.
For example, if those tool calls directly go against what the user said or are dangerous.
If you want to note something to yourself, put it in note, NOT advice.

ALSO IMPORTANT: You will enter phase: after_tool_call after the tool execution.
For efficiency reasons, prefer using a note to that phase INSTEAD of advice (which is blocking)
when a large tool call would take less effort to fix with a follow-up patch.

As an advisor, you may only call the advise tool. If you believe the main agent has not read or verified enough,
you should advise it to do so (call the tools you want, in order to confirm suspicions) rather than calling the tools yourself.
"""
AFTER_TOOL_CALL_PHASE = """phase: after_tool_call

Call advise exactly once.
Use {"advice":"",  "note": ...} when no guidance is needed.
IMPORTANT: If you have nothing of substance to add, prefer staying silent.
If you want to note something to yourself, put it in note, NOT advice.

As an advisor, you may only call the advise tool. If you believe the main agent has not read or verified enough,
you should advise it to do so (call the tools you want, in order to confirm suspicions) rather than calling the tools yourself.
"""
BEFORE_RETURN_PHASE = """phase: before_return

Call advise exactly once.
Use {"advice":"",  "note": ...} when no guidance is needed.
IMPORTANT: This is the last point to intercept before the agent returns a response.
I repeat, it is VERY IMPORTANT for you to verify your suspicions at this time, or else you will not be able to later.

ALSO IMPORTANT: A non-empty advice will abort the pending return and cause the model to loop.
DO NOT provide advice at this time UNLESS you (intentionally) want to LOOP the main agent.
For example, if the main agent has not sufficiently completed or verified its completion of a task and instead returned early.
If you want to note something to yourself, put it in note, NOT advice.

As an advisor, you may only call the advise tool. If you believe the main agent has not read or verified enough,
you should advise it to do so (call the tools you want, in order to confirm suspicions) rather than calling the tools yourself.
"""
ADVISE_FUNCTION = ChatFunctionTool(
    name=ADVISE_TOOL_NAME,
    description="Provide guidance for the main agent.",
    parameters={
        "type": "object",
        "properties": {
            "advice": {
                "type": "string",
                "description": "Guidance for the main agent.",
            },
            "note": {
                "type": "string",
                "description": ("Note passed to the next advice phase. This is non-blocking; writing one does NOT cause aborts or loops."),
            },
        },
        "additionalProperties": False,
    },
    strict=True,
)


def _advisor_memory(state: State) -> dict[str, object]:
    raw = state.memory.get(ADVISOR_THREAD)
    if isinstance(raw, Mapping):
        return dict(raw)
    return {}


def _advisor_note(state: State) -> str | None:
    value = _advisor_memory(state).get("note")
    if not isinstance(value, str) or not value:
        return None
    return value


def _set_advisor_note(state: State, note: str | None) -> None:
    memory = _advisor_memory(state)
    if note is None:
        memory.pop("note", None)
    else:
        memory["note"] = note
    state.memory[ADVISOR_THREAD] = memory


def _advisor_message_memory(msg: ChatMessage) -> dict[str, object]:
    raw = msg.memory.get(ADVISOR_THREAD)
    if isinstance(raw, Mapping):
        return dict(raw)
    return {}


def _is_advisor_artifact(msg: ChatMessage) -> bool:
    return isinstance(_advisor_message_memory(msg).get("call_id"), str)


def _advisor_call_id(msg: ChatMessage) -> str:
    call_id = _advisor_message_memory(msg).get("call_id")
    if not isinstance(call_id, str):
        raise TypeError("advisor artifact is missing its call id")
    return call_id


def _is_advisor_transcript_message(msg: ChatMessage) -> bool:
    return msg.role == "user" and _advisor_message_memory(msg).get("transcript") is True


def _strip_note_from_messages(messages: tuple[ChatMessage, ...], *, advise_tool_name: str) -> list[ChatMessage]:
    stripped: list[ChatMessage] = []
    for message in messages:
        if not message.is_assistant() or not message.tool_calls:
            stripped.append(message)
            continue
        changed = False
        tool_calls: list[ChatToolCall] = []
        for call in message.tool_calls:
            if call.name != advise_tool_name:
                tool_calls.append(call)
                continue
            arguments = msgspec.json.decode(call.arguments)
            if isinstance(arguments, dict) and "note" in arguments:
                changed = True
                arguments = dict(arguments)
                arguments.pop("note", None)
                tool_calls.append(ChatToolCall(id=call.id, name=call.name, arguments=msgspec.json.encode(arguments).decode()))
                continue
            tool_calls.append(call)
        stripped.append(message if not changed else replace(message, tool_calls=tool_calls))
    return stripped


def _final_block_call_id(block: list[ChatMessage]) -> str | None:
    for msg in reversed(block):
        if msg.role == "tool" and msg.content == ADVISOR_TOOL_OUTPUT and msg.tool_call_id is not None:
            return msg.tool_call_id
    return None


def _rebuild_advisor_thread(state: State) -> None:
    existing_blocks: dict[str, list[ChatMessage]] = {}
    current_block: list[ChatMessage] = []
    for entry in state.threads.get(ADVISOR_THREAD, []):
        if _is_advisor_transcript_message(entry):
            if current_block:
                call_id = _final_block_call_id(current_block)
                if call_id is not None:
                    existing_blocks[call_id] = list(current_block)
                current_block = []
            continue
        if entry.role == "user" and not current_block:
            continue
        current_block.append(entry)
    if current_block:
        call_id = _final_block_call_id(current_block)
        if call_id is not None:
            existing_blocks[call_id] = list(current_block)

    history = state.threads["main"]
    new_thread: list[ChatMessage] = []
    current_msgs: list[ChatMessage] = []
    buffered_msgs: list[ChatMessage] = []

    def flush_user():
        if current_msgs:
            rendered = "\n".join(render_main_messages(current_msgs))
            new_thread.append(ChatMessage(role="user", content=rendered, memory={ADVISOR_THREAD: {"transcript": True}}))
            current_msgs.clear()

    for msg in history:
        if _is_advisor_artifact(msg):
            call_id = _advisor_call_id(msg)
            if msg.role == "tool":
                buffered_msgs.append(msg)
                continue
            if msg.role == "developer":
                flush_user()
                block = existing_blocks.get(call_id)
                if block is not None:
                    new_thread.extend(block)
                current_msgs.clear()
                current_msgs.extend(buffered_msgs)
                buffered_msgs.clear()
                continue
        current_msgs.append(msg)

    flush_user()
    if buffered_msgs:
        rendered = "\n".join(render_main_messages(buffered_msgs))
        new_thread.append(ChatMessage(role="user", content=rendered, memory={ADVISOR_THREAD: {"transcript": True}}))
    state.threads[ADVISOR_THREAD] = new_thread


def _phase_instruction(state: State, phase: str, main_request: ChatCompletionRequest) -> str:
    req = requirements_instruction(main_request)
    base = {
        "before_tool_call": BEFORE_TOOL_CALL_PHASE,
        "after_tool_call": AFTER_TOOL_CALL_PHASE,
        "before_return": BEFORE_RETURN_PHASE,
    }[phase]
    parts: list[str] = []
    if req:
        parts.append(req)
    note = _advisor_note(state)
    if note is not None:
        parts.append(note_instruction(note))
    parts.append(base)
    return "\n".join(parts)


def _advisor_request(
    *,
    state: State,
    main_request: ChatCompletionRequest,
    phase_instruction: str,
) -> ChatCompletionRequest:
    advisor = state.config.advisor
    advise_function = server_tools.rename_to_avoid_collisions(ADVISE_FUNCTION, main_request.tools)
    return replace(
        build_chat_request(
            advisor,
            state.request,
            messages=[
                ChatMessage(role="developer", content=ADVISOR_PROMPT),
                *state.threads.get(ADVISOR_THREAD, []),
                ChatMessage(role="user", content=phase_instruction),
            ],
        ),
        tools=[ChatTool(function=advise_function), *main_request.tools],
        tool_choice=ChatToolChoiceFunction(name=advise_function.name),
        parallel_tool_calls=False,
        stream_options=ChatStreamOptions(include_usage=True),
        prompt_cache_key=state.request.prompt_cache_key,
    )


def _advice_fields(call: ChatToolCall) -> tuple[str | None, str | None]:
    arguments = msgspec.json.decode(call.arguments)
    advice_value = arguments.get("advice") if isinstance(arguments, dict) else None
    advice = advice_value.strip() if isinstance(advice_value, str) else ""
    note_value = arguments.get("note") if isinstance(arguments, dict) else None
    note = note_value.strip() if isinstance(note_value, str) else ""
    return advice or None, note or None


def _annotation_text(prefix: str, advice: str | None, note: str | None) -> str | None:
    if advice is None and note is None:
        return None
    if advice is None:
        return f"[advisor] note: {note}"
    parts = [prefix]
    parts.append(f"advice: {advice}")
    if note is not None:
        parts.append(f"note: {note}")
    return " ".join(parts)


async def _emit_annotation(state: State, text: str) -> None:
    if STEALTH or not text:
        return
    coordinator = state.svcs.get(StreamCoordinator)
    await state.ensure_progress()
    await coordinator.summary_delta(SummaryDelta(text=text))
    await coordinator.summary_done(SummaryDone())


async def _run_advisor(
    *,
    state: State,
    main_request: ChatCompletionRequest,
    phase_instruction: str,
    phase: str,
) -> tuple[str | None, str]:
    _rebuild_advisor_thread(state)
    request = _advisor_request(
        state=state,
        main_request=main_request,
        phase_instruction=phase_instruction,
    )

    client = await state.svcs.aget(BudgetedChatCompletionClient)

    try:
        latest_snapshot = await retry_complete(
            client,
            request,
            validators=(retry_on_tool_choice_mismatch, retry_on_unusable_tool_calls),
        )
    except CompletionBudgetExhaustedError:
        budget = state.svcs.get(CompletionBudget)
        logger.info("response.advisor.skipped", phase=phase, reason="budget_exhausted", remaining=budget.remaining)
        return None, ""
    except RetryLimitExceededError:
        logger.warning("response.advisor.skipped", phase=phase, reason="retry_limit_exceeded")
        return None, ""

    result = latest_snapshot.result
    if result is None:  # pragma: no cover - retry_complete returns an accepted terminal snapshot
        raise RuntimeError("advisor completion ended without an accepted result")
    call = result.message.tool_calls[0]
    advice, note = _advice_fields(call)
    logger.info(
        "response.advisor.result",
        advice_chars=len(advice) if advice else 0,
        applied=advice is not None,
        has_note=note is not None,
        phase=phase,
    )

    state.threads[ADVISOR_THREAD].extend(
        _strip_note_from_messages(latest_snapshot.messages, advise_tool_name=request.tools[0].function.name)
    )
    state.threads[ADVISOR_THREAD].append(ChatMessage(role="tool", tool_call_id=call.id, content=ADVISOR_TOOL_OUTPUT))

    _set_advisor_note(state, note)

    return advice, call.id


async def _maybe_advise_after_tool_call(
    *,
    state: State,
    main_request: ChatCompletionRequest,
) -> ChatMessage | None:
    history = state.threads["main"]
    if not history or not history[-1].is_tool():
        return None
    phase_instruction = _phase_instruction(state, "after_tool_call", main_request)
    logger.info("response.advisor.phase", phase="after_tool_call", main_model=main_request.model, main_messages=len(history))
    advice, call_id = await _run_advisor(
        state=state,
        main_request=main_request,
        phase_instruction=phase_instruction,
        phase="after_tool_call",
    )
    note = _advisor_note(state)
    text = _annotation_text("[advisor]", advice, note)
    if text is not None:
        await _emit_annotation(state, text)
    if advice is not None:
        message = ChatMessage(role="developer", content=advice, memory={ADVISOR_THREAD: {"call_id": call_id}})
        state.threads["main"].append(message)
        return message
    return None


async def _maybe_advise_before_tool_call(
    *,
    state: State,
    main_request: ChatCompletionRequest,
    result: ChatCompletionResult,
) -> None:
    if result.finish_reason != ChatFinishReason.TOOL_CALLS:
        return
    open_calls = state.open_calls("main")
    if not open_calls:
        return
    phase_instruction = _phase_instruction(state, "before_tool_call", main_request)
    call_names = [c.name for c in open_calls]
    logger.info("response.advisor.phase", phase="before_tool_call", main_model=main_request.model, pending_calls=call_names)
    advice, call_id = await _run_advisor(
        state=state,
        main_request=main_request,
        phase_instruction=phase_instruction,
        phase="before_tool_call",
    )
    if advice is None:
        return
    state.threads["main"].extend(
        ChatMessage(
            role="tool",
            tool_call_id=call.id,
            content=ABORTED_TOOL_OUTPUT,
            memory={ADVISOR_THREAD: {"call_id": call_id, "tool_name": call.name}},
        )
        for call in open_calls
    )
    joined = ", ".join(call.name for call in open_calls)
    note = _advisor_note(state)
    text = _annotation_text(f"[advisor] blocked tool call(s): {joined}.", advice, note)
    if text is not None:
        await _emit_annotation(state, text)
    state.threads["main"].append(ChatMessage(role="developer", content=advice, memory={ADVISOR_THREAD: {"call_id": call_id}}))


async def _maybe_advise_before_return(
    *,
    state: State,
    main_request: ChatCompletionRequest,
    result: ChatCompletionResult,
) -> None:
    if result.finish_reason != ChatFinishReason.STOP or result.message.tool_calls:
        return
    if state.open_calls("main"):
        return
    phase_instruction = _phase_instruction(state, "before_return", main_request)
    logger.info("response.advisor.phase", phase="before_return", main_model=main_request.model)
    advice, call_id = await _run_advisor(
        state=state,
        main_request=main_request,
        phase_instruction=phase_instruction,
        phase="before_return",
    )
    note = _advisor_note(state)
    text = _annotation_text("[advisor] blocked return.", advice, note)
    if text is not None:
        await _emit_annotation(state, text)
    if advice is not None:
        state.threads["main"].append(ChatMessage(role="developer", content=advice, memory={ADVISOR_THREAD: {"call_id": call_id}}))


@bus.listen("response.turn")
async def advise_turn(
    state: State,
    request: ChatCompletionRequest,
    *,
    next,
) -> ChatCompletionResult:
    advice = await _maybe_advise_after_tool_call(
        state=state,
        main_request=request,
    )
    if advice is not None:
        request = replace(request, messages=[*request.messages, advice])
    result = await next(state=state, request=request)
    await _maybe_advise_before_tool_call(state=state, main_request=request, result=result)
    await _maybe_advise_before_return(state=state, main_request=request, result=result)
    return result


__all__ = [
    "ABORTED_TOOL_OUTPUT",
    "ADVISE_TOOL_NAME",
    "ADVISOR_THREAD",
    "ADVISOR_TOOL_OUTPUT",
]
