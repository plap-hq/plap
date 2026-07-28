# TODO: should probably add a sticky rules system, complementary to notes

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path

import msgspec
import structlog

from plap.bus import bus
from plap.config import CueBox
from plap.errors import ErrorLevel, PlapError, PrivateError
from plap.llms import RetryLimitExceededError
from plap.llms.completions.chat import (
    ChatCompletionRequest,
    ChatFinishReason,
    ChatFunctionTool,
    ChatMessage,
    ChatStreamOptions,
    ChatTool,
    ChatToolCall,
    ChatToolChoiceFunction,
    IChatCompletionClient,
)
from plap.llms.retry import retry_on_tool_choice_mismatch, retry_on_unusable_tool_calls
from plap.llms.retry import stream as retry_stream
from plap.plugins.advisor.markdown import (
    note_instruction,
    render_main_messages,
    requirements_instruction,
)
from plap.plugins.core.ledger import UsageLedger
from plap.plugins.core.loop import StreamResult, response_request
from plap.plugins.core.request import apply_float_transform, apply_int_transform
from plap.responses.ingest.models import MAIN_SIDE
from plap.responses.state import State
from plap.responses.summary import SummaryDelta, SummaryDone

logger = structlog.stdlib.get_logger(__name__)

ADVISOR_SIDE = "advisor"
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
ADVISE_TOOL = ChatTool(
    function=ChatFunctionTool(
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
                    "description": (
                        "Note passed to the next advice phase. This is non-blocking; writing one does NOT cause aborts or loops."
                    ),
                },
            },
            "additionalProperties": False,
        },
        strict=True,
    )
)


def _advisor_sentinel(value: str | bool) -> str:
    return msgspec.json.encode({"advisor": value}).decode()


def _advisor_error(
    *,
    reason: str,
    message: str,
    context: dict[str, object] | None = None,
) -> PlapError:
    return PlapError(
        public=None,
        private=PrivateError(
            event="response.advisor_failed",
            reason=reason,
            message=message,
            level=ErrorLevel.ERROR,
            context={} if context is None else context,
        ),
    )


def _advisor_durable(state: State) -> dict[str, object]:
    raw = state.durable.to_primitive().get(ADVISOR_SIDE)
    if isinstance(raw, Mapping):
        return dict(raw)
    return {}


def _advisor_note(state: State) -> str | None:
    value = _advisor_durable(state).get("note")
    if not isinstance(value, str) or not value:
        return None
    return value


def _set_advisor_note(state: State, note: str | None) -> None:
    durable = _advisor_durable(state)
    if note is None:
        durable.pop("note", None)
    else:
        durable["note"] = note
    state.durable = state.durable.model_copy(update={ADVISOR_SIDE: durable}, deep=True)


def _is_advisor_artifact(msg: ChatMessage) -> bool:
    if msg.reasoning_content is None:
        return False
    try:
        data = msgspec.json.decode(msg.reasoning_content)
        return isinstance(data, dict) and "advisor" in data
    except Exception:
        return False


def _advisor_call_id(msg: ChatMessage) -> str:
    data = msgspec.json.decode(msg.reasoning_content)
    return data["advisor"]


def _is_advisor_transcript_message(msg: ChatMessage) -> bool:
    return msg.role == "user" and msg.reasoning_content == _advisor_sentinel(True)


def _strip_note_from_messages(messages: tuple[ChatMessage, ...]) -> list[ChatMessage]:
    stripped: list[ChatMessage] = []
    for message in messages:
        if not message.is_assistant() or not message.tool_calls:
            stripped.append(message)
            continue
        changed = False
        tool_calls: list[ChatToolCall] = []
        for call in message.tool_calls:
            if call.name != ADVISE_TOOL_NAME:
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


def _rebuild_advisor_side(state: State) -> None:
    existing_blocks: dict[str, list[ChatMessage]] = {}
    current_block: list[ChatMessage] = []
    for entry in state.sides.get(ADVISOR_SIDE, []):
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

    history = state.history(MAIN_SIDE)
    new_side: list[ChatMessage] = []
    current_msgs: list[ChatMessage] = []
    buffered_msgs: list[ChatMessage] = []

    def flush_user():
        if current_msgs:
            rendered = "\n".join(render_main_messages(current_msgs))
            new_side.append(ChatMessage(role="user", content=rendered, reasoning_content=_advisor_sentinel(True)))
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
                    new_side.extend(block)
                current_msgs.clear()
                current_msgs.extend(buffered_msgs)
                buffered_msgs.clear()
                continue
        current_msgs.append(msg)

    flush_user()
    if buffered_msgs:
        rendered = "\n".join(render_main_messages(buffered_msgs))
        new_side.append(ChatMessage(role="user", content=rendered, reasoning_content=_advisor_sentinel(True)))
    state.sides[ADVISOR_SIDE] = new_side


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
    config: CueBox,
    main_request: ChatCompletionRequest,
    phase_instruction: str,
) -> ChatCompletionRequest:
    advisor = config.advisor
    sampling = advisor.sampling
    execution_request = state.prepared.execution_request
    tools = [ADVISE_TOOL]
    for tool in main_request.tools:
        if tool.function.name == ADVISE_TOOL_NAME:
            raise _advisor_error(
                reason="advisor_tool_name_conflict",
                message="main request tool conflicts with internal advisor advise tool",
                context={"tool_name": ADVISE_TOOL_NAME},
            )
        tools.append(tool)
    return ChatCompletionRequest(
        model=advisor.model,
        messages=[
            ChatMessage(role="developer", content=ADVISOR_PROMPT),
            *state.history(ADVISOR_SIDE),
            ChatMessage(role="user", content=phase_instruction),
        ],
        tools=tools,
        tool_choice=ChatToolChoiceFunction(name=ADVISE_TOOL_NAME),
        parallel_tool_calls=False,
        max_completion_tokens=advisor.max_completion_tokens,
        temperature=apply_float_transform(execution_request.temperature, sampling.temperature, minimum=0, maximum=2),
        top_p=apply_float_transform(execution_request.top_p, sampling.top_p, minimum=0, maximum=1),
        min_p=apply_float_transform(None, sampling.min_p, minimum=0, maximum=1),
        top_k=apply_int_transform(None, sampling.top_k, minimum=0),
        frequency_penalty=apply_float_transform(None, sampling.frequency_penalty, minimum=-2, maximum=2),
        presence_penalty=apply_float_transform(None, sampling.presence_penalty, minimum=-2, maximum=2),
        repetition_penalty=apply_float_transform(None, sampling.repetition_penalty, minimum=0, maximum=2),
        seed=apply_int_transform(None, sampling.seed),
        reasoning_effort=advisor.reasoning_effort,
        stream_options=ChatStreamOptions(include_usage=True),
        prompt_cache_key=execution_request.prompt_cache_key,
        service_tier=advisor.service_tier,
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
    await state.ensure_reasoning()
    await state.coordinator.summary_delta(SummaryDelta(text=text))
    await state.coordinator.summary_done(SummaryDone())


async def _run_advisor(
    *,
    state: State,
    config: CueBox,
    ledger: UsageLedger,
    main_request: ChatCompletionRequest,
    phase_instruction: str,
    phase: str,
) -> tuple[str | None, str]:
    _rebuild_advisor_side(state)
    request = _advisor_request(
        state=state,
        config=config,
        main_request=main_request,
        phase_instruction=phase_instruction,
    )

    client = await state.svcs.aget(IChatCompletionClient)
    latest_snapshot = None
    hidden_results_accounted = 0
    budget_exhausted = False

    def next_request(history):
        nonlocal hidden_results_accounted, budget_exhausted
        for result in history.results[hidden_results_accounted:]:
            ledger.hide(config.advisor.public_usage, result.usage)
            hidden_results_accounted += 1
        attempt_index = hidden_results_accounted + 1
        attempt_budget = ledger.cap(config.advisor.public_usage, None)
        attempt_cap = ledger.cap(config.advisor.public_usage, config.advisor.max_completion_tokens)
        if attempt_cap == 0:
            budget_exhausted = True
            logger.info(
                "response.advisor.skipped",
                phase=phase,
                reason="budget_exhausted",
                attempt_budget=attempt_budget,
                attempt_index=attempt_index,
                hidden_history_messages=len(history.messages),
                hidden_results=len(history.results),
                remaining=ledger.remaining(),
            )
            return None
        attempt_request = replace(request, messages=[*request.messages, *history.messages], max_completion_tokens=attempt_cap)
        logger.info(
            "response.advisor.request",
            attempt_index=attempt_index,
            attempt_budget=attempt_budget,
            cap=attempt_cap,
            hidden_history_messages=len(history.messages),
            hidden_results=len(history.results),
            remaining=ledger.remaining(),
            phase=phase,
        )
        logger.bind(log_channel="payload").info(
            "response.advisor.request.payload",
            attempt_index=attempt_index,
            request=asdict(attempt_request),
        )
        return attempt_request

    try:
        async for snapshot in retry_stream(
            client,
            next_request=next_request,
            validators=(retry_on_tool_choice_mismatch, retry_on_unusable_tool_calls),
        ):
            latest_snapshot = snapshot
    except RetryLimitExceededError:
        logger.warning("response.advisor.skipped", phase=phase, reason="retry_limit_exceeded")
        return None, ""

    if budget_exhausted or latest_snapshot is None:
        return None, ""

    result = latest_snapshot.results[-1]
    ledger.hide(config.advisor.public_usage, result.usage)
    call = result.message.tool_calls[0]
    advice, note = _advice_fields(call)
    usage = result.usage

    logger.info(
        "response.advisor.result",
        advisor_cached_tokens=None if usage is None else usage.cached_tokens,
        phase=phase,
        applied=advice is not None,
        advice_chars=len(advice) if advice else 0,
        advisor_input_tokens=None if usage is None else usage.input_tokens,
        advisor_output_tokens=None if usage is None else usage.output_tokens,
        advisor_reasoning_tokens=None if usage is None else usage.reasoning_tokens,
        advisor_total_tokens=None if usage is None else usage.total_tokens,
        remaining=ledger.remaining(),
    )

    state.sides[ADVISOR_SIDE].extend(_strip_note_from_messages(latest_snapshot.messages))
    state.sides[ADVISOR_SIDE].append(ChatMessage(role="tool", tool_call_id=call.id, content=ADVISOR_TOOL_OUTPUT))

    _set_advisor_note(state, note)

    return advice, call.id


async def _maybe_advise_after_tool_call(
    *,
    state: State,
    config: CueBox,
    ledger: UsageLedger,
    main_request: ChatCompletionRequest,
) -> None:
    history = state.history(MAIN_SIDE)
    if not history or not history[-1].is_tool():
        return
    phase_instruction = _phase_instruction(state, "after_tool_call", main_request)
    logger.info("response.advisor.phase", phase="after_tool_call", main_model=main_request.model, main_messages=len(history))
    advice, call_id = await _run_advisor(
        state=state,
        config=config,
        ledger=ledger,
        main_request=main_request,
        phase_instruction=phase_instruction,
        phase="after_tool_call",
    )
    note = _advisor_note(state)
    text = _annotation_text("[advisor]", advice, note)
    if text is not None:
        await _emit_annotation(state, text)
    if advice is not None:
        state.main.append(ChatMessage(role="developer", content=advice, reasoning_content=_advisor_sentinel(call_id)))


async def _maybe_advise_before_tool_call(
    *,
    state: State,
    config: CueBox,
    ledger: UsageLedger,
    main_request: ChatCompletionRequest,
    result: StreamResult,
) -> None:
    accepted = result.accepted
    if accepted is None or accepted.finish_reason != ChatFinishReason.TOOL_CALLS:
        return
    open_calls = state.open_calls(MAIN_SIDE)
    if not open_calls:
        return
    phase_instruction = _phase_instruction(state, "before_tool_call", main_request)
    call_names = [c.name for c in open_calls]
    logger.info("response.advisor.phase", phase="before_tool_call", main_model=main_request.model, pending_calls=call_names)
    advice, call_id = await _run_advisor(
        state=state,
        config=config,
        ledger=ledger,
        main_request=main_request,
        phase_instruction=phase_instruction,
        phase="before_tool_call",
    )
    if advice is None:
        return
    state.main.extend(
        ChatMessage(
            role="tool",
            name=call.name,
            tool_call_id=call.id,
            content=ABORTED_TOOL_OUTPUT,
            reasoning_content=_advisor_sentinel(call_id),
        )
        for call in open_calls
    )
    joined = ", ".join(call.name for call in open_calls)
    note = _advisor_note(state)
    text = _annotation_text(f"[advisor] blocked tool call(s): {joined}.", advice, note)
    if text is not None:
        await _emit_annotation(state, text)
    state.main.append(ChatMessage(role="developer", content=advice, reasoning_content=_advisor_sentinel(call_id)))


async def _maybe_advise_before_return(
    *,
    state: State,
    config: CueBox,
    ledger: UsageLedger,
    main_request: ChatCompletionRequest,
    result: StreamResult,
) -> None:
    accepted = result.accepted
    if accepted is None or accepted.finish_reason != ChatFinishReason.STOP or accepted.message.tool_calls:
        return
    if state.open_calls(MAIN_SIDE):
        return
    phase_instruction = _phase_instruction(state, "before_return", main_request)
    logger.info("response.advisor.phase", phase="before_return", main_model=main_request.model)
    advice, call_id = await _run_advisor(
        state=state,
        config=config,
        ledger=ledger,
        main_request=main_request,
        phase_instruction=phase_instruction,
        phase="before_return",
    )
    note = _advisor_note(state)
    text = _annotation_text("[advisor] blocked return.", advice, note)
    if text is not None:
        await _emit_annotation(state, text)
    if advice is not None:
        state.main.append(ChatMessage(role="developer", content=advice, reasoning_content=_advisor_sentinel(call_id)))


@bus.listen("config.collect")
async def collect(paths: tuple[str, ...], *, next):
    here = Path(__file__).resolve()
    return await next(paths=(*paths, str(here.parent / "schema.cue")))


@bus.listen("response.loop")
async def advise_response(state: State, config: CueBox, ledger: UsageLedger, *, next) -> StreamResult | None:
    if MAIN_SIDE not in state.sides.active or state.open_calls(MAIN_SIDE):
        return await next(state=state, config=config, ledger=ledger)
    main_request = await response_request(state=state, config=config)
    await _maybe_advise_after_tool_call(state=state, config=config, ledger=ledger, main_request=main_request)
    result = await next(state=state, config=config, ledger=ledger)
    if result is None:
        return None
    await _maybe_advise_before_tool_call(state=state, config=config, ledger=ledger, main_request=main_request, result=result)
    await _maybe_advise_before_return(state=state, config=config, ledger=ledger, main_request=main_request, result=result)
    return result


__all__ = [
    "ABORTED_TOOL_OUTPUT",
    "ADVISE_TOOL_NAME",
    "ADVISOR_SIDE",
    "ADVISOR_TOOL_OUTPUT",
]
