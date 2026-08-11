from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import blake3
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
    ChatToolChoiceMode,
)
from plap.llms.retry import (
    RetryValidator,
    complete,
    retry_message,
    retry_on_tool_choice_mismatch,
    retry_on_unusable_tool_calls,
)
from plap.plugins.advisor.markdown import render_main_update, requirements_instruction
from plap.plugins.core.loop import response_request, response_snapshot
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

_PHASES = frozenset({"before_tool_call", "after_tool_call", "before_return"})

ADVISOR_PROMPT = """You are a peer reviewer / watchdog for the main agent.
You receive the agent transcript incrementally, including thoughts.
Investigate suspicions directly with the available tools.
You may take as many exploration rounds as needed and call independent tools in parallel.
The # requirements block describes constraints the main agent must satisfy.
When you are finished, call advise exactly once and by itself.
"""

BEFORE_TOOL_CALL_PHASE = """phase: before_tool_call

The main agent has pending tool calls.

Empty advice allows every pending call to proceed. Non-empty advice cancels every pending call and sends
your guidance to the main agent.

IMPORTANT: Block only when you intentionally want to stop those calls.

Block calls that conflict with the user's request, are unsafe or irreversible, are entirely useless,
or must be corrected before execution.

If the calls are reversible and likely to provide useful evidence, prefer allowing them to run. You
will review their outputs in phase: after_tool_call. If a concern can wait until then, put it in note
instead of blocking.
"""

AFTER_TOOL_CALL_PHASE = """phase: after_tool_call

The main agent's tool outputs are available. Inspect them and verify any unresolved concerns.
"""

BEFORE_RETURN_PHASE = """phase: before_return

The main agent is about to return its answer to the user. This is the final opportunity to verify unresolved concerns.

Empty advice allows the answer to return. Non-empty advice blocks the return and sends your guidance to
the main agent.

IMPORTANT: Block when the work is incomplete, conflicts with the user's request, or is returning prematurely.

Do not use note to defer a concern that must be resolved before the answer is returned.
"""

ADVISE_FUNCTION = ChatFunctionTool(
    name=ADVISE_TOOL_NAME,
    description="Finish the current review phase and optionally guide the main agent.",
    parameters={
        "type": "object",
        "properties": {
            "advice": {
                "type": "string",
                "description": "Guidance for the main agent. Use an empty string when no guidance is needed.",
            },
            "note": {
                "type": "string",
                "description": "Non-blocking observation to yourself for later review phases. The main agent does not see this.",
            },
        },
        "required": ["advice"],
        "additionalProperties": False,
    },
    strict=True,
)


def _advisor_message_memory(message: ChatMessage) -> dict[str, object]:
    raw = message.memory.get(ADVISOR_THREAD)
    if isinstance(raw, Mapping):
        return dict(raw)
    return {}


def _is_main_artifact(message: ChatMessage) -> bool:
    return _advisor_message_memory(message).get("artifact") is True


def _transcript_anchor(message: ChatMessage) -> str | None:
    value = _advisor_message_memory(message).get("transcript_anchor")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("advisor transcript anchor must be a non-empty string")
    return value


def _active_phase(messages: list[ChatMessage]) -> tuple[int, str] | None:
    markers: list[tuple[int, str]] = []
    for index, message in enumerate(messages):
        value = _advisor_message_memory(message).get("phase")
        if value is None:
            continue
        if not isinstance(value, str) or value not in _PHASES:
            raise ValueError("advisor phase marker is invalid")
        markers.append((index, value))
    if len(markers) > 1:
        raise RuntimeError("advisor thread contains multiple active phase markers")
    return markers[0] if markers else None


def _projected_main(state: State) -> list[ChatMessage]:
    return [message for message in state.threads["main"] if not _is_main_artifact(message)]


def _prefix_anchors(messages: list[ChatMessage]) -> list[str]:
    hasher = blake3.blake3()
    anchors: list[str] = []
    for message in messages:
        hasher.update(bytes.fromhex(message.content_hash()))
        anchors.append(hasher.hexdigest())
    return anchors


def _latest_main_turn(messages: list[ChatMessage]) -> list[ChatMessage]:
    if not messages:
        return []
    if not messages[-1].is_tool():
        return [messages[-1]]
    start = len(messages) - 1
    while start > 0 and messages[start - 1].is_tool():
        start -= 1
    if start > 0 and messages[start - 1].is_assistant():
        start -= 1
    return messages[start:]


def _append_main_update(state: State) -> None:
    advisor = state.threads.setdefault(ADVISOR_THREAD)
    projected = _projected_main(state)
    if not projected:
        return

    anchors = _prefix_anchors(projected)
    previous = next(
        (anchor for message in reversed(advisor) if (anchor := _transcript_anchor(message)) is not None),
        None,
    )
    if previous is None:
        start = 0
    else:
        try:
            previous_index = anchors.index(previous)
        except ValueError:
            latest = _latest_main_turn(projected)
            start = len(projected) - len(latest)
            logger.info("response.advisor.reanchored", main_messages=len(projected), update_messages=len(latest))
        else:
            start = previous_index + 1

    if start >= len(projected):
        return
    advisor.append(
        ChatMessage(
            role="user",
            content="\n".join(render_main_update(projected, start)),
            memory={ADVISOR_THREAD: {"transcript_anchor": anchors[-1]}},
        )
    )


def _phase_instruction(phase: str, main_request: ChatCompletionRequest) -> str:
    requirements = requirements_instruction(main_request)
    phase_text = {
        "before_tool_call": BEFORE_TOOL_CALL_PHASE,
        "after_tool_call": AFTER_TOOL_CALL_PHASE,
        "before_return": BEFORE_RETURN_PHASE,
    }[phase]
    return "\n\n".join(part for part in (requirements, phase_text) if part)


def _phase_message(phase: str, main_request: ChatCompletionRequest) -> ChatMessage:
    return ChatMessage(
        role="user",
        content=_phase_instruction(phase, main_request),
        memory={ADVISOR_THREAD: {"phase": phase}},
    )


def _client_tools(main_request: ChatCompletionRequest) -> list[ChatTool]:
    return [tool for tool in main_request.tools if not isinstance(tool.function, server_tools.ServerTool)]


def _advisor_request(state: State, main_request: ChatCompletionRequest) -> tuple[ChatCompletionRequest, str]:
    client_tools = _client_tools(main_request)
    advise_function = server_tools.rename_to_avoid_collisions(ADVISE_FUNCTION, client_tools)
    request = replace(
        build_chat_request(
            state.config.advisor,
            state.request,
            messages=[
                ChatMessage(role="developer", content=ADVISOR_PROMPT),
                *state.threads[ADVISOR_THREAD],
            ],
        ),
        tools=[ChatTool(function=advise_function), *client_tools],
        tool_choice=ChatToolChoiceMode.REQUIRED,
        parallel_tool_calls=True,
        stream_options=ChatStreamOptions(include_usage=True),
        prompt_cache_key=state.request.prompt_cache_key,
    )
    return server_tools.prepare(request), advise_function.name


def _advise_validator(advise_tool_name: str) -> RetryValidator:
    async def validate(result: ChatCompletionResult, request: ChatCompletionRequest) -> str | None:
        _ = request
        advise_calls = [call for call in result.message.tool_calls if call.name == advise_tool_name]
        if not advise_calls:
            return None
        if len(advise_calls) == 1 and len(result.message.tool_calls) == 1:
            return None
        return retry_message(
            problems=("The final `advise` call was mixed with other tool calls.",),
            rules=("Use exploration tools without `advise`, or call `advise` by itself when the review phase is complete.",),
        )

    return validate


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
    parts = [prefix, f"advice: {advice}"]
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


def _start_phase(state: State, phase: str, main_request: ChatCompletionRequest) -> None:
    advisor = state.threads.setdefault(ADVISOR_THREAD)
    if _active_phase(advisor) is not None:
        raise RuntimeError("cannot start an advisor phase while another phase is active")
    _append_main_update(state)
    advisor.append(_phase_message(phase, main_request))
    state.threads.enable(ADVISOR_THREAD)
    state.threads.block("main", by=ADVISOR_THREAD)


def _refresh_phase_instruction(state: State, main_request: ChatCompletionRequest) -> None:
    advisor = state.threads[ADVISOR_THREAD]
    marker = _active_phase(advisor)
    if marker is None:
        raise RuntimeError("active advisor thread is missing its phase marker")
    index, phase = marker

    instruction = _phase_instruction(phase, main_request)
    if advisor[index].content != instruction:
        advisor[index] = replace(advisor[index], content=instruction)


def _remove_phase_message(state: State) -> str:
    advisor = state.threads[ADVISOR_THREAD]
    marker = _active_phase(advisor)
    if marker is None:
        raise RuntimeError("active advisor thread is missing its phase marker")
    index, phase = marker
    advisor.pop(index)
    return phase


async def _complete_advisor(
    state: State,
    main_request: ChatCompletionRequest,
) -> tuple[ChatCompletionResult, str]:
    _refresh_phase_instruction(state, main_request)
    request, advise_tool_name = _advisor_request(state, main_request)
    client = await state.svcs.aget(BudgetedChatCompletionClient)
    snapshot = await complete(
        client,
        request,
        validators=(
            retry_on_tool_choice_mismatch,
            retry_on_unusable_tool_calls,
            _advise_validator(advise_tool_name),
        ),
    )
    snapshot = await response_snapshot(state=state, request=request, snapshot=snapshot)
    result = snapshot.result
    if result is None:  # pragma: no cover - retry_complete returns an accepted terminal snapshot
        raise RuntimeError("advisor completion ended without an accepted result")
    state.threads[ADVISOR_THREAD].extend(snapshot.messages)
    state.threads[ADVISOR_THREAD].extend(await server_tools.execute_calls(state, request, result.message))
    return result, advise_tool_name


def _main_artifact(content: str) -> ChatMessage:
    return ChatMessage(role="developer", content=content, memory={ADVISOR_THREAD: {"artifact": True}})


def _aborted_main_output(call: ChatToolCall) -> ChatMessage:
    return ChatMessage(
        role="tool",
        tool_call_id=call.id,
        content=ABORTED_TOOL_OUTPUT,
        memory={ADVISOR_THREAD: {"artifact": True}},
    )


async def _finalize_phase(
    state: State,
    *,
    advice: str | None,
    note: str | None,
) -> None:
    phase = _remove_phase_message(state)
    state.threads.disable(ADVISOR_THREAD)
    state.threads.unblock("main", by=ADVISOR_THREAD)

    prefix = "[advisor]"
    if phase == "before_tool_call" and advice is not None:
        open_calls = state.open_calls("main")
        state.threads["main"].extend(_aborted_main_output(call) for call in open_calls)
        names = ", ".join(call.name for call in open_calls)
        prefix = f"[advisor] blocked tool call(s): {names}."
        state.threads["main"].append(_main_artifact(advice))
    elif phase == "before_return" and advice is not None:
        prefix = "[advisor] blocked return."
        state.threads["main"].append(_main_artifact(advice))
    elif phase == "after_tool_call" and advice is not None:
        state.threads["main"].append(_main_artifact(advice))

    text = _annotation_text(prefix, advice, note)
    if text is not None:
        await _emit_annotation(state, text)


async def _continue_phase(state: State, main_request: ChatCompletionRequest) -> bool:
    marker = _active_phase(state.threads[ADVISOR_THREAD])
    if marker is None:
        raise RuntimeError("active advisor thread is missing its phase marker")
    phase = marker[1]

    while True:
        try:
            result, advise_tool_name = await _complete_advisor(state, main_request)
        except CompletionBudgetExhaustedError:
            budget = state.svcs.get(CompletionBudget)
            logger.info("response.advisor.skipped", phase=phase, reason="budget_exhausted", remaining=budget.remaining)
            await _finalize_phase(state, advice=None, note=None)
            return True
        except RetryLimitExceededError:
            logger.warning("response.advisor.skipped", phase=phase, reason="retry_limit_exceeded")
            await _finalize_phase(state, advice=None, note=None)
            return True

        advise_call = next((call for call in result.message.tool_calls if call.name == advise_tool_name), None)
        if advise_call is not None:
            advice, note = _advice_fields(advise_call)
            state.threads[ADVISOR_THREAD].append(ChatMessage(role="tool", tool_call_id=advise_call.id, content=ADVISOR_TOOL_OUTPUT))
            logger.info(
                "response.advisor.result",
                advice_chars=len(advice) if advice else 0,
                applied=advice is not None,
                has_note=note is not None,
                phase=phase,
            )
            await _finalize_phase(state, advice=advice, note=note)
            return True

        open_calls = state.open_calls(ADVISOR_THREAD)
        logger.info(
            "response.advisor.exploration",
            phase=phase,
            tool_names=[call.name for call in result.message.tool_calls],
            awaiting_client=bool(open_calls),
        )
        if open_calls:
            return False


async def _start_and_continue_phase(
    state: State,
    main_request: ChatCompletionRequest,
    phase: str,
) -> bool:
    _start_phase(state, phase, main_request)
    logger.info("response.advisor.phase", phase=phase, main_model=main_request.model)
    return await _continue_phase(state, main_request)


def _main_can_run(state: State) -> bool:
    if "main" not in state.threads.active or state.open_calls("main"):
        return False
    main = state.threads["main"]
    return not main or not main[-1].is_assistant()


def _after_tool_phase_pending(state: State) -> bool:
    if "main" not in state.threads.active or state.open_calls("main"):
        return False
    main = state.threads["main"]
    return bool(main) and main[-1].is_tool()


@bus.listen("response.user_turn")
async def interrupt_advisor(state: State, *, next) -> None:
    advisor = state.threads.get(ADVISOR_THREAD)
    marker = None if advisor is None else _active_phase(advisor)
    if marker is not None:
        _append_main_update(state)
        logger.info("response.advisor.interrupted", phase=marker[1], reason="user_turn")
        await _finalize_phase(state, advice=None, note=None)
    await next(state=state)


@bus.listen("response.loop")
async def resume_advisor(state: State, *, next) -> ChatCompletionResult | None:
    handled_phase = False
    if ADVISOR_THREAD in state.threads.active:
        if state.open_calls(ADVISOR_THREAD):
            return None
        main_request = await response_request(state=state)
        handled_phase = True
        if not await _continue_phase(state, main_request):
            return None
    elif _after_tool_phase_pending(state):
        main_request = await response_request(state=state)
        handled_phase = True
        if not await _start_and_continue_phase(state, main_request, "after_tool_call"):
            return None

    if handled_phase and not _main_can_run(state):
        return None
    return await next(state=state)


@bus.listen("response.turn")
async def advise_turn(
    state: State,
    request: ChatCompletionRequest,
    *,
    next,
) -> ChatCompletionResult:
    result = await next(state=state, request=request)
    if result.finish_reason == ChatFinishReason.TOOL_CALLS:
        if state.open_calls("main"):
            await _start_and_continue_phase(state, request, "before_tool_call")
        elif state.threads["main"] and state.threads["main"][-1].is_tool():
            await _start_and_continue_phase(state, request, "after_tool_call")
    elif result.finish_reason == ChatFinishReason.STOP and not result.message.tool_calls and not state.open_calls("main"):
        await _start_and_continue_phase(state, request, "before_return")
    return result


__all__ = [
    "ABORTED_TOOL_OUTPUT",
    "ADVISE_TOOL_NAME",
    "ADVISOR_THREAD",
    "ADVISOR_TOOL_OUTPUT",
]
