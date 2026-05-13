from __future__ import annotations

import secrets
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum

import msgspec
import structlog

from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.keyring import SealingKeyring
from plap.llms.chat import (
    ChatCompletionRequest,
    ChatFunctionTool,
    ChatMessage,
    ChatResponseFormat,
    ChatTool,
    ChatToolCall,
    ChatToolChoiceFunction,
    ChatUsage,
    IChatCompletionClient,
)
from plap.logging import bound_context, log_debug, log_payload
from plap.responses.contracts import (
    FunctionTool,
    ResponseFunctionCallItem,
    ResponseReasoningItem,
    SummaryTextContent,
)
from plap.responses.ingest import (
    SealedCallID,
    compact_transcript,
    content_hash_prefix,
    render_budgeted_spans,
    seal_call_id,
    seal_reasoning_payload,
)
from plap.responses.io import ReasoningDraft, ResponseEventIO
from plap.responses.models import (
    Actor,
    ChatMessageSpan,
    MutableQueues,
    ReasoningPayload,
    Side,
    StateMessage,
    StateToolCall,
    TempMainParts,
    TranscriptMessage,
    UsageLedger,
)
from plap.responses.tokens import measure_prompt_tokens
from plap.responses.tools import ToolPolicy, normalize_function_tool
from plap.responses.tools.mcp import IServerToolExecutor
from plap.settings import RuntimeActorConfig, RuntimeModelProfileConfig

HELD_CLIENT_TOOL_PLACEHOLDER = "This tool call was not executed."
DEBATE_STEP_MAX_ATTEMPTS = 3
CALLED_TOOL_DEFINITIONS_HEADER = "Tool definitions for tools used by the proposed next step:"
logger = structlog.get_logger(__name__)


def _debug_reasoning_summary(*, enabled: bool, texts: Sequence[str | None]) -> list[SummaryTextContent]:
    if not enabled:
        return []
    text = "\n\n".join(part for part in texts if part)
    if not text:
        return []
    return [SummaryTextContent(text=text, type="summary_text")]


def _assistant_output_text(messages: Sequence[StateMessage]) -> list[str | None]:
    return [message.content for message in messages if message.role == "assistant"]


def _single_text(value: str | None) -> list[str | None]:
    return [value]


def _debate_unavailable_error(*, reason: str, private_message: str, cause: BaseException | None = None) -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=503,
            type="server_error",
            code="temporarily_unavailable",
            message="Response generation is temporarily unavailable.",
        ),
        private=PrivateError(
            event="response.unavailable",
            reason=reason,
            message=private_message,
            level=ErrorLevel.WARNING,
            cause=cause,
        ),
    )


def _debate_internal_error(*, reason: str, private_message: str, cause: BaseException | None = None) -> PlapError:
    return PlapError(
        public=None,
        private=PrivateError(
            event="response.internal_error",
            reason=reason,
            message=private_message,
            level=ErrorLevel.ERROR,
            cause=cause,
        ),
    )


def _is_retryable_debate_error(exc: PlapError) -> bool:
    return exc.private.reason in {
        "reviewer_reopen_requires_note",
        "arbitrator_note_required",
        "decision_missing_content",
        "decision_invalid_tail_marker",
        "debate_tool_arguments_invalid_json",
        "debate_tool_arguments_not_object",
    }


async def _retry_debate_step(actor: str, operation) -> object:
    attempts = 0
    while True:
        try:
            return await operation()
        except PlapError as exc:
            if not _is_retryable_debate_error(exc):
                raise
            attempts += 1
            log_debug(
                logger,
                "debate.step.retry",
                actor=actor,
                attempt=attempts,
                max_attempts=DEBATE_STEP_MAX_ATTEMPTS,
                reason=exc.private.reason,
            )
            if attempts >= DEBATE_STEP_MAX_ATTEMPTS:
                raise


REVIEWER_DEVELOPER_PROMPT = """You are checking whether the current proposed next step should be returned now.

You will see:
- the conversation transcript
- the current proposed next step
- on later rounds, the latest response note and possibly a previous next-step note

Definitions:
- current proposed next step: the exact next thing that would be returned now if accepted,
  which may be a direct user-facing message, a tool call, or a combination of both
- response note: another model's reply to the latest review note; it may agree or disagree
- next-step note: a short note about what the next round should focus on

Do not return JSON.

Decision format:
- Your final non-empty line is the decision wrapper line.
- That final line may include a prefix, but its last token must be exactly ACCEPT or REOPEN.
- Examples of valid final lines:
  ACCEPT
  Decision: REOPEN
  Final decision: ACCEPT
- Put no review-note text on the final line. Put all review-note text above it.

Use available tools when they help.
You only have access here to a restricted safe subset of tools. You may also
receive a user message titled `Tool definitions for tools used by the proposed
next step`. If present, it contains tool definitions for tools that are available
to the normal main step for this request and that are already used in
the proposed next step. Use it to understand what those proposed tool calls mean
and whether they are appropriate. The fact that one of those tools is not
callable in this debate step does not mean the normal main step lacks
it. Do not reject or criticize a proposed next step merely because one of those
tools is not callable in this debate step.

Use:
- `ACCEPT` if the current proposed next step is the correct next step to do or return now exactly as-is
- `REOPEN` if another round is needed

If you use `REOPEN`, you MUST write one short review note above the final line saying what seems wrong,
missing, unsupported, or risky about the current proposed next step.
"""

MAIN_DEBATE_DEVELOPER_PROMPT = """You are writing a response note about the current proposed next step.

You will see:
- the full conversation context
- the current proposed next step
- the latest review note

Definitions:
- current proposed next step: the exact next thing that would be returned now if accepted,
  which may be a direct user-facing message, a tool call, or a combination of both
- review note: another model's critique of that next step; it may be correct or incorrect

Write one short response note. The response note should explain, from your own judgment:
- what the review note got right
- what it got wrong
- what matters most for deciding whether the current proposed next step is the correct next step to do or return now

Do not write a replacement answer for the user.
Do not decide whether the current proposed next step should be sent.
You may agree, partly agree, or disagree with the review note.
Use available tools when they help.
You only have access here to a restricted safe subset of tools. You may also
receive a user message titled `Tool definitions for tools used by the proposed
next step`. If present, it contains tool definitions for tools that are available
to the normal main step for this request and that are already used in
the proposed next step. Use it to understand what those proposed tool calls mean
and whether they are appropriate. The fact that one of those tools is not
callable in this debate step does not mean the normal main step lacks
it. Do not reject or criticize a proposed next step merely because one of those
tools is not callable in this debate step.
"""

ARBITRATOR_DEVELOPER_PROMPT = """You are deciding what happens to the current proposed next step.

You will see:
- the conversation transcript
- the current proposed next step
- the latest review note
- the latest response note

Definitions:
- current proposed next step: the exact next thing that would be returned now if accepted,
  which may be a direct user-facing message, a tool call, or a combination of both
- review note: a short note explaining what seems wrong, missing, unsupported, or risky about that next step
- response note: a short note replying to the review note from independent judgment
- next-step note: a short note telling the next round what to focus on

Do not return JSON.

Decision format:
- Your final non-empty line is the decision wrapper line.
- That final line may include a prefix, but its last token must be exactly ACCEPT, REVISE, or REOPEN.
- Examples of valid final lines:
  ACCEPT
  Decision: REVISE
  Final decision: REOPEN
- Put no note text on the final line. Put all note text above it.

Use available tools when they help.
You only have access here to a restricted safe subset of tools. You may also
receive a user message titled `Tool definitions for tools used by the proposed
next step`. If present, it contains tool definitions for tools that are available
to the normal main step for this request and that are already used in
the proposed next step. Use it to understand what those proposed tool calls mean
and whether they are appropriate. The fact that one of those tools is not
callable in this debate step does not mean the normal main step lacks
it. Do not reject or criticize a proposed next step merely because one of those
tools is not callable in this debate step.

Use:
- `ACCEPT` if the current proposed next step is the correct next step to do or return now exactly as-is
- `REVISE` if the current proposed next step should not be sent and the normal main step should try again from scratch
- `REOPEN` if another review/response round is needed

If you use `REVISE`:
- the current proposed next step will not be sent
- the response note will not be sent
- you MUST write one short next-step note above the final line
- that next-step note will be sent to the normal main step, which will choose and write a fresh next step from scratch
- write from the perspective of the main step, unlike you the main step does not know of "review," "reviewer," "arbitrator," "proposed next step," "decision."
- all the main step needs to know is what to do next and what went wrong

If you use `REOPEN`:
- the current proposed next step will not be sent
- the response note will not be sent
- you MUST write one short next-step note above the final line
- that next-step note will be sent into another review/response round
"""


class ReviewerActionType(StrEnum):
    ACCEPT = "accept"
    REOPEN = "reopen"


class ArbitratorActionType(StrEnum):
    ACCEPT = "accept"
    REVISE = "revise"
    REOPEN = "reopen"


class ReviewerDecision(msgspec.Struct, frozen=True):
    action: ReviewerActionType
    note: str | None = None


class ArbitratorDecision(msgspec.Struct, frozen=True):
    action: ArbitratorActionType
    note: str | None = None


@dataclass(slots=True)
class ActorFinished:
    messages: list[StateMessage]
    assistant: StateMessage
    usage: ChatUsage | None
    service_tier: str | None


@dataclass(slots=True)
class ActorAwaitingClientTool:
    messages: list[StateMessage]
    assistant: StateMessage
    tool_calls: list[ChatToolCall]
    usage: ChatUsage | None
    service_tier: str | None


class DebateResult(StrEnum):
    COMPLETED = "completed"
    CONTINUE_MAIN = "continue_main"


def debate_safe_surface(
    tools: Sequence[FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    server_executors: Mapping[str, IServerToolExecutor],
) -> tuple[tuple[FunctionTool, ...], dict[str, ToolPolicy], dict[str, IServerToolExecutor]]:
    safe_tools: list[FunctionTool] = []
    safe_policies: dict[str, ToolPolicy] = {}
    safe_executors: dict[str, IServerToolExecutor] = {}
    for tool in tools:
        policy = tool_policies.get(tool.name)
        if policy is None or policy.effect_class != "safe":
            continue
        safe_tools.append(tool)
        safe_policies[tool.name] = policy
        executor = server_executors.get(tool.name)
        if executor is not None:
            safe_executors[tool.name] = executor
    return tuple(safe_tools), safe_policies, safe_executors


def build_completion_request(
    *,
    actor: str,
    actor_config: RuntimeActorConfig,
    request,
    messages: Sequence[ChatMessage],
    tools: Sequence[FunctionTool],
    tool_choice: ChatToolChoiceFunction | str | None,
    response_format: ChatResponseFormat | None,
    prompt_cache_key_base: str | None,
    max_completion_tokens: int | None,
) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=actor_config.model,
        messages=list(messages),
        tools=[
            ChatTool(
                function=ChatFunctionTool(
                    description=tool.description,
                    name=tool.name,
                    parameters=tool.parameters,
                    strict=tool.strict,
                )
            )
            for tool in tools
        ],
        tool_choice=tool_choice,
        parallel_tool_calls=request.parallel_tool_calls,
        response_format=response_format,
        max_completion_tokens=actor_config.cap_max_completion_tokens(max_completion_tokens),
        temperature=request.temperature,
        top_p=request.top_p,
        top_logprobs=request.top_logprobs,
        reasoning_effort=actor_config.reasoning_effort,
        prompt_cache_key=None if prompt_cache_key_base is None else f"{prompt_cache_key_base}|{actor}",
        service_tier=actor_config.service_tier,
        user=None,
    )


_REVIEWER_ACTIONS_BY_TOKEN = {
    "ACCEPT": ReviewerActionType.ACCEPT,
    "REOPEN": ReviewerActionType.REOPEN,
}

_ARBITRATOR_ACTIONS_BY_TOKEN = {
    "ACCEPT": ArbitratorActionType.ACCEPT,
    "REVISE": ArbitratorActionType.REVISE,
    "REOPEN": ArbitratorActionType.REOPEN,
}


def _tail_decision(message: StateMessage, *, label: str, allowed_actions: Mapping[str, object]) -> tuple[object, str | None]:
    if message.content is None or not message.content.strip():
        raise _debate_unavailable_error(reason="decision_missing_content", private_message=f"{label} is missing content")
    lines = message.content.rstrip().splitlines()
    for index in range(len(lines) - 1, -1, -1):
        decision_line = lines[index].strip()
        if not decision_line:
            continue
        action_token = decision_line.split()[-1]
        action = allowed_actions.get(action_token)
        if action is None:
            expected = ", ".join(allowed_actions)
            raise _debate_unavailable_error(
                reason="decision_invalid_tail_marker",
                private_message=f"{label} final line must end with one of {expected}",
            )
        note = "\n".join(lines[:index]).strip() or None
        return action, note
    raise _debate_unavailable_error(reason="decision_missing_content", private_message=f"{label} is missing content")


def parse_reviewer_decision(message: StateMessage) -> ReviewerDecision:
    action, note = _tail_decision(message, label="reviewer decision", allowed_actions=_REVIEWER_ACTIONS_BY_TOKEN)
    if action == ReviewerActionType.REOPEN and not note:
        raise _debate_unavailable_error(reason="reviewer_reopen_requires_note", private_message="reviewer reopen requires note")
    if action == ReviewerActionType.ACCEPT:
        return ReviewerDecision(action=ReviewerActionType.ACCEPT)
    return ReviewerDecision(action=ReviewerActionType.REOPEN, note=note)


def parse_arbitrator_decision(message: StateMessage) -> ArbitratorDecision:
    action, note = _tail_decision(message, label="arbitrator decision", allowed_actions=_ARBITRATOR_ACTIONS_BY_TOKEN)
    if action in {ArbitratorActionType.REVISE, ArbitratorActionType.REOPEN} and not note:
        raise _debate_unavailable_error(reason="arbitrator_note_required", private_message="arbitrator note is required")
    if action == ArbitratorActionType.ACCEPT:
        return ArbitratorDecision(action=ArbitratorActionType.ACCEPT)
    return ArbitratorDecision(action=action, note=note)


def _compact_candidate(
    parts: TempMainParts,
) -> dict[str, object]:
    if parts.held_candidate is None:
        raise _debate_internal_error(reason="held_candidate_missing", private_message="debate temp state is missing held candidate")
    candidate = parts.held_candidate.message
    outputs = {
        row.message.tool_call_id: row.message.content_text() or ""
        for row in parts.held_hidden_tool_rows
        if row.message.tool_call_id is not None and (row.message.content_text() or "") != HELD_CLIENT_TOOL_PLACEHOLDER
    }
    value: dict[str, object] = {"role": "assistant"}
    if candidate.content_text() is not None:
        value["content"] = candidate.content_text()
    if candidate.tool_calls:
        compact_tool_calls: list[dict[str, object]] = []
        for call in candidate.tool_calls:
            compact_call: dict[str, object] = {
                "name": call.name,
                "arguments": call.arguments_value(),
            }
            output = outputs.get(call.id)
            if output is not None:
                compact_call["output"] = output
            compact_tool_calls.append(compact_call)
        value["tool_calls"] = compact_tool_calls
    return value


def _candidate_called_tool_definitions_message(
    parts: TempMainParts,
    *,
    normal_tools: Sequence[FunctionTool],
    debate_tool_policies: Mapping[str, ToolPolicy],
) -> ChatMessage | None:
    if parts.held_candidate is None:
        raise _debate_internal_error(reason="held_candidate_missing", private_message="debate temp state is missing held candidate")
    tools_by_name = {tool.name: tool for tool in normal_tools}
    definitions: list[dict[str, object]] = []
    seen_tool_names: set[str] = set()
    for call in parts.held_candidate.message.tool_calls:
        if call.name in seen_tool_names or call.name in debate_tool_policies:
            continue
        seen_tool_names.add(call.name)
        tool = tools_by_name.get(call.name)
        if tool is None:
            continue
        definitions.append(normalize_function_tool(tool))
    if not definitions:
        return None
    return ChatMessage(
        role="user",
        content=f"{CALLED_TOOL_DEFINITIONS_HEADER}\n{_json_text(definitions)}",
    )


def _transcript_wrapper(transcript: Sequence[TranscriptMessage]) -> ChatMessage:
    return ChatMessage(role="user", content=f"Conversation transcript:\n{_json_text([message.to_primitive() for message in transcript])}")


def _transcript_rows(
    spans: tuple[ChatMessageSpan, ...],
    *,
    main_developer_message: StateMessage,
) -> tuple[TranscriptMessage, ...]:
    return (
        main_developer_message.to_transcript_message(),
        *compact_transcript(spans, untrusted=True),
    )


def _measure_budgeted_transcript_tokens(
    spans: tuple[ChatMessageSpan, ...],
    *,
    actor_config: RuntimeActorConfig,
    main_developer_message: StateMessage,
) -> int:
    try:
        transcript = _transcript_rows(spans, main_developer_message=main_developer_message)
        return measure_prompt_tokens([_transcript_wrapper(transcript)], actor_config=actor_config)
    except Exception as exc:
        raise _debate_unavailable_error(
            reason="debate_transcript_tokenizer_failed",
            private_message="debate transcript token measurement failed",
            cause=exc,
        ) from exc


def _budgeted_transcript_message(
    spans: Sequence[ChatMessageSpan],
    *,
    actor_config: RuntimeActorConfig,
    main_developer_message: StateMessage,
    recount_margin: int,
    token_budget: int,
) -> ChatMessage:
    measure = None
    if actor_config.tokenizer_hf_repo is not None:
        measure = lambda rendered: _measure_budgeted_transcript_tokens(
            rendered,
            actor_config=actor_config,
            main_developer_message=main_developer_message,
        )
    transcript = _transcript_rows(
        render_budgeted_spans(
            tuple(spans),
            measure=measure,
            recount_margin=recount_margin,
            token_budget=token_budget,
        ),
        main_developer_message=main_developer_message,
    )
    return _transcript_wrapper(transcript)


def _reviewer_initial_turn(
    parts: TempMainParts,
) -> StateMessage:
    return StateMessage(
        role="user",
        content=(
            "Review the current proposed next step below. Decide whether it is the "
            "correct next thing to do or return now, or reopen with one short review "
            "note.\n\n"
            "Current proposed next step:\n"
            f"{_json_text(_compact_candidate(parts))}"
        ),
    )


def _reviewer_reopen_turn(*, latest_response_note: StateMessage, latest_next_step_note: str | None) -> StateMessage:
    parts = []
    if latest_next_step_note is not None:
        parts.append(f"Previous next-step note:\n{latest_next_step_note}")
    parts.append(f"Latest response note:\n{latest_response_note.content_text() or ''}")
    parts.append("Revisit the current proposed next step and decide whether to accept it or reopen again with a new review note.")
    return StateMessage(role="user", content="\n\n".join(parts))


def _main_debate_turn(*, reviewer_decision: ReviewerDecision) -> StateMessage:
    return StateMessage(
        role="user",
        content=(
            "Latest review note:\n"
            f"{reviewer_decision.note or ''}\n\n"
            "Write one short response note about the current proposed next step. "
            "Focus on whether it is the correct next thing to do or return now."
        ),
    )


def _arbitrator_initial_turn(
    *,
    parts: TempMainParts,
    reviewer_decision: ReviewerDecision,
    latest_response_note: StateMessage,
) -> StateMessage:
    return StateMessage(
        role="user",
        content=(
            "Current proposed next step:\n"
            f"{_json_text(_compact_candidate(parts))}"
            "\n\n"
            "Latest review note:\n"
            f"{reviewer_decision.note or ''}\n\n"
            "Latest response note:\n"
            f"{latest_response_note.content_text() or ''}\n\n"
            "Decide whether to accept the current proposed next step, send one "
            "next-step note back to the normal main step for a fresh retry, or reopen "
            "the review cycle."
        ),
    )


def _arbitrator_reopen_turn(*, reviewer_decision: ReviewerDecision, latest_response_note: StateMessage) -> StateMessage:
    return StateMessage(
        role="user",
        content=(
            "Updated review note:\n"
            f"{reviewer_decision.note or ''}\n\n"
            "Latest response note:\n"
            f"{latest_response_note.content_text() or ''}\n\n"
            "Decide whether to accept the current proposed next step, send one "
            "next-step note back to the normal main step for a fresh retry, or reopen "
            "the review cycle."
        ),
    )


def _thread_waiting_after_tool_output(thread: Sequence[StateMessage]) -> bool:
    return bool(thread) and thread[-1].is_tool()


def _thread_messages(rows: Sequence) -> list[StateMessage]:
    return [row.message for row in rows]


def _latest_assistant(messages: Sequence[StateMessage]) -> StateMessage | None:
    for message in reversed(messages):
        if message.is_assistant():
            return message
    return None


def _latest_reviewer_decision(thread: Sequence[StateMessage]) -> ReviewerDecision | None:
    assistant = _latest_assistant(thread)
    if assistant is None:
        return None
    return parse_reviewer_decision(assistant)


def _latest_arbitrator_note(thread: Sequence[StateMessage]) -> str | None:
    assistant = _latest_assistant(thread)
    if assistant is None:
        return None
    return parse_arbitrator_decision(assistant).note


def _reviewer_round_count(reviewer: Sequence) -> int:
    return sum(1 for row in reviewer if row.message.role == "user")


def _state_message_from_result(message: ChatMessage) -> StateMessage:
    return StateMessage(
        role=message.role,
        content=message.content,
        name=message.name,
        tool_call_id=message.tool_call_id,
        tool_calls=[StateToolCall(id=call.id, name=call.name, arguments=call.arguments) for call in message.tool_calls or ()],
        reasoning_content=message.reasoning_content,
        reasoning_details=list(message.reasoning_details or ()),
    )


def _arguments_object(arguments: str, *, label: str) -> dict[str, object]:
    try:
        value = msgspec.json.decode(arguments.encode())
    except msgspec.DecodeError as exc:
        raise _debate_unavailable_error(
            reason="debate_tool_arguments_invalid_json", private_message=f"{label} must be valid JSON", cause=exc
        ) from exc
    if not isinstance(value, dict):
        raise _debate_unavailable_error(reason="debate_tool_arguments_not_object", private_message=f"{label} must be a JSON object")
    return value


def _json_text(value: object) -> str:
    return msgspec.json.encode(value).decode()


async def _execute_actor_turn(
    *,
    actor_name: str,
    actor_config: RuntimeActorConfig,
    request,
    header_messages: Sequence[ChatMessage],
    turn_messages: list[StateMessage],
    tools: Sequence[FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    server_executors: Mapping[str, IServerToolExecutor],
    chat_completion_client: IChatCompletionClient,
    prompt_cache_key_base: str | None,
    usage_ledger: UsageLedger,
    response_format: ChatResponseFormat | None,
) -> ActorFinished | ActorAwaitingClientTool | None:
    with bound_context(actor=actor_name):
        while True:
            cap = usage_ledger.cap_for(actor_config.public_usage)
            if cap == 0:
                return None
            request_messages = [*header_messages, *(message.to_chat_message() for message in turn_messages)]
            actor_request = build_completion_request(
                actor=actor_name,
                actor_config=actor_config,
                request=request,
                messages=request_messages,
                tools=tools,
                tool_choice=None,
                response_format=response_format,
                prompt_cache_key_base=prompt_cache_key_base,
                max_completion_tokens=cap,
            )
            log_debug(logger, "debate.actor.request", actor=actor_name, max_completion_tokens=cap, tool_count=len(tools))
            log_payload(logger, "debate.actor.request.payload", actor=actor_name, request=asdict(actor_request))
            result = await chat_completion_client.complete(actor_request)
            assistant = _state_message_from_result(result.message)
            turn_messages.append(assistant)
            tool_calls = result.message.tool_calls or []
            log_debug(
                logger,
                "debate.actor.result",
                actor=actor_name,
                finish_reason=result.finish_reason,
                has_content=result.message.content is not None,
                tool_call_count=len(tool_calls),
            )
            log_payload(logger, "debate.actor.result.payload", actor=actor_name, result=asdict(result))
            if not tool_calls:
                return ActorFinished(messages=turn_messages, assistant=assistant, usage=result.usage, service_tier=result.service_tier)

            client_calls: list[ChatToolCall] = []
            server_output_seen = False
            for call in tool_calls:
                policy = tool_policies.get(call.name)
                if policy is None or policy.effect_class != "safe":
                    raise _debate_unavailable_error(
                        reason="debate_actor_called_unsupported_tool", private_message="debate actor called unsupported tool"
                    )
                if policy.source == "server":
                    executor = server_executors.get(call.name)
                    if executor is None:
                        raise _debate_internal_error(
                            reason="debate_server_tool_executor_missing", private_message="debate server tool executor is missing"
                        )
                    output = await executor.call_tool(call.name, _arguments_object(call.arguments, label="debate tool arguments"))
                    turn_messages.append(StateMessage(role="tool", tool_call_id=call.id, content=output))
                    server_output_seen = True
                else:
                    client_calls.append(call)

            if client_calls:
                return ActorAwaitingClientTool(
                    messages=turn_messages,
                    assistant=assistant,
                    tool_calls=client_calls,
                    usage=result.usage,
                    service_tier=result.service_tier,
                )

            if server_output_seen:
                usage_ledger.record_hidden(actor_config.public_usage, result.usage)


async def run_reviewer_turn(
    *,
    state: MutableQueues,
    parts: TempMainParts,
    main_developer_message: StateMessage,
    profile: RuntimeModelProfileConfig,
    request,
    normal_tools: Sequence[FunctionTool],
    tools: Sequence[FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    server_executors: Mapping[str, IServerToolExecutor],
    chat_completion_client: IChatCompletionClient,
    prompt_cache_key_base: str | None,
    usage_ledger: UsageLedger,
) -> ActorFinished | ActorAwaitingClientTool | None:
    thread = _thread_messages(state.reviewer)
    header_messages: list[ChatMessage] = [
        ChatMessage(role="developer", content=REVIEWER_DEVELOPER_PROMPT),
        _budgeted_transcript_message(
            state.main_context,
            actor_config=profile.reviewer,
            main_developer_message=main_developer_message,
            recount_margin=profile.transcript_recount_margin,
            token_budget=profile.reviewer_transcript_token_budget,
        ),
    ]
    tool_definitions_message = _candidate_called_tool_definitions_message(
        parts,
        normal_tools=normal_tools,
        debate_tool_policies=tool_policies,
    )
    if tool_definitions_message is not None:
        header_messages.append(tool_definitions_message)
    header_messages.extend(message.to_chat_message() for message in thread)
    if _thread_waiting_after_tool_output(thread):
        turn_messages: list[StateMessage] = []
    elif thread:
        latest_response_note = _latest_assistant([row.message for row in parts.remaining_temp_rows])
        latest_next_step_note = _latest_arbitrator_note(_thread_messages(state.arbitrator))
        if latest_response_note is None:
            raise _debate_internal_error(
                reason="reviewer_reopen_missing_latest_response_note", private_message="reviewer reopen is missing latest response note"
            )
        turn_messages = [
            _reviewer_reopen_turn(
                latest_response_note=latest_response_note,
                latest_next_step_note=latest_next_step_note,
            )
        ]
    else:
        turn_messages = [_reviewer_initial_turn(parts)]
    return await _execute_actor_turn(
        actor_name=Side.REVIEWER.value,
        actor_config=profile.reviewer,
        request=request,
        header_messages=header_messages,
        turn_messages=turn_messages,
        tools=tools,
        tool_policies=tool_policies,
        server_executors=server_executors,
        chat_completion_client=chat_completion_client,
        prompt_cache_key_base=prompt_cache_key_base,
        usage_ledger=usage_ledger,
        response_format=None,
    )


async def run_main_debate_turn(
    *,
    state: MutableQueues,
    parts: TempMainParts,
    profile: RuntimeModelProfileConfig,
    request,
    normal_tools: Sequence[FunctionTool],
    tools: Sequence[FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    server_executors: Mapping[str, IServerToolExecutor],
    chat_completion_client: IChatCompletionClient,
    prompt_cache_key_base: str | None,
    usage_ledger: UsageLedger,
) -> ActorFinished | ActorAwaitingClientTool | None:
    thread = [row.message for row in parts.remaining_temp_rows]
    header_messages: list[ChatMessage] = [
        ChatMessage(role="developer", content=MAIN_DEBATE_DEVELOPER_PROMPT),
        *state.render_effective_main_context(include_citation=False),
    ]
    tool_definitions_message = _candidate_called_tool_definitions_message(
        parts,
        normal_tools=normal_tools,
        debate_tool_policies=tool_policies,
    )
    if tool_definitions_message is not None:
        header_messages.append(tool_definitions_message)
    if _thread_waiting_after_tool_output(thread):
        turn_messages: list[StateMessage] = []
    else:
        reviewer_decision = _latest_reviewer_decision(_thread_messages(state.reviewer))
        if reviewer_decision is None:
            raise _debate_internal_error(
                reason="main_debate_missing_reviewer_decision", private_message="main debate is missing reviewer decision"
            )
        turn_messages = [_main_debate_turn(reviewer_decision=reviewer_decision)]
    return await _execute_actor_turn(
        actor_name=Actor.MAIN_DEBATE.value,
        actor_config=profile.main_debate,
        request=request,
        header_messages=header_messages,
        turn_messages=turn_messages,
        tools=tools,
        tool_policies=tool_policies,
        server_executors=server_executors,
        chat_completion_client=chat_completion_client,
        prompt_cache_key_base=prompt_cache_key_base,
        usage_ledger=usage_ledger,
        response_format=None,
    )


async def run_arbitrator_turn(
    *,
    state: MutableQueues,
    parts: TempMainParts,
    main_developer_message: StateMessage,
    profile: RuntimeModelProfileConfig,
    request,
    normal_tools: Sequence[FunctionTool],
    tools: Sequence[FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    server_executors: Mapping[str, IServerToolExecutor],
    chat_completion_client: IChatCompletionClient,
    prompt_cache_key_base: str | None,
    usage_ledger: UsageLedger,
) -> ActorFinished | ActorAwaitingClientTool | None:
    reviewer_decision = _latest_reviewer_decision(_thread_messages(state.reviewer))
    latest_response_note = _latest_assistant([row.message for row in parts.remaining_temp_rows])
    if reviewer_decision is None or latest_response_note is None:
        raise _debate_internal_error(
            reason="final_decision_missing_review_or_response_note",
            private_message="final decision step is missing review or response note",
        )
    thread = _thread_messages(state.arbitrator)
    header_messages: list[ChatMessage] = [
        ChatMessage(role="developer", content=ARBITRATOR_DEVELOPER_PROMPT),
        _budgeted_transcript_message(
            state.main_context,
            actor_config=profile.arbitrator,
            main_developer_message=main_developer_message,
            recount_margin=profile.transcript_recount_margin,
            token_budget=profile.arbitrator_transcript_token_budget,
        ),
    ]
    tool_definitions_message = _candidate_called_tool_definitions_message(
        parts,
        normal_tools=normal_tools,
        debate_tool_policies=tool_policies,
    )
    if tool_definitions_message is not None:
        header_messages.append(tool_definitions_message)
    header_messages.extend(message.to_chat_message() for message in thread)
    if _thread_waiting_after_tool_output(thread):
        turn_messages: list[StateMessage] = []
    elif thread:
        turn_messages = [
            _arbitrator_reopen_turn(
                reviewer_decision=reviewer_decision,
                latest_response_note=latest_response_note,
            )
        ]
    else:
        turn_messages = [
            _arbitrator_initial_turn(
                parts=parts,
                reviewer_decision=reviewer_decision,
                latest_response_note=latest_response_note,
            )
        ]
    return await _execute_actor_turn(
        actor_name=Side.ARBITRATOR.value,
        actor_config=profile.arbitrator,
        request=request,
        header_messages=header_messages,
        turn_messages=turn_messages,
        tools=tools,
        tool_policies=tool_policies,
        server_executors=server_executors,
        chat_completion_client=chat_completion_client,
        prompt_cache_key_base=prompt_cache_key_base,
        usage_ledger=usage_ledger,
        response_format=None,
    )


def _held_candidate_messages(
    *,
    assistant: StateMessage,
    tool_calls: Sequence[ChatToolCall],
    server_outputs: Mapping[int, str],
) -> list[StateMessage]:
    messages: list[StateMessage] = [assistant]
    for index, call in enumerate(tool_calls):
        messages.append(
            StateMessage(
                role="tool",
                tool_call_id=call.id,
                content=server_outputs.get(index, HELD_CLIENT_TOOL_PLACEHOLDER),
            )
    )
    return messages


async def start_debate_from_candidate(
    *,
    state: MutableQueues,
    out: ResponseEventIO,
    debug_debate_summaries: bool,
    keyring: SealingKeyring,
    assistant: StateMessage,
    tool_calls: Sequence[ChatToolCall],
    server_outputs: Mapping[int, str],
    draft: ReasoningDraft | None = None,
) -> None:
    messages = _held_candidate_messages(assistant=assistant, tool_calls=tool_calls, server_outputs=server_outputs)
    if draft is None:
        await _persist_temp_turn(
            state=state,
            side=Side.MAIN,
            messages=messages,
            continuation_side=Side.REVIEWER,
            out=out,
            debug_debate_summaries=debug_debate_summaries,
            keyring=keyring,
        )
        return

    payload = ReasoningPayload(
        side=Side.MAIN,
        temp=True,
        continuation_side=Side.REVIEWER,
        messages=tuple(messages),
    )
    await out.complete_reasoning_draft(
        draft,
        ResponseReasoningItem(
            encrypted_content=seal_reasoning_payload(payload, keyring=keyring),
            id=draft.item_id,
            status="completed",
            summary=[],
            type="reasoning",
        ),
    )
    for message in messages:
        state.append_main_temp(message)
    state.set_continuation(Side.REVIEWER, in_temp_debate=True)


async def publish_accepted_candidate(
    *,
    state: MutableQueues,
    out: ResponseEventIO,
    debug_debate_summaries: bool,
    keyring: SealingKeyring,
) -> None:
    parts = state.temp_main_parts()
    if parts.held_candidate is None:
        raise _debate_internal_error(reason="held_candidate_missing", private_message="debate temp state is missing held candidate")
    candidate = parts.held_candidate.message
    hidden_outputs = {
        row.message.tool_call_id: row.message.content_text() or ""
        for row in parts.held_hidden_tool_rows
        if row.message.tool_call_id is not None
    }
    emitted_outputs = {call_id: output for call_id, output in hidden_outputs.items() if output != HELD_CLIENT_TOOL_PLACEHOLDER}
    published = await out.publish_main_candidate(
        candidate=candidate,
        keyring=keyring,
        server_outputs=emitted_outputs,
        reasoning_summary=_debug_reasoning_summary(enabled=debug_debate_summaries, texts=_single_text(candidate.content)),
    )

    state.append_main_stable(candidate.without_reasoning(), content_hash=published.assistant_hash)

    for call in candidate.tool_calls:
        output = hidden_outputs.get(call.id)
        if output is None or output == HELD_CLIENT_TOOL_PLACEHOLDER:
            continue
        state.append_main_stable(StateMessage(role="tool", tool_call_id=call.id, content=output))

    state.clear_debate()


def _wrap_revise_note_for_main(note: str) -> str:
    return (
        "Internal retry guidance for the next fresh answer only.\n"
        "This is not a user-facing message.\n"
        "Do not quote it, acknowledge it, apologize for it, or reply to it directly.\n\n"
        f"{note}"
    )


async def resume_main_with_revise_bundle(
    *,
    state: MutableQueues,
    out: ResponseEventIO,
    debug_debate_summaries: bool,
    keyring: SealingKeyring,
    note: str,
) -> None:
    parts = state.temp_main_parts()
    if parts.held_candidate is None:
        raise _debate_internal_error(reason="revise_requires_held_candidate", private_message="revise requires held candidate state")

    note_message = StateMessage(role="assistant", content=_wrap_revise_note_for_main(note))
    bundled_messages = (
        parts.held_candidate.message,
        *[row.message for row in parts.held_hidden_tool_rows],
        note_message,
    )
    reasoning_payload = ReasoningPayload(
        side=Side.MAIN,
        temp=False,
        continuation_side=Side.MAIN,
        messages=bundled_messages,
    )
    await out.output(
        ResponseReasoningItem(
            encrypted_content=seal_reasoning_payload(reasoning_payload, keyring=keyring),
            id=f"rs_{secrets.token_urlsafe(18)}",
            status="completed",
            summary=_debug_reasoning_summary(enabled=debug_debate_summaries, texts=_single_text(note)),
            type="reasoning",
        )
    )
    state.append_main_stable(parts.held_candidate.message, content_hash=parts.held_candidate.content_hash)
    for row in parts.held_hidden_tool_rows:
        state.append_main_stable(row.message, content_hash=row.content_hash)
    state.append_main_stable(note_message)
    state.clear_debate()


async def continue_debate(
    *,
    state: MutableQueues,
    out: ResponseEventIO,
    main_developer_message: StateMessage,
    request,
    profile: RuntimeModelProfileConfig,
    debug_debate_summaries: bool,
    keyring: SealingKeyring,
    tools: Sequence[FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    server_executors: Mapping[str, IServerToolExecutor],
    chat_completion_client: IChatCompletionClient,
    prompt_cache_key_base: str | None,
    usage_ledger: UsageLedger,
    held_anchor_index: int | None = None,
) -> DebateResult:
    safe_tools, safe_tool_policies, safe_server_executors = debate_safe_surface(tools, tool_policies, server_executors)
    parts = state.temp_main_parts()
    if parts.held_candidate is None:
        raise _debate_internal_error(reason="held_candidate_missing", private_message="debate temp state is missing held candidate")

    if profile.debate_max_rounds == 0:
        if held_anchor_index is not None:
            usage_ledger.use_hidden_as_anchor(held_anchor_index)
        await publish_accepted_candidate(
            state=state,
            out=out,
            debug_debate_summaries=debug_debate_summaries,
            keyring=keyring,
        )
        await out.completed(service_tier=request.service_tier, usage=usage_ledger.to_response_usage())
        return DebateResult.COMPLETED

    while True:
        actor = state.current_actor()
        log_debug(logger, "debate.turn", actor=actor, held_anchor_index=held_anchor_index)

        if actor == Actor.REVIEWER:

            async def reviewer_step(parts=parts):
                outcome = await run_reviewer_turn(
                    state=state,
                    parts=parts,
                    main_developer_message=main_developer_message,
                    profile=profile,
                    request=request,
                    normal_tools=tools,
                    tools=safe_tools,
                    tool_policies=safe_tool_policies,
                    server_executors=safe_server_executors,
                    chat_completion_client=chat_completion_client,
                    prompt_cache_key_base=prompt_cache_key_base,
                    usage_ledger=usage_ledger,
                )
                if outcome is None or isinstance(outcome, ActorAwaitingClientTool):
                    return outcome, None
                return outcome, parse_reviewer_decision(outcome.assistant)

            outcome, decision = await _retry_debate_step(Actor.REVIEWER.value, reviewer_step)
            if outcome is None:
                if held_anchor_index is not None and usage_ledger.anchor is None:
                    usage_ledger.use_hidden_as_anchor(held_anchor_index)
                await out.incomplete(service_tier=request.service_tier, usage=usage_ledger.to_response_usage())
                return DebateResult.COMPLETED
            if isinstance(outcome, ActorAwaitingClientTool):
                usage_ledger.set_anchor(outcome.usage)
                await _persist_temp_turn(
                    state=state,
                    side=Side.REVIEWER,
                    messages=outcome.messages,
                    continuation_side=Side.REVIEWER,
                    out=out,
                    debug_debate_summaries=debug_debate_summaries,
                    keyring=keyring,
                )
                await _emit_debate_function_calls(
                    side=Side.REVIEWER,
                    assistant=outcome.assistant,
                    tool_calls=outcome.tool_calls,
                    out=out,
                    keyring=keyring,
                )
                await out.completed(service_tier=request.service_tier, usage=usage_ledger.to_response_usage())
                return DebateResult.COMPLETED

            assert decision is not None
            log_debug(logger, "debate.reviewer.decision", action=decision.action, note=decision.note)
            await _persist_temp_turn(
                state=state,
                side=Side.REVIEWER,
                messages=outcome.messages,
                continuation_side=Side.MAIN,
                out=out,
                debug_debate_summaries=debug_debate_summaries,
                keyring=keyring,
            )
            if decision.action == ReviewerActionType.ACCEPT:
                if held_anchor_index is not None:
                    usage_ledger.record_hidden(profile.reviewer.public_usage, outcome.usage)
                    usage_ledger.use_hidden_as_anchor(held_anchor_index)
                else:
                    usage_ledger.set_anchor(outcome.usage)
                await publish_accepted_candidate(
                    state=state,
                    out=out,
                    debug_debate_summaries=debug_debate_summaries,
                    keyring=keyring,
                )
                await out.completed(service_tier=request.service_tier, usage=usage_ledger.to_response_usage())
                return DebateResult.COMPLETED

            usage_ledger.record_hidden(profile.reviewer.public_usage, outcome.usage)
            parts = state.temp_main_parts()
            continue

        if actor == Actor.MAIN_DEBATE:

            async def main_debate_step(parts=parts):
                return await run_main_debate_turn(
                    state=state,
                    parts=parts,
                    profile=profile,
                    request=request,
                    normal_tools=tools,
                    tools=safe_tools,
                    tool_policies=safe_tool_policies,
                    server_executors=safe_server_executors,
                    chat_completion_client=chat_completion_client,
                    prompt_cache_key_base=prompt_cache_key_base,
                    usage_ledger=usage_ledger,
                )

            outcome = await _retry_debate_step(Actor.MAIN_DEBATE.value, main_debate_step)
            if outcome is None:
                if held_anchor_index is not None and usage_ledger.anchor is None:
                    usage_ledger.use_hidden_as_anchor(held_anchor_index)
                await out.incomplete(service_tier=request.service_tier, usage=usage_ledger.to_response_usage())
                return DebateResult.COMPLETED
            if isinstance(outcome, ActorAwaitingClientTool):
                usage_ledger.set_anchor(outcome.usage)
                await _persist_temp_turn(
                    state=state,
                    side=Side.MAIN,
                    messages=outcome.messages,
                    continuation_side=Side.MAIN,
                    out=out,
                    debug_debate_summaries=debug_debate_summaries,
                    keyring=keyring,
                )
                await _emit_debate_function_calls(
                    side=Side.MAIN,
                    assistant=outcome.assistant,
                    tool_calls=outcome.tool_calls,
                    out=out,
                    keyring=keyring,
                )
                await out.completed(service_tier=request.service_tier, usage=usage_ledger.to_response_usage())
                return DebateResult.COMPLETED

            usage_ledger.record_hidden(profile.main_debate.public_usage, outcome.usage)
            await _persist_temp_turn(
                state=state,
                side=Side.MAIN,
                messages=outcome.messages,
                continuation_side=Side.ARBITRATOR,
                out=out,
                debug_debate_summaries=debug_debate_summaries,
                keyring=keyring,
            )
            parts = state.temp_main_parts()
            continue

        if actor == Actor.ARBITRATOR:

            async def arbitrator_step(parts=parts):
                outcome = await run_arbitrator_turn(
                    state=state,
                    parts=parts,
                    main_developer_message=main_developer_message,
                    profile=profile,
                    request=request,
                    normal_tools=tools,
                    tools=safe_tools,
                    tool_policies=safe_tool_policies,
                    server_executors=safe_server_executors,
                    chat_completion_client=chat_completion_client,
                    prompt_cache_key_base=prompt_cache_key_base,
                    usage_ledger=usage_ledger,
                )
                if outcome is None or isinstance(outcome, ActorAwaitingClientTool):
                    return outcome, None
                return outcome, parse_arbitrator_decision(outcome.assistant)

            outcome, decision = await _retry_debate_step(Actor.ARBITRATOR.value, arbitrator_step)
            if outcome is None:
                if held_anchor_index is not None and usage_ledger.anchor is None:
                    usage_ledger.use_hidden_as_anchor(held_anchor_index)
                await out.incomplete(service_tier=request.service_tier, usage=usage_ledger.to_response_usage())
                return DebateResult.COMPLETED
            if isinstance(outcome, ActorAwaitingClientTool):
                usage_ledger.set_anchor(outcome.usage)
                await _persist_temp_turn(
                    state=state,
                    side=Side.ARBITRATOR,
                    messages=outcome.messages,
                    continuation_side=Side.ARBITRATOR,
                    out=out,
                    debug_debate_summaries=debug_debate_summaries,
                    keyring=keyring,
                )
                await _emit_debate_function_calls(
                    side=Side.ARBITRATOR,
                    assistant=outcome.assistant,
                    tool_calls=outcome.tool_calls,
                    out=out,
                    keyring=keyring,
                )
                await out.completed(service_tier=request.service_tier, usage=usage_ledger.to_response_usage())
                return DebateResult.COMPLETED

            assert decision is not None
            log_debug(logger, "debate.arbitrator.decision", action=decision.action, note=decision.note)
            if decision.action == ArbitratorActionType.REOPEN and _reviewer_round_count(state.reviewer) >= profile.debate_max_rounds:
                decision = ArbitratorDecision(
                    action=ArbitratorActionType.REVISE,
                    note=decision.note,
                )
            continuation = Side.REVIEWER if decision.action == ArbitratorActionType.REOPEN else Side.MAIN
            await _persist_temp_turn(
                state=state,
                side=Side.ARBITRATOR,
                messages=outcome.messages,
                continuation_side=continuation,
                out=out,
                debug_debate_summaries=debug_debate_summaries,
                keyring=keyring,
            )
            if decision.action == ArbitratorActionType.ACCEPT:
                if held_anchor_index is not None:
                    usage_ledger.record_hidden(profile.arbitrator.public_usage, outcome.usage)
                    usage_ledger.use_hidden_as_anchor(held_anchor_index)
                else:
                    usage_ledger.set_anchor(outcome.usage)
                await publish_accepted_candidate(
                    state=state,
                    out=out,
                    debug_debate_summaries=debug_debate_summaries,
                    keyring=keyring,
                )
                await out.completed(service_tier=request.service_tier, usage=usage_ledger.to_response_usage())
                return DebateResult.COMPLETED

            usage_ledger.record_hidden(profile.arbitrator.public_usage, outcome.usage)
            if decision.action == ArbitratorActionType.REVISE:
                await resume_main_with_revise_bundle(
                    state=state,
                    out=out,
                    debug_debate_summaries=debug_debate_summaries,
                    keyring=keyring,
                    note=decision.note or "",
                )
                return DebateResult.CONTINUE_MAIN

            parts = state.temp_main_parts()
            continue

        raise _debate_internal_error(reason="debate_actor_resolution_invalid", private_message="debate actor resolution is invalid")


async def _persist_temp_turn(
    *,
    state: MutableQueues,
    side: Side,
    messages: Sequence[StateMessage],
    continuation_side: Side,
    out: ResponseEventIO,
    debug_debate_summaries: bool,
    keyring: SealingKeyring,
) -> None:
    payload = ReasoningPayload(
        side=side,
        temp=True,
        continuation_side=continuation_side,
        messages=tuple(messages),
    )
    await out.output(
        ResponseReasoningItem(
            encrypted_content=seal_reasoning_payload(payload, keyring=keyring),
            id=f"rs_{secrets.token_urlsafe(18)}",
            status="completed",
            summary=_debug_reasoning_summary(enabled=debug_debate_summaries, texts=_assistant_output_text(messages)),
            type="reasoning",
        )
    )
    if side == Side.MAIN:
        for message in messages:
            state.append_main_temp(message)
    else:
        for message in messages:
            state.append_side(side, message)
    state.set_continuation(continuation_side, in_temp_debate=True)
async def _emit_debate_function_calls(
    *,
    side: Side,
    assistant: StateMessage,
    tool_calls: Sequence[ChatToolCall],
    out: ResponseEventIO,
    keyring: SealingKeyring,
) -> None:
    assistant_hash = assistant.content_hash()
    index_by_id = {call.id: index for index, call in enumerate(assistant.tool_calls)}
    for call in tool_calls:
        index = index_by_id.get(call.id)
        if index is None:
            raise _debate_internal_error(
                reason="debate_assistant_tool_call_missing_from_persisted_state",
                private_message="debate assistant tool call is missing from persisted state",
            )
        await out.output(
            ResponseFunctionCallItem(
                arguments=call.arguments,
                call_id=seal_call_id(
                    SealedCallID(
                        side=side,
                        content_hash_prefix=content_hash_prefix(assistant_hash),
                        tool_call_index=index,
                        upstream_tool_call_id=call.id,
                    ),
                    keyring=keyring,
                ),
                id=f"fc_{secrets.token_urlsafe(18)}",
                name=call.name,
                status="completed",
                type="function_call",
            )
        )
