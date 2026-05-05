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
    OutputTextContent,
    ResponseFunctionCallItem,
    ResponseFunctionCallOutputItem,
    ResponseMessageItem,
    ResponseReasoningItem,
)
from plap.responses.ingest import SealedCallID, content_hash_prefix, seal_call_id, seal_reasoning_payload
from plap.responses.io import ResponseEventIO
from plap.responses.models import (
    Actor,
    MutableQueues,
    ReasoningMessagePatch,
    ReasoningPayload,
    Side,
    StateMessage,
    StateToolCall,
    TempMainParts,
    TranscriptMessage,
    UsageLedger,
    strip_leading_internal_citations,
)
from plap.responses.tools import ToolPolicy
from plap.responses.tools.mcp import IMCPToolProvider
from plap.settings import RuntimeActorConfig, RuntimeModelProfileConfig

HELD_CLIENT_TOOL_PLACEHOLDER = "This tool call was not executed."
DEBATE_STRUCTURED_STEP_MAX_ATTEMPTS = 3
logger = structlog.get_logger(__name__)


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
        "decision_invalid_json",
        "decision_invalid",
        "debate_tool_arguments_invalid_json",
        "debate_tool_arguments_not_object",
    }


async def _retry_structured_debate_step(actor: str, operation) -> object:
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
                max_attempts=DEBATE_STRUCTURED_STEP_MAX_ATTEMPTS,
                reason=exc.private.reason,
            )
            if attempts >= DEBATE_STRUCTURED_STEP_MAX_ATTEMPTS:
                raise


REVIEWER_DEVELOPER_PROMPT = """You are checking whether the current proposed answer should be sent to the user now.

You will see:
- the conversation transcript
- the current proposed answer
- on later rounds, the latest response note and possibly a previous next-step note

Definitions:
- current proposed answer: the next thing that would be returned now if it were accepted,
  which may be a direct answer, a tool call, or a combination of both
- response note: another model's reply to the latest review note; it may agree or disagree
- next-step note: a short note about what the next round should focus on

Return JSON only.

Use available tools when they help.
You only have access here to a restricted safe subset of tools. If the current
proposed answer includes tool availability metadata, treat it as authoritative.
If a tool is unavailable in this debate step, that does not mean the normal
answer-writing step lacks it. Do not claim a tool "doesn't exist" just because
it is unavailable in this restricted debate toolset. If the current proposed
answer would require a non-safe tool in the normal step, describe that as a need
for a non-safe tool, not as a missing or nonexistent tool.

Use:
- `accept` if the current proposed answer is the correct next thing to return exactly as-is
- `reopen` if another round is needed

If you use `reopen`, set `note` to one short review note saying what seems wrong,
missing, unsupported, or risky about the current proposed answer.
"""

MAIN_DEBATE_DEVELOPER_PROMPT = """You are writing a response note about the current proposed answer.

You will see:
- the full conversation context
- the current proposed answer
- the latest review note

Definitions:
- current proposed answer: the next thing that would be returned now if accepted,
  which may be a direct answer, a tool call, or a combination of both
- review note: another model's critique of that answer; it may be correct or incorrect

Write one short response note. The response note should explain, from your own judgment:
- what the review note got right
- what it got wrong
- what matters most for deciding whether the current proposed answer is the correct next thing to return

Do not write a replacement answer for the user.
Do not decide whether the current proposed answer should be sent.
You may agree, partly agree, or disagree with the review note.
Use available tools when they help.
You only have access here to a restricted safe subset of tools. If a tool is not
available in this debate step, that does not mean the normal answer-writing step
lacks it. Do not claim a tool "doesn't exist" just because it is unavailable in
this restricted debate toolset. If the current proposed answer would require a
non-safe tool in the normal step, describe that as a need for a non-safe tool,
not as a missing or nonexistent tool.
"""

ARBITRATOR_DEVELOPER_PROMPT = """You are deciding what happens to the current proposed answer.

You will see:
- the conversation transcript
- the current proposed answer
- the latest review note
- the latest response note

Definitions:
- current proposed answer: the next thing that would be returned now if accepted,
  which may be a direct answer, a tool call, or a combination of both
- review note: a short note explaining what seems wrong, missing, unsupported, or risky about that answer
- response note: a short note replying to the review note from independent judgment
- next-step note: a short note telling the next round what to focus on

Return JSON only.

Use available tools when they help.
You only have access here to a restricted safe subset of tools. If the current
proposed answer includes tool availability metadata, treat it as authoritative.
If a tool is unavailable in this debate step, that does not mean the normal
answer-writing step lacks it. Do not claim a tool "doesn't exist" just because
it is unavailable in this restricted debate toolset. If the current proposed
answer would require a non-safe tool in the normal step, describe that as a need
for a non-safe tool, not as a missing or nonexistent tool.

Use:
- `accept` if the current proposed answer is the correct next thing to return exactly as-is
- `revise` if the current proposed answer should not be sent and the normal answer-writing step should try again from scratch
- `reopen` if another review/response round is needed

If you use `revise`:
- the current proposed answer will not be sent
- the response note will not be sent
- your `note` will be sent to the normal answer-writing step, which will write a fresh new answer from scratch

If you use `reopen`:
- the current proposed answer will not be sent
- the response note will not be sent
- your `note` will be sent into another review/response round

If you use `revise` or `reopen`, set `note` to one short next-step note.
"""

REVIEWER_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["accept", "reopen"]},
        "note": {"type": ["string", "null"]},
    },
    "required": ["action", "note"],
}

ARBITRATOR_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["accept", "revise", "reopen"]},
        "note": {"type": ["string", "null"]},
    },
    "required": ["action", "note"],
}

REVIEWER_RESPONSE_FORMAT = ChatResponseFormat(
    type="json_schema",
    name="reviewer_decision",
    schema=REVIEWER_RESPONSE_SCHEMA,
    strict=True,
    description="Reviewer decision for the current proposed answer.",
)

ARBITRATOR_RESPONSE_FORMAT = ChatResponseFormat(
    type="json_schema",
    name="arbitrator_decision",
    schema=ARBITRATOR_RESPONSE_SCHEMA,
    strict=True,
    description="Final decision for the current proposed answer.",
)


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
    server_executors: Mapping[str, IMCPToolProvider],
) -> tuple[tuple[FunctionTool, ...], dict[str, ToolPolicy], dict[str, IMCPToolProvider]]:
    safe_tools: list[FunctionTool] = []
    safe_policies: dict[str, ToolPolicy] = {}
    safe_executors: dict[str, IMCPToolProvider] = {}
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
        max_completion_tokens=max_completion_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_logprobs=request.top_logprobs,
        reasoning_effort=actor_config.reasoning_effort,
        prompt_cache_key=None if prompt_cache_key_base is None else f"{prompt_cache_key_base}|{actor}",
        service_tier=actor_config.service_tier,
        user=None,
    )


def parse_reviewer_decision(message: StateMessage) -> ReviewerDecision:
    decision = _decode_decision(message, ReviewerDecision, "reviewer decision")
    if decision.action == ReviewerActionType.REOPEN and not decision.note:
        raise _debate_unavailable_error(reason="reviewer_reopen_requires_note", private_message="reviewer reopen requires note")
    if decision.action == ReviewerActionType.ACCEPT:
        return ReviewerDecision(action=ReviewerActionType.ACCEPT)
    return decision


def parse_arbitrator_decision(message: StateMessage) -> ArbitratorDecision:
    decision = _decode_decision(message, ArbitratorDecision, "arbitrator decision")
    if decision.action in {ArbitratorActionType.REVISE, ArbitratorActionType.REOPEN} and not decision.note:
        raise _debate_unavailable_error(reason="arbitrator_note_required", private_message="arbitrator note is required")
    if decision.action == ArbitratorActionType.ACCEPT:
        return ArbitratorDecision(action=ArbitratorActionType.ACCEPT)
    return decision


def _decode_decision(message: StateMessage, typ, label: str):
    if message.content is None:
        raise _debate_unavailable_error(reason="decision_missing_content", private_message=f"{label} is missing content")
    try:
        return msgspec.json.decode(message.content.encode(), type=typ)
    except msgspec.DecodeError as exc:
        raise _debate_unavailable_error(reason="decision_invalid_json", private_message=f"{label} is invalid JSON", cause=exc) from exc
    except msgspec.ValidationError as exc:
        raise _debate_unavailable_error(reason="decision_invalid", private_message=f"{label} is invalid", cause=exc) from exc


def _compact_candidate(
    parts: TempMainParts,
    *,
    normal_tool_policies: Mapping[str, ToolPolicy],
    debate_tool_policies: Mapping[str, ToolPolicy],
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
            normal_policy = normal_tool_policies.get(call.name)
            debate_policy = debate_tool_policies.get(call.name)
            compact_call: dict[str, object] = {
                "name": call.name,
                "arguments": call.arguments_value(),
                "available_in_debate": debate_policy is not None,
                "available_in_normal_step": normal_policy is not None,
            }
            if normal_policy is not None:
                compact_call["normal_effect_class"] = normal_policy.effect_class.value
            output = outputs.get(call.id)
            if output is not None:
                compact_call["output"] = output
            compact_tool_calls.append(compact_call)
        value["tool_calls"] = compact_tool_calls
    return value


def _transcript_wrapper(transcript: Sequence[TranscriptMessage]) -> ChatMessage:
    return ChatMessage(role="user", content=f"Conversation transcript:\n{_json_text([message.to_primitive() for message in transcript])}")


def _reviewer_initial_turn(
    parts: TempMainParts,
    *,
    normal_tool_policies: Mapping[str, ToolPolicy],
    debate_tool_policies: Mapping[str, ToolPolicy],
) -> StateMessage:
    return StateMessage(
        role="user",
        content=(
            "Review the current proposed answer below. Decide whether to accept it as-is or reopen with one short review note.\n\n"
            "Current proposed answer:\n"
            f"{_json_text(_compact_candidate(parts, normal_tool_policies=normal_tool_policies, debate_tool_policies=debate_tool_policies))}"
        ),
    )


def _reviewer_reopen_turn(*, latest_response_note: StateMessage, latest_next_step_note: str | None) -> StateMessage:
    parts = []
    if latest_next_step_note is not None:
        parts.append(f"Previous next-step note:\n{latest_next_step_note}")
    parts.append(f"Latest response note:\n{latest_response_note.content_text() or ''}")
    parts.append("Revisit the current proposed answer and decide whether to accept it or reopen again with a new review note.")
    return StateMessage(role="user", content="\n\n".join(parts))


def _main_debate_turn(*, reviewer_decision: ReviewerDecision) -> StateMessage:
    return StateMessage(
        role="user",
        content=(
            "Latest review note:\n"
            f"{reviewer_decision.note or ''}\n\n"
            "Write one short response note about the current proposed answer. "
            "You may agree, partly agree, or disagree with the review note."
        ),
    )


def _arbitrator_initial_turn(
    *,
    parts: TempMainParts,
    reviewer_decision: ReviewerDecision,
    latest_response_note: StateMessage,
    normal_tool_policies: Mapping[str, ToolPolicy],
    debate_tool_policies: Mapping[str, ToolPolicy],
) -> StateMessage:
    return StateMessage(
        role="user",
        content=(
            "Current proposed answer:\n"
            f"{_json_text(_compact_candidate(parts, normal_tool_policies=normal_tool_policies, debate_tool_policies=debate_tool_policies))}"
            "\n\n"
            "Latest review note:\n"
            f"{reviewer_decision.note or ''}\n\n"
            "Latest response note:\n"
            f"{latest_response_note.content_text() or ''}\n\n"
            "Decide whether to accept the current proposed answer, revise normal main with one next-step note, or reopen the review cycle."
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
            "Decide whether to accept the current proposed answer, revise normal main with one next-step note, or reopen the review cycle."
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
    server_executors: Mapping[str, IMCPToolProvider],
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
    profile: RuntimeModelProfileConfig,
    request,
    tools: Sequence[FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    normal_tool_policies: Mapping[str, ToolPolicy],
    server_executors: Mapping[str, IMCPToolProvider],
    chat_completion_client: IChatCompletionClient,
    prompt_cache_key_base: str | None,
    usage_ledger: UsageLedger,
) -> ActorFinished | ActorAwaitingClientTool | None:
    thread = _thread_messages(state.reviewer)
    header_messages = [
        ChatMessage(role="developer", content=REVIEWER_DEVELOPER_PROMPT),
        _transcript_wrapper(state.compact_transcript(token_budget=profile.reviewer_transcript_token_budget)),
        *(message.to_chat_message() for message in thread),
    ]
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
        turn_messages = [
            _reviewer_initial_turn(
                parts,
                normal_tool_policies=normal_tool_policies,
                debate_tool_policies=tool_policies,
            )
        ]
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
        response_format=REVIEWER_RESPONSE_FORMAT,
    )


async def run_main_debate_turn(
    *,
    state: MutableQueues,
    parts: TempMainParts,
    profile: RuntimeModelProfileConfig,
    request,
    tools: Sequence[FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    server_executors: Mapping[str, IMCPToolProvider],
    chat_completion_client: IChatCompletionClient,
    prompt_cache_key_base: str | None,
    usage_ledger: UsageLedger,
) -> ActorFinished | ActorAwaitingClientTool | None:
    thread = [row.message for row in parts.remaining_temp_rows]
    header_messages = [
        ChatMessage(role="developer", content=MAIN_DEBATE_DEVELOPER_PROMPT),
        *(row.render_for_model(include_citation=False) for row in state.effective_main_context()),
    ]
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
    profile: RuntimeModelProfileConfig,
    request,
    tools: Sequence[FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    normal_tool_policies: Mapping[str, ToolPolicy],
    server_executors: Mapping[str, IMCPToolProvider],
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
    header_messages = [
        ChatMessage(role="developer", content=ARBITRATOR_DEVELOPER_PROMPT),
        _transcript_wrapper(state.compact_transcript(token_budget=profile.arbitrator_transcript_token_budget)),
        *(message.to_chat_message() for message in thread),
    ]
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
                normal_tool_policies=normal_tool_policies,
                debate_tool_policies=tool_policies,
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
        response_format=ARBITRATOR_RESPONSE_FORMAT,
    )


async def start_debate_from_candidate(
    *,
    state: MutableQueues,
    out: ResponseEventIO,
    keyring: SealingKeyring,
    assistant: StateMessage,
    tool_calls: Sequence[ChatToolCall],
    server_outputs: Mapping[int, str],
) -> None:
    messages: list[StateMessage] = [assistant]
    for index, call in enumerate(tool_calls):
        messages.append(
            StateMessage(
                role="tool",
                tool_call_id=call.id,
                content=server_outputs.get(index, HELD_CLIENT_TOOL_PLACEHOLDER),
            )
        )
    await _persist_temp_turn(
        state=state,
        side=Side.MAIN,
        messages=messages,
        continuation_side=Side.REVIEWER,
        out=out,
        keyring=keyring,
    )


async def publish_accepted_candidate(*, state: MutableQueues, out: ResponseEventIO, keyring: SealingKeyring) -> None:
    parts = state.temp_main_parts()
    if parts.held_candidate is None:
        raise _debate_internal_error(reason="held_candidate_missing", private_message="debate temp state is missing held candidate")
    candidate = parts.held_candidate.message
    public_assistant = StateMessage(
        role="assistant",
        content=(
            strip_leading_internal_citations(candidate.content)
            if candidate.content is not None
            else ""
            if candidate.reasoning_content or candidate.reasoning_details or candidate.tool_calls
            else None
        ),
    )
    assistant_hash = public_assistant.content_hash()

    if candidate.reasoning_content or candidate.reasoning_details:
        reasoning_payload = ReasoningPayload(
            side=Side.MAIN,
            temp=False,
            messages=(
                ReasoningMessagePatch(
                    content_hash=assistant_hash,
                    reasoning_content=candidate.reasoning_content,
                    reasoning_details=tuple(candidate.reasoning_details) or None,
                ),
            ),
        )
        await out.output(
            ResponseReasoningItem(
                encrypted_content=seal_reasoning_payload(reasoning_payload, keyring=keyring),
                id=f"rs_{secrets.token_urlsafe(18)}",
                status="completed",
                summary=[],
                type="reasoning",
            ),
            reasoning_side=reasoning_payload.side,
            reasoning_messages=reasoning_payload.messages,
        )

    if public_assistant.content is not None or candidate.reasoning_content or candidate.reasoning_details or candidate.tool_calls:
        await out.output(
            ResponseMessageItem(
                content=[OutputTextContent(text=public_assistant.content or "", type="output_text")],
                id=f"msg_{secrets.token_urlsafe(18)}",
                role="assistant",
                status="completed",
                type="message",
            )
        )

    hidden_outputs = {
        row.message.tool_call_id: row.message.content_text() or ""
        for row in parts.held_hidden_tool_rows
        if row.message.tool_call_id is not None
    }
    function_call_ids: dict[str, str] = {}
    for index, call in enumerate(candidate.tool_calls):
        sealed_call_id = seal_call_id(
            SealedCallID(
                side=Side.MAIN,
                content_hash_prefix=content_hash_prefix(assistant_hash),
                tool_call_index=index,
                upstream_tool_call_id=call.id,
            ),
            keyring=keyring,
        )
        function_call_ids[call.id] = sealed_call_id
        await out.output(
            ResponseFunctionCallItem(
                arguments=call.arguments,
                call_id=sealed_call_id,
                id=f"fc_{secrets.token_urlsafe(18)}",
                name=call.name,
                status="completed",
                type="function_call",
            )
        )

    for call in candidate.tool_calls:
        output = hidden_outputs.get(call.id)
        if output is None or output == HELD_CLIENT_TOOL_PLACEHOLDER:
            continue
        await out.output(
            ResponseFunctionCallOutputItem(
                call_id=function_call_ids[call.id],
                created_by="server",
                id=f"fco_{secrets.token_urlsafe(18)}",
                output=output,
                status="completed",
                type="function_call_output",
            )
        )

    if public_assistant.content is not None or candidate.reasoning_content or candidate.reasoning_details or candidate.tool_calls:
        state.append_main_stable(
            StateMessage(role="assistant", content=public_assistant.content, tool_calls=list(candidate.tool_calls)),
            content_hash=assistant_hash,
        )

    for call in candidate.tool_calls:
        output = hidden_outputs.get(call.id)
        if output is None or output == HELD_CLIENT_TOOL_PLACEHOLDER:
            continue
        state.append_main_stable(StateMessage(role="tool", tool_call_id=call.id, content=output))

    state.clear_debate()


async def resume_main_with_revise_bundle(
    *,
    state: MutableQueues,
    out: ResponseEventIO,
    keyring: SealingKeyring,
    note: str,
) -> None:
    parts = state.temp_main_parts()
    if parts.held_candidate is None:
        raise _debate_internal_error(reason="revise_requires_held_candidate", private_message="revise requires held candidate state")

    note_message = StateMessage(role="assistant", content=note)
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
            summary=[],
            type="reasoning",
        ),
        reasoning_side=reasoning_payload.side,
        reasoning_messages=reasoning_payload.messages,
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
    request,
    profile: RuntimeModelProfileConfig,
    keyring: SealingKeyring,
    tools: Sequence[FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    server_executors: Mapping[str, IMCPToolProvider],
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
        await publish_accepted_candidate(state=state, out=out, keyring=keyring)
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
                    profile=profile,
                    request=request,
                    tools=safe_tools,
                    tool_policies=safe_tool_policies,
                    normal_tool_policies=tool_policies,
                    server_executors=safe_server_executors,
                    chat_completion_client=chat_completion_client,
                    prompt_cache_key_base=prompt_cache_key_base,
                    usage_ledger=usage_ledger,
                )
                if outcome is None or isinstance(outcome, ActorAwaitingClientTool):
                    return outcome, None
                return outcome, parse_reviewer_decision(outcome.assistant)

            outcome, decision = await _retry_structured_debate_step(Actor.REVIEWER.value, reviewer_step)
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
                keyring=keyring,
            )
            if decision.action == ReviewerActionType.ACCEPT:
                if held_anchor_index is not None:
                    usage_ledger.record_hidden(profile.reviewer.public_usage, outcome.usage)
                    usage_ledger.use_hidden_as_anchor(held_anchor_index)
                else:
                    usage_ledger.set_anchor(outcome.usage)
                await publish_accepted_candidate(state=state, out=out, keyring=keyring)
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
                    tools=safe_tools,
                    tool_policies=safe_tool_policies,
                    server_executors=safe_server_executors,
                    chat_completion_client=chat_completion_client,
                    prompt_cache_key_base=prompt_cache_key_base,
                    usage_ledger=usage_ledger,
                )

            outcome = await _retry_structured_debate_step(Actor.MAIN_DEBATE.value, main_debate_step)
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
                keyring=keyring,
            )
            parts = state.temp_main_parts()
            continue

        if actor == Actor.ARBITRATOR:
            async def arbitrator_step(parts=parts):
                outcome = await run_arbitrator_turn(
                    state=state,
                    parts=parts,
                    profile=profile,
                    request=request,
                    tools=safe_tools,
                    tool_policies=safe_tool_policies,
                    normal_tool_policies=tool_policies,
                    server_executors=safe_server_executors,
                    chat_completion_client=chat_completion_client,
                    prompt_cache_key_base=prompt_cache_key_base,
                    usage_ledger=usage_ledger,
                )
                if outcome is None or isinstance(outcome, ActorAwaitingClientTool):
                    return outcome, None
                return outcome, parse_arbitrator_decision(outcome.assistant)

            outcome, decision = await _retry_structured_debate_step(Actor.ARBITRATOR.value, arbitrator_step)
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
                keyring=keyring,
            )
            if decision.action == ArbitratorActionType.ACCEPT:
                if held_anchor_index is not None:
                    usage_ledger.record_hidden(profile.arbitrator.public_usage, outcome.usage)
                    usage_ledger.use_hidden_as_anchor(held_anchor_index)
                else:
                    usage_ledger.set_anchor(outcome.usage)
                await publish_accepted_candidate(state=state, out=out, keyring=keyring)
                await out.completed(service_tier=request.service_tier, usage=usage_ledger.to_response_usage())
                return DebateResult.COMPLETED

            usage_ledger.record_hidden(profile.arbitrator.public_usage, outcome.usage)
            if decision.action == ArbitratorActionType.REVISE:
                await resume_main_with_revise_bundle(state=state, out=out, keyring=keyring, note=decision.note or "")
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
    keyring: SealingKeyring,
) -> None:
    payload = ReasoningPayload(
        side=side,
        temp=True,
        continuation_side=continuation_side,
        messages=tuple(messages),
    )
    summary_messages = _temp_turn_summary_messages(state=state, side=side, messages=messages)
    await out.output(
        ResponseReasoningItem(
            encrypted_content=seal_reasoning_payload(payload, keyring=keyring),
            id=f"rs_{secrets.token_urlsafe(18)}",
            status="completed",
            summary=[],
            type="reasoning",
        ),
        reasoning_side=payload.side,
        reasoning_messages=summary_messages,
    )
    if side == Side.MAIN:
        for message in messages:
            state.append_main_temp(message)
    else:
        for message in messages:
            state.append_side(side, message)
    state.set_continuation(continuation_side, in_temp_debate=True)


def _temp_turn_summary_messages(
    *,
    state: MutableQueues,
    side: Side,
    messages: Sequence[StateMessage],
) -> tuple[StateMessage, ...]:
    if side == Side.MAIN and not state.main_context_temp:
        return tuple(messages)
    parts = state.temp_main_parts()
    if parts.held_candidate is None:
        return tuple(messages)
    return (
        parts.held_candidate.message,
        *(row.message for row in parts.held_hidden_tool_rows),
        *messages,
    )


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
