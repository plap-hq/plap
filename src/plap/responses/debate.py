from __future__ import annotations

import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum

import msgspec
import structlog

from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.keyring import SealingKeyring
from plap.llms.completions.chat import (
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
from plap.responses.json_utils import JSONInvalidError, JSONNotObjectError, parse_json_object_with_repair
from plap.logging import bound_context, log_debug, log_payload
from plap.responses.contracts import (
    FunctionTool,
    ResponseFunctionCallItem,
    ResponseReasoningItem,
    SummaryTextContent,
    TextFormatJSONObject,
    TextFormatJSONSchema,
    ToolChoiceFunction,
)
from plap.responses.ingest import (
    SealedCallID,
    compact_transcript,
    content_hash_prefix,
    seal_call_id,
    seal_reasoning_payload,
    truncate_transcript,
)
from plap.responses.io import ReasoningDraft, ResponseEventIO
from plap.responses.models import (
    Actor,
    ChatMessageSpan,
    DefenderParts,
    MutableQueues,
    ReasoningPayload,
    Side,
    StateMessage,
    StateToolCall,
    TranscriptMessage,
    UsageLedger,
)
from plap.responses.tokens import measure_prompt_tokens
from plap.responses.tools import (
    IToolCallPolicyResolver,
    ToolPolicy,
    canonical_tool_arguments,
    normalize_function_tool,
    resolve_tool_call_policies,
)
from plap.responses.tools.mcp import IServerToolExecutor
from plap.settings import RuntimeActorConfig, RuntimeModelProfileConfig

DEBATE_HELD_TOOL_PLACEHOLDER = "This tool call was intercepted by a reviewer."
DEBATE_UNSAFE_TOOL_PLACEHOLDER = (
    "This tool call is unsafe to run in this environment. "
    "Do not retry the same call. Instead, continue from that result by reasoning about the task, "
    "using other safe tools, or changing approach."
)
DEBATE_INVALID_TOOL_ARGUMENTS_PLACEHOLDER = (
    "This tool call could not be used because its arguments were not a valid JSON object. "
    "If you still need this tool, call it again with corrected JSON object arguments."
)
DEBATE_STEP_MAX_ATTEMPTS = 3
CALLED_TOOL_DEFINITIONS_HEADER = "Tool definitions for tools used by the proposed next step:"
REQUEST_CONSTRAINTS_HEADER = "Request constraints for the proposed next step:"
DECISION_TRAILING_PUNCTUATION = frozenset({".", ",", ":", ";", ")", "]", '"', "'"})
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


def _is_retryable_decision_error(exc: PlapError) -> bool:
    return exc.private.reason in {
        "reviewer_reopen_requires_note",
        "arbitrator_note_required",
        "decision_missing_content",
        "decision_invalid_tail_marker",
        "decision_ambiguous_boundary_markers",
    }


DEBATE_TOOL_AVAILABILITY_PROMPT = """Use available tools when they help.
You only have access here to a restricted callable subset of tools. You may also
receive a user message titled `Tool definitions for tools used by the proposed
next step`. If present, it contains tool definitions for tools that are already used in
the proposed next step. Use it to understand what those proposed tool calls
mean and whether they are appropriate. The fact that one of those tools is not
callable in this debate step does not mean the normal main step lacks it. Do
not reject or criticize a proposed next step merely because one of those tools
is not callable in this debate step."""

DEBATE_REQUEST_CONSTRAINTS_PROMPT = """You may also receive a user message titled `Request constraints for the proposed next step:`.
If present, it describes client-requested tool-choice or output-format
constraints. Treat those constraints as authoritative when judging whether the
current proposed next step is appropriate. Do not criticize a proposed next
step for following them."""

REQUESTED_SCOPE_RULES_PROMPT = """Requested scope rules:
- infer the user's current requested scope from the transcript
- the current requested scope may be the whole task or only a bounded slice
- do not widen the current requested scope
- an in-progress tool step may be intentionally partial
- a user-facing return step may be correct even if more work exists outside the current requested scope"""

REVIEW_CONTEXT_RULES_PROMPT = """If the review context says this is an in-progress tool step:
- judge correctness of the next action, not whole-task completion
- do not treat remaining later work as a defect by itself

If the review context says this is a user-facing return step:
- judge completion of the current requested scope only
- do not treat additional work outside that scope as required continuation"""

REVIEWER_DEVELOPER_PROMPT = f"""You are checking whether the current proposed next step should be returned now.

You will see:
- the conversation transcript
- the current proposed next step
- on later rounds, the latest response note and possibly a guidance note

Definitions:
- current proposed next step: the exact next thing that would be returned now if accepted,
  which may be a direct user-facing message, a tool call, or a combination of both
- response note: another model's reply to the latest review note; it may agree or disagree
- guidance note: a short note about what the next review round should focus on

Do not return JSON.

Decision format:
- Your final non-empty line is the decision wrapper line.
- That final line may include a prefix, but its last token must be exactly ACCEPT or REOPEN.
- Examples of valid final lines:
  ACCEPT
  Decision: REOPEN
  Final decision: ACCEPT
- Put no review-note text on the final line. Put all review-note text above it.

Consistency rules:
- If you write that the current proposed next step is wrong, unsupported, risky, or missing something, you must not use ACCEPT.
- Before finishing, check that your final decision line agrees with the rest of your output.

Verification discipline:
- Compare the current proposed next step, and on later rounds the latest
  response note, against the actual transcript, tool outputs, and candidate
  contents.
- Do not credit claimed checks, fixes, reads, runs, or verifications unless they are supported there.
- If something is claimed as handled, confirmed, safe, or complete but the
  evidence is missing, contradictory, or materially incomplete, treat that as
  a concrete defect.
- If confident wording is covering a real uncertainty, unsupported
  assumption, or overlooked edge case, treat that as unsupported rather than
  acceptable.

{DEBATE_TOOL_AVAILABILITY_PROMPT}

{DEBATE_REQUEST_CONSTRAINTS_PROMPT}

{REQUESTED_SCOPE_RULES_PROMPT}

{REVIEW_CONTEXT_RULES_PROMPT}

Use:
- `ACCEPT` if the current proposed next step is correct for the current requested scope
- `REOPEN` only if you can identify a concrete defect in the current proposed next step for that scope

Do not reopen merely because later work still exists outside the current requested scope.

If you use `REOPEN`:
- you MUST write one short review note above the final line saying what seems
  wrong, missing, unsupported, or risky about the current proposed next step.
- do not add labels such as "Review note:"
"""

DEFENDER_DEVELOPER_PROMPT = f"""You are writing a response note about the current proposed next step.

You will see:
- the full conversation context
- the current proposed next step
- the latest review note

Definitions:
- current proposed next step: the exact next thing that would be returned now if accepted,
  which may be a direct user-facing message, a tool call, or a combination of both
- review note: another model's critique of that next step; it may be correct or incorrect

{REQUESTED_SCOPE_RULES_PROMPT}

{REVIEW_CONTEXT_RULES_PROMPT}

Write one short response note from independent judgment.

Start with exactly one of:
- The review note is wrong:
- The review note is partly right:
- The review note is correct:

Presume the current proposed next step is correct.
The review note must identify a concrete defect in that next step for the current requested scope.
If it does not, treat the review note as wrong.
If the review note widens scope, confuses stage, demands whole-task completion
too early, or demands continuation beyond the current requested scope, say so
explicitly.
Do not partially agree just to sound balanced.
Do not hedge.
Do not soften disagreement.

Do not write a replacement answer for the user.
Do not decide whether the current proposed next step should be sent.
{DEBATE_TOOL_AVAILABILITY_PROMPT}

{DEBATE_REQUEST_CONSTRAINTS_PROMPT}
"""

ARBITRATOR_DEVELOPER_PROMPT = f"""You are deciding what happens to the current proposed next step.

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
- next-step note: a short note telling the next main step what to focus on
- guidance note: a short note about what the next review round should focus on

Do not return JSON.

Decision format:
- Your final non-empty line is the decision wrapper line.
- That final line may include a prefix, but its last token must be exactly ACCEPT, REVISE, or REOPEN.
- Examples of valid final lines:
  ACCEPT
  Decision: REVISE
  Final decision: REOPEN
- Put no note text on the final line. Put all note text above it.

Consistency rules:
- If you write that the current proposed next step is wrong, unsupported, risky, or missing something, you must not use ACCEPT.
- Before finishing, check that your final decision line agrees with the rest of your output.

{DEBATE_TOOL_AVAILABILITY_PROMPT}

{DEBATE_REQUEST_CONSTRAINTS_PROMPT}

{REQUESTED_SCOPE_RULES_PROMPT}

{REVIEW_CONTEXT_RULES_PROMPT}

Use:
- `ACCEPT` if the current proposed next step is correct for the current requested scope
- `REVISE` if the current proposed next step is wrong for that scope and should be discarded in favor of a fresh retry
- `REOPEN` if the current proposed next step is wrong for that scope but another review/response round is still likely useful

If you use `REVISE`:
- the current proposed next step will not be sent
- you MUST write one short next-step note above the final line
- that next-step note will be sent to the normal main step, which will choose and write a fresh next step from scratch
- do not add labels such as "Next-step note:"
- write the next-step note as instructions to the next main turn
- write from the perspective of the main step; it does not know of
  "review," "reviewer," "arbitrator," "proposed next step," or "decision"
- state only what to do next and what went wrong

If you use `REOPEN`:
- the current proposed next step will not be sent
- you MUST write one short guidance note above the final line
- that guidance note will be sent into another review round
- do not add labels such as "Guidance note:"
- put what you want the next review round to focus on in that note
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


def debate_callable_surface(
    tools: Sequence[FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    server_executors: Mapping[str, IServerToolExecutor],
) -> tuple[tuple[FunctionTool, ...], dict[str, ToolPolicy], dict[str, IServerToolExecutor]]:
    callable_tools: list[FunctionTool] = []
    callable_policies: dict[str, ToolPolicy] = {}
    callable_executors: dict[str, IServerToolExecutor] = {}
    for tool in tools:
        policy = tool_policies.get(tool.name)
        if policy is None or policy.effect_class not in {"safe", "contextual"}:
            continue
        callable_tools.append(tool)
        callable_policies[tool.name] = policy
        executor = server_executors.get(tool.name)
        if executor is not None:
            callable_executors[tool.name] = executor
    return tuple(callable_tools), callable_policies, callable_executors


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
        temperature=actor_config.map_temperature(request.temperature),
        top_p=actor_config.map_top_p(request.top_p),
        top_logprobs=actor_config.map_top_logprobs(request.top_logprobs),
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


def _decision_line_action(line: str, *, allowed_actions: Mapping[str, object]) -> object | None:
    if not line:
        return None
    token = line.rsplit(maxsplit=1)[-1]
    action = allowed_actions.get(token)
    if action is not None:
        return action
    if len(token) > 1 and token[-1] in DECISION_TRAILING_PUNCTUATION:
        return allowed_actions.get(token[:-1])
    return None


def _boundary_decision(message: StateMessage, *, label: str, allowed_actions: Mapping[str, object]) -> tuple[object, str | None]:
    if message.content is None or not message.content.strip():
        raise _debate_unavailable_error(reason="decision_missing_content", private_message=f"{label} is missing content")
    lines = message.content.rstrip().splitlines()
    non_empty_indexes = [index for index, line in enumerate(lines) if line.strip()]
    if not non_empty_indexes:
        raise _debate_unavailable_error(reason="decision_missing_content", private_message=f"{label} is missing content")

    first_index = non_empty_indexes[0]
    last_index = non_empty_indexes[-1]
    first_action = _decision_line_action(lines[first_index].strip(), allowed_actions=allowed_actions)
    last_action = _decision_line_action(lines[last_index].strip(), allowed_actions=allowed_actions)

    if first_action is None and last_action is None:
        expected = ", ".join(allowed_actions)
        raise _debate_unavailable_error(
            reason="decision_invalid_tail_marker",
            private_message=f"{label} boundary line must end with one of {expected}",
        )

    if first_index != last_index and first_action is not None and last_action is not None and first_action != last_action:
        raise _debate_unavailable_error(
            reason="decision_ambiguous_boundary_markers",
            private_message=f"{label} first and last decision lines disagree",
        )

    action = first_action if first_action is not None else last_action
    wrapper_indexes = {index for index, decision in ((first_index, first_action), (last_index, last_action)) if decision is not None}
    note = "\n".join(line for index, line in enumerate(lines) if index not in wrapper_indexes).strip() or None
    return action, note


def parse_reviewer_decision(message: StateMessage) -> ReviewerDecision:
    action, note = _boundary_decision(message, label="reviewer decision", allowed_actions=_REVIEWER_ACTIONS_BY_TOKEN)
    if action == ReviewerActionType.REOPEN and not note:
        raise _debate_unavailable_error(reason="reviewer_reopen_requires_note", private_message="reviewer reopen requires note")
    if action == ReviewerActionType.ACCEPT:
        return ReviewerDecision(action=ReviewerActionType.ACCEPT)
    return ReviewerDecision(action=ReviewerActionType.REOPEN, note=note)


def parse_arbitrator_decision(message: StateMessage) -> ArbitratorDecision:
    action, note = _boundary_decision(message, label="arbitrator decision", allowed_actions=_ARBITRATOR_ACTIONS_BY_TOKEN)
    if action in {ArbitratorActionType.REVISE, ArbitratorActionType.REOPEN} and not note:
        raise _debate_unavailable_error(reason="arbitrator_note_required", private_message="arbitrator note is required")
    if action == ArbitratorActionType.ACCEPT:
        return ArbitratorDecision(action=ArbitratorActionType.ACCEPT)
    return ArbitratorDecision(action=action, note=note)


def _compact_candidate(
    parts: DefenderParts,
) -> dict[str, object]:
    if parts.held_candidate is None:
        raise _debate_internal_error(reason="held_candidate_missing", private_message="debate temp state is missing held candidate")
    candidate = parts.held_candidate.message
    outputs = {
        row.message.tool_call_id: row.message.content_text() or ""
        for row in parts.held_hidden_tool_rows
        if row.message.tool_call_id is not None and (row.message.content_text() or "") != DEBATE_HELD_TOOL_PLACEHOLDER
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


def _is_in_progress_tool_step(parts: DefenderParts) -> bool:
    if parts.held_candidate is None:
        raise _debate_internal_error(reason="held_candidate_missing", private_message="debate temp state is missing held candidate")
    return bool(parts.held_candidate.message.tool_calls)


def _review_context_wrapper(parts: DefenderParts) -> ChatMessage:
    if _is_in_progress_tool_step(parts):
        return ChatMessage(
            role="user",
            content=(
                "Review context:\n"
                "- stage: in-progress tool step\n"
                "- standard: judge whether this is the correct next action for the user's current "
                "requested scope; do not require whole-task completion yet"
            ),
        )
    return ChatMessage(
        role="user",
        content=(
            "Review context:\n"
            "- stage: user-facing return step\n"
            "- standard: judge whether this properly completes the user's current requested scope; "
            "do not require continuation beyond that scope"
        ),
    )


def _request_constraints_wrapper(request) -> ChatMessage | None:
    constraints: dict[str, object] = {}

    tool_choice = request.tool_choice
    if isinstance(tool_choice, ToolChoiceFunction):
        constraints["tool_choice"] = {"type": "function", "name": tool_choice.name}
    elif tool_choice not in {None, "auto"}:
        constraints["tool_choice"] = tool_choice

    text_config = request.text
    if text_config is not None and text_config.format is not None:
        text_format = text_config.format
        if isinstance(text_format, TextFormatJSONObject):
            constraints["response_format"] = {"type": "json_object"}
        elif isinstance(text_format, TextFormatJSONSchema):
            constraints["response_format"] = {
                "type": "json_schema",
                "name": text_format.name,
                "strict": text_format.strict,
                "schema": text_format.schema_,
            }

    if not constraints:
        return None

    return ChatMessage(role="user", content=f"{REQUEST_CONSTRAINTS_HEADER}\n{_json_text(constraints)}")


def _candidate_called_tool_definitions_message(
    parts: DefenderParts,
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


def _revise_note_wrapper(note: str) -> str:
    return (
        "Internal retry guidance for the next fresh answer only.\n"
        "This is not a user-facing message.\n"
        "Do not quote it, acknowledge it, apologize for it, or reply to it directly.\n\n"
        f"{note}"
    )


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
    transcript: tuple[TranscriptMessage, ...],
    *,
    actor_config: RuntimeActorConfig,
) -> int:
    try:
        return measure_prompt_tokens([_transcript_wrapper(transcript)], actor_config=actor_config)
    except Exception as exc:
        raise _debate_unavailable_error(
            reason="debate_transcript_tokenizer_failed",
            private_message="debate transcript token measurement failed",
            cause=exc,
        ) from exc


def _truncated_transcript_message(
    spans: Sequence[ChatMessageSpan],
    *,
    actor_config: RuntimeActorConfig,
    main_developer_message: StateMessage,
    max_tokens: int,
) -> ChatMessage:
    transcript = truncate_transcript(
        _transcript_rows(tuple(spans), main_developer_message=main_developer_message),
        measure=lambda candidate: _measure_budgeted_transcript_tokens(candidate, actor_config=actor_config),
        max_tokens=max_tokens,
    )
    return _transcript_wrapper(transcript)


def _reviewer_header_messages(
    *,
    state: MutableQueues,
    parts: DefenderParts,
    main_developer_message: StateMessage,
    profile: RuntimeModelProfileConfig,
    request,
    normal_tools: Sequence[FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    thread: Sequence[StateMessage],
) -> list[ChatMessage]:
    header_messages: list[ChatMessage] = [
        ChatMessage(role="developer", content=REVIEWER_DEVELOPER_PROMPT),
        _truncated_transcript_message(
            state.main_context,
            actor_config=profile.reviewer,
            main_developer_message=main_developer_message,
            max_tokens=profile.reviewer_max_transcript_tokens,
        ),
    ]
    request_constraints = _request_constraints_wrapper(request)
    if request_constraints is not None:
        header_messages.append(request_constraints)
    tool_definitions_message = _candidate_called_tool_definitions_message(
        parts,
        normal_tools=normal_tools,
        debate_tool_policies=tool_policies,
    )
    if tool_definitions_message is not None:
        header_messages.append(tool_definitions_message)
    header_messages.append(_review_context_wrapper(parts))
    header_messages.extend(message.to_chat_message() for message in thread)
    return header_messages


def _defender_header_messages(
    *,
    state: MutableQueues,
    parts: DefenderParts,
    request,
    normal_tools: Sequence[FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
) -> list[ChatMessage]:
    header_messages: list[ChatMessage] = [
        ChatMessage(role="developer", content=DEFENDER_DEVELOPER_PROMPT),
        *_defender_context_messages(state),
    ]
    request_constraints = _request_constraints_wrapper(request)
    if request_constraints is not None:
        header_messages.append(request_constraints)
    tool_definitions_message = _candidate_called_tool_definitions_message(
        parts,
        normal_tools=normal_tools,
        debate_tool_policies=tool_policies,
    )
    if tool_definitions_message is not None:
        header_messages.append(tool_definitions_message)
    header_messages.append(_review_context_wrapper(parts))
    return header_messages


def _arbitrator_header_messages(
    *,
    state: MutableQueues,
    parts: DefenderParts,
    main_developer_message: StateMessage,
    profile: RuntimeModelProfileConfig,
    request,
    normal_tools: Sequence[FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    thread: Sequence[StateMessage],
) -> list[ChatMessage]:
    header_messages: list[ChatMessage] = [
        ChatMessage(role="developer", content=ARBITRATOR_DEVELOPER_PROMPT),
        _truncated_transcript_message(
            state.main_context,
            actor_config=profile.arbitrator,
            main_developer_message=main_developer_message,
            max_tokens=profile.arbitrator_max_transcript_tokens,
        ),
    ]
    request_constraints = _request_constraints_wrapper(request)
    if request_constraints is not None:
        header_messages.append(request_constraints)
    tool_definitions_message = _candidate_called_tool_definitions_message(
        parts,
        normal_tools=normal_tools,
        debate_tool_policies=tool_policies,
    )
    if tool_definitions_message is not None:
        header_messages.append(tool_definitions_message)
    header_messages.append(_review_context_wrapper(parts))
    header_messages.extend(message.to_chat_message() for message in thread)
    return header_messages


def _reviewer_initial_turn(
    parts: DefenderParts,
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


def _reviewer_reopen_turn(
    *,
    latest_response_note: StateMessage,
    latest_guidance_note: str | None,
) -> StateMessage:
    content_parts = []
    if latest_guidance_note is not None:
        content_parts.append(f"Current guidance note:\n{latest_guidance_note}")
    content_parts.append(f"Latest response note:\n{latest_response_note.content_text() or ''}")
    content_parts.append("Revisit the current proposed next step and decide whether to accept it or reopen again with a new review note.")
    return StateMessage(role="user", content="\n\n".join(content_parts))


def _defender_turn(*, reviewer_decision: ReviewerDecision) -> StateMessage:
    return StateMessage(
        role="user",
        content=(
            "Latest review note:\n"
            f"{reviewer_decision.note or ''}\n\n"
            "Write one short response note about the current proposed next step."
        ),
    )


def _arbitrator_initial_turn(
    *,
    parts: DefenderParts,
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
            "next-step note back to the normal main step for a fresh retry, or send one "
            "guidance note back into the review cycle."
        ),
    )


def _arbitrator_reopen_turn(
    *,
    reviewer_decision: ReviewerDecision,
    latest_response_note: StateMessage,
) -> StateMessage:
    return StateMessage(
        role="user",
        content=(
            "Updated review note:\n"
            f"{reviewer_decision.note or ''}\n\n"
            "Latest response note:\n"
            f"{latest_response_note.content_text() or ''}\n\n"
            "Decide whether to accept the current proposed next step, send one "
            "next-step note back to the normal main step for a fresh retry, or send one "
            "guidance note back into the review cycle."
        ),
    )


def _thread_waiting_after_tool_output(thread: Sequence[StateMessage]) -> bool:
    return bool(thread) and thread[-1].is_tool()


def _thread_messages(rows: Sequence) -> list[StateMessage]:
    return [row.message for row in rows]


def _defender_context_messages(state: MutableQueues) -> list[ChatMessage]:
    messages = [row.render_for_model(include_citation=False) for row in state.main_context]
    messages.extend(entry.message.to_chat_message(untrusted=True) for entry in state.defender)
    return messages


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
    count = 0
    for row in reviewer:
        message = row.message
        if not message.is_assistant():
            continue
        try:
            parse_reviewer_decision(message)
        except PlapError:
            continue
        count += 1
    return count


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
        return parse_json_object_with_repair(arguments)
    except JSONInvalidError as exc:
        raise _debate_unavailable_error(
            reason="debate_tool_arguments_invalid_json", private_message=f"{label} must be valid JSON", cause=exc
        ) from exc
    except JSONNotObjectError as exc:
        raise _debate_unavailable_error(
            reason="debate_tool_arguments_not_object",
            private_message=f"{label} must be a JSON object",
            cause=exc,
        ) from exc


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
    tool_call_policy_resolver: IToolCallPolicyResolver,
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
            inline_output_seen = False
            parsed_arguments: dict[str, dict[str, object]] = {}
            supported_calls: list[ChatToolCall] = []
            tools_by_name = {tool.name: tool for tool in tools}
            for call in tool_calls:
                policy = tool_policies.get(call.name)
                if policy is None:
                    log_debug(logger, "debate.actor.tool_stubbed", actor=actor_name, reason="tool_unavailable", tool_name=call.name)
                    turn_messages.append(
                        StateMessage(
                            role="tool",
                            tool_call_id=call.id,
                            content=DEBATE_UNSAFE_TOOL_PLACEHOLDER,
                        )
                    )
                    inline_output_seen = True
                    continue
                try:
                    parsed_arguments[call.id] = _arguments_object(call.arguments, label="debate tool arguments")
                except PlapError as exc:
                    log_debug(
                        logger,
                        "debate.actor.tool_stubbed",
                        actor=actor_name,
                        reason=exc.private.reason,
                        tool_name=call.name,
                    )
                    turn_messages.append(
                        StateMessage(
                            role="tool",
                            tool_call_id=call.id,
                            content=DEBATE_INVALID_TOOL_ARGUMENTS_PLACEHOLDER,
                        )
                    )
                    inline_output_seen = True
                    continue
                supported_calls.append(call)

            if supported_calls:
                resolved_policies = await resolve_tool_call_policies(
                    supported_calls,
                    tools=tools_by_name,
                    tool_policies=tool_policies,
                    resolver=tool_call_policy_resolver,
                )
            else:
                resolved_policies = ()

            for call, policy in zip(supported_calls, resolved_policies, strict=True):
                log_debug(
                    logger,
                    "debate.actor.tool_resolved",
                    actor=actor_name,
                    effect_class=policy.effect_class,
                    source=policy.source,
                    tool_name=call.name,
                )
                if policy.source == "server" and policy.effect_class == "safe":
                    executor = server_executors.get(call.name)
                    if executor is None:
                        raise _debate_internal_error(
                            reason="debate_server_tool_executor_missing", private_message="debate server tool executor is missing"
                        )
                    output = await executor.call_tool(call.name, parsed_arguments[call.id])
                    log_debug(logger, "debate.actor.server_tool_executed", actor=actor_name, tool_name=call.name)
                    turn_messages.append(StateMessage(role="tool", tool_call_id=call.id, content=output))
                    inline_output_seen = True
                    continue
                if policy.effect_class == "safe":
                    client_calls.append(call)
                    continue
                log_debug(
                    logger,
                    "debate.actor.tool_stubbed",
                    actor=actor_name,
                    effect_class=policy.effect_class,
                    source=policy.source,
                    tool_name=call.name,
                )
                turn_messages.append(
                    StateMessage(
                        role="tool",
                        tool_call_id=call.id,
                        content=DEBATE_UNSAFE_TOOL_PLACEHOLDER,
                    )
                )
                inline_output_seen = True

            if client_calls:
                return ActorAwaitingClientTool(
                    messages=turn_messages,
                    assistant=assistant,
                    tool_calls=client_calls,
                    usage=result.usage,
                    service_tier=result.service_tier,
                )

            if inline_output_seen:
                usage_ledger.record_hidden(actor_config.public_usage, result.usage)


def _decision_retry_problem(*, exc: PlapError, allowed_actions: Sequence[str]) -> str:
    reason = exc.private.reason
    allowed = ", ".join(allowed_actions)
    if reason == "decision_missing_content":
        return "Your previous answer was blank or had no usable decision."
    if reason == "decision_invalid_tail_marker":
        return f"Your final decision line was invalid. The final non-empty line must end with exactly one of: {allowed}."
    if reason == "decision_ambiguous_boundary_markers":
        return "Your first and last decision lines disagreed. Use one final decision line only."
    if reason == "reviewer_reopen_requires_note":
        return "If you choose REOPEN, include one short review note above the final line that explains the concrete defect."
    if reason == "arbitrator_note_required":
        return (
            "If you choose REVISE or REOPEN, include one short note above the final line. "
            "For REVISE, write a next-step note for the normal main step. "
            "For REOPEN, write a guidance note for the next review round."
        )
    return exc.private.message


def _decision_retry_message(*, allowed_actions: Sequence[str], note_rule: str, exc: PlapError) -> StateMessage:
    allowed = ", ".join(allowed_actions)
    return StateMessage(
        role="user",
        content=(
            "Your previous answer could not be used as written.\n\n"
            "Problem:\n"
            f"- {_decision_retry_problem(exc=exc, allowed_actions=allowed_actions)}\n\n"
            "Reply again for the same task. Keep the substance of your answer if the substance was correct, "
            "and fix only the unusable part. Change the substance only if you genuinely need to after "
            "re-checking the transcript.\n"
            f"- The final non-empty line must end with exactly one of: {allowed}.\n"
            f"- {note_rule}\n"
            "- Put no note text on the final decision line.\n"
            "- Do not put any text after the final decision line."
        ),
    )


def _reviewer_retry_message(exc: PlapError) -> StateMessage:
    return _decision_retry_message(
        allowed_actions=("ACCEPT", "REOPEN"),
        note_rule=(
            "If you choose REOPEN, write one short review note above the final line explaining the concrete defect. "
            "If you choose ACCEPT, do not add a review note just to satisfy this instruction."
        ),
        exc=exc,
    )


def _arbitrator_retry_message(exc: PlapError) -> StateMessage:
    return _decision_retry_message(
        allowed_actions=("ACCEPT", "REVISE", "REOPEN"),
        note_rule=(
            "If you choose REVISE or REOPEN, write one short note above the final line. "
            "If you choose REVISE, make it a next-step note for the normal main step. "
            "If you choose REOPEN, make it a guidance note for the next review round. "
            "If you choose ACCEPT, do not add a note just to satisfy this instruction."
        ),
        exc=exc,
    )


async def _run_decision_actor_turn(
    *,
    actor_name: str,
    actor_config: RuntimeActorConfig,
    request,
    header_messages: Sequence[ChatMessage],
    turn_messages: list[StateMessage],
    tools: Sequence[FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    server_executors: Mapping[str, IServerToolExecutor],
    tool_call_policy_resolver: IToolCallPolicyResolver,
    chat_completion_client: IChatCompletionClient,
    prompt_cache_key_base: str | None,
    usage_ledger: UsageLedger,
    parse_decision: Callable[[StateMessage], object],
    build_retry_message: Callable[[PlapError], StateMessage],
) -> tuple[ActorFinished | ActorAwaitingClientTool | None, object | None]:
    attempts = 0
    while True:
        outcome = await _execute_actor_turn(
            actor_name=actor_name,
            actor_config=actor_config,
            request=request,
            header_messages=header_messages,
            turn_messages=turn_messages,
            tools=tools,
            tool_policies=tool_policies,
            server_executors=server_executors,
            tool_call_policy_resolver=tool_call_policy_resolver,
            chat_completion_client=chat_completion_client,
            prompt_cache_key_base=prompt_cache_key_base,
            usage_ledger=usage_ledger,
            response_format=None,
        )
        if outcome is None or isinstance(outcome, ActorAwaitingClientTool):
            return outcome, None
        try:
            decision = parse_decision(outcome.assistant)
        except PlapError as exc:
            if not _is_retryable_decision_error(exc):
                raise
            usage_ledger.record_hidden(actor_config.public_usage, outcome.usage)
            attempts += 1
            log_debug(
                logger,
                "debate.step.retry",
                actor=actor_name,
                attempt=attempts,
                max_attempts=DEBATE_STEP_MAX_ATTEMPTS,
                reason=exc.private.reason,
            )
            if attempts >= DEBATE_STEP_MAX_ATTEMPTS:
                raise
            turn_messages.append(build_retry_message(exc))
            continue
        return outcome, decision


async def run_reviewer_turn(
    *,
    state: MutableQueues,
    parts: DefenderParts,
    main_developer_message: StateMessage,
    profile: RuntimeModelProfileConfig,
    request,
    normal_tools: Sequence[FunctionTool],
    tools: Sequence[FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    server_executors: Mapping[str, IServerToolExecutor],
    tool_call_policy_resolver: IToolCallPolicyResolver,
    chat_completion_client: IChatCompletionClient,
    prompt_cache_key_base: str | None,
    usage_ledger: UsageLedger,
) -> tuple[ActorFinished | ActorAwaitingClientTool | None, ReviewerDecision | None]:
    thread = _thread_messages(state.reviewer)
    header_messages = _reviewer_header_messages(
        state=state,
        parts=parts,
        main_developer_message=main_developer_message,
        profile=profile,
        request=request,
        normal_tools=normal_tools,
        tool_policies=tool_policies,
        thread=thread,
    )
    if _thread_waiting_after_tool_output(thread):
        turn_messages: list[StateMessage] = []
    elif thread:
        latest_response_note = _latest_assistant([row.message for row in parts.remaining_temp_rows])
        latest_guidance_note = _latest_arbitrator_note(_thread_messages(state.arbitrator))
        if latest_response_note is None:
            raise _debate_internal_error(
                reason="reviewer_reopen_missing_latest_response_note", private_message="reviewer reopen is missing latest response note"
            )
        turn_messages = [
            _reviewer_reopen_turn(
                latest_response_note=latest_response_note,
                latest_guidance_note=latest_guidance_note,
            )
        ]
    else:
        turn_messages = [_reviewer_initial_turn(parts)]
    return await _run_decision_actor_turn(
        actor_name=Side.REVIEWER.value,
        actor_config=profile.reviewer,
        request=request,
        header_messages=header_messages,
        turn_messages=turn_messages,
        tools=tools,
        tool_policies=tool_policies,
        server_executors=server_executors,
        tool_call_policy_resolver=tool_call_policy_resolver,
        chat_completion_client=chat_completion_client,
        prompt_cache_key_base=prompt_cache_key_base,
        usage_ledger=usage_ledger,
        parse_decision=parse_reviewer_decision,
        build_retry_message=_reviewer_retry_message,
    )


async def run_defender_turn(
    *,
    state: MutableQueues,
    parts: DefenderParts,
    profile: RuntimeModelProfileConfig,
    request,
    normal_tools: Sequence[FunctionTool],
    tools: Sequence[FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    server_executors: Mapping[str, IServerToolExecutor],
    tool_call_policy_resolver: IToolCallPolicyResolver,
    chat_completion_client: IChatCompletionClient,
    prompt_cache_key_base: str | None,
    usage_ledger: UsageLedger,
) -> ActorFinished | ActorAwaitingClientTool | None:
    thread = [row.message for row in parts.remaining_temp_rows]
    header_messages = _defender_header_messages(
        state=state,
        parts=parts,
        request=request,
        normal_tools=normal_tools,
        tool_policies=tool_policies,
    )
    if _thread_waiting_after_tool_output(thread):
        turn_messages: list[StateMessage] = []
    else:
        reviewer_decision = _latest_reviewer_decision(_thread_messages(state.reviewer))
        if reviewer_decision is None:
            raise _debate_internal_error(
                reason="defender_missing_reviewer_decision", private_message="defender is missing reviewer decision"
            )
        turn_messages = [_defender_turn(reviewer_decision=reviewer_decision)]
    return await _execute_actor_turn(
        actor_name=Actor.DEFENDER.value,
        actor_config=profile.defender,
        request=request,
        header_messages=header_messages,
        turn_messages=turn_messages,
        tools=tools,
        tool_policies=tool_policies,
        server_executors=server_executors,
        tool_call_policy_resolver=tool_call_policy_resolver,
        chat_completion_client=chat_completion_client,
        prompt_cache_key_base=prompt_cache_key_base,
        usage_ledger=usage_ledger,
        response_format=None,
    )


async def run_arbitrator_turn(
    *,
    state: MutableQueues,
    parts: DefenderParts,
    main_developer_message: StateMessage,
    profile: RuntimeModelProfileConfig,
    request,
    normal_tools: Sequence[FunctionTool],
    tools: Sequence[FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    server_executors: Mapping[str, IServerToolExecutor],
    tool_call_policy_resolver: IToolCallPolicyResolver,
    chat_completion_client: IChatCompletionClient,
    prompt_cache_key_base: str | None,
    usage_ledger: UsageLedger,
) -> tuple[ActorFinished | ActorAwaitingClientTool | None, ArbitratorDecision | None]:
    reviewer_decision = _latest_reviewer_decision(_thread_messages(state.reviewer))
    latest_response_note = _latest_assistant([row.message for row in parts.remaining_temp_rows])
    if reviewer_decision is None or latest_response_note is None:
        raise _debate_internal_error(
            reason="final_decision_missing_review_or_response_note",
            private_message="final decision step is missing review or response note",
        )
    thread = _thread_messages(state.arbitrator)
    header_messages = _arbitrator_header_messages(
        state=state,
        parts=parts,
        main_developer_message=main_developer_message,
        profile=profile,
        request=request,
        normal_tools=normal_tools,
        tool_policies=tool_policies,
        thread=thread,
    )
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
    return await _run_decision_actor_turn(
        actor_name=Side.ARBITRATOR.value,
        actor_config=profile.arbitrator,
        request=request,
        header_messages=header_messages,
        turn_messages=turn_messages,
        tools=tools,
        tool_policies=tool_policies,
        server_executors=server_executors,
        tool_call_policy_resolver=tool_call_policy_resolver,
        chat_completion_client=chat_completion_client,
        prompt_cache_key_base=prompt_cache_key_base,
        usage_ledger=usage_ledger,
        parse_decision=parse_arbitrator_decision,
        build_retry_message=_arbitrator_retry_message,
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
                content=server_outputs.get(index, DEBATE_HELD_TOOL_PLACEHOLDER),
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
            side=Side.DEFENDER,
            messages=messages,
            continuation_side=Side.REVIEWER,
            out=out,
            debug_debate_summaries=debug_debate_summaries,
            keyring=keyring,
        )
        return

    payload = ReasoningPayload(
        side=Side.DEFENDER,
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
        state.append_side(Side.DEFENDER, message)
    state.set_continuation(Side.REVIEWER)


async def publish_accepted_candidate(
    *,
    state: MutableQueues,
    out: ResponseEventIO,
    debug_debate_summaries: bool,
    keyring: SealingKeyring,
    tool_policies: Mapping[str, ToolPolicy],
    server_executors: Mapping[str, IServerToolExecutor],
) -> DebateResult:
    parts = state.defender_parts()
    if parts.held_candidate is None:
        raise _debate_internal_error(reason="held_candidate_missing", private_message="debate temp state is missing held candidate")
    candidate = parts.held_candidate.message
    hidden_outputs = {
        row.message.tool_call_id: row.message.content_text() or ""
        for row in parts.held_hidden_tool_rows
        if row.message.tool_call_id is not None
    }
    emitted_outputs = {call_id: output for call_id, output in hidden_outputs.items() if output != DEBATE_HELD_TOOL_PLACEHOLDER}
    has_client_handoff = False
    for call in candidate.tool_calls:
        output = hidden_outputs.get(call.id)
        if output is not None and output != DEBATE_HELD_TOOL_PLACEHOLDER:
            continue
        policy = tool_policies.get(call.name)
        if policy is None:
            raise _debate_internal_error(
                reason="debate_accepted_candidate_tool_policy_missing",
                private_message=f"debate accepted candidate tool policy is missing: {call.name}",
            )
        if policy.source == "server":
            executor = server_executors.get(call.name)
            if executor is None:
                raise _debate_internal_error(
                    reason="debate_accepted_candidate_server_tool_executor_missing",
                    private_message=f"debate accepted candidate server tool executor is missing: {call.name}",
                )
            emitted_outputs[call.id] = await executor.call_tool(
                call.name,
                canonical_tool_arguments(call.arguments),
            )
            log_debug(logger, "debate.accepted_candidate.server_tool_executed", tool_name=call.name)
            continue
        has_client_handoff = True
    published = await out.publish_main_candidate(
        candidate=candidate,
        keyring=keyring,
        server_outputs=emitted_outputs,
        reasoning_summary=_debug_reasoning_summary(enabled=debug_debate_summaries, texts=_single_text(candidate.content)),
    )

    state.append_main(candidate, content_hash=published.assistant_hash)

    for call in candidate.tool_calls:
        output = hidden_outputs.get(call.id)
        if output is None or output == DEBATE_HELD_TOOL_PLACEHOLDER:
            output = emitted_outputs.get(call.id)
        if output is None or output == DEBATE_HELD_TOOL_PLACEHOLDER:
            continue
        state.append_main(StateMessage(role="tool", tool_call_id=call.id, content=output))

    state.clear_debate()
    if candidate.tool_calls and not has_client_handoff:
        log_debug(
            logger,
            "debate.accepted_candidate.continue_main",
            executed_server_outputs=len(emitted_outputs),
            tool_call_count=len(candidate.tool_calls),
        )
        return DebateResult.CONTINUE_MAIN
    log_debug(
        logger,
        "debate.accepted_candidate.completed",
        emitted_output_count=len(emitted_outputs),
        has_client_handoff=has_client_handoff,
        tool_call_count=len(candidate.tool_calls),
    )
    return DebateResult.COMPLETED


async def resume_main_with_revise_bundle(
    *,
    state: MutableQueues,
    out: ResponseEventIO,
    debug_debate_summaries: bool,
    keyring: SealingKeyring,
    note: str,
) -> None:
    parts = state.defender_parts()
    if parts.held_candidate is None:
        raise _debate_internal_error(reason="revise_requires_held_candidate", private_message="revise requires held candidate state")

    note_message = StateMessage(role="assistant", content=_revise_note_wrapper(note))
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
    state.append_main(parts.held_candidate.message, content_hash=parts.held_candidate.content_hash)
    for row in parts.held_hidden_tool_rows:
        state.append_main(row.message, content_hash=row.content_hash)
    state.append_main(note_message)
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
    tool_call_policy_resolver: IToolCallPolicyResolver,
    chat_completion_client: IChatCompletionClient,
    prompt_cache_key_base: str | None,
    usage_ledger: UsageLedger,
    held_anchor_index: int | None = None,
) -> DebateResult:
    callable_tools, callable_tool_policies, callable_server_executors = debate_callable_surface(
        tools,
        tool_policies,
        server_executors,
    )
    parts = state.defender_parts()
    if parts.held_candidate is None:
        raise _debate_internal_error(reason="held_candidate_missing", private_message="debate temp state is missing held candidate")

    if profile.debate_max_rounds == 0:
        debate_result = await publish_accepted_candidate(
            state=state,
            out=out,
            debug_debate_summaries=debug_debate_summaries,
            keyring=keyring,
            tool_policies=tool_policies,
            server_executors=server_executors,
        )
        if debate_result == DebateResult.COMPLETED:
            if held_anchor_index is not None:
                usage_ledger.use_hidden_as_anchor(held_anchor_index)
            await out.completed(service_tier=request.service_tier, usage=usage_ledger.to_response_usage())
            return DebateResult.COMPLETED
        return DebateResult.CONTINUE_MAIN

    while True:
        actor = state.current_actor()
        log_debug(logger, "debate.turn", actor=actor, held_anchor_index=held_anchor_index)

        if actor == Actor.REVIEWER:
            outcome, decision = await run_reviewer_turn(
                state=state,
                parts=parts,
                main_developer_message=main_developer_message,
                profile=profile,
                request=request,
                normal_tools=tools,
                tools=callable_tools,
                tool_policies=callable_tool_policies,
                server_executors=callable_server_executors,
                tool_call_policy_resolver=tool_call_policy_resolver,
                chat_completion_client=chat_completion_client,
                prompt_cache_key_base=prompt_cache_key_base,
                usage_ledger=usage_ledger,
            )
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
                continuation_side=Side.DEFENDER,
                out=out,
                debug_debate_summaries=debug_debate_summaries,
                keyring=keyring,
            )
            if decision.action == ReviewerActionType.ACCEPT:
                debate_result = await publish_accepted_candidate(
                    state=state,
                    out=out,
                    debug_debate_summaries=debug_debate_summaries,
                    keyring=keyring,
                    tool_policies=tool_policies,
                    server_executors=server_executors,
                )
                if debate_result == DebateResult.CONTINUE_MAIN:
                    usage_ledger.record_hidden(profile.reviewer.public_usage, outcome.usage)
                    return DebateResult.CONTINUE_MAIN
                if held_anchor_index is not None:
                    usage_ledger.record_hidden(profile.reviewer.public_usage, outcome.usage)
                    usage_ledger.use_hidden_as_anchor(held_anchor_index)
                else:
                    usage_ledger.set_anchor(outcome.usage)
                await out.completed(service_tier=request.service_tier, usage=usage_ledger.to_response_usage())
                return DebateResult.COMPLETED

            usage_ledger.record_hidden(profile.reviewer.public_usage, outcome.usage)
            parts = state.defender_parts()
            continue

        if actor == Actor.DEFENDER:
            outcome = await run_defender_turn(
                state=state,
                parts=parts,
                profile=profile,
                request=request,
                normal_tools=tools,
                tools=callable_tools,
                tool_policies=callable_tool_policies,
                server_executors=callable_server_executors,
                tool_call_policy_resolver=tool_call_policy_resolver,
                chat_completion_client=chat_completion_client,
                prompt_cache_key_base=prompt_cache_key_base,
                usage_ledger=usage_ledger,
            )
            if outcome is None:
                if held_anchor_index is not None and usage_ledger.anchor is None:
                    usage_ledger.use_hidden_as_anchor(held_anchor_index)
                await out.incomplete(service_tier=request.service_tier, usage=usage_ledger.to_response_usage())
                return DebateResult.COMPLETED
            if isinstance(outcome, ActorAwaitingClientTool):
                usage_ledger.set_anchor(outcome.usage)
                await _persist_temp_turn(
                    state=state,
                    side=Side.DEFENDER,
                    messages=outcome.messages,
                    continuation_side=Side.DEFENDER,
                    out=out,
                    debug_debate_summaries=debug_debate_summaries,
                    keyring=keyring,
                )
                await _emit_debate_function_calls(
                    side=Side.DEFENDER,
                    assistant=outcome.assistant,
                    tool_calls=outcome.tool_calls,
                    out=out,
                    keyring=keyring,
                )
                await out.completed(service_tier=request.service_tier, usage=usage_ledger.to_response_usage())
                return DebateResult.COMPLETED

            usage_ledger.record_hidden(profile.defender.public_usage, outcome.usage)
            await _persist_temp_turn(
                state=state,
                side=Side.DEFENDER,
                messages=outcome.messages,
                continuation_side=Side.ARBITRATOR,
                out=out,
                debug_debate_summaries=debug_debate_summaries,
                keyring=keyring,
            )
            parts = state.defender_parts()
            continue

        if actor == Actor.ARBITRATOR:
            outcome, decision = await run_arbitrator_turn(
                state=state,
                parts=parts,
                main_developer_message=main_developer_message,
                profile=profile,
                request=request,
                normal_tools=tools,
                tools=callable_tools,
                tool_policies=callable_tool_policies,
                server_executors=callable_server_executors,
                tool_call_policy_resolver=tool_call_policy_resolver,
                chat_completion_client=chat_completion_client,
                prompt_cache_key_base=prompt_cache_key_base,
                usage_ledger=usage_ledger,
            )
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
            continuation = Side.REVIEWER if decision.action == ArbitratorActionType.REOPEN else Side.DEFENDER
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
                debate_result = await publish_accepted_candidate(
                    state=state,
                    out=out,
                    debug_debate_summaries=debug_debate_summaries,
                    keyring=keyring,
                    tool_policies=tool_policies,
                    server_executors=server_executors,
                )
                if debate_result == DebateResult.CONTINUE_MAIN:
                    usage_ledger.record_hidden(profile.arbitrator.public_usage, outcome.usage)
                    return DebateResult.CONTINUE_MAIN
                if held_anchor_index is not None:
                    usage_ledger.record_hidden(profile.arbitrator.public_usage, outcome.usage)
                    usage_ledger.use_hidden_as_anchor(held_anchor_index)
                else:
                    usage_ledger.set_anchor(outcome.usage)
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

            parts = state.defender_parts()
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
        raise _debate_internal_error(
            reason="persist_temp_turn_main_side_unsupported",
            private_message="persisting temp debate turns on main is no longer supported",
        )
    for message in messages:
        state.append_side(side, message)
    state.set_continuation(continuation_side)
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
                        temp=True,
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
