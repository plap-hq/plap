from __future__ import annotations

import json
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from plap.keyring import SealingKeyring
from plap.llms.chat import (
    ChatCompletionRequest,
    ChatFunctionTool,
    ChatResponseFormat,
    ChatTool,
    ChatToolCall,
    IChatCompletionClient,
)
from plap.llms.chat import (
    ChatMessage as LLMChatMessage,
)
from plap.responses.contracts import (
    FunctionTool,
    OutputTextContent,
    ReasoningItem,
    ResponseFunctionCallItem,
    ResponseFunctionCallOutputItem,
    ResponseMessageItem,
)
from plap.responses.ingest import ChatMessageSpan, ReasoningPayload, SealedCallID, SideMessage
from plap.responses.ingest.sealing import content_hash, content_hash_prefix, seal_call_id, seal_reasoning_payload
from plap.responses.ingest.types import Side
from plap.responses.io import ResponseEventIO
from plap.responses.tokens import estimate_message_tokens
from plap.responses.tools import IToolCallPolicyResolver, ToolCall, ToolPolicy, ToolPolicyError, canonical_tool_arguments
from plap.responses.tools.mcp import IMCPToolProvider
from plap.settings import RuntimeModelProfileConfig

type ReviewerAction = Literal["accept", "challenge"]
type ArbitratorAction = Literal["accept", "answer", "revise", "reopen"]

HELD_CLIENT_TOOL_PLACEHOLDER = "This tool call was not executed."

REVIEWER_DEVELOPER_PROMPT = """You are checking an assistant draft before it is shown.

Decide whether the draft is correct, grounded in the visible conversation, and consistent with the user's request.
Return only JSON matching the requested schema.

Actions:
- accept: no material issue was found; the draft is good enough to publish.
- challenge: a material correctness issue, missing constraint, bad tool call, or unsupported assumption needs review.

Keep rationale and critique short. Do not write a public answer."""

MAIN_DEBATE_DEVELOPER_PROMPT = """You are reconsidering a held assistant draft after a reviewer challenge.

Respond privately. Check the candidate against the conversation and reviewer critique. If tool use helps, use only the available tools.
Do not write for the user unless asked by the arbitration step later."""

ARBITRATOR_DEVELOPER_PROMPT = """You decide how to resolve a private correctness review.

Return only JSON matching the requested schema.

Actions:
- accept: publish the original held draft.
- answer: write the final public assistant message now.
- revise: carry guidance back to the main assistant for another normal attempt.
- reopen: ask the reviewer to check again using your guidance.

Use answer only when the final user-visible response is clear and does not need client tool execution.
Use revise when another main attempt should incorporate guidance."""

REVIEWER_RESPONSE_FORMAT = ChatResponseFormat(
    type="json_schema",
    name="reviewer_verdict",
    strict=True,
    schema={
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "rationale", "critique"],
        "properties": {
            "action": {"type": "string", "enum": ["accept", "challenge"]},
            "rationale": {"type": "string"},
            "critique": {"type": "string"},
        },
    },
)

ADJUDICATOR_RESPONSE_FORMAT = ChatResponseFormat(
    type="json_schema",
    name="arbitrator_verdict",
    strict=True,
    schema={
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "rationale", "message", "guidance"],
        "properties": {
            "action": {"type": "string", "enum": ["accept", "answer", "revise", "reopen"]},
            "rationale": {"type": "string"},
            "message": {"type": "string"},
            "guidance": {"type": "string"},
        },
    },
)


@dataclass(slots=True)
class RuntimeState:
    main_context: list[ChatMessageSpan]
    main_context_temp: list[ChatMessageSpan]
    reviewer: list[SideMessage]
    arbitrator: list[SideMessage]
    cursors: dict[str, int]
    continuation_side: Side
    in_temp_debate: bool


@dataclass(slots=True)
class HeldCandidate:
    assistant: dict[str, Any]
    hidden_tool_messages: list[dict[str, Any]]
    main_debate_messages: list[dict[str, Any]]


@dataclass(slots=True)
class ActorFinished:
    messages: list[dict[str, Any]]
    assistant: dict[str, Any]


@dataclass(slots=True)
class ActorAwaitingTool:
    messages: list[dict[str, Any]]
    tool_calls: list[ChatToolCall]


@dataclass(slots=True)
class DebateResponseCompleted:
    pass


@dataclass(slots=True)
class DebateContinueMain:
    pass


def debate_safe_surface(
    tools: Sequence[FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    server_executors: Mapping[str, IMCPToolProvider],
) -> tuple[list[FunctionTool], dict[str, ToolPolicy], dict[str, IMCPToolProvider]]:
    safe_policies = {name: policy for name, policy in tool_policies.items() if policy.effect_class == "safe"}
    safe_tools = [tool for tool in tools if tool.name in safe_policies]
    safe_executors = {name: executor for name, executor in server_executors.items() if name in safe_policies}
    return safe_tools, safe_policies, safe_executors


def split_main_temp(rows: Sequence[ChatMessageSpan]) -> HeldCandidate:
    if not rows:
        raise ToolPolicyError("debate candidate is missing")
    assistant = dict(rows[0].message)
    if assistant.get("role") != "assistant":
        raise ToolPolicyError("debate candidate must be an assistant message")

    hidden_tool_messages: list[dict[str, Any]] = []
    index = 1
    while index < len(rows) and rows[index].message.get("role") == "tool":
        hidden_tool_messages.append(dict(rows[index].message))
        index += 1
    return HeldCandidate(
        assistant=assistant,
        hidden_tool_messages=hidden_tool_messages,
        main_debate_messages=[dict(row.message) for row in rows[index:]],
    )


def compact_debate_transcript(spans: Sequence[ChatMessageSpan]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    pending_tool_calls: dict[str, dict[str, Any]] = {}
    for span in spans:
        message = span.message
        role = message.get("role")
        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id in pending_tool_calls:
                pending_tool_calls[tool_call_id]["output"] = _message_content_text(message.get("content")) or ""
            continue

        item = _compact_chat_message(message)
        compact.append(item)
        pending_tool_calls = {
            call["_id"]: call
            for call in item.get("tool_calls", [])
            if isinstance(call, dict) and isinstance(call.get("_id"), str)
        }
    return [_strip_internal_tool_call_ids(item) for item in compact]


def _reviewer_round_count(thread: Sequence[SideMessage]) -> int:
    return sum(1 for row in thread if row.message.get("role") == "user")


async def start_debate_from_candidate(
    state: RuntimeState,
    *,
    result_message: LLMChatMessage,
    tool_calls: Sequence[ChatToolCall],
    resolved_policies: Sequence[ToolPolicy],
    server_executors: Mapping[str, IMCPToolProvider],
    out: ResponseEventIO,
    keyring: SealingKeyring,
) -> None:
    held_messages = [_assistant_message_with_tool_calls(result_message)]
    for call, policy in zip(tool_calls, resolved_policies, strict=True):
        if policy.source == "server":
            executor = server_executors.get(call.name)
            if executor is None:
                raise ToolPolicyError("server tool executor is not configured")
            content = await executor.call_tool(call.name, canonical_tool_arguments(call.arguments))
        else:
            content = HELD_CLIENT_TOOL_PLACEHOLDER
        held_messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

    await persist_debate_turn(
        state,
        out=out,
        keyring=keyring,
        side="main",
        continuation_side="reviewer",
        messages=held_messages,
    )


async def continue_debate(
    state: RuntimeState,
    *,
    profile: RuntimeModelProfileConfig,
    main_transcript: Sequence[ChatMessageSpan],
    tools: Sequence[FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    server_executors: Mapping[str, IMCPToolProvider],
    tool_call_policy_resolver: IToolCallPolicyResolver,
    chat_completion_client: IChatCompletionClient,
    keyring: SealingKeyring,
    out: ResponseEventIO,
) -> DebateResponseCompleted | DebateContinueMain:
    if profile.debate_max_rounds == 0:
        await publish_accepted_candidate(state, out=out, keyring=keyring)
        await out.completed()
        return DebateResponseCompleted()

    safe_tools, safe_policies, safe_server_executors = debate_safe_surface(tools, tool_policies, server_executors)
    while True:
        if state.continuation_side == "reviewer":
            held = split_main_temp(state.main_context_temp)
            thread = [dict(row.message) for row in state.reviewer]
            user_turn = (
                None
                if _thread_is_waiting_after_tool_output(thread)
                else _reviewer_user_turn(main_transcript, held, state.arbitrator)
            )
            result = await continue_actor(
                side="reviewer",
                model=profile.reviewer_model,
                developer_prompt=REVIEWER_DEVELOPER_PROMPT,
                thread=thread,
                user_turn=user_turn,
                response_format=REVIEWER_RESPONSE_FORMAT,
                safe_tools=safe_tools,
                safe_policies=safe_policies,
                safe_server_executors=safe_server_executors,
                tool_call_policy_resolver=tool_call_policy_resolver,
                chat_completion_client=chat_completion_client,
            )
            if isinstance(result, ActorAwaitingTool):
                await persist_debate_turn(
                    state,
                    out=out,
                    keyring=keyring,
                    side="reviewer",
                    continuation_side="reviewer",
                    messages=result.messages,
                )
                await _emit_debate_function_calls(
                    out,
                    keyring=keyring,
                    side="reviewer",
                    messages=result.messages,
                    tool_calls=result.tool_calls,
                )
                await out.completed()
                return DebateResponseCompleted()

            await persist_debate_turn(state, out=out, keyring=keyring, side="reviewer", continuation_side="main", messages=result.messages)
            reviewer = _reviewer_result(result.assistant)
            if reviewer["action"] == "accept":
                await publish_accepted_candidate(state, out=out, keyring=keyring)
                await out.completed()
                return DebateResponseCompleted()
            state.continuation_side = "main"
            continue

        if state.continuation_side == "main":
            held = split_main_temp(state.main_context_temp)
            thread = held.main_debate_messages
            user_turn = None if _thread_is_waiting_after_tool_output(thread) else _main_debate_user_turn(held, state.reviewer)
            result = await continue_actor(
                side="main",
                model=profile.main_debate_model,
                developer_prompt=MAIN_DEBATE_DEVELOPER_PROMPT,
                thread=thread,
                user_turn=user_turn,
                response_format=None,
                safe_tools=safe_tools,
                safe_policies=safe_policies,
                safe_server_executors=safe_server_executors,
                tool_call_policy_resolver=tool_call_policy_resolver,
                chat_completion_client=chat_completion_client,
            )
            if isinstance(result, ActorAwaitingTool):
                await persist_debate_turn(state, out=out, keyring=keyring, side="main", continuation_side="main", messages=result.messages)
                await _emit_debate_function_calls(out, keyring=keyring, side="main", messages=result.messages, tool_calls=result.tool_calls)
                await out.completed()
                return DebateResponseCompleted()

            await persist_debate_turn(
                state,
                out=out,
                keyring=keyring,
                side="main",
                continuation_side="arbitrator",
                messages=result.messages,
            )
            state.continuation_side = "arbitrator"
            continue

        held = split_main_temp(state.main_context_temp)
        thread = [dict(row.message) for row in state.arbitrator]
        user_turn = None if _thread_is_waiting_after_tool_output(thread) else _arbitrator_user_turn(main_transcript, held, state.reviewer)
        result = await continue_actor(
            side="arbitrator",
            model=profile.arbitrator_model,
            developer_prompt=ARBITRATOR_DEVELOPER_PROMPT,
            thread=thread,
            user_turn=user_turn,
            response_format=ADJUDICATOR_RESPONSE_FORMAT,
            safe_tools=safe_tools,
            safe_policies=safe_policies,
            safe_server_executors=safe_server_executors,
            tool_call_policy_resolver=tool_call_policy_resolver,
            chat_completion_client=chat_completion_client,
        )
        if isinstance(result, ActorAwaitingTool):
            await persist_debate_turn(
                state,
                out=out,
                keyring=keyring,
                side="arbitrator",
                continuation_side="arbitrator",
                messages=result.messages,
            )
            await _emit_debate_function_calls(
                out,
                keyring=keyring,
                side="arbitrator",
                messages=result.messages,
                tool_calls=result.tool_calls,
            )
            await out.completed()
            return DebateResponseCompleted()

        arbitrator = _arbitrator_result(result.assistant)
        next_side: Side = "reviewer" if arbitrator["action"] == "reopen" else "main"
        await persist_debate_turn(state, out=out, keyring=keyring, side="arbitrator", continuation_side=next_side, messages=result.messages)

        action = arbitrator["action"]
        if action == "accept":
            await publish_accepted_candidate(state, out=out, keyring=keyring)
            await out.completed()
            return DebateResponseCompleted()
        if action == "answer":
            await answer_from_arbitrator(state, out=out, message=arbitrator["message"])
            await out.completed()
            return DebateResponseCompleted()
        if action == "revise":
            await resume_main_with_guidance(state, out=out, keyring=keyring, guidance=arbitrator["guidance"])
            return DebateContinueMain()

        if _reviewer_round_count(state.reviewer) >= profile.debate_max_rounds:
            await resume_main_with_guidance(state, out=out, keyring=keyring, guidance=arbitrator["guidance"])
            return DebateContinueMain()
        state.continuation_side = "reviewer"


async def continue_actor(
    *,
    side: Side,
    model: str,
    developer_prompt: str,
    thread: Sequence[dict[str, Any]],
    user_turn: dict[str, Any] | None,
    response_format: ChatResponseFormat | None,
    safe_tools: Sequence[FunctionTool],
    safe_policies: Mapping[str, ToolPolicy],
    safe_server_executors: Mapping[str, IMCPToolProvider],
    tool_call_policy_resolver: IToolCallPolicyResolver,
    chat_completion_client: IChatCompletionClient,
) -> ActorFinished | ActorAwaitingTool:
    _ = side
    messages: list[dict[str, Any]] = []
    if user_turn is not None:
        messages.append(user_turn)

    while True:
        request = ChatCompletionRequest(
            model=model,
            messages=[LLMChatMessage(role="developer", content=developer_prompt)]
            + [_llm_message_from_dict(message) for message in thread]
            + [_llm_message_from_dict(message) for message in messages],
            tools=[_chat_tool(tool) for tool in safe_tools],
            response_format=response_format,
        )
        result = await chat_completion_client.complete(request)
        assistant = _assistant_message_with_tool_calls(result.message)
        tool_calls = result.message.tool_calls or []
        if not tool_calls:
            messages.append(assistant)
            return ActorFinished(messages=messages, assistant=assistant)

        resolved = await _resolve_safe_actor_tool_calls(
            tool_calls,
            tools=safe_tools,
            policies=safe_policies,
            resolver=tool_call_policy_resolver,
        )
        server_outputs: list[dict[str, Any]] = []
        client_calls: list[ChatToolCall] = []
        for call, policy in zip(tool_calls, resolved, strict=True):
            if policy.source == "server":
                executor = safe_server_executors.get(call.name)
                if executor is None:
                    raise ToolPolicyError("server tool executor is not configured")
                server_outputs.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": await executor.call_tool(call.name, canonical_tool_arguments(call.arguments)),
                    }
                )
            else:
                if policy.effect_class != "safe":
                    raise ToolPolicyError("debate tool call requires unsupported policy path")
                client_calls.append(call)

        messages.append(assistant)
        messages.extend(server_outputs)
        if client_calls:
            return ActorAwaitingTool(messages=messages, tool_calls=client_calls)


async def persist_debate_turn(
    state: RuntimeState,
    *,
    out: ResponseEventIO,
    keyring: SealingKeyring,
    side: Side,
    continuation_side: Side,
    messages: Sequence[dict[str, Any]],
) -> None:
    payload = ReasoningPayload(side=side, temp=True, continuation_side=continuation_side, messages=tuple(messages))
    reasoning_item = ReasoningItem(
        encrypted_content=seal_reasoning_payload(payload, keyring=keyring),
        id=f"rs_{secrets.token_urlsafe(18)}",
        status="completed",
        summary=[],
        type="reasoning",
    )
    await out.output(reasoning_item, reasoning_side=side, reasoning_messages=payload.messages)

    if side == "main":
        _append_main_temp_messages(state, messages)
    elif side == "reviewer":
        state.reviewer.extend(SideMessage(message=dict(message)) for message in messages)
    else:
        state.arbitrator.extend(SideMessage(message=dict(message)) for message in messages)
    state.in_temp_debate = True
    state.continuation_side = continuation_side


async def publish_accepted_candidate(
    state: RuntimeState,
    *,
    out: ResponseEventIO,
    keyring: SealingKeyring,
) -> None:
    held = split_main_temp(state.main_context_temp)
    assistant = held.assistant
    public_assistant = {"role": "assistant"}
    if assistant.get("content") is not None:
        public_assistant["content"] = _message_content_text(assistant.get("content")) or ""
    elif assistant.get("reasoning_content") or assistant.get("reasoning_details") or assistant.get("tool_calls"):
        public_assistant["content"] = ""
    assistant_hash = content_hash(public_assistant)

    if (
        "content" in public_assistant
        or assistant.get("reasoning_content")
        or assistant.get("reasoning_details")
        or assistant.get("tool_calls")
    ):
        await _emit_public_message(out, public_assistant.get("content", ""))

    if assistant.get("reasoning_content") or assistant.get("reasoning_details"):
        reasoning_message: dict[str, Any] = {"content_hash": assistant_hash}
        if assistant.get("reasoning_content") is not None:
            reasoning_message["reasoning_content"] = assistant["reasoning_content"]
        if assistant.get("reasoning_details") is not None:
            reasoning_message["reasoning_details"] = assistant["reasoning_details"]
        payload = ReasoningPayload(side="main", temp=False, continuation_side="main", messages=(reasoning_message,))
        await out.output(
            ReasoningItem(
                encrypted_content=seal_reasoning_payload(payload, keyring=keyring),
                id=f"rs_{secrets.token_urlsafe(18)}",
                status="completed",
                summary=[],
                type="reasoning",
            ),
            reasoning_side="main",
            reasoning_messages=payload.messages,
        )

    tool_calls = _tool_calls_from_assistant_message(assistant)
    public_call_ids: dict[str, str] = {}
    assistant_context_tool_calls: list[dict[str, Any]] = []
    for index, call in enumerate(tool_calls):
        sealed_call_id = seal_call_id(
            SealedCallID(
                side="main",
                content_hash_prefix=content_hash_prefix(assistant_hash),
                tool_call_index=index,
                upstream_tool_call_id=call.id,
            ),
            keyring=keyring,
        )
        public_call_ids[call.id] = sealed_call_id
        assistant_context_tool_calls.append(_tool_call_dict(call))
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

    output_messages = [message for message in held.hidden_tool_messages if message.get("content") != HELD_CLIENT_TOOL_PLACEHOLDER]
    for message in output_messages:
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or tool_call_id not in public_call_ids:
            continue
        await out.output(
            ResponseFunctionCallOutputItem(
                call_id=public_call_ids[tool_call_id],
                created_by="server",
                id=f"fco_{secrets.token_urlsafe(18)}",
                output=_message_content_text(message.get("content")) or "",
                status="completed",
                type="function_call_output",
            )
        )

    if "content" in public_assistant or assistant_context_tool_calls:
        context_message = dict(public_assistant)
        if assistant_context_tool_calls:
            context_message["tool_calls"] = assistant_context_tool_calls
        _append_main_stable_message(state, context_message, content_hash_value=assistant_hash)
    for message in output_messages:
        _append_main_stable_message(
            state,
            {
                "role": "tool",
                "tool_call_id": message["tool_call_id"],
                "content": _message_content_text(message.get("content")) or "",
            },
        )
    _clear_debate(state)


async def answer_from_arbitrator(state: RuntimeState, *, out: ResponseEventIO, message: str) -> None:
    await _emit_public_message(out, message)
    _append_main_stable_message(state, {"role": "assistant", "content": message})
    _clear_debate(state)


async def resume_main_with_guidance(
    state: RuntimeState,
    *,
    out: ResponseEventIO,
    keyring: SealingKeyring,
    guidance: str,
) -> None:
    message = {"role": "assistant", "content": guidance}
    payload = ReasoningPayload(side="main", temp=False, continuation_side="main", messages=(message,))
    await out.output(
        ReasoningItem(
            encrypted_content=seal_reasoning_payload(payload, keyring=keyring),
            id=f"rs_{secrets.token_urlsafe(18)}",
            status="completed",
            summary=[],
            type="reasoning",
        ),
        reasoning_side="main",
        reasoning_messages=payload.messages,
    )
    _append_main_stable_message(state, message)
    _clear_debate(state)


def _append_main_temp_messages(state: RuntimeState, messages: Sequence[dict[str, Any]]) -> None:
    for message in messages:
        ordinal = state.cursors["m"]
        state.cursors["m"] = ordinal + 1
        state.main_context_temp.append(
            ChatMessageSpan(
                start=ordinal,
                end=ordinal,
                message=dict(message),
                token_count=estimate_message_tokens(message),
            )
        )


def _append_main_stable_message(
    state: RuntimeState,
    message: dict[str, Any],
    *,
    content_hash_value: str = "",
) -> None:
    ordinal = state.cursors["m"]
    state.cursors["m"] = ordinal + 1
    state.main_context.append(
        ChatMessageSpan(
            start=ordinal,
            end=ordinal,
            message=dict(message),
            content_hash=content_hash_value,
            token_count=estimate_message_tokens(message),
        )
    )


def _clear_debate(state: RuntimeState) -> None:
    state.main_context_temp.clear()
    state.reviewer.clear()
    state.arbitrator.clear()
    state.in_temp_debate = False
    state.continuation_side = "main"


async def _emit_debate_function_calls(
    out: ResponseEventIO,
    *,
    keyring: SealingKeyring,
    side: Side,
    messages: Sequence[dict[str, Any]],
    tool_calls: Sequence[ChatToolCall],
) -> None:
    assistant = _last_assistant_with_tool_calls(messages)
    assistant_hash = content_hash(assistant)
    all_calls = _tool_calls_from_assistant_message(assistant)
    for call in tool_calls:
        index = next((candidate_index for candidate_index, candidate in enumerate(all_calls) if candidate.id == call.id), None)
        if index is None:
            raise ToolPolicyError("debate tool call is missing from assistant message")
        sealed_call_id = seal_call_id(
            SealedCallID(
                side=side,
                content_hash_prefix=content_hash_prefix(assistant_hash),
                tool_call_index=index,
                upstream_tool_call_id=call.id,
            ),
            keyring=keyring,
        )
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


async def _emit_public_message(out: ResponseEventIO, content: str) -> None:
    await out.output(
        ResponseMessageItem(
            content=[OutputTextContent(text=content, type="output_text")],
            id=f"msg_{secrets.token_urlsafe(18)}",
            role="assistant",
            status="completed",
            type="message",
        )
    )


async def _resolve_safe_actor_tool_calls(
    calls: Sequence[ChatToolCall],
    *,
    tools: Sequence[FunctionTool],
    policies: Mapping[str, ToolPolicy],
    resolver: IToolCallPolicyResolver,
) -> tuple[ToolPolicy, ...]:
    resolved: list[ToolPolicy | None] = []
    client_calls: list[ToolCall] = []
    client_indexes: list[int] = []
    tool_by_name = {tool.name: tool for tool in tools}
    for call in calls:
        policy = policies.get(call.name)
        if policy is None:
            raise ToolPolicyError(f"unknown debate tool call: {call.name}")
        if policy.effect_class != "safe":
            raise ToolPolicyError("debate tool call requires unsupported policy path")
        if policy.source == "server":
            resolved.append(policy)
            continue
        tool = tool_by_name.get(call.name)
        if tool is None:
            raise ToolPolicyError(f"unknown debate client tool call: {call.name}")
        client_indexes.append(len(resolved))
        resolved.append(None)
        client_calls.append(ToolCall(tool=tool, policy=policy, arguments=call.arguments))
    if client_calls:
        client_policies = await resolver.resolve(client_calls)
        for index, policy in zip(client_indexes, client_policies, strict=True):
            if policy.effect_class != "safe":
                raise ToolPolicyError("debate tool call requires unsupported policy path")
            resolved[index] = policy
    if any(policy is None for policy in resolved):
        raise RuntimeError("debate tool policy resolution did not produce all outputs")
    return tuple(policy for policy in resolved if policy is not None)


def _reviewer_user_turn(
    main_transcript: Sequence[ChatMessageSpan],
    held: HeldCandidate,
    arbitrator_thread: Sequence[SideMessage],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "conversation": compact_debate_transcript(main_transcript),
        "candidate": _compact_held_candidate(held),
    }
    latest_main = _last_assistant(held.main_debate_messages)
    if latest_main is not None:
        payload["latest_main_debate"] = _compact_chat_message(latest_main)
    latest_arbitrator = _last_assistant([row.message for row in arbitrator_thread])
    if latest_arbitrator is not None:
        payload["arbitrator_guidance"] = _compact_chat_message(latest_arbitrator)
    return {"role": "user", "content": json.dumps(payload, sort_keys=True)}


def _main_debate_user_turn(held: HeldCandidate, reviewer_thread: Sequence[SideMessage]) -> dict[str, Any]:
    payload = {
        "candidate": _compact_held_candidate(held),
        "reviewer": _compact_chat_message(_require_last_assistant([row.message for row in reviewer_thread])),
    }
    return {"role": "user", "content": json.dumps(payload, sort_keys=True)}


def _arbitrator_user_turn(
    main_transcript: Sequence[ChatMessageSpan],
    held: HeldCandidate,
    reviewer_thread: Sequence[SideMessage],
) -> dict[str, Any]:
    latest_main = _last_assistant(held.main_debate_messages)
    payload: dict[str, Any] = {
        "conversation": compact_debate_transcript(main_transcript),
        "candidate": _compact_held_candidate(held),
        "reviewer": _compact_chat_message(_require_last_assistant([row.message for row in reviewer_thread])),
    }
    if latest_main is not None:
        payload["main_debate"] = _compact_chat_message(latest_main)
    return {"role": "user", "content": json.dumps(payload, sort_keys=True)}


def _reviewer_result(message: dict[str, Any]) -> dict[str, str]:
    payload = _json_object_from_message(message, label="reviewer")
    action = payload.get("action")
    if action not in {"accept", "challenge"}:
        raise ToolPolicyError("reviewer returned invalid action")
    rationale = payload.get("rationale")
    critique = payload.get("critique")
    if not isinstance(rationale, str) or not isinstance(critique, str):
        raise ToolPolicyError("reviewer returned invalid result")
    if action == "challenge" and not critique.strip():
        raise ToolPolicyError("reviewer challenge requires critique")
    return {"action": action, "rationale": rationale, "critique": critique}


def _arbitrator_result(message: dict[str, Any]) -> dict[str, str]:
    payload = _json_object_from_message(message, label="arbitrator")
    action = payload.get("action")
    if action not in {"accept", "answer", "revise", "reopen"}:
        raise ToolPolicyError("arbitrator returned invalid action")
    rationale = payload.get("rationale")
    public_message = payload.get("message")
    guidance = payload.get("guidance")
    if not isinstance(rationale, str) or not isinstance(public_message, str) or not isinstance(guidance, str):
        raise ToolPolicyError("arbitrator returned invalid result")
    if action == "answer" and not public_message.strip():
        raise ToolPolicyError("arbitrator answer requires message")
    if action in {"revise", "reopen"} and not guidance.strip():
        raise ToolPolicyError("arbitrator action requires guidance")
    return {"action": action, "rationale": rationale, "message": public_message, "guidance": guidance}


def _json_object_from_message(message: dict[str, Any], *, label: str) -> dict[str, Any]:
    content = _message_content_text(message.get("content"))
    if content is None:
        raise ToolPolicyError(f"{label} returned empty result")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ToolPolicyError(f"{label} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ToolPolicyError(f"{label} returned invalid result")
    return payload


def _compact_held_candidate(held: HeldCandidate) -> dict[str, Any]:
    candidate = _compact_chat_message(held.assistant)
    outputs = {
        message.get("tool_call_id"): _message_content_text(message.get("content")) or ""
        for message in held.hidden_tool_messages
        if message.get("content") != HELD_CLIENT_TOOL_PLACEHOLDER
    }
    for call in candidate.get("tool_calls", []):
        if isinstance(call, dict) and call.get("_id") in outputs:
            call["output"] = outputs[call["_id"]]
    return _strip_internal_tool_call_ids(candidate)


def _compact_chat_message(message: Mapping[str, Any]) -> dict[str, Any]:
    role = message.get("role")
    if role in {"system", "developer"}:
        role = "user"
        content = f"{str(message.get('role')).capitalize()}-role message:\n{_message_content_text(message.get('content')) or ''}"
    else:
        content = _message_content_text(message.get("content"))
    item: dict[str, Any] = {"role": role, "content": content or ""}
    tool_calls = _compact_tool_calls(message.get("tool_calls"))
    if tool_calls:
        item["tool_calls"] = tool_calls
    return item


def _compact_tool_calls(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    compact: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str):
            continue
        compact.append(
            {
                "_id": item.get("id") if isinstance(item.get("id"), str) else "",
                "name": name,
                "arguments": _json_arguments(function.get("arguments")),
            }
        )
    return compact


def _strip_internal_tool_call_ids(message: dict[str, Any]) -> dict[str, Any]:
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        message = dict(message)
        message["tool_calls"] = [{key: value for key, value in call.items() if key != "_id"} for call in tool_calls]
    return message


def _json_arguments(value: object) -> object:
    if not isinstance(value, str):
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return value
    return decoded


def _assistant_message_with_tool_calls(message: LLMChatMessage) -> dict[str, Any]:
    output: dict[str, Any] = {"role": "assistant"}
    if message.content is not None:
        output["content"] = message.content
    elif message.reasoning_content or message.reasoning_details or message.tool_calls:
        output["content"] = ""
    if message.reasoning_content is not None:
        output["reasoning_content"] = message.reasoning_content
    if message.reasoning_details is not None:
        output["reasoning_details"] = message.reasoning_details
    if message.tool_calls:
        output["tool_calls"] = [_tool_call_dict(call) for call in message.tool_calls]
    return output


def _tool_call_dict(call: ChatToolCall) -> dict[str, Any]:
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": call.arguments,
        },
    }


def _tool_calls_from_assistant_message(message: Mapping[str, Any]) -> list[ChatToolCall]:
    value = message.get("tool_calls")
    if not isinstance(value, list):
        return []
    calls: list[ChatToolCall] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        call_id = item.get("id")
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(call_id, str) or not isinstance(name, str):
            continue
        calls.append(ChatToolCall(id=call_id, name=name, arguments=arguments if isinstance(arguments, str) else "{}"))
    return calls


def _last_assistant(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return dict(message)
    return None


def _require_last_assistant(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    message = _last_assistant(messages)
    if message is None:
        raise ToolPolicyError("debate assistant message is missing")
    return message


def _last_assistant_with_tool_calls(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            return dict(message)
    raise ToolPolicyError("debate assistant tool call message is missing")


def _thread_is_waiting_after_tool_output(messages: Sequence[Mapping[str, Any]]) -> bool:
    return bool(messages) and messages[-1].get("role") == "tool"


def _llm_message_from_dict(message: Mapping[str, Any]) -> LLMChatMessage:
    role = message.get("role")
    if role not in {"system", "developer", "user", "assistant", "tool"}:
        raise ToolPolicyError("debate message role is invalid")
    return LLMChatMessage(
        role=role,
        content=_message_content_text(message.get("content")),
        tool_call_id=message.get("tool_call_id") if isinstance(message.get("tool_call_id"), str) else None,
        tool_calls=_tool_calls_from_assistant_message(message) if role == "assistant" else None,
        reasoning_content=message.get("reasoning_content") if isinstance(message.get("reasoning_content"), str) else None,
        reasoning_details=message.get("reasoning_details") if isinstance(message.get("reasoning_details"), list) else None,
    )


def _chat_tool(tool: FunctionTool) -> ChatTool:
    return ChatTool(
        function=ChatFunctionTool(
            description=tool.description,
            name=tool.name,
            parameters=tool.parameters,
            strict=tool.strict,
        )
    )


def _message_content_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [item["text"] for item in value if isinstance(item, dict) and isinstance(item.get("text"), str)]
        return "\n".join(parts) if parts else None
    return str(value)
