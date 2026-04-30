from __future__ import annotations

import json
import re
import secrets
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Literal

import anyio

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
from plap.responses.contracts import (
    FunctionTool,
    OutputTextContent,
    ReasoningItem,
    ReasoningSummary,
    ResponseCompactionItem,
    ResponseCreateRequest,
    ResponseFunctionCallItem,
    ResponseFunctionCallOutputItem,
    ResponseMessageItem,
    ResponseStreamEvent,
    ResponseUsage,
    ResponseUsageInputTokensDetails,
    ResponseUsageOutputTokensDetails,
    TextFormatJSONObject,
    TextFormatJSONSchema,
    ToolChoiceFunction,
    WebSearchTool,
)
from plap.responses.ingest import (
    ChatMessageSpan,
    CompactionPayload,
    IngestionError,
    ReasoningPayload,
    SealedCallID,
    ingest_response_request,
)
from plap.responses.ingest.sealing import (
    content_hash,
    content_hash_prefix,
    seal_call_id,
    seal_compaction_payload,
    seal_reasoning_payload,
)
from plap.responses.io import ResponseEventIO
from plap.responses.reasoning import IReasoningSummarizer
from plap.responses.tokens import (
    estimate_citation_tokens,
    estimate_message_tokens,
)
from plap.responses.tools import (
    IToolCallPolicyResolver,
    IToolPolicyResolver,
    ToolCall,
    ToolPolicy,
    ToolPolicyError,
    canonical_tool_arguments,
)
from plap.responses.tools.compress import (
    COMPRESS_DEVELOPER_PROMPT,
    COMPRESS_TOOL_NAME,
    compress_policy,
    compress_tool,
)
from plap.responses.tools.mcp import IMCPToolProvider
from plap.responses.tools.web_search import web_search_policy
from plap.settings import RuntimeModelProfileConfig, Settings

SERVER_TOOL_NAMES = frozenset({COMPRESS_TOOL_NAME})
_CITATION_RE = re.compile(r"^~(\d+)(?:_(\d+))?$")
type _CompressionBailout = Literal["none", "soft", "hard"]
type _CompressionReminderLevel = Literal["none", "soft", "hard"]

SOFT_COMPRESSION_REMINDER = (
    "Context is getting long. If earlier cited conversation ranges can be safely "
    "replaced with focused summaries, call `compress` now. If no useful safe "
    "compression is possible, call `compress` with {\"ranges\": []}."
)
HARD_COMPRESSION_REMINDER = (
    "Context is at the compression limit. Before continuing, call `compress` if "
    "any earlier cited ranges can be safely summarized. If no useful safe "
    "compression is possible, call `compress` with {\"ranges\": []}."
)

MAIN_DEVELOPER_PROMPT_TEMPLATE = """You are {model_name}, a capable AI assistant.
Be accurate, direct, and helpful. Follow the user's instructions. Ask clarifying
questions when needed. Use tools when they help you answer better. Do not invent
facts, tool results, or citations. When you make a mistake, correct it plainly."""

def _runtime_model_profile(
    settings: Settings,
    request: ResponseCreateRequest,
) -> RuntimeModelProfileConfig:
    if request.model is None:
        raise IngestionError("model is required")
    profile = settings.runtime_model_profiles.get(request.model)
    if profile is None:
        raise IngestionError("unknown runtime model")
    return profile.for_service_tier(request.service_tier)


def _reasoning_summary_mode(request: ResponseCreateRequest) -> ReasoningSummary | None:
    if request.reasoning is None:
        return None
    return request.reasoning.summary or request.reasoning.generate_summary


def _context_token_count(
    spans: Sequence[ChatMessageSpan], *, include_citations: bool
) -> int:
    if not include_citations:
        return sum(span.token_count for span in spans)
    return sum(
        span.token_count + estimate_citation_tokens(span.citation) for span in spans
    )


def _compression_reminder(
    profile: RuntimeModelProfileConfig,
    token_count: int,
    bailout: _CompressionBailout,
) -> tuple[_CompressionReminderLevel, str | None]:
    if _hard_compression_budget_crossed(profile, token_count):
        if bailout != "hard":
            return "hard", HARD_COMPRESSION_REMINDER
        return "none", None
    if (
        profile.compression_soft_token_budget is not None
        and token_count >= profile.compression_soft_token_budget
        and bailout == "none"
    ):
        return "soft", SOFT_COMPRESSION_REMINDER
    return "none", None


def _hard_compression_budget_crossed(
    profile: RuntimeModelProfileConfig,
    token_count: int,
) -> bool:
    return (
        profile.compression_hard_token_budget is not None
        and token_count >= profile.compression_hard_token_budget
    )


async def prepare_tools(
    request: ResponseCreateRequest,
    resolver: IToolPolicyResolver,
    mcp_tool_provider: IMCPToolProvider | None = None,
) -> tuple[
    tuple[FunctionTool, ...],
    dict[str, ToolPolicy],
    dict[str, IMCPToolProvider],
]:
    client_tools = _client_tools(request.tools or ())

    server_tools: list[FunctionTool] = []
    server_executors: dict[str, IMCPToolProvider] = {}
    mcp_server_provider: IMCPToolProvider | None = None
    if _has_web_search(request.tools or ()):
        if mcp_tool_provider is None:
            raise ToolPolicyError("web_search requested but no MCP provider configured")
        mcp_server_provider = mcp_tool_provider
        server_tools.extend(await mcp_server_provider.tools())

    _reject_server_name_collisions(client_tools, server_tools)

    tools = [*client_tools, *server_tools]
    tool_policies = await resolver.resolve(client_tools)

    for tool in server_tools:
        tool_policies[tool.name] = web_search_policy(tool.name)
        if mcp_server_provider is None:
            raise RuntimeError("server tool executor was not configured")
        server_executors[tool.name] = mcp_server_provider

    return tuple(tools), tool_policies, server_executors


async def resolve_tool_calls(
    calls: Sequence[ChatToolCall],
    *,
    tools: Mapping[str, FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    resolver: IToolCallPolicyResolver,
) -> tuple[ToolPolicy, ...]:
    if not calls:
        return ()
    _validate_tool_call_batch(calls, tool_policies)

    resolved: list[ToolPolicy | None] = []
    client_calls: list[ToolCall] = []
    client_indexes: list[int] = []
    for call in calls:
        policy = tool_policies[call.name]
        if policy.source == "server":
            resolved.append(policy)
            continue

        tool = tools.get(call.name)
        if tool is None:
            raise ToolPolicyError(f"unknown client tool call: {call.name}")
        client_indexes.append(len(resolved))
        resolved.append(None)
        client_calls.append(
            ToolCall(
                tool=tool,
                policy=policy,
                arguments=call.arguments,
            )
        )

    if client_calls:
        client_policies = await resolver.resolve(client_calls)
        for index, policy in zip(client_indexes, client_policies, strict=True):
            resolved[index] = policy

    if any(policy is None for policy in resolved):
        raise RuntimeError("tool call policy resolution did not produce all outputs")
    return tuple(policy for policy in resolved if policy is not None)


def _apply_compression(
    main_context: list[ChatMessageSpan],
    arguments: str,
) -> tuple[list[ChatMessageSpan], bool]:
    try:
        payload = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ToolPolicyError("compress arguments must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ToolPolicyError("compress arguments must be an object")
    ranges = payload.get("ranges")
    if isinstance(ranges, str):
        try:
            ranges = json.loads(ranges)
        except json.JSONDecodeError as exc:
            raise ToolPolicyError("compress ranges must be valid JSON") from exc
    if not isinstance(ranges, list):
        raise ToolPolicyError("compress ranges are required")
    if not ranges:
        return main_context, False

    index_by_citation = {
        span.citation: index for index, span in enumerate(main_context)
    }
    parsed_ranges: list[tuple[int, int, str]] = []
    for item in ranges:
        if not isinstance(item, dict):
            raise ToolPolicyError("compress range must be an object")
        start = item.get("start")
        end = item.get("end")
        summary = item.get("summary")
        if not isinstance(start, str) or not isinstance(end, str):
            raise ToolPolicyError("compress range citations are required")
        if not isinstance(summary, str) or not summary.strip():
            raise ToolPolicyError("compress range summary is required")
        start = _normalize_citation(start)
        end = _normalize_citation(end)
        start_index = index_by_citation.get(start)
        end_index = index_by_citation.get(end)
        if start_index is None or end_index is None:
            raise ToolPolicyError("compress range citation is not visible")
        if start_index > end_index:
            raise ToolPolicyError("compress range start must not follow end")
        parsed_ranges.append((start_index, end_index, summary.strip()))

    parsed_ranges.sort(key=lambda item: (item[0], item[1]))
    previous_end = -1
    for start_index, end_index, _ in parsed_ranges:
        if start_index <= previous_end:
            raise ToolPolicyError("compress ranges must not overlap")
        previous_end = end_index

    compressed: list[ChatMessageSpan] = []
    cursor = 0
    for start_index, end_index, summary in parsed_ranges:
        compressed.extend(main_context[cursor:start_index])
        selected = tuple(main_context[start_index : end_index + 1])
        summary_message = {"role": "assistant", "content": summary}
        summary_token_count = estimate_message_tokens(summary_message)
        selected_token_count = sum(row.token_count for row in selected)
        if summary_token_count >= selected_token_count:
            raise ToolPolicyError("compress summary must reduce token count")
        compressed.append(
            ChatMessageSpan(
                start=selected[0].start,
                end=selected[-1].end,
                message=summary_message,
                token_count=summary_token_count,
                children=selected,
            )
        )
        cursor = end_index + 1
    compressed.extend(main_context[cursor:])
    return compressed, True


def _normalize_citation(value: str) -> str:
    if value.startswith("[") or value.endswith("]"):
        if not (value.startswith("[") and value.endswith("]")):
            raise ToolPolicyError("compress range citation is invalid")
        value = value[1:-1]

    match = _CITATION_RE.fullmatch(value)
    if match is None:
        raise ToolPolicyError("compress range citation is invalid")
    start = int(match.group(1))
    end = int(match.group(2) or start)
    if start > end:
        raise ToolPolicyError("compress range citation is invalid")
    suffix = f"_{end}" if match.group(2) is not None else ""
    return f"[~{start}{suffix}]"


def _chat_message_from_span(
    row: ChatMessageSpan, *, include_citation: bool
) -> ChatMessage:
    message = _chat_message_from_dict(row.message)
    content = message.content or ""
    if include_citation:
        content = f"{row.citation}\n{content}"
    return ChatMessage(
        role=message.role,
        content=content,
        name=message.name,
        refusal=message.refusal,
        tool_calls=message.tool_calls,
        tool_call_id=message.tool_call_id,
        reasoning_content=message.reasoning_content,
        reasoning_details=message.reasoning_details,
    )


def _chat_message_from_dict(message: Mapping[str, Any]) -> ChatMessage:
    role = message.get("role")
    if role not in {"system", "developer", "user", "assistant", "tool"}:
        raise IngestionError("chat message role is invalid")
    return ChatMessage(
        role=role,
        content=_message_content_text(message.get("content")),
        name=_string_or_none(message.get("name")),
        tool_call_id=_string_or_none(message.get("tool_call_id")),
        tool_calls=_chat_tool_calls(message.get("tool_calls")),
        reasoning_content=_string_or_none(message.get("reasoning_content")),
        reasoning_details=_reasoning_details(message.get("reasoning_details")),
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


def _chat_tool_choice(request: ResponseCreateRequest):
    choice = request.tool_choice
    if isinstance(choice, ToolChoiceFunction):
        return ChatToolChoiceFunction(name=choice.name)
    return choice


def _chat_response_format(request: ResponseCreateRequest) -> ChatResponseFormat | None:
    if request.text is None or request.text.format is None:
        return None
    text_format = request.text.format
    if isinstance(text_format, TextFormatJSONObject):
        return ChatResponseFormat(type="json_object")
    if isinstance(text_format, TextFormatJSONSchema):
        return ChatResponseFormat(
            description=text_format.description,
            name=text_format.name,
            schema=text_format.schema_,
            strict=text_format.strict,
            type="json_schema",
        )
    return ChatResponseFormat(type="text")


def _response_usage(usage: ChatUsage | None) -> ResponseUsage | None:
    if usage is None:
        return None
    return ResponseUsage(
        input_tokens=usage.input_tokens,
        input_tokens_details=ResponseUsageInputTokensDetails(
            cached_tokens=usage.cached_tokens or 0
        ),
        output_tokens=usage.output_tokens,
        output_tokens_details=ResponseUsageOutputTokensDetails(
            reasoning_tokens=usage.reasoning_tokens or 0
        ),
        total_tokens=usage.total_tokens,
    )


def _chat_tool_calls(value: object) -> list[ChatToolCall] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise IngestionError("chat message tool_calls is invalid")
    calls: list[ChatToolCall] = []
    for item in value:
        if not isinstance(item, dict):
            raise IngestionError("chat message tool_call is invalid")
        function = item.get("function")
        if not isinstance(function, dict):
            raise IngestionError("chat message tool_call function is invalid")
        call_id = item.get("id")
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(call_id, str) or not isinstance(name, str):
            raise IngestionError("chat message tool_call identity is invalid")
        if not isinstance(arguments, str):
            arguments = "{}"
        calls.append(ChatToolCall(id=call_id, name=name, arguments=arguments))
    return calls


def _message_content_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [
            item["text"]
            for item in value
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "\n".join(parts) if parts else None
    return str(value)


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _reasoning_details(value: object) -> list[dict[str, Any]] | None:
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return value
    return None


def _client_tools(tools: Sequence[object]) -> list[FunctionTool]:
    return [tool for tool in tools if isinstance(tool, FunctionTool)]


def _has_web_search(tools: Sequence[object]) -> bool:
    return any(isinstance(tool, WebSearchTool) for tool in tools)


def _reject_server_name_collisions(
    client_tools: Sequence[FunctionTool],
    server_tools: Sequence[FunctionTool],
) -> None:
    server_names = [tool.name for tool in server_tools]
    if len(set(server_names)) != len(server_names):
        raise ToolPolicyError("server tool names must be unique")
    server_tool_names = set(server_names) | SERVER_TOOL_NAMES
    for tool in client_tools:
        if tool.name in server_tool_names:
            raise ToolPolicyError(f"function tool name is reserved: {tool.name}")


def _validate_tool_call_batch(
    calls: Sequence[ChatToolCall],
    tool_policies: Mapping[str, ToolPolicy],
) -> None:
    for call in calls:
        if call.name not in tool_policies:
            raise ToolPolicyError(f"unknown tool call: {call.name}")
    if any(call.name == COMPRESS_TOOL_NAME for call in calls) and len(calls) != 1:
        raise ToolPolicyError("compress must be called alone")


async def _run_response(
    out: ResponseEventIO,
    request: ResponseCreateRequest,
    *,
    settings: Settings,
    sealing_keyring: SealingKeyring,
    tool_policy_resolver: IToolPolicyResolver,
    tool_call_policy_resolver: IToolCallPolicyResolver,
    chat_completion_client: IChatCompletionClient,
    mcp_tool_provider: IMCPToolProvider | None,
) -> None:
    profile = _runtime_model_profile(settings, request)
    ingested = await ingest_response_request(
        request,
        keyring=sealing_keyring,
        transcript_token_budget=profile.transcript_token_budget,
    )

    main_context = list(ingested.main_context)
    main_context_temp = list(ingested.main_context_temp)
    main_transcript = list(ingested.main_transcript)
    reviewer = list(ingested.reviewer)
    arbitrator = list(ingested.arbitrator)
    cursors = dict(ingested.cursors)
    continuation_side = ingested.continuation_side
    in_temp_debate = ingested.in_temp_debate

    _ = (
        main_transcript,
        main_context_temp,
        reviewer,
        arbitrator,
        cursors,
        continuation_side,
        in_temp_debate,
    )

    base_tools, base_tool_policies, base_server_executors = await prepare_tools(
        request,
        tool_policy_resolver,
        mcp_tool_provider,
    )

    await out.created()
    await out.in_progress()

    compression_rounds = 0
    compression_bailout: _CompressionBailout = "none"
    while True:
        effective_tools = [*base_tools]
        effective_tool_policies = dict(base_tool_policies)
        effective_server_executors = dict(base_server_executors)

        if in_temp_debate:
            effective_tools = [
                tool
                for tool in effective_tools
                if effective_tool_policies[tool.name].effect_class == "safe"
            ]
            effective_tool_policies = {
                name: policy
                for name, policy in effective_tool_policies.items()
                if policy.effect_class == "safe"
            }
            effective_server_executors = {
                name: executor
                for name, executor in effective_server_executors.items()
                if name in effective_tool_policies
            }
        elif (
            compression_rounds < profile.compression_max_rounds
        ):
            effective_tools.append(compress_tool())
            effective_tool_policies[COMPRESS_TOOL_NAME] = compress_policy()

        developer_prompt_parts = [
            MAIN_DEVELOPER_PROMPT_TEMPLATE.format(model_name=profile.display_name)
        ]
        if COMPRESS_TOOL_NAME in effective_tool_policies:
            developer_prompt_parts.append(COMPRESS_DEVELOPER_PROMPT)
        if request.instructions:
            developer_prompt_parts.append(f"User instructions:\n{request.instructions}")

        messages: list[ChatMessage] = [
            ChatMessage(
                role="developer",
                content="\n\n".join(developer_prompt_parts),
            )
        ]
        effective_main_context = [*main_context, *main_context_temp]
        include_citations = COMPRESS_TOOL_NAME in effective_tool_policies
        messages.extend(
            _chat_message_from_span(row, include_citation=include_citations)
            for row in effective_main_context
        )
        token_count = _context_token_count(
            effective_main_context,
            include_citations=include_citations,
        )
        compression_reminder_level: _CompressionReminderLevel = "none"
        compression_reminder = None
        if COMPRESS_TOOL_NAME in effective_tool_policies:
            compression_reminder_level, compression_reminder = _compression_reminder(
                profile,
                token_count,
                compression_bailout,
            )
        if compression_reminder is not None:
            messages.append(ChatMessage(role="user", content=compression_reminder))

        tool_choice = _chat_tool_choice(request)
        forced_compression = False
        if compression_reminder_level == "hard":
            tool_choice = ChatToolChoiceFunction(name=COMPRESS_TOOL_NAME)
            forced_compression = True

        model_request = ChatCompletionRequest(
            model=profile.main_model,
            messages=messages,
            tools=[_chat_tool(tool) for tool in effective_tools],
            tool_choice=tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
            response_format=_chat_response_format(request),
            max_completion_tokens=request.max_output_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_logprobs=request.top_logprobs,
            reasoning_effort=request.reasoning.effort if request.reasoning else None,
            prompt_cache_key=request.prompt_cache_key,
            user=request.user,
        )

        result = await chat_completion_client.complete(model_request)
        tool_calls = result.message.tool_calls or []
        if forced_compression and not any(
            call.name == COMPRESS_TOOL_NAME for call in tool_calls
        ):
            raise ToolPolicyError("hard compression budget requires compress")

        if tool_calls:
            resolved_policies = await resolve_tool_calls(
                tool_calls,
                tools={tool.name: tool for tool in effective_tools},
                tool_policies=effective_tool_policies,
                resolver=tool_call_policy_resolver,
            )
            if len(tool_calls) == 1 and tool_calls[0].name == COMPRESS_TOOL_NAME:
                main_context, compressed = _apply_compression(
                    main_context,
                    tool_calls[0].arguments,
                )
                if compressed:
                    compression_bailout = "none"
                    compaction_payload = CompactionPayload(
                        active=tuple(main_context),
                        cursors=cursors,
                    )
                    await out.output(
                        ResponseCompactionItem(
                            created_by="assistant",
                            encrypted_content=seal_compaction_payload(
                                compaction_payload,
                                keyring=sealing_keyring,
                            ),
                            id=f"cmp_{secrets.token_urlsafe(18)}",
                            type="compaction",
                        )
                    )
                elif compression_reminder_level == "hard":
                    compression_bailout = "hard"
                elif compression_reminder_level == "soft":
                    compression_bailout = "soft"
                compression_rounds += 1
                continue

            if compression_reminder_level == "soft":
                compression_bailout = "soft"

            server_outputs: dict[int, str] = {}
            client_call_indexes: list[int] = []
            for index, (call, policy) in enumerate(
                zip(tool_calls, resolved_policies, strict=True)
            ):
                if policy.source == "server":
                    executor = effective_server_executors.get(call.name)
                    if executor is None:
                        raise ToolPolicyError("server tool executor is not configured")
                    server_outputs[index] = await executor.call_tool(
                        call.name,
                        canonical_tool_arguments(call.arguments),
                    )
                    continue
                if policy.effect_class not in {"safe", "visible"}:
                    raise ToolPolicyError("tool call requires unsupported policy path")
                client_call_indexes.append(index)

        public_assistant_message: dict[str, Any] = {"role": "assistant"}
        if result.message.content is not None:
            public_assistant_message["content"] = result.message.content
        elif (
            result.message.reasoning_content
            or result.message.reasoning_details
            or result.message.tool_calls
        ):
            public_assistant_message["content"] = ""
        assistant_hash = content_hash(public_assistant_message)

        if result.message.content is not None or (
            result.message.reasoning_content
            or result.message.reasoning_details
            or result.message.tool_calls
        ):
            message_item = ResponseMessageItem(
                content=[
                    OutputTextContent(
                        text=result.message.content or "",
                        type="output_text",
                    )
                ],
                id=f"msg_{secrets.token_urlsafe(18)}",
                role="assistant",
                status="completed",
                type="message",
            )
            await out.output(message_item)

        if result.message.reasoning_content or result.message.reasoning_details:
            reasoning_message: dict[str, Any] = {"content_hash": assistant_hash}
            if result.message.reasoning_content is not None:
                reasoning_message["reasoning_content"] = (
                    result.message.reasoning_content
                )
            if result.message.reasoning_details is not None:
                reasoning_message["reasoning_details"] = (
                    result.message.reasoning_details
                )
            reasoning_payload = ReasoningPayload(
                side="main",
                temp=False,
                messages=(reasoning_message,),
            )
            reasoning_item = ReasoningItem(
                encrypted_content=seal_reasoning_payload(
                    reasoning_payload,
                    keyring=sealing_keyring,
                ),
                id=f"rs_{secrets.token_urlsafe(18)}",
                status="completed",
                summary=[],
                type="reasoning",
            )
            await out.output(
                reasoning_item,
                reasoning_side=reasoning_payload.side,
                reasoning_messages=reasoning_payload.messages,
            )

        if result.message.tool_calls:
            function_call_items: list[ResponseFunctionCallItem] = []
            function_call_ids: dict[int, str] = {}
            assistant_tool_calls: list[dict[str, Any]] = []
            for index, call in enumerate(result.message.tool_calls):
                sealed_call_id = seal_call_id(
                    SealedCallID(
                        side="main",
                        content_hash_prefix=content_hash_prefix(assistant_hash),
                        tool_call_index=index,
                        upstream_tool_call_id=call.id,
                    ),
                    keyring=sealing_keyring,
                )
                function_call_ids[index] = sealed_call_id
                assistant_tool_calls.append(
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                    }
                )
                function_call_items.append(
                    ResponseFunctionCallItem(
                        arguments=call.arguments,
                        call_id=sealed_call_id,
                        id=f"fc_{secrets.token_urlsafe(18)}",
                        name=call.name,
                        status="completed",
                        type="function_call",
                    )
                )
            for function_call_item in function_call_items:
                await out.output(function_call_item)

            function_output_items: list[ResponseFunctionCallOutputItem] = []
            for index, output in server_outputs.items():
                function_output_items.append(
                    ResponseFunctionCallOutputItem(
                        call_id=function_call_ids[index],
                        created_by="server",
                        id=f"fco_{secrets.token_urlsafe(18)}",
                        output=output,
                        status="completed",
                        type="function_call_output",
                    )
                )
            for function_output_item in function_output_items:
                await out.output(function_output_item)
        else:
            assistant_tool_calls = []
            server_outputs = {}
            client_call_indexes = []

        if result.message.content is not None or (
            result.message.reasoning_content
            or result.message.reasoning_details
            or result.message.tool_calls
        ):
            next_ordinal = cursors["m"]
            cursors["m"] += 1
            assistant_context_message = dict(public_assistant_message)
            if assistant_tool_calls:
                assistant_context_message["tool_calls"] = assistant_tool_calls
            main_context.append(
                ChatMessageSpan(
                    start=next_ordinal,
                    end=next_ordinal,
                    message=assistant_context_message,
                    content_hash=assistant_hash,
                    token_count=estimate_message_tokens(assistant_context_message),
                )
            )

        for index, output in server_outputs.items():
            next_ordinal = cursors["m"]
            cursors["m"] += 1
            tool_message = {
                "role": "tool",
                "tool_call_id": tool_calls[index].id,
                "content": output,
            }
            main_context.append(
                ChatMessageSpan(
                    start=next_ordinal,
                    end=next_ordinal,
                    message=tool_message,
                    token_count=estimate_message_tokens(tool_message),
                )
            )

        if server_outputs and not client_call_indexes:
            continue

        await out.completed(
            service_tier=result.service_tier,
            usage=_response_usage(result.usage),
        )
        return


async def stream_response_events(
    request: ResponseCreateRequest,
    *,
    settings: Settings,
    sealing_keyring: SealingKeyring,
    tool_policy_resolver: IToolPolicyResolver,
    tool_call_policy_resolver: IToolCallPolicyResolver,
    chat_completion_client: IChatCompletionClient,
    reasoning_summarizer: IReasoningSummarizer,
    mcp_tool_provider: IMCPToolProvider | None = None,
) -> AsyncIterator[ResponseStreamEvent]:
    send, receive = anyio.create_memory_object_stream[ResponseStreamEvent](16)
    producer_error: Exception | None = None

    async def produce() -> None:
        nonlocal producer_error
        async with send:
            try:
                async with anyio.create_task_group() as task_group:
                    out = ResponseEventIO(
                        request=request,
                        reasoning_summarizer=reasoning_summarizer,
                        reasoning_summarizer_model=_runtime_model_profile(
                            settings,
                            request,
                        ).reasoning_summarizer_model,
                        reasoning_summary_mode=_reasoning_summary_mode(request),
                        send=send,
                    )
                    out.start(task_group)
                    try:
                        await _run_response(
                            out,
                            request,
                            settings=settings,
                            sealing_keyring=sealing_keyring,
                            tool_policy_resolver=tool_policy_resolver,
                            tool_call_policy_resolver=tool_call_policy_resolver,
                            chat_completion_client=chat_completion_client,
                            mcp_tool_provider=mcp_tool_provider,
                        )
                    finally:
                        await out.aclose()
            except Exception as exc:
                if isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
                    producer_error = exc.exceptions[0]
                else:
                    producer_error = exc

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(produce)
        async with receive:
            async for event in receive:
                yield event
    if producer_error is not None:
        raise producer_error
