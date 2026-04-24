from __future__ import annotations

import hashlib
import json

from plap.responses.contracts import (
    CompactedResponseObject,
    CompactRequest,
    FunctionTool,
    InputItemsCompactionItem,
    InputItemsFunctionCallItem,
    InputItemsFunctionCallOutputItem,
    InputItemsMessageItem,
    InputItemsPage,
    InputItemsPageItem,
    InputItemsReasoningItem,
    InputTextContent,
    InputTokenCountResponse,
    InputTokensCountRequest,
    OutputTextContent,
    ReasoningTextContent,
    RequestInputItem,
    RequestMessageItem,
    ResponseCompactionItem,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseCreateRequest,
    ResponseDeleted,
    ResponseErrorEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionCallItem,
    ResponseFunctionCallOutputItem,
    ResponseInProgressEvent,
    ResponseMessageItem,
    ResponseObject,
    ResponseOutputItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseReasoningItem,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseStatus,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseUsage,
    ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallItem,
    ResponseWebSearchCallSearchingEvent,
    SummaryTextContent,
    WebSearchActionSearch,
    WebSearchTool,
)

BASE_CREATED_AT = 1_735_689_600


def _stable_id(prefix: str, seed: str) -> str:
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _request_seed(
    request: ResponseCreateRequest | CompactRequest | InputTokensCountRequest,
) -> str:
    return json.dumps(
        request.model_dump(mode="json", exclude_none=True), sort_keys=True
    )


def _extract_input_text(input_value: str | list[RequestInputItem] | None) -> str:
    if input_value is None:
        return ""
    if isinstance(input_value, str):
        return input_value

    chunks: list[str] = []
    for item in input_value:
        if isinstance(item, RequestMessageItem):
            if isinstance(item.content, str):
                chunks.append(item.content)
            else:
                chunks.extend(part.text for part in item.content)
    return " ".join(chunk for chunk in chunks if chunk)


def _count_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(stripped.split())


def _build_usage(input_text: str, output_text: str) -> ResponseUsage:
    input_tokens = _count_tokens(input_text)
    output_tokens = _count_tokens(output_text)
    return ResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details={"cached_tokens": 0},
        output_tokens=output_tokens,
        output_tokens_details={"reasoning_tokens": max(1, output_tokens // 2)},
        total_tokens=input_tokens + output_tokens,
    )


def build_stub_response(
    request: ResponseCreateRequest,
    *,
    response_id: str | None = None,
    status: ResponseStatus = "completed",
) -> ResponseObject:
    seed = _request_seed(request)
    input_text = _extract_input_text(request.input)
    message_text = (
        f"Stub response from plap for {request.model or 'stub-model'}."
        if status != "cancelled"
        else "Stub response cancelled."
    )

    items: list[ResponseOutputItem] = [
        ResponseReasoningItem(
            content=[
                ReasoningTextContent(
                    text="Stub reasoning trace.", type="reasoning_text"
                )
            ],
            id=_stable_id("rs", seed),
            status="completed" if status != "cancelled" else "incomplete",
            summary=[
                SummaryTextContent(text="Stub reasoning summary.", type="summary_text")
            ],
            type="reasoning",
        )
    ]

    if request.tools:
        for index, tool in enumerate(request.tools):
            if isinstance(tool, WebSearchTool):
                items.append(
                    ResponseWebSearchCallItem(
                        action=WebSearchActionSearch(
                            query="stub search",
                            queries=["stub search"],
                            type="search",
                        ),
                        id=_stable_id("ws", f"{seed}:{index}"),
                        status="completed",
                        type="web_search_call",
                    )
                )
            elif isinstance(tool, FunctionTool):
                call_id = _stable_id("call", f"{seed}:{tool.name}:{index}")
                items.append(
                    ResponseFunctionCallItem(
                        arguments="{}",
                        call_id=call_id,
                        id=_stable_id("fc", f"{seed}:{tool.name}:{index}"),
                        name=tool.name,
                        status="completed" if status != "cancelled" else "incomplete",
                        type="function_call",
                    )
                )
                items.append(
                    ResponseFunctionCallOutputItem(
                        call_id=call_id,
                        created_by="plap",
                        id=_stable_id("fco", f"{seed}:{tool.name}:{index}"),
                        output='{"ok":true}',
                        status="completed" if status != "cancelled" else "incomplete",
                        type="function_call_output",
                    )
                )

    items.append(
        ResponseMessageItem(
            content=[OutputTextContent(text=message_text, type="output_text")],
            id=_stable_id("msg", seed),
            phase="final_answer",
            role="assistant",
            status="completed" if status != "cancelled" else "incomplete",
            type="message",
        )
    )

    created_id = response_id or _stable_id("resp", seed)
    return ResponseObject(
        background=request.background,
        completed_at=BASE_CREATED_AT + 1
        if status in {"completed", "cancelled"}
        else None,
        conversation=request.conversation,
        created_at=BASE_CREATED_AT,
        id=created_id,
        instructions=request.instructions,
        max_output_tokens=request.max_output_tokens,
        max_tool_calls=request.max_tool_calls,
        metadata=request.metadata,
        model=request.model or "stub-model",
        output=items,
        parallel_tool_calls=request.parallel_tool_calls,
        previous_response_id=request.previous_response_id,
        prompt=request.prompt,
        prompt_cache_key=request.prompt_cache_key,
        prompt_cache_retention=request.prompt_cache_retention,
        reasoning=request.reasoning,
        safety_identifier=request.safety_identifier,
        service_tier=request.service_tier,
        status=status,
        temperature=request.temperature,
        text=request.text,
        tool_choice=request.tool_choice,
        tools=request.tools,
        top_logprobs=request.top_logprobs,
        top_p=request.top_p,
        truncation=request.truncation,
        usage=_build_usage(input_text, message_text),
        user=request.user,
    )


def build_deleted_response(response_id: str) -> ResponseDeleted:
    return ResponseDeleted(deleted=True, id=response_id)


def build_compacted_response(request: CompactRequest) -> CompactedResponseObject:
    seed = _request_seed(request)
    input_text = _extract_input_text(request.input)
    output = ResponseCompactionItem(
        created_by="plap",
        encrypted_content="stub-compacted-payload",
        id=_stable_id("cmp", seed),
        type="compaction",
    )
    return CompactedResponseObject(
        created_at=BASE_CREATED_AT,
        id=_stable_id("rcmp", seed),
        output=[output],
        usage=_build_usage(input_text, "stub compacted payload"),
    )


def build_input_token_count(
    request: InputTokensCountRequest,
) -> InputTokenCountResponse:
    input_text = _extract_input_text(request.input)
    if request.instructions:
        input_text = f"{request.instructions} {input_text}".strip()
    return InputTokenCountResponse(input_tokens=_count_tokens(input_text))


def build_input_items_page(response_id: str) -> InputItemsPage:
    seed = response_id
    data: list[InputItemsPageItem] = [
        InputItemsMessageItem(
            content="Stub input message.",
            id=_stable_id("imsg", seed),
            role="user",
            type="message",
        ),
        InputItemsFunctionCallItem(
            arguments="{}",
            call_id=_stable_id("call", seed),
            id=_stable_id("ifc", seed),
            name="lookup_context",
            status="completed",
            type="function_call",
        ),
        InputItemsFunctionCallOutputItem(
            call_id=_stable_id("call", seed),
            created_by="plap",
            id=_stable_id("ifco", seed),
            output=[InputTextContent(text="Stub tool output.", type="input_text")],
            status="completed",
            type="function_call_output",
        ),
        InputItemsReasoningItem(
            content=[
                ReasoningTextContent(
                    text="Stub reasoning trace.", type="reasoning_text"
                )
            ],
            id=_stable_id("irs", seed),
            status="completed",
            summary=[
                SummaryTextContent(text="Stub reasoning summary.", type="summary_text")
            ],
            type="reasoning",
        ),
        InputItemsCompactionItem(
            created_by="plap",
            encrypted_content="stub-compacted-payload",
            id=_stable_id("icmp", seed),
            type="compaction",
        ),
    ]
    return InputItemsPage(
        data=data,
        first_id=data[0].id,
        has_more=False,
        last_id=data[-1].id,
    )


def build_stream_events(response: ResponseObject) -> list[ResponseStreamEvent]:
    events: list[ResponseStreamEvent] = []
    sequence_number = 0

    def push(event: ResponseStreamEvent) -> None:
        nonlocal sequence_number
        sequence_number += 1
        payload = event.model_dump(mode="json")
        payload["sequence_number"] = sequence_number
        events.append(type(event).model_validate(payload))

    push(
        ResponseCreatedEvent(
            response=response, sequence_number=0, type="response.created"
        )
    )
    push(
        ResponseInProgressEvent(
            response=response.model_copy(
                update={"completed_at": None, "status": "in_progress"}
            ),
            sequence_number=0,
            type="response.in_progress",
        )
    )

    for output_index, item in enumerate(response.output):
        push(
            ResponseOutputItemAddedEvent(
                item=item,
                output_index=output_index,
                sequence_number=0,
                type="response.output_item.added",
            )
        )

        if isinstance(item, ResponseReasoningItem):
            for summary_index, summary_part in enumerate(item.summary):
                push(
                    ResponseReasoningSummaryPartAddedEvent(
                        item_id=item.id,
                        output_index=output_index,
                        part=summary_part,
                        sequence_number=0,
                        summary_index=summary_index,
                        type="response.reasoning_summary_part.added",
                    )
                )
                push(
                    ResponseReasoningSummaryTextDeltaEvent(
                        delta=summary_part.text,
                        item_id=item.id,
                        output_index=output_index,
                        sequence_number=0,
                        summary_index=summary_index,
                        type="response.reasoning_summary_text.delta",
                    )
                )
                push(
                    ResponseReasoningSummaryTextDoneEvent(
                        item_id=item.id,
                        output_index=output_index,
                        sequence_number=0,
                        summary_index=summary_index,
                        text=summary_part.text,
                        type="response.reasoning_summary_text.done",
                    )
                )
                push(
                    ResponseReasoningSummaryPartDoneEvent(
                        item_id=item.id,
                        output_index=output_index,
                        part=summary_part,
                        sequence_number=0,
                        summary_index=summary_index,
                        type="response.reasoning_summary_part.done",
                    )
                )

            for content_index, content_part in enumerate(item.content or []):
                push(
                    ResponseContentPartAddedEvent(
                        content_index=content_index,
                        item_id=item.id,
                        output_index=output_index,
                        part=content_part,
                        sequence_number=0,
                        type="response.content_part.added",
                    )
                )
                push(
                    ResponseReasoningTextDeltaEvent(
                        content_index=content_index,
                        delta=content_part.text,
                        item_id=item.id,
                        output_index=output_index,
                        sequence_number=0,
                        type="response.reasoning_text.delta",
                    )
                )
                push(
                    ResponseReasoningTextDoneEvent(
                        content_index=content_index,
                        item_id=item.id,
                        output_index=output_index,
                        sequence_number=0,
                        text=content_part.text,
                        type="response.reasoning_text.done",
                    )
                )
                push(
                    ResponseContentPartDoneEvent(
                        content_index=content_index,
                        item_id=item.id,
                        output_index=output_index,
                        part=content_part,
                        sequence_number=0,
                        type="response.content_part.done",
                    )
                )

        if isinstance(item, ResponseWebSearchCallItem):
            push(
                ResponseWebSearchCallInProgressEvent(
                    item_id=item.id,
                    output_index=output_index,
                    sequence_number=0,
                    type="response.web_search_call.in_progress",
                )
            )
            push(
                ResponseWebSearchCallSearchingEvent(
                    item_id=item.id,
                    output_index=output_index,
                    sequence_number=0,
                    type="response.web_search_call.searching",
                )
            )
            push(
                ResponseWebSearchCallCompletedEvent(
                    item_id=item.id,
                    output_index=output_index,
                    sequence_number=0,
                    type="response.web_search_call.completed",
                )
            )

        if isinstance(item, ResponseFunctionCallItem):
            push(
                ResponseFunctionCallArgumentsDeltaEvent(
                    delta=item.arguments,
                    item_id=item.id,
                    output_index=output_index,
                    sequence_number=0,
                    type="response.function_call_arguments.delta",
                )
            )
            push(
                ResponseFunctionCallArgumentsDoneEvent(
                    arguments=item.arguments,
                    item_id=item.id,
                    name=item.name,
                    output_index=output_index,
                    sequence_number=0,
                    type="response.function_call_arguments.done",
                )
            )

        if isinstance(item, ResponseMessageItem):
            for content_index, content_part in enumerate(item.content):
                push(
                    ResponseContentPartAddedEvent(
                        content_index=content_index,
                        item_id=item.id,
                        output_index=output_index,
                        part=content_part,
                        sequence_number=0,
                        type="response.content_part.added",
                    )
                )
                for annotation_index, annotation in enumerate(content_part.annotations):
                    push(
                        ResponseOutputTextAnnotationAddedEvent(
                            annotation=annotation,
                            annotation_index=annotation_index,
                            content_index=content_index,
                            item_id=item.id,
                            output_index=output_index,
                            sequence_number=0,
                            type="response.output_text.annotation.added",
                        )
                    )
                push(
                    ResponseTextDeltaEvent(
                        content_index=content_index,
                        delta=content_part.text,
                        item_id=item.id,
                        output_index=output_index,
                        sequence_number=0,
                        type="response.output_text.delta",
                    )
                )
                push(
                    ResponseTextDoneEvent(
                        content_index=content_index,
                        item_id=item.id,
                        output_index=output_index,
                        sequence_number=0,
                        text=content_part.text,
                        type="response.output_text.done",
                    )
                )
                push(
                    ResponseContentPartDoneEvent(
                        content_index=content_index,
                        item_id=item.id,
                        output_index=output_index,
                        part=content_part,
                        sequence_number=0,
                        type="response.content_part.done",
                    )
                )

        push(
            ResponseOutputItemDoneEvent(
                item=item,
                output_index=output_index,
                sequence_number=0,
                type="response.output_item.done",
            )
        )

    push(
        ResponseCompletedEvent(
            response=response, sequence_number=0, type="response.completed"
        )
    )
    return events


def build_error_event(message: str) -> ResponseErrorEvent:
    return ResponseErrorEvent(
        code="invalid_request_error",
        message=message,
        sequence_number=1,
        type="error",
    )
