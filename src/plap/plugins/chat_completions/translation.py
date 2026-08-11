from __future__ import annotations

from collections.abc import AsyncIterator

from plap.plugins.chat_completions.contracts import (
    ChatAssistantMessage,
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionCreateRequest,
    ChatCompletionLogprobs,
    ChatCompletionMessage,
    ChatCompletionTokenLogprob,
    ChatCompletionTopLogprob,
    ChatCompletionUsage,
    ChatCompletionUsageInputDetails,
    ChatCompletionUsageOutputDetails,
    ChatDeveloperMessage,
    ChatFileContentPart,
    ChatFinishReason,
    ChatFunctionTool,
    ChatImageContentPart,
    ChatReasoningDetail,
    ChatReasoningEncrypted,
    ChatReasoningSummary,
    ChatReasoningText,
    ChatResponseFormatJSONObject,
    ChatResponseFormatJSONSchema,
    ChatResponseFormatText,
    ChatSystemMessage,
    ChatTextContentPart,
    ChatToolCall,
    ChatToolCallDelta,
    ChatToolCallDeltaFunction,
    ChatToolChoiceFunction,
    ChatToolMessage,
    ChatUserMessage,
)
from plap.responses.contracts import (
    FunctionTool,
    InputFileContent,
    InputImageContent,
    InputTextContent,
    OutputRefusalContent,
    OutputTextContent,
    OutputTextLogprob,
    OutputTextLogprobTopLogprob,
    ReasoningConfig,
    ReasoningTextContent,
    RequestFunctionCallItem,
    RequestFunctionCallOutputItem,
    RequestInputItem,
    RequestMessageItem,
    RequestReasoningItem,
    ResponseCompletedEvent,
    ResponseCreateRequest,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallItem,
    ResponseIncompleteEvent,
    ResponseMessageItem,
    ResponseObject,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningItem,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningTextDoneEvent,
    ResponseRefusalDeltaEvent,
    ResponseStreamEvent,
    ResponseTextConfig,
    ResponseTextDeltaEvent,
    ResponseTextEventLogprob,
    ResponseTextEventLogprobTopLogprob,
    ResponseUsage,
    StreamOptions,
    SummaryTextContent,
    TextFormatJSONObject,
    TextFormatJSONSchema,
    TextFormatText,
    ToolChoiceFunction,
)


def _text_parts(parts: list[ChatTextContentPart]) -> list[InputTextContent]:
    return [InputTextContent(text=part.text, type="input_text") for part in parts]


def _text_content(content: str | list[ChatTextContentPart]) -> str | list[InputTextContent]:
    if isinstance(content, str):
        return content
    return _text_parts(content)


def _user_content_part(part):
    if isinstance(part, ChatTextContentPart):
        return InputTextContent(text=part.text, type="input_text")
    if isinstance(part, ChatImageContentPart):
        return InputImageContent(
            detail=part.image_url.detail,
            image_url=part.image_url.url,
            type="input_image",
        )
    if isinstance(part, ChatFileContentPart):
        return InputFileContent(
            file_data=part.file.file_data,
            file_id=part.file.file_id,
            filename=part.file.filename,
            type="input_file",
        )
    raise TypeError(f"unsupported chat content part: {type(part).__name__}")


def _user_content(content):
    if isinstance(content, str):
        return content
    return [_user_content_part(part) for part in content]


def _assistant_content(message: ChatAssistantMessage):
    if message.refusal is None:
        return message.content
    parts: list[OutputTextContent | OutputRefusalContent] = []
    if message.content is not None:
        parts.append(OutputTextContent(annotations=[], text=message.content, type="output_text"))
    parts.append(OutputRefusalContent(refusal=message.refusal, type="refusal"))
    return parts


def _reasoning_items(details: list[ChatReasoningDetail]) -> list[RequestReasoningItem]:
    items: list[RequestReasoningItem] = []
    summary: list[SummaryTextContent] = []
    content: list[ReasoningTextContent] = []
    for detail in details:
        if isinstance(detail, ChatReasoningSummary):
            summary.append(SummaryTextContent(text=detail.summary, type="summary_text"))
            continue
        if isinstance(detail, ChatReasoningText):
            if detail.text is not None:
                content.append(ReasoningTextContent(text=detail.text, type="reasoning_text"))
            continue
        if not isinstance(detail, ChatReasoningEncrypted):  # pragma: no cover - narrowed by the contract
            raise TypeError(f"unsupported reasoning detail: {type(detail).__name__}")
        items.append(
            RequestReasoningItem(
                content=list(content) or None,
                encrypted_content=detail.data,
                id=detail.id,
                summary=list(summary),
                type="reasoning",
            )
        )
        summary.clear()
        content.clear()
    # Orphan summary/text details are untrusted metadata. The visible assistant
    # message below remains usable as fabricated history when encrypted state was dropped.
    return items


def _function_call_item(call: ChatToolCall) -> RequestFunctionCallItem:
    return RequestFunctionCallItem(
        arguments=call.function.arguments,
        call_id=call.id,
        name=call.function.name,
        type="function_call",
    )


def _assistant_items(message: ChatAssistantMessage) -> list[RequestInputItem]:
    reasoning_items = _reasoning_items(message.reasoning_details)
    items: list[RequestInputItem] = list(reasoning_items)
    content = _assistant_content(message)
    if content is not None or not reasoning_items:
        items.append(
            RequestMessageItem(
                content="" if content is None else content,
                role="assistant",
                type="message",
            )
        )
    items.extend(_function_call_item(call) for call in message.tool_calls)
    return items


def _message_items(message) -> list[RequestInputItem]:
    if isinstance(message, ChatSystemMessage | ChatDeveloperMessage):
        return [RequestMessageItem(content=_text_content(message.content), role=message.role, type="message")]
    if isinstance(message, ChatUserMessage):
        return [RequestMessageItem(content=_user_content(message.content), role="user", type="message")]
    if isinstance(message, ChatAssistantMessage):
        return _assistant_items(message)
    if isinstance(message, ChatToolMessage):
        return [
            RequestFunctionCallOutputItem(
                call_id=message.tool_call_id,
                output=_text_content(message.content),
                type="function_call_output",
            )
        ]
    raise TypeError(f"unsupported chat message: {type(message).__name__}")


def _input_items(request: ChatCompletionCreateRequest) -> list[RequestInputItem]:
    items: list[RequestInputItem] = []
    for message in request.messages:
        items.extend(_message_items(message))
    return items


def _function_tool(tool: ChatFunctionTool) -> FunctionTool:
    return FunctionTool(
        description=tool.function.description,
        name=tool.function.name,
        parameters=tool.function.parameters,
        strict=tool.function.strict,
        type="function",
    )


def _tool_choice(request: ChatCompletionCreateRequest):
    choice = request.tool_choice
    if choice is None or isinstance(choice, str):
        return choice
    if isinstance(choice, ChatToolChoiceFunction):
        return ToolChoiceFunction(name=choice.function.name, type="function")
    raise TypeError(f"unsupported chat tool choice: {type(choice).__name__}")


def _text_config(request: ChatCompletionCreateRequest) -> ResponseTextConfig | None:
    response_format = request.response_format
    if response_format is None:
        return None
    if isinstance(response_format, ChatResponseFormatText):
        format_ = TextFormatText(type="text")
    elif isinstance(response_format, ChatResponseFormatJSONObject):
        format_ = TextFormatJSONObject(type="json_object")
    elif isinstance(response_format, ChatResponseFormatJSONSchema):
        schema = response_format.json_schema
        format_ = TextFormatJSONSchema(
            description=schema.description,
            name=schema.name,
            schema=schema.schema_,
            strict=schema.strict,
            type="json_schema",
        )
    else:  # pragma: no cover - narrowed by the contract
        raise TypeError(f"unsupported chat response format: {type(response_format).__name__}")
    return ResponseTextConfig(format=format_)


def _reasoning_config(request: ChatCompletionCreateRequest) -> ReasoningConfig | None:
    if request.reasoning is not None:
        return ReasoningConfig(
            effort=request.reasoning.effort or request.reasoning_effort,
            summary=request.reasoning.summary,
        )
    if request.reasoning_effort is not None:
        return ReasoningConfig(effort=request.reasoning_effort)
    return None


def to_response_request(request: ChatCompletionCreateRequest) -> ResponseCreateRequest:
    return ResponseCreateRequest(
        include=["reasoning.encrypted_content"],
        input=_input_items(request),
        max_output_tokens=request.max_completion_tokens,
        metadata=request.metadata,
        model=request.model,
        parallel_tool_calls=request.parallel_tool_calls,
        prompt_cache_key=request.prompt_cache_key,
        reasoning=_reasoning_config(request),
        safety_identifier=request.safety_identifier,
        service_tier=request.service_tier,
        store=False if request.store is None else request.store,
        stream=request.stream,
        stream_options=StreamOptions(include_obfuscation=False) if request.stream else None,
        temperature=request.temperature,
        text=_text_config(request),
        tool_choice=_tool_choice(request),
        tools=[_function_tool(tool) for tool in request.tools],
        top_logprobs=0 if request.logprobs is True and request.top_logprobs is None else request.top_logprobs,
        top_p=request.top_p,
        user=request.user,
    )


def _usage(usage: ResponseUsage | None) -> ChatCompletionUsage | None:
    if usage is None:
        return None
    return ChatCompletionUsage(
        completion_tokens=usage.output_tokens,
        prompt_tokens=usage.input_tokens,
        total_tokens=usage.total_tokens,
        completion_tokens_details=ChatCompletionUsageOutputDetails(
            reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
        ),
        prompt_tokens_details=ChatCompletionUsageInputDetails(
            cached_tokens=usage.input_tokens_details.cached_tokens,
        ),
    )


def _encrypted_detail(item: ResponseReasoningItem, *, index: int) -> ChatReasoningEncrypted:
    if item.encrypted_content is None:
        raise ValueError(f"reasoning item {item.id!r} has no encrypted content")
    return ChatReasoningEncrypted(
        data=item.encrypted_content,
        id=item.id,
        index=index,
        type="reasoning.encrypted",
    )


def _reasoning_details(item: ResponseReasoningItem, *, start_index: int) -> list[ChatReasoningDetail]:
    details: list[ChatReasoningDetail] = []
    for summary_index, part in enumerate(item.summary):
        details.append(
            ChatReasoningSummary(
                id=f"{item.id}:summary:{summary_index}",
                index=start_index + len(details),
                summary=part.text,
                type="reasoning.summary",
            )
        )
    for content_index, part in enumerate(item.content or []):
        details.append(
            ChatReasoningText(
                id=f"{item.id}:text:{content_index}",
                index=start_index + len(details),
                text=part.text,
                type="reasoning.text",
            )
        )
    details.append(_encrypted_detail(item, index=start_index + len(details)))
    return details


def _top_logprob(value: OutputTextLogprobTopLogprob) -> ChatCompletionTopLogprob:
    return ChatCompletionTopLogprob(
        bytes=list(value.bytes),
        logprob=value.logprob,
        token=value.token,
    )


def _token_logprob(value: OutputTextLogprob) -> ChatCompletionTokenLogprob:
    return ChatCompletionTokenLogprob(
        bytes=list(value.bytes),
        logprob=value.logprob,
        token=value.token,
        top_logprobs=[_top_logprob(item) for item in value.top_logprobs],
    )


def _completion_logprobs(response: ResponseObject) -> ChatCompletionLogprobs | None:
    values: list[ChatCompletionTokenLogprob] = []
    present = False
    for item in response.output:
        if not isinstance(item, ResponseMessageItem):
            continue
        for part in item.content:
            if not isinstance(part, OutputTextContent) or part.logprobs is None:
                continue
            present = True
            values.extend(_token_logprob(value) for value in part.logprobs)
    if not present:
        return None
    return ChatCompletionLogprobs(content=values, refusal=None)


def _stream_top_logprob(value: ResponseTextEventLogprobTopLogprob) -> ChatCompletionTopLogprob:
    if value.logprob is None or value.token is None:
        raise ValueError("streamed top logprob is missing token or logprob")
    return ChatCompletionTopLogprob(
        bytes=list(value.token.encode()),
        logprob=value.logprob,
        token=value.token,
    )


def _stream_token_logprob(value: ResponseTextEventLogprob) -> ChatCompletionTokenLogprob:
    return ChatCompletionTokenLogprob(
        bytes=list(value.token.encode()),
        logprob=value.logprob,
        token=value.token,
        top_logprobs=[_stream_top_logprob(item) for item in value.top_logprobs or []],
    )


def _stream_logprobs(values: list[ResponseTextEventLogprob]) -> ChatCompletionLogprobs | None:
    if not values:
        return None
    return ChatCompletionLogprobs(
        content=[_stream_token_logprob(value) for value in values],
        refusal=None,
    )


def _tool_call(item: ResponseFunctionCallItem) -> ChatToolCall:
    return ChatToolCall(
        function={"arguments": item.arguments, "name": item.name},
        id=item.call_id,
        type="function",
    )


def _completion_message(response: ResponseObject) -> ChatCompletionMessage:
    content_parts: list[str] = []
    refusal_parts: list[str] = []
    reasoning_details: list[ChatReasoningDetail] = []
    tool_calls: list[ChatToolCall] = []
    saw_content = False
    saw_refusal = False
    for item in response.output:
        if isinstance(item, ResponseReasoningItem):
            reasoning_details.extend(_reasoning_details(item, start_index=len(reasoning_details)))
            continue
        if isinstance(item, ResponseFunctionCallItem):
            tool_calls.append(_tool_call(item))
            continue
        if not isinstance(item, ResponseMessageItem):
            continue
        for part in item.content:
            if isinstance(part, OutputTextContent):
                saw_content = True
                content_parts.append(part.text)
            elif isinstance(part, OutputRefusalContent):
                saw_refusal = True
                refusal_parts.append(part.refusal)
    return ChatCompletionMessage(
        content="".join(content_parts) if saw_content else None,
        reasoning_details=reasoning_details or None,
        refusal="".join(refusal_parts) if saw_refusal else None,
        role="assistant",
        tool_calls=tool_calls or None,
    )


def _finish_reason(response: ResponseObject) -> ChatFinishReason:
    if response.status == "incomplete":
        reason = None if response.incomplete_details is None else response.incomplete_details.reason
        return "content_filter" if reason == "content_filter" else "length"
    if any(isinstance(item, ResponseFunctionCallItem) for item in response.output):
        return "tool_calls"
    if response.status != "completed":
        raise ValueError(f"cannot project response with status {response.status!r} as a chat completion")
    return "stop"


def to_chat_completion(response: ResponseObject) -> ChatCompletion:
    if response.model is None:
        raise ValueError("response model is missing")
    return ChatCompletion(
        choices=[
            ChatCompletionChoice(
                finish_reason=_finish_reason(response),
                index=0,
                logprobs=_completion_logprobs(response),
                message=_completion_message(response),
            )
        ],
        created=int(response.created_at),
        id=response.id,
        model=response.model,
        object="chat.completion",
        service_tier=response.service_tier,
        usage=_usage(response.usage),
    )


def _chunk(
    response: ResponseObject,
    *,
    choices: list[ChatCompletionChunkChoice],
    usage: ChatCompletionUsage | None = None,
) -> dict[str, object]:
    if response.model is None:
        raise ValueError("response model is missing")
    return ChatCompletionChunk(
        choices=choices,
        created=int(response.created_at),
        id=response.id,
        model=response.model,
        object="chat.completion.chunk",
        service_tier=response.service_tier,
        usage=usage,
    ).model_dump(mode="json", exclude_none=True)


def _delta_chunk(
    response: ResponseObject,
    delta: ChatCompletionChunkDelta,
    *,
    logprobs: ChatCompletionLogprobs | None = None,
) -> dict[str, object]:
    return _chunk(
        response,
        choices=[
            ChatCompletionChunkChoice(
                delta=delta,
                finish_reason=None,
                index=0,
                logprobs=logprobs,
            )
        ],
    )


def _terminal_chunk(response: ResponseObject) -> dict[str, object]:
    return _chunk(
        response,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChunkDelta(),
                finish_reason=_finish_reason(response),
                index=0,
                logprobs=None,
            )
        ],
    )


def _usage_chunk(response: ResponseObject) -> dict[str, object]:
    return _chunk(response, choices=[], usage=_usage(response.usage))


def _stream_error(*, code: str | None, message: str) -> dict[str, object]:
    return {
        "error": {
            "code": code or "server_error",
            "message": message,
            "param": None,
            "type": "server_error",
        }
    }


async def chat_completion_stream(
    events: AsyncIterator[ResponseStreamEvent],
    *,
    include_usage: bool,
) -> AsyncIterator[dict[str, object]]:
    response: ResponseObject | None = None
    tool_indices: dict[str, int] = {}
    reasoning_index = 0
    async for event in events:
        if response is None and hasattr(event, "response"):
            response = event.response
        if response is None:
            continue
        if event.type == "response.created":
            yield _delta_chunk(response, ChatCompletionChunkDelta(role="assistant"))
            continue
        if isinstance(event, ResponseOutputItemDoneEvent) and isinstance(event.item, ResponseReasoningItem):
            detail = _encrypted_detail(event.item, index=reasoning_index)
            reasoning_index += 1
            yield _delta_chunk(response, ChatCompletionChunkDelta(reasoning_details=[detail]))
            continue
        if isinstance(event, ResponseReasoningSummaryPartDoneEvent):
            detail = ChatReasoningSummary(
                id=f"{event.item_id}:summary:{event.summary_index}",
                index=reasoning_index,
                summary=event.part.text,
                type="reasoning.summary",
            )
            reasoning_index += 1
            yield _delta_chunk(response, ChatCompletionChunkDelta(reasoning_details=[detail]))
            continue
        if isinstance(event, ResponseReasoningTextDoneEvent):
            detail = ChatReasoningText(
                id=f"{event.item_id}:text:{event.content_index}",
                index=reasoning_index,
                text=event.text,
                type="reasoning.text",
            )
            reasoning_index += 1
            yield _delta_chunk(response, ChatCompletionChunkDelta(reasoning_details=[detail]))
            continue
        if isinstance(event, ResponseTextDeltaEvent):
            yield _delta_chunk(
                response,
                ChatCompletionChunkDelta(content=event.delta),
                logprobs=_stream_logprobs(event.logprobs),
            )
            continue
        if isinstance(event, ResponseRefusalDeltaEvent):
            yield _delta_chunk(response, ChatCompletionChunkDelta(refusal=event.delta))
            continue
        if isinstance(event, ResponseOutputItemAddedEvent) and isinstance(event.item, ResponseFunctionCallItem):
            tool_index = len(tool_indices)
            tool_indices[event.item.id] = tool_index
            yield _delta_chunk(
                response,
                ChatCompletionChunkDelta(
                    tool_calls=[
                        ChatToolCallDelta(
                            function=ChatToolCallDeltaFunction(arguments="", name=event.item.name),
                            id=event.item.call_id,
                            index=tool_index,
                            type="function",
                        )
                    ]
                ),
            )
            continue
        if isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
            tool_index = tool_indices.get(event.item_id)
            if tool_index is None:
                raise ValueError(f"function call delta has no added item: {event.item_id}")
            yield _delta_chunk(
                response,
                ChatCompletionChunkDelta(
                    tool_calls=[
                        ChatToolCallDelta(
                            function=ChatToolCallDeltaFunction(arguments=event.delta),
                            index=tool_index,
                        )
                    ]
                ),
            )
            continue
        if isinstance(event, ResponseFailedEvent):
            error = event.response.error
            yield _stream_error(
                code=None if error is None else error.code,
                message="The model failed to generate a response." if error is None else error.message,
            )
            return
        if isinstance(event, ResponseErrorEvent):
            yield _stream_error(code=event.code, message=event.message)
            return
        if isinstance(event, ResponseCompletedEvent | ResponseIncompleteEvent):
            response = event.response
            yield _terminal_chunk(response)
            if include_usage:
                yield _usage_chunk(response)
            return


__all__ = ["chat_completion_stream", "to_chat_completion", "to_response_request"]
