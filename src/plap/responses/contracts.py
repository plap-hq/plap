from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _reject_unsupported_type_variants(
    value: object,
    *,
    allowed: set[str],
    label: str,
) -> object:
    if not isinstance(value, list):
        return value

    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type is None:
            raise ValueError(f"Missing {label} type at index {index}")
        if item_type not in allowed:
            raise ValueError(
                f"Unsupported {label} variant '{item_type}' at index {index}"
            )

    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InputTextContent(StrictModel):
    text: str
    type: Literal["input_text"]


class UrlCitationAnnotation(StrictModel):
    end_index: int
    start_index: int
    title: str | None = None
    type: Literal["url_citation"]
    url: str


class OutputTextContent(StrictModel):
    annotations: list[UrlCitationAnnotation] = Field(default_factory=list)
    logprobs: list[dict[str, Any]] | None = None
    text: str
    type: Literal["output_text"]


class ReasoningTextContent(StrictModel):
    text: str
    type: Literal["reasoning_text"]


class SummaryTextContent(StrictModel):
    text: str
    type: Literal["summary_text"]


type MessageContentPart = Annotated[
    InputTextContent | OutputTextContent,
    Field(discriminator="type"),
]

type ToolOutputContentPart = Annotated[
    InputTextContent,
    Field(discriminator="type"),
]

type ResponseContentPart = Annotated[
    OutputTextContent | ReasoningTextContent,
    Field(discriminator="type"),
]


class RequestMessageItem(StrictModel):
    content: str | list[MessageContentPart]
    id: str | None = None
    phase: Literal["commentary", "final_answer"] | None = None
    role: Literal["user", "assistant", "system", "developer"]
    status: Literal["in_progress", "completed", "incomplete"] | None = None
    type: Literal["message"]

    @field_validator("content", mode="before")
    @classmethod
    def validate_content_variants(cls, value: object) -> object:
        return _reject_unsupported_type_variants(
            value,
            allowed={"input_text", "output_text"},
            label="message content",
        )


class RequestFunctionCallItem(StrictModel):
    arguments: str
    call_id: str
    id: str | None = None
    name: str
    namespace: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None
    type: Literal["function_call"]


class RequestFunctionCallOutputItem(StrictModel):
    call_id: str
    id: str | None = None
    output: str | list[ToolOutputContentPart]
    status: Literal["in_progress", "completed", "incomplete"] | None = None
    type: Literal["function_call_output"]

    @field_validator("output", mode="before")
    @classmethod
    def validate_output_variants(cls, value: object) -> object:
        return _reject_unsupported_type_variants(
            value,
            allowed={"input_text"},
            label="function_call_output content",
        )


class RequestReasoningItem(StrictModel):
    content: list[ReasoningTextContent] | None = None
    encrypted_content: str | None = None
    id: str
    status: Literal["in_progress", "completed", "incomplete"] | None = None
    summary: list[SummaryTextContent]
    type: Literal["reasoning"]


class RequestCompactionItem(StrictModel):
    encrypted_content: str
    id: str | None = None
    type: Literal["compaction"]


type RequestInputItem = Annotated[
    RequestMessageItem
    | RequestFunctionCallItem
    | RequestFunctionCallOutputItem
    | RequestReasoningItem
    | RequestCompactionItem,
    Field(discriminator="type"),
]


class FunctionTool(StrictModel):
    defer_loading: bool | None = None
    description: str | None = None
    name: str
    parameters: dict[str, Any] | None
    strict: bool | None
    type: Literal["function"]


class WebSearchFilters(StrictModel):
    allowed_domains: list[str] | None = None


class WebSearchUserLocation(StrictModel):
    city: str | None = None
    country: str | None = None
    region: str | None = None
    timezone: str | None = None
    type: Literal["approximate"] | None = None


class WebSearchTool(StrictModel):
    filters: WebSearchFilters | None = None
    search_context_size: Literal["low", "medium", "high"] | None = None
    type: Literal["web_search", "web_search_2025_08_26"]
    user_location: WebSearchUserLocation | None = None


type SupportedTool = Annotated[
    FunctionTool | WebSearchTool,
    Field(discriminator="type"),
]


class ToolChoiceFunction(StrictModel):
    name: str
    type: Literal["function"]


type ToolChoice = Literal["none", "auto", "required"] | ToolChoiceFunction


class TextFormatText(StrictModel):
    type: Literal["text"]


class TextFormatJSONObject(StrictModel):
    type: Literal["json_object"]


class TextFormatJSONSchema(StrictModel):
    description: str | None = None
    name: str
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool | None = None
    type: Literal["json_schema"]


type TextFormat = Annotated[
    TextFormatText | TextFormatJSONObject | TextFormatJSONSchema,
    Field(discriminator="type"),
]


class ResponseTextConfig(StrictModel):
    format: TextFormat | None = None


class ContextManagementCompaction(StrictModel):
    compact_threshold: int | None = None
    type: Literal["compaction"]


class ResponseCreateRequest(StrictModel):
    background: bool | None = None
    context_management: list[ContextManagementCompaction] | None = None
    conversation: dict[str, Any] | None = None
    include: list[str] | None = None
    input: str | list[RequestInputItem] | None = None
    instructions: str | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    metadata: dict[str, str] | None = None
    model: str | None = None
    parallel_tool_calls: bool | None = None
    previous_response_id: str | None = None
    prompt: dict[str, Any] | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: Literal["in-memory", "24h"] | None = None
    reasoning: dict[str, Any] | None = None
    safety_identifier: str | None = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None
    store: bool | None = None
    stream: bool | None = None
    stream_options: dict[str, Any] | None = None
    temperature: float | None = None
    text: ResponseTextConfig | None = None
    tool_choice: ToolChoice | None = None
    tools: list[SupportedTool] | None = None
    top_logprobs: int | None = None
    top_p: float | None = None
    truncation: Literal["auto", "disabled"] | None = None
    user: str | None = None

    @field_validator("input", mode="before")
    @classmethod
    def validate_input_variants(cls, value: object) -> object:
        return _reject_unsupported_type_variants(
            value,
            allowed={
                "message",
                "function_call",
                "function_call_output",
                "reasoning",
                "compaction",
            },
            label="input item",
        )

    @field_validator("tools", mode="before")
    @classmethod
    def validate_tool_variants(cls, value: object) -> object:
        return _reject_unsupported_type_variants(
            value,
            allowed={"function", "web_search", "web_search_2025_08_26"},
            label="tool",
        )


class InputTokensCountRequest(StrictModel):
    conversation: dict[str, Any] | None = None
    input: str | list[RequestInputItem] | None = None
    instructions: str | None = None
    model: str | None = None
    parallel_tool_calls: bool | None = None
    previous_response_id: str | None = None
    reasoning: dict[str, Any] | None = None
    text: ResponseTextConfig | None = None
    tool_choice: ToolChoice | None = None
    tools: list[SupportedTool] | None = None
    truncation: Literal["auto", "disabled"] | None = None

    @field_validator("input", mode="before")
    @classmethod
    def validate_input_variants(cls, value: object) -> object:
        return ResponseCreateRequest.validate_input_variants(value)

    @field_validator("tools", mode="before")
    @classmethod
    def validate_tool_variants(cls, value: object) -> object:
        return ResponseCreateRequest.validate_tool_variants(value)


class CompactRequest(StrictModel):
    input: str | list[RequestInputItem] | None = None
    instructions: str | None = None
    model: str | None = None
    previous_response_id: str | None = None
    prompt_cache_key: str | None = None

    @field_validator("input", mode="before")
    @classmethod
    def validate_input_variants(cls, value: object) -> object:
        return ResponseCreateRequest.validate_input_variants(value)


class ResponseMessageItem(StrictModel):
    content: list[OutputTextContent]
    id: str
    phase: Literal["commentary", "final_answer"] | None = None
    role: Literal["assistant"]
    status: Literal["in_progress", "completed", "incomplete"]
    type: Literal["message"]


class ResponseFunctionCallItem(StrictModel):
    arguments: str
    call_id: str
    id: str
    name: str
    namespace: str | None = None
    status: Literal["in_progress", "completed", "incomplete"] | None = None
    type: Literal["function_call"]


class ResponseFunctionCallOutputItem(StrictModel):
    call_id: str
    created_by: str | None = None
    id: str
    output: str | list[ToolOutputContentPart]
    status: Literal["in_progress", "completed", "incomplete"]
    type: Literal["function_call_output"]


class ResponseReasoningItem(StrictModel):
    content: list[ReasoningTextContent] | None = None
    encrypted_content: str | None = None
    id: str
    status: Literal["in_progress", "completed", "incomplete"] | None = None
    summary: list[SummaryTextContent]
    type: Literal["reasoning"]


class ResponseCompactionItem(StrictModel):
    created_by: str | None = None
    encrypted_content: str
    id: str
    type: Literal["compaction"]


class WebSearchActionSearch(StrictModel):
    queries: list[str] | None = None
    query: str
    sources: list[dict[str, Any]] | None = None
    type: Literal["search"]


class WebSearchActionOpenPage(StrictModel):
    type: Literal["open_page"]
    url: str | None = None


class WebSearchActionFindInPage(StrictModel):
    pattern: str
    type: Literal["find_in_page"]
    url: str


type WebSearchAction = Annotated[
    WebSearchActionSearch | WebSearchActionOpenPage | WebSearchActionFindInPage,
    Field(discriminator="type"),
]


class ResponseWebSearchCallItem(StrictModel):
    action: WebSearchAction
    id: str
    status: Literal["in_progress", "searching", "completed", "failed"]
    type: Literal["web_search_call"]


type ResponseOutputItem = Annotated[
    ResponseMessageItem
    | ResponseFunctionCallItem
    | ResponseFunctionCallOutputItem
    | ResponseReasoningItem
    | ResponseCompactionItem
    | ResponseWebSearchCallItem,
    Field(discriminator="type"),
]


class ResponseUsage(StrictModel):
    input_tokens: int
    input_tokens_details: dict[str, Any] | None = None
    output_tokens: int
    output_tokens_details: dict[str, Any] | None = None
    total_tokens: int


class ResponseError(StrictModel):
    code: str | None = None
    message: str
    param: str | None = None


type ResponseStatus = Literal[
    "queued",
    "in_progress",
    "completed",
    "failed",
    "cancelled",
    "incomplete",
]


class ResponseObject(StrictModel):
    background: bool | None = None
    completed_at: int | None = None
    conversation: dict[str, Any] | None = None
    created_at: int
    error: ResponseError | None = None
    id: str
    incomplete_details: dict[str, Any] | None = None
    instructions: str | None = None
    max_output_tokens: int | None = None
    max_tool_calls: int | None = None
    metadata: dict[str, str] | None = None
    model: str | None = None
    object: Literal["response"] = "response"
    output: list[ResponseOutputItem]
    parallel_tool_calls: bool | None = None
    previous_response_id: str | None = None
    prompt: dict[str, Any] | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: Literal["in-memory", "24h"] | None = None
    reasoning: dict[str, Any] | None = None
    safety_identifier: str | None = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None
    status: ResponseStatus
    temperature: float | None = None
    text: ResponseTextConfig | None = None
    tool_choice: ToolChoice | None = None
    tools: list[SupportedTool] | None = None
    top_logprobs: int | None = None
    top_p: float | None = None
    truncation: Literal["auto", "disabled"] | None = None
    usage: ResponseUsage | None = None
    user: str | None = None


class ResponseDeleted(StrictModel):
    deleted: Literal[True]
    id: str
    object: Literal["response"] = "response"


class CompactedResponseObject(StrictModel):
    created_at: int
    id: str
    object: Literal["response.compaction"] = "response.compaction"
    output: list[ResponseCompactionItem]
    usage: ResponseUsage


class InputTokenCountResponse(StrictModel):
    input_tokens: int
    object: Literal["input_token_count"] = "input_token_count"


class InputItemsMessageItem(StrictModel):
    content: str | list[MessageContentPart]
    id: str
    phase: Literal["commentary", "final_answer"] | None = None
    role: Literal["user", "assistant", "system", "developer"]
    status: Literal["in_progress", "completed", "incomplete"] | None = None
    type: Literal["message"]


class InputItemsFunctionCallItem(ResponseFunctionCallItem):
    pass


class InputItemsFunctionCallOutputItem(ResponseFunctionCallOutputItem):
    pass


class InputItemsReasoningItem(ResponseReasoningItem):
    pass


class InputItemsCompactionItem(ResponseCompactionItem):
    pass


type InputItemsPageItem = Annotated[
    InputItemsMessageItem
    | InputItemsFunctionCallItem
    | InputItemsFunctionCallOutputItem
    | InputItemsReasoningItem
    | InputItemsCompactionItem,
    Field(discriminator="type"),
]


class InputItemsPage(StrictModel):
    data: list[InputItemsPageItem]
    first_id: str | None = None
    has_more: bool
    last_id: str | None = None
    object: Literal["list"] = "list"


class ResponseCreatedEvent(StrictModel):
    response: ResponseObject
    sequence_number: int
    type: Literal["response.created"]


class ResponseInProgressEvent(StrictModel):
    response: ResponseObject
    sequence_number: int
    type: Literal["response.in_progress"]


class ResponseCompletedEvent(StrictModel):
    response: ResponseObject
    sequence_number: int
    type: Literal["response.completed"]


class ResponseOutputItemAddedEvent(StrictModel):
    item: ResponseOutputItem
    output_index: int
    sequence_number: int
    type: Literal["response.output_item.added"]


class ResponseOutputItemDoneEvent(StrictModel):
    item: ResponseOutputItem
    output_index: int
    sequence_number: int
    type: Literal["response.output_item.done"]


class ResponseContentPartAddedEvent(StrictModel):
    content_index: int
    item_id: str
    output_index: int
    part: ResponseContentPart
    sequence_number: int
    type: Literal["response.content_part.added"]


class ResponseContentPartDoneEvent(StrictModel):
    content_index: int
    item_id: str
    output_index: int
    part: ResponseContentPart
    sequence_number: int
    type: Literal["response.content_part.done"]


class ResponseTextDeltaEvent(StrictModel):
    content_index: int
    delta: str
    item_id: str
    logprobs: list[dict[str, Any]] = Field(default_factory=list)
    output_index: int
    sequence_number: int
    type: Literal["response.output_text.delta"]


class ResponseTextDoneEvent(StrictModel):
    content_index: int
    item_id: str
    logprobs: list[dict[str, Any]] = Field(default_factory=list)
    output_index: int
    sequence_number: int
    text: str
    type: Literal["response.output_text.done"]


class ResponseOutputTextAnnotationAddedEvent(StrictModel):
    annotation: UrlCitationAnnotation
    annotation_index: int
    content_index: int
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.output_text.annotation.added"]


class ResponseFunctionCallArgumentsDeltaEvent(StrictModel):
    delta: str
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.function_call_arguments.delta"]


class ResponseFunctionCallArgumentsDoneEvent(StrictModel):
    arguments: str
    item_id: str
    name: str
    output_index: int
    sequence_number: int
    type: Literal["response.function_call_arguments.done"]


class ResponseReasoningSummaryPartAddedEvent(StrictModel):
    item_id: str
    output_index: int
    part: SummaryTextContent
    sequence_number: int
    summary_index: int
    type: Literal["response.reasoning_summary_part.added"]


class ResponseReasoningSummaryPartDoneEvent(StrictModel):
    item_id: str
    output_index: int
    part: SummaryTextContent
    sequence_number: int
    summary_index: int
    type: Literal["response.reasoning_summary_part.done"]


class ResponseReasoningSummaryTextDeltaEvent(StrictModel):
    delta: str
    item_id: str
    output_index: int
    sequence_number: int
    summary_index: int
    type: Literal["response.reasoning_summary_text.delta"]


class ResponseReasoningSummaryTextDoneEvent(StrictModel):
    item_id: str
    output_index: int
    sequence_number: int
    summary_index: int
    text: str
    type: Literal["response.reasoning_summary_text.done"]


class ResponseReasoningTextDeltaEvent(StrictModel):
    content_index: int
    delta: str
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.reasoning_text.delta"]


class ResponseReasoningTextDoneEvent(StrictModel):
    content_index: int
    item_id: str
    output_index: int
    sequence_number: int
    text: str
    type: Literal["response.reasoning_text.done"]


class ResponseWebSearchCallInProgressEvent(StrictModel):
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.web_search_call.in_progress"]


class ResponseWebSearchCallSearchingEvent(StrictModel):
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.web_search_call.searching"]


class ResponseWebSearchCallCompletedEvent(StrictModel):
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.web_search_call.completed"]


class ResponseErrorEvent(StrictModel):
    code: str | None = None
    message: str
    param: str | None = None
    sequence_number: int
    type: Literal["error"]


type ResponseStreamEvent = Annotated[
    ResponseCreatedEvent
    | ResponseInProgressEvent
    | ResponseCompletedEvent
    | ResponseOutputItemAddedEvent
    | ResponseOutputItemDoneEvent
    | ResponseContentPartAddedEvent
    | ResponseContentPartDoneEvent
    | ResponseTextDeltaEvent
    | ResponseTextDoneEvent
    | ResponseOutputTextAnnotationAddedEvent
    | ResponseFunctionCallArgumentsDeltaEvent
    | ResponseFunctionCallArgumentsDoneEvent
    | ResponseReasoningSummaryPartAddedEvent
    | ResponseReasoningSummaryPartDoneEvent
    | ResponseReasoningSummaryTextDeltaEvent
    | ResponseReasoningSummaryTextDoneEvent
    | ResponseReasoningTextDeltaEvent
    | ResponseReasoningTextDoneEvent
    | ResponseWebSearchCallInProgressEvent
    | ResponseWebSearchCallSearchingEvent
    | ResponseWebSearchCallCompletedEvent
    | ResponseErrorEvent,
    Field(discriminator="type"),
]


class ResponseCreateClientEvent(StrictModel):
    response: ResponseCreateRequest
    type: Literal["response.create"]
