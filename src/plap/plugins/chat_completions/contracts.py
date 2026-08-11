from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from plap.responses.contracts import ReasoningEffort, ReasoningSummary, ServiceTier


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatTextContentPart(_StrictModel):
    text: str
    type: Literal["text"]


class ChatImageURL(_StrictModel):
    url: str
    detail: Literal["auto", "low", "high", "original"] | None = None


class ChatImageContentPart(_StrictModel):
    image_url: ChatImageURL
    type: Literal["image_url"]


class ChatFile(_StrictModel):
    file_data: str | None = None
    file_id: str | None = None
    filename: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> ChatFile:
        if self.file_data is None and self.file_id is None:
            raise ValueError("chat file requires file_data or file_id")
        return self


class ChatFileContentPart(_StrictModel):
    file: ChatFile
    type: Literal["file"]


type ChatUserContentPart = Annotated[
    ChatTextContentPart | ChatImageContentPart | ChatFileContentPart,
    Field(discriminator="type"),
]


class ChatSystemMessage(_StrictModel):
    content: str | list[ChatTextContentPart]
    role: Literal["system"]


class ChatDeveloperMessage(_StrictModel):
    content: str | list[ChatTextContentPart]
    role: Literal["developer"]


class ChatUserMessage(_StrictModel):
    content: str | list[ChatUserContentPart]
    role: Literal["user"]


class ChatToolCallFunction(_StrictModel):
    arguments: str
    name: str


class ChatToolCall(_StrictModel):
    function: ChatToolCallFunction
    id: str
    type: Literal["function"] = "function"


class ChatReasoningEncrypted(_StrictModel):
    data: str
    type: Literal["reasoning.encrypted"] = "reasoning.encrypted"
    format: str | None = None
    id: str | None = None
    index: int | None = Field(default=None, ge=0)


class ChatReasoningSummary(_StrictModel):
    summary: str
    type: Literal["reasoning.summary"] = "reasoning.summary"
    format: str | None = None
    id: str | None = None
    index: int | None = Field(default=None, ge=0)


class ChatReasoningText(_StrictModel):
    type: Literal["reasoning.text"] = "reasoning.text"
    format: str | None = None
    id: str | None = None
    index: int | None = Field(default=None, ge=0)
    signature: str | None = None
    text: str | None = None


type ChatReasoningDetail = Annotated[
    ChatReasoningEncrypted | ChatReasoningSummary | ChatReasoningText,
    Field(discriminator="type"),
]


class ChatAssistantMessage(_StrictModel):
    role: Literal["assistant"]
    annotations: None = None
    audio: None = None
    content: str | None = None
    function_call: None = None
    reasoning_details: list[ChatReasoningDetail] = Field(default_factory=list)
    refusal: str | None = None
    tool_calls: list[ChatToolCall] = Field(default_factory=list)


class ChatToolMessage(_StrictModel):
    content: str | list[ChatTextContentPart]
    role: Literal["tool"]
    tool_call_id: str


type ChatMessage = Annotated[
    ChatSystemMessage | ChatDeveloperMessage | ChatUserMessage | ChatAssistantMessage | ChatToolMessage,
    Field(discriminator="role"),
]


class ChatFunctionDefinition(_StrictModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class ChatFunctionTool(_StrictModel):
    function: ChatFunctionDefinition
    type: Literal["function"] = "function"


class ChatToolChoiceFunctionName(_StrictModel):
    name: str


class ChatToolChoiceFunction(_StrictModel):
    function: ChatToolChoiceFunctionName
    type: Literal["function"] = "function"


type ChatToolChoice = Literal["none", "auto", "required"] | ChatToolChoiceFunction


class ChatResponseFormatText(_StrictModel):
    type: Literal["text"]


class ChatResponseFormatJSONObject(_StrictModel):
    type: Literal["json_object"]


class ChatJSONSchema(_StrictModel):
    name: str
    description: str | None = None
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool | None = None


class ChatResponseFormatJSONSchema(_StrictModel):
    json_schema: ChatJSONSchema
    type: Literal["json_schema"]


type ChatResponseFormat = Annotated[
    ChatResponseFormatText | ChatResponseFormatJSONObject | ChatResponseFormatJSONSchema,
    Field(discriminator="type"),
]


class ChatReasoningConfig(_StrictModel):
    effort: ReasoningEffort | None = None
    summary: ReasoningSummary | None = None


class ChatStreamOptions(_StrictModel):
    include_usage: bool | None = None


class ChatCompletionCreateRequest(_StrictModel):
    messages: list[ChatMessage] = Field(min_length=1)
    model: str
    max_completion_tokens: int | None = Field(default=None, ge=1)
    logprobs: bool | None = None
    metadata: dict[str, str] | None = None
    n: Literal[1] | None = None
    parallel_tool_calls: bool | None = None
    prompt_cache_key: str | None = None
    reasoning: ChatReasoningConfig | None = None
    reasoning_effort: ReasoningEffort | None = None
    response_format: ChatResponseFormat | None = None
    safety_identifier: str | None = None
    service_tier: ServiceTier | None = None
    store: bool | None = None
    stream: bool = False
    stream_options: ChatStreamOptions | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    tool_choice: ChatToolChoice | None = None
    tools: list[ChatFunctionTool] = Field(default_factory=list)
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    top_p: float | None = Field(default=None, ge=0, le=1)
    user: str | None = None

    @model_validator(mode="after")
    def validate_reasoning(self) -> ChatCompletionCreateRequest:
        nested_effort = None if self.reasoning is None else self.reasoning.effort
        if nested_effort is not None and self.reasoning_effort is not None and nested_effort != self.reasoning_effort:
            raise ValueError("reasoning.effort and reasoning_effort cannot differ")
        if self.top_logprobs is not None and self.logprobs is not True:
            raise ValueError("top_logprobs requires logprobs=true")
        return self


class ChatCompletionUsageInputDetails(_StrictModel):
    cached_tokens: int = Field(ge=0)


class ChatCompletionUsageOutputDetails(_StrictModel):
    reasoning_tokens: int = Field(ge=0)


class ChatCompletionUsage(_StrictModel):
    completion_tokens: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    completion_tokens_details: ChatCompletionUsageOutputDetails | None = None
    prompt_tokens_details: ChatCompletionUsageInputDetails | None = None


class ChatCompletionMessage(_StrictModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    reasoning_details: list[ChatReasoningDetail] | None = None
    refusal: str | None = None
    tool_calls: list[ChatToolCall] | None = None


type ChatFinishReason = Literal["stop", "length", "tool_calls", "content_filter"]


class ChatCompletionTopLogprob(_StrictModel):
    bytes: list[int] | None = None
    logprob: float
    token: str


class ChatCompletionTokenLogprob(_StrictModel):
    bytes: list[int] | None = None
    logprob: float
    token: str
    top_logprobs: list[ChatCompletionTopLogprob]


class ChatCompletionLogprobs(_StrictModel):
    content: list[ChatCompletionTokenLogprob] | None = None
    refusal: list[ChatCompletionTokenLogprob] | None = None


class ChatCompletionChoice(_StrictModel):
    finish_reason: ChatFinishReason
    index: Literal[0] = 0
    logprobs: ChatCompletionLogprobs | None = None
    message: ChatCompletionMessage


class ChatCompletion(_StrictModel):
    choices: list[ChatCompletionChoice]
    created: int
    id: str
    model: str
    object: Literal["chat.completion"] = "chat.completion"
    service_tier: str | None = None
    usage: ChatCompletionUsage | None = None


class ChatToolCallDeltaFunction(_StrictModel):
    arguments: str | None = None
    name: str | None = None


class ChatToolCallDelta(_StrictModel):
    function: ChatToolCallDeltaFunction
    index: int = Field(ge=0)
    id: str | None = None
    type: Literal["function"] | None = None


class ChatCompletionChunkDelta(_StrictModel):
    content: str | None = None
    reasoning_details: list[ChatReasoningDetail] | None = None
    refusal: str | None = None
    role: Literal["assistant"] | None = None
    tool_calls: list[ChatToolCallDelta] | None = None


class ChatCompletionChunkChoice(_StrictModel):
    delta: ChatCompletionChunkDelta
    finish_reason: ChatFinishReason | None = None
    index: Literal[0] = 0
    logprobs: ChatCompletionLogprobs | None = None


class ChatCompletionChunk(_StrictModel):
    choices: list[ChatCompletionChunkChoice]
    created: int
    id: str
    model: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    service_tier: str | None = None
    usage: ChatCompletionUsage | None = None


__all__ = [
    "ChatAssistantMessage",
    "ChatCompletion",
    "ChatCompletionChoice",
    "ChatCompletionChunk",
    "ChatCompletionChunkChoice",
    "ChatCompletionChunkDelta",
    "ChatCompletionCreateRequest",
    "ChatCompletionLogprobs",
    "ChatCompletionMessage",
    "ChatCompletionTokenLogprob",
    "ChatCompletionTopLogprob",
    "ChatCompletionUsage",
    "ChatCompletionUsageInputDetails",
    "ChatCompletionUsageOutputDetails",
    "ChatDeveloperMessage",
    "ChatFileContentPart",
    "ChatFinishReason",
    "ChatFunctionTool",
    "ChatImageContentPart",
    "ChatMessage",
    "ChatReasoningDetail",
    "ChatReasoningEncrypted",
    "ChatReasoningSummary",
    "ChatReasoningText",
    "ChatSystemMessage",
    "ChatTextContentPart",
    "ChatToolCall",
    "ChatToolCallDelta",
    "ChatToolCallDeltaFunction",
    "ChatToolMessage",
    "ChatUserMessage",
]
