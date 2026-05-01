from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable


class ChatRole(StrEnum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatToolChoiceMode(StrEnum):
    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


class ChatResponseFormatType(StrEnum):
    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


class ChatFinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    FUNCTION_CALL = "function_call"


class ReasoningEffort(StrEnum):
    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


type ServiceTier = str

REASONING_EFFORT_VALUES: frozenset[ReasoningEffort] = frozenset(ReasoningEffort)


@dataclass(frozen=True)
class ChatFunctionTool:
    name: str
    parameters: dict[str, Any] | None = None
    strict: bool | None = None
    description: str | None = None


@dataclass(frozen=True)
class ChatTool:
    function: ChatFunctionTool
    type: Literal["function"] = "function"


@dataclass(frozen=True)
class ChatToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ChatToolChoiceFunction:
    name: str
    type: Literal["function"] = "function"


type ChatToolChoice = ChatToolChoiceMode | ChatToolChoiceFunction


@dataclass(frozen=True)
class ChatMessage:
    role: ChatRole
    content: str | None = None
    name: str | None = None
    refusal: str | None = None
    tool_calls: list[ChatToolCall] | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None
    reasoning_details: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", ChatRole(self.role))


@dataclass(frozen=True)
class ChatResponseFormat:
    type: ChatResponseFormatType
    name: str | None = None
    schema: dict[str, Any] | None = None
    strict: bool | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", ChatResponseFormatType(self.type))


@dataclass(frozen=True)
class ChatPrediction:
    content: str
    type: Literal["content"] = "content"


@dataclass(frozen=True)
class ChatStreamOptions:
    include_usage: bool | None = None


@dataclass(frozen=True)
class ChatCompletionRequest:
    model: str
    messages: list[ChatMessage]
    tools: list[ChatTool] = field(default_factory=list)
    tool_choice: ChatToolChoice | None = None
    parallel_tool_calls: bool | None = None
    response_format: ChatResponseFormat | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    logit_bias: dict[str, int] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    n: int | None = None
    reasoning_effort: ReasoningEffort | None = None
    stream_options: ChatStreamOptions | None = None
    user: str | None = None
    prompt_cache_key: str | None = None
    metadata: dict[str, str] | None = None
    service_tier: ServiceTier | None = None
    prediction: ChatPrediction | None = None

    def __post_init__(self) -> None:
        if self.reasoning_effort is not None:
            try:
                object.__setattr__(self, "reasoning_effort", ReasoningEffort(self.reasoning_effort))
            except ValueError as exc:
                allowed = ", ".join(value.value for value in ReasoningEffort)
                raise ValueError(f"reasoning_effort must be one of: {allowed}") from exc
        if isinstance(self.tool_choice, str):
            object.__setattr__(self, "tool_choice", ChatToolChoiceMode(self.tool_choice))


@dataclass(frozen=True)
class ChatUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class ChatCompletionResult:
    id: str | None
    model: str
    created_at: float | None
    message: ChatMessage
    finish_reason: ChatFinishReason | None
    usage: ChatUsage | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None

    def __post_init__(self) -> None:
        if self.finish_reason is not None:
            object.__setattr__(self, "finish_reason", ChatFinishReason(self.finish_reason))


@dataclass(frozen=True)
class ChatToolCallDelta:
    index: int
    id: str | None = None
    name: str | None = None
    arguments_delta: str | None = None


@dataclass(frozen=True)
class ChatCompletionDelta:
    id: str | None
    model: str | None
    created_at: float | None
    choice_index: int
    content_delta: str | None = None
    refusal_delta: str | None = None
    reasoning_delta: str | None = None
    reasoning_details_delta: list[dict[str, Any]] | None = None
    tool_call_delta: ChatToolCallDelta | None = None
    finish_reason: ChatFinishReason | None = None
    usage: ChatUsage | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None

    def __post_init__(self) -> None:
        if self.finish_reason is not None:
            object.__setattr__(self, "finish_reason", ChatFinishReason(self.finish_reason))


@runtime_checkable
class IChatCompletionClient(Protocol):
    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult: ...

    def stream(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionDelta]: ...
