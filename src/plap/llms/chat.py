from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

type ChatRole = Literal["system", "developer", "user", "assistant", "tool"]
type ChatToolChoiceMode = Literal["none", "auto", "required"]
type ChatResponseFormatType = Literal["text", "json_object", "json_schema"]
type ChatFinishReason = Literal[
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "function_call",
]
type ReasoningEffort = str | int | bool
type ServiceTier = str


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
    tool_calls: list[ChatToolCall] | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ChatResponseFormat:
    type: ChatResponseFormatType
    name: str | None = None
    schema: dict[str, Any] | None = None
    strict: bool | None = None
    description: str | None = None


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


@dataclass(frozen=True)
class ChatAssistantMessage:
    content: str | None = None
    refusal: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ChatToolCall] | None = None


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
    message: ChatAssistantMessage
    finish_reason: ChatFinishReason | None
    usage: ChatUsage | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None


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
    tool_call_delta: ChatToolCallDelta | None = None
    finish_reason: ChatFinishReason | None = None
    usage: ChatUsage | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None


@runtime_checkable
class IChatCompletionClient(Protocol):
    async def complete(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResult: ...

    def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionDelta]: ...
