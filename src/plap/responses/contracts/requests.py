from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from plap.responses.contracts.base import (
    Metadata,
    StrictModel,
    _reject_unsupported_type_variants,
    _validate_metadata,
)
from plap.responses.contracts.items import InputTextContent, RequestInputItem
from plap.responses.contracts.tools import (
    SUPPORTED_TOOL_TYPES,
    ResponseIncludable,
    ResponseTextConfig,
    SupportedTool,
    ToolChoice,
)


def _normalize_easy_input_messages(value: object) -> object:
    if not isinstance(value, list):
        return value
    normalized: list[object] = []
    changed = False
    for item in value:
        if isinstance(item, dict) and "type" not in item and "role" in item:
            normalized.append({"type": "message", **item})
            changed = True
            continue
        if isinstance(item, dict) and "type" not in item and set(item) == {"id"}:
            normalized.append({"type": "item_reference", **item})
            changed = True
            continue
        normalized.append(item)
    return normalized if changed else value


class ConversationReference(StrictModel):
    id: str = Field(description="Conversation ID.")


class PromptReference(StrictModel):
    id: str = Field(description="Prompt template ID.")
    variables: dict[str, str | InputTextContent] | None = Field(
        default=None,
        description="Prompt variables supported by this reduced text-only fragment.",
    )
    version: str | None = Field(default=None, description="Prompt template version.")


type ReasoningEffort = Literal[
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
]

type ReasoningSummary = Literal["auto", "concise", "detailed"]

# OpenAI also defines scale; plap does not expose that tier.
type ServiceTier = Literal["auto", "default", "priority", "flex"]


class ReasoningConfig(StrictModel):
    effort: ReasoningEffort | None = Field(
        default=None,
        description="Reasoning effort requested for reasoning-capable models.",
    )
    generate_summary: ReasoningSummary | None = Field(
        default=None,
        description="Deprecated OpenAI summary selector; use summary instead.",
    )
    summary: ReasoningSummary | None = Field(
        default=None,
        description="Reasoning summary mode requested from the model.",
    )


class StreamOptions(StrictModel):
    include_obfuscation: bool | None = Field(
        default=None,
        description="Whether stream delta events should include obfuscation fields.",
    )


class ContextManagementCompaction(StrictModel):
    soft_compact_threshold: int | None = Field(
        default=None,
        ge=0,
        description="Soft token threshold at which dedicated context compaction should be attempted.",
    )
    compact_threshold: int | None = Field(
        default=None,
        ge=0,
        description="Hard token threshold at which dedicated context compaction should be attempted.",
    )
    compact_max_rounds: int | None = Field(
        default=None,
        ge=0,
        description="Maximum compaction rounds allowed for this request.",
    )
    type: Literal["compaction"] = Field(description="Context-management discriminator.")

    @model_validator(mode="after")
    def validate_override(self) -> ContextManagementCompaction:
        if self.soft_compact_threshold is None and self.compact_threshold is None and self.compact_max_rounds is None:
            raise ValueError("compaction context_management requires at least one threshold or round override")
        if (
            self.soft_compact_threshold is not None
            and self.compact_threshold is not None
            and self.compact_threshold <= self.soft_compact_threshold
        ):
            raise ValueError("compact_threshold must exceed soft_compact_threshold")
        return self


class ResponseCreateRequest(StrictModel):
    # background: bool | None = Field(
    #     default=None,
    #     description="Whether to run response generation in the background.",
    # )
    context_management: list[ContextManagementCompaction] | None = Field(
        default=None,
        description=("Context-management entries; currently only compaction is supported."),
    )
    conversation: str | ConversationReference | None = Field(
        default=None,
        description=("Conversation this response belongs to. Cannot be combined with previous_response_id in OpenAI's contract."),
    )
    include: list[ResponseIncludable] | None = Field(
        default=None,
        description=("Additional output data to include, such as message.output_text.logprobs, or reasoning.encrypted_content."),
    )
    input: str | list[RequestInputItem] | None = Field(
        default=None,
        description="Text or supported input items used to generate the response.",
    )
    instructions: str | None = Field(
        default=None,
        description=(
            "System/developer instructions inserted into model context; not carried over automatically when previous_response_id is used."
        ),
    )
    max_output_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Upper bound for visible output plus reasoning tokens.",
    )
    max_tool_calls: int | None = Field(
        default=None,
        ge=1,
        description="Maximum total built-in tool calls processed for the response.",
    )
    metadata: Metadata | None = Field(
        default=None,
        description="Caller metadata attached to the response object.",
    )
    model: str | None = Field(
        default=None,
        description="Model ID requested by the client.",
    )
    parallel_tool_calls: bool | None = Field(
        default=None,
        description="Whether the model may run tool calls in parallel.",
    )
    previous_response_id: str | None = Field(
        default=None,
        description="Previous response ID used for multi-turn continuation.",
    )
    prompt: PromptReference | None = Field(
        default=None,
        description="Reusable prompt template reference and variables.",
    )
    prompt_cache_key: str | None = Field(
        default=None,
        description=("Stable cache key for prompt caching; replaces the legacy user field."),
    )
    # prompt_cache_retention: Literal["in-memory", "24h"] | None = Field(
    #     default=None,
    #     description="Prompt cache retention policy.",
    # )
    reasoning: ReasoningConfig | None = Field(
        default=None,
        description="Reasoning configuration for compatible models.",
    )
    safety_identifier: str | None = Field(
        default=None,
        description=("Stable abuse-detection identifier; prefer hashed user identifiers."),
    )
    service_tier: ServiceTier | None = Field(
        default=None,
        description="Requested processing tier.",
    )
    store: bool | None = Field(
        default=None,
        description="Whether to store the generated response for later retrieval.",
    )
    stream: bool | None = Field(
        default=None,
        description="When true, stream events using SSE for HTTP create calls.",
    )
    stream_options: StreamOptions | None = Field(
        default=None,
        description="Streaming options; only meaningful when stream is true.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0,
        le=2,
        description="Sampling temperature between 0 and 2.",
    )
    text: ResponseTextConfig | None = Field(
        default=None,
        description="Text/structured-output response configuration.",
    )
    tool_choice: ToolChoice | None = Field(
        default=None,
        description=("How the model should choose tools: none, auto, required, or function."),
    )
    tools: list[SupportedTool] | None = Field(
        default=None,
        description="Supported tools: function tools and the server-backed web_search enablement flag with optional user_location.",
    )
    top_logprobs: int | None = Field(
        default=None,
        ge=0,
        le=20,
        description=("Number of likely tokens and log probabilities to return per token."),
    )
    top_p: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=("Nucleus sampling parameter; generally tune instead of temperature."),
    )
    truncation: Literal["auto", "disabled"] | None = Field(
        default=None,
        description="Whether to auto-truncate oversized context or fail when disabled.",
    )
    user: str | None = Field(
        default=None,
        description=("Legacy end-user identifier; prefer safety_identifier/prompt_cache_key."),
    )

    @field_validator("input", mode="before")
    @classmethod
    def validate_input_variants(cls, value: object) -> object:
        value = _normalize_easy_input_messages(value)
        return _reject_unsupported_type_variants(
            value,
            allowed={
                "message",
                "item_reference",
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
            allowed=set(SUPPORTED_TOOL_TYPES),
            label="tool",
        )

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: Metadata | None) -> Metadata | None:
        return _validate_metadata(value)


class CompactRequest(StrictModel):
    input: str | list[RequestInputItem] | None = Field(
        default=None,
        description="Text or supported input items to compact.",
    )
    instructions: str | None = Field(
        default=None,
        description="Instructions associated with the compaction request.",
    )
    model: str | None = Field(
        default=None,
        description="Model ID requested for compaction.",
    )
    previous_response_id: str | None = Field(
        default=None,
        description="Previous response ID whose context should be compacted.",
    )
    prompt_cache_key: str | None = Field(
        default=None,
        description="Prompt cache key associated with compacted context.",
    )

    @field_validator("input", mode="before")
    @classmethod
    def validate_input_variants(cls, value: object) -> object:
        return ResponseCreateRequest.validate_input_variants(value)


class ResponseCreateClientEvent(StrictModel):
    response: ResponseCreateRequest = Field(description="Responses create request payload.")
    type: Literal["response.create"] = Field(description="Websocket client event discriminator.")
