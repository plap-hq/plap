from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from plap.responses.contracts.base import (
    StrictModel,
    _reject_unsupported_type_variants,
)


class InputTextContent(StrictModel):
    text: str = Field(description="Plain text supplied as model input.")
    type: Literal["input_text"] = Field(description="Content discriminator.")


class InputImageContent(StrictModel):
    detail: Literal["low", "high", "auto", "original"] | None = Field(
        default=None,
        description="Requested image detail level.",
    )
    file_id: str | None = Field(default=None, description="Uploaded file ID backing the image input.")
    image_url: str | None = Field(
        default=None,
        description="Image URL or base64 data URL backing the image input.",
    )
    type: Literal["input_image"] = Field(description="Content discriminator.")


class InputFileContent(StrictModel):
    detail: Literal["low", "high"] | None = Field(
        default=None,
        description="Requested file rendering detail level.",
    )
    file_data: str | None = Field(default=None, description="Base64-encoded file data.")
    file_id: str | None = Field(default=None, description="Uploaded file ID backing the file input.")
    file_url: str | None = Field(default=None, description="URL backing the file input.")
    filename: str | None = Field(default=None, description="Filename associated with the file input.")
    type: Literal["input_file"] = Field(description="Content discriminator.")


class UrlCitationAnnotation(StrictModel):
    end_index: int = Field(description="Exclusive character offset where citation ends.")
    start_index: int = Field(description="Inclusive character offset where citation starts.")
    title: str | None = Field(default=None, description="Human-readable page title.")
    type: Literal["url_citation"] = Field(description="Annotation discriminator.")
    url: str = Field(description="Cited URL.")


class OutputTextLogprobTopLogprob(StrictModel):
    bytes: list[int] = Field(description="UTF-8 bytes for this possible token.")
    logprob: float = Field(description="Log probability for this possible token.")
    token: str = Field(description="Possible token text.")


class OutputTextLogprob(StrictModel):
    bytes: list[int] = Field(description="UTF-8 bytes for the emitted token.")
    logprob: float = Field(description="Log probability for the emitted token.")
    token: str = Field(description="Emitted token text.")
    top_logprobs: list[OutputTextLogprobTopLogprob] = Field(description="Most likely alternative tokens at this token position.")


class OutputTextContent(StrictModel):
    annotations: list[UrlCitationAnnotation] = Field(
        default_factory=list,
        description="Annotations over this output text, currently url citations.",
    )
    logprobs: list[OutputTextLogprob] | None = Field(
        default=None,
        description=("Token log probability data when requested via include/top_logprobs."),
    )
    text: str = Field(description="The assistant-visible text content.")
    type: Literal["output_text"] = Field(description="Content discriminator.")


class OutputRefusalContent(StrictModel):
    refusal: str = Field(description="The assistant refusal text.")
    type: Literal["refusal"] = Field(description="Content discriminator.")


class ReasoningTextContent(StrictModel):
    text: str = Field(description="Reasoning text content.")
    type: Literal["reasoning_text"] = Field(description="Content discriminator.")


class SummaryTextContent(StrictModel):
    text: str = Field(description="Reasoning summary text.")
    type: Literal["summary_text"] = Field(description="Content discriminator.")


type RequestMessageContentPart = Annotated[
    InputTextContent | InputImageContent | InputFileContent | OutputTextContent | OutputRefusalContent,
    Field(discriminator="type"),
]

type FunctionCallOutputContentPart = Annotated[
    InputTextContent | InputImageContent | InputFileContent,
    Field(discriminator="type"),
]

type ResponseContentPart = Annotated[
    OutputTextContent | OutputRefusalContent,
    Field(discriminator="type"),
]

type ItemStatus = Literal["in_progress", "completed", "incomplete"]


class RequestMessageItem(StrictModel):
    content: str | list[RequestMessageContentPart] = Field(description="Message text or supported content blocks.")
    id: str | None = Field(default=None, description="Optional ID for replayed items.")
    phase: Literal["commentary", "final_answer"] | None = Field(
        default=None,
        description="Assistant phase label to preserve on follow-up requests.",
    )
    role: Literal["user", "assistant", "system", "developer"] = Field(description="Role that produced the message.")
    status: Literal["in_progress", "completed", "incomplete"] | None = Field(
        default=None,
        description="Item status when replaying persisted input/output items.",
    )
    type: Literal["message"] = Field(description="Input item discriminator.")

    @field_validator("content", mode="before")
    @classmethod
    def validate_content_variants(cls, value: object) -> object:
        return _reject_unsupported_type_variants(
            value,
            allowed={"input_text", "input_image", "input_file", "output_text", "refusal"},
            label="message content",
        )

    @model_validator(mode="after")
    def validate_role_content_variants(self) -> RequestMessageItem:
        if isinstance(self.content, str):
            return self
        part_types = {part.type for part in self.content}
        allowed = {"output_text", "refusal"} if self.role == "assistant" else {"input_text", "input_image", "input_file"}
        unsupported = sorted(part_types - allowed)
        if unsupported:
            joined = ", ".join(unsupported)
            raise ValueError(f"message role {self.role!r} does not support content variants: {joined}")
        return self


class RequestItemReference(StrictModel):
    id: str = Field(description="ID of a previously created response item to include by reference.")
    type: Literal["item_reference"] = Field(description="Input item discriminator.")


class _FunctionCallItemBase(StrictModel):
    arguments: str = Field(description="JSON string of arguments emitted by the model.")
    call_id: str = Field(description="Stable call ID paired with function output.")
    name: str = Field(description="Function name chosen by the model.")
    namespace: str | None = Field(default=None, description="Optional function namespace.")
    status: ItemStatus | None = Field(
        default=None,
        description="Function call item status.",
    )
    type: Literal["function_call"] = Field(description="Item discriminator.")


class RequestFunctionCallItem(_FunctionCallItemBase):
    id: str | None = Field(default=None, description="Optional item ID.")


class _FunctionCallOutputItemBase(StrictModel):
    call_id: str = Field(description="Call ID this output satisfies.")
    output: str | list[FunctionCallOutputContentPart] = Field(description="Function result as a string or supported content blocks.")
    type: Literal["function_call_output"] = Field(description="Item discriminator.")

    @field_validator("output", mode="before")
    @classmethod
    def validate_output_variants(cls, value: object) -> object:
        return _reject_unsupported_type_variants(
            value,
            allowed={"input_text", "input_image", "input_file"},
            label="function_call_output content",
        )


class RequestFunctionCallOutputItem(_FunctionCallOutputItemBase):
    created_by: str | None = Field(default=None, description="Originator of the output when replaying response items verbatim.")
    id: str | None = Field(default=None, description="Optional output item ID.")
    status: ItemStatus | None = Field(
        default=None,
        description="Function output item status.",
    )


class _ReasoningItemBase(StrictModel):
    content: list[ReasoningTextContent] | None = Field(
        default=None,
        description="Optional reasoning text content blocks.",
    )
    encrypted_content: str | None = Field(
        default=None,
        description="Encrypted reasoning payload when included by the API.",
    )
    id: str = Field(description="Reasoning item ID.")
    status: ItemStatus | None = Field(
        default=None,
        description="Reasoning item status.",
    )
    summary: list[SummaryTextContent] = Field(description="Summaries of the reasoning content.")
    type: Literal["reasoning"] = Field(description="Item discriminator.")


class RequestReasoningItem(_ReasoningItemBase):
    id: str | None = Field(default=None, description="Optional reasoning item ID.")


class ResponseReasoningItem(_ReasoningItemBase):
    id: str = Field(description="Reasoning item ID.")


class _CompactionItemBase(StrictModel):
    encrypted_content: str = Field(description="Encrypted compacted context payload.")
    type: Literal["compaction"] = Field(description="Item discriminator.")


class RequestCompactionItem(_CompactionItemBase):
    created_by: str | None = Field(default=None, description="Originator of compaction when replaying response items verbatim.")
    id: str | None = Field(default=None, description="Optional compaction item ID.")


type RequestInputItem = Annotated[
    RequestMessageItem
    | RequestItemReference
    | RequestFunctionCallItem
    | RequestFunctionCallOutputItem
    | RequestReasoningItem
    | RequestCompactionItem,
    Field(discriminator="type"),
]


class ResponseMessageItem(StrictModel):
    content: list[ResponseContentPart] = Field(description="Output message content blocks.")
    id: str = Field(description="Unique output message item ID.")
    phase: Literal["commentary", "final_answer"] | None = Field(
        default=None,
        description="Assistant phase label for commentary or final answer.",
    )
    role: Literal["assistant"] = Field(description="Output role; always assistant.")
    status: Literal["in_progress", "completed", "incomplete"] = Field(description="Message generation status.")
    type: Literal["message"] = Field(description="Output item discriminator.")


class ResponseFunctionCallItem(_FunctionCallItemBase):
    id: str = Field(description="Unique function call item ID.")


class ResponseFunctionCallOutputItem(_FunctionCallOutputItemBase):
    created_by: str | None = Field(default=None, description="Originator of the output.")
    id: str = Field(description="Unique function output item ID.")
    status: ItemStatus = Field(description="Function output item status.")


class ResponseCompactionItem(_CompactionItemBase):
    created_by: str | None = Field(default=None, description="Originator of compaction.")
    id: str = Field(description="Unique compaction item ID.")


# OpenAI exposes web_search_call output items. plap emits web search as a
# server-owned function_call/function_call_output pair instead.
# class WebSearchActionSearchSource(StrictModel): ...
# class WebSearchActionSearch(StrictModel): ...
# class WebSearchActionOpenPage(StrictModel): ...
# class WebSearchActionFindInPage(StrictModel): ...
# type WebSearchAction = ...
# class ResponseWebSearchCallItem(StrictModel): ...


type ResponseOutputItem = Annotated[
    ResponseMessageItem | ResponseFunctionCallItem | ResponseFunctionCallOutputItem | ResponseReasoningItem | ResponseCompactionItem,
    # | ResponseWebSearchCallItem,
    Field(discriminator="type"),
]


class InputItemsMessageItem(StrictModel):
    content: str | list[RequestMessageContentPart] = Field(description="Message text or supported content blocks.")
    id: str = Field(description="Input item ID.")
    phase: Literal["commentary", "final_answer"] | None = Field(
        default=None,
        description="Assistant phase label, when present.",
    )
    role: Literal["user", "assistant", "system", "developer"] = Field(description="Role that produced this input item.")
    status: Literal["in_progress", "completed", "incomplete"] | None = Field(
        default=None,
        description="Input item status.",
    )
    type: Literal["message"] = Field(description="Input item discriminator.")


type InputItemsPageItem = Annotated[
    InputItemsMessageItem
    | RequestItemReference
    | ResponseFunctionCallItem
    | ResponseFunctionCallOutputItem
    | ResponseReasoningItem
    | ResponseCompactionItem,
    Field(discriminator="type"),
]
