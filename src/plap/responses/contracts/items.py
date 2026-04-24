from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator

from plap.responses.contracts.base import (
    StrictModel,
    _reject_unsupported_type_variants,
)


class InputTextContent(StrictModel):
    text: str = Field(description="Plain text supplied as model input.")
    type: Literal["input_text"] = Field(description="Content discriminator.")


class UrlCitationAnnotation(StrictModel):
    end_index: int = Field(
        description="Exclusive character offset where citation ends."
    )
    start_index: int = Field(
        description="Inclusive character offset where citation starts."
    )
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
    top_logprobs: list[OutputTextLogprobTopLogprob] = Field(
        description="Most likely alternative tokens at this token position."
    )


class OutputTextContent(StrictModel):
    annotations: list[UrlCitationAnnotation] = Field(
        default_factory=list,
        description="Annotations over this output text, currently url citations.",
    )
    logprobs: list[OutputTextLogprob] | None = Field(
        default=None,
        description=(
            "Token log probability data when requested via include/top_logprobs."
        ),
    )
    text: str = Field(description="The assistant-visible text content.")
    type: Literal["output_text"] = Field(description="Content discriminator.")


class ReasoningTextContent(StrictModel):
    text: str = Field(description="Reasoning text content.")
    type: Literal["reasoning_text"] = Field(description="Content discriminator.")


class SummaryTextContent(StrictModel):
    text: str = Field(description="Reasoning summary text.")
    type: Literal["summary_text"] = Field(description="Content discriminator.")


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

type ItemStatus = Literal["in_progress", "completed", "incomplete"]


class RequestMessageItem(StrictModel):
    content: str | list[MessageContentPart] = Field(
        description="Message text or supported content blocks."
    )
    id: str | None = Field(default=None, description="Optional ID for replayed items.")
    phase: Literal["commentary", "final_answer"] | None = Field(
        default=None,
        description="Assistant phase label to preserve on follow-up requests.",
    )
    role: Literal["user", "assistant", "system", "developer"] = Field(
        description="Role that produced the message."
    )
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
            allowed={"input_text", "output_text"},
            label="message content",
        )


class _FunctionCallItemBase(StrictModel):
    arguments: str = Field(description="JSON string of arguments emitted by the model.")
    call_id: str = Field(description="Stable call ID paired with function output.")
    name: str = Field(description="Function name chosen by the model.")
    namespace: str | None = Field(
        default=None, description="Optional function namespace."
    )
    status: ItemStatus | None = Field(
        default=None,
        description="Function call item status.",
    )
    type: Literal["function_call"] = Field(description="Item discriminator.")


class RequestFunctionCallItem(_FunctionCallItemBase):
    id: str | None = Field(default=None, description="Optional item ID.")


class _FunctionCallOutputItemBase(StrictModel):
    call_id: str = Field(description="Call ID this output satisfies.")
    output: str | list[ToolOutputContentPart] = Field(
        description="Function result as a string or supported content blocks."
    )
    type: Literal["function_call_output"] = Field(description="Item discriminator.")

    @field_validator("output", mode="before")
    @classmethod
    def validate_output_variants(cls, value: object) -> object:
        return _reject_unsupported_type_variants(
            value,
            allowed={"input_text"},
            label="function_call_output content",
        )


class RequestFunctionCallOutputItem(_FunctionCallOutputItemBase):
    id: str | None = Field(default=None, description="Optional output item ID.")
    status: ItemStatus | None = Field(
        default=None,
        description="Function output item status.",
    )


class ReasoningItem(StrictModel):
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
    summary: list[SummaryTextContent] = Field(
        description="Summaries of the reasoning content."
    )
    type: Literal["reasoning"] = Field(description="Item discriminator.")


class _CompactionItemBase(StrictModel):
    encrypted_content: str = Field(description="Encrypted compacted context payload.")
    type: Literal["compaction"] = Field(description="Item discriminator.")


class RequestCompactionItem(_CompactionItemBase):
    id: str | None = Field(default=None, description="Optional compaction item ID.")


type RequestInputItem = Annotated[
    RequestMessageItem
    | RequestFunctionCallItem
    | RequestFunctionCallOutputItem
    | ReasoningItem
    | RequestCompactionItem,
    Field(discriminator="type"),
]


class ResponseMessageItem(StrictModel):
    content: list[OutputTextContent] = Field(
        description="Output message content blocks."
    )
    id: str = Field(description="Unique output message item ID.")
    phase: Literal["commentary", "final_answer"] | None = Field(
        default=None,
        description="Assistant phase label for commentary or final answer.",
    )
    role: Literal["assistant"] = Field(description="Output role; always assistant.")
    status: Literal["in_progress", "completed", "incomplete"] = Field(
        description="Message generation status."
    )
    type: Literal["message"] = Field(description="Output item discriminator.")


class ResponseFunctionCallItem(_FunctionCallItemBase):
    id: str = Field(description="Unique function call item ID.")


class ResponseFunctionCallOutputItem(_FunctionCallOutputItemBase):
    created_by: str | None = Field(
        default=None, description="Originator of the output."
    )
    id: str = Field(description="Unique function output item ID.")
    status: ItemStatus = Field(description="Function output item status.")


class ResponseCompactionItem(_CompactionItemBase):
    created_by: str | None = Field(
        default=None, description="Originator of compaction."
    )
    id: str = Field(description="Unique compaction item ID.")


class WebSearchActionSearchSource(StrictModel):
    type: Literal["url"] = Field(description="Web-search source discriminator.")
    url: str = Field(description="Source URL used by the web search call.")


class WebSearchActionSearch(StrictModel):
    queries: list[str] | None = Field(default=None, description="Queries used.")
    query: str = Field(description="Primary query used for the search.")
    sources: list[WebSearchActionSearchSource] | None = Field(
        default=None,
        description="Search result sources when included in the response.",
    )
    type: Literal["search"] = Field(description="Web-search action discriminator.")


class WebSearchActionOpenPage(StrictModel):
    type: Literal["open_page"] = Field(description="Web-search action discriminator.")
    url: str | None = Field(default=None, description="URL opened by web search.")


class WebSearchActionFindInPage(StrictModel):
    pattern: str = Field(description="Search pattern used within the page.")
    type: Literal["find_in_page"] = Field(
        description="Web-search action discriminator."
    )
    url: str = Field(description="URL searched within.")


type WebSearchAction = Annotated[
    WebSearchActionSearch | WebSearchActionOpenPage | WebSearchActionFindInPage,
    Field(discriminator="type"),
]


class ResponseWebSearchCallItem(StrictModel):
    action: WebSearchAction = Field(description="Specific web-search action details.")
    id: str = Field(description="Unique web search call item ID.")
    status: Literal["in_progress", "searching", "completed", "failed"] = Field(
        description="Web search call lifecycle status."
    )
    type: Literal["web_search_call"] = Field(description="Output item discriminator.")


type ResponseOutputItem = Annotated[
    ResponseMessageItem
    | ResponseFunctionCallItem
    | ResponseFunctionCallOutputItem
    | ReasoningItem
    | ResponseCompactionItem
    | ResponseWebSearchCallItem,
    Field(discriminator="type"),
]


class InputItemsMessageItem(StrictModel):
    content: str | list[MessageContentPart] = Field(
        description="Message text or supported content blocks."
    )
    id: str = Field(description="Input item ID.")
    phase: Literal["commentary", "final_answer"] | None = Field(
        default=None,
        description="Assistant phase label, when present.",
    )
    role: Literal["user", "assistant", "system", "developer"] = Field(
        description="Role that produced this input item."
    )
    status: Literal["in_progress", "completed", "incomplete"] | None = Field(
        default=None,
        description="Input item status.",
    )
    type: Literal["message"] = Field(description="Input item discriminator.")


type InputItemsPageItem = Annotated[
    InputItemsMessageItem
    | ResponseFunctionCallItem
    | ResponseFunctionCallOutputItem
    | ReasoningItem
    | ResponseCompactionItem,
    Field(discriminator="type"),
]
