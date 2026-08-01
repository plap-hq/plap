from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from plap.responses.contracts.base import StrictModel
from plap.responses.contracts.items import (
    ResponseContentPart,
    ResponseOutputItem,
    SummaryTextContent,
    UrlCitationAnnotation,
)
from plap.responses.contracts.responses import ResponseObject


class ResponseCreatedEvent(StrictModel):
    response: ResponseObject = Field(description="Response snapshot for this event.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    type: Literal["response.created"] = Field(description="Stream event discriminator.")


class ResponseInProgressEvent(StrictModel):
    response: ResponseObject = Field(description="Response snapshot for this event.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    type: Literal["response.in_progress"] = Field(description="Stream event discriminator.")


class ResponseCompletedEvent(StrictModel):
    response: ResponseObject = Field(description="Final response snapshot.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    type: Literal["response.completed"] = Field(description="Stream event discriminator.")


class ResponseFailedEvent(StrictModel):
    response: ResponseObject = Field(description="Final failed response snapshot.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    type: Literal["response.failed"] = Field(description="Stream event discriminator.")


class ResponseIncompleteEvent(StrictModel):
    response: ResponseObject = Field(description="Final incomplete response snapshot.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    type: Literal["response.incomplete"] = Field(description="Stream event discriminator.")


class ResponseOutputItemAddedEvent(StrictModel):
    item: ResponseOutputItem = Field(description="Output item snapshot at add time.")
    output_index: int = Field(description="Index of this item in response.output.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    type: Literal["response.output_item.added"] = Field(description="Stream event discriminator.")


class ResponseOutputItemDoneEvent(StrictModel):
    item: ResponseOutputItem = Field(description="Completed output item.")
    output_index: int = Field(description="Index of this item in response.output.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    type: Literal["response.output_item.done"] = Field(description="Stream event discriminator.")


class ResponseContentPartAddedEvent(StrictModel):
    content_index: int = Field(description="Index within the item's content array.")
    item_id: str = Field(description="Output item ID that owns this part.")
    output_index: int = Field(description="Index of the owning output item.")
    part: ResponseContentPart = Field(description="Content part snapshot.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    type: Literal["response.content_part.added"] = Field(description="Stream event discriminator.")


class ResponseContentPartDoneEvent(StrictModel):
    content_index: int = Field(description="Index within the item's content array.")
    item_id: str = Field(description="Output item ID that owns this part.")
    output_index: int = Field(description="Index of the owning output item.")
    part: ResponseContentPart = Field(description="Completed content part.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    type: Literal["response.content_part.done"] = Field(description="Stream event discriminator.")


class ResponseTextEventLogprobTopLogprob(StrictModel):
    logprob: float | None = Field(default=None, description="Log probability for this possible token.")
    token: str | None = Field(default=None, description="Possible token text.")


class ResponseTextEventLogprob(StrictModel):
    logprob: float = Field(description="Log probability for the emitted token.")
    token: str = Field(description="Emitted token text.")
    top_logprobs: list[ResponseTextEventLogprobTopLogprob] | None = Field(
        default=None,
        description="Most likely alternative tokens at this token position.",
    )


class _ObfuscatableDeltaEvent(StrictModel):
    obfuscation: str | None = Field(
        default=None,
        description="Optional random padding used to normalize streamed delta payload sizes.",
    )


class ResponseTextDeltaEvent(_ObfuscatableDeltaEvent):
    content_index: int = Field(description="Index within the message content array.")
    delta: str = Field(description="Incremental output text chunk.")
    item_id: str = Field(description="Message item ID receiving this delta.")
    logprobs: list[ResponseTextEventLogprob] = Field(
        default_factory=list,
        description="Token logprobs for this delta when requested.",
    )
    output_index: int = Field(description="Index of the message in response.output.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    type: Literal["response.output_text.delta"] = Field(description="Stream event discriminator.")


class ResponseTextDoneEvent(StrictModel):
    content_index: int = Field(description="Index within the message content array.")
    item_id: str = Field(description="Message item ID that owns this text.")
    logprobs: list[ResponseTextEventLogprob] = Field(
        default_factory=list,
        description="Token logprobs for the completed text when requested.",
    )
    output_index: int = Field(description="Index of the message in response.output.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    text: str = Field(description="Completed output text.")
    type: Literal["response.output_text.done"] = Field(description="Stream event discriminator.")


class ResponseOutputTextAnnotationAddedEvent(StrictModel):
    annotation: UrlCitationAnnotation = Field(description="Annotation added to text.")
    annotation_index: int = Field(description="Index within the annotations array.")
    content_index: int = Field(description="Index within the message content array.")
    item_id: str = Field(description="Message item ID that owns the annotation.")
    output_index: int = Field(description="Index of the message in response.output.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    type: Literal["response.output_text.annotation.added"] = Field(description="Stream event discriminator.")


class ResponseRefusalDeltaEvent(_ObfuscatableDeltaEvent):
    content_index: int = Field(description="Index within the message content array.")
    delta: str = Field(description="Incremental refusal text chunk.")
    item_id: str = Field(description="Message item ID receiving this refusal delta.")
    output_index: int = Field(description="Index of the message in response.output.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    type: Literal["response.refusal.delta"] = Field(description="Stream event discriminator.")


class ResponseRefusalDoneEvent(StrictModel):
    content_index: int = Field(description="Index within the message content array.")
    item_id: str = Field(description="Message item ID that owns this refusal.")
    output_index: int = Field(description="Index of the message in response.output.")
    refusal: str = Field(description="Completed refusal text.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    type: Literal["response.refusal.done"] = Field(description="Stream event discriminator.")


class ResponseFunctionCallArgumentsDeltaEvent(_ObfuscatableDeltaEvent):
    delta: str = Field(description="Incremental JSON argument string chunk.")
    item_id: str = Field(description="Function call item ID receiving this delta.")
    output_index: int = Field(description="Index of the function call in response.output.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    type: Literal["response.function_call_arguments.delta"] = Field(description="Stream event discriminator.")


class ResponseFunctionCallArgumentsDoneEvent(StrictModel):
    arguments: str = Field(description="Completed JSON argument string.")
    item_id: str = Field(description="Function call item ID.")
    name: str = Field(description="Function name selected by the model.")
    output_index: int = Field(description="Index of the function call in response.output.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    type: Literal["response.function_call_arguments.done"] = Field(description="Stream event discriminator.")


class ResponseReasoningSummaryPartAddedEvent(StrictModel):
    item_id: str = Field(description="Reasoning item ID that owns this summary.")
    output_index: int = Field(description="Index of the reasoning item in response.output.")
    part: SummaryTextContent = Field(description="Reasoning summary part snapshot.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    summary_index: int = Field(description="Index within the reasoning summaries.")
    type: Literal["response.reasoning_summary_part.added"] = Field(description="Stream event discriminator.")


class ResponseReasoningSummaryPartDoneEvent(StrictModel):
    item_id: str = Field(description="Reasoning item ID that owns this summary.")
    output_index: int = Field(description="Index of the reasoning item in response.output.")
    part: SummaryTextContent = Field(description="Completed reasoning summary part.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    summary_index: int = Field(description="Index within the reasoning summaries.")
    type: Literal["response.reasoning_summary_part.done"] = Field(description="Stream event discriminator.")


class ResponseReasoningSummaryTextDeltaEvent(_ObfuscatableDeltaEvent):
    delta: str = Field(description="Incremental reasoning summary text chunk.")
    item_id: str = Field(description="Reasoning item ID receiving this delta.")
    output_index: int = Field(description="Index of the reasoning item in response.output.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    summary_index: int = Field(description="Index within the reasoning summaries.")
    type: Literal["response.reasoning_summary_text.delta"] = Field(description="Stream event discriminator.")


class ResponseReasoningSummaryTextDoneEvent(StrictModel):
    item_id: str = Field(description="Reasoning item ID.")
    output_index: int = Field(description="Index of the reasoning item in response.output.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    summary_index: int = Field(description="Index within the reasoning summaries.")
    text: str = Field(description="Completed reasoning summary text.")
    type: Literal["response.reasoning_summary_text.done"] = Field(description="Stream event discriminator.")


class ResponseReasoningTextDeltaEvent(_ObfuscatableDeltaEvent):
    content_index: int = Field(description="Index within the reasoning content array.")
    delta: str = Field(description="Incremental reasoning text chunk.")
    item_id: str = Field(description="Reasoning item ID receiving this delta.")
    output_index: int = Field(description="Index of the reasoning item in response.output.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    type: Literal["response.reasoning_text.delta"] = Field(description="Stream event discriminator.")


class ResponseReasoningTextDoneEvent(StrictModel):
    content_index: int = Field(description="Index within the reasoning content array.")
    item_id: str = Field(description="Reasoning item ID.")
    output_index: int = Field(description="Index of the reasoning item in response.output.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    text: str = Field(description="Completed reasoning text.")
    type: Literal["response.reasoning_text.done"] = Field(description="Stream event discriminator.")


# OpenAI exposes response.web_search_call.* stream events. plap emits web search
# as ordinary function_call/function_call_output events instead.
# class ResponseWebSearchCallInProgressEvent(StrictModel): ...
# class ResponseWebSearchCallSearchingEvent(StrictModel): ...
# class ResponseWebSearchCallCompletedEvent(StrictModel): ...


class ResponseErrorEvent(StrictModel):
    code: str | None = Field(default=None, description="Machine-readable error code.")
    message: str = Field(description="Human-readable error message.")
    param: str | None = Field(default=None, description="Related parameter when known.")
    sequence_number: int = Field(description="Monotonic stream event sequence number.")
    type: Literal["error"] = Field(description="Stream event discriminator.")


type ResponseStreamEvent = Annotated[
    ResponseCreatedEvent
    | ResponseInProgressEvent
    | ResponseCompletedEvent
    | ResponseFailedEvent
    | ResponseIncompleteEvent
    | ResponseOutputItemAddedEvent
    | ResponseOutputItemDoneEvent
    | ResponseContentPartAddedEvent
    | ResponseContentPartDoneEvent
    | ResponseTextDeltaEvent
    | ResponseTextDoneEvent
    | ResponseOutputTextAnnotationAddedEvent
    | ResponseRefusalDeltaEvent
    | ResponseRefusalDoneEvent
    | ResponseFunctionCallArgumentsDeltaEvent
    | ResponseFunctionCallArgumentsDoneEvent
    | ResponseReasoningSummaryPartAddedEvent
    | ResponseReasoningSummaryPartDoneEvent
    | ResponseReasoningSummaryTextDeltaEvent
    | ResponseReasoningSummaryTextDoneEvent
    | ResponseReasoningTextDeltaEvent
    | ResponseReasoningTextDoneEvent
    # | ResponseWebSearchCallInProgressEvent
    # | ResponseWebSearchCallSearchingEvent
    # | ResponseWebSearchCallCompletedEvent
    | ResponseErrorEvent,
    Field(discriminator="type"),
]
