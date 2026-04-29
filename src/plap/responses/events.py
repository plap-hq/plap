from __future__ import annotations

from plap.responses.contracts import (
    ReasoningItem,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseErrorEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionCallItem,
    ResponseInProgressEvent,
    ResponseMessageItem,
    ResponseObject,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
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
            response=response,
            sequence_number=0,
            type="response.created",
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

        if isinstance(item, ReasoningItem):
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
                for annotation_index, annotation in enumerate(
                    content_part.annotations
                ):
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
            response=response,
            sequence_number=0,
            type="response.completed",
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
