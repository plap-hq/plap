from __future__ import annotations

import secrets
import time
from collections.abc import Sequence
from enum import StrEnum
from typing import cast

import anyio
from anyio.abc import ObjectSendStream, TaskGroup

from plap.llms.chat import ReasoningEffort, ServiceTier
from plap.responses.contracts import (
    ConversationReference,
    ReasoningItem,
    ReasoningSummary,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseCreateRequest,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionCallItem,
    ResponseIncompleteDetails,
    ResponseInProgressEvent,
    ResponseMessageItem,
    ResponseObject,
    ResponseOutputItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseStatus,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseUsage,
    SummaryTextContent,
)
from plap.responses.models import ReasoningMessagePatch, Side, StateMessage
from plap.responses.reasoning import IReasoningSummarizer
from plap.responses.store import PreparedRequest, ResponseStore


class _CommitKind(StrEnum):
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    OUTPUT = "output"
    COMPLETED = "completed"


type _ReasoningMessages = tuple[StateMessage | ReasoningMessagePatch, ...]
type _ReasoningMetadata = tuple[Side, _ReasoningMessages]
type _OutputMetadata = _ReasoningMetadata | None
type _Commit = tuple[_CommitKind, ResponseObject | ResponseOutputItem, _OutputMetadata]


class ResponseEventIO:
    def __init__(
        self,
        *,
        request: ResponseCreateRequest,
        prepared: PreparedRequest,
        response_store: ResponseStore,
        send: ObjectSendStream[ResponseStreamEvent],
        reasoning_summarizer: IReasoningSummarizer,
        reasoning_summarizer_model: str,
        reasoning_summarizer_prompt_cache_key_base: str | None,
        reasoning_summarizer_reasoning_effort: ReasoningEffort | None,
        reasoning_summarizer_service_tier: ServiceTier | None,
        reasoning_summary_mode: ReasoningSummary | None,
    ) -> None:
        self._response = _response_object(request, status="in_progress")
        self._prepared = prepared
        self._response_store = response_store
        self._send = send
        self._reasoning_summarizer = reasoning_summarizer
        self._reasoning_summarizer_model = reasoning_summarizer_model
        self._reasoning_summarizer_prompt_cache_key_base = reasoning_summarizer_prompt_cache_key_base
        self._reasoning_summarizer_reasoning_effort = reasoning_summarizer_reasoning_effort
        self._reasoning_summarizer_service_tier = reasoning_summarizer_service_tier
        self._reasoning_summary_mode = reasoning_summary_mode
        self._commit_send, self._commit_receive = anyio.create_memory_object_stream[_Commit](16)
        self._output_items: list[ResponseOutputItem] = []
        self._sequence_number = 0

    def start(self, task_group: TaskGroup) -> None:
        task_group.start_soon(self._emit_commits)

    async def created(self) -> None:
        await self._commit_send.send((_CommitKind.CREATED, self._response, None))

    async def in_progress(self) -> None:
        await self._commit_send.send((_CommitKind.IN_PROGRESS, self._response, None))

    async def output(
        self,
        item: ResponseOutputItem,
        *,
        reasoning_side: Side | None = None,
        reasoning_messages: Sequence[StateMessage | ReasoningMessagePatch] | None = None,
    ) -> None:
        metadata: _OutputMetadata = None
        if reasoning_messages is not None:
            if not isinstance(item, ReasoningItem):
                raise TypeError("reasoning_messages can only be attached to reasoning items")
            if self._reasoning_summary_mode is not None:
                if reasoning_side is None:
                    raise TypeError("reasoning_side is required for reasoning messages")
                metadata = (reasoning_side, tuple(reasoning_messages))
        await self._commit_send.send((_CommitKind.OUTPUT, item, metadata))

    async def completed(
        self,
        *,
        service_tier: str | None = None,
        usage: ResponseUsage | None = None,
    ) -> None:
        response = self._response.model_copy(
            update={
                "completed_at": time.time(),
                "service_tier": service_tier or self._response.service_tier,
                "status": "completed",
                "usage": usage,
            }
        )
        await self._commit_send.send((_CommitKind.COMPLETED, response, None))

    async def incomplete(
        self,
        *,
        service_tier: str | None = None,
        usage: ResponseUsage | None = None,
    ) -> None:
        response = self._response.model_copy(
            update={
                "completed_at": time.time(),
                "service_tier": service_tier or self._response.service_tier,
                "status": "incomplete",
                "incomplete_details": ResponseIncompleteDetails(reason="max_output_tokens"),
                "usage": usage,
            }
        )
        await self._commit_send.send((_CommitKind.COMPLETED, response, None))

    async def aclose(self) -> None:
        await self._commit_send.aclose()

    async def _emit_commits(self) -> None:
        async with self._commit_receive:
            async for kind, value, metadata in self._commit_receive:
                if kind == _CommitKind.CREATED:
                    await self._response_store.begin_response(self._prepared, cast(ResponseObject, value))
                    await self._send_event(
                        ResponseCreatedEvent(
                            response=cast(ResponseObject, value),
                            sequence_number=0,
                            type="response.created",
                        )
                    )
                elif kind == _CommitKind.IN_PROGRESS:
                    await self._send_event(
                        ResponseInProgressEvent(
                            response=cast(ResponseObject, value),
                            sequence_number=0,
                            type="response.in_progress",
                        )
                    )
                elif kind == _CommitKind.OUTPUT:
                    await self._emit_output(cast(ResponseOutputItem, value), metadata)
                else:
                    response = cast(ResponseObject, value).model_copy(update={"output": self._output_items})
                    await self._response_store.finish_response(self._prepared, response)
                    await self._send_event(
                        ResponseCompletedEvent(
                            response=response,
                            sequence_number=0,
                            type="response.completed",
                        )
                    )

    async def _send_event(self, event: ResponseStreamEvent) -> None:
        self._sequence_number += 1
        payload = event.model_dump(mode="json")
        payload["sequence_number"] = self._sequence_number
        await self._send.send(type(event).model_validate(payload))

    async def _emit_output(
        self,
        item: ResponseOutputItem,
        metadata: _OutputMetadata,
    ) -> None:
        output_index = len(self._output_items)
        if isinstance(item, ReasoningItem) and metadata is not None:
            completed_item = await self._emit_reasoning_with_summary(
                item,
                metadata,
                output_index=output_index,
            )
        else:
            completed_item = item
            await self._emit_output_item_events(item, output_index=output_index)
        self._output_items.append(completed_item)
        await self._response_store.append_output_item(
            self._prepared,
            self._response.id,
            output_index,
            completed_item.model_dump(mode="json", exclude_none=True),
        )

    async def _emit_output_item_events(
        self,
        item: ResponseOutputItem,
        *,
        output_index: int,
    ) -> None:
        await self._send_event(
            ResponseOutputItemAddedEvent(
                item=item,
                output_index=output_index,
                sequence_number=0,
                type="response.output_item.added",
            )
        )
        if isinstance(item, ReasoningItem):
            await self._emit_reasoning_events(item, output_index=output_index)
        if isinstance(item, ResponseFunctionCallItem):
            await self._send_event(
                ResponseFunctionCallArgumentsDeltaEvent(
                    delta=item.arguments,
                    item_id=item.id,
                    output_index=output_index,
                    sequence_number=0,
                    type="response.function_call_arguments.delta",
                )
            )
            await self._send_event(
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
            await self._emit_message_events(item, output_index=output_index)
        await self._send_event(
            ResponseOutputItemDoneEvent(
                item=item,
                output_index=output_index,
                sequence_number=0,
                type="response.output_item.done",
            )
        )

    async def _emit_reasoning_with_summary(
        self,
        item: ReasoningItem,
        metadata: _ReasoningMetadata,
        *,
        output_index: int,
    ) -> ReasoningItem:
        side, messages = metadata
        added_item = item.model_copy(update={"status": "in_progress", "summary": []})
        await self._send_event(
            ResponseOutputItemAddedEvent(
                item=added_item,
                output_index=output_index,
                sequence_number=0,
                type="response.output_item.added",
            )
        )

        summary_text = ""
        summary_index = 0
        await self._send_event(
            ResponseReasoningSummaryPartAddedEvent(
                item_id=item.id,
                output_index=output_index,
                part=SummaryTextContent(text="", type="summary_text"),
                sequence_number=0,
                summary_index=summary_index,
                type="response.reasoning_summary_part.added",
            )
        )
        try:
            async for delta in self._reasoning_summarizer.stream(
                model=self._reasoning_summarizer_model,
                prompt_cache_key=self._reasoning_summarizer_prompt_cache_key(),
                reasoning_effort=self._reasoning_summarizer_reasoning_effort,
                service_tier=self._reasoning_summarizer_service_tier,
                mode=cast(ReasoningSummary, self._reasoning_summary_mode),
                side=side,
                messages=messages,
            ):
                summary_text += delta
                await self._send_event(
                    ResponseReasoningSummaryTextDeltaEvent(
                        delta=delta,
                        item_id=item.id,
                        output_index=output_index,
                        sequence_number=0,
                        summary_index=summary_index,
                        type="response.reasoning_summary_text.delta",
                    )
                )
        except Exception:
            summary_text = ""

        summary_part = SummaryTextContent(text=summary_text, type="summary_text")
        await self._send_event(
            ResponseReasoningSummaryTextDoneEvent(
                item_id=item.id,
                output_index=output_index,
                sequence_number=0,
                summary_index=summary_index,
                text=summary_text,
                type="response.reasoning_summary_text.done",
            )
        )
        await self._send_event(
            ResponseReasoningSummaryPartDoneEvent(
                item_id=item.id,
                output_index=output_index,
                part=summary_part,
                sequence_number=0,
                summary_index=summary_index,
                type="response.reasoning_summary_part.done",
            )
        )

        completed_item = item.model_copy(update={"summary": [summary_part]})
        await self._send_event(
            ResponseOutputItemDoneEvent(
                item=completed_item,
                output_index=output_index,
                sequence_number=0,
                type="response.output_item.done",
            )
        )
        return completed_item

    def _reasoning_summarizer_prompt_cache_key(self) -> str | None:
        if self._reasoning_summarizer_prompt_cache_key_base is None:
            return None
        return f"{self._reasoning_summarizer_prompt_cache_key_base}|reasoning_summarizer"

    async def _emit_reasoning_events(
        self,
        item: ReasoningItem,
        *,
        output_index: int,
    ) -> None:
        for summary_index, summary_part in enumerate(item.summary):
            await self._send_event(
                ResponseReasoningSummaryPartAddedEvent(
                    item_id=item.id,
                    output_index=output_index,
                    part=summary_part,
                    sequence_number=0,
                    summary_index=summary_index,
                    type="response.reasoning_summary_part.added",
                )
            )
            await self._send_event(
                ResponseReasoningSummaryTextDeltaEvent(
                    delta=summary_part.text,
                    item_id=item.id,
                    output_index=output_index,
                    sequence_number=0,
                    summary_index=summary_index,
                    type="response.reasoning_summary_text.delta",
                )
            )
            await self._send_event(
                ResponseReasoningSummaryTextDoneEvent(
                    item_id=item.id,
                    output_index=output_index,
                    sequence_number=0,
                    summary_index=summary_index,
                    text=summary_part.text,
                    type="response.reasoning_summary_text.done",
                )
            )
            await self._send_event(
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
            await self._send_event(
                ResponseContentPartAddedEvent(
                    content_index=content_index,
                    item_id=item.id,
                    output_index=output_index,
                    part=content_part,
                    sequence_number=0,
                    type="response.content_part.added",
                )
            )
            await self._send_event(
                ResponseReasoningTextDeltaEvent(
                    content_index=content_index,
                    delta=content_part.text,
                    item_id=item.id,
                    output_index=output_index,
                    sequence_number=0,
                    type="response.reasoning_text.delta",
                )
            )
            await self._send_event(
                ResponseReasoningTextDoneEvent(
                    content_index=content_index,
                    item_id=item.id,
                    output_index=output_index,
                    sequence_number=0,
                    text=content_part.text,
                    type="response.reasoning_text.done",
                )
            )
            await self._send_event(
                ResponseContentPartDoneEvent(
                    content_index=content_index,
                    item_id=item.id,
                    output_index=output_index,
                    part=content_part,
                    sequence_number=0,
                    type="response.content_part.done",
                )
            )

    async def _emit_message_events(
        self,
        item: ResponseMessageItem,
        *,
        output_index: int,
    ) -> None:
        for content_index, content_part in enumerate(item.content):
            await self._send_event(
                ResponseContentPartAddedEvent(
                    content_index=content_index,
                    item_id=item.id,
                    output_index=output_index,
                    part=content_part,
                    sequence_number=0,
                    type="response.content_part.added",
                )
            )
            for annotation_index, annotation in enumerate(content_part.annotations):
                await self._send_event(
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
            await self._send_event(
                ResponseTextDeltaEvent(
                    content_index=content_index,
                    delta=content_part.text,
                    item_id=item.id,
                    output_index=output_index,
                    sequence_number=0,
                    type="response.output_text.delta",
                )
            )
            await self._send_event(
                ResponseTextDoneEvent(
                    content_index=content_index,
                    item_id=item.id,
                    output_index=output_index,
                    sequence_number=0,
                    text=content_part.text,
                    type="response.output_text.done",
                )
            )
            await self._send_event(
                ResponseContentPartDoneEvent(
                    content_index=content_index,
                    item_id=item.id,
                    output_index=output_index,
                    part=content_part,
                    sequence_number=0,
                    type="response.content_part.done",
                )
            )


def _response_object(
    request: ResponseCreateRequest,
    *,
    response_id: str | None = None,
    status: ResponseStatus = "completed",
    usage: ResponseUsage | None = None,
) -> ResponseObject:
    created_at = time.time()
    return ResponseObject(
        completed_at=created_at if status in {"completed", "cancelled"} else None,
        conversation=_response_conversation(request),
        created_at=created_at,
        id=response_id or f"resp_{secrets.token_urlsafe(18)}",
        instructions=request.instructions,
        max_output_tokens=request.max_output_tokens,
        max_tool_calls=request.max_tool_calls,
        metadata=request.metadata,
        model=request.model,
        output=[],
        parallel_tool_calls=request.parallel_tool_calls,
        previous_response_id=request.previous_response_id,
        prompt=request.prompt,
        prompt_cache_key=request.prompt_cache_key,
        reasoning=request.reasoning,
        safety_identifier=request.safety_identifier,
        service_tier=request.service_tier,
        status=status,
        temperature=request.temperature,
        text=request.text,
        tool_choice=request.tool_choice,
        tools=request.tools,
        top_logprobs=request.top_logprobs,
        top_p=request.top_p,
        truncation=request.truncation,
        usage=usage,
        user=request.user,
    )


def _response_conversation(
    request: ResponseCreateRequest,
) -> ConversationReference | None:
    if request.conversation is None:
        return None
    if isinstance(request.conversation, str):
        return ConversationReference(id=request.conversation)
    return request.conversation
