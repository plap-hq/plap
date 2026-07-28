from __future__ import annotations

import secrets
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from litestar.channels import ChannelsPlugin

from plap.errors import PublicError
from plap.keyring import SealingKeyring
from plap.responses.contracts import (
    ConversationReference,
    OutputRefusalContent,
    OutputTextContent,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseCreateRequest,
    ResponseErrorEvent,
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
    ResponseReasoningItem,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseRefusalDeltaEvent,
    ResponseRefusalDoneEvent,
    ResponseStatus,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseUsage,
    SummaryTextContent,
)
from plap.responses.ingest.models import ReasoningPayload, SidesUpdate
from plap.responses.ingest.patch import JSONPatch
from plap.responses.ingest.sealing import seal_reasoning_payload
from plap.responses.store import PreparedRequest, ResponseStore
from plap.responses.summary import SummaryDelta, SummaryDone


def channel_name(response_id: str) -> str:
    return f"response:{response_id}"


def _conversation(request: ResponseCreateRequest) -> ConversationReference | None:
    if request.conversation is None:
        return None
    if isinstance(request.conversation, str):
        return ConversationReference(id=request.conversation)
    return request.conversation


def _new_response(
    request: ResponseCreateRequest,
    *,
    response_id: str | None = None,
    status: ResponseStatus = "in_progress",
    usage: ResponseUsage | None = None,
) -> ResponseObject:
    created_at = time.time()
    completed_at = created_at if status in {"completed", "failed", "cancelled", "incomplete"} else None
    return ResponseObject(
        completed_at=completed_at,
        conversation=_conversation(request),
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


def _new_reasoning_id() -> str:
    return f"rs_{secrets.token_urlsafe(18)}"


@dataclass(frozen=True, slots=True)
class _Lineage:
    item_id: str
    previous_reasoning_id: str | None


@dataclass(slots=True)
class _Chain:
    last_reasoning_id: str | None = None

    def next_reasoning(self) -> _Lineage:
        return _Lineage(
            item_id=_new_reasoning_id(),
            previous_reasoning_id=self.last_reasoning_id,
        )

    def record_reasoning(self, item_id: str) -> None:
        self.last_reasoning_id = item_id


@dataclass(slots=True)
class _Draft:
    output_index: int
    lineage: _Lineage
    machine: JSONPatch
    sides: SidesUpdate
    summary_parts: list[SummaryTextContent] = field(default_factory=list)
    summary_pending: str = ""


class StreamCoordinator:
    def __init__(
        self,
        *,
        request: ResponseCreateRequest,
        channels: ChannelsPlugin,
        prepared: PreparedRequest | None = None,
        response_store: ResponseStore | None = None,
        response_id: str | None = None,
        sealing_keyring: SealingKeyring | None = None,
        last_reasoning_id: str | None = None,
    ) -> None:
        if (prepared is None) != (response_store is None):
            raise ValueError("prepared and response_store must either both be provided or both be omitted")
        self._channels = channels
        self._prepared = prepared
        self._response_store = response_store
        self._sealing_keyring = sealing_keyring
        self._response = _new_response(request, response_id=response_id)
        self._items: list[ResponseOutputItem] = []
        self._chain = _Chain(last_reasoning_id=last_reasoning_id)
        self._draft: _Draft | None = None
        self._sequence_number = 0

    @property
    def response_id(self) -> str:
        return self._response.id

    @property
    def channel(self) -> str:
        return channel_name(self._response.id)

    def current_response(self) -> ResponseObject:
        return self._response.model_copy(update={"output": list(self._items)})

    def _terminal_response(
        self,
        status: ResponseStatus,
        *,
        service_tier: str | None = None,
        usage: ResponseUsage | None = None,
        incomplete_reason: str | None = None,
    ) -> ResponseObject:
        updates: dict[str, object] = {
            "completed_at": time.time(),
            "output": list(self._items),
            "service_tier": service_tier or self._response.service_tier,
            "status": status,
            "usage": usage,
            "incomplete_details": None,
        }
        if incomplete_reason is not None:
            updates["incomplete_details"] = ResponseIncompleteDetails(reason=incomplete_reason)
        return self._response.model_copy(update=updates)

    def _require_sealing_keyring(self) -> SealingKeyring:
        if self._sealing_keyring is None:
            raise RuntimeError("stream coordinator requires a sealing keyring for reasoning items")
        return self._sealing_keyring

    async def _publish(self, event: object) -> None:
        self._sequence_number += 1
        payload = event.model_copy(update={"sequence_number": self._sequence_number}).model_dump(mode="json", exclude_none=True)
        await self._channels.wait_published(payload, self.channel)

    async def _append_item(self, item: ResponseOutputItem) -> int:
        output_index = len(self._items)
        if self._response_store is not None and self._prepared is not None:
            await self._response_store.append_output_item(
                self._prepared,
                self._response.id,
                output_index,
                item.model_dump(mode="json", exclude_none=True),
            )
        self._items.append(item)
        return output_index

    async def _replace_item(self, *, output_index: int, item: ResponseOutputItem) -> None:
        if self._response_store is not None and self._prepared is not None:
            await self._response_store.replace_output_item(
                self._prepared,
                self._response.id,
                output_index,
                item.model_dump(mode="json", exclude_none=True),
            )
        self._items[output_index] = item

    def _active_draft(self) -> _Draft:
        if self._draft is None:
            raise RuntimeError("reasoning item is not active")
        return self._draft

    def _active_reasoning_item(self) -> ResponseReasoningItem:
        draft = self._active_draft()
        item = self._items[draft.output_index]
        if not isinstance(item, ResponseReasoningItem):
            raise TypeError("active output item is not a reasoning item")
        return item

    def _assert_no_pending_summary(self, draft: _Draft) -> None:
        if draft.summary_pending:
            raise RuntimeError("cannot finish reasoning item while summary text is still pending")

    def _reasoning_item(
        self,
        *,
        lineage: _Lineage,
        machine: JSONPatch,
        sides: SidesUpdate,
        summary: Sequence[SummaryTextContent],
        status: str,
    ) -> ResponseReasoningItem:
        payload = ReasoningPayload(
            id=lineage.item_id,
            previous_reasoning_id=lineage.previous_reasoning_id,
            machine=machine,
            sides=sides,
        )
        return ResponseReasoningItem(
            encrypted_content=seal_reasoning_payload(payload, keyring=self._require_sealing_keyring()),
            id=lineage.item_id,
            status=status,
            summary=list(summary),
            type="reasoning",
        )

    def _draft_item(self, *, draft: _Draft, status: str) -> ResponseReasoningItem:
        return self._reasoning_item(
            lineage=draft.lineage,
            machine=draft.machine,
            sides=draft.sides,
            summary=list(draft.summary_parts),
            status=status,
        )

    async def _emit_message_events(self, item: ResponseMessageItem, *, output_index: int) -> None:
        for content_index, content_part in enumerate(item.content):
            await self._publish(
                ResponseContentPartAddedEvent(
                    content_index=content_index,
                    item_id=item.id,
                    output_index=output_index,
                    part=content_part,
                    sequence_number=0,
                    type="response.content_part.added",
                )
            )
            if isinstance(content_part, OutputTextContent):
                for annotation_index, annotation in enumerate(content_part.annotations):
                    await self._publish(
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
                await self._publish(
                    ResponseTextDeltaEvent(
                        content_index=content_index,
                        delta=content_part.text,
                        item_id=item.id,
                        output_index=output_index,
                        sequence_number=0,
                        type="response.output_text.delta",
                    )
                )
                await self._publish(
                    ResponseTextDoneEvent(
                        content_index=content_index,
                        item_id=item.id,
                        output_index=output_index,
                        sequence_number=0,
                        text=content_part.text,
                        type="response.output_text.done",
                    )
                )
            elif isinstance(content_part, OutputRefusalContent):
                await self._publish(
                    ResponseRefusalDeltaEvent(
                        content_index=content_index,
                        delta=content_part.refusal,
                        item_id=item.id,
                        output_index=output_index,
                        sequence_number=0,
                        type="response.refusal.delta",
                    )
                )
                await self._publish(
                    ResponseRefusalDoneEvent(
                        content_index=content_index,
                        item_id=item.id,
                        output_index=output_index,
                        refusal=content_part.refusal,
                        sequence_number=0,
                        type="response.refusal.done",
                    )
                )
            await self._publish(
                ResponseContentPartDoneEvent(
                    content_index=content_index,
                    item_id=item.id,
                    output_index=output_index,
                    part=content_part,
                    sequence_number=0,
                    type="response.content_part.done",
                )
            )

    async def _emit_reasoning_events(self, item: ResponseReasoningItem, *, output_index: int) -> None:
        for summary_index, summary_part in enumerate(item.summary):
            await self._publish(
                ResponseReasoningSummaryPartAddedEvent(
                    item_id=item.id,
                    output_index=output_index,
                    part=summary_part,
                    sequence_number=0,
                    summary_index=summary_index,
                    type="response.reasoning_summary_part.added",
                )
            )
            await self._publish(
                ResponseReasoningSummaryTextDeltaEvent(
                    delta=summary_part.text,
                    item_id=item.id,
                    output_index=output_index,
                    sequence_number=0,
                    summary_index=summary_index,
                    type="response.reasoning_summary_text.delta",
                )
            )
            await self._publish(
                ResponseReasoningSummaryTextDoneEvent(
                    item_id=item.id,
                    output_index=output_index,
                    sequence_number=0,
                    summary_index=summary_index,
                    text=summary_part.text,
                    type="response.reasoning_summary_text.done",
                )
            )
            await self._publish(
                ResponseReasoningSummaryPartDoneEvent(
                    item_id=item.id,
                    output_index=output_index,
                    part=summary_part,
                    sequence_number=0,
                    summary_index=summary_index,
                    type="response.reasoning_summary_part.done",
                )
            )

    async def _emit_item_events(self, item: ResponseOutputItem, *, output_index: int) -> None:
        await self._publish(
            ResponseOutputItemAddedEvent(item=item, output_index=output_index, sequence_number=0, type="response.output_item.added")
        )
        if isinstance(item, ResponseReasoningItem):
            await self._emit_reasoning_events(item, output_index=output_index)
        if isinstance(item, ResponseFunctionCallItem):
            await self._publish(
                ResponseFunctionCallArgumentsDeltaEvent(
                    delta=item.arguments,
                    item_id=item.id,
                    output_index=output_index,
                    sequence_number=0,
                    type="response.function_call_arguments.delta",
                )
            )
            await self._publish(
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
        await self._publish(
            ResponseOutputItemDoneEvent(item=item, output_index=output_index, sequence_number=0, type="response.output_item.done")
        )

    async def created(self) -> None:
        response = self.current_response()
        if self._response_store is not None and self._prepared is not None:
            await self._response_store.begin_response(self._prepared, response)
        await self._publish(ResponseCreatedEvent(response=response, sequence_number=0, type="response.created"))

    async def in_progress(self) -> None:
        await self._publish(ResponseInProgressEvent(response=self.current_response(), sequence_number=0, type="response.in_progress"))

    async def emit(self, item: ResponseOutputItem) -> None:
        output_index = await self._append_item(item)
        await self._emit_item_events(item, output_index=output_index)

    async def begin_reasoning(
        self,
        *,
        machine: JSONPatch,
        sides: SidesUpdate,
    ) -> str:
        if self._draft is not None:
            raise RuntimeError("reasoning item is already active")
        lineage = self._chain.next_reasoning()
        item = self._reasoning_item(
            lineage=lineage,
            machine=machine,
            sides=sides,
            summary=(),
            status="in_progress",
        )
        output_index = await self._append_item(item)
        self._draft = _Draft(
            output_index=output_index,
            lineage=lineage,
            machine=machine,
            sides=sides,
        )
        await self._publish(
            ResponseOutputItemAddedEvent(item=item, output_index=output_index, sequence_number=0, type="response.output_item.added")
        )
        return item.id

    async def replace_reasoning(
        self,
        *,
        machine: JSONPatch,
        sides: SidesUpdate,
    ) -> None:
        draft = self._active_draft()
        draft.machine = machine
        draft.sides = sides
        item = self._draft_item(draft=draft, status="in_progress")
        await self._replace_item(output_index=draft.output_index, item=item)

    async def summary_delta(self, update: SummaryDelta) -> None:
        draft = self._active_draft()
        item = self._active_reasoning_item()
        summary_index = len(draft.summary_parts)
        if not draft.summary_pending:
            await self._publish(
                ResponseReasoningSummaryPartAddedEvent(
                    item_id=item.id,
                    output_index=draft.output_index,
                    part=SummaryTextContent(text="", type="summary_text"),
                    sequence_number=0,
                    summary_index=summary_index,
                    type="response.reasoning_summary_part.added",
                )
            )
        draft.summary_pending += update.text
        await self._publish(
            ResponseReasoningSummaryTextDeltaEvent(
                delta=update.text,
                item_id=item.id,
                output_index=draft.output_index,
                sequence_number=0,
                summary_index=summary_index,
                type="response.reasoning_summary_text.delta",
            )
        )

    async def summary_done(self, update: SummaryDone) -> None:
        draft = self._active_draft()
        item = self._active_reasoning_item()
        summary_index = len(draft.summary_parts)
        summary_text = draft.summary_pending
        draft.summary_pending = ""
        summary_part = SummaryTextContent(text=summary_text, type="summary_text")
        draft.summary_parts.append(summary_part)
        self._items[draft.output_index] = self._draft_item(draft=draft, status="in_progress")
        await self._publish(
            ResponseReasoningSummaryTextDoneEvent(
                item_id=item.id,
                output_index=draft.output_index,
                sequence_number=0,
                summary_index=summary_index,
                text=summary_text,
                type="response.reasoning_summary_text.done",
            )
        )
        await self._publish(
            ResponseReasoningSummaryPartDoneEvent(
                item_id=item.id,
                output_index=draft.output_index,
                part=summary_part,
                sequence_number=0,
                summary_index=summary_index,
                type="response.reasoning_summary_part.done",
            )
        )

    async def finish_reasoning(
        self,
        *,
        machine: JSONPatch,
        sides: SidesUpdate,
    ) -> str:
        draft = self._active_draft()
        self._assert_no_pending_summary(draft)
        draft.machine = machine
        draft.sides = sides
        item = self._draft_item(draft=draft, status="completed")
        await self._replace_item(output_index=draft.output_index, item=item)
        await self._publish(
            ResponseOutputItemDoneEvent(
                item=item,
                output_index=draft.output_index,
                sequence_number=0,
                type="response.output_item.done",
            )
        )
        self._chain.record_reasoning(draft.lineage.item_id)
        self._draft = None
        return item.id

    async def completed(
        self,
        *,
        service_tier: str | None = None,
        usage: ResponseUsage | None = None,
    ) -> None:
        if self._draft is not None:
            raise RuntimeError("cannot complete response while a reasoning item is active")
        self._response = self._terminal_response("completed", service_tier=service_tier, usage=usage)
        if self._response_store is not None and self._prepared is not None:
            await self._response_store.finish_response(self._prepared, self._response)
        await self._publish(ResponseCompletedEvent(response=self._response, sequence_number=0, type="response.completed"))

    async def incomplete(
        self,
        *,
        service_tier: str | None = None,
        usage: ResponseUsage | None = None,
    ) -> None:
        if self._draft is not None:
            raise RuntimeError("cannot mark response incomplete while a reasoning item is active")
        self._response = self._terminal_response(
            "incomplete",
            service_tier=service_tier,
            usage=usage,
            incomplete_reason="max_output_tokens",
        )
        if self._response_store is not None and self._prepared is not None:
            await self._response_store.finish_response(self._prepared, self._response)
        await self._publish(ResponseCompletedEvent(response=self._response, sequence_number=0, type="response.completed"))

    async def cancelled(
        self,
        *,
        service_tier: str | None = None,
        usage: ResponseUsage | None = None,
    ) -> None:
        draft = self._draft
        if draft is not None:
            item = self._draft_item(draft=draft, status="in_progress")
            await self._replace_item(output_index=draft.output_index, item=item)
            self._draft = None
        self._response = self._terminal_response("cancelled", service_tier=service_tier, usage=usage)
        if self._response_store is not None and self._prepared is not None:
            await self._response_store.cancel_response(self._prepared, self._response)
        await self._publish(ResponseCompletedEvent(response=self._response, sequence_number=0, type="response.completed"))

    async def fail(self, public: PublicError) -> None:
        draft = self._draft
        if draft is not None:
            item = self._draft_item(draft=draft, status="in_progress")
            await self._replace_item(output_index=draft.output_index, item=item)
            self._draft = None
        self._response = self._terminal_response("failed")
        if self._response_store is not None and self._prepared is not None:
            await self._response_store.fail_response(self._prepared, self._response.id)
        await self._publish(
            ResponseErrorEvent(
                code=public.code,
                message=public.message,
                param=public.param,
                sequence_number=0,
                type="error",
            )
        )
