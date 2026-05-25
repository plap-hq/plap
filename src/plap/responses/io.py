from __future__ import annotations

import secrets
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import anyio
import structlog
from anyio.abc import ObjectSendStream, TaskGroup

from plap.errors import ErrorLevel, PlapError, PrivateError
from plap.keyring import SealingKeyring
from plap.llms.completions.chat import ReasoningEffort, ServiceTier
from plap.logging import log_debug, log_payload
from plap.responses.contracts import (
    ConversationReference,
    OutputTextContent,
    ReasoningSummary,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseCreateRequest,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionCallItem,
    ResponseFunctionCallOutputItem,
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
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseStatus,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseUsage,
    SummaryTextContent,
)
from plap.responses.ingest.sealing import content_hash_prefix, seal_call_id, seal_reasoning_payload
from plap.responses.models import ReasoningMessagePatch, ReasoningPayload, SealedCallID, Side, StateMessage
from plap.responses.projection import ResponseProjection
from plap.responses.reasoning import IReasoningSummarizer, ReasoningSummaryPartSource
from plap.responses.store import PreparedRequest, ResponseStore

logger = structlog.get_logger(__name__)


class _CommitKind(StrEnum):
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    OUTPUT = "output"
    REASONING_DRAFT_BEGIN = "reasoning_draft_begin"
    REASONING_DRAFT_REPLACE = "reasoning_draft_replace"
    REASONING_DRAFT_SUMMARY = "reasoning_draft_summary"
    REASONING_DRAFT_COMPLETE = "reasoning_draft_complete"
    COMPLETED = "completed"


def _io_internal_error(*, reason: str, private_message: str, cause: BaseException | None = None) -> PlapError:
    return PlapError(
        public=None,
        private=PrivateError(
            event="response.internal_error",
            reason=reason,
            message=private_message,
            level=ErrorLevel.ERROR,
            cause=cause,
        ),
    )


@dataclass(frozen=True, slots=True)
class ReasoningDraft:
    item_id: str
    output_index: int


@dataclass(frozen=True, slots=True)
class PublishedMainCandidate:
    assistant_hash: str


@dataclass(frozen=True, slots=True)
class _ReasoningDraftBegin:
    item: ResponseReasoningItem


@dataclass(frozen=True, slots=True)
class _ReasoningDraftReplace:
    item: ResponseReasoningItem
    output_index: int


@dataclass(frozen=True, slots=True)
class _ReasoningDraftSummary:
    item_id: str
    output_index: int
    source: ReasoningSummaryPartSource


@dataclass(frozen=True, slots=True)
class _ReasoningDraftComplete:
    item: ResponseReasoningItem
    output_index: int


type _CommitValue = (
    ResponseObject | ResponseOutputItem | _ReasoningDraftBegin | _ReasoningDraftReplace | _ReasoningDraftSummary | _ReasoningDraftComplete
)


def _public_assistant_message(candidate: StateMessage) -> StateMessage | None:
    if candidate.content is None or candidate.content == "":
        return None
    return StateMessage(role="assistant", content=candidate.content)


def _assistant_anchor_hash(candidate: StateMessage, public_assistant: StateMessage | None) -> str:
    if public_assistant is not None:
        return public_assistant.content_hash()
    return candidate.content_hash()


def _stable_reasoning_payload(
    *,
    candidate: StateMessage,
    anchor_hash: str,
    public_assistant: StateMessage | None,
) -> ReasoningPayload:
    if public_assistant is not None and (candidate.reasoning_content or candidate.reasoning_details):
        return ReasoningPayload(
            side="main",
            temp=False,
            continuation_side=Side.MAIN,
            messages=(
                ReasoningMessagePatch(
                    content_hash=anchor_hash,
                    reasoning_content=candidate.reasoning_content,
                    reasoning_details=tuple(candidate.reasoning_details) or None,
                ),
            ),
        )
    return ReasoningPayload(side="main", temp=False, continuation_side=Side.MAIN, messages=(candidate,))


@dataclass(slots=True)
class _Commit:
    kind: _CommitKind
    value: _CommitValue
    ack: anyio.Event | None = None
    acked: bool = False
    error: BaseException | None = None
    result: object | None = None


class ResponseEventIO:
    def __init__(
        self,
        *,
        request: ResponseCreateRequest,
        projection: ResponseProjection,
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
        self._client_attached = True
        self._reasoning_summarizer = reasoning_summarizer
        self._reasoning_summarizer_model = reasoning_summarizer_model
        self._reasoning_summarizer_prompt_cache_key_base = reasoning_summarizer_prompt_cache_key_base
        self._reasoning_summarizer_reasoning_effort = reasoning_summarizer_reasoning_effort
        self._reasoning_summarizer_service_tier = reasoning_summarizer_service_tier
        self._reasoning_summary_mode = reasoning_summary_mode
        self._projection = projection
        self._commit_send, self._commit_receive = anyio.create_memory_object_stream[_Commit](16)
        self._output_items: list[ResponseOutputItem] = []
        self._sequence_number = 0

    def start(self, task_group: TaskGroup) -> None:
        task_group.start_soon(self._emit_commits)

    @property
    def response_id(self) -> str:
        return self._response.id

    async def created(self) -> None:
        await self._enqueue_commit(_CommitKind.CREATED, self._response)

    async def in_progress(self) -> None:
        await self._enqueue_commit(_CommitKind.IN_PROGRESS, self._response)

    def detach_client(self) -> None:
        self._client_attached = False

    async def output(
        self,
        item: ResponseOutputItem,
    ) -> None:
        log_debug(logger, "response.io.output.queued", item_type=item.type, reasoning_metadata=False)
        log_payload(logger, "response.io.output.payload", item=item.model_dump(mode="json", exclude_none=True))
        await self._enqueue_commit(_CommitKind.OUTPUT, item, wait=True)

    async def begin_reasoning_draft(self, item: ResponseReasoningItem) -> ReasoningDraft:
        if item.status != "in_progress":
            raise _io_internal_error(
                reason="reasoning_draft_begin_status_invalid",
                private_message="reasoning draft must begin in_progress",
            )
        result = await self._enqueue_commit(
            _CommitKind.REASONING_DRAFT_BEGIN,
            _ReasoningDraftBegin(item=item),
            wait=True,
        )
        if not isinstance(result, ReasoningDraft):
            raise _io_internal_error(
                reason="reasoning_draft_begin_result_invalid",
                private_message="reasoning draft begin did not return a handle",
            )
        return result

    async def replace_reasoning_draft(self, draft: ReasoningDraft, item: ResponseReasoningItem) -> None:
        await self._enqueue_commit(
            _CommitKind.REASONING_DRAFT_REPLACE,
            _ReasoningDraftReplace(item=item, output_index=draft.output_index),
            wait=True,
        )

    async def append_reasoning_draft_summary(
        self,
        draft: ReasoningDraft,
        source: ReasoningSummaryPartSource,
    ) -> str:
        if self._reasoning_summary_mode is None:
            raise _io_internal_error(
                reason="reasoning_draft_summary_mode_missing",
                private_message="reasoning summary mode is required for draft summary parts",
            )
        result = await self._enqueue_commit(
            _CommitKind.REASONING_DRAFT_SUMMARY,
            _ReasoningDraftSummary(
                item_id=draft.item_id,
                output_index=draft.output_index,
                source=source,
            ),
            wait=True,
        )
        if not isinstance(result, str):
            raise _io_internal_error(
                reason="reasoning_draft_summary_result_invalid",
                private_message="reasoning draft summary did not return text",
            )
        return result

    async def complete_reasoning_draft(self, draft: ReasoningDraft, item: ResponseReasoningItem) -> None:
        if item.status != "completed":
            raise _io_internal_error(
                reason="reasoning_draft_complete_status_invalid",
                private_message="reasoning draft must complete with completed status",
            )
        await self._enqueue_commit(
            _CommitKind.REASONING_DRAFT_COMPLETE,
            _ReasoningDraftComplete(item=item, output_index=draft.output_index),
            wait=True,
        )

    async def publish_main_candidate(
        self,
        *,
        candidate: StateMessage,
        keyring: SealingKeyring,
        server_outputs: Mapping[str, str],
        reasoning_summary: Sequence[SummaryTextContent] = (),
        reasoning_draft: ReasoningDraft | None = None,
    ) -> PublishedMainCandidate:
        public_assistant = _public_assistant_message(candidate)
        assistant_hash = _assistant_anchor_hash(candidate, public_assistant)
        needs_reasoning_anchor = bool(candidate.reasoning_content or candidate.reasoning_details or public_assistant is None)

        if reasoning_draft is not None:
            reasoning_payload = _stable_reasoning_payload(
                candidate=candidate,
                anchor_hash=assistant_hash,
                public_assistant=public_assistant,
            )
            await self.complete_reasoning_draft(
                reasoning_draft,
                ResponseReasoningItem(
                    encrypted_content=seal_reasoning_payload(reasoning_payload, keyring=keyring),
                    id=reasoning_draft.item_id,
                    status="completed",
                    summary=list(reasoning_summary),
                    type="reasoning",
                ),
            )
        elif needs_reasoning_anchor:
            reasoning_payload = _stable_reasoning_payload(
                candidate=candidate,
                anchor_hash=assistant_hash,
                public_assistant=public_assistant,
            )
            await self.output(
                ResponseReasoningItem(
                    encrypted_content=seal_reasoning_payload(reasoning_payload, keyring=keyring),
                    id=f"rs_{secrets.token_urlsafe(18)}",
                    status="completed",
                    summary=list(reasoning_summary),
                    type="reasoning",
                )
            )

        if public_assistant is not None:
            await self.output(
                ResponseMessageItem(
                    content=[OutputTextContent(text=public_assistant.content or "", type="output_text")],
                    id=f"msg_{secrets.token_urlsafe(18)}",
                    role="assistant",
                    status="completed",
                    type="message",
                )
            )

        function_call_ids: dict[str, str] = {}
        for index, call in enumerate(candidate.tool_calls):
            sealed_call_id = seal_call_id(
                SealedCallID(
                    side="main",
                    temp=False,
                    content_hash_prefix=content_hash_prefix(assistant_hash),
                    tool_call_index=index,
                    upstream_tool_call_id=call.id,
                ),
                keyring=keyring,
            )
            function_call_ids[call.id] = sealed_call_id
            await self.output(
                ResponseFunctionCallItem(
                    arguments=call.arguments,
                    call_id=sealed_call_id,
                    id=f"fc_{secrets.token_urlsafe(18)}",
                    name=call.name,
                    status="completed",
                    type="function_call",
                )
            )

        for call in candidate.tool_calls:
            output = server_outputs.get(call.id)
            if output is None:
                continue
            await self.output(
                ResponseFunctionCallOutputItem(
                    call_id=function_call_ids[call.id],
                    created_by="server",
                    id=f"fco_{secrets.token_urlsafe(18)}",
                    output=output,
                    status="completed",
                    type="function_call_output",
                )
            )

        return PublishedMainCandidate(assistant_hash=assistant_hash)

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
        await self._enqueue_commit(_CommitKind.COMPLETED, response)

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
        await self._enqueue_commit(_CommitKind.COMPLETED, response)

    def cancelled_response(
        self,
        *,
        service_tier: str | None = None,
        usage: ResponseUsage | None = None,
    ) -> ResponseObject:
        return self._response.model_copy(
            update={
                "completed_at": time.time(),
                "output": list(self._output_items),
                "service_tier": service_tier or self._response.service_tier,
                "status": "cancelled",
                "usage": usage,
            }
        )

    async def aclose(self) -> None:
        await self._commit_send.aclose()

    async def _enqueue_commit(
        self,
        kind: _CommitKind,
        value: _CommitValue,
        *,
        wait: bool = False,
    ) -> object | None:
        commit = _Commit(kind=kind, value=value, ack=anyio.Event() if wait else None)
        await self._commit_send.send(commit)
        if commit.ack is None:
            return None
        await commit.ack.wait()
        if commit.error is not None:
            raise commit.error
        return commit.result

    @staticmethod
    def _ack_commit(commit: _Commit, *, result: object | None = None, error: BaseException | None = None) -> None:
        if commit.ack is None or commit.acked:
            return
        commit.error = error
        commit.result = result
        commit.acked = True
        commit.ack.set()

    async def _emit_commits(self) -> None:
        async with self._commit_receive:
            async for commit in self._commit_receive:
                try:
                    if commit.kind == _CommitKind.CREATED:
                        response = cast(ResponseObject, commit.value)
                        await self._response_store.begin_response(self._prepared, cast(ResponseObject, commit.value))
                        self._response = response
                        self._ack_commit(commit)
                        await self._send_event(
                            ResponseCreatedEvent(
                                response=response,
                                sequence_number=0,
                                type="response.created",
                            )
                        )
                    elif commit.kind == _CommitKind.IN_PROGRESS:
                        response = cast(ResponseObject, commit.value)
                        self._response = response
                        self._ack_commit(commit)
                        await self._send_event(
                            ResponseInProgressEvent(
                                response=response,
                                sequence_number=0,
                                type="response.in_progress",
                            )
                        )
                    elif commit.kind == _CommitKind.OUTPUT:
                        await self._emit_output(commit)
                    elif commit.kind == _CommitKind.REASONING_DRAFT_BEGIN:
                        await self._emit_reasoning_draft_begin(commit)
                    elif commit.kind == _CommitKind.REASONING_DRAFT_REPLACE:
                        await self._emit_reasoning_draft_replace(commit)
                    elif commit.kind == _CommitKind.REASONING_DRAFT_SUMMARY:
                        await self._emit_reasoning_draft_summary(commit)
                    elif commit.kind == _CommitKind.REASONING_DRAFT_COMPLETE:
                        await self._emit_reasoning_draft_complete(commit)
                    else:
                        response = cast(ResponseObject, commit.value).model_copy(update={"output": self._output_items})
                        await self._response_store.finish_response(self._prepared, response)
                        self._response = response
                        self._ack_commit(commit)
                        await self._send_event(
                            ResponseCompletedEvent(
                                response=response,
                                sequence_number=0,
                                type="response.completed",
                            )
                        )
                except BaseException as exc:
                    self._ack_commit(commit, error=exc)
                    raise
                else:
                    self._ack_commit(commit)

    async def _send_event(self, event: ResponseStreamEvent) -> None:
        if not self._client_attached:
            return
        self._sequence_number += 1
        payload = self._projection.stream_payload(event, sequence_number=self._sequence_number)
        log_debug(logger, "response.io.event.sent", event_type=payload.get("type"), sequence_number=self._sequence_number)
        log_payload(logger, "response.io.event.payload", payload=payload)
        await self._send.send(type(event).model_validate(payload))

    async def _emit_output(
        self,
        commit: _Commit,
    ) -> None:
        item = cast(ResponseOutputItem, commit.value)
        output_index = len(self._output_items)
        await self._response_store.append_output_item(
            self._prepared,
            self._response.id,
            output_index,
            item.model_dump(mode="json", exclude_none=True),
        )
        completed_item = item
        self._output_items.append(completed_item)
        self._ack_commit(commit)
        await self._emit_output_item_events(item, output_index=output_index)

    async def _emit_reasoning_draft_begin(self, commit: _Commit) -> None:
        draft = cast(_ReasoningDraftBegin, commit.value)
        output_index = len(self._output_items)
        await self._response_store.append_output_item(
            self._prepared,
            self._response.id,
            output_index,
            draft.item.model_dump(mode="json", exclude_none=True),
        )
        self._output_items.append(draft.item)
        self._ack_commit(commit, result=ReasoningDraft(item_id=draft.item.id, output_index=output_index))
        await self._send_event(
            ResponseOutputItemAddedEvent(
                item=draft.item,
                output_index=output_index,
                sequence_number=0,
                type="response.output_item.added",
            )
        )

    async def _emit_reasoning_draft_replace(self, commit: _Commit) -> None:
        draft = cast(_ReasoningDraftReplace, commit.value)
        item = self._merge_reasoning_draft_summary(draft.item, output_index=draft.output_index)
        await self._response_store.replace_output_item(
            self._prepared,
            self._response.id,
            draft.output_index,
            item.model_dump(mode="json", exclude_none=True),
        )
        self._output_items[draft.output_index] = item
        self._ack_commit(commit)

    async def _emit_reasoning_draft_summary(self, commit: _Commit) -> None:
        summary = cast(_ReasoningDraftSummary, commit.value)
        item = self._reasoning_draft_item(summary.output_index)
        summary_index = len(item.summary)
        summary_part = await self._emit_reasoning_summary_part_from_source(
            item_id=summary.item_id,
            output_index=summary.output_index,
            summary_index=summary_index,
            source=summary.source,
        )
        updated_item = item.model_copy(update={"summary": [*item.summary, summary_part]})
        await self._response_store.replace_output_item(
            self._prepared,
            self._response.id,
            summary.output_index,
            updated_item.model_dump(mode="json", exclude_none=True),
        )
        self._output_items[summary.output_index] = updated_item
        self._ack_commit(commit, result=summary_part.text)

    async def _emit_reasoning_draft_complete(self, commit: _Commit) -> None:
        draft = cast(_ReasoningDraftComplete, commit.value)
        item = self._merge_reasoning_draft_summary(draft.item, output_index=draft.output_index)
        await self._response_store.replace_output_item(
            self._prepared,
            self._response.id,
            draft.output_index,
            item.model_dump(mode="json", exclude_none=True),
        )
        self._output_items[draft.output_index] = item
        self._ack_commit(commit)
        await self._send_event(
            ResponseOutputItemDoneEvent(
                item=item,
                output_index=draft.output_index,
                sequence_number=0,
                type="response.output_item.done",
            )
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
        if isinstance(item, ResponseReasoningItem):
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

    async def _emit_reasoning_summary_part_from_source(
        self,
        *,
        item_id: str,
        output_index: int,
        summary_index: int,
        source: ReasoningSummaryPartSource,
    ) -> SummaryTextContent:
        summary_text = ""
        log_debug(
            logger,
            "response.io.reasoning_summary.start",
            item_id=item_id,
            mode=self._reasoning_summary_mode,
            output_index=output_index,
            summary_index=summary_index,
        )
        await self._send_event(
            ResponseReasoningSummaryPartAddedEvent(
                item_id=item_id,
                output_index=output_index,
                part=SummaryTextContent(text="", type="summary_text"),
                sequence_number=0,
                summary_index=summary_index,
                type="response.reasoning_summary_part.added",
            )
        )
        try:
            async for delta in self._reasoning_summarizer.stream_part(
                model=self._reasoning_summarizer_model,
                prompt_cache_key=self._reasoning_summarizer_prompt_cache_key(),
                reasoning_effort=self._reasoning_summarizer_reasoning_effort,
                service_tier=self._reasoning_summarizer_service_tier,
                mode=cast(ReasoningSummary, self._reasoning_summary_mode),
                source=source,
            ):
                summary_text += delta
                await self._send_event(
                    ResponseReasoningSummaryTextDeltaEvent(
                        delta=delta,
                        item_id=item_id,
                        output_index=output_index,
                        sequence_number=0,
                        summary_index=summary_index,
                        type="response.reasoning_summary_text.delta",
                    )
                )
        except Exception:
            log_debug(logger, "response.io.reasoning_summary.failed", item_id=item_id, output_index=output_index)
            summary_text = ""

        summary_part = SummaryTextContent(text=summary_text, type="summary_text")
        await self._send_event(
            ResponseReasoningSummaryTextDoneEvent(
                item_id=item_id,
                output_index=output_index,
                sequence_number=0,
                summary_index=summary_index,
                text=summary_text,
                type="response.reasoning_summary_text.done",
            )
        )
        await self._send_event(
            ResponseReasoningSummaryPartDoneEvent(
                item_id=item_id,
                output_index=output_index,
                part=summary_part,
                sequence_number=0,
                summary_index=summary_index,
                type="response.reasoning_summary_part.done",
            )
        )
        log_debug(
            logger,
            "response.io.reasoning_summary.done",
            item_id=item_id,
            output_index=output_index,
            summary_index=summary_index,
            summary_length=len(summary_text),
        )
        return summary_part

    def _reasoning_draft_item(self, output_index: int) -> ResponseReasoningItem:
        try:
            item = self._output_items[output_index]
        except IndexError as exc:
            raise _io_internal_error(
                reason="reasoning_draft_output_index_invalid",
                private_message="reasoning draft output index is invalid",
                cause=exc,
            ) from exc
        if not isinstance(item, ResponseReasoningItem):
            raise _io_internal_error(
                reason="reasoning_draft_target_invalid",
                private_message="reasoning draft target is not a reasoning item",
            )
        return item

    def _merge_reasoning_draft_summary(
        self,
        item: ResponseReasoningItem,
        *,
        output_index: int,
    ) -> ResponseReasoningItem:
        current = self._reasoning_draft_item(output_index)
        if item.summary or not current.summary:
            return item
        return item.model_copy(update={"summary": list(current.summary)})

    def _reasoning_summarizer_prompt_cache_key(self) -> str | None:
        if self._reasoning_summarizer_prompt_cache_key_base is None:
            return None
        return f"{self._reasoning_summarizer_prompt_cache_key_base}|reasoning_summarizer"

    async def _emit_reasoning_events(
        self,
        item: ResponseReasoningItem,
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
