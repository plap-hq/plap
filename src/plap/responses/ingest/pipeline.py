from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.keyring import SealingKeyring
from plap.responses.contracts import (
    RequestCompactionItem,
    RequestFunctionCallItem,
    RequestFunctionCallOutputItem,
    RequestMessageItem,
    RequestReasoningItem,
    ResponseCreateRequest,
)
from plap.responses.ingest.sealing import open_call_id, open_compaction_payload, open_reasoning_payload
from plap.responses.models import (
    ChatMessageSpan,
    CompactionPayload,
    IngestedQueues,
    ReasoningMessagePatch,
    ReasoningPayload,
    SealedCallID,
    Side,
    SideMessage,
    StateMessage,
    StateToolCall,
    append_main_context_row,
)


def _input_replay_error(*, reason: str, private_message: str, cause: BaseException | None = None) -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code="invalid_input_replay",
            message="Input replay items are invalid.",
            param="input",
        ),
        private=PrivateError(
            event="response.invalid_request",
            reason=reason,
            message=private_message,
            level=ErrorLevel.WARNING,
            cause=cause,
        ),
    )


def _reasoning_replay_error(*, reason: str, private_message: str, cause: BaseException | None = None) -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code="invalid_reasoning_replay",
            message="Reasoning replay data is invalid.",
            param="input",
        ),
        private=PrivateError(
            event="response.invalid_request",
            reason=reason,
            message=private_message,
            level=ErrorLevel.WARNING,
            cause=cause,
        ),
    )


def _tool_replay_error(*, reason: str, private_message: str, cause: BaseException | None = None) -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code="invalid_tool_replay",
            message="Tool replay data is invalid.",
            param="input",
        ),
        private=PrivateError(
            event="response.invalid_request",
            reason=reason,
            message=private_message,
            level=ErrorLevel.WARNING,
            cause=cause,
        ),
    )


def _compaction_replay_error(*, reason: str, private_message: str, cause: BaseException | None = None) -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code="invalid_compaction_replay",
            message="Compaction replay data is invalid.",
            param="input",
        ),
        private=PrivateError(
            event="response.invalid_request",
            reason=reason,
            message=private_message,
            level=ErrorLevel.WARNING,
            cause=cause,
        ),
    )


class _EventKind(StrEnum):
    MESSAGE = "message"
    REASONING = "reasoning"
    FUNCTION_CALL = "function_call"
    FUNCTION_CALL_OUTPUT = "function_call_output"
    FABRICATED_FUNCTION_CALL = "fabricated_function_call"
    FABRICATED_FUNCTION_CALL_OUTPUT = "fabricated_function_call_output"


async def ingest_response_request(
    request: ResponseCreateRequest,
    *,
    keyring: SealingKeyring,
) -> IngestedQueues:
    input_items = _normalize_input_items(request)
    compaction, remaining = _open_compaction_root(input_items, keyring=keyring)
    decoded = _decode_sealed_items(remaining, keyring=keyring)
    reordered = _hoist_interleaved_reset_candidates_before_temp_debate(decoded)
    pruned = _prune_temp_debate_globally(reordered)
    routed = _route_items_by_side(pruned)
    queues = _associate_side_queues(compaction, routed)
    return IngestedQueues(
        main_context=tuple(queues.main.context_rows),
        defender=tuple(queues.defender.rows),
        reviewer=tuple(queues.reviewer.rows),
        arbitrator=tuple(queues.arbitrator.rows),
        continuation_side=routed.continuation_side,
        cursors=queues.cursors,
    )


@dataclass(slots=True)
class _DecodedItem:
    item: object
    reasoning: ReasoningPayload | None = None
    call_id: SealedCallID | None = None
    temp_related: bool = False


@dataclass(slots=True)
class _SideEvent:
    kind: _EventKind
    item: object | None = None
    reasoning: ReasoningPayload | None = None
    call_id: SealedCallID | None = None
    call: RequestFunctionCallItem | None = None
    output: RequestFunctionCallOutputItem | None = None
    temp: bool = False


@dataclass(slots=True)
class _RoutedItems:
    main: list[_SideEvent] = field(default_factory=list)
    defender: list[_SideEvent] = field(default_factory=list)
    reviewer: list[_SideEvent] = field(default_factory=list)
    arbitrator: list[_SideEvent] = field(default_factory=list)
    continuation_side: Side = Side.MAIN

    def side(self, side: Side) -> list[_SideEvent]:
        if side == Side.MAIN:
            return self.main
        if side == Side.DEFENDER:
            return self.defender
        if side == Side.REVIEWER:
            return self.reviewer
        return self.arbitrator


@dataclass(slots=True)
class _AssociatedQueues:
    main: _MainQueue
    defender: _PrivateSideQueue
    reviewer: _PrivateSideQueue
    arbitrator: _PrivateSideQueue
    cursors: dict[str, int]


class _QueueBase:
    def __init__(self, side: Side) -> None:
        self.side = side
        self._entries: list[ChatMessageSpan | SideMessage] = []
        self._pending_tool_call_ids: set[str] = set()
        self._pending_reasoning_patch: ReasoningMessagePatch | None = None
        self._current_assistant_entry: ChatMessageSpan | SideMessage | None = None
        self._tool_outputs_started = False
        self._unreplayed_reasoning_tool_call_ids: set[str] = set()
        self._reasoning_tool_call_ids_seen: set[str] = set()
        self._deferred_reasoning_tool_outputs: list[tuple[StateMessage, bool]] = []
        self._deferred_reasoning_tool_output_ids: set[str] = set()

    def add_reasoning(self, payload: ReasoningPayload) -> None:
        self._ensure_reasoning_can_start()
        patch_seen = False
        first_message_is_assistant = (
            bool(payload.messages)
            and isinstance(payload.messages[0], StateMessage)
            and payload.messages[0].is_assistant()
        )
        for index, item in enumerate(payload.messages):
            if isinstance(item, ReasoningMessagePatch):
                if patch_seen:
                    raise _reasoning_replay_error(
                        reason="reasoning_message_invalid",
                        private_message="multiple reasoning patches are not allowed",
                    )
                self._register_reasoning_patch(item)
                patch_seen = True
                continue
            if not isinstance(item, StateMessage):
                raise _reasoning_replay_error(reason="reasoning_message_invalid", private_message="reasoning message is invalid")
            if patch_seen and not item.is_tool():
                raise _reasoning_replay_error(
                    reason="reasoning_message_invalid",
                    private_message="reasoning patch can only be followed by hidden tool rows",
                )
            if self._is_deferred_reasoning_tool_output(
                item,
                patch_seen=patch_seen,
                first_message_is_assistant=first_message_is_assistant,
                index=index,
            ):
                self._defer_reasoning_tool_output(item, temp=payload.temp)
                continue
            self._append_reasoning_message(item, temp=payload.temp)
        self._settle_deferred_reasoning_tool_outputs()

    def associate_function_call(
        self,
        item: RequestFunctionCallItem,
        call_id: SealedCallID,
    ) -> None:
        target = self._sealed_function_call_target(call_id)
        if call_id.tool_call_index < len(target.tool_calls):
            existing = target.tool_call_at(call_id.tool_call_index)
            if existing.id != call_id.upstream_tool_call_id:
                raise _tool_replay_error(
                    reason="sealed_function_call_upstream_id_mismatch", private_message="sealed function_call upstream id mismatch"
                )
            self._mark_reasoning_tool_call_replayed(call_id.upstream_tool_call_id)
            self._mark_pending_tool_call(call_id.upstream_tool_call_id)
            self._settle_deferred_reasoning_tool_outputs()
            return
        if call_id.tool_call_index != len(target.tool_calls):
            raise _tool_replay_error(
                reason="sealed_function_call_index_not_contiguous", private_message="sealed function_call index is not contiguous"
            )
        target.append_tool_call(_chat_tool_call(item, call_id))
        self._mark_pending_tool_call(call_id.upstream_tool_call_id)
        self._settle_deferred_reasoning_tool_outputs()

    def associate_function_call_output(
        self,
        item: RequestFunctionCallOutputItem,
        call_id: SealedCallID,
        *,
        temp: bool = False,
    ) -> None:
        self._settle_deferred_reasoning_tool_outputs()
        self._require_tool_call_output(call_id)
        self._consume_pending_tool_call(call_id.upstream_tool_call_id)
        self._append_tool_output(
            StateMessage(
                role="tool",
                tool_call_id=call_id.upstream_tool_call_id,
                content=_function_output_text(item),
            ),
            temp=temp,
        )

    def _append_message(self, message: StateMessage, *, temp: bool = False) -> StateMessage:
        raise NotImplementedError

    def _append_tool_output(self, message: StateMessage, *, temp: bool = False) -> None:
        _ = temp
        self._append_message(message)

    def _ensure_reasoning_can_start(self) -> None:
        if self._pending_reasoning_patch is not None:
            raise _reasoning_replay_error(
                reason="reasoning_content_hash_target_missing",
                private_message="reasoning patch target assistant must appear immediately after its reasoning item",
            )
        if self._pending_tool_call_ids:
            raise _tool_replay_error(
                reason="pending_tool_outputs_block_message",
                private_message="same-side reasoning cannot appear before pending tool outputs",
            )

    def _register_reasoning_patch(self, patch: ReasoningMessagePatch) -> None:
        self._pending_reasoning_patch = patch
        self._clear_current_assistant_entry()

    def _is_deferred_reasoning_tool_output(
        self,
        message: StateMessage,
        *,
        patch_seen: bool,
        first_message_is_assistant: bool,
        index: int,
    ) -> bool:
        if not message.is_tool():
            return False
        if patch_seen:
            return True
        return first_message_is_assistant and index > 0

    def _append_reasoning_message(self, message: StateMessage, *, temp: bool) -> None:
        appended = self._append_message(message, temp=temp)
        self._track_reasoning_message(appended)

    def _apply_reasoning_patch(self, patch: ReasoningMessagePatch, target: StateMessage) -> None:
        patch.apply_to(target)
        if patch.tool_calls is not None or patch.role == "tool" or (target.is_tool() and patch.tool_call_id is not None):
            self._track_reasoning_message(target)

    def _assert_message_matches_pending_reasoning_patch(self, message: StateMessage) -> None:
        patch = self._pending_reasoning_patch
        if patch is None:
            return
        if not message.is_assistant() or message.content_hash() != patch.content_hash:
            raise _reasoning_replay_error(
                reason="reasoning_content_hash_target_missing",
                private_message="reasoning patch target assistant must appear immediately after its reasoning item",
            )

    def _apply_pending_reasoning_patch(self, entry: ChatMessageSpan | SideMessage) -> None:
        patch = self._pending_reasoning_patch
        if patch is None:
            return
        if not entry.message.is_assistant() or entry.content_hash != patch.content_hash:
            raise _reasoning_replay_error(
                reason="reasoning_content_hash_target_missing",
                private_message="reasoning patch target assistant must appear immediately after its reasoning item",
            )
        self._pending_reasoning_patch = None
        self._apply_reasoning_patch(patch, entry.message)

    def _remember_entry(self, entry: ChatMessageSpan | SideMessage) -> None:
        self._entries.append(entry)
        self._apply_pending_reasoning_patch(entry)
        message = entry.message
        if message.is_assistant():
            self._set_current_assistant_entry(entry)
            self._settle_deferred_reasoning_tool_outputs()
            return
        if message.is_tool():
            if self._current_assistant_entry is not None:
                self._tool_outputs_started = True
            return
        self._clear_current_assistant_entry()

    def _set_current_assistant_entry(self, entry: ChatMessageSpan | SideMessage) -> None:
        self._current_assistant_entry = entry
        self._tool_outputs_started = False

    def _clear_current_assistant_entry(self) -> None:
        self._current_assistant_entry = None
        self._tool_outputs_started = False

    def _defer_reasoning_tool_output(self, message: StateMessage, *, temp: bool) -> None:
        if not message.is_tool() or not message.tool_call_id:
            raise _reasoning_replay_error(
                reason="reasoning_message_invalid",
                private_message="hidden reasoning tool output must be a tool message with tool_call_id",
            )
        if message.tool_call_id in self._deferred_reasoning_tool_output_ids:
            raise _reasoning_replay_error(
                reason="duplicate_reasoning_tool_output",
                private_message="duplicate hidden reasoning tool output",
            )
        self._deferred_reasoning_tool_outputs.append((message, temp))
        self._deferred_reasoning_tool_output_ids.add(message.tool_call_id)

    def _settle_deferred_reasoning_tool_outputs(self) -> None:
        if not self._deferred_reasoning_tool_outputs:
            return
        if self._pending_reasoning_patch is not None:
            return
        if self._current_assistant_message() is None:
            return
        if self._unreplayed_reasoning_tool_call_ids - self._deferred_reasoning_tool_output_ids:
            return
        deferred_outputs = list(self._deferred_reasoning_tool_outputs)
        self._deferred_reasoning_tool_outputs.clear()
        self._deferred_reasoning_tool_output_ids.clear()
        for message, temp in deferred_outputs:
            self._append_tool_output(message, temp=temp)
            self._satisfy_reasoning_tool_call_from_hidden_output(message)

    def _ensure_deferred_reasoning_tool_outputs_settled(
        self,
        *,
        private_message: str,
        next_message: StateMessage,
    ) -> None:
        if not self._deferred_reasoning_tool_outputs:
            return
        if self._pending_reasoning_patch is not None and next_message.is_assistant():
            return
        if next_message.is_tool():
            return
        self._settle_deferred_reasoning_tool_outputs()
        if self._deferred_reasoning_tool_outputs:
            raise _reasoning_replay_error(
                reason="reasoning_message_invalid",
                private_message=private_message,
            )

    def _current_assistant_message(self) -> StateMessage | None:
        if self._current_assistant_entry is None:
            return None
        return self._current_assistant_entry.message

    def _sealed_function_call_target(self, call_id: SealedCallID) -> StateMessage:
        if self._pending_reasoning_patch is not None:
            raise _reasoning_replay_error(
                reason="reasoning_content_hash_target_missing",
                private_message="reasoning patch target assistant must appear immediately after its reasoning item",
            )
        entry = self._current_assistant_entry
        if entry is None or not entry.message.is_assistant():
            raise _tool_replay_error(
                reason="sealed_function_call_content_hash_target_missing",
                private_message="sealed function call content_hash target is missing",
            )
        if not entry.content_hash.startswith(call_id.content_hash_prefix.hex()):
            raise _tool_replay_error(
                reason="sealed_function_call_content_hash_target_missing",
                private_message="sealed function call content_hash target is missing",
            )
        if self._tool_outputs_started:
            raise _tool_replay_error(
                reason="sealed_function_call_after_function_call_output",
                private_message="sealed function_call cannot appear after function_call_output for the same assistant turn",
            )
        return entry.message

    def _require_tool_call_output(self, call_id: SealedCallID) -> None:
        target = self._current_assistant_message()
        if self._pending_reasoning_patch is not None:
            raise _reasoning_replay_error(
                reason="reasoning_content_hash_target_missing",
                private_message="reasoning patch target assistant must appear immediately after its reasoning item",
            )
        if target is None or self._current_assistant_entry is None:
            raise _tool_replay_error(
                reason="sealed_function_call_content_hash_target_missing",
                private_message="sealed function call content_hash target is missing",
            )
        if not self._current_assistant_entry.content_hash.startswith(call_id.content_hash_prefix.hex()):
            raise _tool_replay_error(
                reason="sealed_function_call_content_hash_target_missing",
                private_message="sealed function call content_hash target is missing",
            )
        if not target.tool_calls:
            raise _tool_replay_error(
                reason="sealed_function_call_output_target_has_no_tool_calls",
                private_message="sealed function_call_output target has no tool_calls",
            )
        if call_id.tool_call_index >= len(target.tool_calls):
            raise _tool_replay_error(
                reason="sealed_function_call_output_index_out_of_range", private_message="sealed function_call_output index is out of range"
            )
        existing = target.tool_call_at(call_id.tool_call_index)
        if existing.id != call_id.upstream_tool_call_id:
            raise _tool_replay_error(
                reason="sealed_function_call_output_upstream_id_mismatch",
                private_message="sealed function_call_output upstream id mismatch",
            )

    def _ensure_no_pending_tool_calls(self) -> None:
        if self._pending_tool_call_ids:
            raise _tool_replay_error(
                reason="pending_tool_outputs_block_message", private_message="same-side message cannot appear before pending tool outputs"
            )

    def _mark_pending_tool_call(self, call_id: str) -> None:
        if call_id in self._pending_tool_call_ids:
            raise _tool_replay_error(reason="duplicate_pending_function_call", private_message="duplicate pending function_call")
        self._pending_tool_call_ids.add(call_id)

    def _mark_reasoning_tool_call_replayed(self, call_id: str) -> None:
        self._unreplayed_reasoning_tool_call_ids.discard(call_id)

    def _consume_pending_tool_call(self, call_id: str) -> None:
        if call_id not in self._pending_tool_call_ids:
            raise _tool_replay_error(
                reason="function_call_output_without_pending_function_call",
                private_message="function_call_output has no pending function_call",
            )
        self._pending_tool_call_ids.remove(call_id)

    def _track_reasoning_tool_calls(self, message: StateMessage) -> None:
        if not message.is_assistant():
            return
        if not message.tool_calls:
            return
        for tool_call in message.tool_calls:
            tool_call_id = tool_call.id
            if not tool_call_id:
                raise _reasoning_replay_error(reason="reasoning_tool_call_id_missing", private_message="reasoning tool call id is missing")
            if tool_call_id in self._reasoning_tool_call_ids_seen:
                raise _reasoning_replay_error(reason="duplicate_reasoning_tool_call", private_message="duplicate reasoning tool call")
            self._reasoning_tool_call_ids_seen.add(tool_call_id)
            self._unreplayed_reasoning_tool_call_ids.add(tool_call_id)

    def _satisfy_reasoning_tool_call_from_hidden_output(self, message: StateMessage) -> None:
        if not message.is_tool():
            return
        tool_call_id = message.tool_call_id
        if not tool_call_id:
            raise _reasoning_replay_error(
                reason="reasoning_tool_output_call_id_missing", private_message="reasoning tool output tool_call_id is missing"
            )
        if tool_call_id not in self._unreplayed_reasoning_tool_call_ids:
            raise _reasoning_replay_error(
                reason="reasoning_tool_output_without_pending_call", private_message="reasoning tool output has no pending tool call"
            )
        self._unreplayed_reasoning_tool_call_ids.remove(tool_call_id)

    def _track_reasoning_message(self, message: StateMessage) -> None:
        self._track_reasoning_tool_calls(message)
        self._satisfy_reasoning_tool_call_from_hidden_output(message)

    def assert_no_pending_tool_calls(self) -> None:
        self._settle_deferred_reasoning_tool_outputs()
        if self._pending_reasoning_patch is not None:
            raise _reasoning_replay_error(
                reason="reasoning_content_hash_target_missing", private_message="reasoning content_hash target is missing"
            )
        if self._deferred_reasoning_tool_outputs:
            raise _reasoning_replay_error(
                reason="reasoning_message_invalid",
                private_message="hidden reasoning tool outputs could not be attached to their assistant turn",
            )
        if self._unreplayed_reasoning_tool_call_ids:
            raise _reasoning_replay_error(
                reason="reasoning_tool_call_missing_function_call_item", private_message="reasoning tool call is missing function_call item"
            )
        if self._pending_tool_call_ids:
            raise _tool_replay_error(
                reason="function_call_missing_function_call_output", private_message="function_call is missing function_call_output"
            )


class _MainQueue(_QueueBase):
    def __init__(self, cursors: dict[str, int]) -> None:
        super().__init__(Side.MAIN)
        self._cursors = cursors
        self._seed_rows: list[ChatMessageSpan] = []
        self._stable_rows: list[ChatMessageSpan] = []

    @property
    def context_rows(self) -> list[ChatMessageSpan]:
        return [*self._seed_rows, *self._stable_rows]

    @property
    def stable_rows(self) -> list[ChatMessageSpan]:
        return self._stable_rows

    def add_existing_row(self, row: ChatMessageSpan) -> None:
        self._seed_rows.append(row)
        self._entries.append(row)

    def add_message(self, message: StateMessage) -> ChatMessageSpan:
        return self._append_main_row(message)

    def attach_fabricated_call(self, call: RequestFunctionCallItem) -> None:
        if self._pending_reasoning_patch is not None:
            raise _reasoning_replay_error(
                reason="reasoning_content_hash_target_missing",
                private_message="reasoning patch target assistant must appear immediately after its reasoning item",
            )
        target = self._current_assistant_message()
        if target is None or self._tool_outputs_started:
            raise _tool_replay_error(
                reason="fabricated_function_call_without_previous_assistant",
                private_message="fabricated function_call has no previous assistant message",
            )
        target.append_tool_call(
            StateToolCall(
                id=call.call_id,
                name=call.name,
                arguments=call.arguments,
            )
        )
        self._mark_pending_tool_call(call.call_id)

    def attach_fabricated_output(self, output: RequestFunctionCallOutputItem) -> None:
        self._consume_pending_tool_call(output.call_id)
        self._append_tool_output(
            StateMessage(
                role="tool",
                tool_call_id=output.call_id,
                content=_function_output_text(output),
            )
        )

    def _append_message(self, message: StateMessage, *, temp: bool = False) -> StateMessage:
        return self._append_main_row(message, temp=temp).message

    def _append_tool_output(self, message: StateMessage, *, temp: bool = False) -> None:
        self._append_main_row(message, allow_pending=True, temp=temp)

    def _append_main_row(
        self,
        message: StateMessage,
        *,
        allow_pending: bool = False,
        temp: bool = False,
    ) -> ChatMessageSpan:
        if not allow_pending:
            self._ensure_no_pending_tool_calls()
        self._ensure_deferred_reasoning_tool_outputs_settled(
            private_message="hidden reasoning tool outputs cannot be followed by additional messages before they attach",
            next_message=message,
        )
        self._assert_message_matches_pending_reasoning_patch(message)
        if temp:
            raise _input_replay_error(
                reason="main_temp_debate_replay_unsupported",
                private_message="main temp debate replay is no longer supported",
            )
        try:
            row = append_main_context_row(self._stable_rows, self._cursors, message)
        except ValueError as exc:
            raise _tool_replay_error(
                reason="function_call_output_without_main_segment",
                private_message="function_call_output requires a preceding main segment",
                cause=exc,
            ) from exc
        self._remember_entry(row)
        return row


class _PrivateSideQueue(_QueueBase):
    def __init__(self, side: Side) -> None:
        if side == Side.MAIN:
            raise ValueError("main side must use _MainQueue")
        super().__init__(side)

    @property
    def rows(self) -> list[SideMessage]:
        return [entry for entry in self._entries if isinstance(entry, SideMessage)]

    def _append_message(self, message: StateMessage, *, temp: bool = False) -> StateMessage:
        self._ensure_no_pending_tool_calls()
        self._ensure_deferred_reasoning_tool_outputs_settled(
            private_message="hidden reasoning tool outputs cannot be followed by additional messages before they attach",
            next_message=message,
        )
        self._assert_message_matches_pending_reasoning_patch(message)
        row = SideMessage(message=message)
        self._remember_entry(row)
        return row.message

    def _append_tool_output(self, message: StateMessage, *, temp: bool = False) -> None:
        _ = temp
        self._ensure_deferred_reasoning_tool_outputs_settled(
            private_message="hidden reasoning tool outputs cannot be followed by additional messages before they attach",
            next_message=message,
        )
        self._assert_message_matches_pending_reasoning_patch(message)
        row = SideMessage(message=message)
        self._remember_entry(row)


def _normalize_input_items(request: ResponseCreateRequest) -> list[object]:
    if request.input is None:
        return []
    if isinstance(request.input, str):
        return [RequestMessageItem(content=request.input, role="user", type="message")]
    return list(request.input)


def _open_compaction_root(
    input_items: list[object],
    *,
    keyring: SealingKeyring,
) -> tuple[CompactionPayload | None, list[object]]:
    last_index = _last_compaction_index(input_items)
    if last_index is None:
        return None, input_items
    compaction_item = input_items[last_index]
    if not isinstance(compaction_item, RequestCompactionItem):
        raise TypeError("last compaction index did not point at compaction item")
    return (
        open_compaction_payload(compaction_item.encrypted_content, keyring=keyring),
        input_items[last_index + 1 :],
    )


def _last_compaction_index(input_items: list[object]) -> int | None:
    last_index: int | None = None
    for index, item in enumerate(input_items):
        if isinstance(item, RequestCompactionItem):
            last_index = index
    return last_index


def _decode_sealed_items(input_items: list[object], *, keyring: SealingKeyring) -> list[_DecodedItem]:
    decoded: list[_DecodedItem] = []
    for item in input_items:
        if isinstance(item, RequestMessageItem):
            decoded.append(_DecodedItem(item=item))
            continue
        if isinstance(item, RequestReasoningItem):
            payload = _open_reasoning_item(item, keyring=keyring)
            decoded.append(
                _DecodedItem(
                    item=item,
                    reasoning=payload,
                    temp_related=payload.temp,
                )
            )
            continue
        if isinstance(item, RequestFunctionCallItem | RequestFunctionCallOutputItem):
            call_id = _open_call_id_or_none(item.call_id, keyring=keyring)
            decoded.append(
                _DecodedItem(
                    item=item,
                    call_id=call_id,
                    temp_related=call_id is not None and call_id.temp,
                )
            )
    return decoded


def _open_reasoning_item(item: RequestReasoningItem, *, keyring: SealingKeyring) -> ReasoningPayload:
    if item.encrypted_content is None:
        raise _reasoning_replay_error(
            reason="unsealed_reasoning_input_untrusted", private_message="unsealed reasoning input is not trusted"
        )
    return open_reasoning_payload(item.encrypted_content, keyring=keyring)


def _is_hoistable_interleaved_non_temp(item: _DecodedItem) -> bool:
    return not item.temp_related and item.reasoning is None


def _hoist_interleaved_reset_candidates_before_temp_debate(items: list[_DecodedItem]) -> list[_DecodedItem]:
    reordered: list[_DecodedItem] = []
    index = 0

    while index < len(items):
        item = items[index]
        if not item.temp_related:
            reordered.append(item)
            index += 1
            continue

        hoisted_messages: list[_DecodedItem] = []
        temp_items: list[_DecodedItem] = []
        tail_messages: list[_DecodedItem] = []

        while index < len(items):
            item = items[index]
            if item.temp_related:
                if tail_messages:
                    hoisted_messages.extend(tail_messages)
                    tail_messages.clear()
                temp_items.append(item)
                index += 1
                continue
            if _is_hoistable_interleaved_non_temp(item):
                tail_messages.append(item)
                index += 1
                continue
            break

        reordered.extend(hoisted_messages)
        reordered.extend(temp_items)
        reordered.extend(tail_messages)

    return reordered


def _prune_temp_debate_globally(items: list[_DecodedItem]) -> list[_DecodedItem]:
    retained: list[_DecodedItem] = []
    for item in items:
        if not item.temp_related:
            retained = [candidate for candidate in retained if not candidate.temp_related]
        retained.append(item)
    return retained


def _route_items_by_side(items: list[_DecodedItem]) -> _RoutedItems:
    routed = _RoutedItems()
    pending_unsealed: dict[str, RequestFunctionCallItem] = {}
    for item in items:
        if isinstance(item.item, RequestMessageItem):
            routed.continuation_side = Side.MAIN
            routed.main.append(_SideEvent(kind=_EventKind.MESSAGE, item=item.item))
            continue
        if item.reasoning is not None:
            routed.continuation_side = item.reasoning.continuation_side
            routed.side(item.reasoning.side).append(_SideEvent(kind=_EventKind.REASONING, reasoning=item.reasoning))
            continue
        if isinstance(item.item, RequestFunctionCallItem):
            if item.call_id is None:
                if item.item.call_id in pending_unsealed:
                    raise _tool_replay_error(
                        reason="duplicate_pending_unsealed_function_call", private_message="duplicate pending unsealed function_call"
                    )
                pending_unsealed[item.item.call_id] = item.item
                routed.continuation_side = Side.MAIN
                routed.main.append(_SideEvent(kind=_EventKind.FABRICATED_FUNCTION_CALL, call=item.item))
            else:
                routed.continuation_side = item.call_id.side
                routed.side(item.call_id.side).append(
                    _SideEvent(
                        kind=_EventKind.FUNCTION_CALL,
                        item=item.item,
                        call_id=item.call_id,
                    )
                )
            continue
        if isinstance(item.item, RequestFunctionCallOutputItem):
            if item.call_id is None:
                call = pending_unsealed.pop(item.item.call_id, None)
                if call is None:
                    raise _tool_replay_error(
                        reason="unsealed_function_call_output_without_matching_function_call",
                        private_message="unsealed function_call_output has no matching function_call",
                    )
                routed.continuation_side = Side.MAIN
                routed.main.append(_SideEvent(kind=_EventKind.FABRICATED_FUNCTION_CALL_OUTPUT, output=item.item))
            else:
                routed.continuation_side = item.call_id.side
                routed.side(item.call_id.side).append(
                    _SideEvent(
                        kind=_EventKind.FUNCTION_CALL_OUTPUT,
                        item=item.item,
                        call_id=item.call_id,
                        temp=item.temp_related,
                    )
                )
    if pending_unsealed:
        raise _tool_replay_error(
            reason="unsealed_function_call_missing_output", private_message="unsealed function_call is missing function_call_output"
        )
    return routed


def _associate_side_queues(
    compaction: CompactionPayload | None,
    routed: _RoutedItems,
) -> _AssociatedQueues:
    cursors = _initial_cursors(compaction)
    queues = _AssociatedQueues(
        main=_MainQueue(cursors),
        defender=_PrivateSideQueue(Side.DEFENDER),
        reviewer=_PrivateSideQueue(Side.REVIEWER),
        arbitrator=_PrivateSideQueue(Side.ARBITRATOR),
        cursors=cursors,
    )
    if compaction is not None:
        for span in compaction.active:
            queues.main.add_existing_row(span)
    _apply_side_events(queues.main, routed.main)
    _apply_side_events(queues.defender, routed.defender)
    _apply_side_events(queues.reviewer, routed.reviewer)
    _apply_side_events(queues.arbitrator, routed.arbitrator)
    _assert_no_pending_tool_calls(queues)
    return queues


def _apply_side_events(queue: _QueueBase, events: list[_SideEvent]) -> None:
    for event in events:
        if event.kind == _EventKind.MESSAGE:
            if not isinstance(event.item, RequestMessageItem):
                raise TypeError("message event did not contain a RequestMessageItem")
            queue._append_message(_message_item_to_chat(event.item))
            continue
        if event.kind == _EventKind.REASONING:
            if event.reasoning is None:
                raise TypeError("reasoning event did not contain a payload")
            queue.add_reasoning(event.reasoning)
            continue
        if event.kind == _EventKind.FUNCTION_CALL:
            if not isinstance(event.item, RequestFunctionCallItem) or event.call_id is None:
                raise TypeError("function_call event is malformed")
            queue.associate_function_call(event.item, event.call_id)
            continue
        if event.kind == _EventKind.FUNCTION_CALL_OUTPUT:
            if not isinstance(event.item, RequestFunctionCallOutputItem) or event.call_id is None:
                raise TypeError("function_call_output event is malformed")
            queue.associate_function_call_output(event.item, event.call_id, temp=event.temp)
            continue
        if event.kind == _EventKind.FABRICATED_FUNCTION_CALL:
            if not isinstance(queue, _MainQueue):
                raise _tool_replay_error(
                    reason="fabricated_function_call_wrong_side", private_message="fabricated function_call can only route to main"
                )
            if event.call is None:
                raise TypeError("fabricated_function_call event is malformed")
            queue.attach_fabricated_call(event.call)
            continue
        if event.kind == _EventKind.FABRICATED_FUNCTION_CALL_OUTPUT:
            if not isinstance(queue, _MainQueue):
                raise _tool_replay_error(
                    reason="fabricated_function_call_output_wrong_side", private_message="fabricated function_call can only route to main"
                )
            if event.output is None:
                raise TypeError("fabricated_function_call_output event is malformed")
            queue.attach_fabricated_output(event.output)


def _assert_no_pending_tool_calls(queues: _AssociatedQueues) -> None:
    queues.main.assert_no_pending_tool_calls()
    queues.defender.assert_no_pending_tool_calls()
    queues.reviewer.assert_no_pending_tool_calls()
    queues.arbitrator.assert_no_pending_tool_calls()


def _initial_cursors(compaction: CompactionPayload | None) -> dict[str, int]:
    if compaction is None:
        return {"m": 0}
    cursors = {"m": 0, **compaction.cursors}
    if cursors["m"] < 0:
        raise _compaction_replay_error(reason="compaction_cursors_negative", private_message="compaction cursors must be non-negative")
    return cursors


def _open_call_id_or_none(value: str, *, keyring: SealingKeyring) -> SealedCallID | None:
    try:
        return open_call_id(value, keyring=keyring)
    except PlapError:
        return None


def _message_item_to_chat(item: RequestMessageItem) -> StateMessage:
    return StateMessage(role=item.role, content=_message_text(item))


def _message_text(item: RequestMessageItem) -> str:
    if isinstance(item.content, str):
        return item.content
    return " ".join(part.text for part in item.content)


def _function_output_text(item: RequestFunctionCallOutputItem) -> str:
    if isinstance(item.output, str):
        return item.output
    return " ".join(part.text for part in item.output)


def _chat_tool_call(item: RequestFunctionCallItem, call_id: SealedCallID) -> StateToolCall:
    return StateToolCall(
        id=call_id.upstream_tool_call_id,
        name=item.name,
        arguments=item.arguments,
    )
