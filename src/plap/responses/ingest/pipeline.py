from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from plap.keyring import SealingKeyring
from plap.responses.contracts import (
    RequestCompactionItem,
    RequestFunctionCallItem,
    RequestFunctionCallOutputItem,
    RequestMessageItem,
    RequestReasoningItem,
    ResponseCreateRequest,
)
from plap.responses.errors import ResponseError
from plap.responses.ingest.sealing import (
    CALL_ID_PREFIX,
    open_call_id,
    open_compaction_payload,
    open_reasoning_payload,
)
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
    decoded, in_temp_debate = _decode_sealed_items(remaining, keyring=keyring)
    pruned = _prune_temp_debate_globally(decoded)
    routed = _route_items_by_side(pruned)
    queues = _associate_side_queues(compaction, routed)
    return IngestedQueues(
        main_context=tuple(queues.main.context_rows),
        main_context_temp=tuple(queues.main.context_temp_rows),
        reviewer=tuple(queues.reviewer.rows),
        arbitrator=tuple(queues.arbitrator.rows),
        continuation_side=routed.continuation_side,
        in_temp_debate=in_temp_debate,
        compaction=compaction,
        cursors=queues.cursors,
    )


@dataclass(slots=True)
class _DecodedItem:
    item: object
    reasoning: ReasoningPayload | None = None
    call_id: SealedCallID | None = None
    temp_related: bool = False
    resets_temp_debate: bool = False


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
    reviewer: list[_SideEvent] = field(default_factory=list)
    arbitrator: list[_SideEvent] = field(default_factory=list)
    continuation_side: Side = Side.MAIN

    def side(self, side: Side) -> list[_SideEvent]:
        if side == Side.MAIN:
            return self.main
        if side == Side.REVIEWER:
            return self.reviewer
        return self.arbitrator


@dataclass(slots=True)
class _AssociatedQueues:
    main: _MainQueue
    reviewer: _PrivateSideQueue
    arbitrator: _PrivateSideQueue
    cursors: dict[str, int]


@dataclass(frozen=True, slots=True)
class _ExpansionCandidate:
    index: int
    summary_fidelity: int
    token_delta: int
    start: int
    end: int


class _QueueBase:
    def __init__(self, side: Side) -> None:
        self.side = side
        self._entries: list[ChatMessageSpan | SideMessage] = []
        self._pending_tool_call_ids: set[str] = set()
        self._unreplayed_reasoning_tool_call_ids: set[str] = set()
        self._reasoning_tool_call_ids_seen: set[str] = set()

    def add_reasoning(self, payload: ReasoningPayload) -> None:
        for message in payload.messages:
            if isinstance(message, StateMessage):
                appended = self._append_message(message, temp=payload.temp)
                self._track_reasoning_message(appended)
                continue
            if not isinstance(message, ReasoningMessagePatch):
                raise ResponseError.ingestion(private_message="reasoning message is invalid")
            target = self._message_by_hash(message.content_hash)
            if target is None:
                raise ResponseError.ingestion(private_message="reasoning content_hash target is missing")
            message.apply_to(target)
            if message.tool_calls is not None or message.role == "tool" or (target.is_tool() and message.tool_call_id is not None):
                self._track_reasoning_message(target)

    def associate_function_call(
        self,
        item: RequestFunctionCallItem,
        call_id: SealedCallID,
    ) -> None:
        target = self._message_for_call(call_id)
        if call_id.tool_call_index < len(target.tool_calls):
            existing = target.tool_call_at(call_id.tool_call_index)
            if existing.id != call_id.upstream_tool_call_id:
                raise ResponseError.ingestion(private_message="sealed function_call upstream id mismatch")
            self._mark_reasoning_tool_call_replayed(call_id.upstream_tool_call_id)
            self._mark_pending_tool_call(call_id.upstream_tool_call_id)
            return
        if call_id.tool_call_index != len(target.tool_calls):
            raise ResponseError.ingestion(private_message="sealed function_call index is not contiguous")
        target.append_tool_call(_chat_tool_call(item, call_id))
        self._mark_pending_tool_call(call_id.upstream_tool_call_id)

    def associate_function_call_output(
        self,
        item: RequestFunctionCallOutputItem,
        call_id: SealedCallID,
        *,
        temp: bool = False,
    ) -> None:
        self._require_tool_call(call_id)
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

    def _message_by_hash(self, hash_value: str) -> StateMessage | None:
        for entry in reversed(self._entries):
            if entry.content_hash == hash_value:
                return entry.message
        return None

    def _message_for_call(self, call_id: SealedCallID) -> StateMessage:
        prefix = call_id.content_hash_prefix.hex()
        for entry in reversed(self._entries):
            if entry.content_hash.startswith(prefix) and entry.message.is_assistant():
                return entry.message
        raise ResponseError.ingestion(private_message="sealed function call content_hash target is missing")

    def _require_tool_call(self, call_id: SealedCallID) -> None:
        target = self._message_for_call(call_id)
        if not target.tool_calls:
            raise ResponseError.ingestion(private_message="sealed function_call_output target has no tool_calls")
        if call_id.tool_call_index >= len(target.tool_calls):
            raise ResponseError.ingestion(private_message="sealed function_call_output index is out of range")
        existing = target.tool_call_at(call_id.tool_call_index)
        if existing.id != call_id.upstream_tool_call_id:
            raise ResponseError.ingestion(private_message="sealed function_call_output upstream id mismatch")

    def _ensure_no_pending_tool_calls(self) -> None:
        if self._pending_tool_call_ids:
            raise ResponseError.ingestion(private_message="same-side message cannot appear before pending tool outputs")

    def _mark_pending_tool_call(self, call_id: str) -> None:
        if call_id in self._pending_tool_call_ids:
            raise ResponseError.ingestion(private_message="duplicate pending function_call")
        self._pending_tool_call_ids.add(call_id)

    def _mark_reasoning_tool_call_replayed(self, call_id: str) -> None:
        self._unreplayed_reasoning_tool_call_ids.discard(call_id)

    def _consume_pending_tool_call(self, call_id: str) -> None:
        if call_id not in self._pending_tool_call_ids:
            raise ResponseError.ingestion(private_message="function_call_output has no pending function_call")
        self._pending_tool_call_ids.remove(call_id)

    def _track_reasoning_tool_calls(self, message: StateMessage) -> None:
        if not message.is_assistant():
            return
        if not message.tool_calls:
            return
        for tool_call in message.tool_calls:
            tool_call_id = tool_call.id
            if not tool_call_id:
                raise ResponseError.ingestion(private_message="reasoning tool call id is missing")
            if tool_call_id in self._reasoning_tool_call_ids_seen:
                raise ResponseError.ingestion(private_message="duplicate reasoning tool call")
            self._reasoning_tool_call_ids_seen.add(tool_call_id)
            self._unreplayed_reasoning_tool_call_ids.add(tool_call_id)

    def _satisfy_reasoning_tool_call_from_hidden_output(self, message: StateMessage) -> None:
        if not message.is_tool():
            return
        tool_call_id = message.tool_call_id
        if not tool_call_id:
            raise ResponseError.ingestion(private_message="reasoning tool output tool_call_id is missing")
        if tool_call_id not in self._unreplayed_reasoning_tool_call_ids:
            raise ResponseError.ingestion(private_message="reasoning tool output has no pending tool call")
        self._unreplayed_reasoning_tool_call_ids.remove(tool_call_id)

    def _track_reasoning_message(self, message: StateMessage) -> None:
        self._track_reasoning_tool_calls(message)
        self._satisfy_reasoning_tool_call_from_hidden_output(message)

    def assert_no_pending_tool_calls(self) -> None:
        if self._unreplayed_reasoning_tool_call_ids:
            raise ResponseError.ingestion(private_message="reasoning tool call is missing function_call item")
        if self._pending_tool_call_ids:
            raise ResponseError.ingestion(private_message="function_call is missing function_call_output")


class _MainQueue(_QueueBase):
    def __init__(self, cursors: dict[str, int]) -> None:
        super().__init__(Side.MAIN)
        self._cursors = cursors
        self._seed_rows: list[ChatMessageSpan] = []
        self._stable_rows: list[ChatMessageSpan] = []
        self._temp_rows: list[ChatMessageSpan] = []

    @property
    def context_rows(self) -> list[ChatMessageSpan]:
        return [*self._seed_rows, *self._stable_rows]

    @property
    def context_temp_rows(self) -> list[ChatMessageSpan]:
        return self._temp_rows

    @property
    def stable_rows(self) -> list[ChatMessageSpan]:
        return self._stable_rows

    @property
    def temp_rows(self) -> list[ChatMessageSpan]:
        return self._temp_rows

    def add_existing_row(self, row: ChatMessageSpan) -> None:
        self._seed_rows.append(row)
        self._entries.append(row)

    def add_message(self, message: StateMessage) -> ChatMessageSpan:
        return self._append_main_row(message)

    def attach_fabricated_call(self, call: RequestFunctionCallItem) -> None:
        target = self._closest_previous_assistant()
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
        ordinal = self._cursors.get("m", 0)
        row = ChatMessageSpan(
            start=ordinal,
            end=ordinal,
            message=message,
            token_count=message.estimated_token_count(),
        )
        self._cursors["m"] = ordinal + 1
        if temp:
            self._temp_rows.append(row)
        else:
            self._stable_rows.append(row)
        self._entries.append(row)
        return row

    def _closest_previous_assistant(self) -> StateMessage:
        for entry in reversed(self._entries):
            if entry.message.is_assistant():
                return entry.message
        raise ResponseError.ingestion(private_message="fabricated function_call has no previous assistant message")


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
        row = SideMessage(message=message)
        self._entries.append(row)
        return row.message

    def _append_tool_output(self, message: StateMessage, *, temp: bool = False) -> None:
        _ = temp
        row = SideMessage(message=message)
        self._entries.append(row)


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


def _decode_sealed_items(input_items: list[object], *, keyring: SealingKeyring) -> tuple[list[_DecodedItem], bool]:
    decoded: list[_DecodedItem] = []
    in_temp_debate = False
    for item in input_items:
        if isinstance(item, RequestMessageItem):
            decoded.append(_DecodedItem(item=item, resets_temp_debate=in_temp_debate))
            if in_temp_debate:
                in_temp_debate = False
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
            in_temp_debate = payload.temp
            continue
        if isinstance(item, RequestFunctionCallItem | RequestFunctionCallOutputItem):
            call_id = _open_call_id_or_none(item.call_id, keyring=keyring)
            resets_temp_debate = in_temp_debate and call_id is None
            decoded.append(
                _DecodedItem(
                    item=item,
                    call_id=call_id,
                    temp_related=in_temp_debate and call_id is not None,
                    resets_temp_debate=resets_temp_debate,
                )
            )
            if resets_temp_debate:
                in_temp_debate = False
    return decoded, in_temp_debate


def _open_reasoning_item(item: RequestReasoningItem, *, keyring: SealingKeyring) -> ReasoningPayload:
    if item.encrypted_content is None:
        raise ResponseError.ingestion(private_message="unsealed reasoning input is not trusted")
    return open_reasoning_payload(item.encrypted_content, keyring=keyring)


def _prune_temp_debate_globally(items: list[_DecodedItem]) -> list[_DecodedItem]:
    retained: list[_DecodedItem] = []
    for item in items:
        if item.resets_temp_debate or (item.reasoning is not None and not item.reasoning.temp):
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
            routed.continuation_side = item.reasoning.continuation_side or item.reasoning.side
            routed.side(item.reasoning.side).append(_SideEvent(kind=_EventKind.REASONING, reasoning=item.reasoning))
            continue
        if isinstance(item.item, RequestFunctionCallItem):
            if item.call_id is None:
                if item.item.call_id in pending_unsealed:
                    raise ResponseError.ingestion(private_message="duplicate pending unsealed function_call")
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
                    raise ResponseError.ingestion(private_message="unsealed function_call_output has no matching function_call")
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
        raise ResponseError.ingestion(private_message="unsealed function_call is missing function_call_output")
    return routed


def _associate_side_queues(
    compaction: CompactionPayload | None,
    routed: _RoutedItems,
) -> _AssociatedQueues:
    cursors = _initial_cursors(compaction)
    queues = _AssociatedQueues(
        main=_MainQueue(cursors),
        reviewer=_PrivateSideQueue(Side.REVIEWER),
        arbitrator=_PrivateSideQueue(Side.ARBITRATOR),
        cursors=cursors,
    )
    if compaction is not None:
        for row in compaction.active:
            queues.main.add_existing_row(row)
    _apply_side_events(queues.main, routed.main)
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
                raise ResponseError.ingestion(private_message="fabricated function_call can only route to main")
            if event.call is None:
                raise TypeError("fabricated_function_call event is malformed")
            queue.attach_fabricated_call(event.call)
            continue
        if event.kind == _EventKind.FABRICATED_FUNCTION_CALL_OUTPUT:
            if not isinstance(queue, _MainQueue):
                raise ResponseError.ingestion(private_message="fabricated function_call can only route to main")
            if event.output is None:
                raise TypeError("fabricated_function_call_output event is malformed")
            queue.attach_fabricated_output(event.output)


def _assert_no_pending_tool_calls(queues: _AssociatedQueues) -> None:
    queues.main.assert_no_pending_tool_calls()
    queues.reviewer.assert_no_pending_tool_calls()
    queues.arbitrator.assert_no_pending_tool_calls()


def _initial_cursors(compaction: CompactionPayload | None) -> dict[str, int]:
    if compaction is None:
        return {"m": 0}
    cursors = {"m": 0, **compaction.cursors}
    if cursors["m"] < 0:
        raise ResponseError.ingestion(private_message="compaction cursors must be non-negative")
    return cursors


def _open_call_id_or_none(value: str, *, keyring: SealingKeyring) -> SealedCallID | None:
    if not value.startswith(CALL_ID_PREFIX):
        return None
    return open_call_id(value, keyring=keyring)


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
