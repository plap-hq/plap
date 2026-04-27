from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from plap.keyring import SealingKeyring
from plap.responses.contracts import (
    ReasoningItem,
    RequestCompactionItem,
    RequestFunctionCallItem,
    RequestFunctionCallOutputItem,
    RequestMessageItem,
    ResponseCreateRequest,
)
from plap.responses.ingest.sealing import (
    CALL_ID_PREFIX,
    open_call_id,
    open_compaction_payload,
    open_reasoning_payload,
)
from plap.responses.ingest.types import (
    ChatMessage,
    ChatMessageWithOrdinal,
    CompactionPayload,
    IngestedQueues,
    IngestionError,
    ReasoningPayload,
    SealedCallID,
    Side,
    SideMessage,
)
from plap.responses.tools import (
    StaticToolPolicyResolver,
    ToolPolicy,
    ToolPolicyResolver,
)

type _EventKind = Literal[
    "message",
    "reasoning",
    "function_call",
    "function_call_output",
    "fabricated_function_call",
    "fabricated_function_call_output",
]


async def ingest_response_request(
    request: ResponseCreateRequest,
    *,
    keyring: SealingKeyring,
    tool_policy_resolver: ToolPolicyResolver | None = None,
) -> IngestedQueues:
    tool_policies = await _classify_tools(request, tool_policy_resolver)
    input_items = _normalize_input_items(request)
    compaction, remaining = _open_compaction_root(input_items, keyring=keyring)
    decoded = _decode_sealed_items(remaining, keyring=keyring)
    pruned = _prune_temp_debate_globally(decoded)
    routed = _route_items_by_side(pruned)
    queues = _associate_side_queues(compaction, routed)
    return IngestedQueues(
        main=tuple(queues.main.rows),
        reviewer=tuple(queues.reviewer.rows),
        arbitrator=tuple(queues.arbitrator.rows),
        continuation_side=routed.continuation_side,
        compaction=compaction,
        source=() if compaction is None else compaction.source,
        cursors=queues.cursors,
        tool_policies=tool_policies,
    )


async def _classify_tools(
    request: ResponseCreateRequest,
    resolver: ToolPolicyResolver | None,
) -> dict[str, ToolPolicy]:
    return await (resolver or StaticToolPolicyResolver()).resolve(request.tools or [])


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


@dataclass(slots=True)
class _RoutedItems:
    main: list[_SideEvent] = field(default_factory=list)
    reviewer: list[_SideEvent] = field(default_factory=list)
    arbitrator: list[_SideEvent] = field(default_factory=list)
    continuation_side: Side = "main"

    def side(self, side: Side) -> list[_SideEvent]:
        if side == "main":
            return self.main
        if side == "reviewer":
            return self.reviewer
        return self.arbitrator


@dataclass(slots=True)
class _AssociatedQueues:
    main: _MainQueue
    reviewer: _PrivateSideQueue
    arbitrator: _PrivateSideQueue
    cursors: dict[str, int]


class _QueueBase:
    def __init__(self, side: Side) -> None:
        self.side = side
        self._entries: list[ChatMessageWithOrdinal | SideMessage] = []
        self._pending_tool_call_ids: set[str] = set()

    def add_reasoning(self, payload: ReasoningPayload) -> None:
        for message in payload.messages:
            content_hash_value = message.get("content_hash")
            if content_hash_value is None:
                self._append_message(dict(message))
                continue
            if not isinstance(content_hash_value, str):
                raise IngestionError("reasoning content_hash must be a string")
            target = self._message_by_hash(content_hash_value)
            if target is None:
                raise IngestionError("reasoning content_hash target is missing")
            for key, value in message.items():
                if key != "content_hash":
                    target[key] = value

    def associate_function_call(
        self,
        item: RequestFunctionCallItem,
        call_id: SealedCallID,
    ) -> None:
        target = self._message_for_call(call_id)
        tool_calls = target.setdefault("tool_calls", [])
        if not isinstance(tool_calls, list):
            raise IngestionError("target tool_calls is not an array")
        if call_id.tool_call_index < len(tool_calls):
            existing = tool_calls[call_id.tool_call_index]
            if not isinstance(existing, dict):
                raise IngestionError("target tool call is malformed")
            if existing.get("id") != call_id.upstream_tool_call_id:
                raise IngestionError("sealed function_call upstream id mismatch")
            self._mark_pending_tool_call(call_id.upstream_tool_call_id)
            return
        if call_id.tool_call_index != len(tool_calls):
            raise IngestionError("sealed function_call index is not contiguous")
        tool_calls.append(_chat_tool_call(item, call_id))
        self._mark_pending_tool_call(call_id.upstream_tool_call_id)

    def associate_function_call_output(
        self,
        item: RequestFunctionCallOutputItem,
        call_id: SealedCallID,
    ) -> None:
        self._require_tool_call(call_id)
        self._consume_pending_tool_call(call_id.upstream_tool_call_id)
        self._append_tool_output(
            {
                "role": "tool",
                "tool_call_id": call_id.upstream_tool_call_id,
                "content": _function_output_text(item),
            }
        )

    def _append_message(self, message: ChatMessage) -> ChatMessage:
        raise NotImplementedError

    def _append_tool_output(self, message: ChatMessage) -> None:
        self._append_message(message)

    def _message_by_hash(self, hash_value: str) -> ChatMessage | None:
        for entry in reversed(self._entries):
            if entry.content_hash == hash_value:
                return entry.message
        return None

    def _message_for_call(self, call_id: SealedCallID) -> ChatMessage:
        prefix = call_id.content_hash_prefix.hex()
        for entry in reversed(self._entries):
            if (
                entry.content_hash.startswith(prefix)
                and entry.message.get("role") == "assistant"
            ):
                return entry.message
        raise IngestionError("sealed function call content_hash target is missing")

    def _require_tool_call(self, call_id: SealedCallID) -> None:
        target = self._message_for_call(call_id)
        tool_calls = target.get("tool_calls")
        if not isinstance(tool_calls, list):
            raise IngestionError("sealed function_call_output target has no tool_calls")
        if call_id.tool_call_index >= len(tool_calls):
            raise IngestionError("sealed function_call_output index is out of range")
        existing = tool_calls[call_id.tool_call_index]
        if not isinstance(existing, dict):
            raise IngestionError("target tool call is malformed")
        if existing.get("id") != call_id.upstream_tool_call_id:
            raise IngestionError("sealed function_call_output upstream id mismatch")

    def _ensure_no_pending_tool_calls(self) -> None:
        if self._pending_tool_call_ids:
            raise IngestionError(
                "same-side message cannot appear before pending tool outputs"
            )

    def _mark_pending_tool_call(self, call_id: str) -> None:
        if call_id in self._pending_tool_call_ids:
            raise IngestionError("duplicate pending function_call")
        self._pending_tool_call_ids.add(call_id)

    def _consume_pending_tool_call(self, call_id: str) -> None:
        if call_id not in self._pending_tool_call_ids:
            raise IngestionError("function_call_output has no pending function_call")
        self._pending_tool_call_ids.remove(call_id)

    def assert_no_pending_tool_calls(self) -> None:
        if self._pending_tool_call_ids:
            raise IngestionError("function_call is missing function_call_output")


class _MainQueue(_QueueBase):
    def __init__(self, cursors: dict[str, int]) -> None:
        super().__init__("main")
        self._cursors = cursors

    @property
    def rows(self) -> list[ChatMessageWithOrdinal]:
        return [
            entry
            for entry in self._entries
            if isinstance(entry, ChatMessageWithOrdinal)
        ]

    def add_existing_row(self, row: ChatMessageWithOrdinal) -> None:
        self._entries.append(row)

    def add_message(self, message: ChatMessage) -> ChatMessageWithOrdinal:
        return self._append_main_row(message)

    def attach_fabricated_call(self, call: RequestFunctionCallItem) -> None:
        target = self._closest_previous_assistant()
        tool_calls = target.setdefault("tool_calls", [])
        if not isinstance(tool_calls, list):
            raise IngestionError("target tool_calls is not an array")
        tool_calls.append(
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
        )
        self._mark_pending_tool_call(call.call_id)

    def attach_fabricated_output(self, output: RequestFunctionCallOutputItem) -> None:
        self._consume_pending_tool_call(output.call_id)
        self._append_tool_output(
            {
                "role": "tool",
                "tool_call_id": output.call_id,
                "content": _function_output_text(output),
            }
        )

    def _append_message(self, message: ChatMessage) -> ChatMessage:
        return self.add_message(message).message

    def _append_tool_output(self, message: ChatMessage) -> None:
        self._append_main_row(message, allow_pending=True)

    def _append_main_row(
        self, message: ChatMessage, *, allow_pending: bool = False
    ) -> ChatMessageWithOrdinal:
        if not allow_pending:
            self._ensure_no_pending_tool_calls()
        ordinal = self._cursors.get("m", 0)
        row = ChatMessageWithOrdinal(namespace="m", ordinal=ordinal, message=message)
        self._cursors["m"] = ordinal + 1
        self._entries.append(row)
        return row

    def _closest_previous_assistant(self) -> ChatMessage:
        for entry in reversed(self._entries):
            if entry.message.get("role") == "assistant":
                return entry.message
        raise IngestionError(
            "fabricated function_call has no previous assistant message"
        )


class _PrivateSideQueue(_QueueBase):
    def __init__(self, side: Side) -> None:
        if side == "main":
            raise ValueError("main side must use _MainQueue")
        super().__init__(side)

    @property
    def rows(self) -> list[SideMessage]:
        return [entry for entry in self._entries if isinstance(entry, SideMessage)]

    def _append_message(self, message: ChatMessage) -> ChatMessage:
        self._ensure_no_pending_tool_calls()
        row = SideMessage(message=message)
        self._entries.append(row)
        return row.message

    def _append_tool_output(self, message: ChatMessage) -> None:
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


def _decode_sealed_items(
    input_items: list[object], *, keyring: SealingKeyring
) -> list[_DecodedItem]:
    decoded: list[_DecodedItem] = []
    in_temp_debate = False
    for item in input_items:
        if isinstance(item, RequestMessageItem):
            decoded.append(_DecodedItem(item=item))
            continue
        if isinstance(item, ReasoningItem):
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
            decoded.append(
                _DecodedItem(
                    item=item,
                    call_id=call_id,
                    temp_related=in_temp_debate and call_id is not None,
                )
            )
    return decoded


def _open_reasoning_item(
    item: ReasoningItem, *, keyring: SealingKeyring
) -> ReasoningPayload:
    if item.encrypted_content is None:
        raise IngestionError("unsealed reasoning input is not trusted")
    return open_reasoning_payload(item.encrypted_content, keyring=keyring)


def _prune_temp_debate_globally(items: list[_DecodedItem]) -> list[_DecodedItem]:
    retained: list[_DecodedItem] = []
    for item in items:
        if item.reasoning is not None and not item.reasoning.temp:
            retained = [
                candidate for candidate in retained if not candidate.temp_related
            ]
        retained.append(item)
    return retained


def _route_items_by_side(items: list[_DecodedItem]) -> _RoutedItems:
    routed = _RoutedItems()
    pending_unsealed: dict[str, RequestFunctionCallItem] = {}
    for item in items:
        if isinstance(item.item, RequestMessageItem):
            routed.continuation_side = "main"
            routed.main.append(_SideEvent(kind="message", item=item.item))
            continue
        if item.reasoning is not None:
            routed.continuation_side = item.reasoning.side
            routed.side(item.reasoning.side).append(
                _SideEvent(kind="reasoning", reasoning=item.reasoning)
            )
            continue
        if isinstance(item.item, RequestFunctionCallItem):
            if item.call_id is None:
                if item.item.call_id in pending_unsealed:
                    raise IngestionError("duplicate pending unsealed function_call")
                pending_unsealed[item.item.call_id] = item.item
                routed.continuation_side = "main"
                routed.main.append(
                    _SideEvent(kind="fabricated_function_call", call=item.item)
                )
            else:
                routed.continuation_side = item.call_id.side
                routed.side(item.call_id.side).append(
                    _SideEvent(
                        kind="function_call",
                        item=item.item,
                        call_id=item.call_id,
                    )
                )
            continue
        if isinstance(item.item, RequestFunctionCallOutputItem):
            if item.call_id is None:
                call = pending_unsealed.pop(item.item.call_id, None)
                if call is None:
                    raise IngestionError(
                        "unsealed function_call_output has no matching function_call"
                    )
                routed.continuation_side = "main"
                routed.main.append(
                    _SideEvent(kind="fabricated_function_call_output", output=item.item)
                )
            else:
                routed.continuation_side = item.call_id.side
                routed.side(item.call_id.side).append(
                    _SideEvent(
                        kind="function_call_output",
                        item=item.item,
                        call_id=item.call_id,
                    )
                )
    if pending_unsealed:
        raise IngestionError("unsealed function_call is missing function_call_output")
    return routed


def _associate_side_queues(
    compaction: CompactionPayload | None,
    routed: _RoutedItems,
) -> _AssociatedQueues:
    cursors = _initial_cursors(compaction)
    queues = _AssociatedQueues(
        main=_MainQueue(cursors),
        reviewer=_PrivateSideQueue("reviewer"),
        arbitrator=_PrivateSideQueue("arbitrator"),
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
        if event.kind == "message":
            if not isinstance(event.item, RequestMessageItem):
                raise TypeError("message event did not contain a RequestMessageItem")
            queue._append_message(_message_item_to_chat(event.item))
            continue
        if event.kind == "reasoning":
            if event.reasoning is None:
                raise TypeError("reasoning event did not contain a payload")
            queue.add_reasoning(event.reasoning)
            continue
        if event.kind == "function_call":
            if (
                not isinstance(event.item, RequestFunctionCallItem)
                or event.call_id is None
            ):
                raise TypeError("function_call event is malformed")
            queue.associate_function_call(event.item, event.call_id)
            continue
        if event.kind == "function_call_output":
            if (
                not isinstance(event.item, RequestFunctionCallOutputItem)
                or event.call_id is None
            ):
                raise TypeError("function_call_output event is malformed")
            queue.associate_function_call_output(event.item, event.call_id)
            continue
        if event.kind == "fabricated_function_call":
            if not isinstance(queue, _MainQueue):
                raise IngestionError("fabricated function_call can only route to main")
            if event.call is None:
                raise TypeError("fabricated_function_call event is malformed")
            queue.attach_fabricated_call(event.call)
            continue
        if event.kind == "fabricated_function_call_output":
            if not isinstance(queue, _MainQueue):
                raise IngestionError("fabricated function_call can only route to main")
            if event.output is None:
                raise TypeError("fabricated_function_call_output event is malformed")
            queue.attach_fabricated_output(event.output)


def _assert_no_pending_tool_calls(queues: _AssociatedQueues) -> None:
    queues.main.assert_no_pending_tool_calls()
    queues.reviewer.assert_no_pending_tool_calls()
    queues.arbitrator.assert_no_pending_tool_calls()


def _initial_cursors(compaction: CompactionPayload | None) -> dict[str, int]:
    if compaction is None:
        return {"m": 0, "s": 0}
    cursors = {"m": 0, "s": 0, **compaction.cursors}
    if cursors["m"] < 0 or cursors["s"] < 0:
        raise IngestionError("compaction cursors must be non-negative")
    return cursors


def _open_call_id_or_none(
    value: str, *, keyring: SealingKeyring
) -> SealedCallID | None:
    if not value.startswith(CALL_ID_PREFIX):
        return None
    return open_call_id(value, keyring=keyring)


def _message_item_to_chat(item: RequestMessageItem) -> ChatMessage:
    return {"role": item.role, "content": _message_text(item)}


def _message_text(item: RequestMessageItem) -> str:
    if isinstance(item.content, str):
        return item.content
    return " ".join(part.text for part in item.content)


def _function_output_text(item: RequestFunctionCallOutputItem) -> str:
    if isinstance(item.output, str):
        return item.output
    return " ".join(part.text for part in item.output)


def _chat_tool_call(
    item: RequestFunctionCallItem, call_id: SealedCallID
) -> dict[str, Any]:
    return {
        "id": call_id.upstream_tool_call_id,
        "type": "function",
        "function": {"name": item.name, "arguments": item.arguments},
    }
