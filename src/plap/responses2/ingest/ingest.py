from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import jsonpatch

from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.keyring import SealingKeyring
from plap.responses2.contracts import (
    RequestCompactionItem,
    RequestFunctionCallItem,
    RequestFunctionCallOutputItem,
    RequestInputItem,
    RequestMessageItem,
    RequestReasoningItem,
    ResponseCreateRequest,
)
from plap.responses2.ingest.models import (
    NON_MAIN_SIDES,
    CallID,
    CompactionPayload,
    Ingested,
    Message,
    MessagePatch,
    ReasoningPayload,
    Side,
    Sides,
)
from plap.responses2.ingest.sealing import content_hash, open_call_id, open_compaction_payload, open_reasoning_payload


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


def _normalize_input_items(request: ResponseCreateRequest) -> list[RequestInputItem]:
    if request.input is None:
        return []
    if isinstance(request.input, str):
        return [RequestMessageItem(content=request.input, role="user", type="message")]
    return list(request.input)


def _last_compaction_index(items: list[RequestInputItem]) -> int | None:
    last_index: int | None = None
    for index, item in enumerate(items):
        if isinstance(item, RequestCompactionItem):
            last_index = index
    return last_index


def _slice_to_last_compaction(items: list[RequestInputItem]) -> list[RequestInputItem]:
    last_index = _last_compaction_index(items)
    if last_index is None:
        return items
    return items[last_index:]


def _message_text(item: RequestMessageItem) -> str:
    if isinstance(item.content, str):
        return item.content
    return " ".join(part.text for part in item.content)


def _function_output_text(item: RequestFunctionCallOutputItem) -> str:
    if isinstance(item.output, str):
        return item.output
    return " ".join(part.text for part in item.output)


def _decode_message_item(item: RequestMessageItem) -> Message:
    return Message(role=item.role, content=_message_text(item))


def _open_compaction_item(item: RequestCompactionItem, *, keyring: SealingKeyring) -> CompactionPayload:
    return open_compaction_payload(item.encrypted_content, keyring=keyring)


def _open_reasoning_item(item: RequestReasoningItem, *, keyring: SealingKeyring) -> ReasoningPayload:
    if item.encrypted_content is None:
        raise _reasoning_replay_error(
            reason="unsealed_reasoning_input_untrusted",
            private_message="unsealed reasoning input is not trusted",
        )
    return open_reasoning_payload(item.encrypted_content, keyring=keyring)


def _open_call_id_or_none(value: str, *, keyring: SealingKeyring) -> CallID | None:
    try:
        return open_call_id(value, keyring=keyring)
    except PlapError:
        return None


@dataclass(frozen=True, slots=True)
class _DecodedCompaction:
    payload: CompactionPayload


@dataclass(frozen=True, slots=True)
class _DecodedReasoning:
    payload: ReasoningPayload


@dataclass(frozen=True, slots=True)
class _DecodedMessage:
    message: Message


@dataclass(frozen=True, slots=True)
class _DecodedSealedFunctionCall:
    item: RequestFunctionCallItem
    call_id: CallID


@dataclass(frozen=True, slots=True)
class _DecodedFabricatedFunctionCall:
    item: RequestFunctionCallItem


@dataclass(frozen=True, slots=True)
class _DecodedSealedFunctionCallOutput:
    item: RequestFunctionCallOutputItem
    call_id: CallID


@dataclass(frozen=True, slots=True)
class _DecodedFabricatedFunctionCallOutput:
    item: RequestFunctionCallOutputItem


type _DecodedInput = (
    _DecodedCompaction
    | _DecodedReasoning
    | _DecodedMessage
    | _DecodedSealedFunctionCall
    | _DecodedFabricatedFunctionCall
    | _DecodedSealedFunctionCallOutput
    | _DecodedFabricatedFunctionCallOutput
)


def _decode_item(item: RequestInputItem, *, keyring: SealingKeyring) -> _DecodedInput:
    if isinstance(item, RequestCompactionItem):
        return _DecodedCompaction(payload=_open_compaction_item(item, keyring=keyring))
    if isinstance(item, RequestReasoningItem):
        return _DecodedReasoning(payload=_open_reasoning_item(item, keyring=keyring))
    if isinstance(item, RequestMessageItem):
        return _DecodedMessage(message=_decode_message_item(item))
    if isinstance(item, RequestFunctionCallItem):
        call_id = _open_call_id_or_none(item.call_id, keyring=keyring)
        if call_id is None:
            return _DecodedFabricatedFunctionCall(item=item)
        return _DecodedSealedFunctionCall(item=item, call_id=call_id)
    if isinstance(item, RequestFunctionCallOutputItem):
        call_id = _open_call_id_or_none(item.call_id, keyring=keyring)
        if call_id is None:
            return _DecodedFabricatedFunctionCallOutput(item=item)
        return _DecodedSealedFunctionCallOutput(item=item, call_id=call_id)
    raise TypeError(f"unsupported request input item: {type(item).__name__}")


def _decode_queue(items: list[RequestInputItem], *, keyring: SealingKeyring) -> list[_DecodedInput]:
    return [_decode_item(item, keyring=keyring) for item in items]


def _apply_machine_patch(machine: dict[str, object], patch: list[dict[str, object]]) -> dict[str, object]:
    try:
        result = jsonpatch.apply_patch(machine, patch, in_place=False)
    except (jsonpatch.JsonPatchException, TypeError, ValueError) as exc:
        raise _reasoning_replay_error(reason="reasoning_machine_patch_invalid", private_message=str(exc), cause=exc) from exc
    if not isinstance(result, Mapping):
        raise _reasoning_replay_error(
            reason="reasoning_machine_patch_invalid",
            private_message="reasoning machine patch result must be an object",
        )
    return dict(result)


def _apply_side_patch(messages: list[Message], patch: list[dict[str, object]], *, side: Side) -> list[Message]:
    primitives = [message.to_primitive() for message in messages]
    try:
        result = jsonpatch.apply_patch(primitives, patch, in_place=False)
    except (jsonpatch.JsonPatchException, TypeError, ValueError) as exc:
        raise _reasoning_replay_error(
            reason="reasoning_side_patch_invalid",
            private_message=f"{side} patch failed: {exc}",
            cause=exc,
        ) from exc
    if not isinstance(result, list):
        raise _reasoning_replay_error(
            reason="reasoning_side_patch_invalid",
            private_message=f"{side} patch result must be an array",
        )
    rebuilt: list[Message] = []
    for index, item in enumerate(result):
        try:
            rebuilt.append(Message.from_primitive(item))
        except (TypeError, ValueError) as exc:
            raise _reasoning_replay_error(
                reason="reasoning_side_patch_invalid",
                private_message=f"{side} patch result[{index}] is invalid: {exc}",
                cause=exc,
            ) from exc
    return rebuilt


@dataclass(slots=True)
class _TrackedCall:
    anchor_content_hash: str
    tool_call_index: int
    has_function_call_item: bool
    has_output_message: bool


@dataclass(slots=True)
class _SideCalls:
    calls_by_id: dict[str, _TrackedCall] = field(default_factory=dict)


def _empty_calls_by_side() -> dict[Side, _SideCalls]:
    return {side: _SideCalls() for side in Side}


def _is_open_function_call(call: _TrackedCall) -> bool:
    return not call.has_output_message


def _has_open_function_calls(calls_by_side: dict[Side, _SideCalls]) -> bool:
    return any(_is_open_function_call(call) for side_calls in calls_by_side.values() for call in side_calls.calls_by_id.values())


def _call_matches_call_id(call: _TrackedCall, call_id: CallID) -> bool:
    return call.tool_call_index == call_id.tool_call_index and call.anchor_content_hash.startswith(call_id.content_hash_prefix.hex())


def _anchor_has_output_message(side_calls: _SideCalls, anchor_content_hash: str) -> bool:
    return any(call.anchor_content_hash == anchor_content_hash and call.has_output_message for call in side_calls.calls_by_id.values())


def _register_assistant_calls(side_calls: _SideCalls, message: Message) -> None:
    anchor_content_hash = content_hash(message)
    for index, tool_call in enumerate(message.tool_calls):
        if tool_call.id in side_calls.calls_by_id:
            raise _tool_replay_error(
                reason="duplicate_tool_call_id_in_history",
                private_message="tool call id appears more than once in side history",
            )
        side_calls.calls_by_id[tool_call.id] = _TrackedCall(
            anchor_content_hash=anchor_content_hash,
            tool_call_index=index,
            has_function_call_item=False,
            has_output_message=False,
        )


def _mark_output_message(side_calls: _SideCalls, message: Message) -> None:
    tool_call_id = message.tool_call_id
    if tool_call_id is None:
        raise _tool_replay_error(
            reason="function_call_output_without_pending_function_call",
            private_message="function_call_output has no pending function_call",
        )
    call = side_calls.calls_by_id.get(tool_call_id)
    if call is None or call.has_output_message:
        raise _tool_replay_error(
            reason="function_call_output_without_pending_function_call",
            private_message="function_call_output has no pending function_call",
        )
    call.has_output_message = True
    call.has_function_call_item = False


def _rebuild_calls(messages: list[Message]) -> _SideCalls:
    side_calls = _SideCalls()
    for message in messages:
        if message.is_assistant() and message.tool_calls:
            _register_assistant_calls(side_calls, message)
            continue
        if message.is_tool():
            _mark_output_message(side_calls, message)
    return side_calls


def _get_call(side_calls: _SideCalls, call_id: CallID) -> _TrackedCall:
    call = side_calls.calls_by_id.get(call_id.upstream_tool_call_id)
    if call is None or not _call_matches_call_id(call, call_id):
        raise _tool_replay_error(
            reason="sealed_function_call_content_hash_target_missing",
            private_message="sealed function call content_hash target is missing",
        )
    return call


@dataclass(slots=True)
class _Replay:
    machine: dict[str, object]
    sides: Sides
    last_side: Side | None
    calls_by_side: dict[Side, _SideCalls] = field(default_factory=_empty_calls_by_side)

    def _rebuild_all_calls(self) -> None:
        self.calls_by_side = _empty_calls_by_side()
        for side in Side:
            self._rebuild_side_calls(side)

    def _rebuild_side_calls(self, side: Side) -> None:
        self.calls_by_side[side] = _rebuild_calls(self.sides.messages(side))

    def _assert_no_open_function_calls_before_reasoning(self) -> None:
        if _has_open_function_calls(self.calls_by_side):
            raise _tool_replay_error(
                reason="pending_tool_outputs_block_message",
                private_message="reasoning cannot appear before open function calls are closed",
            )

    def _step_compaction(self, payload: CompactionPayload) -> None:
        self.machine = dict(payload.machine)
        self.sides = Sides.from_primitive(payload.sides.to_primitive())
        self._rebuild_all_calls()
        self.last_side = None

    def _apply_main_update(self, update: Message | MessagePatch) -> None:
        if isinstance(update, MessagePatch):
            raise NotImplementedError("responses2 ingest main message patch replay not implemented yet")
        self.sides.main.append(update)

    def _step_reasoning(self, payload: ReasoningPayload) -> None:
        self._assert_no_open_function_calls_before_reasoning()
        self.machine = _apply_machine_patch(self.machine, payload.machine)
        for side in NON_MAIN_SIDES:
            if not payload.sides.others[side]:
                continue
            self.sides.others[side] = _apply_side_patch(self.sides.others[side], payload.sides.others[side], side=side)
            self._rebuild_side_calls(side)
        if payload.sides.main:
            for update in payload.sides.main:
                self._apply_main_update(update)
            self._rebuild_side_calls(Side.MAIN)
        self.last_side = None

    def _step_non_main_function_call(self, call_id: CallID) -> None:
        side_calls = self.calls_by_side[call_id.side]
        call = _get_call(side_calls, call_id)
        if call.has_output_message:
            raise _tool_replay_error(
                reason="function_call_already_satisfied",
                private_message="function_call already has a tool output in history",
            )
        if call.has_function_call_item:
            raise _tool_replay_error(
                reason="duplicate_pending_function_call",
                private_message="duplicate pending function_call",
            )
        if _anchor_has_output_message(side_calls, call.anchor_content_hash):
            raise _tool_replay_error(
                reason="sealed_function_call_after_function_call_output",
                private_message="sealed function_call cannot appear after function_call_output for the same assistant turn",
            )
        call.has_function_call_item = True

    def _step_non_main_function_call_output(self, item: RequestFunctionCallOutputItem, call_id: CallID) -> None:
        side_calls = self.calls_by_side[call_id.side]
        call = _get_call(side_calls, call_id)
        if not call.has_function_call_item or call.has_output_message:
            raise _tool_replay_error(
                reason="function_call_output_without_pending_function_call",
                private_message="function_call_output has no pending function_call",
            )
        self.sides.messages(call_id.side).append(
            Message(
                role="tool",
                tool_call_id=call_id.upstream_tool_call_id,
                content=_function_output_text(item),
            )
        )
        call.has_output_message = True
        call.has_function_call_item = False
        self.last_side = call_id.side

    def _step_main_item(self, item: _DecodedInput) -> None:
        if isinstance(item, _DecodedMessage):
            raise NotImplementedError("responses2 standalone main message replay not implemented yet")
        if isinstance(item, _DecodedSealedFunctionCall):
            raise NotImplementedError("responses2 standalone main function_call replay not implemented yet")
        if isinstance(item, _DecodedSealedFunctionCallOutput):
            raise NotImplementedError("responses2 standalone main function_call_output replay not implemented yet")
        if isinstance(item, _DecodedFabricatedFunctionCall):
            raise NotImplementedError("responses2 standalone main fabricated function_call replay not implemented yet")
        if isinstance(item, _DecodedFabricatedFunctionCallOutput):
            raise NotImplementedError("responses2 standalone main fabricated function_call_output replay not implemented yet")
        raise TypeError(f"unsupported main item: {type(item).__name__}")

    def _validate_all_calls_closed(self) -> None:
        for side_calls in self.calls_by_side.values():
            for call in side_calls.calls_by_id.values():
                if call.has_output_message:
                    continue
                if call.has_function_call_item:
                    raise _tool_replay_error(
                        reason="function_call_missing_function_call_output",
                        private_message="function_call is missing function_call_output",
                    )
                raise _reasoning_replay_error(
                    reason="reasoning_tool_call_missing_function_call_item",
                    private_message="reasoning tool call is missing function_call item",
                )

    def finish(self) -> Ingested:
        self._validate_all_calls_closed()
        return Ingested(
            machine=self.machine,
            sides=self.sides,
            last_side=self.last_side,
        )

    def step(self, item: _DecodedInput) -> None:
        if isinstance(item, _DecodedCompaction):
            self._step_compaction(item.payload)
            return
        if isinstance(item, _DecodedReasoning):
            self._step_reasoning(item.payload)
            return
        if isinstance(item, _DecodedSealedFunctionCall):
            if item.call_id.side == Side.MAIN:
                self._step_main_item(item)
                return
            self._step_non_main_function_call(item.call_id)
            return
        if isinstance(item, _DecodedSealedFunctionCallOutput):
            if item.call_id.side == Side.MAIN:
                self._step_main_item(item)
                return
            self._step_non_main_function_call_output(item.item, item.call_id)
            return
        self._step_main_item(item)


def _replay_decoded_queue(items: list[_DecodedInput]) -> Ingested:
    replay = _Replay(machine={}, sides=Sides(), last_side=None)
    for item in items:
        replay.step(item)
    return replay.finish()


async def ingest_response_request(
    request: ResponseCreateRequest,
    *,
    keyring: SealingKeyring,
) -> Ingested:
    input_items = _normalize_input_items(request)
    replay_items = _slice_to_last_compaction(input_items)
    decoded_items = _decode_queue(replay_items, keyring=keyring)
    return _replay_decoded_queue(decoded_items)
