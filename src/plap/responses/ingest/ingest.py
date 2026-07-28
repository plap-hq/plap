from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import jsonpatch

from plap.errors import PlapError
from plap.keyring import SealingKeyring
from plap.responses.contracts import (
    RequestCompactionItem,
    RequestFunctionCallItem,
    RequestFunctionCallOutputItem,
    RequestInputItem,
    RequestMessageItem,
    RequestReasoningItem,
    ResponseCreateRequest,
)
from plap.responses.ingest import content
from plap.responses.ingest.errors import (
    compaction_replay_error as _compaction_replay_error,
)
from plap.responses.ingest.errors import (
    reasoning_replay_error as _reasoning_replay_error,
)
from plap.responses.ingest.errors import (
    tool_replay_error as _tool_replay_error,
)
from plap.responses.ingest.main import CallPhase, MainReplay
from plap.responses.ingest.models import (
    MAIN_SIDE,
    CallID,
    CompactionPayload,
    Ingested,
    Message,
    ReasoningCheckpoint,
    ReasoningPatch,
    ReasoningPayload,
    Side,
    Sides,
)
from plap.responses.ingest.patch import JSONPatch, JSONValue
from plap.responses.ingest.sealing import open_call_id, open_compaction_payload, open_reasoning_payload

INTERRUPTED_TOOL_OUTPUT = "Tool call aborted by user."


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


def _decode_message_item(item: RequestMessageItem) -> Message:
    return content.message(item)


def _open_compaction_item(item: RequestCompactionItem, *, keyring: SealingKeyring) -> CompactionPayload:
    payload = open_compaction_payload(item.encrypted_content, keyring=keyring)
    if item.id is not None and item.id != payload.id:
        raise _compaction_replay_error(
            reason="compaction_item_id_mismatch",
            private_message="compaction item id does not match sealed payload id",
        )
    return payload


def _open_reasoning_item(item: RequestReasoningItem, *, keyring: SealingKeyring) -> ReasoningPayload:
    if item.encrypted_content is None:
        raise _reasoning_replay_error(
            reason="unsealed_reasoning_input_untrusted",
            private_message="unsealed reasoning input is not trusted",
        )
    payload = open_reasoning_payload(item.encrypted_content, keyring=keyring)
    if item.id is not None and item.id != payload.id:
        raise _reasoning_replay_error(
            reason="reasoning_item_id_mismatch",
            private_message="reasoning item id does not match sealed payload id",
        )
    return payload


def _open_call_id_or_none(value: str, *, keyring: SealingKeyring, side_codes: dict[str, int]) -> CallID | None:
    try:
        return open_call_id(value, keyring=keyring, side_codes=side_codes)
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


def _decode_item(item: RequestInputItem, *, keyring: SealingKeyring, side_codes: dict[str, int]) -> _DecodedInput:
    if isinstance(item, RequestCompactionItem):
        return _DecodedCompaction(payload=_open_compaction_item(item, keyring=keyring))
    if isinstance(item, RequestReasoningItem):
        return _DecodedReasoning(payload=_open_reasoning_item(item, keyring=keyring))
    if isinstance(item, RequestMessageItem):
        return _DecodedMessage(message=_decode_message_item(item))
    if isinstance(item, RequestFunctionCallItem):
        call_id = _open_call_id_or_none(item.call_id, keyring=keyring, side_codes=side_codes)
        if call_id is None:
            return _DecodedFabricatedFunctionCall(item=item)
        return _DecodedSealedFunctionCall(item=item, call_id=call_id)
    if isinstance(item, RequestFunctionCallOutputItem):
        call_id = _open_call_id_or_none(item.call_id, keyring=keyring, side_codes=side_codes)
        if call_id is None:
            return _DecodedFabricatedFunctionCallOutput(item=item)
        return _DecodedSealedFunctionCallOutput(item=item, call_id=call_id)
    raise TypeError(f"unsupported request input item: {type(item).__name__}")


def _decode_queue(items: list[RequestInputItem], *, keyring: SealingKeyring, side_codes: dict[str, int]) -> list[_DecodedInput]:
    return [_decode_item(item, keyring=keyring, side_codes=side_codes) for item in items]


def _apply_durable_patch(durable: dict[str, JSONValue], patch: JSONPatch) -> dict[str, JSONValue]:
    try:
        result = jsonpatch.apply_patch(durable, patch, in_place=False)
    except (jsonpatch.JsonPatchException, TypeError, ValueError) as exc:
        raise _reasoning_replay_error(reason="reasoning_durable_patch_invalid", private_message=str(exc), cause=exc) from exc
    if not isinstance(result, Mapping):
        raise _reasoning_replay_error(
            reason="reasoning_durable_patch_invalid",
            private_message="reasoning durable patch result must be an object",
        )
    return dict(result)


def _apply_side_patch(messages: list[Message], patch: JSONPatch, *, side: Side) -> list[Message]:
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
class _SideCalls:
    by_id: dict[str, CallPhase] = field(default_factory=dict)

    @classmethod
    def rebuild(cls, messages: list[Message]) -> _SideCalls:
        side_calls = cls()
        for message in messages:
            if not message.is_tool() and side_calls.has_unfinished():
                raise _tool_replay_error(
                    reason="pending_tool_outputs_block_message",
                    private_message="message cannot appear before open function calls are closed",
                )
            if message.is_assistant() and message.tool_calls:
                side_calls.register(message)
                continue
            if message.is_tool():
                side_calls.settle_output(message)
        return side_calls

    def has_declared(self) -> bool:
        return any(phase == CallPhase.DECLARED for phase in self.by_id.values())

    def has_open(self) -> bool:
        return any(phase == CallPhase.OPEN for phase in self.by_id.values())

    def has_unfinished(self) -> bool:
        return any(phase != CallPhase.CLOSED for phase in self.by_id.values())

    def phase(self, call_id: str) -> CallPhase | None:
        return self.by_id.get(call_id)

    def register(self, message: Message) -> None:
        for tool_call in message.tool_calls:
            if tool_call.id in self.by_id:
                raise _tool_replay_error(
                    reason="duplicate_tool_call_id_in_history",
                    private_message="tool call id appears more than once in side history",
                )
            self.by_id[tool_call.id] = CallPhase.DECLARED

    def settle_output(self, message: Message) -> None:
        tool_call_id = message.tool_call_id
        if tool_call_id is None:
            raise _tool_replay_error(
                reason="function_call_output_without_pending_function_call",
                private_message="function_call_output has no pending function_call",
            )
        phase = self.by_id.get(tool_call_id)
        if phase is None or phase == CallPhase.CLOSED:
            raise _tool_replay_error(
                reason="function_call_output_without_pending_function_call",
                private_message="function_call_output has no pending function_call",
            )
        self.by_id[tool_call_id] = CallPhase.CLOSED

    def open(self, call_id: str) -> bool:
        phase = self.phase(call_id)
        if phase is None:
            return False
        if phase == CallPhase.CLOSED:
            raise _tool_replay_error(
                reason="function_call_already_satisfied",
                private_message="function_call already has a tool output in history",
            )
        if phase == CallPhase.OPEN:
            raise _tool_replay_error(
                reason="duplicate_pending_function_call",
                private_message="duplicate pending function_call",
            )
        self.by_id[call_id] = CallPhase.OPEN
        return True

    def close(self, call_id: str) -> bool:
        phase = self.phase(call_id)
        if phase is None:
            return False
        if phase != CallPhase.OPEN:
            raise _tool_replay_error(
                reason="function_call_output_without_pending_function_call",
                private_message="function_call_output has no pending function_call",
            )
        self.by_id[call_id] = CallPhase.CLOSED
        return True

    def validate_completion(self, *, active: bool) -> None:
        if self.has_open():
            raise _tool_replay_error(
                reason="function_call_missing_function_call_output",
                private_message="function_call is missing function_call_output",
            )
        if active and self.has_declared():
            raise _reasoning_replay_error(
                reason="reasoning_tool_call_missing_function_call_item",
                private_message="reasoning tool call is missing function_call item",
            )


@dataclass(slots=True)
class _Replay:
    durable: dict[str, JSONValue]
    sides: Sides
    allowed_sides: set[Side]
    calls_by_side: dict[Side, _SideCalls] = field(default_factory=dict)
    main: MainReplay = field(default_factory=MainReplay)
    last_reasoning_id: str | None = None
    last_compaction_id: str | None = None
    checkpoint_required: bool = False
    deferred_main_interrupt: bool = False

    def _main_calls(self) -> _SideCalls:
        return _SideCalls(by_id=self.main.phases())

    def _sync_main(self) -> None:
        messages = self.main.current_messages()
        if messages or MAIN_SIDE in self.sides.messages:
            self.sides[MAIN_SIDE] = messages

    def _rebuild_non_main_calls(self, side: Side) -> None:
        self.calls_by_side[side] = _SideCalls.rebuild(self.sides.get(side, []) or [])

    def _rebuild_all_non_main_calls(self) -> None:
        self.calls_by_side = {}
        for side in self.sides.messages:
            if side != MAIN_SIDE:
                self._rebuild_non_main_calls(side)

    def _validate_sides(self) -> None:
        unknown = (self.sides.active | set(self.sides.messages)) - self.allowed_sides
        if unknown:
            names = ", ".join(sorted(unknown))
            raise _reasoning_replay_error(
                reason="reasoning_active_side_invalid",
                private_message=f"reasoning activates unconfigured sides: {names}",
            )

    def _validate_call_boundary(self, *, allow_deferred_main: bool) -> None:
        self.main.assert_no_pending_patch()
        main_calls = self._main_calls()
        if main_calls.has_open() or (MAIN_SIDE in self.sides.active and main_calls.has_declared() and not allow_deferred_main):
            raise _tool_replay_error(
                reason="pending_tool_outputs_block_message",
                private_message="reasoning cannot appear before active main function calls are closed",
            )
        for side, side_calls in self.calls_by_side.items():
            if side_calls.has_open() or (side in self.sides.active and side_calls.has_declared()):
                raise _tool_replay_error(
                    reason="pending_tool_outputs_block_message",
                    private_message="reasoning cannot appear before active function calls are closed",
                )

    def _validate_reasoning(self, payload: ReasoningPayload) -> None:
        is_checkpoint = isinstance(payload.state, ReasoningCheckpoint)
        self._validate_call_boundary(allow_deferred_main=is_checkpoint and self.deferred_main_interrupt)
        if payload.previous_compaction_id != self.last_compaction_id:
            raise _reasoning_replay_error(
                reason="reasoning_previous_compaction_id_mismatch",
                private_message="reasoning payload previous_compaction_id does not match replay history",
            )
        if is_checkpoint:
            if not self.checkpoint_required:
                raise _reasoning_replay_error(
                    reason="reasoning_checkpoint_unexpected",
                    private_message="reasoning checkpoint does not follow a user message",
                )
            if payload.previous_reasoning_id is not None:
                raise _reasoning_replay_error(
                    reason="reasoning_checkpoint_has_predecessor",
                    private_message="reasoning checkpoint must not reference a previous reasoning item",
                )
            return
        if self.checkpoint_required:
            raise _reasoning_replay_error(
                reason="reasoning_checkpoint_required",
                private_message="first reasoning item after a user message must be a checkpoint",
            )
        if payload.previous_reasoning_id != self.last_reasoning_id:
            raise _reasoning_replay_error(
                reason="reasoning_previous_reasoning_id_mismatch",
                private_message="reasoning payload previous_reasoning_id does not match replay history",
            )

    def _step_compaction(self, payload: CompactionPayload) -> None:
        self.durable = dict(payload.durable)
        self.sides = Sides.from_primitive(payload.sides.to_primitive())
        self._validate_sides()
        self.main.load_snapshot(self.sides.get(MAIN_SIDE) or [])
        self._rebuild_all_non_main_calls()
        calls = [self._main_calls(), *self.calls_by_side.values()]
        if any(side_calls.has_unfinished() for side_calls in calls):
            raise _compaction_replay_error(
                reason="compaction_contains_unresolved_tool_call",
                private_message="compaction side histories must not contain unresolved tool calls",
            )
        self.last_reasoning_id = None
        self.last_compaction_id = payload.id
        self.checkpoint_required = False
        self.deferred_main_interrupt = False

    def _apply_checkpoint(self, state: ReasoningCheckpoint) -> None:
        self._sync_main()
        main_present = MAIN_SIDE in self.sides.messages
        main_messages = self.main.current_messages()
        messages = {side: list(side_messages) for side, side_messages in state.sides.items()}
        if main_present or main_messages:
            messages[MAIN_SIDE] = main_messages
        self.durable = dict(state.durable)
        self.sides = Sides(active=set(state.active), messages=messages)
        self._rebuild_all_non_main_calls()

    def _apply_patch(self, state: ReasoningPatch) -> None:
        self.durable = _apply_durable_patch(self.durable, state.durable)
        for side, patch in state.sides.items():
            current = self.sides.get(side, []) or []
            self.sides[side] = list(current) if not patch else _apply_side_patch(current, patch, side=side)
            self._rebuild_non_main_calls(side)
        if state.active is not None:
            self.sides.active = set(state.active)

    def _step_reasoning(self, payload: ReasoningPayload) -> None:
        self._validate_reasoning(payload)
        if isinstance(payload.state, ReasoningCheckpoint):
            if self.deferred_main_interrupt and not self.main.patch_matches_authenticated(payload.main):
                self.main.interrupt_declared(output=INTERRUPTED_TOOL_OUTPUT)
            self.deferred_main_interrupt = False
            self._apply_checkpoint(payload.state)
            self.checkpoint_required = False
        else:
            self._apply_patch(payload.state)
        self._validate_sides()
        self.main.apply_update(payload.main)
        self._sync_main()
        self.last_reasoning_id = payload.id

    def _step_non_main_function_call(self, call_id: CallID) -> None:
        side_calls = self.calls_by_side.get(call_id.side)
        if side_calls is None or side_calls.phase(call_id.upstream_tool_call_id) is None:
            return
        if call_id.side not in self.sides.active:
            raise _tool_replay_error(
                reason="inactive_side_function_call",
                private_message="public function_call belongs to an inactive side",
            )
        side_calls.open(call_id.upstream_tool_call_id)

    def _step_non_main_function_call_output(self, item: RequestFunctionCallOutputItem, call_id: CallID) -> None:
        side_calls = self.calls_by_side.get(call_id.side)
        if side_calls is None or side_calls.phase(call_id.upstream_tool_call_id) is None:
            return
        side_calls.close(call_id.upstream_tool_call_id)
        self.sides.setdefault(call_id.side).append(
            Message(
                role="tool",
                tool_call_id=call_id.upstream_tool_call_id,
                content=content.tool_output(item),
            )
        )
        self._rebuild_non_main_calls(call_id.side)

    def _activate_main_for_message(self, message: Message) -> None:
        if message.role not in {"user", "assistant"}:
            return
        if MAIN_SIDE not in self.sides.active:
            if message.role == "user" and self._main_calls().has_declared():
                self.deferred_main_interrupt = True
            else:
                self.main.interrupt_declared(output=INTERRUPTED_TOOL_OUTPUT)
        self.sides.active.add(MAIN_SIDE)
        if message.role == "user":
            self.checkpoint_required = True

    def _step_main_item(self, item: _DecodedInput) -> None:
        if isinstance(item, _DecodedMessage):
            self._activate_main_for_message(item.message)
            self.main.append_message(item.message)
        elif isinstance(item, _DecodedSealedFunctionCall):
            phase = self.main.phases().get(item.call_id.upstream_tool_call_id)
            if MAIN_SIDE not in self.sides.active and phase == CallPhase.DECLARED:
                raise _tool_replay_error(
                    reason="inactive_side_function_call",
                    private_message="public function_call belongs to an inactive side",
                )
            self.main.add_call(item.item, call_id=item.call_id.upstream_tool_call_id)
        elif isinstance(item, _DecodedFabricatedFunctionCall):
            self.main.add_call(item.item, call_id=item.item.call_id)
        elif isinstance(item, _DecodedSealedFunctionCallOutput):
            self.main.add_output(item.item, call_id=item.call_id.upstream_tool_call_id)
        elif isinstance(item, _DecodedFabricatedFunctionCallOutput):
            self.main.add_output(item.item, call_id=item.item.call_id)
        else:
            raise TypeError(f"unsupported standalone main item: {type(item).__name__}")
        self._sync_main()

    def _validate_finish(self) -> None:
        if self.deferred_main_interrupt:
            self.main.interrupt_declared(output=INTERRUPTED_TOOL_OUTPUT)
            self.deferred_main_interrupt = False
        self.main.assert_no_pending_patch()
        self._main_calls().validate_completion(active=MAIN_SIDE in self.sides.active)
        for side, side_calls in self.calls_by_side.items():
            side_calls.validate_completion(active=side in self.sides.active)

    def finish(self) -> Ingested:
        self._validate_finish()
        self._sync_main()
        return Ingested(
            durable=self.durable,
            sides=self.sides,
            last_reasoning_id=self.last_reasoning_id,
            last_compaction_id=self.last_compaction_id,
            checkpoint_required=self.checkpoint_required,
        )

    def step(self, item: _DecodedInput) -> None:
        if isinstance(item, _DecodedCompaction):
            self._step_compaction(item.payload)
            return
        if isinstance(item, _DecodedReasoning):
            self._step_reasoning(item.payload)
            return
        if isinstance(item, _DecodedSealedFunctionCall):
            if item.call_id.side == MAIN_SIDE:
                self._step_main_item(item)
            else:
                self._step_non_main_function_call(item.call_id)
            return
        if isinstance(item, _DecodedSealedFunctionCallOutput):
            if item.call_id.side == MAIN_SIDE:
                self._step_main_item(item)
            else:
                self._step_non_main_function_call_output(item.item, item.call_id)
            return
        self._step_main_item(item)


def _replay_decoded_queue(items: list[_DecodedInput], *, allowed_sides: set[Side]) -> Ingested:
    replay = _Replay(durable={}, sides=Sides(), allowed_sides=allowed_sides)
    replay._validate_sides()
    for item in items:
        replay.step(item)
    return replay.finish()


async def ingest_response_request(
    request: ResponseCreateRequest,
    *,
    keyring: SealingKeyring,
    side_codes: dict[str, int],
) -> Ingested:
    input_items = _normalize_input_items(request)
    replay_items = _slice_to_last_compaction(input_items)
    decoded_items = _decode_queue(replay_items, keyring=keyring, side_codes=side_codes)
    return _replay_decoded_queue(decoded_items, allowed_sides=set(side_codes))
