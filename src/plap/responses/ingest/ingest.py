# TODO: this ingestion algorithm still has a key flaw where reasoning is always required to be an unbroken chain (sans compaction nodes).
# Normally, in serving stacks there is a weaker assumption where the reasoning chain is simply unbroken
# since the last user message due to tool call reasoning persistence.
# This can be fixed by re-full-checkpointing at the first reasoning item after every user message,
# but will lead to higher data transfer, storage, and complexity.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import jsonpatch

from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
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
from plap.responses.ingest.models import (
    MAIN_SIDE,
    CallID,
    CompactionPayload,
    Ingested,
    Message,
    MessagePatch,
    ReasoningPayload,
    Side,
    Sides,
    SidesUpdate,
    ToolCall,
    split_tail,
)
from plap.responses.ingest.patch import JSONPatch, JSONValue
from plap.responses.ingest.sealing import open_call_id, open_compaction_payload, open_reasoning_payload

INTERRUPTED_TOOL_OUTPUT = "Tool call aborted by user."


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


def _apply_machine_patch(machine: dict[str, JSONValue], patch: JSONPatch) -> dict[str, JSONValue]:
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


class _Phase(StrEnum):
    DECLARED = "declared"
    OPEN = "open"
    CLOSED = "closed"


@dataclass(slots=True)
class _SideCalls:
    by_id: dict[str, _Phase] = field(default_factory=dict)

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
        return any(phase == _Phase.DECLARED for phase in self.by_id.values())

    def has_open(self) -> bool:
        return any(phase == _Phase.OPEN for phase in self.by_id.values())

    def has_unfinished(self) -> bool:
        return any(phase != _Phase.CLOSED for phase in self.by_id.values())

    def phase(self, call_id: str) -> _Phase | None:
        return self.by_id.get(call_id)

    def register(self, message: Message) -> None:
        for tool_call in message.tool_calls:
            if tool_call.id in self.by_id:
                raise _tool_replay_error(
                    reason="duplicate_tool_call_id_in_history",
                    private_message="tool call id appears more than once in side history",
                )
            self.by_id[tool_call.id] = _Phase.DECLARED

    def settle_output(self, message: Message) -> None:
        tool_call_id = message.tool_call_id
        if tool_call_id is None:
            raise _tool_replay_error(
                reason="function_call_output_without_pending_function_call",
                private_message="function_call_output has no pending function_call",
            )
        phase = self.by_id.get(tool_call_id)
        if phase is None or phase == _Phase.CLOSED:
            raise _tool_replay_error(
                reason="function_call_output_without_pending_function_call",
                private_message="function_call_output has no pending function_call",
            )
        self.by_id[tool_call_id] = _Phase.CLOSED

    def record(self, call: _Call) -> None:
        if call.id in self.by_id:
            raise _tool_replay_error(
                reason="duplicate_tool_call_id_in_history",
                private_message="tool call id appears more than once in side history",
            )
        self.by_id[call.id] = call.phase

    def merge(self, extra: _SideCalls) -> _SideCalls:
        merged = _SideCalls(by_id=dict(self.by_id))
        for call_id, phase in extra.by_id.items():
            if call_id in merged.by_id:
                raise _tool_replay_error(
                    reason="duplicate_tool_call_id_in_history",
                    private_message="tool call id appears more than once in side history",
                )
            merged.by_id[call_id] = phase
        return merged

    def open(self, call_id: str) -> bool:
        phase = self.phase(call_id)
        if phase is None:
            return False
        if phase == _Phase.CLOSED:
            raise _tool_replay_error(
                reason="function_call_already_satisfied",
                private_message="function_call already has a tool output in history",
            )
        if phase == _Phase.OPEN:
            raise _tool_replay_error(
                reason="duplicate_pending_function_call",
                private_message="duplicate pending function_call",
            )
        self.by_id[call_id] = _Phase.OPEN
        return True

    def close(self, call_id: str) -> bool:
        phase = self.phase(call_id)
        if phase is None:
            return False
        if phase != _Phase.OPEN:
            raise _tool_replay_error(
                reason="function_call_output_without_pending_function_call",
                private_message="function_call_output has no pending function_call",
            )
        self.by_id[call_id] = _Phase.CLOSED
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


def _copy_message(
    message: Message,
    *,
    tool_calls: list[ToolCall] | None = None,
    reasoning_content: str | None = None,
) -> Message:
    return Message(
        role=message.role,
        content=message.content,
        name=message.name,
        refusal=message.refusal,
        tool_call_id=message.tool_call_id,
        tool_calls=list(message.tool_calls if tool_calls is None else tool_calls),
        reasoning_content=message.reasoning_content if reasoning_content is None else reasoning_content,
    )


@dataclass(slots=True)
class _Call:
    id: str
    name: str
    arguments: str
    phase: _Phase

    @classmethod
    def declared(cls, tool_call: ToolCall) -> _Call:
        return cls(
            id=tool_call.id,
            name=tool_call.name,
            arguments=tool_call.arguments,
            phase=_Phase.DECLARED,
        )

    @classmethod
    def opened(cls, item: RequestFunctionCallItem, *, call_id: str) -> _Call:
        return cls(
            id=call_id,
            name=item.name,
            arguments=item.arguments,
            phase=_Phase.OPEN,
        )

    def to_tool_call(self) -> ToolCall:
        return ToolCall(id=self.id, name=self.name, arguments=self.arguments)

    def is_open(self) -> bool:
        return self.phase == _Phase.OPEN

    def is_closed(self) -> bool:
        return self.phase == _Phase.CLOSED

    def open(self) -> None:
        if self.phase == _Phase.CLOSED:
            raise _tool_replay_error(
                reason="function_call_already_satisfied",
                private_message="function_call already has a tool output in history",
            )
        if self.phase == _Phase.OPEN:
            raise _tool_replay_error(
                reason="duplicate_pending_function_call",
                private_message="duplicate pending function_call",
            )
        self.phase = _Phase.OPEN

    def settle(self) -> None:
        if self.phase == _Phase.CLOSED:
            raise _tool_replay_error(
                reason="function_call_output_without_pending_function_call",
                private_message="function_call_output has no pending function_call",
            )
        self.phase = _Phase.CLOSED

    def close(self) -> None:
        if self.phase != _Phase.OPEN:
            raise _tool_replay_error(
                reason="function_call_output_without_pending_function_call",
                private_message="function_call_output has no pending function_call",
            )
        self.phase = _Phase.CLOSED


@dataclass(slots=True)
class _Turn:
    assistant: Message
    calls: list[_Call] = field(default_factory=list)
    outputs: list[Message] = field(default_factory=list)

    def call(self, call_id: str) -> _Call | None:
        for call in self.calls:
            if call.id == call_id:
                return call
        return None

    def settled(self) -> bool:
        return all(call.is_closed() for call in self.calls)

    def add_call(self, item: RequestFunctionCallItem, *, call_id: str) -> None:
        self.calls.append(_Call.opened(item, call_id=call_id))

    def add_output(self, message: Message) -> None:
        self.outputs.append(message)

    def render(self) -> list[Message]:
        assistant = _copy_message(
            self.assistant,
            tool_calls=[call.to_tool_call() for call in self.calls],
        )
        return [assistant, *self.outputs]

    def side_calls(self) -> _SideCalls:
        side_calls = _SideCalls()
        for call in self.calls:
            side_calls.record(call)
        return side_calls


type _Cluster = Message | _Turn


@dataclass(slots=True)
class _Anchor:
    assistant: Message
    patch: MessagePatch | None
    declared: list[_Call]
    added: list[_Call]
    outputs: list[Message]

    @classmethod
    def from_hidden(cls, message: Message, suffix: list[Message]) -> _Anchor:
        anchor = cls(
            assistant=message,
            patch=None,
            declared=[_Call.declared(tool_call) for tool_call in message.tool_calls],
            added=[],
            outputs=[],
        )
        anchor.apply_hidden_suffix(suffix)
        return anchor

    @classmethod
    def from_message(cls, message: Message) -> _Anchor:
        return cls(
            assistant=message,
            patch=None,
            declared=[],
            added=[],
            outputs=[],
        )

    def pending(self) -> bool:
        return self.patch is not None

    def call(self, call_id: str) -> _Call | None:
        for call in self.declared:
            if call.id == call_id:
                return call
        for call in self.added:
            if call.id == call_id:
                return call
        return None

    def all_calls(self) -> list[_Call]:
        return [*self.declared, *self.added]

    def message(self) -> Message:
        return _copy_message(
            self.assistant,
            tool_calls=[call.to_tool_call() for call in self.all_calls()],
        )

    def released(self) -> bool:
        return all(call.is_closed() for call in self.declared)

    def add_call(self, item: RequestFunctionCallItem, *, call_id: str) -> None:
        self.added.append(_Call.opened(item, call_id=call_id))

    def add_output(self, message: Message) -> None:
        self.outputs.append(message)

    def apply_hidden_suffix(self, suffix: list[Message]) -> None:
        for message in suffix:
            if message.tool_call_id is None:
                raise _reasoning_replay_error(
                    reason="reasoning_message_invalid",
                    private_message="hidden main tool output must include tool_call_id",
                )
            call = self.call(message.tool_call_id)
            if call is None:
                raise _reasoning_replay_error(
                    reason="reasoning_message_invalid",
                    private_message="hidden main tool output does not match an unresolved anchor tool call",
                )
            if call.is_closed():
                raise _reasoning_replay_error(
                    reason="reasoning_message_invalid",
                    private_message="hidden main tool output duplicates an existing output",
                )
            call.settle()
            self.outputs.append(message)

    def stage(self, patch: MessagePatch) -> None:
        if self.pending():
            raise _reasoning_replay_error(
                reason="reasoning_message_patch_invalid",
                private_message="main assistant is already pending public materialization",
            )
        if self.message() != patch.message:
            raise _reasoning_replay_error(
                reason="reasoning_message_patch_mismatch",
                private_message="message patch does not match the final logical main assistant",
            )
        if not content.assistant_output(patch.message):
            raise _reasoning_replay_error(
                reason="reasoning_message_patch_invalid",
                private_message="message patch assistant has no public output",
            )
        self.patch = patch

    def resolves(self, message: Message) -> bool:
        if self.patch is None:
            return False
        output = content.assistant_output(message)
        return bool(output) and output == content.assistant_output(self.patch.message)

    def resolve(self) -> None:
        if self.patch is None:
            raise RuntimeError("message patch resolution requires a pending patch")
        self.patch = None

    def render(self) -> list[Message]:
        if self.pending():
            return []
        return [self.message(), *self.outputs]

    def side_calls(self) -> _SideCalls:
        side_calls = _SideCalls()
        if self.pending():
            return side_calls
        for call in self.all_calls():
            side_calls.record(call)
        return side_calls


@dataclass(slots=True)
class _Bundle:
    anchor: _Anchor
    past_anchor: bool
    before: list[_Cluster] = field(default_factory=list)
    after: list[_Cluster] = field(default_factory=list)

    def current(self) -> list[_Cluster]:
        return self.after if self.past_anchor else self.before

    def _last_turn(self, clusters: list[_Cluster]) -> _Turn | None:
        for cluster in reversed(clusters):
            if isinstance(cluster, _Turn):
                return cluster
        return None

    def _render_clusters(self, clusters: list[_Cluster]) -> list[Message]:
        rendered: list[Message] = []
        for cluster in clusters:
            if isinstance(cluster, _Turn):
                rendered.extend(cluster.render())
                continue
            rendered.append(cluster)
        return rendered

    def _cluster_calls(self, clusters: list[_Cluster]) -> _SideCalls:
        side_calls = _SideCalls()
        for cluster in clusters:
            if not isinstance(cluster, _Turn):
                continue
            side_calls = side_calls.merge(cluster.side_calls())
        return side_calls

    def released(self) -> bool:
        return not self.anchor.pending() and self.anchor.released()

    def settled(self) -> bool:
        if self.anchor.pending():
            return False
        if not all(call.is_closed() for call in self.anchor.all_calls()):
            return False
        return all(not isinstance(cluster, _Turn) or cluster.settled() for cluster in [*self.before, *self.after])

    def can_handoff(self) -> bool:
        return self.past_anchor and self.released() and self.settled()

    def append_message(self, message: Message) -> None:
        if message.is_assistant():
            self.current().append(_Turn(assistant=message))
            return
        self.current().append(message)

    def owner(self, call_id: str) -> tuple[_Turn | _Anchor, _Call] | None:
        for cluster in reversed(self.after):
            if not isinstance(cluster, _Turn):
                continue
            call = cluster.call(call_id)
            if call is not None:
                return cluster, call
        if not self.anchor.pending():
            call = self.anchor.call(call_id)
            if call is not None:
                return self.anchor, call
        for cluster in reversed(self.before):
            if not isinstance(cluster, _Turn):
                continue
            call = cluster.call(call_id)
            if call is not None:
                return cluster, call
        return None

    def pending_owner(self, call_id: str) -> tuple[_Turn | _Anchor, _Call] | None:
        owner = self.owner(call_id)
        if owner is None:
            return None
        call_owner, call = owner
        if not call.is_open():
            return None
        return call_owner, call

    def fabricated_owner(self) -> _Turn | _Anchor | None:
        turn = self._last_turn(self.current())
        if turn is not None:
            return turn
        if not self.anchor.pending():
            return self.anchor
        return None

    def sealed_owner(self) -> _Turn | _Anchor | None:
        turn = self._last_turn(self.current())
        if turn is not None:
            return turn
        if not self.anchor.pending():
            return self.anchor
        return None

    def render(self) -> list[Message]:
        return [*self._render_clusters(self.before), *self.anchor.render(), *self._render_clusters(self.after)]

    def side_calls(self) -> _SideCalls:
        return self.anchor.side_calls().merge(self._cluster_calls(self.before).merge(self._cluster_calls(self.after)))


@dataclass(slots=True)
class _Main:
    committed: list[Message] = field(default_factory=list)
    bundle: _Bundle | None = None

    def load_attachable_snapshot(self, messages: list[Message]) -> None:
        before, anchor, suffix, after = split_tail(messages)
        self.committed = list(before)
        if anchor is None:
            self.committed.extend(after)
            self.bundle = None
            return
        if not isinstance(anchor, Message):  # pragma: no cover
            raise TypeError("main snapshot anchor must be a message")
        self.bundle = _Bundle(anchor=_Anchor.from_hidden(anchor, suffix), past_anchor=True, after=list(after))

    def _finalize(self) -> None:
        if self.bundle is None:
            return
        if self.bundle.anchor.pending():
            raise _reasoning_replay_error(
                reason="main_message_patch_target_missing",
                private_message="main message patch target is missing",
            )
        self.committed.extend(self.bundle.render())
        self.bundle = None

    def current_messages(self) -> list[Message]:
        if self.bundle is None:
            return list(self.committed)
        return [*self.committed, *self.bundle.render()]

    def current_calls(self) -> _SideCalls:
        committed_calls = _SideCalls.rebuild(self.committed)
        if self.bundle is None:
            return committed_calls
        return committed_calls.merge(self.bundle.side_calls())

    def validate(self) -> None:
        if self.bundle is not None and self.bundle.anchor.pending():
            raise _reasoning_replay_error(
                reason="main_message_patch_target_missing",
                private_message="main message patch target is missing",
            )
        _SideCalls.rebuild(self.current_messages())

    def commit_before_reasoning(self) -> None:
        self.validate()
        self._finalize()

    def _stage_patch(self, patch: MessagePatch) -> None:
        if self.bundle is None or self.bundle.after:
            raise _reasoning_replay_error(
                reason="main_message_patch_target_missing",
                private_message="message patch requires a final logical main assistant",
            )
        anchor = self.bundle.anchor
        if anchor.outputs and all(call.is_closed() for call in anchor.all_calls()):
            raise _reasoning_replay_error(
                reason="main_message_patch_target_missing",
                private_message="message patch cannot target a fully settled hidden assistant",
            )
        anchor.stage(patch)
        self.bundle.past_anchor = False

    def apply_main_update(self, baseline: list[Message], sides: SidesUpdate) -> None:
        self.load_attachable_snapshot(baseline)
        leading_outputs, prefix, anchor, suffix, after, patch = sides.split_main()
        if leading_outputs and self.bundle is None:
            raise _reasoning_replay_error(
                reason="reasoning_message_invalid",
                private_message="leading main outputs require a post-patch baseline assistant anchor",
            )
        if leading_outputs:
            if self.bundle is None:  # pragma: no cover - narrowed above
                raise RuntimeError("leading main outputs require a baseline bundle")
            self.bundle.anchor.apply_hidden_suffix(leading_outputs)
            self.bundle.past_anchor = True

        if prefix or anchor is not None or after:
            if self.bundle is not None and not self.bundle.can_handoff():
                raise _reasoning_replay_error(
                    reason="reasoning_message_invalid",
                    private_message="main update cannot cross unresolved baseline calls",
                )
            self._finalize()
            self.committed.extend(prefix)
            if anchor is None:
                self.committed.extend(after)
            else:
                self.bundle = _Bundle(
                    anchor=_Anchor.from_hidden(anchor, suffix),
                    past_anchor=bool(suffix or after),
                    after=list(after),
                )

        if patch is not None:
            self._stage_patch(patch)

    def _append_message(self, message: Message) -> None:
        if self.bundle is None:
            if message.is_assistant():
                self.bundle = _Bundle(anchor=_Anchor.from_message(message), past_anchor=True)
                return
            self.committed.append(message)
            return
        anchor = self.bundle.anchor
        if anchor.pending():
            if message.is_assistant() and anchor.resolves(message):
                anchor.resolve()
                self.bundle.past_anchor = True
                return
            self.bundle.append_message(message)
            return
        if message.is_assistant() and self.bundle.can_handoff():
            self._finalize()
            self.bundle = _Bundle(anchor=_Anchor.from_message(message), past_anchor=True)
            return
        self.bundle.append_message(message)

    def _fabricated_call(self, item: RequestFunctionCallItem) -> None:
        if self.bundle is None:
            raise _tool_replay_error(
                reason="fabricated_function_call_without_previous_assistant",
                private_message="fabricated function_call has no previous assistant message",
            )
        existing = self.bundle.owner(item.call_id)
        if existing is not None:
            existing[1].open()
            if isinstance(existing[0], _Anchor):
                self.bundle.past_anchor = True
            return
        owner = self.bundle.fabricated_owner()
        if owner is None:
            raise _tool_replay_error(
                reason="fabricated_function_call_without_previous_assistant",
                private_message="fabricated function_call has no previous assistant message",
            )
        owner.add_call(item, call_id=item.call_id)
        if isinstance(owner, _Anchor):
            self.bundle.past_anchor = True

    def _sealed_call(self, item: RequestFunctionCallItem, call_id: CallID) -> None:
        if self.bundle is None:
            raise _tool_replay_error(
                reason="sealed_function_call_without_attachment_owner",
                private_message="sealed function call has no attachment owner",
            )
        existing = self.bundle.owner(call_id.upstream_tool_call_id)
        if existing is not None:
            existing[1].open()
            if isinstance(existing[0], _Anchor):
                self.bundle.past_anchor = True
            return
        owner = self.bundle.sealed_owner()
        if owner is None:
            raise _tool_replay_error(
                reason="sealed_function_call_without_attachment_owner",
                private_message="sealed function call has no attachment owner",
            )
        owner.add_call(item, call_id=call_id.upstream_tool_call_id)
        if isinstance(owner, _Anchor):
            self.bundle.past_anchor = True

    def _output(self, item: RequestFunctionCallOutputItem, *, call_id: str) -> None:
        if self.bundle is None:
            raise _tool_replay_error(
                reason="function_call_output_without_pending_function_call",
                private_message="function_call_output has no pending function_call",
            )
        owner = self.bundle.pending_owner(call_id)
        if owner is None:
            raise _tool_replay_error(
                reason="function_call_output_without_pending_function_call",
                private_message="function_call_output has no pending function_call",
            )
        owner_state, call = owner
        owner_state.add_output(Message(role="tool", tool_call_id=call_id, content=content.tool_output(item)))
        call.close()
        if isinstance(owner_state, _Anchor):
            self.bundle.past_anchor = True

    def add_standalone_main_item(self, item: _DecodedInput) -> None:
        if isinstance(item, _DecodedMessage):
            self._append_message(item.message)
            return
        if isinstance(item, _DecodedSealedFunctionCall):
            self._sealed_call(item.item, item.call_id)
            return
        if isinstance(item, _DecodedSealedFunctionCallOutput):
            self._output(item.item, call_id=item.call_id.upstream_tool_call_id)
            return
        if isinstance(item, _DecodedFabricatedFunctionCall):
            self._fabricated_call(item.item)
            return
        if isinstance(item, _DecodedFabricatedFunctionCallOutput):
            self._output(item.item, call_id=item.item.call_id)
            return
        raise TypeError(f"unsupported standalone main item: {type(item).__name__}")

    def interrupt_parked_calls(self, *, output: str) -> None:
        self.validate()
        calls = self.current_calls()
        if calls.has_open():
            calls.validate_completion(active=True)
        if not calls.has_declared():
            self._finalize()
            return
        if self.bundle is None:
            self.load_attachable_snapshot(self.committed)
        if self.bundle is None:  # pragma: no cover - stable declarations require an assistant anchor
            raise RuntimeError("parked main calls require an assistant anchor")
        parked = [call for call in self.bundle.anchor.declared if call.phase == _Phase.DECLARED]
        declared_ids = {call_id for call_id, phase in calls.by_id.items() if phase == _Phase.DECLARED}
        if {call.id for call in parked} != declared_ids:
            raise _tool_replay_error(
                reason="parked_main_call_not_on_final_anchor",
                private_message="parked main calls must belong to the final assistant anchor",
            )
        for call in parked:
            self.bundle.anchor.add_output(Message(role="tool", tool_call_id=call.id, content=output))
            call.settle()
        self._finalize()


@dataclass(slots=True)
class _Replay:
    machine: dict[str, JSONValue]
    sides: Sides
    allowed_sides: set[Side]
    calls_by_side: dict[Side, _SideCalls] = field(default_factory=dict)
    main: _Main = field(default_factory=_Main)
    current_compaction_id: str | None = None
    last_reasoning_id: str | None = None

    def _sync_main(self) -> None:
        messages = self.main.current_messages()
        if messages or MAIN_SIDE in self.sides.messages:
            self.sides[MAIN_SIDE] = messages
        self.calls_by_side[MAIN_SIDE] = self.main.current_calls()

    def _rebuild_non_main_calls(self, side: Side) -> None:
        self.calls_by_side[side] = _SideCalls.rebuild(self.sides.get(side, []) or [])

    def _rebuild_all_calls(self) -> None:
        self.calls_by_side = {}
        self._sync_main()
        for side in self.sides.messages:
            if side == MAIN_SIDE:
                continue
            self._rebuild_non_main_calls(side)
        self.main.validate()

    def _validate_side_patch(self, side: Side, shape: JSONValue) -> None:
        if self.sides.shape(side) != shape:
            raise _reasoning_replay_error(
                reason="reasoning_sides_shape_mismatch",
                private_message=f"reasoning {side} shape does not match replay history",
            )

    def _validate_sides(self) -> None:
        unknown = (self.sides.active | set(self.sides.messages)) - self.allowed_sides
        if unknown:
            names = ", ".join(sorted(unknown))
            raise _reasoning_replay_error(
                reason="reasoning_active_side_invalid",
                private_message=f"reasoning activates unconfigured sides: {names}",
            )

    def _validate_reasoning(self, payload: ReasoningPayload) -> None:
        for side, side_calls in self.calls_by_side.items():
            if side_calls.has_open() or (side in self.sides.active and side_calls.has_declared()):
                raise _tool_replay_error(
                    reason="pending_tool_outputs_block_message",
                    private_message="reasoning cannot appear before active function calls are closed",
                )

        if payload.previous_reasoning_id != self.last_reasoning_id:
            raise _reasoning_replay_error(
                reason="reasoning_previous_reasoning_id_mismatch",
                private_message="reasoning payload previous_reasoning_id does not match replay history",
            )
        if payload.previous_compaction_id != self.current_compaction_id:
            raise _reasoning_replay_error(
                reason="reasoning_previous_compaction_id_mismatch",
                private_message="reasoning payload previous_compaction_id does not match replay history",
            )

    def _step_compaction(self, payload: CompactionPayload) -> None:
        self.machine = dict(payload.machine)
        self.sides = Sides.from_primitive(payload.sides.to_primitive())
        self._validate_sides()
        self.main.load_attachable_snapshot(self.sides.get(MAIN_SIDE) or [])
        self._rebuild_all_calls()
        if any(side_calls.has_unfinished() for side_calls in self.calls_by_side.values()):
            raise _compaction_replay_error(
                reason="compaction_contains_unresolved_tool_call",
                private_message="compaction side histories must not contain unresolved tool calls",
            )
        self.current_compaction_id = payload.id
        self.last_reasoning_id = None

    def _step_reasoning(self, payload: ReasoningPayload) -> None:
        self._validate_reasoning(payload)
        self.main.commit_before_reasoning()
        self._sync_main()
        for side, guarded in payload.sides.patches.items():
            self._validate_side_patch(side, guarded.shape)
        self.machine = _apply_machine_patch(self.machine, payload.machine)
        for side, guarded in payload.sides.patches.items():
            patch = guarded.patch
            if patch is None:
                continue
            if side == MAIN_SIDE:
                next_main = [] if not patch else _apply_side_patch(self.sides.get(MAIN_SIDE, []) or [], patch, side=side)
                self.sides[MAIN_SIDE] = next_main
                continue
            self.sides[side] = [] if not patch else _apply_side_patch(self.sides.get(side, []) or [], patch, side=side)
            self._rebuild_non_main_calls(side)
        if payload.sides.active is not None:
            self.sides.active = set(payload.sides.active)
        self._validate_sides()
        self.main.apply_main_update(self.sides.get(MAIN_SIDE, []) or [], payload.sides)
        self._sync_main()
        self.last_reasoning_id = payload.id

    def _step_non_main_function_call(self, call_id: CallID) -> None:
        side_calls = self.calls_by_side.get(call_id.side)
        if side_calls is None:
            return
        if side_calls.phase(call_id.upstream_tool_call_id) is None:
            return
        if call_id.side not in self.sides.active:
            raise _tool_replay_error(
                reason="inactive_side_function_call",
                private_message="public function_call belongs to an inactive side",
            )
        side_calls.open(call_id.upstream_tool_call_id)

    def _step_non_main_function_call_output(self, item: RequestFunctionCallOutputItem, call_id: CallID) -> None:
        side_calls = self.calls_by_side.get(call_id.side)
        if side_calls is None:
            return
        if side_calls.phase(call_id.upstream_tool_call_id) is None:
            return
        side_calls.close(call_id.upstream_tool_call_id)
        side_messages = self.sides.setdefault(call_id.side)
        side_messages.append(
            Message(
                role="tool",
                tool_call_id=call_id.upstream_tool_call_id,
                content=content.tool_output(item),
            )
        )
        self._rebuild_non_main_calls(call_id.side)

    def _step_standalone_main_item(self, item: _DecodedInput) -> None:
        if isinstance(item, _DecodedMessage) and item.message.role in {"user", "assistant"}:
            if MAIN_SIDE not in self.sides.active:
                self.main.interrupt_parked_calls(output=INTERRUPTED_TOOL_OUTPUT)
            self.sides.active.add(MAIN_SIDE)
        if isinstance(item, _DecodedSealedFunctionCall) and MAIN_SIDE not in self.sides.active:
            phase = self.main.current_calls().phase(item.call_id.upstream_tool_call_id)
            if phase == _Phase.DECLARED:
                # TODO: This ID may come from another branch that activated and published
                # the same parked call. Demote it to a fabricated call with a fresh internal
                # ID so it cannot open or discard the parked work; doing so requires aliasing
                # the subsequent sealed output to the fresh ID for the rest of this request.
                raise _tool_replay_error(
                    reason="inactive_side_function_call",
                    private_message="public function_call belongs to an inactive side",
                )
        self.main.add_standalone_main_item(item)
        self._sync_main()

    def _validate_finish(self) -> None:
        self.main.validate()
        for side, side_calls in self.calls_by_side.items():
            side_calls.validate_completion(active=side in self.sides.active)

    def finish(self) -> Ingested:
        self._validate_finish()
        return Ingested(
            machine=self.machine,
            sides=self.sides,
            last_reasoning_id=self.last_reasoning_id,
            current_compaction_id=self.current_compaction_id,
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
                self._step_standalone_main_item(item)
                return
            self._step_non_main_function_call(item.call_id)
            return
        if isinstance(item, _DecodedSealedFunctionCallOutput):
            if item.call_id.side == MAIN_SIDE:
                self._step_standalone_main_item(item)
                return
            self._step_non_main_function_call_output(item.item, item.call_id)
            return
        self._step_standalone_main_item(item)


def _replay_decoded_queue(items: list[_DecodedInput], *, allowed_sides: set[Side]) -> Ingested:
    replay = _Replay(machine={}, sides=Sides(), allowed_sides=allowed_sides)
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
