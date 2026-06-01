from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

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
from plap.responses.ingest.models import (
    NON_MAIN_SIDES,
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
)
from plap.responses.ingest.sealing import open_call_id, open_compaction_payload, open_reasoning_payload
from plap.responses.patch import JSONPatch, JSONValue


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


@dataclass(slots=True)
class _TrackedCall:
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


def _register_assistant_calls(side_calls: _SideCalls, message: Message) -> None:
    for tool_call in message.tool_calls:
        if tool_call.id in side_calls.calls_by_id:
            raise _tool_replay_error(
                reason="duplicate_tool_call_id_in_history",
                private_message="tool call id appears more than once in side history",
            )
        side_calls.calls_by_id[tool_call.id] = _TrackedCall(
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


def _merge_side_calls(base: _SideCalls, extra: _SideCalls) -> _SideCalls:
    merged = _SideCalls(calls_by_id=dict(base.calls_by_id))
    for call_id, call in extra.calls_by_id.items():
        if call_id in merged.calls_by_id:
            raise _tool_replay_error(
                reason="duplicate_tool_call_id_in_history",
                private_message="tool call id appears more than once in side history",
            )
        merged.calls_by_id[call_id] = call
    return merged


def _tracked_call(side_calls: _SideCalls, call_id: str) -> _TrackedCall | None:
    return side_calls.calls_by_id.get(call_id)


@dataclass(slots=True)
class _MainCall:
    id: str
    name: str
    arguments: str
    has_function_call_item: bool
    has_output_message: bool


def _tool_call_from_main_call(call: _MainCall) -> ToolCall:
    return ToolCall(id=call.id, name=call.name, arguments=call.arguments)


def _tracked_call_from_main_call(call: _MainCall) -> _TrackedCall:
    return _TrackedCall(
        has_function_call_item=call.has_function_call_item,
        has_output_message=call.has_output_message,
    )


def _main_call_from_tool_call(tool_call: ToolCall) -> _MainCall:
    return _MainCall(
        id=tool_call.id,
        name=tool_call.name,
        arguments=tool_call.arguments,
        has_function_call_item=False,
        has_output_message=False,
    )


def _copy_message(
    message: Message,
    *,
    tool_calls: list[ToolCall] | None = None,
    reasoning_content: str | None = None,
    reasoning_details: list[object] | None = None,
) -> Message:
    return Message(
        role=message.role,
        content=message.content,
        name=message.name,
        tool_call_id=message.tool_call_id,
        tool_calls=list(message.tool_calls if tool_calls is None else tool_calls),
        reasoning_content=message.reasoning_content if reasoning_content is None else reasoning_content,
        reasoning_details=list(message.reasoning_details if reasoning_details is None else reasoning_details),
    )


@dataclass(slots=True)
class _AssistantCluster:
    assistant: Message
    calls: list[_MainCall] = field(default_factory=list)
    outputs: list[Message] = field(default_factory=list)


type _MainCluster = Message | _AssistantCluster


@dataclass(slots=True)
class _MainAnchor:
    assistant: Message | None
    patch: MessagePatch | None
    hidden_suffix_outputs: list[Message]
    calls: list[_MainCall]
    fabricated_calls: list[_MainCall]
    outputs: list[Message]
    strict_declared_ids: set[str]


@dataclass(slots=True)
class _MainBundle:
    anchor: _MainAnchor
    after_anchor: bool
    pre: list[_MainCluster] = field(default_factory=list)
    post: list[_MainCluster] = field(default_factory=list)


def _pending_patch(anchor: _MainAnchor | None) -> bool:
    return anchor is not None and anchor.assistant is None and anchor.patch is not None


def _build_hidden_anchor(message: Message, suffix_outputs: list[Message]) -> _MainAnchor:
    anchor = _MainAnchor(
        assistant=message,
        patch=None,
        hidden_suffix_outputs=[],
        calls=[_main_call_from_tool_call(tool_call) for tool_call in message.tool_calls],
        fabricated_calls=[],
        outputs=[],
        strict_declared_ids={tool_call.id for tool_call in message.tool_calls},
    )
    _apply_hidden_suffix_outputs_to_anchor(anchor, suffix_outputs)
    return anchor


def _build_patch_anchor(patch: MessagePatch, suffix_outputs: list[Message]) -> _MainAnchor:
    strict_ids = {tool_call.id for tool_call in patch.tool_calls} if patch.tool_calls else set()
    return _MainAnchor(
        assistant=None,
        patch=patch,
        hidden_suffix_outputs=list(suffix_outputs),
        calls=[],
        fabricated_calls=[],
        outputs=[],
        strict_declared_ids=strict_ids,
    )


def _build_public_anchor(message: Message) -> _MainAnchor:
    return _MainAnchor(
        assistant=message,
        patch=None,
        hidden_suffix_outputs=[],
        calls=[],
        fabricated_calls=[],
        outputs=[],
        strict_declared_ids=set(),
    )


def _anchor_call(anchor: _MainAnchor, call_id: str) -> _MainCall | None:
    for call in anchor.calls:
        if call.id == call_id:
            return call
    return None


def _anchor_owned_call(anchor: _MainAnchor, call_id: str) -> _MainCall | None:
    for call in _anchor_all_calls(anchor):
        if call.id == call_id:
            return call
    return None


def _anchor_all_calls(anchor: _MainAnchor) -> list[_MainCall]:
    return [*anchor.calls, *anchor.fabricated_calls]


def _cluster_call(cluster: _AssistantCluster, call_id: str) -> _MainCall | None:
    for call in cluster.calls:
        if call.id == call_id:
            return call
    return None


def _current_clusters(bundle: _MainBundle) -> list[_MainCluster]:
    return bundle.post if bundle.after_anchor else bundle.pre


def _nearest_assistant_cluster(clusters: list[_MainCluster]) -> _AssistantCluster | None:
    for cluster in reversed(clusters):
        if isinstance(cluster, _AssistantCluster):
            return cluster
    return None


def _bundle_released(bundle: _MainBundle) -> bool:
    if _pending_patch(bundle.anchor):
        return False
    if not bundle.anchor.strict_declared_ids:
        return True
    for call_id in bundle.anchor.strict_declared_ids:
        call = _anchor_call(bundle.anchor, call_id)
        if call is None or not call.has_output_message:
            return False
    return True


def _cluster_settled(cluster: _AssistantCluster) -> bool:
    return all(call.has_output_message for call in cluster.calls)


def _bundle_settled(bundle: _MainBundle) -> bool:
    if _pending_patch(bundle.anchor):
        return False
    if not all(call.has_output_message for call in _anchor_all_calls(bundle.anchor)):
        return False
    return all(
        not isinstance(cluster, _AssistantCluster) or _cluster_settled(cluster)
        for cluster in [*bundle.pre, *bundle.post]
    )


def _can_handoff(bundle: _MainBundle) -> bool:
    return bundle.after_anchor and _bundle_released(bundle) and _bundle_settled(bundle)


def _call_owner(bundle: _MainBundle, call_id: str) -> tuple[_AssistantCluster | _MainAnchor, _MainCall] | None:
    for cluster in reversed(bundle.post):
        if not isinstance(cluster, _AssistantCluster):
            continue
        call = _cluster_call(cluster, call_id)
        if call is not None:
            return cluster, call
    call = _anchor_owned_call(bundle.anchor, call_id)
    if call is not None:
        return bundle.anchor, call
    for cluster in reversed(bundle.pre):
        if not isinstance(cluster, _AssistantCluster):
            continue
        call = _cluster_call(cluster, call_id)
        if call is not None:
            return cluster, call
    return None


def _pending_call_owner(bundle: _MainBundle, call_id: str) -> tuple[_AssistantCluster | _MainAnchor, _MainCall] | None:
    owner = _call_owner(bundle, call_id)
    if owner is None:
        return None
    call_owner, call = owner
    if not call.has_function_call_item or call.has_output_message:
        return None
    return call_owner, call


def _attachment_owner(bundle: _MainBundle, *, prefer_clusters: bool) -> _AssistantCluster | _MainAnchor | None:
    if prefer_clusters:
        cluster = _nearest_assistant_cluster(_current_clusters(bundle))
        if cluster is not None:
            return cluster
    if bundle.anchor.assistant is not None:
        return bundle.anchor
    if not prefer_clusters:
        return _nearest_assistant_cluster(_current_clusters(bundle))
    return None


def _owner_calls(owner: _AssistantCluster | _MainAnchor) -> list[_MainCall]:
    if isinstance(owner, _MainAnchor):
        return owner.fabricated_calls
    return owner.calls


def _append_fabricated_call(owner: _AssistantCluster | _MainAnchor, item: RequestFunctionCallItem) -> None:
    _owner_calls(owner).append(
        _MainCall(
            id=item.call_id,
            name=item.name,
            arguments=item.arguments,
            has_function_call_item=True,
            has_output_message=False,
        )
    )


def _append_sealed_call(
    owner: _AssistantCluster | _MainAnchor,
    item: RequestFunctionCallItem,
    *,
    upstream_tool_call_id: str,
) -> None:
    owner.calls.append(
        _MainCall(
            id=upstream_tool_call_id,
            name=item.name,
            arguments=item.arguments,
            has_function_call_item=True,
            has_output_message=False,
        )
    )


def _mark_call_open(call: _MainCall) -> None:
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
    call.has_function_call_item = True


def _append_tool_output_to_owner(owner: _AssistantCluster | _MainAnchor, message: Message) -> None:
    owner.outputs.append(message)


def _mark_call_closed(call: _MainCall) -> None:
    call.has_output_message = True
    call.has_function_call_item = False


def _apply_hidden_suffix_outputs_to_anchor(anchor: _MainAnchor, suffix_outputs: list[Message]) -> None:
    if _pending_patch(anchor):
        anchor.hidden_suffix_outputs = list(suffix_outputs)
        return
    for message in suffix_outputs:
        if message.tool_call_id is None:
            raise _reasoning_replay_error(
                reason="reasoning_message_invalid",
                private_message="hidden main tool output must include tool_call_id",
            )
        call = _anchor_call(anchor, message.tool_call_id)
        if call is None:
            raise _reasoning_replay_error(
                reason="reasoning_message_invalid",
                private_message="hidden main tool output does not match an unresolved anchor tool call",
            )
        if call.has_output_message:
            raise _reasoning_replay_error(
                reason="reasoning_message_invalid",
                private_message="hidden main tool output duplicates an existing output",
            )
        _mark_call_closed(call)
        anchor.outputs.append(message)


def _resolve_patch_anchor(anchor: _MainAnchor, message: Message) -> None:
    if anchor.patch is None:
        raise RuntimeError("patch anchor resolution requires a pending patch")
    if not message.is_assistant():
        raise _tool_replay_error(
            reason="main_message_patch_target_missing",
            private_message="main message patch target must resolve to an assistant message",
        )
    tool_calls = [] if anchor.patch.tool_calls is None else list(anchor.patch.tool_calls)
    resolved = _copy_message(
        message,
        tool_calls=tool_calls,
        reasoning_content=anchor.patch.reasoning_content,
        reasoning_details=[] if anchor.patch.reasoning_details is None else list(anchor.patch.reasoning_details),
    )
    anchor.assistant = resolved
    anchor.calls = [_main_call_from_tool_call(tool_call) for tool_call in tool_calls]
    pending_outputs = list(anchor.hidden_suffix_outputs)
    anchor.patch = None
    anchor.hidden_suffix_outputs = []
    _apply_hidden_suffix_outputs_to_anchor(anchor, pending_outputs)


def _append_cluster_message(clusters: list[_MainCluster], message: Message) -> None:
    if message.is_assistant():
        clusters.append(_AssistantCluster(assistant=message))
        return
    clusters.append(message)


def _render_assistant_cluster(cluster: _AssistantCluster) -> list[Message]:
    assistant = _copy_message(
        cluster.assistant,
        tool_calls=[_tool_call_from_main_call(call) for call in cluster.calls],
    )
    return [assistant, *cluster.outputs]


def _render_anchor(anchor: _MainAnchor | None) -> list[Message]:
    if anchor is None or anchor.assistant is None:
        return []
    tool_calls = [
        *[_tool_call_from_main_call(call) for call in anchor.calls],
        *[_tool_call_from_main_call(call) for call in anchor.fabricated_calls],
    ]
    assistant = _copy_message(anchor.assistant, tool_calls=tool_calls)
    return [assistant, *anchor.outputs]


def _render_clusters(clusters: list[_MainCluster]) -> list[Message]:
    rendered: list[Message] = []
    for cluster in clusters:
        if isinstance(cluster, _AssistantCluster):
            rendered.extend(_render_assistant_cluster(cluster))
            continue
        rendered.append(cluster)
    return rendered


def _render_bundle(bundle: _MainBundle) -> list[Message]:
    return [*_render_clusters(bundle.pre), *_render_anchor(bundle.anchor), *_render_clusters(bundle.post)]


def _build_calls_from_cluster(cluster: _AssistantCluster) -> _SideCalls:
    side_calls = _SideCalls()
    for call in cluster.calls:
        side_calls.calls_by_id[call.id] = _tracked_call_from_main_call(call)
    return side_calls


def _build_calls_from_anchor(anchor: _MainAnchor | None) -> _SideCalls:
    side_calls = _SideCalls()
    if anchor is None or anchor.assistant is None:
        return side_calls
    for call in anchor.calls:
        side_calls.calls_by_id[call.id] = _tracked_call_from_main_call(call)
    for call in anchor.fabricated_calls:
        side_calls.calls_by_id[call.id] = _tracked_call_from_main_call(call)
    return side_calls


def _build_calls_from_clusters(clusters: list[_MainCluster]) -> _SideCalls:
    side_calls = _SideCalls()
    for cluster in clusters:
        if not isinstance(cluster, _AssistantCluster):
            continue
        side_calls = _merge_side_calls(side_calls, _build_calls_from_cluster(cluster))
    return side_calls


def _build_calls_from_bundle(bundle: _MainBundle | None) -> _SideCalls:
    if bundle is None:
        return _SideCalls()
    return _merge_side_calls(
        _build_calls_from_anchor(bundle.anchor),
        _merge_side_calls(_build_calls_from_clusters(bundle.pre), _build_calls_from_clusters(bundle.post)),
    )


@dataclass(slots=True)
class _MainReplay:
    committed: list[Message] = field(default_factory=list)
    bundle: _MainBundle | None = None

    def load_snapshot(self, messages: list[Message]) -> None:
        self.committed = list(messages)
        self.bundle = None

    def _finalize_bundle(self) -> None:
        if self.bundle is None:
            return
        if _pending_patch(self.bundle.anchor):
            raise _reasoning_replay_error(
                reason="main_message_patch_target_missing",
                private_message="main message patch target is missing",
            )
        self.committed.extend(_render_bundle(self.bundle))
        self.bundle = None

    def commit_before_reasoning(self) -> None:
        self._finalize_bundle()

    def apply_hidden_main_updates(self, sides: SidesUpdate) -> None:
        prefix, anchor, suffix = sides.split_main()
        self.committed.extend(prefix)
        if anchor is None:
            return
        if isinstance(anchor, MessagePatch):
            self.bundle = _MainBundle(anchor=_build_patch_anchor(anchor, suffix), after_anchor=False)
            return
        self.bundle = _MainBundle(anchor=_build_hidden_anchor(anchor, suffix), after_anchor=False)

    def _append_public_message(self, message: Message) -> None:
        if self.bundle is None:
            if message.is_assistant():
                self.bundle = _MainBundle(anchor=_build_public_anchor(message), after_anchor=True)
                return
            self.committed.append(message)
            return
        anchor = self.bundle.anchor
        if _pending_patch(anchor):
            if message.is_assistant() and message.content_hash() == anchor.patch.content_hash:
                _resolve_patch_anchor(anchor, message)
                self.bundle.after_anchor = True
                return
            _append_cluster_message(_current_clusters(self.bundle), message)
            return
        if message.is_assistant() and _can_handoff(self.bundle):
            self._finalize_bundle()
            self.bundle = _MainBundle(anchor=_build_public_anchor(message), after_anchor=True)
            return
        _append_cluster_message(_current_clusters(self.bundle), message)

    def _attach_fabricated_call(self, item: RequestFunctionCallItem) -> None:
        if self.bundle is None:
            raise _tool_replay_error(
                reason="fabricated_function_call_without_previous_assistant",
                private_message="fabricated function_call has no previous assistant message",
            )
        existing = _call_owner(self.bundle, item.call_id)
        if existing is not None:
            _mark_call_open(existing[1])
            return
        owner = _attachment_owner(self.bundle, prefer_clusters=True)
        if owner is None:
            raise _tool_replay_error(
                reason="fabricated_function_call_without_previous_assistant",
                private_message="fabricated function_call has no previous assistant message",
            )
        _append_fabricated_call(owner, item)
        if isinstance(owner, _MainAnchor):
            self.bundle.after_anchor = True

    def _attach_sealed_call(self, item: RequestFunctionCallItem, call_id: CallID) -> None:
        if self.bundle is None:
            raise _tool_replay_error(
                reason="sealed_function_call_without_attachment_owner",
                private_message="sealed function call has no attachment owner",
            )
        if _pending_patch(self.bundle.anchor) and call_id.upstream_tool_call_id in self.bundle.anchor.strict_declared_ids:
            raise _tool_replay_error(
                reason="sealed_function_call_before_patch_target",
                private_message="sealed function call cannot attach before its patch target resolves",
            )
        existing = _call_owner(self.bundle, call_id.upstream_tool_call_id)
        if existing is not None:
            _mark_call_open(existing[1])
            if isinstance(existing[0], _MainAnchor):
                self.bundle.after_anchor = True
            return
        owner = _attachment_owner(self.bundle, prefer_clusters=self.bundle.after_anchor)
        if owner is None:
            raise _tool_replay_error(
                reason="sealed_function_call_without_attachment_owner",
                private_message="sealed function call has no attachment owner",
            )
        _append_sealed_call(owner, item, upstream_tool_call_id=call_id.upstream_tool_call_id)
        if isinstance(owner, _MainAnchor):
            self.bundle.after_anchor = True

    def _attach_output(self, item: RequestFunctionCallOutputItem, *, call_id: str) -> None:
        if self.bundle is None:
            raise _tool_replay_error(
                reason="function_call_output_without_pending_function_call",
                private_message="function_call_output has no pending function_call",
            )
        owner = _pending_call_owner(self.bundle, call_id)
        if owner is None:
            raise _tool_replay_error(
                reason="function_call_output_without_pending_function_call",
                private_message="function_call_output has no pending function_call",
            )
        owner_state, call = owner
        _append_tool_output_to_owner(
            owner_state,
            Message(role="tool", tool_call_id=call_id, content=_function_output_text(item)),
        )
        _mark_call_closed(call)
        if isinstance(owner_state, _MainAnchor):
            self.bundle.after_anchor = True

    def add_standalone_main_item(self, item: _DecodedInput) -> None:
        if isinstance(item, _DecodedMessage):
            self._append_public_message(item.message)
            return
        if isinstance(item, _DecodedSealedFunctionCall):
            self._attach_sealed_call(item.item, item.call_id)
            return
        if isinstance(item, _DecodedSealedFunctionCallOutput):
            self._attach_output(item.item, call_id=item.call_id.upstream_tool_call_id)
            return
        if isinstance(item, _DecodedFabricatedFunctionCall):
            self._attach_fabricated_call(item.item)
            return
        if isinstance(item, _DecodedFabricatedFunctionCallOutput):
            self._attach_output(item.item, call_id=item.item.call_id)
            return
        raise TypeError(f"unsupported standalone main item: {type(item).__name__}")

    def current_messages(self) -> list[Message]:
        if self.bundle is None:
            return list(self.committed)
        return [*self.committed, *_render_bundle(self.bundle)]

    def current_calls(self) -> _SideCalls:
        committed_calls = _rebuild_calls(self.committed)
        return _merge_side_calls(committed_calls, _build_calls_from_bundle(self.bundle))

    def history_changed_by_last_event(self, item: _DecodedInput) -> bool:
        return not isinstance(item, _DecodedSealedFunctionCall)

    def assert_finished(self) -> None:
        if self.bundle is not None and _pending_patch(self.bundle.anchor):
            raise _reasoning_replay_error(
                reason="main_message_patch_target_missing",
                private_message="main message patch target is missing",
            )


@dataclass(slots=True)
class _Replay:
    machine: dict[str, JSONValue]
    sides: Sides
    last_side: Side | None
    calls_by_side: dict[Side, _SideCalls] = field(default_factory=_empty_calls_by_side)
    main: _MainReplay = field(default_factory=_MainReplay)
    current_compaction_id: str | None = None
    last_reasoning_id: str | None = None

    def _sync_main(self) -> None:
        self.sides.main = self.main.current_messages()
        self.calls_by_side[Side.MAIN] = self.main.current_calls()

    def _rebuild_non_main_calls(self, side: Side) -> None:
        self.calls_by_side[side] = _rebuild_calls(self.sides.messages(side))

    def _rebuild_all_calls(self) -> None:
        self.calls_by_side = _empty_calls_by_side()
        self._sync_main()
        for side in NON_MAIN_SIDES:
            self._rebuild_non_main_calls(side)

    def _assert_no_open_function_calls_before_reasoning(self) -> None:
        if _has_open_function_calls(self.calls_by_side):
            raise _tool_replay_error(
                reason="pending_tool_outputs_block_message",
                private_message="reasoning cannot appear before open function calls are closed",
            )

    def _assert_reasoning_chain_matches_state(self, payload: ReasoningPayload) -> None:
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
        self.main.load_snapshot(self.sides.main)
        self._rebuild_all_calls()
        self.current_compaction_id = payload.id
        self.last_reasoning_id = None
        self.last_side = None

    def _step_reasoning(self, payload: ReasoningPayload) -> None:
        self._assert_no_open_function_calls_before_reasoning()
        self._assert_reasoning_chain_matches_state(payload)
        self.main.commit_before_reasoning()
        self.machine = _apply_machine_patch(self.machine, payload.machine)
        for side in NON_MAIN_SIDES:
            if not payload.sides.others[side]:
                continue
            self.sides.others[side] = _apply_side_patch(self.sides.others[side], payload.sides.others[side], side=side)
            self._rebuild_non_main_calls(side)
        if payload.sides.main:
            self.main.apply_hidden_main_updates(payload.sides)
            self._sync_main()
        self.last_reasoning_id = payload.id
        self.last_side = None

    def _step_non_main_function_call(self, call_id: CallID) -> None:
        side_calls = self.calls_by_side[call_id.side]
        call = _tracked_call(side_calls, call_id.upstream_tool_call_id)
        if call is None:
            return
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
        call.has_function_call_item = True

    def _step_non_main_function_call_output(self, item: RequestFunctionCallOutputItem, call_id: CallID) -> None:
        side_calls = self.calls_by_side[call_id.side]
        call = _tracked_call(side_calls, call_id.upstream_tool_call_id)
        if call is None:
            return
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
        self._rebuild_non_main_calls(call_id.side)
        self.last_side = call_id.side

    def _step_standalone_main_item(self, item: _DecodedInput) -> None:
        self.main.add_standalone_main_item(item)
        self._sync_main()
        if self.main.history_changed_by_last_event(item):
            self.last_side = Side.MAIN

    def _validate_all_calls_closed(self) -> None:
        self.main.assert_finished()
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
            if item.call_id.side == Side.MAIN:
                self._step_standalone_main_item(item)
                return
            self._step_non_main_function_call(item.call_id)
            return
        if isinstance(item, _DecodedSealedFunctionCallOutput):
            if item.call_id.side == Side.MAIN:
                self._step_standalone_main_item(item)
                return
            self._step_non_main_function_call_output(item.item, item.call_id)
            return
        self._step_standalone_main_item(item)


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
