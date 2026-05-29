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
    SidesUpdate,
    ToolCall,
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


def _get_call(side_calls: _SideCalls, call_id: CallID) -> _TrackedCall:
    call = side_calls.calls_by_id.get(call_id.upstream_tool_call_id)
    if call is None or not _call_matches_call_id(call, call_id):
        raise _tool_replay_error(
            reason="sealed_function_call_content_hash_target_missing",
            private_message="sealed function call content_hash target is missing",
        )
    return call


@dataclass(slots=True)
class _MainCall:
    id: str
    name: str
    arguments: str
    sealed_index: int | None
    has_function_call_item: bool
    has_output_message: bool


def _tool_call_from_main_call(call: _MainCall) -> ToolCall:
    return ToolCall(id=call.id, name=call.name, arguments=call.arguments)


def _tracked_call_from_main_call(*, anchor_hash: str, tool_call_index: int, call: _MainCall) -> _TrackedCall:
    return _TrackedCall(
        anchor_content_hash=anchor_hash,
        tool_call_index=tool_call_index,
        has_function_call_item=call.has_function_call_item,
        has_output_message=call.has_output_message,
    )


def _main_call_from_tool_call(tool_call: ToolCall, *, sealed_index: int) -> _MainCall:
    return _MainCall(
        id=tool_call.id,
        name=tool_call.name,
        arguments=tool_call.arguments,
        sealed_index=sealed_index,
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
    stable_hash: str | None
    patch: MessagePatch | None
    hidden_suffix_outputs: list[Message]
    sealed_calls: list[_MainCall]
    fabricated_calls: list[_MainCall]
    outputs: list[Message]


def _pending_patch(anchor: _MainAnchor | None) -> bool:
    return anchor is not None and anchor.assistant is None and anchor.patch is not None


def _build_hidden_anchor(message: Message, suffix_outputs: list[Message]) -> _MainAnchor:
    anchor = _MainAnchor(
        assistant=message,
        stable_hash=content_hash(message),
        patch=None,
        hidden_suffix_outputs=[],
        sealed_calls=[_main_call_from_tool_call(tool_call, sealed_index=index) for index, tool_call in enumerate(message.tool_calls)],
        fabricated_calls=[],
        outputs=[],
    )
    _apply_hidden_suffix_outputs_to_anchor(anchor, suffix_outputs)
    return anchor


def _build_patch_anchor(patch: MessagePatch, suffix_outputs: list[Message]) -> _MainAnchor:
    return _MainAnchor(
        assistant=None,
        stable_hash=None,
        patch=patch,
        hidden_suffix_outputs=list(suffix_outputs),
        sealed_calls=[],
        fabricated_calls=[],
        outputs=[],
    )


def _anchor_sealed_call(anchor: _MainAnchor, tool_call_index: int) -> _MainCall | None:
    if tool_call_index < 0 or tool_call_index >= len(anchor.sealed_calls):
        return None
    return anchor.sealed_calls[tool_call_index]


def _anchor_has_matching_hash(anchor: _MainAnchor | None, call_id: CallID) -> bool:
    return anchor is not None and anchor.stable_hash is not None and anchor.stable_hash.startswith(call_id.content_hash_prefix.hex())


def _anchor_all_calls(anchor: _MainAnchor) -> list[_MainCall]:
    return [*anchor.sealed_calls, *anchor.fabricated_calls]


def _mark_hidden_anchor_output(anchor: _MainAnchor, message: Message) -> None:
    if message.tool_call_id is None:
        raise _reasoning_replay_error(
            reason="reasoning_message_invalid",
            private_message="hidden main tool output must include tool_call_id",
        )
    for call in _anchor_all_calls(anchor):
        if call.id != message.tool_call_id:
            continue
        if call.has_output_message:
            raise _reasoning_replay_error(
                reason="reasoning_message_invalid",
                private_message="hidden main tool output duplicates an existing output",
            )
        call.has_output_message = True
        call.has_function_call_item = False
        anchor.outputs.append(message)
        return
    raise _reasoning_replay_error(
        reason="reasoning_message_invalid",
        private_message="hidden main tool output does not match an unresolved anchor tool call",
    )


def _apply_hidden_suffix_outputs_to_anchor(anchor: _MainAnchor, suffix_outputs: list[Message]) -> None:
    if _pending_patch(anchor):
        anchor.hidden_suffix_outputs = list(suffix_outputs)
        return
    for message in suffix_outputs:
        _mark_hidden_anchor_output(anchor, message)


def _resolve_patch_anchor(anchor: _MainAnchor, message: Message) -> None:
    if anchor.patch is None:
        raise RuntimeError("patch anchor resolution requires a pending patch")
    if not message.is_assistant():
        raise _tool_replay_error(
            reason="main_message_patch_target_missing",
            private_message="main message patch target must resolve to an assistant message",
        )
    stable_hash = content_hash(message)
    tool_calls = [] if anchor.patch.tool_calls is None else list(anchor.patch.tool_calls)
    resolved = _copy_message(
        message,
        tool_calls=tool_calls,
        reasoning_content=anchor.patch.reasoning_content,
        reasoning_details=[] if anchor.patch.reasoning_details is None else list(anchor.patch.reasoning_details),
    )
    anchor.assistant = resolved
    anchor.stable_hash = stable_hash
    anchor.sealed_calls = [_main_call_from_tool_call(tool_call, sealed_index=index) for index, tool_call in enumerate(resolved.tool_calls)]
    pending_outputs = list(anchor.hidden_suffix_outputs)
    anchor.patch = None
    anchor.hidden_suffix_outputs = []
    _apply_hidden_suffix_outputs_to_anchor(anchor, pending_outputs)


def _append_cluster_message(clusters: list[_MainCluster], message: Message) -> None:
    if message.is_assistant():
        clusters.append(_AssistantCluster(assistant=message))
        return
    clusters.append(message)


def _current_cluster_list(*, pre: list[_MainCluster], post: list[_MainCluster], bundle_started: bool) -> list[_MainCluster]:
    return post if bundle_started else pre


def _nearest_assistant_cluster(clusters: list[_MainCluster]) -> _AssistantCluster | None:
    for cluster in reversed(clusters):
        if isinstance(cluster, _AssistantCluster):
            return cluster
    return None


def _cluster_pending_fabricated_call(cluster: _AssistantCluster, call_id: str) -> _MainCall | None:
    for call in cluster.calls:
        if call.sealed_index is None and call.id == call_id and call.has_function_call_item and not call.has_output_message:
            return call
    return None


def _anchor_pending_fabricated_call(anchor: _MainAnchor | None, call_id: str) -> _MainCall | None:
    if anchor is None:
        return None
    for call in anchor.fabricated_calls:
        if call.id == call_id and call.has_function_call_item and not call.has_output_message:
            return call
    return None


def _render_assistant_cluster(cluster: _AssistantCluster) -> list[Message]:
    tool_calls = [*cluster.assistant.tool_calls, *[_tool_call_from_main_call(call) for call in cluster.calls]]
    assistant = _copy_message(cluster.assistant, tool_calls=tool_calls)
    return [assistant, *cluster.outputs]


def _render_anchor(anchor: _MainAnchor | None) -> list[Message]:
    if anchor is None or anchor.assistant is None:
        return []
    tool_calls = [
        *[_tool_call_from_main_call(call) for call in anchor.sealed_calls],
        *[_tool_call_from_main_call(call) for call in anchor.fabricated_calls],
    ]
    assistant = _copy_message(
        anchor.assistant,
        tool_calls=tool_calls,
    )
    return [assistant, *anchor.outputs]


def _render_clusters(clusters: list[_MainCluster]) -> list[Message]:
    rendered: list[Message] = []
    for cluster in clusters:
        if isinstance(cluster, _AssistantCluster):
            rendered.extend(_render_assistant_cluster(cluster))
            continue
        rendered.append(cluster)
    return rendered


def _build_calls_from_cluster(cluster: _AssistantCluster) -> _SideCalls:
    side_calls = _SideCalls()
    anchor_hash = content_hash(cluster.assistant)
    base_index = len(cluster.assistant.tool_calls)
    for offset, call in enumerate(cluster.calls):
        side_calls.calls_by_id[call.id] = _tracked_call_from_main_call(
            anchor_hash=anchor_hash,
            tool_call_index=base_index + offset,
            call=call,
        )
    return side_calls


def _build_calls_from_anchor(anchor: _MainAnchor | None) -> _SideCalls:
    side_calls = _SideCalls()
    if anchor is None or anchor.assistant is None or anchor.stable_hash is None:
        return side_calls
    for call in anchor.sealed_calls:
        if call.sealed_index is None:
            raise RuntimeError("sealed anchor call must have a sealed index")
        side_calls.calls_by_id[call.id] = _tracked_call_from_main_call(
            anchor_hash=anchor.stable_hash,
            tool_call_index=call.sealed_index,
            call=call,
        )
    for offset, call in enumerate(anchor.fabricated_calls, start=len(anchor.sealed_calls)):
        side_calls.calls_by_id[call.id] = _tracked_call_from_main_call(
            anchor_hash=anchor.stable_hash,
            tool_call_index=offset,
            call=call,
        )
    return side_calls


@dataclass(slots=True)
class _ParsedMain:
    messages: list[Message]
    calls: _SideCalls
    has_pending_patch: bool


@dataclass(frozen=True, slots=True)
class _HiddenMainMessage:
    message: Message
    suffix_outputs: list[Message]


@dataclass(frozen=True, slots=True)
class _HiddenMainPatch:
    patch: MessagePatch
    suffix_outputs: list[Message]


@dataclass(frozen=True, slots=True)
class _PublicMainMessage:
    message: Message


@dataclass(frozen=True, slots=True)
class _SealedMainFunctionCall:
    item: RequestFunctionCallItem
    call_id: CallID


@dataclass(frozen=True, slots=True)
class _SealedMainFunctionCallOutput:
    item: RequestFunctionCallOutputItem
    call_id: CallID


@dataclass(frozen=True, slots=True)
class _FabricatedMainFunctionCall:
    item: RequestFunctionCallItem


@dataclass(frozen=True, slots=True)
class _FabricatedMainFunctionCallOutput:
    item: RequestFunctionCallOutputItem


type _MainEvent = (
    _HiddenMainMessage
    | _HiddenMainPatch
    | _PublicMainMessage
    | _SealedMainFunctionCall
    | _SealedMainFunctionCallOutput
    | _FabricatedMainFunctionCall
    | _FabricatedMainFunctionCallOutput
)


def _parse_main_events(events: list[_MainEvent]) -> _ParsedMain:
    pre: list[_MainCluster] = []
    post: list[_MainCluster] = []
    anchor: _MainAnchor | None = None
    bundle_started = False

    def target_clusters() -> list[_MainCluster]:
        return _current_cluster_list(pre=pre, post=post, bundle_started=bundle_started)

    def ensure_anchor_for_main_call(call_id: CallID) -> _MainAnchor:
        if anchor is None or anchor.assistant is None or anchor.patch is not None:
            raise _tool_replay_error(
                reason="sealed_function_call_content_hash_target_missing",
                private_message="sealed function call content_hash target is missing",
            )
        if not _anchor_has_matching_hash(anchor, call_id):
            raise _tool_replay_error(
                reason="sealed_function_call_content_hash_target_missing",
                private_message="sealed function call content_hash target is missing",
            )
        return anchor

    def append_anchor_fabricated_call(item: RequestFunctionCallItem) -> None:
        if anchor is None or anchor.assistant is None or anchor.patch is not None:
            raise _tool_replay_error(
                reason="fabricated_function_call_without_previous_assistant",
                private_message="fabricated function_call has no previous assistant message",
            )
        anchor.fabricated_calls.append(
            _MainCall(
                id=item.call_id,
                name=item.name,
                arguments=item.arguments,
                sealed_index=None,
                has_function_call_item=True,
                has_output_message=False,
            )
        )

    def append_cluster_fabricated_call(cluster: _AssistantCluster, item: RequestFunctionCallItem) -> None:
        cluster.calls.append(
            _MainCall(
                id=item.call_id,
                name=item.name,
                arguments=item.arguments,
                sealed_index=None,
                has_function_call_item=True,
                has_output_message=False,
            )
        )

    def find_pending_fabricated_call(call_id: str) -> tuple[_AssistantCluster | _MainAnchor, _MainCall] | None:
        cluster = _nearest_assistant_cluster(target_clusters())
        while cluster is not None:
            call = _cluster_pending_fabricated_call(cluster, call_id)
            if call is not None:
                return cluster, call
            clusters = target_clusters()
            index = clusters.index(cluster)
            cluster = _nearest_assistant_cluster(clusters[:index])
        anchor_call = _anchor_pending_fabricated_call(anchor, call_id)
        if anchor_call is None or anchor is None:
            return None
        return anchor, anchor_call

    for event in events:
        if isinstance(event, _HiddenMainMessage):
            if anchor is not None:
                raise _reasoning_replay_error(
                    reason="reasoning_message_invalid",
                    private_message="hidden main delta may define at most one anchor",
                )
            anchor = _build_hidden_anchor(event.message, event.suffix_outputs)
            continue

        if isinstance(event, _HiddenMainPatch):
            if anchor is not None:
                raise _reasoning_replay_error(
                    reason="reasoning_message_invalid",
                    private_message="hidden main delta may define at most one anchor",
                )
            anchor = _build_patch_anchor(event.patch, event.suffix_outputs)
            continue

        if isinstance(event, _PublicMainMessage):
            if _pending_patch(anchor):
                if event.message.is_assistant() and content_hash(event.message) == anchor.patch.content_hash:
                    _resolve_patch_anchor(anchor, event.message)
                    bundle_started = True
                    continue
                _append_cluster_message(target_clusters(), event.message)
                continue

            if anchor is None:
                if event.message.is_assistant():
                    anchor = _build_hidden_anchor(event.message, ())
                    bundle_started = True
                    continue
                _append_cluster_message(target_clusters(), event.message)
                continue

            _append_cluster_message(target_clusters(), event.message)
            continue

        if isinstance(event, _SealedMainFunctionCall):
            current_anchor = ensure_anchor_for_main_call(event.call_id)
            sealed_call = _anchor_sealed_call(current_anchor, event.call_id.tool_call_index)
            if sealed_call is not None:
                if sealed_call.id != event.call_id.upstream_tool_call_id:
                    raise _tool_replay_error(
                        reason="sealed_function_call_content_hash_target_missing",
                        private_message="sealed function call content_hash target is missing",
                    )
                if sealed_call.has_output_message:
                    raise _tool_replay_error(
                        reason="function_call_already_satisfied",
                        private_message="function_call already has a tool output in history",
                    )
                if sealed_call.has_function_call_item:
                    raise _tool_replay_error(
                        reason="duplicate_pending_function_call",
                        private_message="duplicate pending function_call",
                    )
                sealed_call.has_function_call_item = True
                bundle_started = True
                continue
            if event.call_id.tool_call_index > len(current_anchor.sealed_calls):
                raise _tool_replay_error(
                    reason="sealed_function_call_index_not_contiguous",
                    private_message="sealed function_call index is not contiguous",
                )
            current_anchor.sealed_calls.append(
                _MainCall(
                    id=event.call_id.upstream_tool_call_id,
                    name=event.item.name,
                    arguments=event.item.arguments,
                    sealed_index=event.call_id.tool_call_index,
                    has_function_call_item=True,
                    has_output_message=False,
                )
            )
            bundle_started = True
            continue

        if isinstance(event, _SealedMainFunctionCallOutput):
            current_anchor = ensure_anchor_for_main_call(event.call_id)
            sealed_call = _anchor_sealed_call(current_anchor, event.call_id.tool_call_index)
            if sealed_call is None:
                raise _tool_replay_error(
                    reason="function_call_output_without_pending_function_call",
                    private_message="function_call_output has no pending function_call",
                )
            if sealed_call.id != event.call_id.upstream_tool_call_id:
                raise _tool_replay_error(
                    reason="sealed_function_call_output_upstream_id_mismatch",
                    private_message="sealed function_call_output upstream id mismatch",
                )
            if not sealed_call.has_function_call_item or sealed_call.has_output_message:
                raise _tool_replay_error(
                    reason="function_call_output_without_pending_function_call",
                    private_message="function_call_output has no pending function_call",
                )
            current_anchor.outputs.append(
                Message(
                    role="tool",
                    tool_call_id=event.call_id.upstream_tool_call_id,
                    content=_function_output_text(event.item),
                )
            )
            sealed_call.has_output_message = True
            sealed_call.has_function_call_item = False
            bundle_started = True
            continue

        if isinstance(event, _FabricatedMainFunctionCall):
            cluster = _nearest_assistant_cluster(target_clusters())
            if cluster is not None:
                append_cluster_fabricated_call(cluster, event.item)
                continue
            append_anchor_fabricated_call(event.item)
            bundle_started = True
            continue

        if isinstance(event, _FabricatedMainFunctionCallOutput):
            owner = find_pending_fabricated_call(event.item.call_id)
            if owner is None:
                raise _tool_replay_error(
                    reason="function_call_output_without_pending_function_call",
                    private_message="function_call_output has no pending function_call",
                )
            owner_state, call = owner
            output = Message(role="tool", tool_call_id=event.item.call_id, content=_function_output_text(event.item))
            if isinstance(owner_state, _AssistantCluster):
                owner_state.outputs.append(output)
            else:
                owner_state.outputs.append(output)
                bundle_started = True
            call.has_output_message = True
            call.has_function_call_item = False
            continue

        raise TypeError(f"unsupported main event: {type(event).__name__}")

    messages = [*_render_clusters(pre), *_render_anchor(anchor), *_render_clusters(post)]
    calls = _merge_side_calls(
        _SideCalls(),
        _merge_side_calls(
            _build_calls_from_anchor(anchor),
            _merge_side_calls(
                _build_calls_from_clusters(pre),
                _build_calls_from_clusters(post),
            ),
        ),
    )
    return _ParsedMain(
        messages=messages,
        calls=calls,
        has_pending_patch=_pending_patch(anchor),
    )


def _build_calls_from_clusters(clusters: list[_MainCluster]) -> _SideCalls:
    side_calls = _SideCalls()
    for cluster in clusters:
        if not isinstance(cluster, _AssistantCluster):
            continue
        side_calls = _merge_side_calls(side_calls, _build_calls_from_cluster(cluster))
    return side_calls


@dataclass(slots=True)
class _MainReplay:
    committed: list[Message] = field(default_factory=list)
    events: list[_MainEvent] = field(default_factory=list)

    def _parse(self) -> _ParsedMain:
        return _parse_main_events(self.events)

    def load_snapshot(self, messages: list[Message]) -> None:
        self.committed = list(messages)
        self.events = []

    def commit_before_reasoning(self) -> None:
        if not self.events:
            return
        parsed = self._parse()
        if parsed.has_pending_patch:
            raise _reasoning_replay_error(
                reason="main_message_patch_target_missing",
                private_message="main message patch target is missing",
            )
        self.committed.extend(parsed.messages)
        self.events = []

    def apply_hidden_main_updates(self, sides: SidesUpdate) -> None:
        prefix, anchor, suffix = sides.split_main()
        self.committed.extend(prefix)
        if anchor is None:
            return
        if isinstance(anchor, MessagePatch):
            self.events.append(_HiddenMainPatch(patch=anchor, suffix_outputs=suffix))
            return
        self.events.append(_HiddenMainMessage(message=anchor, suffix_outputs=suffix))

    def add_standalone_main_item(self, item: _DecodedInput) -> None:
        if isinstance(item, _DecodedMessage):
            self.events.append(_PublicMainMessage(message=item.message))
            return
        if isinstance(item, _DecodedSealedFunctionCall):
            self.events.append(_SealedMainFunctionCall(item=item.item, call_id=item.call_id))
            return
        if isinstance(item, _DecodedSealedFunctionCallOutput):
            self.events.append(_SealedMainFunctionCallOutput(item=item.item, call_id=item.call_id))
            return
        if isinstance(item, _DecodedFabricatedFunctionCall):
            self.events.append(_FabricatedMainFunctionCall(item=item.item))
            return
        if isinstance(item, _DecodedFabricatedFunctionCallOutput):
            self.events.append(_FabricatedMainFunctionCallOutput(item=item.item))
            return
        raise TypeError(f"unsupported standalone main item: {type(item).__name__}")

    def current_messages(self) -> list[Message]:
        parsed = self._parse()
        return [*self.committed, *parsed.messages]

    def current_calls(self) -> _SideCalls:
        parsed = self._parse()
        return _merge_side_calls(_rebuild_calls(self.committed), parsed.calls)

    def history_changed_by_last_event(self, item: _DecodedInput) -> bool:
        return not isinstance(item, _DecodedSealedFunctionCall)

    def assert_finished(self) -> None:
        parsed = self._parse()
        if parsed.has_pending_patch:
            raise _reasoning_replay_error(
                reason="main_message_patch_target_missing",
                private_message="main message patch target is missing",
            )


@dataclass(slots=True)
class _Replay:
    machine: dict[str, object]
    sides: Sides
    last_side: Side | None
    calls_by_side: dict[Side, _SideCalls] = field(default_factory=_empty_calls_by_side)
    main: _MainReplay = field(default_factory=_MainReplay)

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

    def _step_compaction(self, payload: CompactionPayload) -> None:
        self.machine = dict(payload.machine)
        self.sides = Sides.from_primitive(payload.sides.to_primitive())
        self.main.load_snapshot(self.sides.main)
        self._rebuild_all_calls()
        self.last_side = None

    def _step_reasoning(self, payload: ReasoningPayload) -> None:
        self._assert_no_open_function_calls_before_reasoning()
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
        return Ingested(machine=self.machine, sides=self.sides, last_side=self.last_side)

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
