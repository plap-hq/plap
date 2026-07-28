from __future__ import annotations

from dataclasses import dataclass
from itertools import count

import pytest
from pydantic import ValidationError

from plap.errors import PlapError
from plap.keyring import SealingKeyring
from plap.llms.completions.chat import ChatContentFile, ChatContentImage, ChatContentText, ChatFile, ChatImageURL
from plap.responses.contracts import (
    InputFileContent,
    InputImageContent,
    InputTextContent,
    OutputRefusalContent,
    OutputTextContent,
    RequestCompactionItem,
    RequestFunctionCallItem,
    RequestFunctionCallOutputItem,
    RequestMessageItem,
    RequestReasoningItem,
    ResponseCreateRequest,
    SummaryTextContent,
)
from plap.responses.ingest.content import message as decode_message
from plap.responses.ingest.content import tool_output as decode_tool_output
from plap.responses.ingest.ingest import (
    _decode_queue as _decode_queue_impl,
)
from plap.responses.ingest.ingest import (
    _DecodedCompaction,
    _DecodedFabricatedFunctionCall,
    _DecodedFabricatedFunctionCallOutput,
    _DecodedMessage,
    _DecodedReasoning,
    _DecodedSealedFunctionCall,
    _DecodedSealedFunctionCallOutput,
    _last_compaction_index,
    _normalize_input_items,
    _slice_to_last_compaction,
)
from plap.responses.ingest.ingest import (
    ingest_response_request as _ingest_response_request_impl,
)
from plap.responses.ingest.models import (
    CallID,
    CompactionPayload,
    Message,
    MessagePatch,
    ReasoningCheckpoint,
    ReasoningPatch,
    ReasoningPayload,
    Side,
    Sides,
    ToolCall,
    split_main_updates,
)
from plap.responses.ingest.sealing import (
    _pack_call_id as _pack_call_id_impl,
)
from plap.responses.ingest.sealing import (
    _unpack_call_id as _unpack_call_id_impl,
)
from plap.responses.ingest.sealing import (
    open_call_id as _open_call_id_impl,
)
from plap.responses.ingest.sealing import (
    seal_call_id as _seal_call_id_impl,
)
from plap.responses.ingest.sealing import (
    seal_compaction_payload,
    seal_reasoning_payload,
)


def _compaction(label: str) -> RequestCompactionItem:
    return RequestCompactionItem(encrypted_content=label, type="compaction")


def _message(label: str) -> RequestMessageItem:
    return RequestMessageItem(content=label, role="user", type="message")


def _keyring() -> SealingKeyring:
    return SealingKeyring(roots=(b"i" * 32,))


def _side_codes() -> dict[str, int]:
    return {
        "main": 0,
        "defender": 1024,
        "reviewer": 1025,
        "arbitrator": 1026,
    }


def _decode_queue(items, *, keyring: SealingKeyring):
    return _decode_queue_impl(items, keyring=keyring, side_codes=_side_codes())


async def ingest_response_request(request: ResponseCreateRequest, *, keyring: SealingKeyring):
    return await _ingest_response_request_impl(request, keyring=keyring, side_codes=_side_codes())


def _pack_call_id(value: CallID) -> bytes:
    return _pack_call_id_impl(value, side_codes=_side_codes())


def _unpack_call_id(value: bytes) -> CallID:
    return _unpack_call_id_impl(value, side_codes=_side_codes())


def seal_call_id(value: CallID, *, keyring: SealingKeyring) -> str:
    return _seal_call_id_impl(value, keyring=keyring, side_codes=_side_codes())


def open_call_id(value: str, *, keyring: SealingKeyring) -> CallID:
    return _open_call_id_impl(value, keyring=keyring, side_codes=_side_codes())


_REASONING_PAYLOAD_COUNTER = count()
_COMPACTION_PAYLOAD_COUNTER = count()


@dataclass(frozen=True, slots=True)
class _Update:
    active: set[Side] | None
    main: list[Message | MessagePatch]
    patches: dict[Side, list[dict[str, object]]]


def _next_reasoning_payload_id() -> str:
    return f"rs_payload_{next(_REASONING_PAYLOAD_COUNTER)}"


def _next_compaction_payload_id() -> str:
    return f"cmp_payload_{next(_COMPACTION_PAYLOAD_COUNTER)}"


def _compaction_payload(
    *,
    durable: dict[str, object],
    sides: Sides,
    payload_id: str | None = None,
) -> CompactionPayload:
    return CompactionPayload(id=payload_id or _next_compaction_payload_id(), durable=durable, sides=sides)


def _reasoning_payload(
    *,
    durable: list[dict[str, object]],
    sides: _Update,
    payload_id: str | None = None,
    previous_reasoning_id: str | None = None,
    previous_compaction_id: str | None = None,
    checkpoint: bool = False,
) -> ReasoningPayload:
    state = (
        ReasoningCheckpoint(
            durable={},
            active={"main"} if sides.active is None else sides.active,
            sides={},
        )
        if checkpoint
        else ReasoningPatch(durable=durable, active=sides.active, sides=sides.patches)
    )
    return ReasoningPayload(
        id=payload_id or _next_reasoning_payload_id(),
        previous_reasoning_id=previous_reasoning_id,
        previous_compaction_id=previous_compaction_id,
        state=state,
        main=sides.main,
    )


def _checkpoint_payload(
    *,
    durable: dict[str, object],
    sides: _Update,
    payload_id: str | None = None,
    snapshots: dict[Side, list[Message]] | None = None,
    previous_compaction_id: str | None = None,
) -> ReasoningPayload:
    return ReasoningPayload(
        id=payload_id or _next_reasoning_payload_id(),
        previous_reasoning_id=None,
        previous_compaction_id=previous_compaction_id,
        state=ReasoningCheckpoint(
            durable=durable,
            active={"main"} if sides.active is None else sides.active,
            sides={} if snapshots is None else snapshots,
        ),
        main=sides.main,
    )


def _sides_update(
    *,
    active: set[Side] | None = None,
    main: list[Message | MessagePatch] | None = None,
    patches: dict[Side, list[dict[str, object]]] | None = None,
) -> _Update:
    update = _Update(active=active, main=[] if main is None else list(main), patches={} if patches is None else patches)
    split_main_updates(update.main)
    ReasoningPatch(durable=[], active=active, sides=update.patches)
    return update


def _sealed_compaction(payload: CompactionPayload) -> RequestCompactionItem:
    return RequestCompactionItem(
        encrypted_content=seal_compaction_payload(payload, keyring=_keyring()),
        id=payload.id,
        type="compaction",
    )


def _sealed_reasoning(payload: ReasoningPayload) -> RequestReasoningItem:
    return RequestReasoningItem(
        encrypted_content=seal_reasoning_payload(payload, keyring=_keyring()),
        id=payload.id,
        summary=[SummaryTextContent(text="sealed", type="summary_text")],
        type="reasoning",
    )


def _sealed_call_id(side: str, upstream_tool_call_id: str) -> str:
    return seal_call_id(
        CallID(
            side=side,
            upstream_tool_call_id=upstream_tool_call_id,
        ),
        keyring=_keyring(),
    )


def _sealed_call_id_for_message(side: str, upstream_tool_call_id: str, message: Message, *, tool_call_index: int = 0) -> str:
    _ = message
    _ = tool_call_index
    return seal_call_id(
        CallID(
            side=side,
            upstream_tool_call_id=upstream_tool_call_id,
        ),
        keyring=_keyring(),
    )


def _parallel_active_calls() -> tuple[ReasoningPayload, str, str]:
    main_assistant = Message(
        role="assistant",
        tool_calls=[ToolCall(id="up_main_0", name="read_file", arguments='{"path":"main"}')],
    )
    reviewer_assistant = Message(
        role="assistant",
        tool_calls=[ToolCall(id="up_reviewer_0", name="read_file", arguments='{"path":"reviewer"}')],
    )
    payload = _reasoning_payload(
        durable=[],
        sides=_sides_update(
            active={"main", "reviewer"},
            main=[main_assistant],
            patches={"reviewer": [{"op": "add", "path": "/0", "value": reviewer_assistant.to_primitive()}]},
        ),
    )
    return payload, _sealed_call_id("main", "up_main_0"), _sealed_call_id("reviewer", "up_reviewer_0")


def test_normalize_input_items_wraps_string_as_user_message() -> None:
    result = _normalize_input_items(ResponseCreateRequest(model="plap/test", input="hello"))

    assert result == [RequestMessageItem(content="hello", role="user", type="message")]


def test_last_compaction_index_returns_none_without_compaction() -> None:
    items = [_message("a"), _message("b")]

    assert _last_compaction_index(items) is None


def test_slice_to_last_compaction_keeps_full_queue_without_compaction() -> None:
    items = [_message("a"), _message("b")]

    assert _slice_to_last_compaction(items) == items


def test_slice_to_last_compaction_drops_items_before_single_compaction() -> None:
    items = [_message("a"), _compaction("cmp1"), _message("b")]

    assert _slice_to_last_compaction(items) == [_compaction("cmp1"), _message("b")]


def test_slice_to_last_compaction_uses_last_compaction_when_multiple_exist() -> None:
    items = [_message("a"), _compaction("cmp1"), _message("b"), _compaction("cmp2"), _message("c")]

    assert _slice_to_last_compaction(items) == [_compaction("cmp2"), _message("c")]


def test_slice_to_last_compaction_leaves_leading_compaction_in_place() -> None:
    items = [_compaction("cmp1"), _message("a"), _message("b")]

    assert _slice_to_last_compaction(items) == items


def test_slice_to_last_compaction_preserves_compaction_only_queue() -> None:
    items = [_compaction("cmp1")]

    assert _slice_to_last_compaction(items) == items


def test_decode_queue_opens_compaction_payload() -> None:
    payload = _compaction_payload(durable={"active": []}, sides=Sides())

    decoded = _decode_queue([_sealed_compaction(payload)], keyring=_keyring())

    assert decoded == [_DecodedCompaction(payload=payload)]


def test_decode_queue_opens_reasoning_payload() -> None:
    payload = _reasoning_payload(
        durable=[{"op": "add", "path": "/active", "value": []}],
        sides=_sides_update(),
    )

    decoded = _decode_queue([_sealed_reasoning(payload)], keyring=_keyring())

    assert decoded == [_DecodedReasoning(payload=payload)]


def test_reasoning_payload_allows_empty_delta() -> None:
    payload = _reasoning_payload(durable=[], sides=_sides_update())

    assert payload.state == ReasoningPatch(durable=[])
    assert payload.main == []


def test_decode_queue_rejects_reasoning_item_id_mismatch() -> None:
    payload = _reasoning_payload(durable=[{"op": "add", "path": "/active", "value": []}], sides=_sides_update())
    item = RequestReasoningItem(
        encrypted_content=seal_reasoning_payload(payload, keyring=_keyring()),
        id="rs_wrong",
        summary=[SummaryTextContent(text="sealed", type="summary_text")],
        type="reasoning",
    )

    with pytest.raises(PlapError) as exc_info:
        _decode_queue([item], keyring=_keyring())

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "reasoning_item_id_mismatch"


def test_decode_queue_rejects_compaction_item_id_mismatch() -> None:
    payload = _compaction_payload(durable={"active": []}, sides=Sides())
    item = RequestCompactionItem(
        encrypted_content=seal_compaction_payload(payload, keyring=_keyring()),
        id="cmp_wrong",
        type="compaction",
    )

    with pytest.raises(PlapError) as exc_info:
        _decode_queue([item], keyring=_keyring())

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "compaction_item_id_mismatch"


def test_decode_queue_decodes_message_item_to_internal_message() -> None:
    decoded = _decode_queue(
        [
            RequestMessageItem(
                content=[
                    OutputTextContent(text="hello", type="output_text"),
                    OutputTextContent(text="world", type="output_text"),
                ],
                role="assistant",
                type="message",
            )
        ],
        keyring=_keyring(),
    )

    assert decoded == [
        _DecodedMessage(
            message=Message(
                role="assistant",
                content=[
                    ChatContentText(text="hello"),
                    ChatContentText(text="world"),
                ],
            )
        )
    ]


def test_request_message_item_rejects_refusal_content_for_user_role() -> None:
    with pytest.raises(ValidationError, match="does not support content variants"):
        RequestMessageItem(
            content=[OutputRefusalContent(refusal="no", type="refusal")],
            role="user",
            type="message",
        )


def test_decode_queue_normalizes_assistant_refusal_to_top_level_runtime_field() -> None:
    decoded = _decode_queue(
        [
            RequestMessageItem(
                content=[
                    OutputTextContent(text="hello", type="output_text"),
                    OutputRefusalContent(refusal="nope", type="refusal"),
                ],
                role="assistant",
                type="message",
            )
        ],
        keyring=_keyring(),
    )

    assert decoded == [
        _DecodedMessage(
            message=Message(
                role="assistant",
                content="hello",
                refusal="nope",
            )
        )
    ]


def test_decode_queue_normalizes_refusal_only_assistant_content_to_none() -> None:
    decoded = _decode_queue(
        [
            RequestMessageItem(
                content=[OutputRefusalContent(refusal="nope", type="refusal")],
                role="assistant",
                type="message",
            )
        ],
        keyring=_keyring(),
    )

    assert decoded == [_DecodedMessage(message=Message(role="assistant", content=None, refusal="nope"))]


def test_decode_queue_decodes_structured_message_parts_to_chat_superset() -> None:
    decoded = _decode_queue(
        [
            RequestMessageItem(
                content=[
                    InputTextContent(text="look", type="input_text"),
                    InputImageContent(image_url="https://example.com/image.png", detail="original", type="input_image"),
                    InputFileContent(file_url="https://example.com/report.pdf", filename="report.pdf", detail="high", type="input_file"),
                ],
                role="user",
                type="message",
            )
        ],
        keyring=_keyring(),
    )

    assert decoded == [
        _DecodedMessage(
            message=Message(
                role="user",
                content=[
                    ChatContentText(text="look"),
                    ChatContentImage(image_url=ChatImageURL(url="https://example.com/image.png", detail="original")),
                    ChatContentFile(file=ChatFile(file_url="https://example.com/report.pdf", filename="report.pdf", detail="high")),
                ],
            )
        )
    ]


def test_decode_queue_decodes_structured_function_call_output_to_tool_message_content() -> None:
    item = RequestFunctionCallOutputItem(
        call_id="call_0",
        output=[
            InputTextContent(text="tool text", type="input_text"),
            InputImageContent(image_url="https://example.com/tool.png", detail="high", type="input_image"),
        ],
        type="function_call_output",
    )

    assert decode_tool_output(item) == [
        ChatContentText(text="tool text"),
        ChatContentImage(image_url=ChatImageURL(url="https://example.com/tool.png", detail="high")),
    ]


def test_decode_queue_classifies_sealed_function_call() -> None:
    item = RequestFunctionCallItem(
        arguments='{"path":"README.md"}',
        call_id=_sealed_call_id("reviewer", "up_reviewer_0"),
        name="read_file",
        type="function_call",
    )

    decoded = _decode_queue([item], keyring=_keyring())

    assert decoded == [
        _DecodedSealedFunctionCall(
            item=item,
            call_id=CallID(
                side="reviewer",
                upstream_tool_call_id="up_reviewer_0",
            ),
        )
    ]


def test_decode_queue_classifies_unopenable_function_call_as_fabricated() -> None:
    item = RequestFunctionCallItem(
        arguments='{"path":"README.md"}',
        call_id="not_openable",
        name="read_file",
        type="function_call",
    )

    decoded = _decode_queue([item], keyring=_keyring())

    assert decoded == [_DecodedFabricatedFunctionCall(item=item)]


def test_decode_queue_classifies_sealed_function_call_output() -> None:
    item = RequestFunctionCallOutputItem(
        call_id=_sealed_call_id("defender", "up_defender_0"),
        output="done",
        type="function_call_output",
    )

    decoded = _decode_queue([item], keyring=_keyring())

    assert decoded == [
        _DecodedSealedFunctionCallOutput(
            item=item,
            call_id=CallID(
                side="defender",
                upstream_tool_call_id="up_defender_0",
            ),
        )
    ]


def test_decode_queue_classifies_unopenable_function_call_output_as_fabricated() -> None:
    item = RequestFunctionCallOutputItem(
        call_id="not_openable",
        output="done",
        type="function_call_output",
    )

    decoded = _decode_queue([item], keyring=_keyring())

    assert decoded == [_DecodedFabricatedFunctionCallOutput(item=item)]


def test_call_id_roundtrips_zero_based_main_side() -> None:
    value = CallID(
        side="main",
        upstream_tool_call_id="up_main_0",
    )

    token = seal_call_id(value, keyring=_keyring())

    assert open_call_id(token, keyring=_keyring()) == value


def test_pack_call_id_uses_base64url6_codec_when_possible() -> None:
    packed = _pack_call_id(CallID(side="main", upstream_tool_call_id="up_main_0"))

    assert packed[0] == 0x22


def test_pack_call_id_uses_ascii7_codec_for_ascii_non_base64url_ids() -> None:
    packed = _pack_call_id(CallID(side="main", upstream_tool_call_id="call:1"))

    assert packed[0] == 0x21


def test_pack_call_id_uses_utf8_codec_for_non_ascii_ids() -> None:
    packed = _pack_call_id(CallID(side="main", upstream_tool_call_id="café"))

    assert packed[0] == 0x20


def test_unpack_call_id_rejects_reserved_codec() -> None:
    with pytest.raises(PlapError) as exc_info:
        _unpack_call_id(bytes([0x23, 0x00, 0x00, ord("a")]))

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "unsupported_function_call_id_codec"


def test_unpack_call_id_rejects_invalid_packed_meta() -> None:
    with pytest.raises(PlapError) as exc_info:
        _unpack_call_id(bytes([0x21, 0x00, 0x00, 0x80, 0x00]))

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "function_call_id_packed_meta_invalid"


def test_unpack_call_id_rejects_invalid_packed_payload() -> None:
    with pytest.raises(PlapError) as exc_info:
        _unpack_call_id(bytes([0x21, 0x00, 0x00, 0x02, 0x00]))

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "function_call_id_packed_payload_invalid"


def test_message_patch_round_trips_complete_assistant() -> None:
    assistant = Message(
        role="assistant",
        content="answer",
        refusal="partial refusal",
        reasoning_content="hidden",
        tool_calls=[ToolCall(id="call_1", name="read_file", arguments="{}")],
    )
    patch = MessagePatch(message=assistant)

    assert MessagePatch.from_primitive(patch.to_primitive()) == patch


def test_message_patch_rejects_non_assistant_message() -> None:
    with pytest.raises(ValueError, match="must wrap an assistant"):
        MessagePatch(message=Message(role="user", content="not an assistant"))


def test_sides_assignment_is_replaceable_and_copies_assigned_lists() -> None:
    sides = Sides()
    first = Message(role="assistant", content="first")
    second = Message(role="assistant", content="second")

    replacement = [first]
    sides["main"] = replacement
    replacement.append(second)

    assert sides["main"] == [first]
    sides["main"].append(second)
    assert sides["main"] == [first, second]


def test_sides_update_main_accepts_terminal_patch_after_hidden_tool_messages() -> None:
    assistant = Message(
        role="assistant",
        content="anchor",
        tool_calls=[ToolCall(id="call_1", name="read_file", arguments="{}")],
    )
    update = _sides_update(
        main=[
            assistant,
            Message(role="tool", tool_call_id="call_1", content="hidden output"),
            MessagePatch(message=assistant),
        ]
    )

    assert len(update.main) == 3


def test_sides_update_main_rejects_second_patch() -> None:
    first = Message(role="assistant", content="first")
    second = Message(role="assistant", content="second")
    with pytest.raises(ValueError, match="at most one message patch"):
        _sides_update(
            main=[
                MessagePatch(message=first),
                MessagePatch(message=second),
            ]
        )


def test_sides_update_main_rejects_non_tool_message_after_patch() -> None:
    assistant = Message(role="assistant", content="anchor")
    with pytest.raises(ValueError, match="message patch must be the final main update"):
        _sides_update(
            main=[
                MessagePatch(message=assistant),
                Message(role="assistant", content="later assistant"),
            ]
        )


def test_sides_update_main_rejects_tool_message_after_patch() -> None:
    assistant = Message(role="assistant", content="anchor")
    with pytest.raises(ValueError, match="message patch must be the final main update"):
        _sides_update(
            main=[
                MessagePatch(message=assistant),
                Message(role="tool", content="missing id"),
            ]
        )


def test_sides_update_main_accepts_closed_prefix_before_patch_anchor() -> None:
    anchor = Message(
        role="assistant",
        content="anchor",
        tool_calls=[ToolCall(id="call_1", name="read_file", arguments="{}")],
    )
    update = _sides_update(
        main=[
            Message(
                role="assistant",
                content="prefix tool turn",
                tool_calls=[ToolCall(id="pref_0", name="read_file", arguments="{}")],
            ),
            Message(role="tool", tool_call_id="pref_0", content="prefix output"),
            anchor,
            Message(role="tool", tool_call_id="call_1", content="anchor hidden output"),
            MessagePatch(message=anchor),
        ]
    )

    assert len(update.main) == 5


def test_sides_update_main_accepts_closed_prefix_before_assistant_anchor() -> None:
    update = _sides_update(
        main=[
            Message(
                role="assistant",
                content="prefix tool turn",
                tool_calls=[ToolCall(id="pref_0", name="read_file", arguments="{}")],
            ),
            Message(role="tool", tool_call_id="pref_0", content="prefix output"),
            Message(
                role="assistant",
                content="anchor",
                tool_calls=[ToolCall(id="anchor_0", name="read_file", arguments="{}")],
            ),
            Message(role="tool", tool_call_id="anchor_0", content="anchor hidden output"),
        ]
    )

    assert len(update.main) == 4


def test_sides_update_main_accepts_closed_non_assistant_tail_without_anchor() -> None:
    update = _sides_update(
        main=[
            Message(role="user", content="u"),
            Message(role="system", content="s"),
            Message(role="developer", content="d"),
        ]
    )

    assert len(update.main) == 3


def test_sides_update_main_accepts_closed_assistant_with_user_tail() -> None:
    update = _sides_update(
        main=[
            Message(role="assistant", content="done"),
            Message(role="user", content="tail"),
        ]
    )

    assert len(update.main) == 2


def test_sides_update_main_rejects_message_patch_with_user_tail() -> None:
    assistant = Message(role="assistant", content="anchor")
    with pytest.raises(ValueError, match="message patch target may not have a trailing non-assistant tail"):
        _sides_update(
            main=[
                assistant,
                Message(role="user", content="tail"),
                MessagePatch(message=assistant),
            ]
        )


def test_sides_update_main_rejects_open_assistant_with_user_tail() -> None:
    with pytest.raises(ValueError, match="unresolved anchor tool calls may not have trailing non-assistant tail"):
        _sides_update(
            main=[
                Message(
                    role="assistant",
                    content="anchor",
                    tool_calls=[ToolCall(id="anchor_0", name="read_file", arguments="{}")],
                ),
                Message(role="user", content="tail"),
            ]
        )


def test_sides_update_main_rejects_unclosed_prefix_before_patch_anchor() -> None:
    anchor = Message(role="assistant", content="anchor")
    with pytest.raises(ValueError, match="must satisfy all prefix tool calls before the anchor"):
        _sides_update(
            main=[
                Message(
                    role="assistant",
                    content="prefix tool turn",
                    tool_calls=[ToolCall(id="pref_0", name="read_file", arguments="{}")],
                ),
                anchor,
                MessagePatch(message=anchor),
            ]
        )


def test_sides_update_main_rejects_unclosed_prefix_before_assistant_anchor() -> None:
    with pytest.raises(ValueError, match="must satisfy all prefix tool calls before the anchor"):
        _sides_update(
            main=[
                Message(
                    role="assistant",
                    content="prefix tool turn",
                    tool_calls=[ToolCall(id="pref_0", name="read_file", arguments="{}")],
                ),
                Message(role="assistant", content="anchor"),
            ]
        )


def test_sides_update_main_rejects_prefix_message_before_pending_tool_output_closes() -> None:
    anchor = Message(role="assistant", content="anchor")
    with pytest.raises(ValueError, match="cannot appear before earlier tool calls are satisfied"):
        _sides_update(
            main=[
                Message(
                    role="assistant",
                    content="prefix tool turn",
                    tool_calls=[ToolCall(id="pref_0", name="read_file", arguments="{}")],
                ),
                Message(role="user", content="interrupting prefix"),
                Message(role="tool", tool_call_id="pref_0", content="prefix output"),
                anchor,
                MessagePatch(message=anchor),
            ]
        )


def test_sides_update_main_rejects_suffix_tool_for_unknown_anchor_call() -> None:
    anchor = Message(
        role="assistant",
        content="anchor",
        tool_calls=[ToolCall(id="call_1", name="read_file", arguments="{}")],
    )
    with pytest.raises(ValueError, match="does not match an unresolved anchor tool call"):
        _sides_update(
            main=[
                anchor,
                Message(role="tool", tool_call_id="wrong", content="hidden output"),
                MessagePatch(message=anchor),
            ]
        )


def test_decode_queue_preserves_item_order() -> None:
    compaction = _sealed_compaction(_compaction_payload(durable={"active": []}, sides=Sides()))
    message = _message("hello")
    sealed_call = RequestFunctionCallItem(
        arguments='{"path":"README.md"}',
        call_id=_sealed_call_id("reviewer", "up_reviewer_0"),
        name="read_file",
        type="function_call",
    )
    fabricated_output = RequestFunctionCallOutputItem(call_id="not_openable", output="done", type="function_call_output")

    decoded = _decode_queue([compaction, message, sealed_call, fabricated_output], keyring=_keyring())

    assert [type(item) for item in decoded] == [
        _DecodedCompaction,
        _DecodedMessage,
        _DecodedSealedFunctionCall,
        _DecodedFabricatedFunctionCallOutput,
    ]


def test_decode_queue_rejects_unsealed_reasoning_input() -> None:
    item = RequestReasoningItem(
        encrypted_content=None,
        id=None,
        summary=[SummaryTextContent(text="sealed", type="summary_text")],
        type="reasoning",
    )

    with pytest.raises(PlapError):
        _decode_queue([item], keyring=_keyring())


async def test_ingest_response_request_returns_compaction_snapshot_for_carrier_only_queue() -> None:
    payload = _compaction_payload(
        durable={"active": ["reviewer"]},
        sides=Sides(
            messages={
                "main": [Message(role="assistant", content="main snapshot")],
                "reviewer": [Message(role="assistant", content="review snapshot")],
            }
        ),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[_sealed_compaction(payload)]),
        keyring=_keyring(),
    )

    assert result.durable == payload.durable
    assert result.sides == payload.sides
    assert result.last_reasoning_id is None
    assert result.last_compaction_id == payload.id


async def test_ingest_response_request_rejects_unresolved_call_inside_compaction_snapshot() -> None:
    payload = _compaction_payload(
        durable={},
        sides=Sides(
            active=set(),
            messages={
                "reviewer": [
                    Message(
                        role="assistant",
                        tool_calls=[ToolCall(id="up_reviewer_0", name="read_file", arguments='{"path":"README.md"}')],
                    )
                ]
            },
        ),
    )

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(model="plap/test", input=[_sealed_compaction(payload)]),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "compaction_contains_unresolved_tool_call"


async def test_ingest_response_request_accepts_reasoning_chain_after_compaction_snapshot() -> None:
    compaction = _compaction_payload(
        payload_id="cmp_root",
        durable={"active": []},
        sides=Sides(messages={"main": [Message(role="assistant", content="snapshot")]}),
    )
    first = _reasoning_payload(
        payload_id="rs_first",
        previous_compaction_id=compaction.id,
        durable=[{"op": "add", "path": "/meta", "value": {"step": 1}}],
        sides=_sides_update(main=[Message(role="assistant", content="first")]),
    )
    second = _reasoning_payload(
        payload_id="rs_second",
        previous_reasoning_id=first.id,
        previous_compaction_id=compaction.id,
        durable=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(
            main=[Message(role="assistant", content="second")],
        ),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[_sealed_compaction(compaction), _sealed_reasoning(first), _sealed_reasoning(second)],
        ),
        keyring=_keyring(),
    )

    assert result.durable == {"active": ["reviewer"], "meta": {"step": 1}}
    assert result.sides["main"] == [
        Message(role="assistant", content="snapshot"),
        Message(role="assistant", content="first"),
        Message(role="assistant", content="second"),
    ]
    assert result.last_reasoning_id == second.id
    assert result.last_compaction_id == compaction.id


async def test_ingest_response_request_applies_reasoning_durable_patch() -> None:
    payload = _reasoning_payload(
        durable=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(payload)]),
        keyring=_keyring(),
    )

    assert result.durable == {"active": ["reviewer"]}
    assert result.sides == Sides()
    assert result.last_reasoning_id == payload.id


async def test_ingest_response_request_applies_reasoning_non_main_side_patch() -> None:
    payload = _reasoning_payload(
        durable=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(
            patches={
                "reviewer": [
                    {"op": "add", "path": "/0", "value": {"role": "assistant", "content": "review hidden"}},
                ]
            }
        ),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(payload)]),
        keyring=_keyring(),
    )

    assert result.sides["reviewer"] == [Message(role="assistant", content="review hidden")]


async def test_ingest_response_request_appends_main_messages_from_reasoning() -> None:
    payload = _reasoning_payload(
        durable=[{"op": "add", "path": "/active", "value": []}],
        sides=_sides_update(main=[Message(role="assistant", content="main hidden")]),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(payload)]),
        keyring=_keyring(),
    )

    assert result.sides["main"] == [Message(role="assistant", content="main hidden")]


async def test_ingest_response_request_materializes_new_postfix_message_patch() -> None:
    assistant = Message(role="assistant", content="answer", reasoning_content="hidden")
    payload = _reasoning_payload(
        durable=[],
        sides=_sides_update(main=[assistant, MessagePatch(message=assistant)]),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(payload),
                RequestMessageItem(content="answer", role="assistant", type="message"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides["main"] == [assistant]


async def test_ingest_response_request_materializes_text_and_refusal_from_sealed_message() -> None:
    assistant = Message(
        role="assistant",
        content="partial answer",
        refusal="remaining request refused",
        reasoning_content="hidden",
    )
    payload = _reasoning_payload(
        durable=[],
        sides=_sides_update(main=[assistant, MessagePatch(message=assistant)]),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(payload),
                RequestMessageItem(
                    content=[
                        OutputTextContent(text="partial answer", type="output_text"),
                        OutputRefusalContent(refusal="remaining request refused", type="refusal"),
                    ],
                    role="assistant",
                    type="message",
                ),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides["main"] == [assistant]


async def test_ingest_response_request_materializes_parked_postfix_message_patch() -> None:
    assistant = Message(role="assistant", content="delayed answer", reasoning_content="hidden")
    parked = _reasoning_payload(
        durable=[],
        sides=_sides_update(active=set(), main=[assistant]),
    )
    published = _reasoning_payload(
        durable=[],
        sides=_sides_update(active={"main"}, main=[MessagePatch(message=assistant)]),
        previous_reasoning_id=parked.id,
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(parked),
                _sealed_reasoning(published),
                RequestMessageItem(content="delayed answer", role="assistant", type="message"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides.active == {"main"}
    assert result.sides["main"] == [assistant]


async def test_ingest_response_request_timewarps_parked_patch_after_fabricated_turn() -> None:
    assistant = Message(role="assistant", content="delayed answer", reasoning_content="hidden")
    parked = _reasoning_payload(
        durable=[],
        sides=_sides_update(active=set(), main=[assistant]),
    )
    published = _checkpoint_payload(
        durable={},
        sides=_sides_update(active={"main"}, main=[MessagePatch(message=assistant)]),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(parked),
                RequestMessageItem(content="new question", role="user", type="message"),
                RequestMessageItem(content="fabricated answer", role="assistant", type="message"),
                _sealed_reasoning(published),
                RequestMessageItem(content="delayed answer", role="assistant", type="message"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides["main"] == [
        Message(role="user", content="new question"),
        Message(role="assistant", content="fabricated answer"),
        assistant,
    ]


async def test_ingest_response_request_timewarp_preserves_equal_fabricated_assistant_multiplicity() -> None:
    assistant = Message(role="assistant", content="same")
    parked = _reasoning_payload(
        durable=[],
        sides=_sides_update(active=set(), main=[assistant]),
    )
    published = _checkpoint_payload(
        durable={},
        sides=_sides_update(active={"main"}, main=[MessagePatch(message=assistant)]),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(parked),
                RequestMessageItem(content="new question", role="user", type="message"),
                RequestMessageItem(content="same", role="assistant", type="message"),
                _sealed_reasoning(published),
                RequestMessageItem(content="same", role="assistant", type="message"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides["main"] == [
        Message(role="user", content="new question"),
        assistant,
        assistant,
    ]


async def test_ingest_response_request_timewarp_preserves_fabricated_assistant_mini_turn() -> None:
    assistant = Message(role="assistant", content="delayed answer", reasoning_content="hidden")
    parked = _reasoning_payload(
        durable=[],
        sides=_sides_update(active=set(), main=[assistant]),
    )
    published = _reasoning_payload(
        durable=[],
        sides=_sides_update(active={"main"}, main=[MessagePatch(message=assistant)]),
        previous_reasoning_id=parked.id,
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(parked),
                RequestMessageItem(content="fabricated answer", role="assistant", type="message"),
                RequestFunctionCallItem(arguments="{}", call_id="fab_0", name="fabricated", type="function_call"),
                RequestFunctionCallOutputItem(call_id="fab_0", output="fabricated result", type="function_call_output"),
                _sealed_reasoning(published),
                RequestMessageItem(content="delayed answer", role="assistant", type="message"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides["main"] == [
        Message(
            role="assistant",
            content="fabricated answer",
            tool_calls=[ToolCall(id="fab_0", name="fabricated", arguments="{}")],
        ),
        Message(role="tool", tool_call_id="fab_0", content="fabricated result"),
        assistant,
    ]


async def test_ingest_response_request_timewarps_across_irrelevant_reasoning_chain() -> None:
    assistant = Message(role="assistant", content="delayed answer", reasoning_content="hidden")
    payloads = [
        _reasoning_payload(
            payload_id="rs_1",
            durable=[],
            sides=_sides_update(active=set(), main=[assistant]),
        )
    ]
    payloads.append(
        _checkpoint_payload(
            payload_id="rs_2",
            durable={"step_2": True},
            sides=_sides_update(),
        )
    )
    for index in range(3, 10):
        payloads.append(
            _reasoning_payload(
                payload_id=f"rs_{index}",
                previous_reasoning_id=payloads[-1].id,
                durable=[{"op": "add", "path": f"/step_{index}", "value": True}],
                sides=_sides_update(),
            )
        )
    payloads.append(
        _reasoning_payload(
            payload_id="rs_10",
            previous_reasoning_id=payloads[-1].id,
            durable=[],
            sides=_sides_update(active={"main"}, main=[MessagePatch(message=assistant)]),
        )
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(payloads[0]),
                RequestMessageItem(content="new question", role="user", type="message"),
                RequestMessageItem(content="fabricated answer", role="assistant", type="message"),
                *[_sealed_reasoning(payload) for payload in payloads[1:]],
                RequestMessageItem(content="delayed answer", role="assistant", type="message"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides["main"] == [
        Message(role="user", content="new question"),
        Message(role="assistant", content="fabricated answer"),
        assistant,
    ]
    assert result.durable == {f"step_{index}": True for index in range(2, 10)}


async def test_ingest_response_request_timewarps_parked_call_without_interruption_stub() -> None:
    assistant = Message(
        role="assistant",
        content="delayed call",
        tool_calls=[ToolCall(id="up_main_0", name="client_tool", arguments="{}")],
    )
    parked = _reasoning_payload(
        durable=[],
        sides=_sides_update(active=set(), main=[assistant]),
    )
    published = _checkpoint_payload(
        durable={},
        sides=_sides_update(active={"main"}, main=[MessagePatch(message=assistant)]),
    )
    call_id = _sealed_call_id("main", "up_main_0")

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(parked),
                RequestMessageItem(content="new question", role="user", type="message"),
                RequestMessageItem(content="fabricated answer", role="assistant", type="message"),
                _sealed_reasoning(published),
                RequestMessageItem(content="delayed call", role="assistant", type="message"),
                RequestFunctionCallItem(arguments="{}", call_id=call_id, name="client_tool", type="function_call"),
                RequestFunctionCallOutputItem(call_id=call_id, output="client result", type="function_call_output"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides["main"] == [
        Message(role="user", content="new question"),
        Message(role="assistant", content="fabricated answer"),
        assistant,
        Message(role="tool", tool_call_id="up_main_0", content="client result"),
    ]


async def test_ingest_response_request_timewarps_partial_hidden_settlement() -> None:
    assistant = Message(
        role="assistant",
        content="delayed call",
        tool_calls=[
            ToolCall(id="up_main_0", name="server_tool", arguments="{}"),
            ToolCall(id="up_main_1", name="client_tool", arguments="{}"),
        ],
    )
    hidden_output = Message(role="tool", tool_call_id="up_main_0", content="server result")
    parked = _reasoning_payload(
        durable=[],
        sides=_sides_update(active=set(), main=[assistant]),
    )
    published = _checkpoint_payload(
        durable={},
        sides=_sides_update(
            active={"main"},
            main=[hidden_output, MessagePatch(message=assistant)],
        ),
    )
    call_id = _sealed_call_id("main", "up_main_1")

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(parked),
                RequestMessageItem(content="new question", role="user", type="message"),
                RequestMessageItem(content="fabricated answer", role="assistant", type="message"),
                _sealed_reasoning(published),
                RequestMessageItem(content="delayed call", role="assistant", type="message"),
                RequestFunctionCallItem(arguments="{}", call_id=call_id, name="client_tool", type="function_call"),
                RequestFunctionCallOutputItem(call_id=call_id, output="client result", type="function_call_output"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides["main"] == [
        Message(role="user", content="new question"),
        Message(role="assistant", content="fabricated answer"),
        assistant,
        hidden_output,
        Message(role="tool", tool_call_id="up_main_1", content="client result"),
    ]


async def test_ingest_response_request_timewarps_hidden_settlement_from_user_checkpoint() -> None:
    assistant = Message(
        role="assistant",
        content="delayed call",
        tool_calls=[
            ToolCall(id="up_main_0", name="server_tool", arguments="{}"),
            ToolCall(id="up_main_1", name="client_tool", arguments="{}"),
        ],
    )
    hidden_output = Message(role="tool", tool_call_id="up_main_0", content="server result")
    parked = _reasoning_payload(
        payload_id="rs_parked",
        durable=[],
        sides=_sides_update(active=set(), main=[assistant]),
    )
    settled = _checkpoint_payload(
        payload_id="rs_settled",
        durable={},
        sides=_sides_update(active={"main"}, main=[hidden_output, MessagePatch(message=assistant)]),
    )
    call_id = _sealed_call_id("main", "up_main_1")

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(parked),
                RequestMessageItem(content="new question", role="user", type="message"),
                RequestMessageItem(content="fabricated answer", role="assistant", type="message"),
                _sealed_reasoning(settled),
                RequestMessageItem(content="delayed call", role="assistant", type="message"),
                RequestFunctionCallItem(arguments="{}", call_id=call_id, name="client_tool", type="function_call"),
                RequestFunctionCallOutputItem(call_id=call_id, output="client result", type="function_call_output"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides["main"] == [
        Message(role="user", content="new question"),
        Message(role="assistant", content="fabricated answer"),
        assistant,
        hidden_output,
        Message(role="tool", tool_call_id="up_main_1", content="client result"),
    ]


async def test_ingest_response_request_normalizes_multiple_timewarps_in_one_queue() -> None:
    first = Message(role="assistant", content="first delayed", reasoning_content="first hidden")
    second = Message(role="assistant", content="second delayed", reasoning_content="second hidden")
    first_parked = _reasoning_payload(
        payload_id="rs_first_parked",
        durable=[],
        sides=_sides_update(active=set(), main=[first]),
    )
    first_published = _reasoning_payload(
        payload_id="rs_first_published",
        previous_reasoning_id=first_parked.id,
        durable=[],
        sides=_sides_update(active={"main"}, main=[MessagePatch(message=first)]),
    )
    second_parked = _reasoning_payload(
        payload_id="rs_second_parked",
        previous_reasoning_id=first_published.id,
        durable=[],
        sides=_sides_update(active=set(), main=[second]),
    )
    second_published = _reasoning_payload(
        payload_id="rs_second_published",
        previous_reasoning_id=second_parked.id,
        durable=[],
        sides=_sides_update(active={"main"}, main=[MessagePatch(message=second)]),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(first_parked),
                RequestMessageItem(content="first fabricated", role="assistant", type="message"),
                _sealed_reasoning(first_published),
                RequestMessageItem(content="first delayed", role="assistant", type="message"),
                _sealed_reasoning(second_parked),
                RequestMessageItem(content="second fabricated", role="assistant", type="message"),
                _sealed_reasoning(second_published),
                RequestMessageItem(content="second delayed", role="assistant", type="message"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides["main"] == [
        Message(role="assistant", content="first fabricated"),
        first,
        Message(role="assistant", content="second fabricated"),
        second,
    ]


async def test_ingest_response_request_preserves_equal_message_multiplicity_before_patch() -> None:
    assistant = Message(role="assistant", content="same", reasoning_content="hidden")
    first = _reasoning_payload(
        durable=[],
        sides=_sides_update(active=set(), main=[assistant]),
    )
    second = _reasoning_payload(
        durable=[],
        sides=_sides_update(
            active={"main"},
            main=[assistant, MessagePatch(message=assistant)],
        ),
        previous_reasoning_id=first.id,
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(first),
                _sealed_reasoning(second),
                RequestMessageItem(content="same", role="assistant", type="message"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides["main"] == [assistant, assistant]


async def test_ingest_response_request_does_not_deduplicate_unpatched_equal_assistant() -> None:
    hidden = Message(role="assistant", content="same", reasoning_content="hidden")
    payload = _reasoning_payload(
        durable=[],
        sides=_sides_update(active={"main"}, main=[hidden]),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(payload),
                RequestMessageItem(content="same", role="assistant", type="message"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides["main"] == [hidden, Message(role="assistant", content="same")]


async def test_ingest_response_request_preserves_unrelated_tail_for_postfix_source_mismatch() -> None:
    assistant = Message(role="assistant", content="answer", reasoning_content="hidden")
    other = Message(role="assistant", content="other", reasoning_content="other hidden")
    payload = _reasoning_payload(
        durable=[],
        sides=_sides_update(main=[assistant, MessagePatch(message=other)]),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(payload),
                RequestMessageItem(content="edited public", role="assistant", type="message"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides["main"] == [
        assistant,
        Message(role="assistant", content="edited public", reasoning_content="other hidden"),
    ]


async def test_ingest_response_request_applies_multiple_reasoning_items_in_order() -> None:
    first_payload = _reasoning_payload(
        durable=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(main=[Message(role="assistant", content="first")]),
    )
    second_payload = _reasoning_payload(
        durable=[{"op": "add", "path": "/meta", "value": {"step": 2}}],
        sides=_sides_update(main=[Message(role="assistant", content="second")]),
        previous_reasoning_id=first_payload.id,
    )
    first = _sealed_reasoning(first_payload)
    second = _sealed_reasoning(second_payload)

    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[first, second]),
        keyring=_keyring(),
    )

    assert result.durable == {"active": ["reviewer"], "meta": {"step": 2}}
    assert result.sides["main"] == [
        Message(role="assistant", content="first"),
        Message(role="assistant", content="second"),
    ]


async def test_ingest_response_request_rejects_first_reasoning_with_non_none_previous_reasoning_id() -> None:
    payload = _reasoning_payload(
        payload_id="rs_first",
        previous_reasoning_id="rs_missing",
        durable=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(),
    )

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(payload)]),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "reasoning_previous_reasoning_id_mismatch"


def test_sides_update_rejects_main_patch() -> None:
    with pytest.raises(ValueError, match="may not target main"):
        _sides_update(patches={"main": []})


async def test_ingest_response_request_rejects_second_reasoning_with_none_previous_reasoning_id() -> None:
    first = _reasoning_payload(
        payload_id="rs_first",
        durable=[{"op": "add", "path": "/meta", "value": {"step": 1}}],
        sides=_sides_update(),
    )
    second = _reasoning_payload(
        payload_id="rs_second",
        durable=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(),
    )

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(first), _sealed_reasoning(second)]),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "reasoning_previous_reasoning_id_mismatch"


async def test_ingest_response_request_accepts_duplicate_reasoning_payload_id_when_chain_matches() -> None:
    first = _reasoning_payload(
        payload_id="rs_duplicate",
        durable=[{"op": "add", "path": "/meta", "value": {"step": 1}}],
        sides=_sides_update(),
    )
    second = _reasoning_payload(
        payload_id="rs_duplicate",
        previous_reasoning_id=first.id,
        durable=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(first), _sealed_reasoning(second)]),
        keyring=_keyring(),
    )

    assert result.durable == {"active": ["reviewer"], "meta": {"step": 1}}


async def test_ingest_response_request_accepts_hidden_non_main_call_with_public_pair() -> None:
    assistant = Message(
        role="assistant",
        content="review hidden",
        tool_calls=[ToolCall(id="up_reviewer_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    payload = _reasoning_payload(
        durable=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(
            active={"main", "reviewer"},
            patches={
                "reviewer": [
                    {"op": "add", "path": "/0", "value": assistant.to_primitive()},
                ]
            },
        ),
    )
    call_id = _sealed_call_id_for_message("reviewer", "up_reviewer_0", assistant)

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(payload),
                RequestFunctionCallItem(
                    arguments='{"path":"README.md"}',
                    call_id=call_id,
                    name="read_file",
                    type="function_call",
                ),
                RequestFunctionCallOutputItem(call_id=call_id, output="review result", type="function_call_output"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides["reviewer"] == [
        assistant,
        Message(role="tool", tool_call_id="up_reviewer_0", content="review result"),
    ]


async def test_ingest_response_request_accepts_hidden_call_satisfied_by_hidden_output_only() -> None:
    assistant = Message(
        role="assistant",
        content="review hidden",
        tool_calls=[ToolCall(id="up_reviewer_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    hidden_output = {"role": "tool", "tool_call_id": "up_reviewer_0", "content": "hidden result"}
    payload = _reasoning_payload(
        durable=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(
            patches={
                "reviewer": [
                    {"op": "add", "path": "/0", "value": assistant.to_primitive()},
                    {"op": "add", "path": "/1", "value": hidden_output},
                ]
            }
        ),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(payload)]),
        keyring=_keyring(),
    )

    assert result.sides["reviewer"] == [
        assistant,
        Message(role="tool", tool_call_id="up_reviewer_0", content="hidden result"),
    ]


async def test_ingest_response_request_parks_inactive_call_across_reasoning() -> None:
    assistant = Message(
        role="assistant",
        content="review hidden",
        tool_calls=[ToolCall(id="up_reviewer_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    first = _reasoning_payload(
        durable=[],
        sides=_sides_update(
            patches={"reviewer": [{"op": "add", "path": "/0", "value": assistant.to_primitive()}]},
        ),
    )
    second = _reasoning_payload(
        durable=[{"op": "add", "path": "/step", "value": 2}],
        sides=_sides_update(),
        previous_reasoning_id=first.id,
    )

    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(first), _sealed_reasoning(second)]),
        keyring=_keyring(),
    )

    assert result.durable == {"step": 2}
    assert result.sides.active == {"main"}
    assert result.sides["reviewer"] == [assistant]


async def test_ingest_response_request_rejects_public_replay_of_parked_call() -> None:
    assistant = Message(
        role="assistant",
        tool_calls=[ToolCall(id="up_reviewer_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    payload = _reasoning_payload(
        durable=[],
        sides=_sides_update(
            patches={"reviewer": [{"op": "add", "path": "/0", "value": assistant.to_primitive()}]},
        ),
    )
    call_id = _sealed_call_id("reviewer", "up_reviewer_0")

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(
                model="plap/test",
                input=[
                    _sealed_reasoning(payload),
                    RequestFunctionCallItem(arguments='{"path":"README.md"}', call_id=call_id, name="read_file", type="function_call"),
                ],
            ),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "inactive_side_function_call"


async def test_ingest_response_request_replays_call_after_side_activation() -> None:
    assistant = Message(
        role="assistant",
        tool_calls=[ToolCall(id="up_reviewer_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    parked = _reasoning_payload(
        durable=[],
        sides=_sides_update(
            patches={"reviewer": [{"op": "add", "path": "/0", "value": assistant.to_primitive()}]},
        ),
    )
    activated = _reasoning_payload(
        durable=[],
        sides=_sides_update(active={"main", "reviewer"}),
        previous_reasoning_id=parked.id,
    )
    call_id = _sealed_call_id("reviewer", "up_reviewer_0")

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(parked),
                _sealed_reasoning(activated),
                RequestFunctionCallItem(arguments='{"path":"README.md"}', call_id=call_id, name="read_file", type="function_call"),
                RequestFunctionCallOutputItem(call_id=call_id, output="review result", type="function_call_output"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides.active == {"main", "reviewer"}
    assert result.sides["reviewer"][-1] == Message(
        role="tool",
        tool_call_id="up_reviewer_0",
        content="review result",
    )


async def test_ingest_response_request_requires_every_active_side_call_output() -> None:
    payload, main_call_id, reviewer_call_id = _parallel_active_calls()

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(
                model="plap/test",
                input=[
                    _sealed_reasoning(payload),
                    RequestFunctionCallItem(arguments='{"path":"main"}', call_id=main_call_id, name="read_file", type="function_call"),
                    RequestFunctionCallItem(
                        arguments='{"path":"reviewer"}', call_id=reviewer_call_id, name="read_file", type="function_call"
                    ),
                    RequestFunctionCallOutputItem(call_id=reviewer_call_id, output="review result", type="function_call_output"),
                ],
            ),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "function_call_missing_function_call_output"


async def test_ingest_response_request_accepts_reversed_outputs_for_all_active_sides() -> None:
    payload, main_call_id, reviewer_call_id = _parallel_active_calls()

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(payload),
                RequestFunctionCallItem(arguments='{"path":"main"}', call_id=main_call_id, name="read_file", type="function_call"),
                RequestFunctionCallItem(arguments='{"path":"reviewer"}', call_id=reviewer_call_id, name="read_file", type="function_call"),
                RequestFunctionCallOutputItem(call_id=reviewer_call_id, output="review result", type="function_call_output"),
                RequestFunctionCallOutputItem(call_id=main_call_id, output="main result", type="function_call_output"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides["main"][-1] == Message(role="tool", tool_call_id="up_main_0", content="main result")
    assert result.sides["reviewer"][-1] == Message(
        role="tool",
        tool_call_id="up_reviewer_0",
        content="review result",
    )


async def test_ingest_response_request_user_interrupts_only_parked_main_calls() -> None:
    assistant = Message(
        role="assistant",
        content="hidden",
        tool_calls=[ToolCall(id="up_main_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    payload = _reasoning_payload(
        durable=[],
        sides=_sides_update(active={"reviewer"}, main=[assistant]),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(payload), _message("new request")]),
        keyring=_keyring(),
    )

    assert result.sides.active == {"main", "reviewer"}
    assert result.sides["main"] == [
        assistant,
        Message(role="tool", tool_call_id="up_main_0", content="Tool call aborted by user."),
        Message(role="user", content="new request"),
    ]


async def test_ingest_response_request_fabricated_assistant_interrupts_parked_main_calls() -> None:
    assistant = Message(
        role="assistant",
        content="hidden",
        tool_calls=[ToolCall(id="up_main_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    payload = _reasoning_payload(
        durable=[],
        sides=_sides_update(active=set(), main=[assistant]),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(payload),
                RequestMessageItem(content="imported answer", role="assistant", type="message"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides.active == {"main"}
    assert result.sides["main"] == [
        assistant,
        Message(role="tool", tool_call_id="up_main_0", content="Tool call aborted by user."),
        Message(role="assistant", content="imported answer"),
    ]


async def test_ingest_response_request_fabricated_pair_attaches_to_inactive_main() -> None:
    assistant = Message(role="assistant", content="hidden")
    payload = _reasoning_payload(
        durable=[],
        sides=_sides_update(active=set(), main=[assistant]),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(payload),
                RequestFunctionCallItem(arguments="{}", call_id="fab_0", name="fabricated", type="function_call"),
                RequestFunctionCallOutputItem(call_id="fab_0", output="fabricated result", type="function_call_output"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides.active == set()
    assert result.sides["main"] == [
        Message(
            role="assistant",
            content="hidden",
            tool_calls=[ToolCall(id="fab_0", name="fabricated", arguments="{}")],
        ),
        Message(role="tool", tool_call_id="fab_0", content="fabricated result"),
    ]


async def test_ingest_response_request_fabricated_pair_preserves_other_parked_call() -> None:
    assistant = Message(
        role="assistant",
        content="hidden",
        tool_calls=[ToolCall(id="parked_0", name="parked", arguments="{}")],
    )
    payload = _reasoning_payload(
        durable=[],
        sides=_sides_update(active=set(), main=[assistant]),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(payload),
                RequestFunctionCallItem(arguments="{}", call_id="fab_0", name="fabricated", type="function_call"),
                RequestFunctionCallOutputItem(call_id="fab_0", output="fabricated result", type="function_call_output"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides.active == set()
    assert result.sides["main"] == [
        Message(
            role="assistant",
            content="hidden",
            tool_calls=[
                ToolCall(id="parked_0", name="parked", arguments="{}"),
                ToolCall(id="fab_0", name="fabricated", arguments="{}"),
            ],
        ),
        Message(role="tool", tool_call_id="fab_0", content="fabricated result"),
    ]


async def test_ingest_response_request_fabricated_pair_settles_matching_parked_call() -> None:
    assistant = Message(
        role="assistant",
        content="hidden",
        tool_calls=[ToolCall(id="parked_0", name="parked", arguments="{}")],
    )
    payload = _reasoning_payload(
        durable=[],
        sides=_sides_update(active=set(), main=[assistant]),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(payload),
                RequestFunctionCallItem(arguments="{}", call_id="parked_0", name="parked", type="function_call"),
                RequestFunctionCallOutputItem(call_id="parked_0", output="fabricated result", type="function_call_output"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides.active == set()
    assert result.sides["main"] == [
        assistant,
        Message(role="tool", tool_call_id="parked_0", content="fabricated result"),
    ]


async def test_ingest_response_request_sealed_transplant_attaches_to_inactive_main() -> None:
    assistant = Message(role="assistant", content="hidden")
    payload = _reasoning_payload(
        durable=[],
        sides=_sides_update(active=set(), main=[assistant]),
    )
    call_id = _sealed_call_id("main", "transplanted_0")

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(payload),
                RequestFunctionCallItem(arguments="{}", call_id=call_id, name="transplanted", type="function_call"),
                RequestFunctionCallOutputItem(call_id=call_id, output="transplanted result", type="function_call_output"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides.active == set()
    assert result.sides["main"] == [
        Message(
            role="assistant",
            content="hidden",
            tool_calls=[ToolCall(id="transplanted_0", name="transplanted", arguments="{}")],
        ),
        Message(role="tool", tool_call_id="transplanted_0", content="transplanted result"),
    ]


async def test_ingest_response_request_sealed_transplant_preserves_other_parked_call() -> None:
    assistant = Message(
        role="assistant",
        content="hidden",
        tool_calls=[ToolCall(id="parked_0", name="parked", arguments="{}")],
    )
    payload = _reasoning_payload(
        durable=[],
        sides=_sides_update(active=set(), main=[assistant]),
    )
    call_id = _sealed_call_id("main", "transplanted_0")

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(payload),
                RequestFunctionCallItem(arguments="{}", call_id=call_id, name="transplanted", type="function_call"),
                RequestFunctionCallOutputItem(call_id=call_id, output="transplanted result", type="function_call_output"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides.active == set()
    assert result.sides["main"] == [
        Message(
            role="assistant",
            content="hidden",
            tool_calls=[
                ToolCall(id="parked_0", name="parked", arguments="{}"),
                ToolCall(id="transplanted_0", name="transplanted", arguments="{}"),
            ],
        ),
        Message(role="tool", tool_call_id="transplanted_0", content="transplanted result"),
    ]


async def test_ingest_response_request_rejects_sealed_replay_of_inactive_main_parked_call() -> None:
    assistant = Message(
        role="assistant",
        content="hidden",
        tool_calls=[ToolCall(id="parked_0", name="parked", arguments="{}")],
    )
    payload = _reasoning_payload(
        durable=[],
        sides=_sides_update(active=set(), main=[assistant]),
    )
    call_id = _sealed_call_id("main", "parked_0")

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(
                model="plap/test",
                input=[
                    _sealed_reasoning(payload),
                    RequestFunctionCallItem(arguments="{}", call_id=call_id, name="parked", type="function_call"),
                ],
            ),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "inactive_side_function_call"


async def test_ingest_response_request_materializes_partially_settled_baseline() -> None:
    assistant = Message(
        role="assistant",
        content="working",
        tool_calls=[
            ToolCall(id="up_main_0", name="subagent", arguments="{}"),
            ToolCall(id="up_main_1", name="client_tool", arguments="{}"),
        ],
    )
    parked = _reasoning_payload(
        durable=[],
        sides=_sides_update(active=set(), main=[assistant]),
    )
    hidden_output = Message(role="tool", tool_call_id="up_main_0", content="subagent result")
    published = _reasoning_payload(
        durable=[],
        sides=_sides_update(
            active={"main"},
            main=[hidden_output, MessagePatch(message=assistant)],
        ),
        previous_reasoning_id=parked.id,
    )
    call_id = _sealed_call_id_for_message("main", "up_main_1", assistant, tool_call_index=1)

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(parked),
                _sealed_reasoning(published),
                RequestMessageItem(content="working", role="assistant", type="message"),
                RequestFunctionCallItem(
                    arguments="{}",
                    call_id=call_id,
                    name="client_tool",
                    type="function_call",
                ),
                RequestFunctionCallOutputItem(call_id=call_id, output="client result", type="function_call_output"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides["main"] == [
        assistant,
        hidden_output,
        Message(role="tool", tool_call_id="up_main_1", content="client result"),
    ]


async def test_ingest_response_request_keeps_fully_settled_baseline_hidden() -> None:
    assistant = Message(
        role="assistant",
        content="working",
        tool_calls=[ToolCall(id="up_main_0", name="subagent", arguments="{}")],
    )
    parked = _reasoning_payload(
        durable=[],
        sides=_sides_update(active=set(), main=[assistant]),
    )
    hidden_output = Message(role="tool", tool_call_id="up_main_0", content="subagent result")
    settled = _reasoning_payload(
        durable=[],
        sides=_sides_update(main=[hidden_output]),
        previous_reasoning_id=parked.id,
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[_sealed_reasoning(parked), _sealed_reasoning(settled)],
        ),
        keyring=_keyring(),
    )

    assert result.sides["main"] == [assistant, hidden_output]


async def test_ingest_response_request_user_does_not_satisfy_open_main_call() -> None:
    assistant = Message(
        role="assistant",
        tool_calls=[ToolCall(id="up_main_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    payload = _reasoning_payload(durable=[], sides=_sides_update(main=[assistant]))
    call_id = _sealed_call_id("main", "up_main_0")

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(
                model="plap/test",
                input=[
                    _sealed_reasoning(payload),
                    RequestFunctionCallItem(
                        arguments='{"path":"README.md"}',
                        call_id=call_id,
                        name="read_file",
                        type="function_call",
                    ),
                    _message("new request"),
                ],
            ),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "function_call_missing_function_call_output"


async def test_ingest_response_request_accepts_main_hidden_call_with_public_pair() -> None:
    assistant = Message(
        role="assistant",
        content="main hidden",
        tool_calls=[ToolCall(id="up_main_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    payload = _reasoning_payload(
        durable=[{"op": "add", "path": "/active", "value": ["main"]}],
        sides=_sides_update(main=[assistant]),
    )
    call_id = _sealed_call_id_for_message("main", "up_main_0", assistant)

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(payload),
                RequestFunctionCallItem(
                    arguments='{"path":"README.md"}',
                    call_id=call_id,
                    name="read_file",
                    type="function_call",
                ),
                RequestFunctionCallOutputItem(call_id=call_id, output="main result", type="function_call_output"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides["main"] == [
        assistant,
        Message(role="tool", tool_call_id="up_main_0", content="main result"),
    ]


async def test_ingest_response_request_accepts_fabricated_main_pair_after_empty_reasoning_step() -> None:
    initial = Message(role="assistant", content="main hidden")
    first = _reasoning_payload(
        durable=[{"op": "add", "path": "/active", "value": ["main"]}],
        sides=_sides_update(main=[initial]),
    )
    second = _reasoning_payload(
        durable=[{"op": "add", "path": "/meta", "value": {"step": 2}}],
        sides=_sides_update(),
        previous_reasoning_id=first.id,
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(first),
                _sealed_reasoning(second),
                RequestFunctionCallItem(
                    arguments='{"path":"README.md"}',
                    call_id="fab_0",
                    name="read_file",
                    type="function_call",
                ),
                RequestFunctionCallOutputItem(call_id="fab_0", output="main result", type="function_call_output"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides["main"] == [
        Message(
            role="assistant",
            content="main hidden",
            tool_calls=[ToolCall(id="fab_0", name="read_file", arguments='{"path":"README.md"}')],
        ),
        Message(role="tool", tool_call_id="fab_0", content="main result"),
    ]


async def test_ingest_response_request_rejects_reasoning_after_hidden_main_open_call() -> None:
    payload = _reasoning_payload(
        durable=[{"op": "add", "path": "/active", "value": ["main"]}],
        sides=_sides_update(
            main=[
                Message(
                    role="assistant",
                    content="main hidden",
                    tool_calls=[ToolCall(id="up_main_0", name="read_file", arguments='{"path":"README.md"}')],
                )
            ]
        ),
    )
    later = _sealed_reasoning(
        _reasoning_payload(
            durable=[{"op": "add", "path": "/meta", "value": {"step": 2}}],
            sides=_sides_update(),
            previous_reasoning_id=payload.id,
        )
    )

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(payload), later]),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "pending_tool_outputs_block_message"


async def test_ingest_response_request_rejects_hidden_call_missing_public_function_call_item() -> None:
    assistant = Message(
        role="assistant",
        content="review hidden",
        tool_calls=[ToolCall(id="up_reviewer_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    payload = _reasoning_payload(
        durable=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(
            active={"main", "reviewer"},
            patches={
                "reviewer": [
                    {"op": "add", "path": "/0", "value": assistant.to_primitive()},
                ]
            },
        ),
    )

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(payload)]),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "reasoning_tool_call_missing_function_call_item"


async def test_ingest_response_request_rejects_function_call_output_without_pending_function_call() -> None:
    assistant = Message(
        role="assistant",
        content="review hidden",
        tool_calls=[ToolCall(id="up_reviewer_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    payload = _reasoning_payload(
        durable=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(
            active={"main", "reviewer"},
            patches={
                "reviewer": [
                    {"op": "add", "path": "/0", "value": assistant.to_primitive()},
                ]
            },
        ),
    )
    call_id = _sealed_call_id_for_message("reviewer", "up_reviewer_0", assistant)

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(
                model="plap/test",
                input=[
                    _sealed_reasoning(payload),
                    RequestFunctionCallOutputItem(call_id=call_id, output="review result", type="function_call_output"),
                ],
            ),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "function_call_output_without_pending_function_call"


async def test_ingest_response_request_rejects_duplicate_public_function_call_item() -> None:
    assistant = Message(
        role="assistant",
        content="review hidden",
        tool_calls=[ToolCall(id="up_reviewer_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    payload = _reasoning_payload(
        durable=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(
            active={"main", "reviewer"},
            patches={
                "reviewer": [
                    {"op": "add", "path": "/0", "value": assistant.to_primitive()},
                ]
            },
        ),
    )
    call_id = _sealed_call_id_for_message("reviewer", "up_reviewer_0", assistant)
    function_call = RequestFunctionCallItem(
        arguments='{"path":"README.md"}',
        call_id=call_id,
        name="read_file",
        type="function_call",
    )

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(payload), function_call, function_call]),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "duplicate_pending_function_call"


async def test_ingest_response_request_rejects_same_side_reasoning_before_public_output() -> None:
    assistant = Message(
        role="assistant",
        content="review hidden",
        tool_calls=[ToolCall(id="up_reviewer_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    first_payload = _reasoning_payload(
        durable=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(
            active={"main", "reviewer"},
            patches={
                "reviewer": [
                    {"op": "add", "path": "/0", "value": assistant.to_primitive()},
                ]
            },
        ),
    )
    second_payload = _reasoning_payload(
        durable=[{"op": "add", "path": "/meta", "value": {"step": 2}}],
        sides=_sides_update(
            patches={
                "reviewer": [
                    {"op": "add", "path": "/1", "value": {"role": "assistant", "content": "interleaving"}},
                ]
            },
        ),
        previous_reasoning_id=first_payload.id,
    )
    first = _sealed_reasoning(first_payload)
    second = _sealed_reasoning(second_payload)
    call_id = _sealed_call_id_for_message("reviewer", "up_reviewer_0", assistant)

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(
                model="plap/test",
                input=[
                    first,
                    RequestFunctionCallItem(
                        arguments='{"path":"README.md"}',
                        call_id=call_id,
                        name="read_file",
                        type="function_call",
                    ),
                    second,
                ],
            ),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "pending_tool_outputs_block_message"


async def test_ingest_response_request_rejects_durable_only_reasoning_while_waiting_for_output() -> None:
    assistant = Message(
        role="assistant",
        content="review hidden",
        tool_calls=[ToolCall(id="up_reviewer_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    first_payload = _reasoning_payload(
        durable=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(
            active={"main", "reviewer"},
            patches={
                "reviewer": [
                    {"op": "add", "path": "/0", "value": assistant.to_primitive()},
                ]
            },
        ),
    )
    second_payload = _reasoning_payload(
        durable=[{"op": "add", "path": "/meta", "value": {"step": 2}}],
        sides=_sides_update(),
        previous_reasoning_id=first_payload.id,
    )
    first = _sealed_reasoning(first_payload)
    second = _sealed_reasoning(second_payload)
    call_id = _sealed_call_id_for_message("reviewer", "up_reviewer_0", assistant)

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(
                model="plap/test",
                input=[
                    first,
                    RequestFunctionCallItem(
                        arguments='{"path":"README.md"}',
                        call_id=call_id,
                        name="read_file",
                        type="function_call",
                    ),
                    second,
                    RequestFunctionCallOutputItem(call_id=call_id, output="review result", type="function_call_output"),
                ],
            ),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "pending_tool_outputs_block_message"


async def test_ingest_response_request_rejects_source_less_patch_without_reasoning_slice() -> None:
    assistant = Message(role="assistant", content="answer", reasoning_content="hidden assistant state")
    payload = _reasoning_payload(
        durable=[{"op": "add", "path": "/active", "value": []}],
        sides=_sides_update(main=[MessagePatch(message=assistant)]),
    )

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(payload)]),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "main_message_patch_target_missing"


async def test_ingest_response_request_accepts_standalone_main_message() -> None:
    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[_message("hello")]),
        keyring=_keyring(),
    )

    assert result.sides["main"] == [Message(role="user", content="hello")]
    assert result.sides.active == {"main"}


async def test_ingest_response_request_discards_naked_non_main_sealed_function_call() -> None:
    item = RequestFunctionCallItem(
        arguments='{"path":"README.md"}',
        call_id=_sealed_call_id("reviewer", "up_reviewer_0"),
        name="read_file",
        type="function_call",
    )

    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[item]),
        keyring=_keyring(),
    )

    assert result.sides.get("reviewer", []) == []


async def test_ingest_response_request_rejects_naked_main_sealed_pair_without_anchor() -> None:
    synthetic = Message(role="assistant", content="")
    call_id = _sealed_call_id_for_message("main", "syn_0", synthetic)

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(
                model="plap/test",
                input=[
                    RequestFunctionCallItem(
                        arguments='{"path":"README.md"}',
                        call_id=call_id,
                        name="read_file",
                        type="function_call",
                    ),
                    RequestFunctionCallOutputItem(call_id=call_id, output="fo_0", type="function_call_output"),
                ],
            ),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "sealed_function_call_without_attachment_owner"


async def test_ingest_response_request_accepts_closed_anchor_then_stripped_hidden_empty_pair() -> None:
    synthetic = Message(role="assistant", content="")
    call_id = _sealed_call_id_for_message("main", "syn_0", synthetic)

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                RequestMessageItem(content="m", role="assistant", type="message"),
                RequestFunctionCallItem(
                    arguments='{"path":"README.md"}',
                    call_id=call_id,
                    name="read_file",
                    type="function_call",
                ),
                RequestFunctionCallOutputItem(call_id=call_id, output="fo_0", type="function_call_output"),
            ],
        ),
        keyring=_keyring(),
    )

    assert result.sides["main"] == [
        Message(role="assistant", content="m", tool_calls=[ToolCall(id="syn_0", name="read_file", arguments='{"path":"README.md"}')]),
        Message(role="tool", tool_call_id="syn_0", content="fo_0"),
    ]


async def test_ingest_response_request_discards_naked_non_main_sealed_pair_even_without_history_match() -> None:
    item = RequestFunctionCallItem(
        arguments='{"path":"README.md"}',
        call_id=_sealed_call_id("reviewer", "up_reviewer_0"),
        name="read_file",
        type="function_call",
    )

    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[item]),
        keyring=_keyring(),
    )

    assert result.sides.get("reviewer", []) == []


async def test_ingest_response_request_fails_closed_on_invalid_durable_patch() -> None:
    payload = _reasoning_payload(
        durable=[{"op": "replace", "path": "/missing", "value": 1}],
        sides=_sides_update(),
    )

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(payload)]),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "reasoning_durable_patch_invalid"


async def test_ingest_response_request_fails_closed_on_invalid_side_patch() -> None:
    payload = _reasoning_payload(
        durable=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(
            patches={
                "reviewer": [
                    {"op": "add", "path": "/0", "value": 1},
                ]
            }
        ),
    )

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(payload)]),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "reasoning_side_patch_invalid"


def test_assistant_message_content_roundtrip_normalizes_single_text_part() -> None:
    item = RequestMessageItem(role="assistant", content=[OutputTextContent(text="hello world", type="output_text")], type="message")
    decoded = decode_message(item)

    assert isinstance(decoded.content, str)
    assert decoded.content == "hello world"
    assert decoded.refusal is None
