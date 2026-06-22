from __future__ import annotations

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
    MAIN_SIDE,
    CallID,
    CompactionPayload,
    GuardedPatch,
    Message,
    MessagePatch,
    ReasoningPayload,
    Side,
    Sides,
    SidesUpdate,
    ToolCall,
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
        MAIN_SIDE: 0,
        "defender": 1,
        "reviewer": 2,
        "arbitrator": 3,
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


def _next_reasoning_payload_id() -> str:
    return f"rs_payload_{next(_REASONING_PAYLOAD_COUNTER)}"


def _next_compaction_payload_id() -> str:
    return f"cmp_payload_{next(_COMPACTION_PAYLOAD_COUNTER)}"


def _compaction_payload(
    *,
    machine: dict[str, object],
    sides: Sides,
    payload_id: str | None = None,
) -> CompactionPayload:
    return CompactionPayload(id=payload_id or _next_compaction_payload_id(), machine=machine, sides=sides)


def _reasoning_payload(
    *,
    machine: list[dict[str, object]],
    sides: SidesUpdate,
    payload_id: str | None = None,
    previous_reasoning_id: str | None = None,
    previous_compaction_id: str | None = None,
) -> ReasoningPayload:
    return ReasoningPayload(
        id=payload_id or _next_reasoning_payload_id(),
        previous_reasoning_id=previous_reasoning_id,
        previous_compaction_id=previous_compaction_id,
        machine=machine,
        sides=sides,
    )


def _sides_update(
    *,
    main: list[Message | MessagePatch] | None = None,
    patches: dict[Side, list[dict[str, object]] | None] | None = None,
    current: Sides | None = None,
) -> SidesUpdate:
    current_sides = Sides() if current is None else current
    normalized_patches = {
        side: _guarded_patch(side, current_sides.get(side), patch) for side, patch in ({} if patches is None else patches).items()
    }
    return SidesUpdate(main=[] if main is None else list(main), patches=normalized_patches)


def _guarded_patch(side: Side, current: list[Message] | None, patch: list[dict[str, object]] | None) -> GuardedPatch:
    return GuardedPatch(
        shape=None if current is None else Sides(messages={side: list(current)}).shape(side),
        patch=patch,
    )


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
    payload = _compaction_payload(machine={"active": []}, sides=Sides())

    decoded = _decode_queue([_sealed_compaction(payload)], keyring=_keyring())

    assert decoded == [_DecodedCompaction(payload=payload)]


def test_decode_queue_opens_reasoning_payload() -> None:
    payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": []}],
        sides=_sides_update(),
    )

    decoded = _decode_queue([_sealed_reasoning(payload)], keyring=_keyring())

    assert decoded == [_DecodedReasoning(payload=payload)]


def test_reasoning_payload_allows_empty_delta() -> None:
    payload = _reasoning_payload(machine=[], sides=_sides_update())

    assert payload.machine == []
    assert payload.sides == _sides_update()


def test_decode_queue_rejects_reasoning_item_id_mismatch() -> None:
    payload = _reasoning_payload(machine=[{"op": "add", "path": "/active", "value": []}], sides=_sides_update())
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
    payload = _compaction_payload(machine={"active": []}, sides=Sides())
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
                content=[ChatContentText(text="hello")],
                refusal="nope",
            )
        )
    ]


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


def test_sides_update_main_accepts_single_patch_followed_by_trailing_tool_messages() -> None:
    update = _sides_update(
        main=[
            Message(role="assistant", content="prefix"),
            MessagePatch(content_hash="abcd", tool_calls=[ToolCall(id="call_1", name="read_file", arguments="{}")]),
            Message(role="tool", tool_call_id="call_1", content="hidden output"),
        ]
    )

    assert len(update.main) == 3


def test_sides_update_main_rejects_second_patch() -> None:
    with pytest.raises(ValueError, match="at most one message patch"):
        _sides_update(
            main=[
                MessagePatch(content_hash="abcd", reasoning_content="first"),
                MessagePatch(content_hash="efgh", reasoning_content="second"),
            ]
        )


def test_sides_update_main_rejects_non_tool_message_after_patch() -> None:
    with pytest.raises(ValueError, match="message patch must be the last non-tool main update"):
        _sides_update(
            main=[
                MessagePatch(content_hash="abcd", reasoning_content="hidden"),
                Message(role="assistant", content="later assistant"),
            ]
        )


def test_sides_update_main_rejects_tool_message_without_tool_call_id_after_patch() -> None:
    with pytest.raises(ValueError, match="must be a tool message with tool_call_id after the anchor"):
        _sides_update(
            main=[
                MessagePatch(content_hash="abcd", tool_calls=[ToolCall(id="call_1", name="read_file", arguments="{}")]),
                Message(role="tool", content="missing id"),
            ]
        )


def test_sides_update_main_accepts_closed_prefix_before_patch_anchor() -> None:
    update = _sides_update(
        main=[
            Message(
                role="assistant",
                content="prefix tool turn",
                tool_calls=[ToolCall(id="pref_0", name="read_file", arguments="{}")],
            ),
            Message(role="tool", tool_call_id="pref_0", content="prefix output"),
            MessagePatch(content_hash="abcd", tool_calls=[ToolCall(id="call_1", name="read_file", arguments="{}")]),
            Message(role="tool", tool_call_id="call_1", content="anchor hidden output"),
        ]
    )

    assert len(update.main) == 4


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
    with pytest.raises(ValueError, match="message patch anchor may not have trailing non-assistant tail"):
        _sides_update(
            main=[
                MessagePatch(content_hash="abcd", reasoning_content="hidden"),
                Message(role="user", content="tail"),
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
    with pytest.raises(ValueError, match="must satisfy all prefix tool calls before the anchor"):
        _sides_update(
            main=[
                Message(
                    role="assistant",
                    content="prefix tool turn",
                    tool_calls=[ToolCall(id="pref_0", name="read_file", arguments="{}")],
                ),
                MessagePatch(content_hash="abcd", reasoning_content="hidden"),
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
                MessagePatch(content_hash="abcd", reasoning_content="hidden"),
            ]
        )


def test_sides_update_main_rejects_suffix_tool_for_unknown_anchor_call() -> None:
    with pytest.raises(ValueError, match="does not match an unresolved anchor tool call"):
        _sides_update(
            main=[
                MessagePatch(content_hash="abcd", tool_calls=[ToolCall(id="call_1", name="read_file", arguments="{}")]),
                Message(role="tool", tool_call_id="wrong", content="hidden output"),
            ]
        )


def test_guarded_patch_from_primitive_requires_shape_key() -> None:
    with pytest.raises(ValueError, match="guarded patch is missing keys: shape"):
        GuardedPatch.from_primitive({"patch": []})


def test_guarded_patch_from_primitive_requires_patch_key() -> None:
    with pytest.raises(ValueError, match="guarded patch is missing keys: patch"):
        GuardedPatch.from_primitive({"shape": None})


def test_sides_shape_ignores_textual_leaves() -> None:
    first = Sides(
        messages={
            MAIN_SIDE: [
                Message(
                    role="assistant",
                    content="hello",
                    tool_calls=[ToolCall(id="call_0", name="read_file", arguments='{"path":"README.md"}')],
                    reasoning_content="because",
                ),
                Message(role="tool", tool_call_id="call_0", content="first output"),
            ],
            "reviewer": [Message(role="assistant", content="review one")],
        },
    )
    second = Sides(
        messages={
            MAIN_SIDE: [
                Message(
                    role="assistant",
                    content="goodbye",
                    tool_calls=[ToolCall(id="call_0", name="list_files", arguments='{"path":"docs"}')],
                    reasoning_content="therefore",
                ),
                Message(role="tool", tool_call_id="call_0", content="second output"),
            ],
            "reviewer": [Message(role="assistant", content="review two")],
        },
    )

    assert first.shape(MAIN_SIDE) == second.shape(MAIN_SIDE)


def test_sides_shape_preserves_tool_call_edges() -> None:
    first = Sides(
        messages={
            MAIN_SIDE: [
                Message(role="assistant", tool_calls=[ToolCall(id="call_0", name="read_file", arguments="{}")]),
                Message(role="tool", tool_call_id="call_0", content="output"),
            ]
        }
    )
    second = Sides(
        messages={
            MAIN_SIDE: [
                Message(role="assistant", tool_calls=[ToolCall(id="call_1", name="read_file", arguments="{}")]),
                Message(role="tool", tool_call_id="call_1", content="output"),
            ]
        }
    )

    assert first.shape(MAIN_SIDE) != second.shape(MAIN_SIDE)


def test_decode_queue_preserves_item_order() -> None:
    compaction = _sealed_compaction(_compaction_payload(machine={"active": []}, sides=Sides()))
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
        machine={"active": ["reviewer"]},
        sides=Sides(
            messages={
                MAIN_SIDE: [Message(role="assistant", content="main snapshot")],
                "reviewer": [Message(role="assistant", content="review snapshot")],
            }
        ),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[_sealed_compaction(payload)]),
        keyring=_keyring(),
    )

    assert result.machine == payload.machine
    assert result.sides == payload.sides
    assert result.last_side is None
    assert result.last_reasoning_id is None
    assert result.current_compaction_id == payload.id


async def test_ingest_response_request_accepts_reasoning_chain_anchored_to_compaction() -> None:
    compaction = _compaction_payload(
        payload_id="cmp_root",
        machine={"active": []},
        sides=Sides(messages={MAIN_SIDE: [Message(role="assistant", content="snapshot")]}),
    )
    first = _reasoning_payload(
        payload_id="rs_first",
        previous_compaction_id=compaction.id,
        machine=[{"op": "add", "path": "/meta", "value": {"step": 1}}],
        sides=_sides_update(main=[Message(role="assistant", content="first")], current=compaction.sides),
    )
    second = _reasoning_payload(
        payload_id="rs_second",
        previous_reasoning_id=first.id,
        previous_compaction_id=compaction.id,
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(
            main=[Message(role="assistant", content="second")],
            current=Sides(
                messages={
                    MAIN_SIDE: [
                        Message(role="assistant", content="snapshot"),
                        Message(role="assistant", content="first"),
                    ]
                }
            ),
        ),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[_sealed_compaction(compaction), _sealed_reasoning(first), _sealed_reasoning(second)],
        ),
        keyring=_keyring(),
    )

    assert result.machine == {"active": ["reviewer"], "meta": {"step": 1}}
    assert result.sides[MAIN_SIDE] == [
        Message(role="assistant", content="snapshot"),
        Message(role="assistant", content="first"),
        Message(role="assistant", content="second"),
    ]
    assert result.last_side is None
    assert result.last_reasoning_id == second.id
    assert result.current_compaction_id == compaction.id


async def test_ingest_response_request_applies_reasoning_machine_patch() -> None:
    payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(payload)]),
        keyring=_keyring(),
    )

    assert result.machine == {"active": ["reviewer"]}
    assert result.sides == Sides()
    assert result.last_side is None
    assert result.last_reasoning_id == payload.id
    assert result.current_compaction_id is None


async def test_ingest_response_request_applies_reasoning_non_main_side_patch() -> None:
    payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
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
    assert result.last_side is None


async def test_ingest_response_request_appends_main_messages_from_reasoning() -> None:
    payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": []}],
        sides=_sides_update(main=[Message(role="assistant", content="main hidden")]),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(payload)]),
        keyring=_keyring(),
    )

    assert result.sides[MAIN_SIDE] == [Message(role="assistant", content="main hidden")]
    assert result.last_side is None


async def test_ingest_response_request_applies_multiple_reasoning_items_in_order() -> None:
    first_payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(main=[Message(role="assistant", content="first")]),
    )
    second_payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/meta", "value": {"step": 2}}],
        sides=_sides_update(
            main=[Message(role="assistant", content="second")],
            current=Sides(messages={MAIN_SIDE: [Message(role="assistant", content="first")]}),
        ),
        previous_reasoning_id=first_payload.id,
    )
    first = _sealed_reasoning(first_payload)
    second = _sealed_reasoning(second_payload)

    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[first, second]),
        keyring=_keyring(),
    )

    assert result.machine == {"active": ["reviewer"], "meta": {"step": 2}}
    assert result.sides[MAIN_SIDE] == [
        Message(role="assistant", content="first"),
        Message(role="assistant", content="second"),
    ]
    assert result.last_side is None


async def test_ingest_response_request_rejects_first_reasoning_with_non_none_previous_reasoning_id() -> None:
    payload = _reasoning_payload(
        payload_id="rs_first",
        previous_reasoning_id="rs_missing",
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(),
    )

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(payload)]),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "reasoning_previous_reasoning_id_mismatch"


async def test_ingest_response_request_rejects_first_reasoning_with_non_none_previous_compaction_id() -> None:
    payload = _reasoning_payload(
        payload_id="rs_first",
        previous_compaction_id="cmp_missing",
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(),
    )

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(payload)]),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "reasoning_previous_compaction_id_mismatch"


async def test_ingest_response_request_rejects_reasoning_with_none_previous_compaction_id_after_compaction() -> None:
    compaction = _compaction_payload(payload_id="cmp_root", machine={"active": []}, sides=Sides())
    payload = _reasoning_payload(
        payload_id="rs_first",
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(),
    )

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(model="plap/test", input=[_sealed_compaction(compaction), _sealed_reasoning(payload)]),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "reasoning_previous_compaction_id_mismatch"


async def test_ingest_response_request_rejects_reasoning_when_guarded_patch_shape_drifts() -> None:
    compaction = _compaction_payload(
        payload_id="cmp_root",
        machine={"active": []},
        sides=Sides(messages={MAIN_SIDE: [Message(role="assistant", content="snapshot")]}),
    )
    payload = _reasoning_payload(
        payload_id="rs_first",
        previous_compaction_id=compaction.id,
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(
            patches={
                MAIN_SIDE: [{"op": "add", "path": "/1", "value": {"role": "assistant", "content": "hidden"}}],
            },
            current=compaction.sides,
        ),
    )

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(model="plap/test", input=[_sealed_compaction(compaction), _message("drift"), _sealed_reasoning(payload)]),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "reasoning_sides_shape_mismatch"


async def test_ingest_response_request_rejects_second_reasoning_with_none_previous_reasoning_id() -> None:
    first = _reasoning_payload(
        payload_id="rs_first",
        machine=[{"op": "add", "path": "/meta", "value": {"step": 1}}],
        sides=_sides_update(),
    )
    second = _reasoning_payload(
        payload_id="rs_second",
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
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
        machine=[{"op": "add", "path": "/meta", "value": {"step": 1}}],
        sides=_sides_update(),
    )
    second = _reasoning_payload(
        payload_id="rs_duplicate",
        previous_reasoning_id=first.id,
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(first), _sealed_reasoning(second)]),
        keyring=_keyring(),
    )

    assert result.machine == {"active": ["reviewer"], "meta": {"step": 1}}
    assert result.last_side is None


async def test_ingest_response_request_accepts_hidden_non_main_call_with_public_pair() -> None:
    assistant = Message(
        role="assistant",
        content="review hidden",
        tool_calls=[ToolCall(id="up_reviewer_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(
            patches={
                "reviewer": [
                    {"op": "add", "path": "/0", "value": assistant.to_primitive()},
                ]
            }
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
    assert result.last_side == "reviewer"


async def test_ingest_response_request_accepts_hidden_call_satisfied_by_hidden_output_only() -> None:
    assistant = Message(
        role="assistant",
        content="review hidden",
        tool_calls=[ToolCall(id="up_reviewer_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    hidden_output = {"role": "tool", "tool_call_id": "up_reviewer_0", "content": "hidden result"}
    payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
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
    assert result.last_side is None


async def test_ingest_response_request_accepts_main_hidden_call_with_public_pair() -> None:
    assistant = Message(
        role="assistant",
        content="main hidden",
        tool_calls=[ToolCall(id="up_main_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": ["main"]}],
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

    assert result.sides[MAIN_SIDE] == [
        assistant,
        Message(role="tool", tool_call_id="up_main_0", content="main result"),
    ]
    assert result.last_side == MAIN_SIDE


async def test_ingest_response_request_accepts_main_patched_open_cluster_with_empty_main_updates() -> None:
    assistant = Message(
        role="assistant",
        content="main hidden",
        tool_calls=[ToolCall(id="up_main_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": ["main"]}],
        sides=_sides_update(
            patches={
                MAIN_SIDE: [
                    {"op": "add", "path": "/0", "value": assistant.to_primitive()},
                ]
            }
        ),
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

    assert result.sides[MAIN_SIDE] == [
        assistant,
        Message(role="tool", tool_call_id="up_main_0", content="main result"),
    ]
    assert result.last_side == MAIN_SIDE


async def test_ingest_response_request_accepts_main_patched_closed_cluster_with_fabricated_pair() -> None:
    assistant = Message(
        role="assistant",
        content="main hidden",
        tool_calls=[ToolCall(id="up_main_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": ["main"]}],
        sides=_sides_update(
            patches={
                MAIN_SIDE: [
                    {"op": "add", "path": "/0", "value": assistant.to_primitive()},
                    {"op": "add", "path": "/1", "value": {"role": "tool", "tool_call_id": "up_main_0", "content": "hidden result"}},
                ]
            }
        ),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(payload),
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

    assert result.sides[MAIN_SIDE] == [
        Message(
            role="assistant",
            content="main hidden",
            tool_calls=[
                ToolCall(id="up_main_0", name="read_file", arguments='{"path":"README.md"}'),
                ToolCall(id="fab_0", name="read_file", arguments='{"path":"README.md"}'),
            ],
        ),
        Message(role="tool", tool_call_id="up_main_0", content="hidden result"),
        Message(role="tool", tool_call_id="fab_0", content="main result"),
    ]
    assert result.last_side == MAIN_SIDE


async def test_ingest_response_request_accepts_fabricated_main_pair_after_closed_cluster_and_user() -> None:
    assistant = Message(
        role="assistant",
        content="main hidden",
        tool_calls=[ToolCall(id="up_main_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": ["main"]}],
        sides=_sides_update(
            patches={
                MAIN_SIDE: [
                    {"op": "add", "path": "/0", "value": assistant.to_primitive()},
                    {"op": "add", "path": "/1", "value": {"role": "tool", "tool_call_id": "up_main_0", "content": "hidden result"}},
                    {"op": "add", "path": "/2", "value": Message(role="user", content="later user").to_primitive()},
                ]
            }
        ),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[
                _sealed_reasoning(payload),
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

    assert result.sides[MAIN_SIDE] == [
        Message(
            role="assistant",
            content="main hidden",
            tool_calls=[
                ToolCall(id="up_main_0", name="read_file", arguments='{"path":"README.md"}'),
                ToolCall(id="fab_0", name="read_file", arguments='{"path":"README.md"}'),
            ],
        ),
        Message(role="tool", tool_call_id="up_main_0", content="hidden result"),
        Message(role="tool", tool_call_id="fab_0", content="main result"),
        Message(role="user", content="later user"),
    ]
    assert result.last_side == MAIN_SIDE


async def test_ingest_response_request_accepts_fabricated_main_pair_after_empty_reasoning_step() -> None:
    initial = Message(role="assistant", content="main hidden")
    first = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": ["main"]}],
        sides=_sides_update(main=[initial]),
    )
    second = _reasoning_payload(
        machine=[{"op": "add", "path": "/meta", "value": {"step": 2}}],
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

    assert result.sides[MAIN_SIDE] == [
        Message(
            role="assistant",
            content="main hidden",
            tool_calls=[ToolCall(id="fab_0", name="read_file", arguments='{"path":"README.md"}')],
        ),
        Message(role="tool", tool_call_id="fab_0", content="main result"),
    ]
    assert result.last_side == MAIN_SIDE


async def test_ingest_response_request_rejects_reasoning_after_hidden_main_open_call() -> None:
    payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": ["main"]}],
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
            machine=[{"op": "add", "path": "/meta", "value": {"step": 2}}],
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
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(
            patches={
                "reviewer": [
                    {"op": "add", "path": "/0", "value": assistant.to_primitive()},
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
    assert exc_info.value.private.reason == "reasoning_tool_call_missing_function_call_item"


async def test_ingest_response_request_rejects_function_call_output_without_pending_function_call() -> None:
    assistant = Message(
        role="assistant",
        content="review hidden",
        tool_calls=[ToolCall(id="up_reviewer_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(
            patches={
                "reviewer": [
                    {"op": "add", "path": "/0", "value": assistant.to_primitive()},
                ]
            }
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
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(
            patches={
                "reviewer": [
                    {"op": "add", "path": "/0", "value": assistant.to_primitive()},
                ]
            }
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
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(
            patches={
                "reviewer": [
                    {"op": "add", "path": "/0", "value": assistant.to_primitive()},
                ]
            }
        ),
    )
    second_payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/meta", "value": {"step": 2}}],
        sides=_sides_update(
            patches={
                "reviewer": [
                    {"op": "add", "path": "/1", "value": {"role": "assistant", "content": "interleaving"}},
                ]
            },
            current=Sides(messages={"reviewer": [assistant]}),
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


async def test_ingest_response_request_rejects_machine_only_reasoning_while_waiting_for_output() -> None:
    assistant = Message(
        role="assistant",
        content="review hidden",
        tool_calls=[ToolCall(id="up_reviewer_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    first_payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=_sides_update(
            patches={
                "reviewer": [
                    {"op": "add", "path": "/0", "value": assistant.to_primitive()},
                ]
            }
        ),
    )
    second_payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/meta", "value": {"step": 2}}],
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


async def test_ingest_response_request_rejects_main_message_patch_until_supported() -> None:
    payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": []}],
        sides=_sides_update(
            main=[
                MessagePatch(
                    content_hash="abcd",
                    reasoning_content="hidden assistant state",
                )
            ]
        ),
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

    assert result.sides[MAIN_SIDE] == [Message(role="user", content="hello")]
    assert result.last_side == MAIN_SIDE


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
    assert result.last_side is None


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

    assert result.sides[MAIN_SIDE] == [
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


async def test_ingest_response_request_fails_closed_on_invalid_machine_patch() -> None:
    payload = _reasoning_payload(
        machine=[{"op": "replace", "path": "/missing", "value": 1}],
        sides=_sides_update(),
    )

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(payload)]),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "reasoning_machine_patch_invalid"


async def test_ingest_response_request_fails_closed_on_invalid_side_patch() -> None:
    payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
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
