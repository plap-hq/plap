from __future__ import annotations

from itertools import count

import pytest

import plap.responses.ingest.ingest as ingest_module
from plap.errors import PlapError
from plap.keyring import SealingKeyring
from plap.responses.contracts import (
    InputTextContent,
    RequestCompactionItem,
    RequestFunctionCallItem,
    RequestFunctionCallOutputItem,
    RequestMessageItem,
    RequestReasoningItem,
    ResponseCreateRequest,
    SummaryTextContent,
)
from plap.responses.ingest.ingest import (
    _decode_queue,
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
    ingest_response_request,
)
from plap.responses.ingest.models import (
    CallID,
    CompactionPayload,
    Message,
    MessagePatch,
    ReasoningPayload,
    Side,
    Sides,
    SidesUpdate,
    ToolCall,
)
from plap.responses.ingest.sealing import (
    content_hash,
    content_hash_prefix,
    open_call_id,
    seal_call_id,
    seal_compaction_payload,
    seal_reasoning_payload,
)


def _compaction(label: str) -> RequestCompactionItem:
    return RequestCompactionItem(encrypted_content=label, type="compaction")


def _message(label: str) -> RequestMessageItem:
    return RequestMessageItem(content=label, role="user", type="message")


def _keyring() -> SealingKeyring:
    return SealingKeyring(roots=(b"i" * 32,))


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
            content_hash_prefix=bytes.fromhex("0102030405060708"),
            tool_call_index=0,
            upstream_tool_call_id=upstream_tool_call_id,
        ),
        keyring=_keyring(),
    )


def _sealed_call_id_for_message(side: str, upstream_tool_call_id: str, message: Message, *, tool_call_index: int = 0) -> str:
    return seal_call_id(
        CallID(
            side=side,
            content_hash_prefix=content_hash_prefix(content_hash(message)),
            tool_call_index=tool_call_index,
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
        sides=SidesUpdate(),
    )

    decoded = _decode_queue([_sealed_reasoning(payload)], keyring=_keyring())

    assert decoded == [_DecodedReasoning(payload=payload)]


def test_decode_queue_rejects_reasoning_item_id_mismatch() -> None:
    payload = _reasoning_payload(machine=[{"op": "add", "path": "/active", "value": []}], sides=SidesUpdate())
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
                    InputTextContent(text="hello", type="input_text"),
                    InputTextContent(text="world", type="input_text"),
                ],
                role="assistant",
                type="message",
            )
        ],
        keyring=_keyring(),
    )

    assert decoded == [_DecodedMessage(message=Message(role="assistant", content="hello world"))]


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
                content_hash_prefix=bytes.fromhex("0102030405060708"),
                tool_call_index=0,
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
                content_hash_prefix=bytes.fromhex("0102030405060708"),
                tool_call_index=0,
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


def test_call_id_roundtrips_zero_based_main_side_with_fixed_width_index() -> None:
    value = CallID(
        side="main",
        content_hash_prefix=bytes.fromhex("0102030405060708"),
        tool_call_index=65535,
        upstream_tool_call_id="up_main_65535",
    )

    token = seal_call_id(value, keyring=_keyring())

    assert open_call_id(token, keyring=_keyring()) == value


def test_seal_call_id_rejects_tool_call_index_above_u16() -> None:
    with pytest.raises(PlapError) as excinfo:
        seal_call_id(
            CallID(
                side="main",
                content_hash_prefix=bytes.fromhex("0102030405060708"),
                tool_call_index=65536,
                upstream_tool_call_id="up_main_65536",
            ),
            keyring=_keyring(),
        )

    assert excinfo.value.private.reason == "tool_call_index_too_large"


def test_sides_update_main_accepts_single_patch_followed_by_trailing_tool_messages() -> None:
    update = SidesUpdate(
        main=[
            Message(role="assistant", content="prefix"),
            MessagePatch(content_hash="abcd", tool_calls=[ToolCall(id="call_1", name="read_file", arguments="{}")]),
            Message(role="tool", tool_call_id="call_1", content="hidden output"),
        ]
    )

    assert len(update.main) == 3


def test_sides_update_main_rejects_second_patch() -> None:
    with pytest.raises(ValueError, match="at most one message patch"):
        SidesUpdate(
            main=[
                MessagePatch(content_hash="abcd", reasoning_content="first"),
                MessagePatch(content_hash="efgh", reasoning_content="second"),
            ]
        )


def test_sides_update_main_rejects_non_tool_message_after_patch() -> None:
    with pytest.raises(ValueError, match="message patch must be the last non-tool main update"):
        SidesUpdate(
            main=[
                MessagePatch(content_hash="abcd", reasoning_content="hidden"),
                Message(role="assistant", content="later assistant"),
            ]
        )


def test_sides_update_main_rejects_tool_message_without_tool_call_id_after_patch() -> None:
    with pytest.raises(ValueError, match="must be a tool message with tool_call_id after the anchor"):
        SidesUpdate(
            main=[
                MessagePatch(content_hash="abcd", tool_calls=[ToolCall(id="call_1", name="read_file", arguments="{}")]),
                Message(role="tool", content="missing id"),
            ]
        )


def test_sides_update_main_accepts_closed_prefix_before_patch_anchor() -> None:
    update = SidesUpdate(
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
    update = SidesUpdate(
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


def test_sides_update_main_rejects_unclosed_prefix_before_patch_anchor() -> None:
    with pytest.raises(ValueError, match="must satisfy all prefix tool calls before the anchor"):
        SidesUpdate(
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
        SidesUpdate(
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
        SidesUpdate(
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
        SidesUpdate(
            main=[
                MessagePatch(content_hash="abcd", tool_calls=[ToolCall(id="call_1", name="read_file", arguments="{}")]),
                Message(role="tool", tool_call_id="wrong", content="hidden output"),
            ]
        )


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
            main=[Message(role="assistant", content="main snapshot")],
            others={Side.REVIEWER: [Message(role="assistant", content="review snapshot")]},
        ),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[_sealed_compaction(payload)]),
        keyring=_keyring(),
    )

    assert result.machine == payload.machine
    assert result.sides == payload.sides
    assert result.last_side is None


async def test_ingest_response_request_accepts_reasoning_chain_anchored_to_compaction() -> None:
    compaction = _compaction_payload(
        payload_id="cmp_root",
        machine={"active": []},
        sides=Sides(main=[Message(role="assistant", content="snapshot")]),
    )
    first = _reasoning_payload(
        payload_id="rs_first",
        previous_compaction_id=compaction.id,
        machine=[{"op": "add", "path": "/meta", "value": {"step": 1}}],
        sides=SidesUpdate(main=[Message(role="assistant", content="first")]),
    )
    second = _reasoning_payload(
        payload_id="rs_second",
        previous_reasoning_id=first.id,
        previous_compaction_id=compaction.id,
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=SidesUpdate(main=[Message(role="assistant", content="second")]),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(
            model="plap/test",
            input=[_sealed_compaction(compaction), _sealed_reasoning(first), _sealed_reasoning(second)],
        ),
        keyring=_keyring(),
    )

    assert result.machine == {"active": ["reviewer"], "meta": {"step": 1}}
    assert result.sides.main == [
        Message(role="assistant", content="snapshot"),
        Message(role="assistant", content="first"),
        Message(role="assistant", content="second"),
    ]
    assert result.last_side is None


async def test_ingest_response_request_applies_reasoning_machine_patch() -> None:
    payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=SidesUpdate(),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(payload)]),
        keyring=_keyring(),
    )

    assert result.machine == {"active": ["reviewer"]}
    assert result.sides == Sides()
    assert result.last_side is None


async def test_ingest_response_request_applies_reasoning_non_main_side_patch() -> None:
    payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=SidesUpdate(
            others={
                Side.REVIEWER: [
                    {"op": "add", "path": "/0", "value": {"role": "assistant", "content": "review hidden"}},
                ]
            }
        ),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(payload)]),
        keyring=_keyring(),
    )

    assert result.sides.others[Side.REVIEWER] == [Message(role="assistant", content="review hidden")]
    assert result.last_side is None


async def test_ingest_response_request_appends_main_messages_from_reasoning() -> None:
    payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": []}],
        sides=SidesUpdate(main=[Message(role="assistant", content="main hidden")]),
    )

    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[_sealed_reasoning(payload)]),
        keyring=_keyring(),
    )

    assert result.sides.main == [Message(role="assistant", content="main hidden")]
    assert result.last_side is None


async def test_ingest_response_request_applies_multiple_reasoning_items_in_order() -> None:
    first_payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=SidesUpdate(main=[Message(role="assistant", content="first")]),
    )
    second_payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/meta", "value": {"step": 2}}],
        sides=SidesUpdate(main=[Message(role="assistant", content="second")]),
        previous_reasoning_id=first_payload.id,
    )
    first = _sealed_reasoning(first_payload)
    second = _sealed_reasoning(second_payload)

    result = await ingest_response_request(
        ResponseCreateRequest(model="plap/test", input=[first, second]),
        keyring=_keyring(),
    )

    assert result.machine == {"active": ["reviewer"], "meta": {"step": 2}}
    assert result.sides.main == [
        Message(role="assistant", content="first"),
        Message(role="assistant", content="second"),
    ]
    assert result.last_side is None


async def test_ingest_response_request_rejects_first_reasoning_with_non_none_previous_reasoning_id() -> None:
    payload = _reasoning_payload(
        payload_id="rs_first",
        previous_reasoning_id="rs_missing",
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=SidesUpdate(),
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
        sides=SidesUpdate(),
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
        sides=SidesUpdate(),
    )

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(model="plap/test", input=[_sealed_compaction(compaction), _sealed_reasoning(payload)]),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "reasoning_previous_compaction_id_mismatch"


async def test_ingest_response_request_rejects_second_reasoning_with_none_previous_reasoning_id() -> None:
    first = _reasoning_payload(
        payload_id="rs_first",
        machine=[{"op": "add", "path": "/meta", "value": {"step": 1}}],
        sides=SidesUpdate(),
    )
    second = _reasoning_payload(
        payload_id="rs_second",
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=SidesUpdate(),
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
        sides=SidesUpdate(),
    )
    second = _reasoning_payload(
        payload_id="rs_duplicate",
        previous_reasoning_id=first.id,
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=SidesUpdate(),
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
        sides=SidesUpdate(
            others={
                Side.REVIEWER: [
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

    assert result.sides.others[Side.REVIEWER] == [
        assistant,
        Message(role="tool", tool_call_id="up_reviewer_0", content="review result"),
    ]
    assert result.last_side == Side.REVIEWER


async def test_ingest_response_request_accepts_hidden_call_satisfied_by_hidden_output_only() -> None:
    assistant = Message(
        role="assistant",
        content="review hidden",
        tool_calls=[ToolCall(id="up_reviewer_0", name="read_file", arguments='{"path":"README.md"}')],
    )
    hidden_output = {"role": "tool", "tool_call_id": "up_reviewer_0", "content": "hidden result"}
    payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": ["reviewer"]}],
        sides=SidesUpdate(
            others={
                Side.REVIEWER: [
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

    assert result.sides.others[Side.REVIEWER] == [
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
        sides=SidesUpdate(main=[assistant]),
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

    assert result.sides.main == [
        assistant,
        Message(role="tool", tool_call_id="up_main_0", content="main result"),
    ]
    assert result.last_side == Side.MAIN


async def test_ingest_response_request_rejects_reasoning_after_hidden_main_open_call() -> None:
    payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/active", "value": ["main"]}],
        sides=SidesUpdate(
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
            sides=SidesUpdate(),
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
        sides=SidesUpdate(
            others={
                Side.REVIEWER: [
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
        sides=SidesUpdate(
            others={
                Side.REVIEWER: [
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
        sides=SidesUpdate(
            others={
                Side.REVIEWER: [
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
        sides=SidesUpdate(
            others={
                Side.REVIEWER: [
                    {"op": "add", "path": "/0", "value": assistant.to_primitive()},
                ]
            }
        ),
    )
    second_payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/meta", "value": {"step": 2}}],
        sides=SidesUpdate(
            others={
                Side.REVIEWER: [
                    {"op": "add", "path": "/1", "value": {"role": "assistant", "content": "interleaving"}},
                ]
            }
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
        sides=SidesUpdate(
            others={
                Side.REVIEWER: [
                    {"op": "add", "path": "/0", "value": assistant.to_primitive()},
                ]
            }
        ),
    )
    second_payload = _reasoning_payload(
        machine=[{"op": "add", "path": "/meta", "value": {"step": 2}}],
        sides=SidesUpdate(),
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
        sides=SidesUpdate(
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

    assert result.sides.main == [Message(role="user", content="hello")]
    assert result.last_side == Side.MAIN


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

    assert result.sides.others[Side.REVIEWER] == []
    assert result.last_side is None


async def test_ingest_response_request_strict_phase1_rejects_naked_main_synthetic_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingest_module, "ENABLE_PHASE2", False)
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
    assert exc_info.value.private.reason == "sealed_function_call_content_hash_target_missing"


async def test_ingest_response_request_strict_phase1_rejects_closed_anchor_then_synthetic_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingest_module, "ENABLE_PHASE2", False)
    synthetic = Message(role="assistant", content="")
    call_id = _sealed_call_id_for_message("main", "syn_0", synthetic)

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
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

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "sealed_function_call_content_hash_target_missing"


async def test_ingest_response_request_strict_phase1_rejects_naked_non_main_sealed_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingest_module, "ENABLE_PHASE2", False)
    item = RequestFunctionCallItem(
        arguments='{"path":"README.md"}',
        call_id=_sealed_call_id("reviewer", "up_reviewer_0"),
        name="read_file",
        type="function_call",
    )

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            ResponseCreateRequest(model="plap/test", input=[item]),
            keyring=_keyring(),
        )

    assert exc_info.value.private is not None
    assert exc_info.value.private.reason == "sealed_function_call_content_hash_target_missing"


async def test_ingest_response_request_fails_closed_on_invalid_machine_patch() -> None:
    payload = _reasoning_payload(
        machine=[{"op": "replace", "path": "/missing", "value": 1}],
        sides=SidesUpdate(),
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
        sides=SidesUpdate(
            others={
                Side.REVIEWER: [
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
