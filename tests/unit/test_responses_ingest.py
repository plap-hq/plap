from __future__ import annotations

import base64
import importlib

import msgspec
import pytest
import zstandard as zstd
from nacl.secret import Aead

from plap.errors import PlapError
from plap.keyring import SealingKeyring, purpose_label
from plap.responses.contracts import (
    RequestCompactionItem,
    RequestFunctionCallItem,
    RequestFunctionCallOutputItem,
    RequestMessageItem,
    RequestReasoningItem,
    ResponseCreateRequest,
    SummaryTextContent,
)
from plap.responses.ingest import (
    CALL_ID_CONTENT_HASH_PREFIX_BYTES,
    ChatMessageSpan,
    CompactionPayload,
    IngestedQueues,
    MutableQueues,
    ReasoningPayload,
    SealedCallID,
    content_hash,
    content_hash_prefix,
    open_call_id,
    open_compaction_payload,
    seal_call_id,
    seal_compaction_payload,
    seal_reasoning_payload,
)
from plap.responses.ingest import (
    ingest_response_request as _ingest_response_request,
)
from plap.responses.ingest.sealing import (
    COMPACTION_PURPOSE,
    PAYLOAD_FORMAT_VERSION,
)
from plap.responses.models import ReasoningMessagePatch, StateMessage


async def ingest_response_request(
    request: ResponseCreateRequest,
    *,
    keyring: SealingKeyring,
) -> IngestedQueues:
    return await _ingest_response_request(
        request,
        keyring=keyring,
    )


def _assert_plap_error(
    exc: PlapError,
    *,
    code: str | None = None,
    param: str | None = None,
    private_reason: str | None = None,
) -> None:
    if code is not None:
        assert exc.public is not None
        assert exc.public.code == code
    if param is not None:
        assert exc.public is not None
        assert exc.public.param == param
    if private_reason is not None:
        assert exc.private.reason == private_reason


def _main_transcript(
    queues: IngestedQueues,
    *,
    token_budget: int = 0,
) -> tuple[ChatMessageSpan, ...]:
    return MutableQueues.from_ingested(queues).main_transcript(token_budget=token_budget)


async def test_ingestion_preserves_last_compaction_spans() -> None:
    result = await ingest_response_request(
        _request(
            input=[
                _message("user", "before"),
                _compaction_item("discarded", 10),
                _message("user", "between"),
                _compaction_item("kept", 2),
                _message("user", "after"),
            ]
        ),
        keyring=_keyring(),
    )

    assert [(row.start, row.end, row.message.content) for row in result.main_context] == [
        (0, 0, "kept source"),
        (1, 1, "kept summarized source"),
        (2, 2, "after"),
    ]
    assert [row.message.content for row in _main_transcript(result)] == [
        "kept source",
        "kept summarized source",
        "after",
    ]
    assert result.cursors == {"m": 3}
    assert result.continuation_side == "main"


def test_mutable_queues_append_helpers() -> None:
    queues = MutableQueues(
        main_context=[],
        main_context_temp=[],
        reviewer=[],
        arbitrator=[],
        cursors={"m": 0},
        continuation_side="main",
        in_temp_debate=False,
    )

    stable = queues.append_main_stable(StateMessage(role="user", content="stable"), content_hash="stable-hash")
    temp = queues.append_main_temp(StateMessage(role="assistant", content="temp"))
    reviewer = queues.append_side("reviewer", StateMessage(role="assistant", content="review"), content_hash="review-hash")
    arbitrator = queues.append_side("arbitrator", StateMessage(role="assistant", content="decide"))

    assert (stable.start, stable.end, stable.content_hash) == (0, 0, "stable-hash")
    assert (temp.start, temp.end, temp.message.content) == (1, 1, "temp")
    assert reviewer.content_hash == "review-hash"
    assert reviewer.message.content == "review"
    assert arbitrator.message.content == "decide"
    assert queues.cursors == {"m": 2}

    with pytest.raises(ValueError, match="append_side does not accept main"):
        queues.append_side("main", StateMessage(role="assistant", content="bad"))


def test_request_accepts_verbatim_replayed_function_output_created_by() -> None:
    request = ResponseCreateRequest(
        model="test/model",
        input=[
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "tool result",
                "created_by": "server",
            }
        ],
    )

    item = request.input[0]
    assert isinstance(item, RequestFunctionCallOutputItem)
    assert item.created_by == "server"


def test_request_accepts_verbatim_replayed_compaction_created_by() -> None:
    value = _compaction_item("kept", 2).model_dump(mode="python")
    value["created_by"] = "assistant"

    request = ResponseCreateRequest(
        model="test/model",
        input=[value],
    )

    item = request.input[0]
    assert isinstance(item, RequestCompactionItem)
    assert item.created_by == "assistant"


async def test_ingestion_compaction_only_continues_main() -> None:
    result = await ingest_response_request(
        _request(input=[_compaction_item("only", 2)]),
        keyring=_keyring(),
    )

    assert [(row.start, row.end) for row in result.main_context] == [
        (0, 0),
        (1, 1),
    ]
    assert [(row.start, row.end) for row in _main_transcript(result)] == [
        (0, 0),
        (1, 1),
    ]
    assert result.continuation_side == "main"


async def test_ingestion_assigns_m_ordinals_without_compaction() -> None:
    result = await ingest_response_request(
        _request(input=[_message("user", "u0"), _message("assistant", "a0")]),
        keyring=_keyring(),
    )

    assert [(row.start, row.end, row.message.content) for row in result.main_context] == [
        (0, 0, "u0"),
        (1, 1, "a0"),
    ]
    assert result.cursors == {"m": 2}
    assert result.main_context == _main_transcript(result)
    assert result.continuation_side == "main"


async def test_ingestion_routes_reasoning_by_sealed_side_with_hashes() -> None:
    result = await ingest_response_request(
        _request(input=[_reasoning_item("reviewer", False, [{"role": "assistant", "content": "review"}])]),
        keyring=_keyring(),
    )

    assert result.main_context == ()
    assert _main_transcript(result) == ()
    assert [(row.message.content, row.content_hash) for row in result.reviewer] == [
        ("review", _message_hash({"role": "assistant", "content": "review"}))
    ]
    assert result.arbitrator == ()
    assert result.continuation_side == "reviewer"


async def test_ingestion_arbitrator_reasoning_sets_continuation_side() -> None:
    result = await ingest_response_request(
        _request(input=[_reasoning_item("arbitrator", False, [{"role": "assistant", "content": "decide"}])]),
        keyring=_keyring(),
    )

    assert result.arbitrator[0].message.content == "decide"
    assert result.continuation_side == "arbitrator"


async def test_ingestion_reasoning_continuation_side_overrides_next_side_only_when_last() -> None:
    result = await ingest_response_request(
        _request(
            input=[
                _reasoning_item(
                    "main",
                    True,
                    [{"role": "assistant", "content": "candidate mutation"}],
                    continuation_side="reviewer",
                )
            ]
        ),
        keyring=_keyring(),
    )

    assert [row.message.content for row in result.main_context_temp] == ["candidate mutation"]
    assert result.reviewer == ()
    assert result.continuation_side == "reviewer"
    assert result.in_temp_debate is True

    result = await ingest_response_request(
        _request(
            input=[
                _reasoning_item(
                    "main",
                    False,
                    [{"role": "assistant", "content": "main reasoning"}],
                    continuation_side="reviewer",
                ),
                _message("user", "follow-up"),
            ]
        ),
        keyring=_keyring(),
    )

    assert [row.message.content for row in result.main_context] == ["main reasoning", "follow-up"]
    assert result.continuation_side == "main"


async def test_ingestion_temp_false_prunes_entire_temp_debate() -> None:
    temp_message = {"role": "assistant", "content": "temp reviewer"}
    call_id = _call_id(
        side="reviewer",
        content_hash_value=_message_hash(temp_message),
        upstream_tool_call_id="up_temp_0",
    )

    result = await ingest_response_request(
        _request(
            input=[
                _reasoning_item("reviewer", True, [temp_message]),
                _function_call(call_id),
                _function_output(call_id, "temp output"),
                _reasoning_item(
                    "main",
                    False,
                    [{"role": "assistant", "content": "final debate result"}],
                ),
            ]
        ),
        keyring=_keyring(),
    )

    assert [row.message.content for row in result.main_context] == ["final debate result"]
    assert result.main_context_temp == ()
    assert result.main_context == _main_transcript(result)
    assert result.reviewer == ()
    assert result.continuation_side == "main"
    assert result.in_temp_debate is False


async def test_ingestion_message_after_temp_prunes_entire_temp_debate() -> None:
    temp_message = {
        "role": "assistant",
        "content": "temp reviewer",
        "tool_calls": [_tool_call("up_temp_0")],
    }
    call_id = _call_id(
        side="reviewer",
        content_hash_value=_message_hash(temp_message),
        upstream_tool_call_id="up_temp_0",
    )

    result = await ingest_response_request(
        _request(
            input=[
                _reasoning_item("reviewer", True, [temp_message]),
                _function_call(call_id),
                _function_output(call_id, "temp output"),
                _message("user", "new mainline request"),
            ]
        ),
        keyring=_keyring(),
    )

    assert [row.message.content for row in result.main_context] == ["new mainline request"]
    assert result.main_context_temp == ()
    assert result.main_context == _main_transcript(result)
    assert result.reviewer == ()
    assert result.continuation_side == "main"
    assert result.in_temp_debate is False


async def test_ingestion_message_after_temp_prunes_forward_reasoning_refs_too() -> None:
    target = {"role": "assistant", "content": "stable public answer"}

    result = await ingest_response_request(
        _request(
            input=[
                _reasoning_item(
                    "reviewer",
                    True,
                    [
                        {"role": "assistant", "content": "temp reviewer"},
                        {
                            "content_hash": _message_hash(target),
                            "reasoning_content": "should not leak",
                        },
                    ],
                ),
                _message("assistant", "stable public answer"),
            ]
        ),
        keyring=_keyring(),
    )

    assert [row.message.to_primitive() for row in result.main_context] == [target]
    assert result.main_context_temp == ()
    assert result.reviewer == ()
    assert result.continuation_side == "main"
    assert result.in_temp_debate is False


async def test_ingestion_fabricated_call_after_temp_prunes_temp_debate() -> None:
    result = await ingest_response_request(
        _request(
            input=[
                _message("assistant", "stable assistant"),
                _reasoning_item(
                    "reviewer",
                    True,
                    [{"role": "assistant", "content": "temp reviewer"}],
                ),
                RequestFunctionCallItem(
                    arguments='{"path":"README.md"}',
                    call_id="client_call_0",
                    name="read_file",
                    type="function_call",
                ),
                _function_output("client_call_0", "client output"),
            ]
        ),
        keyring=_keyring(),
    )

    assert [row.message.to_primitive() for row in result.main_context] == [
        {
            "role": "assistant",
            "content": "stable assistant",
            "tool_calls": [
                {
                    "id": "client_call_0",
                    "name": "read_file",
                    "arguments": '{"path":"README.md"}',
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "client_call_0",
            "content": "client output",
        },
    ]
    assert result.main_context_temp == ()
    assert result.main_context == _main_transcript(result)
    assert result.reviewer == ()
    assert result.continuation_side == "main"
    assert result.in_temp_debate is False


async def test_ingestion_exposes_active_temp_debate_state() -> None:
    result = await ingest_response_request(
        _request(
            input=[
                _reasoning_item(
                    "main",
                    True,
                    [{"role": "assistant", "content": "temp debate tail"}],
                )
            ]
        ),
        keyring=_keyring(),
    )

    assert result.main_context == ()
    assert [row.message.content for row in result.main_context_temp] == ["temp debate tail"]
    assert _main_transcript(result) == ()
    assert result.continuation_side == "main"
    assert result.in_temp_debate is True


async def test_ingestion_routes_sealed_reviewer_call_and_output() -> None:
    assistant = {
        "role": "assistant",
        "content": "need file",
        "tool_calls": [_tool_call("up_reviewer_0")],
    }
    call_id = _call_id(
        side="reviewer",
        content_hash_value=_message_hash(assistant),
        upstream_tool_call_id="up_reviewer_0",
    )

    result = await ingest_response_request(
        _request(
            input=[
                _reasoning_item("reviewer", False, [assistant]),
                _function_call(call_id),
                _function_output(call_id, "review file"),
            ]
        ),
        keyring=_keyring(),
    )

    assert result.main_context == ()
    assert _main_transcript(result) == ()
    assert [row.message.to_primitive() for row in result.reviewer] == [
        assistant,
        {"role": "tool", "tool_call_id": "up_reviewer_0", "content": "review file"},
    ]
    assert result.continuation_side == "reviewer"


async def test_ingestion_routes_sealed_main_call_and_tool_output_to_m_rows() -> None:
    assistant = {"role": "assistant", "content": "public assistant"}
    call_id = _call_id(
        side="main",
        content_hash_value=_message_hash(assistant),
        upstream_tool_call_id="up_main_0",
    )

    result = await ingest_response_request(
        _request(
            input=[
                _message("assistant", "public assistant"),
                _function_call(call_id),
                _function_output(call_id, "main output"),
            ]
        ),
        keyring=_keyring(),
    )

    assert [(row.start, row.end, row.message.to_primitive()) for row in result.main_context] == [
        (
            0,
            0,
            {
                "role": "assistant",
                "content": "public assistant",
                "tool_calls": [_tool_call("up_main_0")],
            },
        ),
        (
            1,
            1,
            {"role": "tool", "tool_call_id": "up_main_0", "content": "main output"},
        ),
    ]
    assert result.cursors == {"m": 2}
    assert result.main_context == _main_transcript(result)
    assert result.continuation_side == "main"


async def test_ingestion_fabricated_unsealed_pair_routes_to_main_only() -> None:
    result = await ingest_response_request(
        _request(
            input=[
                _message("assistant", "client fabricated assistant"),
                RequestFunctionCallItem(
                    arguments='{"path":"README.md"}',
                    call_id="client_call_0",
                    name="read_file",
                    type="function_call",
                ),
                _function_output("client_call_0", "client output"),
            ]
        ),
        keyring=_keyring(),
    )

    assert [(row.start, row.end, row.message.to_primitive()) for row in result.main_context] == [
        (
            0,
            0,
            {
                "role": "assistant",
                "content": "client fabricated assistant",
                "tool_calls": [{"id": "client_call_0", "name": "read_file", "arguments": '{"path":"README.md"}'}],
            },
        ),
        (
            1,
            1,
            {
                "role": "tool",
                "tool_call_id": "client_call_0",
                "content": "client output",
            },
        ),
    ]
    assert result.main_context == _main_transcript(result)
    assert result.reviewer == ()
    assert result.arbitrator == ()
    assert result.continuation_side == "main"


async def test_ingestion_rejects_unsealed_call_interleaving_before_output() -> None:
    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            _request(
                input=[
                    _message("assistant", "first assistant"),
                    RequestFunctionCallItem(
                        arguments='{"path":"README.md"}',
                        call_id="client_call_0",
                        name="read_file",
                        type="function_call",
                    ),
                    _message("assistant", "second assistant"),
                    _function_output("client_call_0", "client output"),
                ]
            ),
            keyring=_keyring(),
        )

    _assert_plap_error(exc_info.value, code="invalid_tool_replay", param="input", private_reason="pending_tool_outputs_block_message")


async def test_ingestion_rejects_duplicate_unsealed_pending_call_ids() -> None:
    first_call = RequestFunctionCallItem(
        arguments='{"path":"a"}',
        call_id="client_call_0",
        name="read_file",
        type="function_call",
    )
    second_call = RequestFunctionCallItem(
        arguments='{"path":"b"}',
        call_id="client_call_0",
        name="read_file",
        type="function_call",
    )

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            _request(input=[_message("assistant", "anchor"), first_call, second_call]),
            keyring=_keyring(),
        )

    _assert_plap_error(exc_info.value, code="invalid_tool_replay", param="input", private_reason="duplicate_pending_unsealed_function_call")


async def test_ingestion_rejects_sealed_call_interleaving_before_output() -> None:
    assistant = {"role": "assistant", "content": "need file"}
    call_id = _call_id(
        side="reviewer",
        content_hash_value=_message_hash(assistant),
        upstream_tool_call_id="up_reviewer_0",
    )

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            _request(
                input=[
                    _reasoning_item("reviewer", False, [assistant]),
                    _function_call(call_id),
                    _reasoning_item(
                        "reviewer",
                        False,
                        [{"role": "assistant", "content": "interleaving"}],
                    ),
                    _function_output(call_id, "review file"),
                ]
            ),
            keyring=_keyring(),
        )

    _assert_plap_error(exc_info.value, code="invalid_tool_replay", param="input", private_reason="pending_tool_outputs_block_message")


async def test_ingestion_allows_stripped_tool_call_association() -> None:
    stripped = {"role": "assistant", "content": "need file"}
    call_id = _call_id(
        side="reviewer",
        content_hash_value=_message_hash(stripped),
        upstream_tool_call_id="up_stripped_0",
    )

    result = await ingest_response_request(
        _request(
            input=[
                _reasoning_item("reviewer", False, [stripped]),
                _function_call(call_id),
                _function_output(call_id, "stripped output"),
            ]
        ),
        keyring=_keyring(),
    )

    assert [row.message.to_primitive() for row in result.reviewer] == [
        {
            "role": "assistant",
            "content": "need file",
            "tool_calls": [_tool_call("up_stripped_0")],
        },
        {"role": "tool", "tool_call_id": "up_stripped_0", "content": "stripped output"},
    ]


async def test_ingestion_requires_reasoning_tool_call_public_replay() -> None:
    assistant = {
        "role": "assistant",
        "content": "need file",
        "tool_calls": [_tool_call("up_reasoning_0")],
    }

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            _request(input=[_reasoning_item("reviewer", False, [assistant])]),
            keyring=_keyring(),
        )

    _assert_plap_error(
        exc_info.value, code="invalid_reasoning_replay", param="input", private_reason="reasoning_tool_call_missing_function_call_item"
    )


async def test_ingestion_accepts_reasoning_tool_call_satisfied_by_hidden_output() -> None:
    assistant = {
        "role": "assistant",
        "content": "need file",
        "tool_calls": [_tool_call("up_reasoning_0")],
    }
    hidden_output = {
        "role": "tool",
        "tool_call_id": "up_reasoning_0",
        "content": "intercepted by reviewer",
    }

    result = await ingest_response_request(
        _request(input=[_reasoning_item("reviewer", False, [assistant, hidden_output])]),
        keyring=_keyring(),
    )

    assert [row.message.to_primitive() for row in result.reviewer] == [assistant, hidden_output]
    assert result.continuation_side == "reviewer"


async def test_ingestion_accepts_reasoning_tool_call_with_public_pair() -> None:
    assistant = {
        "role": "assistant",
        "content": "need file",
        "tool_calls": [_tool_call("up_reasoning_0")],
    }
    call_id = _call_id(
        side="reviewer",
        content_hash_value=_message_hash(assistant),
        upstream_tool_call_id="up_reasoning_0",
    )

    result = await ingest_response_request(
        _request(
            input=[
                _reasoning_item("reviewer", False, [assistant]),
                _function_call(call_id),
                _function_output(call_id, "reasoning output"),
            ]
        ),
        keyring=_keyring(),
    )

    assert [row.message.to_primitive() for row in result.reviewer] == [
        assistant,
        {
            "role": "tool",
            "tool_call_id": "up_reasoning_0",
            "content": "reasoning output",
        },
    ]


async def test_ingestion_requires_output_for_replayed_reasoning_tool_call() -> None:
    assistant = {
        "role": "assistant",
        "content": "need file",
        "tool_calls": [_tool_call("up_reasoning_0")],
    }
    call_id = _call_id(
        side="reviewer",
        content_hash_value=_message_hash(assistant),
        upstream_tool_call_id="up_reasoning_0",
    )

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            _request(
                input=[
                    _reasoning_item("reviewer", False, [assistant]),
                    _function_call(call_id),
                ]
            ),
            keyring=_keyring(),
        )

    _assert_plap_error(
        exc_info.value, code="invalid_tool_replay", param="input", private_reason="function_call_missing_function_call_output"
    )


async def test_ingestion_accepts_reasoning_forward_refs() -> None:
    target = {"role": "assistant", "content": "target"}

    result = await ingest_response_request(
        _request(
            input=[
                _reasoning_item(
                    "reviewer",
                    False,
                    [
                        {
                            "content_hash": _message_hash(target),
                            "reasoning_content": "hidden",
                        },
                        target,
                    ],
                )
            ]
        ),
        keyring=_keyring(),
    )

    assert [row.message.to_primitive() for row in result.reviewer] == [
        {
            "role": "assistant",
            "content": "target",
            "reasoning_content": "hidden",
        }
    ]


async def test_ingestion_missing_reasoning_forward_ref_fails_closed() -> None:
    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            _request(
                input=[
                    _reasoning_item(
                        "reviewer",
                        False,
                        [
                            {
                                "content_hash": _message_hash({"role": "assistant", "content": "missing"}),
                                "reasoning_content": "hidden",
                            }
                        ],
                    )
                ]
            ),
            keyring=_keyring(),
        )

    _assert_plap_error(
        exc_info.value, code="invalid_reasoning_replay", param="input", private_reason="reasoning_content_hash_target_missing"
    )


async def test_ingestion_main_reasoning_refs_merge_without_new_ordinal() -> None:
    anchor = {"role": "assistant", "content": "anchor"}

    result = await ingest_response_request(
        _request(
            input=[
                _message("assistant", "anchor"),
                _reasoning_item(
                    "main",
                    False,
                    [
                        {
                            "content_hash": _message_hash(anchor),
                            "reasoning_content": "anchor reasoning",
                        },
                        {
                            "role": "assistant",
                            "content": "new reasoning message",
                            "reasoning_content": "new hidden",
                        },
                    ],
                ),
            ]
        ),
        keyring=_keyring(),
    )

    assert [(row.start, row.end, row.message.to_primitive()) for row in result.main_context] == [
        (
            0,
            0,
            {
                "role": "assistant",
                "content": "anchor",
                "reasoning_content": "anchor reasoning",
            },
        ),
        (
            1,
            1,
            {
                "role": "assistant",
                "content": "new reasoning message",
                "reasoning_content": "new hidden",
            },
        ),
    ]
    assert result.cursors == {"m": 2}
    assert result.main_context == _main_transcript(result)


async def test_ingestion_missing_content_hash_target_fails_closed() -> None:
    call_id = _call_id(
        side="reviewer",
        content_hash_prefix_value=b"\xff" * CALL_ID_CONTENT_HASH_PREFIX_BYTES,
        upstream_tool_call_id="up_missing_0",
    )

    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            _request(input=[_function_call(call_id)]),
            keyring=_keyring(),
        )

    _assert_plap_error(
        exc_info.value,
        code="invalid_tool_replay",
        param="input",
        private_reason="sealed_function_call_content_hash_target_missing",
    )


async def test_ingestion_uses_nearest_backward_hash_prefix_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_hash(message: StateMessage) -> str:
        content = str(message.content or "")
        return "0102030405060708" + ("a" if content.endswith("a") else "b") * 48

    monkeypatch.setattr(StateMessage, "content_hash", fake_hash)
    call_id = _call_id(
        side="reviewer",
        content_hash_prefix_value=bytes.fromhex("0102030405060708"),
        upstream_tool_call_id="up_ambiguous_0",
    )

    result = await ingest_response_request(
        _request(
            input=[
                _reasoning_item(
                    "reviewer",
                    False,
                    [
                        {"role": "assistant", "content": "a"},
                        {"role": "assistant", "content": "b"},
                    ],
                ),
                _function_call(call_id),
                _function_output(call_id, "nearest output"),
            ]
        ),
        keyring=_keyring(),
    )

    assert [row.message.to_primitive() for row in result.reviewer] == [
        {"role": "assistant", "content": "a"},
        {
            "role": "assistant",
            "content": "b",
            "tool_calls": [_tool_call("up_ambiguous_0")],
        },
        {
            "role": "tool",
            "tool_call_id": "up_ambiguous_0",
            "content": "nearest output",
        },
    ]


async def test_ingestion_invalid_sealed_artifact_fails_closed() -> None:
    with pytest.raises(PlapError) as exc_info:
        await ingest_response_request(
            _request(
                input=[
                    RequestReasoningItem(
                        encrypted_content="not-valid",
                        id="rs_bad",
                        summary=[SummaryTextContent(text="bad", type="summary_text")],
                        type="reasoning",
                    )
                ]
            ),
            keyring=_keyring(),
        )

    _assert_plap_error(exc_info.value, code="invalid_input_replay", param="input", private_reason="sealed_payload_not_base64url")


def test_payload_domain_objects_do_not_expose_version_or_type_truths() -> None:
    assert "version" not in CompactionPayload.__dataclass_fields__
    assert "type" not in CompactionPayload.__dataclass_fields__
    assert "version" not in ReasoningPayload.__dataclass_fields__
    assert "type" not in ReasoningPayload.__dataclass_fields__


def test_chat_message_span_citation_uses_model_facing_syntax() -> None:
    leaf = ChatMessageSpan(
        start=0,
        end=0,
        message=_state_message({"role": "user"}),
        token_count=1,
    )
    range_span = ChatMessageSpan(
        start=0,
        end=7,
        message=_state_message({"role": "assistant"}),
        token_count=1,
        children_pruned=True,
        summary_fidelity=3,
    )

    assert leaf.citation == "[~0]"
    assert range_span.citation == "[~0_7]"


def test_sealed_compaction_rejects_wrong_payload_type() -> None:
    token = _seal_raw_payload(
        COMPACTION_PURPOSE,
        {
            "version": PAYLOAD_FORMAT_VERSION,
            "type": "reasoning",
            "active": [],
            "source": [],
            "cursors": {"m": 0},
        },
    )

    with pytest.raises(PlapError) as exc_info:
        open_compaction_payload(token, keyring=_keyring())

    _assert_plap_error(exc_info.value, code="invalid_compaction_replay", param="input", private_reason="unsupported_compaction_payload")


def test_sealed_compaction_rejects_active_spans_outside_cursors() -> None:
    token = seal_compaction_payload(
        CompactionPayload(
            active=(
                ChatMessageSpan(
                    start=5,
                    end=5,
                    message=_state_message({"role": "user", "content": "bad"}),
                    token_count=1,
                ),
            ),
            cursors={"m": 1},
        ),
        keyring=_keyring(),
    )

    with pytest.raises(PlapError) as exc_info:
        open_compaction_payload(token, keyring=_keyring())

    _assert_plap_error(
        exc_info.value, code="invalid_compaction_replay", param="input", private_reason="compaction_active_span_outside_cursor"
    )


def test_sealed_compaction_requires_token_count() -> None:
    token = _seal_raw_payload(
        COMPACTION_PURPOSE,
        {
            "version": PAYLOAD_FORMAT_VERSION,
            "type": "compaction",
            "active": [
                {
                    "start": 0,
                    "end": 0,
                    "message": {"role": "user", "content": "missing count"},
                }
            ],
            "cursors": {"m": 1},
        },
    )

    with pytest.raises(PlapError) as exc_info:
        open_compaction_payload(token, keyring=_keyring())

    _assert_plap_error(exc_info.value, code="invalid_compaction_replay", param="input", private_reason="compaction_payload_invalid")


def test_sealed_compaction_requires_summary_fidelity_for_summary_spans() -> None:
    token = _seal_raw_payload(
        COMPACTION_PURPOSE,
        {
            "version": PAYLOAD_FORMAT_VERSION,
            "type": "compaction",
            "active": [
                {
                    "start": 0,
                    "end": 1,
                    "message": {"role": "assistant", "content": "summary"},
                    "token_count": 1,
                    "children_token_count": 2,
                    "expanded_token_count": 2,
                    "children": [
                        {"start": 0, "end": 0, "message": {"role": "user", "content": "a"}, "token_count": 1},
                        {"start": 1, "end": 1, "message": {"role": "user", "content": "b"}, "token_count": 1},
                    ],
                }
            ],
            "cursors": {"m": 2},
        },
    )

    with pytest.raises(PlapError) as exc_info:
        open_compaction_payload(token, keyring=_keyring())

    _assert_plap_error(
        exc_info.value, code="invalid_compaction_replay", param="input", private_reason="compaction_summary_fidelity_missing"
    )


def test_sealed_compaction_rejects_overlapping_active_spans() -> None:
    token = seal_compaction_payload(
        CompactionPayload(
            active=(
                ChatMessageSpan(
                    start=0,
                    end=1,
                    message=_state_message({"role": "user", "content": "first"}),
                    token_count=1,
                    children_pruned=True,
                    summary_fidelity=3,
                ),
                ChatMessageSpan(
                    start=1,
                    end=1,
                    message=_state_message({"role": "user", "content": "second"}),
                    token_count=1,
                ),
            ),
            cursors={"m": 2},
        ),
        keyring=_keyring(),
    )

    with pytest.raises(PlapError) as exc_info:
        open_compaction_payload(token, keyring=_keyring())

    _assert_plap_error(exc_info.value, code="invalid_compaction_replay", param="input", private_reason="compaction_active_spans_overlap")


async def test_ingestion_main_context_active_transcript_budgeted() -> None:
    first_summary = ChatMessageSpan(
        start=0,
        end=1,
        message=_state_message({"role": "assistant", "content": "summary 0-1"}),
        token_count=1,
        summary_fidelity=4,
        children=(
            ChatMessageSpan(
                start=0,
                end=0,
                message=_state_message({"role": "user", "content": "m0"}),
                token_count=2,
            ),
            ChatMessageSpan(
                start=1,
                end=1,
                message=_state_message({"role": "assistant", "content": "m1"}),
                token_count=3,
            ),
        ),
    )
    second_summary = ChatMessageSpan(
        start=2,
        end=3,
        message=_state_message({"role": "assistant", "content": "summary 2-3"}),
        token_count=1,
        summary_fidelity=2,
        children=(
            ChatMessageSpan(
                start=2,
                end=2,
                message=_state_message({"role": "user", "content": "m2"}),
                token_count=2,
            ),
            ChatMessageSpan(
                start=3,
                end=3,
                message=_state_message({"role": "assistant", "content": "m3"}),
                token_count=3,
            ),
        ),
    )
    compaction = RequestCompactionItem(
        encrypted_content=seal_compaction_payload(
            CompactionPayload(
                active=(first_summary, second_summary),
                cursors={"m": 4},
            ),
            keyring=_keyring(),
        ),
        type="compaction",
    )

    result = await ingest_response_request(
        _request(input=[compaction]),
        keyring=_keyring(),
    )

    assert [row.citation for row in result.main_context] == ["[~0_1]", "[~2_3]"]
    assert [row.summary_fidelity for row in result.main_context] == [4, 2]
    assert [row.citation for row in _main_transcript(result, token_budget=6)] == [
        "[~0_1]",
        "[~2]",
        "[~3]",
    ]


async def test_ingestion_transcript_expansion_ties_prefer_newer_spans() -> None:
    first_summary = ChatMessageSpan(
        start=0,
        end=1,
        message=_state_message({"role": "assistant", "content": "summary 0-1"}),
        token_count=1,
        summary_fidelity=3,
        children=(
            ChatMessageSpan(start=0, end=0, message=_state_message({"role": "user", "content": "m0"}), token_count=2),
            ChatMessageSpan(start=1, end=1, message=_state_message({"role": "assistant", "content": "m1"}), token_count=3),
        ),
    )
    second_summary = ChatMessageSpan(
        start=2,
        end=3,
        message=_state_message({"role": "assistant", "content": "summary 2-3"}),
        token_count=1,
        summary_fidelity=3,
        children=(
            ChatMessageSpan(start=2, end=2, message=_state_message({"role": "user", "content": "m2"}), token_count=2),
            ChatMessageSpan(start=3, end=3, message=_state_message({"role": "assistant", "content": "m3"}), token_count=3),
        ),
    )
    compaction = RequestCompactionItem(
        encrypted_content=seal_compaction_payload(
            CompactionPayload(active=(first_summary, second_summary), cursors={"m": 4}),
            keyring=_keyring(),
        ),
        type="compaction",
    )

    result = await ingest_response_request(
        _request(input=[compaction]),
        keyring=_keyring(),
    )

    assert [row.citation for row in _main_transcript(result, token_budget=6)] == [
        "[~0_1]",
        "[~2]",
        "[~3]",
    ]


def test_call_id_binary_encoding_is_compact_and_roundtrips() -> None:
    value = SealedCallID(
        side="arbitrator",
        content_hash_prefix=bytes.fromhex("0102030405060708"),
        tool_call_index=300,
        upstream_tool_call_id="provider_123",
    )

    token = seal_call_id(value, keyring=_keyring())

    assert token.startswith("call_")
    assert len(token) < 60
    assert open_call_id(token, keyring=_keyring()) == value


def test_old_tools_call_ids_module_is_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("plap.responses.tools.call_ids")


def _request(
    *,
    input: list[object] | None = None,
    tools: list[object] | None = None,
) -> ResponseCreateRequest:
    return ResponseCreateRequest(input=input, model="test/model", tools=tools)


def _message(role: str, content: str) -> RequestMessageItem:
    return RequestMessageItem(content=content, role=role, type="message")


def _compaction_item(label: str, cursor: int) -> RequestCompactionItem:
    if cursor < 1:
        raise ValueError("cursor must be positive")
    source = tuple(
        ChatMessageSpan(
            start=ordinal,
            end=ordinal,
            message=_state_message(
                {
                    "role": "user",
                    "content": (f"{label} source" if ordinal == 0 else f"{label} summarized source"),
                }
            ),
            token_count=1,
        )
        for ordinal in range(cursor)
    )
    active = [source[0]]
    if cursor == 2:
        active.append(source[1])
    elif cursor > 2:
        active.append(
            ChatMessageSpan(
                start=1,
                end=cursor - 1,
                message=_state_message({"role": "assistant", "content": f"{label} summary"}),
                token_count=1,
                summary_fidelity=3,
                children=source[1:],
            )
        )
    payload = CompactionPayload(
        active=tuple(active),
        cursors={"m": cursor},
    )
    return RequestCompactionItem(
        encrypted_content=seal_compaction_payload(payload, keyring=_keyring()),
        type="compaction",
    )


def _reasoning_item(
    side: str,
    temp: bool,
    messages: list[dict[str, object]],
    *,
    continuation_side: str | None = None,
    item_id: str | None = None,
) -> RequestReasoningItem:
    payload = ReasoningPayload(
        side=side,
        temp=temp,
        messages=tuple(_reasoning_message(message) for message in messages),
        continuation_side=continuation_side,
    )
    return RequestReasoningItem(
        encrypted_content=seal_reasoning_payload(payload, keyring=_keyring()),
        id=item_id,
        summary=[SummaryTextContent(text="sealed", type="summary_text")],
        type="reasoning",
    )


def _call_id(
    *,
    side: str,
    upstream_tool_call_id: str,
    content_hash_value: str | None = None,
    content_hash_prefix_value: bytes | None = None,
    tool_call_index: int = 0,
) -> str:
    if content_hash_prefix_value is None:
        if content_hash_value is None:
            raise ValueError("content_hash_value or content_hash_prefix_value is required")
        content_hash_prefix_value = content_hash_prefix(content_hash_value)
    return seal_call_id(
        SealedCallID(
            side=side,
            content_hash_prefix=content_hash_prefix_value,
            tool_call_index=tool_call_index,
            upstream_tool_call_id=upstream_tool_call_id,
        ),
        keyring=_keyring(),
    )


def _function_call(call_id: str) -> RequestFunctionCallItem:
    return RequestFunctionCallItem(
        arguments='{"path":"README.md"}',
        call_id=call_id,
        name="read_file",
        type="function_call",
    )


def _function_output(call_id: str, output: str) -> RequestFunctionCallOutputItem:
    return RequestFunctionCallOutputItem(
        call_id=call_id,
        output=output,
        type="function_call_output",
    )


def _tool_call(upstream_id: str) -> dict[str, object]:
    return {
        "id": upstream_id,
        "name": "read_file",
        "arguments": '{"path":"README.md"}',
    }


def _state_message(value: dict[str, object]) -> StateMessage:
    return StateMessage.from_primitive(value)


def _reasoning_message(value: dict[str, object]) -> StateMessage | ReasoningMessagePatch:
    if isinstance(value.get("content_hash"), str):
        return ReasoningMessagePatch.from_primitive(value)
    return StateMessage.from_primitive(value)


def _message_hash(value: dict[str, object]) -> str:
    return content_hash(_state_message(value))


def _seal_raw_payload(purpose: str, value: object) -> str:
    compressed = zstd.ZstdCompressor().compress(msgspec.json.encode(value, order="deterministic"))
    encrypted = Aead(_keyring().active(purpose)).encrypt(compressed, purpose_label(purpose))
    return base64.urlsafe_b64encode(bytes(encrypted)).rstrip(b"=").decode()


def _keyring() -> SealingKeyring:
    return SealingKeyring(roots=(b"i" * 32,))
