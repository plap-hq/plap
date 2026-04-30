from __future__ import annotations

import base64
import importlib

import msgspec
import pytest
import zstandard as zstd
from nacl.secret import Aead

from plap.keyring import SealingKeyring, purpose_label
from plap.responses.contracts import (
    ReasoningItem,
    RequestCompactionItem,
    RequestFunctionCallItem,
    RequestFunctionCallOutputItem,
    RequestMessageItem,
    ResponseCreateRequest,
    SummaryTextContent,
)
from plap.responses.ingest import (
    CALL_ID_CONTENT_HASH_PREFIX_BYTES,
    ChatMessageSpan,
    CompactionPayload,
    IngestedQueues,
    IngestionError,
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


async def ingest_response_request(
    request: ResponseCreateRequest,
    *,
    keyring: SealingKeyring,
    transcript_token_budget: int = 0,
) -> IngestedQueues:
    return await _ingest_response_request(
        request,
        keyring=keyring,
        transcript_token_budget=transcript_token_budget,
    )


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

    assert [(row.start, row.end, row.message["content"]) for row in result.main_context] == [
        (0, 0, "kept source"),
        (1, 1, "kept summarized source"),
        (2, 2, "after"),
    ]
    assert [row.message["content"] for row in result.main_transcript] == [
        "kept source",
        "kept summarized source",
        "after",
    ]
    assert result.cursors == {"m": 3}
    assert result.continuation_side == "main"


async def test_ingestion_compaction_only_continues_main() -> None:
    result = await ingest_response_request(
        _request(input=[_compaction_item("only", 2)]),
        keyring=_keyring(),
    )

    assert [(row.start, row.end) for row in result.main_context] == [
        (0, 0),
        (1, 1),
    ]
    assert [(row.start, row.end) for row in result.main_transcript] == [
        (0, 0),
        (1, 1),
    ]
    assert result.continuation_side == "main"


async def test_ingestion_assigns_m_ordinals_without_compaction() -> None:
    result = await ingest_response_request(
        _request(input=[_message("user", "u0"), _message("assistant", "a0")]),
        keyring=_keyring(),
    )

    assert [(row.start, row.end, row.message["content"]) for row in result.main_context] == [
        (0, 0, "u0"),
        (1, 1, "a0"),
    ]
    assert result.cursors == {"m": 2}
    assert result.main_context == result.main_transcript
    assert result.continuation_side == "main"


async def test_ingestion_routes_reasoning_by_sealed_side_with_hashes() -> None:
    result = await ingest_response_request(
        _request(input=[_reasoning_item("reviewer", False, [{"role": "assistant", "content": "review"}])]),
        keyring=_keyring(),
    )

    assert result.main_context == ()
    assert result.main_transcript == ()
    assert [(row.message["content"], row.content_hash) for row in result.reviewer] == [
        ("review", content_hash({"role": "assistant", "content": "review"}))
    ]
    assert result.arbitrator == ()
    assert result.continuation_side == "reviewer"


async def test_ingestion_arbitrator_reasoning_sets_continuation_side() -> None:
    result = await ingest_response_request(
        _request(input=[_reasoning_item("arbitrator", False, [{"role": "assistant", "content": "decide"}])]),
        keyring=_keyring(),
    )

    assert result.arbitrator[0].message["content"] == "decide"
    assert result.continuation_side == "arbitrator"


async def test_ingestion_temp_false_prunes_entire_temp_debate() -> None:
    temp_message = {"role": "assistant", "content": "temp reviewer"}
    call_id = _call_id(
        side="reviewer",
        content_hash_value=content_hash(temp_message),
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

    assert [row.message["content"] for row in result.main_context] == ["final debate result"]
    assert result.main_context_temp == ()
    assert result.main_context == result.main_transcript
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
        content_hash_value=content_hash(temp_message),
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

    assert [row.message["content"] for row in result.main_context] == ["new mainline request"]
    assert result.main_context_temp == ()
    assert result.main_context == result.main_transcript
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

    assert [row.message for row in result.main_context] == [
        {
            "role": "assistant",
            "content": "stable assistant",
            "tool_calls": [
                {
                    "id": "client_call_0",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    },
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
    assert result.main_context == result.main_transcript
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
    assert [row.message["content"] for row in result.main_context_temp] == ["temp debate tail"]
    assert result.main_transcript == ()
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
        content_hash_value=content_hash(assistant),
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
    assert result.main_transcript == ()
    assert [row.message for row in result.reviewer] == [
        assistant,
        {"role": "tool", "tool_call_id": "up_reviewer_0", "content": "review file"},
    ]
    assert result.continuation_side == "reviewer"


async def test_ingestion_routes_sealed_main_call_and_tool_output_to_m_rows() -> None:
    assistant = {"role": "assistant", "content": "public assistant"}
    call_id = _call_id(
        side="main",
        content_hash_value=content_hash(assistant),
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

    assert [(row.start, row.end, row.message) for row in result.main_context] == [
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
    assert result.main_context == result.main_transcript
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

    assert [(row.start, row.end, row.message) for row in result.main_context] == [
        (
            0,
            0,
            {
                "role": "assistant",
                "content": "client fabricated assistant",
                "tool_calls": [
                    {
                        "id": "client_call_0",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"README.md"}',
                        },
                    }
                ],
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
    assert result.main_context == result.main_transcript
    assert result.reviewer == ()
    assert result.arbitrator == ()
    assert result.continuation_side == "main"


async def test_ingestion_rejects_unsealed_call_interleaving_before_output() -> None:
    with pytest.raises(IngestionError, match="pending tool outputs"):
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

    with pytest.raises(IngestionError, match="duplicate pending unsealed"):
        await ingest_response_request(
            _request(input=[_message("assistant", "anchor"), first_call, second_call]),
            keyring=_keyring(),
        )


async def test_ingestion_rejects_sealed_call_interleaving_before_output() -> None:
    assistant = {"role": "assistant", "content": "need file"}
    call_id = _call_id(
        side="reviewer",
        content_hash_value=content_hash(assistant),
        upstream_tool_call_id="up_reviewer_0",
    )

    with pytest.raises(IngestionError, match="pending tool outputs"):
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


async def test_ingestion_allows_stripped_tool_call_association() -> None:
    stripped = {"role": "assistant", "content": "need file"}
    call_id = _call_id(
        side="reviewer",
        content_hash_value=content_hash(stripped),
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

    assert [row.message for row in result.reviewer] == [
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

    with pytest.raises(IngestionError, match="missing function_call item"):
        await ingest_response_request(
            _request(input=[_reasoning_item("reviewer", False, [assistant])]),
            keyring=_keyring(),
        )


async def test_ingestion_accepts_reasoning_tool_call_with_public_pair() -> None:
    assistant = {
        "role": "assistant",
        "content": "need file",
        "tool_calls": [_tool_call("up_reasoning_0")],
    }
    call_id = _call_id(
        side="reviewer",
        content_hash_value=content_hash(assistant),
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

    assert [row.message for row in result.reviewer] == [
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
        content_hash_value=content_hash(assistant),
        upstream_tool_call_id="up_reasoning_0",
    )

    with pytest.raises(IngestionError, match="missing function_call_output"):
        await ingest_response_request(
            _request(
                input=[
                    _reasoning_item("reviewer", False, [assistant]),
                    _function_call(call_id),
                ]
            ),
            keyring=_keyring(),
        )


async def test_ingestion_rejects_reasoning_forward_refs() -> None:
    target = {"role": "assistant", "content": "target"}

    with pytest.raises(IngestionError, match="content_hash target is missing"):
        await ingest_response_request(
            _request(
                input=[
                    _reasoning_item(
                        "reviewer",
                        False,
                        [
                            {
                                "content_hash": content_hash(target),
                                "reasoning_content": "hidden",
                            },
                            target,
                        ],
                    )
                ]
            ),
            keyring=_keyring(),
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
                            "content_hash": content_hash(anchor),
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

    assert [(row.start, row.end, row.message) for row in result.main_context] == [
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
    assert result.main_context == result.main_transcript


async def test_ingestion_missing_content_hash_target_fails_closed() -> None:
    call_id = _call_id(
        side="reviewer",
        content_hash_prefix_value=b"\xff" * CALL_ID_CONTENT_HASH_PREFIX_BYTES,
        upstream_tool_call_id="up_missing_0",
    )

    with pytest.raises(IngestionError, match="content_hash target is missing"):
        await ingest_response_request(
            _request(input=[_function_call(call_id)]),
            keyring=_keyring(),
        )


async def test_ingestion_uses_nearest_backward_hash_prefix_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_hash(message: dict[str, object]) -> str:
        content = str(message.get("content", ""))
        return "0102030405060708" + ("a" if content.endswith("a") else "b") * 48

    monkeypatch.setattr("plap.responses.ingest.types.chat_message_hash", fake_hash)
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

    assert [row.message for row in result.reviewer] == [
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
    with pytest.raises(IngestionError):
        await ingest_response_request(
            _request(
                input=[
                    ReasoningItem(
                        encrypted_content="not-valid",
                        id="rs_bad",
                        summary=[SummaryTextContent(text="bad", type="summary_text")],
                        type="reasoning",
                    )
                ]
            ),
            keyring=_keyring(),
        )


def test_payload_domain_objects_do_not_expose_version_or_type_truths() -> None:
    assert "version" not in CompactionPayload.__dataclass_fields__
    assert "type" not in CompactionPayload.__dataclass_fields__
    assert "version" not in ReasoningPayload.__dataclass_fields__
    assert "type" not in ReasoningPayload.__dataclass_fields__


def test_chat_message_span_citation_uses_model_facing_syntax() -> None:
    leaf = ChatMessageSpan(
        start=0,
        end=0,
        message={"role": "user"},
        token_count=1,
    )
    range_span = ChatMessageSpan(
        start=0,
        end=7,
        message={"role": "assistant"},
        token_count=1,
        children_pruned=True,
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

    with pytest.raises(IngestionError, match="unsupported compaction payload"):
        open_compaction_payload(token, keyring=_keyring())


def test_sealed_compaction_rejects_active_spans_outside_cursors() -> None:
    token = seal_compaction_payload(
        CompactionPayload(
            active=(
                ChatMessageSpan(
                    start=5,
                    end=5,
                    message={"role": "user", "content": "bad"},
                    token_count=1,
                ),
            ),
            cursors={"m": 1},
        ),
        keyring=_keyring(),
    )

    with pytest.raises(IngestionError, match="outside cursor"):
        open_compaction_payload(token, keyring=_keyring())


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

    with pytest.raises(IngestionError, match="token_count"):
        open_compaction_payload(token, keyring=_keyring())


def test_sealed_compaction_rejects_overlapping_active_spans() -> None:
    token = seal_compaction_payload(
        CompactionPayload(
            active=(
                ChatMessageSpan(
                    start=0,
                    end=1,
                    message={"role": "user", "content": "first"},
                    token_count=1,
                    children_pruned=True,
                ),
                ChatMessageSpan(
                    start=1,
                    end=1,
                    message={"role": "user", "content": "second"},
                    token_count=1,
                ),
            ),
            cursors={"m": 2},
        ),
        keyring=_keyring(),
    )

    with pytest.raises(IngestionError, match="overlap"):
        open_compaction_payload(token, keyring=_keyring())


async def test_ingestion_main_context_active_transcript_budgeted() -> None:
    first_summary = ChatMessageSpan(
        start=0,
        end=1,
        message={"role": "assistant", "content": "summary 0-1"},
        token_count=1,
        children=(
            ChatMessageSpan(
                start=0,
                end=0,
                message={"role": "user", "content": "m0"},
                token_count=2,
            ),
            ChatMessageSpan(
                start=1,
                end=1,
                message={"role": "assistant", "content": "m1"},
                token_count=3,
            ),
        ),
    )
    second_summary = ChatMessageSpan(
        start=2,
        end=3,
        message={"role": "assistant", "content": "summary 2-3"},
        token_count=1,
        children=(
            ChatMessageSpan(
                start=2,
                end=2,
                message={"role": "user", "content": "m2"},
                token_count=2,
            ),
            ChatMessageSpan(
                start=3,
                end=3,
                message={"role": "assistant", "content": "m3"},
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
        transcript_token_budget=6,
    )

    assert [row.citation for row in result.main_context] == ["[~0_1]", "[~2_3]"]
    assert [row.citation for row in result.main_transcript] == [
        "[~0]",
        "[~1]",
        "[~2_3]",
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
            message={
                "role": "user",
                "content": (f"{label} source" if ordinal == 0 else f"{label} summarized source"),
            },
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
                message={"role": "assistant", "content": f"{label} summary"},
                token_count=1,
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
) -> ReasoningItem:
    payload = ReasoningPayload(side=side, temp=temp, messages=tuple(messages))
    return ReasoningItem(
        encrypted_content=seal_reasoning_payload(payload, keyring=_keyring()),
        id="rs_test",
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
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
    }


def _seal_raw_payload(purpose: str, value: object) -> str:
    compressed = zstd.ZstdCompressor().compress(msgspec.json.encode(value, order="deterministic"))
    encrypted = Aead(_keyring().active(purpose)).encrypt(compressed, purpose_label(purpose))
    return base64.urlsafe_b64encode(bytes(encrypted)).rstrip(b"=").decode()


def _keyring() -> SealingKeyring:
    return SealingKeyring(roots=(b"i" * 32,))
