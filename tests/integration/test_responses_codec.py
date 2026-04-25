from __future__ import annotations

from uuid import uuid4

import pytest

from plap.responses.codec import (
    decode_response,
    encode_output_items,
    fields_from_request,
)
from plap.responses.contracts import (
    OutputTextContent,
    ReasoningItem,
    ResponseCreateRequest,
    ResponseMessageItem,
    SummaryTextContent,
)
from plap.responses.state import ResponseRepository


def _request(model: str = "test/model") -> ResponseCreateRequest:
    return ResponseCreateRequest(
        metadata={"trace_id": "abc"},
        model=model,
        temperature=0.2,
        top_p=1,
    )


def _message(item_id: str, text: str) -> ResponseMessageItem:
    return ResponseMessageItem(
        content=[OutputTextContent(text=text, type="output_text")],
        id=item_id,
        role="assistant",
        status="completed",
        type="message",
    )


async def _previous_cursors(
    repository: ResponseRepository,
    scope_id,
    response_id: str | None,
) -> dict[str, int]:
    if response_id is None:
        return {"m": 0, "r": 0, "s": 0}
    cursors = await repository.get_namespace_cursors(scope_id, response_id)
    return {cursor.namespace: cursor.next_ordinal for cursor in cursors}


async def _create_wire_response(
    repository: ResponseRepository,
    scope_id,
    response_id: str,
    previous_response_id: str | None,
    output_items,
    request: ResponseCreateRequest,
    *,
    conversation_id: str | None = None,
):
    state_items, output_manifest, namespace_cursors = encode_output_items(
        output_items,
        await _previous_cursors(repository, scope_id, previous_response_id),
    )
    result = await repository.append_response(
        scope_id,
        response_id,
        previous_response_id,
        state_items,
        namespace_cursors,
        output_items=output_manifest,
        fields=fields_from_request(request),
    )
    if conversation_id is not None:
        await repository.move_conversation_head(scope_id, conversation_id, response_id)
    return await repository.get_response_record(scope_id, result.response_id)


@pytest.mark.asyncio
async def test_codec_creates_retrieves_and_lists_current_state(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        repository = ResponseRepository(session)
        first = await _create_wire_response(
            repository,
            scope_id,
            "resp_1",
            None,
            [_message("msg_1", "one")],
            _request(),
        )
        second = await _create_wire_response(
            repository,
            scope_id,
            "resp_2",
            "resp_1",
            [_message("msg_2", "two")],
            _request(model="test/model-2"),
        )
        await session.commit()

        assert second is not None
        state_items = await repository.list_items(scope_id, second.state_root_id)
        output_entries = await repository.list_response_outputs(scope_id, "resp_2")
        retrieved = decode_response(second, output_entries)

    assert first is not None
    assert first.response_id == "resp_1"
    assert second.previous_response_id == "resp_1"
    assert [(item.namespace, item.ordinal, item.position) for item in state_items] == [
        ("m", 0, 0),
        ("m", 1, 1),
    ]
    assert "id" not in state_items[0].payload
    assert "type" not in state_items[0].payload
    assert retrieved.id == "resp_2"
    assert retrieved.previous_response_id == "resp_1"
    assert retrieved.model == "test/model-2"
    assert retrieved.metadata == {"trace_id": "abc"}
    assert [item.id for item in retrieved.output] == ["msg_2"]


@pytest.mark.asyncio
async def test_codec_assigns_reasoning_to_reasoning_namespace(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    reasoning = ReasoningItem(
        id="rsn_1",
        summary=[SummaryTextContent(text="thought summary", type="summary_text")],
        type="reasoning",
    )

    async with db_session_maker() as session:
        repository = ResponseRepository(session)
        record = await _create_wire_response(
            repository,
            scope_id,
            "resp_reasoning",
            None,
            [_message("msg_reasoning", "answer"), reasoning],
            _request(),
        )
        await session.commit()

        assert record is not None
        state_items = await repository.list_items(scope_id, record.state_root_id)
        output_entries = await repository.list_response_outputs(
            scope_id,
            "resp_reasoning",
        )
        retrieved = decode_response(record, output_entries)

    assert [(item.namespace, item.ordinal) for item in state_items] == [
        ("m", 0),
        ("r", 0),
    ]
    assert [item.type for item in retrieved.output] == ["message", "reasoning"]


@pytest.mark.asyncio
async def test_codec_moves_conversation_head_with_retention(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        repository = ResponseRepository(session)
        await _create_wire_response(
            repository,
            scope_id,
            "resp_conversation",
            None,
            [_message("msg_conversation", "hello")],
            _request(),
            conversation_id="conv_1",
        )
        await session.commit()

        head = await repository.get_response_record(scope_id, "resp_conversation")
        assert head is not None
        conversation_items = await repository.list_items(scope_id, head.state_root_id)

    assert head.response_id == "resp_conversation"
    assert [item.payload["content"][0]["text"] for item in conversation_items] == [
        "hello"
    ]


@pytest.mark.asyncio
async def test_codec_returns_none_for_missing_retrieve(db_session_maker) -> None:
    async with db_session_maker() as session:
        repository = ResponseRepository(session)
        missing_record = await repository.get_response_record(uuid4(), "missing")
        assert missing_record is None
