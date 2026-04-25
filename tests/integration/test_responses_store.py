from __future__ import annotations

from uuid import uuid4

import pytest

from plap.responses.contracts import (
    OutputTextContent,
    ReasoningItem,
    ResponseCreateRequest,
    ResponseMessageItem,
    SummaryTextContent,
)
from plap.responses.state import ResponseRepository, ResponseStore


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


@pytest.mark.asyncio
async def test_response_store_creates_retrieves_and_lists_current_state(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        store = ResponseStore(ResponseRepository(session))
        first = await store.create_response(
            scope_id,
            "resp_1",
            None,
            [_message("msg_1", "one")],
            _request(),
        )
        second = await store.create_response(
            scope_id,
            "resp_2",
            "resp_1",
            [_message("msg_2", "two")],
            _request(model="test/model-2"),
        )
        await session.commit()

        state_items = await store.list_state_items(scope_id, "resp_2")
        second_record = await store.get_record(scope_id, "resp_2")
        retrieved = await store.retrieve_response(scope_id, "resp_2")

    assert first.response_id == "resp_1"
    assert second.previous_response_id == "resp_1"
    assert second_record is not None
    assert second_record.output_state_root_id != second_record.state_root_id
    assert [(item.namespace, item.ordinal, item.position) for item in state_items] == [
        ("m", 0, 0),
        ("m", 1, 1),
    ]
    assert retrieved is not None
    assert retrieved.id == "resp_2"
    assert retrieved.previous_response_id == "resp_1"
    assert retrieved.model == "test/model-2"
    assert retrieved.metadata == {"trace_id": "abc"}
    assert [item.id for item in retrieved.output] == ["msg_2"]


@pytest.mark.asyncio
async def test_response_store_assigns_reasoning_to_reasoning_namespace(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    reasoning = ReasoningItem(
        id="rsn_1",
        summary=[SummaryTextContent(text="thought summary", type="summary_text")],
        type="reasoning",
    )

    async with db_session_maker() as session:
        store = ResponseStore(ResponseRepository(session))
        await store.create_response(
            scope_id,
            "resp_reasoning",
            None,
            [_message("msg_reasoning", "answer"), reasoning],
            _request(),
        )
        await session.commit()

        state_items = await store.list_state_items(scope_id, "resp_reasoning")
        retrieved = await store.retrieve_response(scope_id, "resp_reasoning")

    assert [(item.namespace, item.ordinal) for item in state_items] == [
        ("m", 0),
        ("r", 0),
    ]
    assert retrieved is not None
    assert [item.type for item in retrieved.output] == ["message", "reasoning"]


@pytest.mark.asyncio
async def test_response_store_moves_conversation_head_with_retention(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        store = ResponseStore(ResponseRepository(session))
        await store.create_response(
            scope_id,
            "resp_conversation",
            None,
            [_message("msg_conversation", "hello")],
            _request(),
            conversation_id="conv_1",
        )
        await session.commit()

        head = await store.get_record(scope_id, "resp_conversation")
        conversation_items = await store.list_state_items(scope_id, "resp_conversation")

    assert head is not None
    assert head.response_id == "resp_conversation"
    assert [item.item.id for item in conversation_items] == ["msg_conversation"]


@pytest.mark.asyncio
async def test_response_store_returns_none_for_missing_retrieve(
    db_session_maker,
) -> None:
    async with db_session_maker() as session:
        store = ResponseStore(ResponseRepository(session))
        assert await store.retrieve_response(uuid4(), "missing") is None
        assert await store.list_state_items(uuid4(), "missing") == []
