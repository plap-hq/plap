from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text

from plap.responses.state import NamespaceCursor, ResponseRepository, StateItem


def _cursors(
    message_next_ordinal: int,
    summary_next_ordinal: int = 0,
    reasoning_next_ordinal: int = 0,
):
    return (
        NamespaceCursor(namespace="m", next_ordinal=message_next_ordinal),
        NamespaceCursor(namespace="r", next_ordinal=reasoning_next_ordinal),
        NamespaceCursor(namespace="s", next_ordinal=summary_next_ordinal),
    )


def _message(ordinal: int, text: str) -> StateItem:
    return StateItem(
        namespace="m",
        ordinal=ordinal,
        payload={"type": "message", "text": text},
    )


def _summary(ordinal: int, text: str) -> StateItem:
    return StateItem(
        namespace="s",
        ordinal=ordinal,
        payload={"type": "summary", "text": text},
    )


@pytest.mark.asyncio
async def test_response_state_repository_builds_and_lists_tree(
    db_session_maker,
) -> None:
    scope_id = uuid4()
    items = [_message(0, "one"), _message(1, "two")]

    async with db_session_maker() as session:
        repository = ResponseRepository(session)
        state_root_id = await repository.build_tree(scope_id, items)
        await session.commit()

        listed = await repository.list_items(scope_id, state_root_id)

    assert [
        (item.namespace, item.ordinal, item.payload["text"]) for item in listed
    ] == [
        ("m", 0, "one"),
        ("m", 1, "two"),
    ]
    assert all(item.payload_hash for item in listed)


@pytest.mark.asyncio
async def test_response_state_repository_rejects_mismatched_payload_hash(
    db_session_maker,
) -> None:
    scope_id = uuid4()
    item = StateItem(
        namespace="m",
        ordinal=0,
        payload={"type": "message", "text": "one"},
        payload_hash="0" * 64,
    )

    async with db_session_maker() as session:
        repository = ResponseRepository(session)
        with pytest.raises(ValueError, match="payload_hash does not match"):
            await repository.build_tree(scope_id, [item])


@pytest.mark.asyncio
async def test_response_state_repository_appends_and_reads_records(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        repository = ResponseRepository(session)
        first = await repository.append_response(
            scope_id,
            "resp_1",
            None,
            [_message(0, "one")],
            _cursors(1),
            fields={"metadata": {"trace_id": "abc"}, "model": "test/model"},
        )
        second = await repository.append_response(
            scope_id,
            "resp_2",
            "resp_1",
            [_message(1, "two")],
            _cursors(2),
        )
        await session.commit()

        first_record = await repository.get_response_record(scope_id, "resp_1")
        second_record = await repository.get_response_record(scope_id, "resp_2")
        listed = await repository.list_items(scope_id, second.state_root_id)

    assert first.response_id == "resp_1"
    assert first_record is not None
    assert first_record.previous_response_id is None
    assert first_record.status == "completed"
    assert first_record.completed_at is not None
    assert first_record.fields == {
        "metadata": {"trace_id": "abc"},
        "model": "test/model",
    }
    assert second_record is not None
    assert second_record.previous_response_id == "resp_1"
    assert second_record.state_root_id == second.state_root_id
    assert [item.payload["text"] for item in listed] == ["one", "two"]


@pytest.mark.asyncio
async def test_response_state_repository_splices_summary_state(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        repository = ResponseRepository(session)
        base_root = await repository.build_tree(
            scope_id,
            [
                _message(0, "m1"),
                _message(1, "m2"),
                _message(2, "m3"),
                _message(3, "m4"),
            ],
        )
        summary_root = await repository.build_tree(scope_id, [_summary(0, "s1")])
        compacted_root = await repository.splice_tree(
            scope_id,
            base_root,
            1,
            2,
            insert_state_root_id=summary_root,
        )
        assert compacted_root is not None

        await repository.create_response_record(
            scope_id,
            "resp_compacted",
            None,
            compacted_root,
            _cursors(4, 1),
            status="in_progress",
            completed_at=None,
            fields={"metadata": {"flow": "summary"}},
        )
        await repository.move_conversation_head(scope_id, "conv_1", "resp_compacted")
        await session.commit()

        head = await repository.get_conversation_head(scope_id, "conv_1")
        listed = await repository.list_items(scope_id, compacted_root)

    assert head is not None
    assert head.response_id == "resp_compacted"
    assert head.status == "in_progress"
    assert head.completed_at is None
    assert head.fields == {"metadata": {"flow": "summary"}}
    assert [
        (item.namespace, item.ordinal, item.payload["text"]) for item in listed
    ] == [
        ("m", 0, "m1"),
        ("s", 0, "s1"),
        ("m", 3, "m4"),
    ]


@pytest.mark.asyncio
async def test_response_state_repository_scopes_response_ids(
    db_session_maker,
) -> None:
    first_scope_id = uuid4()
    second_scope_id = uuid4()

    async with db_session_maker() as session:
        repository = ResponseRepository(session)
        first = await repository.append_response(
            first_scope_id,
            "resp_same",
            None,
            [_message(0, "first scope")],
            _cursors(1),
        )
        second = await repository.append_response(
            second_scope_id,
            "resp_same",
            None,
            [_message(0, "second scope")],
            _cursors(1),
        )
        await session.commit()

        first_record = await repository.get_response_record(first_scope_id, "resp_same")
        second_record = await repository.get_response_record(
            second_scope_id, "resp_same"
        )
        missing = await repository.get_response_record(uuid4(), "resp_same")
        first_items = await repository.list_items(first_scope_id, first.state_root_id)
        second_items = await repository.list_items(
            second_scope_id, second.state_root_id
        )

    assert first_record is not None
    assert second_record is not None
    assert missing is None
    assert first_items[0].payload["text"] == "first scope"
    assert second_items[0].payload["text"] == "second scope"


@pytest.mark.asyncio
async def test_response_state_repository_preserves_explicit_completed_at(
    db_session_maker,
) -> None:
    scope_id = uuid4()
    completed_at = datetime(2026, 4, 24, 12, 30, tzinfo=UTC)

    async with db_session_maker() as session:
        repository = ResponseRepository(session)
        await repository.append_response(
            scope_id,
            "resp_completed_at",
            None,
            [_message(0, "done")],
            _cursors(1),
            completed_at=completed_at,
            fields={"service_tier": "default"},
        )
        await session.commit()

        record = await repository.get_response_record(scope_id, "resp_completed_at")

    assert record is not None
    assert record.completed_at == completed_at
    assert record.fields == {"service_tier": "default"}


@pytest.mark.asyncio
async def test_response_state_repository_sets_conversation_retention(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        repository = ResponseRepository(session)
        await repository.append_response(
            scope_id,
            "resp_retention",
            None,
            [_message(0, "retained")],
            _cursors(1),
        )
        await repository.move_conversation_head(
            scope_id,
            "conv_retention",
            "resp_retention",
            retention=timedelta(hours=2),
        )
        await session.commit()

        retention_seconds = (
            await session.execute(
                text(
                    """
                    select extract(epoch from (retention_expires_at - now()))
                      from responses.conversations
                     where scope_id = :scope_id
                       and conversation_id = 'conv_retention'
                    """
                ),
                {"scope_id": scope_id},
            )
        ).scalar_one()

    assert 60 * 60 < retention_seconds <= 2 * 60 * 60
