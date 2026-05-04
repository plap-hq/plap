from __future__ import annotations

import json
from datetime import timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError


def _items(*items: dict[str, object]) -> str:
    return json.dumps(list(items))


async def _append_response(
    session,
    scope_id,
    response_id: str,
    *,
    prev_response_id: str | None = None,
    input_items: list[dict[str, object]] | None = None,
    output_items: list[dict[str, object]] | None = None,
    retention: timedelta | None = None,
) -> tuple[str, str]:
    row = (
        await session.execute(
            text(
                """
                select response_id, replay_base_response_id
                  from responses.append_response(
                    :scope_id,
                    :response_id,
                    :prev_response_id,
                    cast(:input_items as jsonb),
                    cast(:output_items as jsonb),
                    :retention
                  )
                """
            ),
            {
                "scope_id": scope_id,
                "response_id": response_id,
                "prev_response_id": prev_response_id,
                "input_items": _items(*(input_items or [])),
                "output_items": _items(*(output_items or [])),
                "retention": retention,
            },
        )
    ).one()
    return row.response_id, row.replay_base_response_id


async def _response_refcounts(session, scope_id, response_id: str) -> tuple[int, int]:
    row = (
        await session.execute(
            text(
                """
                select child_refcount, lease_refcount
                  from responses.response_records
                 where scope_id = :scope_id
                   and response_id = :response_id
                """
            ),
            {"scope_id": scope_id, "response_id": response_id},
        )
    ).one()
    return row.child_refcount, row.lease_refcount


async def _replay_rows(session, scope_id, response_id: str) -> list[tuple[str, str, int, str]]:
    rows = (
        await session.execute(
            text(
                """
                select replay_response_id, direction, item_index, payload ->> 'type' as item_type
                  from responses.list_response_replay(:scope_id, :response_id)
                 order by replay_response_id, direction, item_index
                """
            ),
            {"scope_id": scope_id, "response_id": response_id},
        )
    ).all()
    return [(row.replay_response_id, row.direction, row.item_index, row.item_type) for row in rows]


async def _replay_payloads(session, scope_id, response_id: str) -> list[dict[str, object]]:
    rows = (
        await session.execute(
            text(
                """
                select payload
                  from responses.list_response_replay(:scope_id, :response_id)
                 order by replay_response_id, direction, item_index
                """
            ),
            {"scope_id": scope_id, "response_id": response_id},
        )
    ).all()
    return [row.payload for row in rows]


async def test_responses_payload_dedupe_returns_same_payload_id(db_session_maker) -> None:
    scope_id = uuid4()
    payload = {"type": "message", "role": "user", "content": "hello"}

    async with db_session_maker() as session:
        first_id = (
            await session.execute(
                text(
                    """
                    select responses.get_or_create_payload(:scope_id, cast(:payload_json as jsonb))
                    """
                ),
                {"scope_id": scope_id, "payload_json": json.dumps(payload)},
            )
        ).scalar_one()
        second_id = (
            await session.execute(
                text(
                    """
                    select responses.get_or_create_payload(:scope_id, cast(:payload_json as jsonb))
                    """
                ),
                {"scope_id": scope_id, "payload_json": json.dumps(payload)},
            )
        ).scalar_one()

        assert first_id == second_id

        count = (
            await session.execute(
                text("select count(*) from responses.payloads where scope_id = :scope_id"),
                {"scope_id": scope_id},
            )
        ).scalar_one()

    assert count == 1


async def test_responses_create_response_record_rejects_missing_previous_response(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        with pytest.raises(SQLAlchemyError, match="previous response"):
            await session.execute(
                text(
                    """
                    select responses.create_response_record(
                      :scope_id,
                      'resp_missing_prev',
                      'resp_unknown',
                      cast(:input_items as jsonb),
                      '[]'::jsonb,
                      null
                    )
                    """
                ),
                {
                    "scope_id": scope_id,
                    "input_items": _items({"type": "message", "role": "user", "content": "hi"}),
                },
            )


async def test_responses_create_response_record_rejects_self_referential_previous_response(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        with pytest.raises(SQLAlchemyError, match="cannot reference itself"):
            await session.execute(
                text(
                    """
                    select responses.create_response_record(
                      :scope_id,
                      'resp_self_prev',
                      'resp_self_prev',
                      cast(:input_items as jsonb),
                      '[]'::jsonb,
                      null
                    )
                    """
                ),
                {
                    "scope_id": scope_id,
                    "input_items": _items({"type": "message", "role": "user", "content": "hi"}),
                },
            )


async def test_responses_append_response_builds_lineage_and_inherits_replay_base(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        _, base_replay = await _append_response(
            session,
            scope_id,
            "resp_1",
            input_items=[{"type": "message", "role": "user", "content": "first"}],
            output_items=[{"type": "message", "role": "assistant", "content": "one"}],
            retention=None,
        )
        _, child_replay = await _append_response(
            session,
            scope_id,
            "resp_2",
            prev_response_id="resp_1",
            input_items=[{"type": "message", "role": "user", "content": "second"}],
            output_items=[{"type": "message", "role": "assistant", "content": "two"}],
            retention=None,
        )
        await session.commit()

        replay_rows = await _replay_rows(session, scope_id, "resp_2")
        counts_1 = await _response_refcounts(session, scope_id, "resp_1")
        counts_2 = await _response_refcounts(session, scope_id, "resp_2")

    assert base_replay == "resp_1"
    assert child_replay == "resp_1"
    assert replay_rows == [
        ("resp_1", "input", 0, "message"),
        ("resp_1", "output", 0, "message"),
        ("resp_2", "input", 0, "message"),
        ("resp_2", "output", 0, "message"),
    ]
    assert counts_1 == (1, 0)
    assert counts_2 == (0, 0)


async def test_responses_compaction_resets_replay_base(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        await _append_response(
            session,
            scope_id,
            "resp_1",
            input_items=[{"type": "message", "role": "user", "content": "first"}],
            output_items=[{"type": "message", "role": "assistant", "content": "one"}],
            retention=None,
        )
        await _append_response(
            session,
            scope_id,
            "resp_2",
            prev_response_id="resp_1",
            input_items=[{"type": "message", "role": "user", "content": "second"}],
            output_items=[{"type": "message", "role": "assistant", "content": "two"}],
            retention=None,
        )
        _, compact_replay = await _append_response(
            session,
            scope_id,
            "resp_3",
            prev_response_id="resp_2",
            input_items=[{"type": "message", "role": "user", "content": "compact"}],
            output_items=[{"type": "compaction", "id": "cmp_1", "created_by": "server"}],
            retention=None,
        )
        _, tail_replay = await _append_response(
            session,
            scope_id,
            "resp_4",
            prev_response_id="resp_3",
            input_items=[{"type": "message", "role": "user", "content": "after compact"}],
            output_items=[{"type": "message", "role": "assistant", "content": "done"}],
            retention=None,
        )
        await session.commit()

        replay_rows = await _replay_rows(session, scope_id, "resp_4")

    assert compact_replay == "resp_3"
    assert tail_replay == "resp_3"
    assert replay_rows == [
        ("resp_3", "input", 0, "message"),
        ("resp_3", "output", 0, "compaction"),
        ("resp_4", "input", 0, "message"),
        ("resp_4", "output", 0, "message"),
    ]


async def test_responses_list_response_replay_orders_input_before_output(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        await _append_response(
            session,
            scope_id,
            "resp_1",
            input_items=[
                {"type": "message", "role": "user", "content": "input a"},
                {"type": "message", "role": "user", "content": "input b"},
            ],
            output_items=[
                {"type": "reasoning", "id": "rs_1"},
                {"type": "message", "role": "assistant", "content": "output a"},
            ],
            retention=None,
        )
        await session.commit()

        replay_payloads = await _replay_payloads(session, scope_id, "resp_1")

    assert replay_payloads == [
        {"type": "message", "role": "user", "content": "input a"},
        {"type": "message", "role": "user", "content": "input b"},
        {"type": "reasoning", "id": "rs_1"},
        {"type": "message", "role": "assistant", "content": "output a"},
    ]


async def test_responses_create_response_record_gets_response_owned_retention_lease(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        await session.execute(
            text(
                """
                select responses.create_response_record(
                  :scope_id,
                  'resp_retained',
                  null,
                  cast(:input_items as jsonb),
                  '[]'::jsonb
                )
                """
            ),
            {
                "scope_id": scope_id,
                "input_items": _items({"type": "message", "role": "user", "content": "retained"}),
            },
        )
        await session.commit()

        lease = (
            await session.execute(
                text(
                    """
                    select owner_type, owner_id, status, expires_at > now() as future
                      from responses.response_leases
                     where scope_id = :scope_id
                       and response_id = 'resp_retained'
                    """
                ),
                {"scope_id": scope_id},
            )
        ).one()

    assert (lease.owner_type, lease.owner_id, lease.status, lease.future) == ("response", "resp_retained", "live", True)


async def test_responses_lease_and_conversation_helpers_move_lineage_ownership(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        await _append_response(
            session,
            scope_id,
            "resp_1",
            input_items=[{"type": "message", "role": "user", "content": "first"}],
            retention=timedelta(days=30),
        )
        await _append_response(
            session,
            scope_id,
            "resp_2",
            prev_response_id="resp_1",
            input_items=[{"type": "message", "role": "user", "content": "second"}],
            retention=timedelta(days=30),
        )

        lease_id = (
            await session.execute(
                text(
                    """
                    select responses.create_or_refresh_response_lease(
                      :scope_id,
                      'resp_1',
                      'manual',
                      'owner_1',
                      now() + interval '1 hour'
                    )
                    """
                ),
                {"scope_id": scope_id},
            )
        ).scalar_one()
        refreshed_lease_id = (
            await session.execute(
                text(
                    """
                    select responses.create_or_refresh_response_lease(
                      :scope_id,
                      'resp_2',
                      'manual',
                      'owner_1',
                      now() + interval '2 hours'
                    )
                    """
                ),
                {"scope_id": scope_id},
            )
        ).scalar_one()
        await session.execute(
            text("select responses.move_conversation_head(:scope_id, 'conv_1', 'resp_1')"),
            {"scope_id": scope_id},
        )
        await session.execute(
            text("select responses.move_conversation_head(:scope_id, 'conv_1', 'resp_2')"),
            {"scope_id": scope_id},
        )
        await session.execute(
            text("select responses.release_response_lease(:scope_id, 'manual', 'owner_1')"),
            {"scope_id": scope_id},
        )
        await session.commit()

        resp_1_counts = await _response_refcounts(session, scope_id, "resp_1")
        resp_2_counts = await _response_refcounts(session, scope_id, "resp_2")
        live_leases = (
            await session.execute(
                text(
                    """
                    select owner_type, owner_id, response_id
                      from responses.response_leases
                     where scope_id = :scope_id
                       and status = 'live'
                     order by owner_type, owner_id
                    """
                ),
                {"scope_id": scope_id},
            )
        ).all()

    assert refreshed_lease_id == lease_id
    assert resp_1_counts == (1, 1)
    assert resp_2_counts == (0, 2)
    assert [(row.owner_type, row.owner_id, row.response_id) for row in live_leases] == [
        ("conversation", "conv_1", "resp_2"),
        ("response", "resp_1", "resp_1"),
        ("response", "resp_2", "resp_2"),
    ]
