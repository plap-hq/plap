from __future__ import annotations

import json
from uuid import uuid4

import blake3
import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError


def _canonical_payload_hash(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return blake3.blake3(canonical.encode()).hexdigest()


def _items(*payloads: dict[str, object], start_ord: int = 0) -> str:
    return json.dumps(
        [
            {
                "namespace": "m",
                "ord": start_ord + index,
                "payload_hash": _canonical_payload_hash(payload),
                "payload": payload,
            }
            for index, payload in enumerate(payloads)
        ]
    )


async def _append_response(
    session,
    scope_id,
    response_id: str,
    payload: dict[str, object],
    *,
    prev_response_id: str | None = None,
    next_ord: int = 1,
) -> int:
    row = (
        await session.execute(
            text(
                """
                select root_node_id
                  from response_state.append_items(
                    :scope_id,
                    :response_id,
                    :prev_response_id,
                    cast(:items as jsonb),
                    cast(:namespace_counters as jsonb),
                    '[]'::jsonb
                  )
                """
            ),
            {
                "scope_id": scope_id,
                "response_id": response_id,
                "prev_response_id": prev_response_id,
                "items": _items(payload),
                "namespace_counters": json.dumps(
                    [{"namespace": "m", "next_ord": next_ord}]
                ),
            },
        )
    ).one()
    return row.root_node_id


async def _response_refcounts(session, scope_id, response_id: str) -> tuple[int, int]:
    row = (
        await session.execute(
            text(
                """
                select child_refcount, lease_refcount
                  from responses
                 where scope_id = :scope_id
                   and response_id = :response_id
                """
            ),
            {"scope_id": scope_id, "response_id": response_id},
        )
    ).one()
    return row.child_refcount, row.lease_refcount


@pytest.mark.asyncio
async def test_response_state_payload_dedupe_and_collision_guard(
    db_session_maker,
) -> None:
    scope_id = uuid4()
    payload = {"type": "message", "text": "hello"}
    payload_hash = bytes.fromhex(_canonical_payload_hash(payload))

    async with db_session_maker() as session:
        first_id = (
            await session.execute(
                text(
                    """
                    select response_state.get_or_create_payload(
                      :scope_id,
                      :payload_hash,
                      cast(:payload_json as jsonb)
                    )
                    """
                ),
                {
                    "scope_id": scope_id,
                    "payload_hash": payload_hash,
                    "payload_json": json.dumps(payload),
                },
            )
        ).scalar_one()
        second_id = (
            await session.execute(
                text(
                    """
                    select response_state.get_or_create_payload(
                      :scope_id,
                      :payload_hash,
                      cast(:payload_json as jsonb)
                    )
                    """
                ),
                {
                    "scope_id": scope_id,
                    "payload_hash": payload_hash,
                    "payload_json": json.dumps(payload),
                },
            )
        ).scalar_one()

        assert second_id == first_id

        with pytest.raises(SQLAlchemyError, match="non-canonical payload hash"):
            await session.execute(
                text(
                    """
                    select response_state.get_or_create_payload(
                      :scope_id,
                      :payload_hash,
                      cast(:payload_json as jsonb)
                    )
                    """
                ),
                {
                    "scope_id": scope_id,
                    "payload_hash": payload_hash,
                    "payload_json": json.dumps({"type": "message", "text": "bye"}),
                },
            )


@pytest.mark.asyncio
async def test_response_state_create_leaf_and_concat_update_refcounts(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        left_id = (
            await session.execute(
                text(
                    """
                    select response_state.create_leaf(:scope_id, cast(:items as jsonb))
                    """
                ),
                {
                    "scope_id": scope_id,
                    "items": _items({"type": "message", "text": "left"}),
                },
            )
        ).scalar_one()
        right_id = (
            await session.execute(
                text(
                    """
                    select response_state.create_leaf(:scope_id, cast(:items as jsonb))
                    """
                ),
                {
                    "scope_id": scope_id,
                    "items": _items({"type": "message", "text": "right"}),
                },
            )
        ).scalar_one()
        concat_id = (
            await session.execute(
                text(
                    """
                    select response_state.create_concat(
                      :scope_id,
                      :left_id,
                      :right_id
                    )
                    """
                ),
                {"scope_id": scope_id, "left_id": left_id, "right_id": right_id},
            )
        ).scalar_one()
        await session.commit()

        rows = (
            await session.execute(
                text(
                    """
                    select node_id, kind, item_count, refcount
                      from state_nodes
                     where scope_id = :scope_id
                       and node_id in (:left_id, :right_id, :concat_id)
                     order by node_id
                    """
                ),
                {
                    "scope_id": scope_id,
                    "left_id": left_id,
                    "right_id": right_id,
                    "concat_id": concat_id,
                },
            )
        ).all()

    assert [(row.kind, row.item_count, row.refcount) for row in rows] == [
        ("leaf", 1, 1),
        ("leaf", 1, 1),
        ("concat", 2, 0),
    ]


@pytest.mark.asyncio
async def test_response_state_append_items_creates_response_chain(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        first_root = await _append_response(
            session,
            scope_id,
            "resp_1",
            {"type": "message", "text": "first"},
        )
        second_root = await _append_response(
            session,
            scope_id,
            "resp_2",
            {"type": "message", "text": "second"},
            prev_response_id="resp_1",
            next_ord=2,
        )
        await session.commit()

        first_counts = await _response_refcounts(session, scope_id, "resp_1")
        second_counts = await _response_refcounts(session, scope_id, "resp_2")
        second_root_kind = (
            await session.execute(
                text(
                    """
                    select kind
                      from state_nodes
                     where scope_id = :scope_id
                       and node_id = :node_id
                    """
                ),
                {"scope_id": scope_id, "node_id": second_root},
            )
        ).scalar_one()
        counters = (
            await session.execute(
                text(
                    """
                    select next_ord
                      from response_namespace_counters
                     where scope_id = :scope_id
                       and response_id = 'resp_2'
                    """
                ),
                {"scope_id": scope_id},
            )
        ).scalar_one()

    assert first_root != second_root
    assert first_counts == (1, 0)
    assert second_counts == (0, 0)
    assert second_root_kind == "concat"
    assert counters == 2


@pytest.mark.asyncio
async def test_response_state_lease_and_conversation_helpers_move_roots(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        await _append_response(
            session,
            scope_id,
            "resp_1",
            {"type": "message", "text": "first"},
        )
        await _append_response(
            session,
            scope_id,
            "resp_2",
            {"type": "message", "text": "second"},
            prev_response_id="resp_1",
            next_ord=2,
        )

        lease_id = (
            await session.execute(
                text(
                    """
                    select response_state.create_or_refresh_lease(
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
                    select response_state.create_or_refresh_lease(
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
            text(
                """
                select response_state.move_conversation(
                  :scope_id,
                  'conv_1',
                  'resp_1'
                )
                """
            ),
            {"scope_id": scope_id},
        )
        await session.execute(
            text(
                """
                select response_state.move_conversation(
                  :scope_id,
                  'conv_1',
                  'resp_2'
                )
                """
            ),
            {"scope_id": scope_id},
        )
        await session.execute(
            text(
                """
                select response_state.release_lease(
                  :scope_id,
                  'manual',
                  'owner_1'
                )
                """
            ),
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
                      from response_leases
                     where scope_id = :scope_id
                       and status = 'live'
                     order by owner_type, owner_id
                    """
                ),
                {"scope_id": scope_id},
            )
        ).all()

    assert refreshed_lease_id == lease_id
    assert resp_1_counts == (1, 0)
    assert resp_2_counts == (0, 1)
    assert [(row.owner_type, row.owner_id, row.response_id) for row in live_leases] == [
        ("conversation", "conv_1", "resp_2")
    ]
