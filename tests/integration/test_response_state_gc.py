from __future__ import annotations

import json
from uuid import uuid4

import pytest
from sqlalchemy import text


async def _create_payload(session, scope_id, marker: int):
    return (
        await session.execute(
            text(
                """
                insert into response_state.payload_objects (
                  scope_id,
                  payload_hash,
                  payload_json
                ) values (
                  :scope_id,
                  :payload_hash,
                  cast(:payload_json as jsonb)
                )
                returning payload_id
                """
            ),
            {
                "scope_id": scope_id,
                "payload_hash": bytes([marker]) * 32,
                "payload_json": json.dumps({"type": "message", "text": str(marker)}),
            },
        )
    ).scalar_one()


async def _create_leaf(session, scope_id, marker: int) -> int:
    payload_id = await _create_payload(session, scope_id, marker)
    namespace_id = (
        await session.execute(
            text(
                """
                select namespace_id
                  from response_state.ordinal_namespaces
                 where namespace_name = 'm'
                """
            )
        )
    ).scalar_one()
    node_id = (
        await session.execute(
            text(
                """
                insert into response_state.state_nodes (scope_id, kind, item_count)
                values (:scope_id, 'leaf', 1)
                returning node_id
                """
            ),
            {"scope_id": scope_id},
        )
    ).scalar_one()
    await session.execute(
        text(
            """
            insert into response_state.state_leaves (scope_id, node_id, entry_count)
            values (:scope_id, :node_id, 1)
            """
        ),
        {"scope_id": scope_id, "node_id": node_id},
    )
    await session.execute(
        text(
            """
            insert into response_state.state_leaf_entries (
              scope_id,
              node_id,
              pos,
              namespace_id,
              ord,
              payload_id
            ) values (
              :scope_id,
              :node_id,
              0,
              :namespace_id,
              0,
              :payload_id
            )
            """
        ),
        {
            "scope_id": scope_id,
            "node_id": node_id,
            "namespace_id": namespace_id,
            "payload_id": payload_id,
        },
    )
    return node_id


async def _create_response(
    session,
    scope_id,
    response_id: str,
    root_id: int,
    *,
    prev_response_id: str | None = None,
) -> None:
    await session.execute(
        text(
            """
            insert into response_state.responses (
              scope_id,
              response_id,
              prev_response_id,
              full_state_root_id
            ) values (
              :scope_id,
              :response_id,
              :prev_response_id,
              :root_id
            )
            """
        ),
        {
            "scope_id": scope_id,
            "response_id": response_id,
            "prev_response_id": prev_response_id,
            "root_id": root_id,
        },
    )


async def _table_count(session, table_name: str, scope_id) -> int:
    return (
        await session.execute(
            text(
                f"select count(*) from response_state.{table_name} "
                "where scope_id = :scope_id"
            ),
            {"scope_id": scope_id},
        )
    ).scalar_one()


@pytest.mark.asyncio
async def test_response_state_gc_registers_cron_jobs(db_session_maker) -> None:
    async with db_session_maker() as session:
        jobs = (
            await session.execute(
                text(
                    """
                        select jobname, command
                          from cron.job
                         where jobname in (
                           'expire-response-leases',
                           'prune-stale-conversations',
                           'prune-unreferenced-responses',
                           'prune-zero-ref-payloads'
                         )
                         order by jobname
                        """
                )
            )
        ).all()

    assert [job.jobname for job in jobs] == [
        "expire-response-leases",
        "prune-stale-conversations",
        "prune-unreferenced-responses",
        "prune-zero-ref-payloads",
    ]
    assert all("response_state." in job.command for job in jobs)


@pytest.mark.asyncio
async def test_response_state_gc_expires_lease_and_deletes_suffix(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        first_root = await _create_leaf(session, scope_id, 1)
        second_root = await _create_leaf(session, scope_id, 2)
        await _create_response(session, scope_id, "resp_1", first_root)
        await _create_response(
            session,
            scope_id,
            "resp_2",
            second_root,
            prev_response_id="resp_1",
        )
        await session.execute(
            text(
                """
                insert into response_state.response_leases (
                  scope_id,
                  response_id,
                  owner_type,
                  owner_id,
                  expires_at
                ) values (
                  :scope_id,
                  'resp_2',
                  'manual',
                  'lease_1',
                  now() - interval '1 hour'
                )
                """
            ),
            {"scope_id": scope_id},
        )
        await session.commit()

        await session.execute(text("call response_state.gc_expire_leases(10)"))
        await session.commit()

        assert await _table_count(session, "responses", scope_id) == 0
        assert await _table_count(session, "state_nodes", scope_id) == 0
        assert await _table_count(session, "payload_objects", scope_id) == 0
        assert await _table_count(session, "response_leases", scope_id) == 0


@pytest.mark.asyncio
async def test_response_state_gc_prunes_stale_conversations(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        root_id = await _create_leaf(session, scope_id, 1)
        await _create_response(session, scope_id, "resp_stale", root_id)
        await session.execute(
            text(
                """
                insert into response_state.conversations (
                  scope_id,
                  conversation_id,
                  current_response_id,
                  last_used_at
                ) values (
                  :scope_id,
                  'conv_stale',
                  'resp_stale',
                  now() - interval '31 days'
                )
                """
            ),
            {"scope_id": scope_id},
        )
        await session.commit()

        await session.execute(
            text("call response_state.gc_prune_conversations(10, interval '30 days')")
        )
        await session.commit()

        assert await _table_count(session, "conversations", scope_id) == 0
        assert await _table_count(session, "response_leases", scope_id) == 0
        assert await _table_count(session, "responses", scope_id) == 0
        assert await _table_count(session, "state_nodes", scope_id) == 0


@pytest.mark.asyncio
async def test_response_state_gc_prunes_unreferenced_responses(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        root_id = await _create_leaf(session, scope_id, 1)
        await _create_response(session, scope_id, "resp_unreferenced", root_id)
        await session.commit()

        await session.execute(
            text("call response_state.gc_prune_unreferenced_responses(10)")
        )
        await session.commit()

        assert await _table_count(session, "responses", scope_id) == 0
        assert await _table_count(session, "state_nodes", scope_id) == 0
        assert await _table_count(session, "payload_objects", scope_id) == 0


@pytest.mark.asyncio
async def test_response_state_gc_prunes_zero_ref_payloads(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        await _create_payload(session, scope_id, 1)
        await session.commit()

        await session.execute(text("call response_state.gc_prune_payloads(10)"))
        await session.commit()

        assert await _table_count(session, "payload_objects", scope_id) == 0
