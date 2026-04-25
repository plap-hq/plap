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
                insert into responses.payload_objects (
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
                  from responses.ordinal_namespaces
                 where namespace_name = 'm'
                """
            )
        )
    ).scalar_one()
    node_id = (
        await session.execute(
            text(
                """
                insert into responses.state_nodes (
                  scope_id,
                  kind,
                  height,
                  item_count,
                  child_count
                ) values (
                  :scope_id,
                  'leaf',
                  0,
                  1,
                  0
                )
                returning node_id
                """
            ),
            {"scope_id": scope_id},
        )
    ).scalar_one()
    await session.execute(
        text(
            """
            insert into responses.state_leaves (scope_id, node_id, entry_count)
            values (:scope_id, :node_id, 1)
            """
        ),
        {"scope_id": scope_id, "node_id": node_id},
    )
    await session.execute(
        text(
            """
            insert into responses.state_leaf_entries (
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
            insert into responses.responses (
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


def _namespace_counters(message_next_ord: int, summary_next_ord: int = 0) -> str:
    return json.dumps(
        [
            {"namespace": "m", "next_ord": message_next_ord},
            {"namespace": "s", "next_ord": summary_next_ord},
        ]
    )


async def _table_count(session, table_name: str, scope_id) -> int:
    return (
        await session.execute(
            text(
                f"select count(*) from responses.{table_name} "
                "where scope_id = :scope_id"
            ),
            {"scope_id": scope_id},
        )
    ).scalar_one()


@pytest.mark.asyncio
async def test_responses_gc_registers_cron_jobs(db_session_maker) -> None:
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
                            'prune-zero-ref-payloads',
                            'prune-zero-ref-state-nodes'
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
        "prune-zero-ref-state-nodes",
    ]
    assert all("responses." in job.command for job in jobs)


@pytest.mark.asyncio
async def test_responses_gc_expires_lease_and_deletes_suffix(
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
                insert into responses.response_leases (
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

        await session.execute(text("call responses.gc_expire_leases(10)"))
        await session.commit()

        assert await _table_count(session, "responses", scope_id) == 0
        assert await _table_count(session, "state_nodes", scope_id) == 0
        assert await _table_count(session, "payload_objects", scope_id) == 0
        assert await _table_count(session, "response_leases", scope_id) == 0


@pytest.mark.asyncio
async def test_responses_gc_prunes_stale_conversations(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        root_id = await _create_leaf(session, scope_id, 1)
        await _create_response(session, scope_id, "resp_stale", root_id)
        await session.execute(
            text(
                """
                insert into responses.conversations (
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
            text("call responses.gc_prune_conversations(10, interval '30 days')")
        )
        await session.commit()

        assert await _table_count(session, "conversations", scope_id) == 0
        assert await _table_count(session, "response_leases", scope_id) == 0
        assert await _table_count(session, "responses", scope_id) == 0
        assert await _table_count(session, "state_nodes", scope_id) == 0


@pytest.mark.asyncio
async def test_responses_gc_prunes_unreferenced_responses(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        root_id = await _create_leaf(session, scope_id, 1)
        await _create_response(session, scope_id, "resp_unreferenced", root_id)
        await session.commit()

        await session.execute(
            text("call responses.gc_prune_unreferenced_responses(10)")
        )
        await session.commit()

        assert await _table_count(session, "responses", scope_id) == 0
        assert await _table_count(session, "state_nodes", scope_id) == 0
        assert await _table_count(session, "payload_objects", scope_id) == 0


@pytest.mark.asyncio
async def test_responses_created_via_mechanics_get_response_owned_retention(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        root_id = await _create_leaf(session, scope_id, 1)
        await session.execute(
            text(
                """
                select responses.create_response(
                  :scope_id,
                  'resp_retained',
                  null,
                  :root_id,
                  cast(:namespace_counters as jsonb),
                  '[]'::jsonb
                )
                """
            ),
            {
                "scope_id": scope_id,
                "root_id": root_id,
                "namespace_counters": _namespace_counters(1),
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
        assert (lease.owner_type, lease.owner_id, lease.status, lease.future) == (
            "response",
            "resp_retained",
            "live",
            True,
        )

        await session.execute(
            text("call responses.gc_prune_unreferenced_responses(10)")
        )
        await session.commit()
        assert await _table_count(session, "responses", scope_id) == 1

        await session.execute(
            text(
                """
                update responses.response_leases
                   set expires_at = now() - interval '1 hour'
                 where scope_id = :scope_id
                   and response_id = 'resp_retained'
                """
            ),
            {"scope_id": scope_id},
        )
        await session.commit()

        assert await _table_count(session, "response_leases", scope_id) == 1
        assert await _table_count(session, "responses", scope_id) == 1
        assert await _table_count(session, "state_nodes", scope_id) == 1
        assert await _table_count(session, "payload_objects", scope_id) == 1

        await session.execute(text("call responses.gc_expire_leases(10)"))
        await session.commit()

        assert await _table_count(session, "responses", scope_id) == 0
        assert await _table_count(session, "state_nodes", scope_id) == 0
        assert await _table_count(session, "payload_objects", scope_id) == 0


@pytest.mark.asyncio
async def test_responses_conversation_lease_survives_response_retention_expiry(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        root_id = await _create_leaf(session, scope_id, 1)
        await session.execute(
            text(
                """
                select responses.create_response(
                  :scope_id,
                  'resp_active_conversation',
                  null,
                  :root_id,
                  cast(:namespace_counters as jsonb),
                  '[]'::jsonb
                )
                """
            ),
            {
                "scope_id": scope_id,
                "root_id": root_id,
                "namespace_counters": _namespace_counters(1),
            },
        )
        await session.execute(
            text(
                """
                select responses.move_conversation(
                  :scope_id,
                  'conv_active',
                  'resp_active_conversation'
                )
                """
            ),
            {"scope_id": scope_id},
        )
        await session.execute(
            text(
                """
                update responses.response_leases
                   set expires_at = now() - interval '1 hour'
                 where scope_id = :scope_id
                   and response_id = 'resp_active_conversation'
                   and owner_type = 'response'
                """
            ),
            {"scope_id": scope_id},
        )
        await session.commit()

        await session.execute(text("call responses.gc_expire_leases(10)"))
        await session.commit()

        assert await _table_count(session, "conversations", scope_id) == 1
        assert await _table_count(session, "response_leases", scope_id) == 1
        assert await _table_count(session, "responses", scope_id) == 1
        assert await _table_count(session, "state_nodes", scope_id) == 1

        await session.execute(
            text(
                """
                update responses.conversations
                   set last_used_at = now() - interval '31 days'
                 where scope_id = :scope_id
                   and conversation_id = 'conv_active'
                """
            ),
            {"scope_id": scope_id},
        )
        await session.execute(
            text("call responses.gc_prune_conversations(10, interval '30 days')")
        )
        await session.commit()

        assert await _table_count(session, "conversations", scope_id) == 0
        assert await _table_count(session, "response_leases", scope_id) == 0
        assert await _table_count(session, "responses", scope_id) == 0
        assert await _table_count(session, "state_nodes", scope_id) == 0
        assert await _table_count(session, "payload_objects", scope_id) == 0


@pytest.mark.asyncio
async def test_responses_response_owned_head_lease_retains_previous_chain(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        first_root = await _create_leaf(session, scope_id, 1)
        second_root = await _create_leaf(session, scope_id, 2)
        await session.execute(
            text(
                """
                select responses.create_response(
                  :scope_id,
                  'resp_chain_1',
                  null,
                  :root_id,
                  cast(:namespace_counters as jsonb),
                  '[]'::jsonb,
                  null
                )
                """
            ),
            {
                "scope_id": scope_id,
                "root_id": first_root,
                "namespace_counters": _namespace_counters(1),
            },
        )
        await session.execute(
            text(
                """
                select responses.create_response(
                  :scope_id,
                  'resp_chain_2',
                  'resp_chain_1',
                  :root_id,
                  cast(:namespace_counters as jsonb),
                  '[]'::jsonb
                )
                """
            ),
            {
                "scope_id": scope_id,
                "root_id": second_root,
                "namespace_counters": _namespace_counters(2),
            },
        )
        await session.commit()

        await session.execute(
            text("call responses.gc_prune_unreferenced_responses(10)")
        )
        await session.commit()
        assert await _table_count(session, "responses", scope_id) == 2

        await session.execute(
            text(
                """
                update responses.response_leases
                   set expires_at = now() - interval '1 hour'
                 where scope_id = :scope_id
                   and response_id = 'resp_chain_2'
                """
            ),
            {"scope_id": scope_id},
        )
        await session.execute(text("call responses.gc_expire_leases(10)"))
        await session.commit()

        assert await _table_count(session, "responses", scope_id) == 0
        assert await _table_count(session, "state_nodes", scope_id) == 0
        assert await _table_count(session, "payload_objects", scope_id) == 0


@pytest.mark.asyncio
async def test_responses_gc_prunes_zero_ref_payloads(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        await _create_payload(session, scope_id, 1)
        await session.commit()

        await session.execute(text("call responses.gc_prune_payloads(10)"))
        await session.commit()

        assert await _table_count(session, "payload_objects", scope_id) == 0


@pytest.mark.asyncio
async def test_responses_gc_prunes_unattached_state_nodes(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        left_id = await _create_leaf(session, scope_id, 1)
        right_id = await _create_leaf(session, scope_id, 2)
        internal_id = (
            await session.execute(
                text(
                    """
                    select responses.create_internal_node(
                      :scope_id,
                      array[:left_id, :right_id]::bigint[]
                    )
                    """
                ),
                {"scope_id": scope_id, "left_id": left_id, "right_id": right_id},
            )
        ).scalar_one()
        await session.commit()

        assert await _table_count(session, "state_nodes", scope_id) == 3
        assert await _table_count(session, "payload_objects", scope_id) == 2

        await session.execute(text("call responses.gc_prune_state_nodes(10)"))
        await session.commit()

        assert internal_id is not None
        assert await _table_count(session, "state_nodes", scope_id) == 0
        assert await _table_count(session, "payload_objects", scope_id) == 0
