from __future__ import annotations

import json
from datetime import timedelta
from uuid import uuid4

from sqlalchemy import text


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
) -> None:
    await session.execute(
        text(
            """
            select responses.append_response(
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


async def _create_payload(session, scope_id, marker: int):
    return (
        await session.execute(
            text(
                """
                select responses.get_or_create_payload(
                  :scope_id,
                  cast(:payload_json as jsonb)
                )
                """
            ),
            {
                "scope_id": scope_id,
                "payload_json": json.dumps({"type": "message", "text": str(marker)}),
            },
        )
    ).scalar_one()


async def _table_count(session, table_name: str, scope_id) -> int:
    return (
        await session.execute(
            text(f"select count(*) from responses.{table_name} where scope_id = :scope_id"),
            {"scope_id": scope_id},
        )
    ).scalar_one()


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
    assert all("responses." in job.command for job in jobs)


async def test_responses_gc_expires_lease_and_deletes_suffix(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        await _append_response(
            session,
            scope_id,
            "resp_1",
            input_items=[{"type": "message", "role": "user", "content": "first"}],
            retention=None,
        )
        await _append_response(
            session,
            scope_id,
            "resp_2",
            prev_response_id="resp_1",
            input_items=[{"type": "message", "role": "user", "content": "second"}],
            retention=None,
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

        await session.execute(text("call responses.gc_expire_response_leases(10)"))
        await session.commit()

        assert await _table_count(session, "response_records", scope_id) == 0
        assert await _table_count(session, "response_input_items", scope_id) == 0
        assert await _table_count(session, "response_output_items", scope_id) == 0
        assert await _table_count(session, "payloads", scope_id) == 0
        assert await _table_count(session, "response_leases", scope_id) == 0


async def test_responses_gc_prunes_expired_conversations(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        await _append_response(
            session,
            scope_id,
            "resp_stale",
            input_items=[{"type": "message", "role": "user", "content": "stale"}],
            retention=None,
        )
        await session.execute(
            text(
                """
                insert into responses.conversations (
                  scope_id,
                  conversation_id,
                  current_response_id,
                  retention_expires_at
                ) values (
                  :scope_id,
                  'conv_stale',
                  'resp_stale',
                  now() - interval '1 hour'
                )
                """
            ),
            {"scope_id": scope_id},
        )
        await session.commit()

        await session.execute(text("call responses.gc_prune_conversations(10)"))
        await session.commit()

        assert await _table_count(session, "conversations", scope_id) == 0
        assert await _table_count(session, "response_leases", scope_id) == 0
        assert await _table_count(session, "response_records", scope_id) == 0
        assert await _table_count(session, "payloads", scope_id) == 0


async def test_responses_gc_prunes_unreferenced_responses(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        await _append_response(
            session,
            scope_id,
            "resp_unreferenced",
            input_items=[{"type": "message", "role": "user", "content": "orphan"}],
            retention=None,
        )
        await session.commit()

        await session.execute(text("call responses.gc_prune_unreferenced_responses(10)"))
        await session.commit()

        assert await _table_count(session, "response_records", scope_id) == 0
        assert await _table_count(session, "payloads", scope_id) == 0


async def test_responses_gc_prune_unreferenced_responses_respects_batch_budget(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        await _append_response(
            session,
            scope_id,
            "resp_budget_1",
            input_items=[{"type": "message", "role": "user", "content": "first"}],
            retention=None,
        )
        await _append_response(
            session,
            scope_id,
            "resp_budget_2",
            prev_response_id="resp_budget_1",
            input_items=[{"type": "message", "role": "user", "content": "second"}],
            retention=None,
        )
        await _append_response(
            session,
            scope_id,
            "resp_budget_3",
            prev_response_id="resp_budget_2",
            input_items=[{"type": "message", "role": "user", "content": "third"}],
            retention=None,
        )
        await session.commit()

        await session.execute(text("call responses.gc_prune_unreferenced_responses(1)"))
        await session.commit()

        assert await _table_count(session, "response_records", scope_id) == 2
        assert await _table_count(session, "payloads", scope_id) == 2

        await session.execute(text("call responses.gc_prune_unreferenced_responses(1)"))
        await session.commit()

        assert await _table_count(session, "response_records", scope_id) == 1
        assert await _table_count(session, "payloads", scope_id) == 1

        await session.execute(text("call responses.gc_prune_unreferenced_responses(1)"))
        await session.commit()

        assert await _table_count(session, "response_records", scope_id) == 0
        assert await _table_count(session, "payloads", scope_id) == 0


async def test_responses_create_response_record_gets_response_owned_retention(db_session_maker) -> None:
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

        await session.execute(text("call responses.gc_prune_unreferenced_responses(10)"))
        await session.commit()
        assert await _table_count(session, "response_records", scope_id) == 1

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

        await session.execute(text("call responses.gc_expire_response_leases(10)"))
        await session.commit()

        assert (lease.owner_type, lease.owner_id, lease.status, lease.future) == ("response", "resp_retained", "live", True)
        assert await _table_count(session, "response_records", scope_id) == 0
        assert await _table_count(session, "payloads", scope_id) == 0


async def test_responses_gc_expire_response_leases_respects_batch_budget(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        await _append_response(
            session,
            scope_id,
            "resp_lease_budget_1",
            input_items=[{"type": "message", "role": "user", "content": "first"}],
            retention=None,
        )
        await _append_response(
            session,
            scope_id,
            "resp_lease_budget_2",
            prev_response_id="resp_lease_budget_1",
            input_items=[{"type": "message", "role": "user", "content": "second"}],
            retention=None,
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
                  'resp_lease_budget_2',
                  'manual',
                  'lease_budget_1',
                  now() - interval '1 hour'
                )
                """
            ),
            {"scope_id": scope_id},
        )
        await session.commit()

        await session.execute(text("call responses.gc_expire_response_leases(1)"))
        await session.commit()

        assert await _table_count(session, "response_records", scope_id) == 1
        assert await _table_count(session, "payloads", scope_id) == 1
        assert await _table_count(session, "response_leases", scope_id) == 0

        await session.execute(text("call responses.gc_prune_unreferenced_responses(1)"))
        await session.commit()

        assert await _table_count(session, "response_records", scope_id) == 0
        assert await _table_count(session, "payloads", scope_id) == 0


async def test_responses_conversation_lease_survives_response_retention_expiry(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        await session.execute(
            text(
                """
                select responses.create_response_record(
                  :scope_id,
                  'resp_active_conversation',
                  null,
                  cast(:input_items as jsonb),
                  '[]'::jsonb
                )
                """
            ),
            {
                "scope_id": scope_id,
                "input_items": _items({"type": "message", "role": "user", "content": "active"}),
            },
        )
        await session.execute(
            text(
                """
                select responses.move_conversation_head(
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

        await session.execute(text("call responses.gc_expire_response_leases(10)"))
        await session.commit()

        assert await _table_count(session, "conversations", scope_id) == 1
        assert await _table_count(session, "response_leases", scope_id) == 1
        assert await _table_count(session, "response_records", scope_id) == 1

        await session.execute(
            text(
                """
                update responses.conversations
                   set retention_expires_at = now() - interval '1 hour'
                 where scope_id = :scope_id
                   and conversation_id = 'conv_active'
                """
            ),
            {"scope_id": scope_id},
        )
        await session.execute(text("call responses.gc_prune_conversations(10)"))
        await session.commit()

        assert await _table_count(session, "conversations", scope_id) == 0
        assert await _table_count(session, "response_leases", scope_id) == 0
        assert await _table_count(session, "response_records", scope_id) == 0
        assert await _table_count(session, "payloads", scope_id) == 0


async def test_responses_gc_ignores_indefinitely_retained_conversations(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        await _append_response(
            session,
            scope_id,
            "resp_indefinite",
            input_items=[{"type": "message", "role": "user", "content": "indefinite"}],
            retention=None,
        )
        await session.execute(
            text(
                """
                select responses.move_conversation_head(
                  :scope_id,
                  'conv_indefinite',
                  'resp_indefinite',
                  null
                )
                """
            ),
            {"scope_id": scope_id},
        )
        await session.commit()

        retention_expires_at = (
            await session.execute(
                text(
                    """
                    select retention_expires_at
                      from responses.conversations
                     where scope_id = :scope_id
                       and conversation_id = 'conv_indefinite'
                    """
                ),
                {"scope_id": scope_id},
            )
        ).scalar_one()

        await session.execute(text("call responses.gc_prune_conversations(10)"))
        await session.commit()

        assert retention_expires_at is None
        assert await _table_count(session, "conversations", scope_id) == 1
        assert await _table_count(session, "response_leases", scope_id) == 1
        assert await _table_count(session, "response_records", scope_id) == 1


async def test_responses_response_owned_head_lease_retains_previous_chain(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        await session.execute(
            text(
                """
                select responses.create_response_record(
                  :scope_id,
                  'resp_chain_1',
                  null,
                  cast(:input_items as jsonb),
                  '[]'::jsonb,
                  null
                )
                """
            ),
            {
                "scope_id": scope_id,
                "input_items": _items({"type": "message", "role": "user", "content": "chain 1"}),
            },
        )
        await session.execute(
            text(
                """
                select responses.create_response_record(
                  :scope_id,
                  'resp_chain_2',
                  'resp_chain_1',
                  cast(:input_items as jsonb),
                  '[]'::jsonb
                )
                """
            ),
            {
                "scope_id": scope_id,
                "input_items": _items({"type": "message", "role": "user", "content": "chain 2"}),
            },
        )
        await session.commit()

        await session.execute(text("call responses.gc_prune_unreferenced_responses(10)"))
        await session.commit()
        assert await _table_count(session, "response_records", scope_id) == 2

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
        await session.execute(text("call responses.gc_expire_response_leases(10)"))
        await session.commit()

        assert await _table_count(session, "response_records", scope_id) == 0
        assert await _table_count(session, "payloads", scope_id) == 0


async def test_responses_gc_prunes_zero_ref_payloads(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        await _create_payload(session, scope_id, 1)
        await session.commit()

        await session.execute(text("call responses.gc_prune_payloads(10)"))
        await session.commit()

        assert await _table_count(session, "payloads", scope_id) == 0
