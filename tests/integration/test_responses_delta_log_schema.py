from __future__ import annotations

import json
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError


async def test_responses_gc_and_fk_indexes_are_shaped_for_deletes(db_session_maker) -> None:
    expected_indexes = {
        "ix_payloads_gc": "(created_at, scope_id, payload_id)",
        "ix_response_records_prev": "(scope_id, prev_response_id)",
        "ix_response_records_replay_base": "(scope_id, replay_base_response_id, response_id)",
        "ix_response_records_gc": "(created_at, scope_id, response_id)",
        "ix_response_input_items_payload_lookup": "(scope_id, payload_id, response_id, input_index)",
        "ix_response_output_items_payload_lookup": "(scope_id, payload_id, response_id, output_index)",
        "ix_response_leases_expiration": "(expires_at, scope_id, lease_id)",
        "ix_response_leases_response": "(scope_id, response_id, lease_id)",
        "ix_conversations_last_used_at": "(last_used_at, scope_id, conversation_id)",
        "ix_conversations_current_response": "(scope_id, current_response_id, conversation_id)",
    }

    async with db_session_maker() as session:
        index_definitions = dict(
            (
                await session.execute(
                    text(
                        """
                        select indexname, indexdef
                          from pg_indexes
                         where schemaname = 'responses'
                           and indexname = any(:index_names)
                        """
                    ),
                    {"index_names": list(expected_indexes)},
                )
            ).all()
        )

        old_tables = [
            (
                await session.execute(
                    text("select to_regclass(:name)"),
                    {"name": name},
                )
            ).scalar_one()
            for name in [
                "responses.state_nodes",
                "responses.state_node_children",
                "responses.state_leaves",
                "responses.state_leaf_entries",
                "responses.response_checkpoints",
                "responses.checkpoint_namespace_cursors",
            ]
        ]

    assert index_definitions.keys() == expected_indexes.keys()
    for index_name, column_order in expected_indexes.items():
        assert column_order in index_definitions[index_name]
    assert old_tables == [None, None, None, None, None, None]


async def test_responses_triggers_update_payload_and_lease_refcounts(db_session_maker) -> None:
    scope_id = uuid4()
    shared_item = json.dumps({"type": "message", "role": "user", "content": "hello"})

    async with db_session_maker() as session:
        await session.execute(
            text(
                """
                select responses.create_response_record(
                  :scope_id,
                  'resp_valid',
                  null,
                  cast(:input_items as jsonb),
                  cast(:output_items as jsonb)
                )
                """
            ),
            {
                "scope_id": scope_id,
                "input_items": json.dumps([json.loads(shared_item)]),
                "output_items": json.dumps([json.loads(shared_item)]),
            },
        )
        await session.execute(
            text(
                """
                insert into responses.conversations (
                  scope_id,
                  conversation_id,
                  current_response_id
                ) values (
                  :scope_id,
                  'conv_valid',
                  'resp_valid'
                )
                """
            ),
            {"scope_id": scope_id},
        )
        await session.commit()

        payload_refcount = (
            await session.execute(
                text(
                    """
                    select refcount
                      from responses.payloads
                     where scope_id = :scope_id
                    """
                ),
                {"scope_id": scope_id},
            )
        ).scalar_one()
        lease_refcount = (
            await session.execute(
                text(
                    """
                    select lease_refcount
                      from responses.response_records
                     where scope_id = :scope_id
                       and response_id = 'resp_valid'
                    """
                ),
                {"scope_id": scope_id},
            )
        ).scalar_one()

    assert payload_refcount == 2
    assert lease_refcount == 2


async def test_responses_rejects_response_identity_updates(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        await session.execute(
            text(
                """
                select responses.create_response_record(
                  :scope_id,
                  'resp_identity',
                  null,
                  cast(:input_items as jsonb),
                  '[]'::jsonb,
                  null
                )
                """
            ),
            {
                "scope_id": scope_id,
                "input_items": json.dumps([{"type": "message", "role": "user", "content": "hello"}]),
            },
        )
        await session.commit()

        with pytest.raises(SQLAlchemyError, match="structurally immutable"):
            await session.execute(
                text(
                    """
                    update responses.response_records
                       set replay_base_response_id = 'resp_other'
                     where scope_id = :scope_id
                       and response_id = 'resp_identity'
                    """
                ),
                {"scope_id": scope_id},
            )


async def test_responses_rejects_conversation_identity_updates(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        await session.execute(
            text(
                """
                select responses.create_response_record(
                  :scope_id,
                  'resp_conversation_identity',
                  null,
                  cast(:input_items as jsonb),
                  '[]'::jsonb,
                  null
                )
                """
            ),
            {
                "scope_id": scope_id,
                "input_items": json.dumps([{"type": "message", "role": "user", "content": "hello"}]),
            },
        )
        await session.execute(
            text(
                """
                insert into responses.conversations (
                  scope_id,
                  conversation_id,
                  current_response_id
                ) values (
                  :scope_id,
                  'conv_identity',
                  'resp_conversation_identity'
                )
                """
            ),
            {"scope_id": scope_id},
        )
        await session.commit()

        with pytest.raises(SQLAlchemyError, match="structurally immutable"):
            await session.execute(
                text(
                    """
                    update responses.conversations
                       set conversation_id = 'conv_renamed'
                     where scope_id = :scope_id
                       and conversation_id = 'conv_identity'
                    """
                ),
                {"scope_id": scope_id},
            )
