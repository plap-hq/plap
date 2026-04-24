from __future__ import annotations

import json
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError


@pytest.mark.asyncio
async def test_response_state_seeds_core_ordinal_namespaces(db_session_maker) -> None:
    async with db_session_maker() as session:
        namespace_names = (
            (
                await session.execute(
                    text(
                        """
                    select namespace_name
                      from ordinal_namespaces
                     where namespace_name in ('m', 's')
                     order by namespace_name
                    """
                    )
                )
            )
            .scalars()
            .all()
        )

    assert namespace_names == ["m", "s"]


async def _create_payload(session, scope_id) -> str:
    return (
        await session.execute(
            text(
                """
                insert into payload_objects (
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
                "payload_hash": b"x" * 32,
                "payload_json": json.dumps({"type": "message", "text": "hello"}),
            },
        )
    ).scalar_one()


async def _create_leaf(session, scope_id, payload_id) -> int:
    namespace_id = (
        await session.execute(
            text(
                """
                insert into ordinal_namespaces (namespace_name)
                values ('message')
                on conflict (namespace_name) do update
                  set namespace_name = excluded.namespace_name
                returning namespace_id
                """
            )
        )
    ).scalar_one()

    node_id = (
        await session.execute(
            text(
                """
                insert into state_nodes (scope_id, kind, item_count)
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
            insert into state_leaves (scope_id, node_id, entry_count)
            values (:scope_id, :node_id, 1)
            """
        ),
        {"scope_id": scope_id, "node_id": node_id},
    )
    await session.execute(
        text(
            """
            insert into state_leaf_entries (
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


@pytest.mark.asyncio
async def test_response_state_triggers_update_refcounts_and_conversation_lease(
    db_session_maker,
) -> None:
    scope_id = uuid4()
    response_id = "resp_valid"
    conversation_id = "conv_valid"

    async with db_session_maker() as session:
        payload_id = await _create_payload(session, scope_id)
        node_id = await _create_leaf(session, scope_id, payload_id)
        await session.execute(
            text(
                """
                insert into responses (
                  scope_id,
                  response_id,
                  full_state_root_id
                ) values (
                  :scope_id,
                  :response_id,
                  :node_id
                )
                """
            ),
            {
                "scope_id": scope_id,
                "response_id": response_id,
                "node_id": node_id,
            },
        )
        await session.execute(
            text(
                """
                insert into conversations (
                  scope_id,
                  conversation_id,
                  current_response_id
                ) values (
                  :scope_id,
                  :conversation_id,
                  :response_id
                )
                """
            ),
            {
                "scope_id": scope_id,
                "conversation_id": conversation_id,
                "response_id": response_id,
            },
        )
        await session.commit()

        payload_refcount = (
            await session.execute(
                text(
                    """
                    select refcount
                      from payload_objects
                     where scope_id = :scope_id
                       and payload_id = :payload_id
                    """
                ),
                {"scope_id": scope_id, "payload_id": payload_id},
            )
        ).scalar_one()
        root_refcount = (
            await session.execute(
                text(
                    """
                    select refcount
                      from state_nodes
                     where scope_id = :scope_id
                       and node_id = :node_id
                    """
                ),
                {"scope_id": scope_id, "node_id": node_id},
            )
        ).scalar_one()
        lease_refcount = (
            await session.execute(
                text(
                    """
                    select lease_refcount
                      from responses
                     where scope_id = :scope_id
                       and response_id = :response_id
                    """
                ),
                {"scope_id": scope_id, "response_id": response_id},
            )
        ).scalar_one()

    assert payload_refcount == 1
    assert root_refcount == 1
    assert lease_refcount == 1


@pytest.mark.asyncio
async def test_response_state_rejects_sparse_leaf_positions(db_session_maker) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        payload_id = await _create_payload(session, scope_id)
        namespace_id = (
            await session.execute(
                text(
                    """
                    insert into ordinal_namespaces (namespace_name)
                    values ('message')
                    returning namespace_id
                    """
                )
            )
        ).scalar_one()
        node_id = (
            await session.execute(
                text(
                    """
                    insert into state_nodes (scope_id, kind, item_count)
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
                insert into state_leaves (scope_id, node_id, entry_count)
                values (:scope_id, :node_id, 1)
                """
            ),
            {"scope_id": scope_id, "node_id": node_id},
        )
        await session.execute(
            text(
                """
                insert into state_leaf_entries (
                  scope_id,
                  node_id,
                  pos,
                  namespace_id,
                  ord,
                  payload_id
                ) values (
                  :scope_id,
                  :node_id,
                  1,
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

        with pytest.raises(SQLAlchemyError, match="positions are not dense"):
            await session.commit()


@pytest.mark.asyncio
async def test_response_state_rejects_structural_node_updates(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        payload_id = await _create_payload(session, scope_id)
        node_id = await _create_leaf(session, scope_id, payload_id)
        await session.commit()

    async with db_session_maker() as session:
        with pytest.raises(SQLAlchemyError, match="structurally immutable"):
            await session.execute(
                text(
                    """
                    update state_nodes
                       set item_count = 2
                     where scope_id = :scope_id
                       and node_id = :node_id
                    """
                ),
                {"scope_id": scope_id, "node_id": node_id},
            )


@pytest.mark.asyncio
async def test_response_state_rejects_concat_with_duplicate_child(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        payload_id = await _create_payload(session, scope_id)
        node_id = await _create_leaf(session, scope_id, payload_id)

        with pytest.raises(SQLAlchemyError):
            await session.execute(
                text(
                    """
                    insert into state_nodes (
                      scope_id,
                      kind,
                      left_id,
                      right_id,
                      item_count
                    ) values (
                      :scope_id,
                      'concat',
                      :node_id,
                      :node_id,
                      2
                    )
                    """
                ),
                {"scope_id": scope_id, "node_id": node_id},
            )


@pytest.mark.asyncio
async def test_response_state_rejects_conversation_identity_updates(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        payload_id = await _create_payload(session, scope_id)
        node_id = await _create_leaf(session, scope_id, payload_id)
        await session.execute(
            text(
                """
                insert into responses (
                  scope_id,
                  response_id,
                  full_state_root_id
                ) values (
                  :scope_id,
                  'resp_conversation_identity',
                  :node_id
                )
                """
            ),
            {"scope_id": scope_id, "node_id": node_id},
        )
        await session.execute(
            text(
                """
                insert into conversations (
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

    async with db_session_maker() as session:
        with pytest.raises(SQLAlchemyError, match="structurally immutable"):
            await session.execute(
                text(
                    """
                    update conversations
                       set conversation_id = 'conv_renamed'
                     where scope_id = :scope_id
                       and conversation_id = 'conv_identity'
                    """
                ),
                {"scope_id": scope_id},
            )


@pytest.mark.asyncio
async def test_response_state_rejects_namespace_counter_updates(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        payload_id = await _create_payload(session, scope_id)
        node_id = await _create_leaf(session, scope_id, payload_id)
        namespace_id = (
            await session.execute(
                text(
                    """
                    select namespace_id
                      from ordinal_namespaces
                     where namespace_name = 'message'
                    """
                )
            )
        ).scalar_one()
        await session.execute(
            text(
                """
                insert into responses (
                  scope_id,
                  response_id,
                  full_state_root_id
                ) values (
                  :scope_id,
                  'resp_counter_immutable',
                  :node_id
                )
                """
            ),
            {"scope_id": scope_id, "node_id": node_id},
        )
        await session.execute(
            text(
                """
                insert into response_namespace_counters (
                  scope_id,
                  response_id,
                  namespace_id,
                  next_ord
                ) values (
                  :scope_id,
                  'resp_counter_immutable',
                  :namespace_id,
                  1
                )
                """
            ),
            {"scope_id": scope_id, "namespace_id": namespace_id},
        )
        await session.commit()

    async with db_session_maker() as session:
        with pytest.raises(SQLAlchemyError, match="rows are immutable"):
            await session.execute(
                text(
                    """
                    update response_namespace_counters
                       set next_ord = 2
                     where scope_id = :scope_id
                       and response_id = 'resp_counter_immutable'
                       and namespace_id = :namespace_id
                    """
                ),
                {"scope_id": scope_id, "namespace_id": namespace_id},
            )
