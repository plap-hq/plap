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


def _namespace_counters(message_next_ord: int, summary_next_ord: int = 0) -> str:
    return json.dumps(
        [
            {"namespace": "m", "next_ord": message_next_ord},
            {"namespace": "s", "next_ord": summary_next_ord},
        ]
    )


def _numbered_items(count: int, *, start_ord: int = 0) -> str:
    payloads = [
        {"type": "message", "text": f"message {start_ord + index}"}
        for index in range(count)
    ]
    return _items(*payloads, start_ord=start_ord)


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
                  from responses.append_items(
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
                "items": _items(payload, start_ord=next_ord - 1),
                "namespace_counters": _namespace_counters(next_ord),
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
                  from responses.responses
                 where scope_id = :scope_id
                   and response_id = :response_id
                """
            ),
            {"scope_id": scope_id, "response_id": response_id},
        )
    ).one()
    return row.child_refcount, row.lease_refcount


async def _create_numbered_tree(session, scope_id, count: int) -> int:
    return (
        await session.execute(
            text(
                """
                select responses.create_items_tree(
                  :scope_id,
                  cast(:items as jsonb)
                )
                """
            ),
            {"scope_id": scope_id, "items": _numbered_items(count)},
        )
    ).scalar_one()


async def _tree_texts(session, scope_id, root_id: int) -> list[str]:
    return (
        (
            await session.execute(
                text(
                    """
                    select payload ->> 'text'
                      from responses.list_state_items(:scope_id, :root_id)
                     order by item_position
                    """
                ),
                {"scope_id": scope_id, "root_id": root_id},
            )
        )
        .scalars()
        .all()
    )


async def _tree_count(session, scope_id, root_id: int | None) -> int:
    if root_id is None:
        return 0
    return (
        await session.execute(
            text(
                """
                select count(*)
                  from responses.list_state_items(:scope_id, :root_id)
                """
            ),
            {"scope_id": scope_id, "root_id": root_id},
        )
    ).scalar_one()


@pytest.mark.asyncio
async def test_responses_payload_dedupe_and_collision_guard(
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
                    select responses.get_or_create_payload(
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
                    select responses.get_or_create_payload(
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
                    select responses.get_or_create_payload(
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
async def test_responses_create_leaf_rejects_same_batch_payload_hash_collision(
    db_session_maker,
) -> None:
    scope_id = uuid4()
    first_payload = {"type": "message", "text": "first"}
    second_payload = {"type": "message", "text": "second"}
    reused_hash = _canonical_payload_hash(first_payload)

    async with db_session_maker() as session:
        with pytest.raises(SQLAlchemyError, match="non-canonical payload hash"):
            await session.execute(
                text(
                    """
                    select responses.create_leaf(:scope_id, cast(:items as jsonb))
                    """
                ),
                {
                    "scope_id": scope_id,
                    "items": json.dumps(
                        [
                            {
                                "namespace": "m",
                                "ord": 0,
                                "payload_hash": reused_hash,
                                "payload": first_payload,
                            },
                            {
                                "namespace": "m",
                                "ord": 1,
                                "payload_hash": reused_hash,
                                "payload": second_payload,
                            },
                        ]
                    ),
                },
            )


@pytest.mark.asyncio
async def test_responses_create_leaf_rejects_duplicate_ordinals(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        with pytest.raises(SQLAlchemyError, match=r"duplicate ordinal m\.4"):
            await session.execute(
                text(
                    """
                    select responses.create_leaf(:scope_id, cast(:items as jsonb))
                    """
                ),
                {
                    "scope_id": scope_id,
                    "items": _items(
                        {"type": "message", "text": "first"},
                        {"type": "message", "text": "second"},
                        start_ord=4,
                    ).replace('"ord": 5', '"ord": 4'),
                },
            )


@pytest.mark.asyncio
async def test_responses_create_leaf_rejects_invalid_ord_cleanly(
    db_session_maker,
) -> None:
    scope_id = uuid4()
    payload = {"type": "message", "text": "bad ord"}

    async with db_session_maker() as session:
        with pytest.raises(SQLAlchemyError, match="entry at position 0 is malformed"):
            await session.execute(
                text(
                    """
                    select responses.create_leaf(:scope_id, cast(:items as jsonb))
                    """
                ),
                {
                    "scope_id": scope_id,
                    "items": json.dumps(
                        [
                            {
                                "namespace": "m",
                                "ord": "abc",
                                "payload_hash": _canonical_payload_hash(payload),
                                "payload": payload,
                            }
                        ]
                    ),
                },
            )


@pytest.mark.asyncio
async def test_responses_create_items_tree_rejects_cross_leaf_duplicate_ordinals(
    db_session_maker,
) -> None:
    scope_id = uuid4()
    payloads = [{"type": "message", "text": f"message {index}"} for index in range(129)]
    items = [
        {
            "namespace": "m",
            "ord": 0 if index == 128 else index,
            "payload_hash": _canonical_payload_hash(payload),
            "payload": payload,
        }
        for index, payload in enumerate(payloads)
    ]

    async with db_session_maker() as session:
        with pytest.raises(SQLAlchemyError, match=r"duplicate ordinal m\.0"):
            await session.execute(
                text(
                    """
                    select responses.create_items_tree(
                      :scope_id,
                      cast(:items as jsonb)
                    )
                    """
                ),
                {"scope_id": scope_id, "items": json.dumps(items)},
            )


@pytest.mark.asyncio
async def test_responses_create_items_tree_rejects_normalized_duplicate_ordinals(
    db_session_maker,
) -> None:
    scope_id = uuid4()
    payloads = [{"type": "message", "text": f"message {index}"} for index in range(129)]
    items = [
        {
            "namespace": "m",
            "ord": "01" if index == 128 else str(index),
            "payload_hash": _canonical_payload_hash(payload),
            "payload": payload,
        }
        for index, payload in enumerate(payloads)
    ]

    async with db_session_maker() as session:
        with pytest.raises(SQLAlchemyError, match=r"duplicate ordinal m\.1"):
            await session.execute(
                text(
                    """
                    select responses.create_items_tree(
                      :scope_id,
                      cast(:items as jsonb)
                    )
                    """
                ),
                {"scope_id": scope_id, "items": json.dumps(items)},
            )


@pytest.mark.asyncio
async def test_responses_create_items_tree_rejects_duplicate_huge_ord_cleanly(
    db_session_maker,
) -> None:
    scope_id = uuid4()
    huge_ord = "922337203685477580800"
    payloads = [
        {"type": "message", "text": "first huge ord"},
        {"type": "message", "text": "second huge ord"},
    ]
    items = [
        {
            "namespace": "m",
            "ord": huge_ord,
            "payload_hash": _canonical_payload_hash(payload),
            "payload": payload,
        }
        for payload in payloads
    ]

    async with db_session_maker() as session:
        with pytest.raises(SQLAlchemyError, match="entry at position 0 is malformed"):
            await session.execute(
                text(
                    """
                    select responses.create_items_tree(
                      :scope_id,
                      cast(:items as jsonb)
                    )
                    """
                ),
                {"scope_id": scope_id, "items": json.dumps(items)},
            )


@pytest.mark.asyncio
async def test_responses_create_leaf_and_internal_node_update_refcounts(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        left_id = (
            await session.execute(
                text(
                    """
                    select responses.create_leaf(:scope_id, cast(:items as jsonb))
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
                    select responses.create_leaf(:scope_id, cast(:items as jsonb))
                    """
                ),
                {
                    "scope_id": scope_id,
                    "items": _items({"type": "message", "text": "right"}),
                },
            )
        ).scalar_one()
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

        rows = (
            await session.execute(
                text(
                    """
                    select node_id, kind, height, item_count, child_count, refcount
                      from responses.state_nodes
                     where scope_id = :scope_id
                       and node_id in (:left_id, :right_id, :internal_id)
                     order by node_id
                    """
                ),
                {
                    "scope_id": scope_id,
                    "left_id": left_id,
                    "right_id": right_id,
                    "internal_id": internal_id,
                },
            )
        ).all()

    assert [
        (row.kind, row.height, row.item_count, row.child_count, row.refcount)
        for row in rows
    ] == [
        ("leaf", 0, 1, 0, 1),
        ("leaf", 0, 1, 0, 1),
        ("internal", 1, 2, 2, 0),
    ]


@pytest.mark.asyncio
async def test_responses_create_internal_node_rejects_duplicate_children(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        leaf_id = (
            await session.execute(
                text(
                    """
                    select responses.create_leaf(:scope_id, cast(:items as jsonb))
                    """
                ),
                {
                    "scope_id": scope_id,
                    "items": _items({"type": "message", "text": "leaf"}),
                },
            )
        ).scalar_one()

        with pytest.raises(SQLAlchemyError, match="more than once"):
            await session.execute(
                text(
                    """
                    select responses.create_internal_node(
                      :scope_id,
                      array[:leaf_id, :leaf_id]::bigint[]
                    )
                    """
                ),
                {"scope_id": scope_id, "leaf_id": leaf_id},
            )


@pytest.mark.asyncio
async def test_responses_concat_merges_same_height_internal_children_when_possible(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        leaf_ids = [
            (
                await session.execute(
                    text(
                        """
                        select responses.create_leaf(
                          :scope_id,
                          cast(:items as jsonb)
                        )
                        """
                    ),
                    {
                        "scope_id": scope_id,
                        "items": _items(
                            {"type": "message", "text": f"leaf {index}"},
                            start_ord=index,
                        ),
                    },
                )
            ).scalar_one()
            for index in range(4)
        ]
        left_root = (
            await session.execute(
                text(
                    """
                    select responses.create_internal_node(
                      :scope_id,
                      array[:first_id, :second_id]::bigint[]
                    )
                    """
                ),
                {
                    "scope_id": scope_id,
                    "first_id": leaf_ids[0],
                    "second_id": leaf_ids[1],
                },
            )
        ).scalar_one()
        right_root = (
            await session.execute(
                text(
                    """
                    select responses.create_internal_node(
                      :scope_id,
                      array[:third_id, :fourth_id]::bigint[]
                    )
                    """
                ),
                {
                    "scope_id": scope_id,
                    "third_id": leaf_ids[2],
                    "fourth_id": leaf_ids[3],
                },
            )
        ).scalar_one()
        merged_root = (
            await session.execute(
                text(
                    """
                    select responses.concat_balanced(
                      :scope_id,
                      :left_root,
                      :right_root
                    )
                    """
                ),
                {
                    "scope_id": scope_id,
                    "left_root": left_root,
                    "right_root": right_root,
                },
            )
        ).scalar_one()
        await session.commit()

        root = (
            await session.execute(
                text(
                    """
                    select height, child_count, item_count
                      from responses.state_nodes
                     where scope_id = :scope_id
                       and node_id = :root_id
                    """
                ),
                {"scope_id": scope_id, "root_id": merged_root},
            )
        ).one()
        child_ids = (
            (
                await session.execute(
                    text(
                        """
                        select child_node_id
                          from responses.state_node_children
                         where scope_id = :scope_id
                           and parent_node_id = :root_id
                         order by slot
                        """
                    ),
                    {"scope_id": scope_id, "root_id": merged_root},
                )
            )
            .scalars()
            .all()
        )

    assert (root.height, root.child_count, root.item_count) == (1, 4, 4)
    assert child_ids == leaf_ids


@pytest.mark.asyncio
async def test_responses_append_items_creates_response_chain(
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
        second_root = (
            await session.execute(
                text(
                    """
                    select kind, item_count
                      from responses.state_nodes
                     where scope_id = :scope_id
                       and node_id = :node_id
                    """
                ),
                {"scope_id": scope_id, "node_id": second_root},
            )
        ).one()
        counters = (
            await session.execute(
                text(
                    """
                    select c.next_ord
                      from responses.response_namespace_counters c
                      join responses.ordinal_namespaces n
                        on n.namespace_id = c.namespace_id
                     where scope_id = :scope_id
                       and response_id = 'resp_2'
                       and n.namespace_name = 'm'
                    """
                ),
                {"scope_id": scope_id},
            )
        ).scalar_one()

    assert first_root != second_root
    assert first_counts == (1, 1)
    assert second_counts == (0, 1)
    assert (second_root.kind, second_root.item_count) == ("internal", 2)
    assert counters == 2


@pytest.mark.asyncio
async def test_responses_append_items_balances_repeated_appends(
    db_session_maker,
) -> None:
    scope_id = uuid4()
    previous_response_id: str | None = None

    async with db_session_maker() as session:
        for index in range(130):
            response_id = f"resp_{index + 1}"
            await _append_response(
                session,
                scope_id,
                response_id,
                {"type": "message", "text": f"message {index + 1}"},
                prev_response_id=previous_response_id,
                next_ord=index + 1,
            )
            previous_response_id = response_id
        await session.commit()

        row = (
            await session.execute(
                text(
                    """
                    select root.item_count, root.height, root.child_count
                      from responses.responses response
                      join responses.state_nodes root
                        on root.scope_id = response.scope_id
                       and root.node_id = response.full_state_root_id
                     where response.scope_id = :scope_id
                       and response.response_id = 'resp_130'
                    """
                ),
                {"scope_id": scope_id},
            )
        ).one()

    assert row.item_count == 130
    assert row.height <= 2
    assert row.child_count >= 2


@pytest.mark.asyncio
async def test_responses_create_node_tree_avoids_single_child_carry_groups(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        leaf_ids = []
        for index in range(65):
            leaf_id = (
                await session.execute(
                    text(
                        """
                        select responses.create_leaf(
                          :scope_id,
                          cast(:items as jsonb)
                        )
                        """
                    ),
                    {
                        "scope_id": scope_id,
                        "items": _items(
                            {"type": "message", "text": f"message {index}"},
                            start_ord=index,
                        ),
                    },
                )
            ).scalar_one()
            leaf_ids.append(leaf_id)

        root_id = (
            await session.execute(
                text(
                    """
                    select responses.create_node_tree_from_children(
                      :scope_id,
                      cast(:child_ids as bigint[])
                    )
                    """
                ),
                {"scope_id": scope_id, "child_ids": leaf_ids},
            )
        ).scalar_one()
        await session.commit()

        root = (
            await session.execute(
                text(
                    """
                    select height, child_count, item_count
                      from responses.state_nodes
                     where scope_id = :scope_id
                       and node_id = :root_id
                    """
                ),
                {"scope_id": scope_id, "root_id": root_id},
            )
        ).one()
        child_counts = (
            (
                await session.execute(
                    text(
                        """
                        select child.child_count
                          from responses.state_node_children edge
                          join responses.state_nodes child
                            on child.scope_id = edge.scope_id
                           and child.node_id = edge.child_node_id
                         where edge.scope_id = :scope_id
                           and edge.parent_node_id = :root_id
                         order by edge.slot
                        """
                    ),
                    {"scope_id": scope_id, "root_id": root_id},
                )
            )
            .scalars()
            .all()
        )

    assert (root.height, root.child_count, root.item_count) == (2, 2, 65)
    assert child_counts == [33, 32]


@pytest.mark.asyncio
async def test_responses_create_items_tree_handles_leaf_boundary_plus_one(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        root_id = await _create_numbered_tree(session, scope_id, 129)
        await session.commit()

        root = (
            await session.execute(
                text(
                    """
                    select kind, height, item_count, child_count
                      from responses.state_nodes
                     where scope_id = :scope_id
                       and node_id = :root_id
                    """
                ),
                {"scope_id": scope_id, "root_id": root_id},
            )
        ).one()
        boundary_rows = (
            await session.execute(
                text(
                    """
                    select item_position, payload ->> 'text' as text
                      from responses.list_state_items(:scope_id, :root_id, 127, 2)
                     order by item_position
                    """
                ),
                {"scope_id": scope_id, "root_id": root_id},
            )
        ).all()

    assert (root.kind, root.height, root.item_count, root.child_count) == (
        "internal",
        1,
        129,
        2,
    )
    assert [(row.item_position, row.text) for row in boundary_rows] == [
        (127, "message 127"),
        (128, "message 128"),
    ]


@pytest.mark.asyncio
async def test_responses_create_items_tree_handles_multilevel_8193_items(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        root_id = await _create_numbered_tree(session, scope_id, 8193)
        await session.commit()

        root = (
            await session.execute(
                text(
                    """
                    select height, item_count, child_count
                      from responses.state_nodes
                     where scope_id = :scope_id
                       and node_id = :root_id
                    """
                ),
                {"scope_id": scope_id, "root_id": root_id},
            )
        ).one()
        edge_rows = (
            await session.execute(
                text(
                    """
                    select item_position, payload ->> 'text' as text
                      from responses.list_state_items(:scope_id, :root_id, 8191, 2)
                     order by item_position
                    """
                ),
                {"scope_id": scope_id, "root_id": root_id},
            )
        ).all()

    assert (root.height, root.item_count, root.child_count) == (2, 8193, 2)
    assert [(row.item_position, row.text) for row in edge_rows] == [
        (8191, "message 8191"),
        (8192, "message 8192"),
    ]


@pytest.mark.asyncio
async def test_responses_split_at_index_handles_edges_and_boundaries(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        root_id = await _create_numbered_tree(session, scope_id, 8193)
        await session.commit()

        expected_counts = {
            0: (0, 8193),
            37: (37, 8156),
            128: (128, 8065),
            4224: (4224, 3969),
            8193: (8193, 0),
        }
        observed_counts = {}
        for split_index in expected_counts:
            split = (
                await session.execute(
                    text(
                        """
                        select left_root_id, right_root_id
                          from responses.split_at_index(
                            :scope_id,
                            :root_id,
                            :split_index
                          )
                        """
                    ),
                    {
                        "scope_id": scope_id,
                        "root_id": root_id,
                        "split_index": split_index,
                    },
                )
            ).one()
            observed_counts[split_index] = (
                await _tree_count(session, scope_id, split.left_root_id),
                await _tree_count(session, scope_id, split.right_root_id),
            )

        old_root_sample = (
            (
                await session.execute(
                    text(
                        """
                    select payload ->> 'text'
                      from responses.list_state_items(:scope_id, :root_id, 4223, 2)
                     order by item_position
                    """
                    ),
                    {"scope_id": scope_id, "root_id": root_id},
                )
            )
            .scalars()
            .all()
        )

    assert observed_counts == expected_counts
    assert old_root_sample == ["message 4223", "message 4224"]


@pytest.mark.asyncio
async def test_responses_list_state_items_reads_ranges_from_multilevel_tree(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        leaf_ids = [
            (
                await session.execute(
                    text(
                        """
                        select responses.create_leaf(
                          :scope_id,
                          cast(:items as jsonb)
                        )
                        """
                    ),
                    {
                        "scope_id": scope_id,
                        "items": _items(
                            {"type": "message", "text": f"message {index}"},
                            start_ord=index,
                        ),
                    },
                )
            ).scalar_one()
            for index in range(65)
        ]
        root_id = (
            await session.execute(
                text(
                    """
                    select responses.create_node_tree_from_children(
                      :scope_id,
                      cast(:child_ids as bigint[])
                    )
                    """
                ),
                {"scope_id": scope_id, "child_ids": leaf_ids},
            )
        ).scalar_one()
        await session.commit()

        rows = (
            await session.execute(
                text(
                    """
                    select item_position, payload ->> 'text' as text
                      from responses.list_state_items(:scope_id, :root_id, 31, 3)
                     order by item_position
                    """
                ),
                {"scope_id": scope_id, "root_id": root_id},
            )
        ).all()

    assert [(row.item_position, row.text) for row in rows] == [
        (31, "message 31"),
        (32, "message 32"),
        (33, "message 33"),
    ]


@pytest.mark.asyncio
async def test_responses_list_and_splice_support_compaction_shape(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        root_id = (
            await session.execute(
                text(
                    """
                    select responses.create_items_tree(
                      :scope_id,
                      cast(:items as jsonb)
                    )
                    """
                ),
                {
                    "scope_id": scope_id,
                    "items": _items(
                        {"type": "message", "text": "m1"},
                        {"type": "message", "text": "m2"},
                        {"type": "message", "text": "m3"},
                        {"type": "message", "text": "m4"},
                    ),
                },
            )
        ).scalar_one()
        summary_payload = {"type": "summary", "text": "s1"}
        summary_root_id = (
            await session.execute(
                text(
                    """
                    select responses.create_items_tree(
                      :scope_id,
                      cast(:items as jsonb)
                    )
                    """
                ),
                {
                    "scope_id": scope_id,
                    "items": json.dumps(
                        [
                            {
                                "namespace": "s",
                                "ord": 0,
                                "payload_hash": _canonical_payload_hash(
                                    summary_payload
                                ),
                                "payload": summary_payload,
                            }
                        ]
                    ),
                },
            )
        ).scalar_one()
        spliced_root_id = (
            await session.execute(
                text(
                    """
                    select responses.splice(
                      :scope_id,
                      :root_id,
                      1,
                      2,
                      :summary_root_id
                    )
                    """
                ),
                {
                    "scope_id": scope_id,
                    "root_id": root_id,
                    "summary_root_id": summary_root_id,
                },
            )
        ).scalar_one()
        rows = (
            await session.execute(
                text(
                    """
                    select namespace, ord, payload ->> 'text' as text
                      from responses.list_state_items(:scope_id, :root_id)
                     order by item_position
                    """
                ),
                {"scope_id": scope_id, "root_id": spliced_root_id},
            )
        ).all()
        old_rows = (
            await session.execute(
                text(
                    """
                    select namespace, ord, payload ->> 'text' as text
                      from responses.list_state_items(:scope_id, :root_id)
                     order by item_position
                    """
                ),
                {"scope_id": scope_id, "root_id": root_id},
            )
        ).all()

    assert [(row.namespace, row.ord, row.text) for row in rows] == [
        ("m", 0, "m1"),
        ("s", 0, "s1"),
        ("m", 3, "m4"),
    ]
    assert [(row.namespace, row.ord, row.text) for row in old_rows] == [
        ("m", 0, "m1"),
        ("m", 1, "m2"),
        ("m", 2, "m3"),
        ("m", 3, "m4"),
    ]


@pytest.mark.asyncio
async def test_responses_gc_preserves_shared_nodes_after_splice(
    db_session_maker,
) -> None:
    scope_id = uuid4()
    summary_payload = {"type": "summary", "text": "summary 0"}

    async with db_session_maker() as session:
        old_root_id = await _create_numbered_tree(session, scope_id, 257)
        shared_leaf_ids = (
            (
                await session.execute(
                    text(
                        """
                        select child_node_id
                          from responses.state_node_children
                         where scope_id = :scope_id
                           and parent_node_id = :root_id
                           and slot in (0, 2)
                         order by slot
                        """
                    ),
                    {"scope_id": scope_id, "root_id": old_root_id},
                )
            )
            .scalars()
            .all()
        )
        summary_root_id = (
            await session.execute(
                text(
                    """
                    select responses.create_items_tree(
                      :scope_id,
                      cast(:items as jsonb)
                    )
                    """
                ),
                {
                    "scope_id": scope_id,
                    "items": json.dumps(
                        [
                            {
                                "namespace": "s",
                                "ord": 0,
                                "payload_hash": _canonical_payload_hash(
                                    summary_payload
                                ),
                                "payload": summary_payload,
                            }
                        ]
                    ),
                },
            )
        ).scalar_one()
        new_root_id = (
            await session.execute(
                text(
                    """
                    select responses.splice(
                      :scope_id,
                      :root_id,
                      129,
                      1,
                      :summary_root_id
                    )
                    """
                ),
                {
                    "scope_id": scope_id,
                    "root_id": old_root_id,
                    "summary_root_id": summary_root_id,
                },
            )
        ).scalar_one()
        await session.execute(
            text(
                """
                select responses.create_response(
                  :scope_id,
                  'resp_old_root',
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
                "root_id": old_root_id,
                "namespace_counters": _namespace_counters(257),
            },
        )
        await session.execute(
            text(
                """
                select responses.create_response(
                  :scope_id,
                  'resp_new_root',
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
                "root_id": new_root_id,
                "namespace_counters": _namespace_counters(257, 1),
            },
        )
        await session.commit()

        before_count = (
            await session.execute(
                text(
                    """
                    select count(*)
                      from responses.state_nodes
                     where scope_id = :scope_id
                    """
                ),
                {"scope_id": scope_id},
            )
        ).scalar_one()
        await session.execute(text("call responses.gc_prune_state_nodes(1000)"))
        await session.commit()
        after_count = (
            await session.execute(
                text(
                    """
                    select count(*)
                      from responses.state_nodes
                     where scope_id = :scope_id
                    """
                ),
                {"scope_id": scope_id},
            )
        ).scalar_one()
        shared_nodes_exist = (
            await session.execute(
                text(
                    """
                    select count(*)
                      from responses.state_nodes
                     where scope_id = :scope_id
                       and node_id = any(:node_ids)
                    """
                ),
                {"scope_id": scope_id, "node_ids": shared_leaf_ids},
            )
        ).scalar_one()
        old_sample = (
            (
                await session.execute(
                    text(
                        """
                    select payload ->> 'text'
                      from responses.list_state_items(:scope_id, :root_id, 128, 3)
                     order by item_position
                    """
                    ),
                    {"scope_id": scope_id, "root_id": old_root_id},
                )
            )
            .scalars()
            .all()
        )
        new_sample = (
            (
                await session.execute(
                    text(
                        """
                    select payload ->> 'text'
                      from responses.list_state_items(:scope_id, :root_id, 128, 3)
                     order by item_position
                    """
                    ),
                    {"scope_id": scope_id, "root_id": new_root_id},
                )
            )
            .scalars()
            .all()
        )

    assert after_count < before_count
    assert shared_nodes_exist == len(shared_leaf_ids)
    assert old_sample == ["message 128", "message 129", "message 130"]
    assert new_sample == ["message 128", "summary 0", "message 130"]


@pytest.mark.asyncio
async def test_responses_append_items_requires_complete_namespace_counters(
    db_session_maker,
) -> None:
    scope_id = uuid4()

    async with db_session_maker() as session:
        with pytest.raises(SQLAlchemyError, match="missing required namespace s"):
            await session.execute(
                text(
                    """
                    select root_node_id
                      from responses.append_items(
                        :scope_id,
                        'resp_partial_counters',
                        null,
                        cast(:items as jsonb),
                        cast(:namespace_counters as jsonb),
                        '[]'::jsonb
                      )
                    """
                ),
                {
                    "scope_id": scope_id,
                    "items": _items({"type": "message", "text": "partial"}),
                    "namespace_counters": json.dumps(
                        [{"namespace": "m", "next_ord": 1}]
                    ),
                },
            )


@pytest.mark.asyncio
async def test_responses_lease_and_conversation_helpers_move_roots(
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
                    select responses.create_or_refresh_lease(
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
                    select responses.create_or_refresh_lease(
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
                select responses.move_conversation(
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
                select responses.move_conversation(
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
                select responses.release_lease(
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
