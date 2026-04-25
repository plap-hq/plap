from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import blake3
import msgspec
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from plap.responses.state.types import (
    AppendResponseResult,
    JSONPayload,
    NamespaceCursor,
    ResponseRecord,
    StateCheckpoint,
    StateItem,
)


class ResponseStateRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    @staticmethod
    def payload_hash(payload: JSONPayload) -> str:
        canonical = msgspec.json.encode(payload, order="deterministic")
        return blake3.blake3(canonical).hexdigest()

    async def build_tree(
        self,
        scope_id: UUID,
        items: Sequence[StateItem],
    ) -> int:
        return (
            await self._session.execute(
                text(
                    """
                    select responses.build_state_tree(
                      :scope_id,
                      cast(:items as jsonb)
                    )
                    """
                ),
                {"scope_id": scope_id, "items": self._items_json(items)},
            )
        ).scalar_one()

    async def list_items(
        self,
        scope_id: UUID,
        state_root_id: int,
        *,
        start_index: int = 0,
        limit: int | None = None,
    ) -> list[StateItem]:
        rows = (
            await self._session.execute(
                text(
                    """
                    select namespace, ordinal, payload_hash, payload
                      from responses.list_state_items(
                        :scope_id,
                        :state_root_id,
                        :start_index,
                        :limit
                      )
                     order by item_position
                    """
                ),
                {
                    "scope_id": scope_id,
                    "state_root_id": state_root_id,
                    "start_index": start_index,
                    "limit": limit,
                },
            )
        ).all()

        return [
            StateItem(
                namespace=row.namespace,
                ordinal=row.ordinal,
                payload=row.payload,
                payload_hash=row.payload_hash,
            )
            for row in rows
        ]

    async def splice_tree(
        self,
        scope_id: UUID,
        state_root_id: int,
        start_index: int,
        delete_count: int,
        *,
        insert_state_root_id: int | None = None,
    ) -> int | None:
        return (
            await self._session.execute(
                text(
                    """
                    select responses.splice_state_tree(
                      :scope_id,
                      :state_root_id,
                      :start_index,
                      :delete_count,
                      :insert_state_root_id
                    )
                    """
                ),
                {
                    "scope_id": scope_id,
                    "state_root_id": state_root_id,
                    "start_index": start_index,
                    "delete_count": delete_count,
                    "insert_state_root_id": insert_state_root_id,
                },
            )
        ).scalar_one()

    async def create_response_record(
        self,
        scope_id: UUID,
        response_id: str,
        previous_response_id: str | None,
        state_root_id: int,
        namespace_cursors: Sequence[NamespaceCursor],
        *,
        checkpoints: Sequence[StateCheckpoint] = (),
        retention: timedelta | None = timedelta(days=30),
        status: str = "completed",
        completed_at: datetime | None = None,
        fields: JSONPayload | None = None,
    ) -> str:
        return (
            await self._session.execute(
                text(
                    """
                    select responses.create_response_record(
                      :scope_id,
                      :response_id,
                      :previous_response_id,
                      :state_root_id,
                      cast(:namespace_cursors as jsonb),
                      cast(:checkpoints as jsonb),
                      :retention,
                      :status,
                      :completed_at,
                      cast(:fields as jsonb)
                    )
                    """
                ),
                {
                    "scope_id": scope_id,
                    "response_id": response_id,
                    "previous_response_id": previous_response_id,
                    "state_root_id": state_root_id,
                    "namespace_cursors": self._namespace_cursors_json(
                        namespace_cursors
                    ),
                    "checkpoints": self._checkpoints_json(checkpoints),
                    "retention": retention,
                    "status": status,
                    "completed_at": completed_at,
                    "fields": self._fields_json(fields),
                },
            )
        ).scalar_one()

    async def append_response(
        self,
        scope_id: UUID,
        response_id: str,
        previous_response_id: str | None,
        items: Sequence[StateItem],
        namespace_cursors: Sequence[NamespaceCursor],
        *,
        checkpoints: Sequence[StateCheckpoint] = (),
        retention: timedelta | None = timedelta(days=30),
        status: str = "completed",
        completed_at: datetime | None = None,
        fields: JSONPayload | None = None,
    ) -> AppendResponseResult:
        row = (
            await self._session.execute(
                text(
                    """
                    select response_id, state_root_id
                      from responses.append_response(
                        :scope_id,
                        :response_id,
                        :previous_response_id,
                        cast(:items as jsonb),
                        cast(:namespace_cursors as jsonb),
                        cast(:checkpoints as jsonb),
                        :retention,
                        :status,
                        :completed_at,
                        cast(:fields as jsonb)
                      )
                    """
                ),
                {
                    "scope_id": scope_id,
                    "response_id": response_id,
                    "previous_response_id": previous_response_id,
                    "items": self._items_json(items),
                    "namespace_cursors": self._namespace_cursors_json(
                        namespace_cursors
                    ),
                    "checkpoints": self._checkpoints_json(checkpoints),
                    "retention": retention,
                    "status": status,
                    "completed_at": completed_at,
                    "fields": self._fields_json(fields),
                },
            )
        ).one()
        return AppendResponseResult(
            response_id=row.response_id,
            state_root_id=row.state_root_id,
        )

    async def get_response_record(
        self,
        scope_id: UUID,
        response_id: str,
    ) -> ResponseRecord | None:
        row = (
            await self._session.execute(
                text(
                    """
                    select response_id,
                           prev_response_id,
                           state_root_id,
                           status,
                           created_at,
                           completed_at,
                           fields
                      from responses.response_records
                     where scope_id = :scope_id
                       and response_id = :response_id
                    """
                ),
                {"scope_id": scope_id, "response_id": response_id},
            )
        ).one_or_none()
        return self._response_record(row)

    async def get_conversation_head(
        self,
        scope_id: UUID,
        conversation_id: str,
    ) -> ResponseRecord | None:
        row = (
            await self._session.execute(
                text(
                    """
                    select record.response_id,
                           record.prev_response_id,
                           record.state_root_id,
                           record.status,
                           record.created_at,
                           record.completed_at,
                           record.fields
                      from responses.conversations conversation
                      join responses.response_records record
                        on record.scope_id = conversation.scope_id
                       and record.response_id = conversation.current_response_id
                     where conversation.scope_id = :scope_id
                       and conversation.conversation_id = :conversation_id
                    """
                ),
                {"scope_id": scope_id, "conversation_id": conversation_id},
            )
        ).one_or_none()
        return self._response_record(row)

    async def move_conversation_head(
        self,
        scope_id: UUID,
        conversation_id: str,
        response_id: str,
        *,
        retention: timedelta | None = timedelta(days=30),
    ) -> None:
        await self._session.execute(
            text(
                """
                select responses.move_conversation_head(
                  :scope_id,
                  :conversation_id,
                  :response_id,
                  :retention
                )
                """
            ),
            {
                "scope_id": scope_id,
                "conversation_id": conversation_id,
                "response_id": response_id,
                "retention": retention,
            },
        )

    async def create_or_refresh_response_lease(
        self,
        scope_id: UUID,
        response_id: str,
        owner_type: str,
        owner_id: str,
        *,
        expires_at: Any = None,
    ) -> UUID:
        return (
            await self._session.execute(
                text(
                    """
                    select responses.create_or_refresh_response_lease(
                      :scope_id,
                      :response_id,
                      :owner_type,
                      :owner_id,
                      :expires_at
                    )
                    """
                ),
                {
                    "scope_id": scope_id,
                    "response_id": response_id,
                    "owner_type": owner_type,
                    "owner_id": owner_id,
                    "expires_at": expires_at,
                },
            )
        ).scalar_one()

    async def release_response_lease(
        self,
        scope_id: UUID,
        owner_type: str,
        owner_id: str,
    ) -> None:
        await self._session.execute(
            text(
                """
                select responses.release_response_lease(
                  :scope_id,
                  :owner_type,
                  :owner_id
                )
                """
            ),
            {"scope_id": scope_id, "owner_type": owner_type, "owner_id": owner_id},
        )

    @classmethod
    def _items_json(cls, items: Sequence[StateItem]) -> str:
        return msgspec.json.encode([cls._item_to_db(item) for item in items]).decode()

    @staticmethod
    def _namespace_cursors_json(cursors: Sequence[NamespaceCursor]) -> str:
        return msgspec.json.encode(
            [
                {"namespace": cursor.namespace, "next_ordinal": cursor.next_ordinal}
                for cursor in cursors
            ]
        ).decode()

    @classmethod
    def _checkpoints_json(cls, checkpoints: Sequence[StateCheckpoint]) -> str:
        return msgspec.json.encode(
            [
                {
                    "state_root_id": checkpoint.state_root_id,
                    "namespace_cursors": msgspec.json.decode(
                        cls._namespace_cursors_json(checkpoint.namespace_cursors)
                    ),
                }
                for checkpoint in checkpoints
            ]
        ).decode()

    @staticmethod
    def _fields_json(fields: JSONPayload | None) -> str:
        return msgspec.json.encode({} if fields is None else fields).decode()

    @classmethod
    def _item_to_db(cls, item: StateItem) -> dict[str, object]:
        payload_hash = cls.payload_hash(item.payload)
        if item.payload_hash is not None and item.payload_hash != payload_hash:
            raise ValueError("payload_hash does not match canonical payload hash")
        return {
            "namespace": item.namespace,
            "ordinal": item.ordinal,
            "payload_hash": payload_hash,
            "payload": item.payload,
        }

    @staticmethod
    def _response_record(row: Any) -> ResponseRecord | None:
        if row is None:
            return None
        return ResponseRecord(
            response_id=row.response_id,
            previous_response_id=row.prev_response_id,
            state_root_id=row.state_root_id,
            status=row.status,
            created_at=row.created_at,
            completed_at=row.completed_at,
            fields=row.fields,
        )
