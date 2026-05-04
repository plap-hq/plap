from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID

import msgspec
from pydantic import TypeAdapter
from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.sql.elements import TextClause

from plap.auth import AuthContext
from plap.persistence import Database
from plap.responses.contracts import (
    ConversationReference,
    InputItemsPage,
    InputItemsPageItem,
    RequestInputItem,
    RequestMessageItem,
    ResponseCreateRequest,
    ResponseObject,
)
from plap.responses.errors import ResponseError

_REQUEST_INPUT_ADAPTER = TypeAdapter(RequestInputItem)
_INPUT_ITEMS_PAGE_ADAPTER = TypeAdapter(InputItemsPageItem)

type PayloadObject = dict[str, object]
type ResponseFields = dict[str, object]


@dataclass(slots=True)
class PreparedRequest:
    scope_id: UUID
    response_request: ResponseCreateRequest
    execution_request: ResponseCreateRequest
    current_input_items: list[RequestInputItem]
    parent_response_id: str | None
    conversation_id: str | None
    persist_response: bool


class ResponseStore:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def prepare_request(self, auth_context: AuthContext, request: ResponseCreateRequest) -> PreparedRequest:
        scope_id = self._scope_id(auth_context)
        conversation_id = self._conversation_id(request.conversation)
        if request.previous_response_id is not None and conversation_id is not None:
            raise ResponseError.invalid_request(
                private_message="previous_response_id and conversation cannot be combined",
                param="previous_response_id",
            )
        if conversation_id is not None and request.store is False:
            raise ResponseError.invalid_request(
                private_message="conversation continuation requires a stored response",
                param="store",
            )

        current_input_items = self._current_input_items(request)
        parent_response_id = request.previous_response_id
        replay_items: list[RequestInputItem] = []
        if parent_response_id is not None or conversation_id is not None:
            async with self._database.connection() as connection:
                if parent_response_id is not None:
                    await self._require_replayable_response(
                        connection,
                        scope_id,
                        parent_response_id,
                        param="previous_response_id",
                    )
                else:
                    parent_response_id = await self._conversation_head(connection, scope_id, conversation_id)

                if parent_response_id is not None:
                    replay_items = await self._replay_items(connection, scope_id, parent_response_id)

        response_request = request.model_copy(update={"previous_response_id": parent_response_id})
        execution_request = request.model_copy(update={"input": [*replay_items, *current_input_items]})
        return PreparedRequest(
            scope_id=scope_id,
            response_request=response_request,
            execution_request=execution_request,
            current_input_items=current_input_items,
            parent_response_id=parent_response_id,
            conversation_id=conversation_id,
            persist_response=request.store is not False,
        )

    async def begin_response(self, prepared: PreparedRequest, response: ResponseObject) -> None:
        if not prepared.persist_response:
            return
        input_items = self._stored_input_payloads(response.id, prepared.current_input_items)
        async with self._database.connection_transaction() as connection:
            create_result = await connection.execute(
                text(
                    """
                    select responses.create_response_record(
                        :scope_id,
                        :response_id,
                        :prev_response_id,
                        cast(:input_items as jsonb),
                        '[]'::jsonb,
                        :retention,
                        :status,
                        null,
                        cast(:fields as jsonb)
                      )
                    """
                ),
                {
                    "scope_id": prepared.scope_id,
                    "response_id": response.id,
                    "prev_response_id": prepared.parent_response_id,
                    "input_items": self._json_text(input_items),
                    "retention": timedelta(days=30),
                    "status": "in_progress",
                    "fields": self._json_text(self._response_fields(response)),
                },
            )
            create_result.scalar_one()
            create_result.close()

    async def append_output_item(
        self,
        prepared: PreparedRequest,
        response_id: str,
        output_index: int,
        item: object,
    ) -> None:
        if not prepared.persist_response:
            return
        async with self._database.connection_transaction() as connection:
            insert_result = await connection.execute(
                text(
                    """
                    with payload as (
                      select responses.get_or_create_payload(:scope_id, cast(:payload as jsonb)) as payload_id
                    )
                    insert into responses.response_output_items (
                      scope_id,
                      response_id,
                      output_index,
                      payload_id
                    )
                    select :scope_id, :response_id, :output_index, payload_id
                      from payload
                    """
                ),
                {
                    "scope_id": prepared.scope_id,
                    "response_id": response_id,
                    "output_index": output_index,
                    "payload": self._json_text(item),
                },
            )
            insert_result.close()

    async def finish_response(self, prepared: PreparedRequest, response: ResponseObject) -> None:
        if not prepared.persist_response:
            return
        async with self._database.connection_transaction() as connection:
            update_result = await connection.execute(
                text(
                    """
                    update responses.response_records
                       set status = :status,
                           completed_at = :completed_at,
                           fields = cast(:fields as jsonb)
                     where scope_id = :scope_id
                       and response_id = :response_id
                    """
                ),
                {
                    "status": response.status,
                    "completed_at": self._completed_at(response),
                    "fields": self._json_text(self._response_fields(response)),
                    "scope_id": prepared.scope_id,
                    "response_id": response.id,
                },
            )
            update_result.close()
            if prepared.conversation_id is not None:
                move_head_result = await connection.execute(
                    text(
                        "select responses.move_conversation_head(:scope_id, :conversation_id, :response_id, :retention)"
                    ),
                    {
                        "scope_id": prepared.scope_id,
                        "conversation_id": prepared.conversation_id,
                        "response_id": response.id,
                        "retention": None,
                    },
                )
                move_head_result.scalar_one_or_none()
                move_head_result.close()

    async def get_response(self, auth_context: AuthContext, response_id: str) -> ResponseObject | None:
        scope_id = self._scope_id(auth_context)
        async with self._database.connection() as connection:
            record = await self._mappings_one_or_none(
                connection,
                text(
                    """
                    select response_id, status, completed_at, fields
                      from responses.response_records record
                     where record.scope_id = :scope_id
                       and record.response_id = :response_id
                       and not exists (
                         select 1
                           from responses.response_tombstones tombstone
                          where tombstone.scope_id = record.scope_id
                            and tombstone.response_id = record.response_id
                       )
                    """
                ),
                {"scope_id": scope_id, "response_id": response_id},
            )
            if record is None:
                return None
            output_items = await self._output_payloads(connection, scope_id, response_id)
            return self._response_from_record(record, output_items)

    async def list_input_items(
        self,
        auth_context: AuthContext,
        response_id: str,
        *,
        after: str | None,
        limit: int | None,
        order: str | None,
    ) -> InputItemsPage:
        scope_id = self._scope_id(auth_context)
        direction = order or "asc"
        if direction not in {"asc", "desc"}:
            raise ResponseError.invalid_request(private_message="input_items order must be asc or desc", param="order")
        page_limit = limit or 20
        async with self._database.connection() as connection:
            exists = await self._response_exists(connection, scope_id, response_id)
            if not exists:
                raise ResponseError.not_found(private_message=f"response not found for input_items: {response_id}")
            payloads = await self._input_payloads(connection, scope_id, response_id)

        items = [self._input_items_page_item_from_payload(payload) for payload in payloads]
        if direction == "desc":
            items.reverse()
        start = 0
        if after is not None:
            for index, item in enumerate(items):
                if item.id == after:
                    start = index + 1
                    break
            else:
                raise ResponseError.invalid_request(private_message="input_items after cursor was not found", param="after")
        page_items = items[start : start + page_limit]
        return InputItemsPage(
            data=page_items,
            first_id=page_items[0].id if page_items else None,
            has_more=start + page_limit < len(items),
            last_id=page_items[-1].id if page_items else None,
        )

    async def delete_response(self, auth_context: AuthContext, response_id: str) -> bool:
        scope_id = self._scope_id(auth_context)
        async with self._database.connection_transaction() as connection:
            exists = await self._response_exists(connection, scope_id, response_id)
            if not exists:
                return False
            inserted_result = await connection.execute(
                text(
                    """
                    insert into responses.response_tombstones (scope_id, response_id)
                    values (:scope_id, :response_id)
                    on conflict do nothing
                    returning response_id
                    """
                ),
                {"scope_id": scope_id, "response_id": response_id},
            )
            inserted = inserted_result.scalar_one_or_none()
            inserted_result.close()
            if inserted is None:
                return False
            delete_conversation_result = await connection.execute(
                text(
                    "delete from responses.conversations where scope_id = :scope_id and current_response_id = :response_id"
                ),
                {"scope_id": scope_id, "response_id": response_id},
            )
            delete_conversation_result.close()
            release_result = await connection.execute(
                text("select responses.release_response_lease(:scope_id, 'response', :response_id)"),
                {"scope_id": scope_id, "response_id": response_id},
            )
            release_result.scalar_one_or_none()
            release_result.close()
            return True

    @staticmethod
    def _scope_id(auth_context: AuthContext) -> UUID:
        return auth_context.organization_id or auth_context.user_id

    @staticmethod
    def _conversation_id(conversation: str | ConversationReference | None) -> str | None:
        if conversation is None:
            return None
        if isinstance(conversation, str):
            return conversation
        return conversation.id

    @staticmethod
    def _current_input_items(request: ResponseCreateRequest) -> list[RequestInputItem]:
        if request.input is None:
            return []
        if isinstance(request.input, str):
            return [RequestMessageItem(content=request.input, role="user", type="message")]
        return list(request.input)

    async def _require_replayable_response(
        self,
        connection: AsyncConnection,
        scope_id: UUID,
        response_id: str,
        *,
        param: str,
    ) -> None:
        exists = await self._response_exists(connection, scope_id, response_id)
        if not exists:
            raise ResponseError.not_found(
                private_message=f"previous response was not found: {response_id}",
                public_message="Previous response not found.",
                public_code="previous_response_not_found",
                param=param,
            )

    async def _conversation_head(self, connection: AsyncConnection, scope_id: UUID, conversation_id: str) -> str | None:
        result = await connection.execute(
            text(
                """
                select current_response_id
                  from responses.conversations
                 where scope_id = :scope_id
                   and conversation_id = :conversation_id
                """
            ),
            {"scope_id": scope_id, "conversation_id": conversation_id},
        )
        try:
            return result.scalar_one_or_none()
        finally:
            result.close()

    async def _replay_items(self, connection: AsyncConnection, scope_id: UUID, response_id: str) -> list[RequestInputItem]:
        result = await connection.execute(
            text(
                """
                select payload
                  from responses.list_response_replay(:scope_id, :response_id)
                """
            ),
            {"scope_id": scope_id, "response_id": response_id},
        )
        try:
            rows = result.scalars().all()
        finally:
            result.close()
        return [self._request_input_from_payload(payload) for payload in rows]

    @staticmethod
    def _request_input_from_payload(payload: Mapping[str, object]) -> RequestInputItem:
        normalized = dict(payload)
        if normalized.get("type") in {"compaction", "function_call_output"}:
            normalized.pop("created_by", None)
        return _REQUEST_INPUT_ADAPTER.validate_python(normalized)

    @staticmethod
    def _stored_input_payloads(response_id: str, items: list[RequestInputItem]) -> list[PayloadObject]:
        payloads: list[PayloadObject] = []
        for index, item in enumerate(items):
            payload = item.model_dump(mode="json", exclude_none=True)
            if payload.get("id") is None:
                payload["id"] = f"in_{response_id}_{index}"
            payloads.append(payload)
        return payloads

    @staticmethod
    def _response_fields(response: ResponseObject) -> ResponseFields:
        return cast(
            ResponseFields,
            response.model_dump(
                mode="json",
                exclude={"completed_at", "id", "object", "output", "status"},
            ),
        )

    @staticmethod
    def _completed_at(response: ResponseObject) -> datetime | None:
        if response.completed_at is None:
            return None
        return datetime.fromtimestamp(response.completed_at, UTC)

    @staticmethod
    def _json_text(value: object) -> str:
        return msgspec.json.encode(value).decode()

    async def _response_exists(self, connection: AsyncConnection, scope_id: UUID, response_id: str) -> bool:
        result = await connection.execute(
            text(
                """
                select 1
                  from responses.response_records record
                 where record.scope_id = :scope_id
                   and record.response_id = :response_id
                   and not exists (
                     select 1
                       from responses.response_tombstones tombstone
                      where tombstone.scope_id = record.scope_id
                        and tombstone.response_id = record.response_id
                   )
                """
            ),
            {"scope_id": scope_id, "response_id": response_id},
        )
        try:
            return bool(result.scalar_one_or_none())
        finally:
            result.close()

    async def _output_payloads(self, connection: AsyncConnection, scope_id: UUID, response_id: str) -> list[PayloadObject]:
        result = await connection.execute(
            text(
                """
                select payload.payload_json
                  from responses.response_output_items item
                  join responses.payloads payload
                    on payload.scope_id = item.scope_id
                   and payload.payload_id = item.payload_id
                 where item.scope_id = :scope_id
                   and item.response_id = :response_id
                 order by item.output_index
                """
            ),
            {"scope_id": scope_id, "response_id": response_id},
        )
        try:
            return [dict(cast(Mapping[str, object], payload)) for payload in result.scalars().all()]
        finally:
            result.close()

    async def _input_payloads(self, connection: AsyncConnection, scope_id: UUID, response_id: str) -> list[PayloadObject]:
        result = await connection.execute(
            text(
                """
                select payload.payload_json
                  from responses.response_input_items item
                  join responses.payloads payload
                    on payload.scope_id = item.scope_id
                   and payload.payload_id = item.payload_id
                 where item.scope_id = :scope_id
                   and item.response_id = :response_id
                 order by item.input_index
                """
            ),
            {"scope_id": scope_id, "response_id": response_id},
        )
        try:
            return [dict(cast(Mapping[str, object], payload)) for payload in result.scalars().all()]
        finally:
            result.close()

    @staticmethod
    def _input_items_page_item_from_payload(payload: Mapping[str, object]) -> InputItemsPageItem:
        normalized = dict(payload)
        if normalized.get("type") in {"compaction", "function_call_output"}:
            normalized.pop("created_by", None)
        return _INPUT_ITEMS_PAGE_ADAPTER.validate_python(normalized)

    @staticmethod
    def _response_from_record(record: RowMapping, output_items: list[PayloadObject]) -> ResponseObject:
        completed_at_value = cast(datetime | None, record["completed_at"])
        completed_at = completed_at_value.timestamp() if completed_at_value is not None else None
        value = dict(cast(Mapping[str, object], record["fields"]))
        value.update(
            {
                "completed_at": completed_at,
                "id": str(record["response_id"]),
                "object": "response",
                "output": output_items,
                "status": str(record["status"]),
            }
        )
        return ResponseObject.model_validate(value)

    async def _mappings_one_or_none(
        self,
        connection: AsyncConnection,
        statement: TextClause,
        parameters: Mapping[str, object],
    ) -> RowMapping | None:
        result = await connection.execute(statement, parameters)
        try:
            return result.mappings().one_or_none()
        finally:
            result.close()
