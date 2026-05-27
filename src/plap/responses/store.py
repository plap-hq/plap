from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID

import msgspec
import structlog
from pydantic import TypeAdapter
from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncConnection
from sqlalchemy.sql.elements import TextClause

from plap.auth import AuthContext
from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.logging import log_debug, log_payload
from plap.persistence import Database
from plap.responses.contracts import (
    ConversationReference,
    InputItemsPage,
    InputItemsPageItem,
    RequestInputItem,
    RequestItemReference,
    RequestMessageItem,
    RequestReasoningItem,
    ResponseCreateRequest,
    ResponseObject,
)

_REQUEST_INPUT_ADAPTER = TypeAdapter(RequestInputItem)
_INPUT_ITEMS_PAGE_ADAPTER = TypeAdapter(InputItemsPageItem)
logger = structlog.get_logger(__name__)

type PayloadObject = dict[str, object]
type ResponseFields = dict[str, object]


def _conflicting_continuation_parameters_error() -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code="invalid_request_conflict",
            message="Parameters 'previous_response_id' and 'conversation' cannot be used together.",
            param="previous_response_id",
        ),
        private=PrivateError(
            event="response.store.invalid_request",
            reason="previous_response_and_conversation_conflict",
            message="previous_response_id and conversation cannot be combined",
            level=ErrorLevel.WARNING,
        ),
    )


def _conversation_requires_store_error() -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code="conversation_requires_store",
            message="Conversation continuation requires 'store' to be enabled.",
            param="store",
        ),
        private=PrivateError(
            event="response.store.invalid_request",
            reason="conversation_requires_store",
            message="conversation continuation requires a stored response",
            level=ErrorLevel.WARNING,
        ),
    )


def _previous_response_not_found_error(response_id: str, *, param: str) -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=404,
            type="not_found_error",
            code="previous_response_not_found",
            message=f"Previous response '{response_id}' not found.",
            param=param,
        ),
        private=PrivateError(
            event="response.store.not_found",
            reason="previous_response_not_found",
            message=f"previous response was not found: {response_id}",
            level=ErrorLevel.WARNING,
            context={"response_id": response_id},
        ),
    )


def _response_not_found_error(response_id: str, *, action: str) -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=404,
            type="not_found_error",
            code="response_not_found",
            message=f"Response '{response_id}' not found.",
        ),
        private=PrivateError(
            event="response.store.not_found",
            reason="response_not_found",
            message=f"response not found for {action}: {response_id}",
            level=ErrorLevel.WARNING,
            context={"action": action, "response_id": response_id},
        ),
    )


def _input_items_order_error() -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code="invalid_cursor_order",
            message="Parameter 'order' must be 'asc' or 'desc'.",
            param="order",
        ),
        private=PrivateError(
            event="response.store.invalid_request",
            reason="invalid_input_items_order",
            message="input_items order must be asc or desc",
            level=ErrorLevel.WARNING,
        ),
    )


def _input_items_after_not_found_error() -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code="cursor_not_found",
            message="Input items cursor was not found.",
            param="after",
        ),
        private=PrivateError(
            event="response.store.invalid_request",
            reason="input_items_after_not_found",
            message="input_items after cursor was not found",
            level=ErrorLevel.WARNING,
        ),
    )


def _item_reference_not_found_error(item_id: str) -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code="item_not_found",
            message=f"Item with id '{item_id}' not found.",
            param="input",
        ),
        private=PrivateError(
            event="response.store.invalid_request",
            reason="item_reference_not_found",
            message=f"item_reference target was not found: {item_id}",
            level=ErrorLevel.WARNING,
            context={"item_id": item_id},
        ),
    )


def _item_reference_ambiguous_error(item_id: str) -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code="item_reference_ambiguous",
            message=f"Item reference '{item_id}' is ambiguous.",
            param="input",
        ),
        private=PrivateError(
            event="response.store.invalid_request",
            reason="item_reference_ambiguous",
            message=f"item_reference matched multiple stored items: {item_id}",
            level=ErrorLevel.WARNING,
            context={"item_id": item_id},
        ),
    )


@dataclass(slots=True)
class PreparedRequest:
    scope_id: UUID
    response_request: ResponseCreateRequest
    execution_request: ResponseCreateRequest
    current_input_items: list[RequestInputItem]
    stored_input_items: list[RequestInputItem]
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
            raise _conflicting_continuation_parameters_error()
        if conversation_id is not None and request.store is False:
            raise _conversation_requires_store_error()

        current_input_items = self._current_input_items(request)
        stored_input_items = self._stored_current_input_items(current_input_items)
        parent_response_id = request.previous_response_id
        replay_items: list[RequestInputItem] = []
        execution_replay_items: list[RequestInputItem] = []
        execution_current_input_items: list[RequestInputItem] = list(current_input_items)
        if parent_response_id is not None or conversation_id is not None or self._needs_execution_resolution(current_input_items):
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
                execution_replay_items = await self._resolve_execution_items(connection, scope_id, replay_items)
                execution_current_input_items = await self._resolve_execution_items(connection, scope_id, current_input_items)
        else:
            execution_replay_items = replay_items

        response_request = request.model_copy(update={"previous_response_id": parent_response_id})
        execution_request = request.model_copy(update={"input": [*execution_replay_items, *execution_current_input_items]})
        log_debug(
            logger,
            "response.store.prepared",
            conversation_id=conversation_id,
            current_input_items=len(current_input_items),
            parent_response_id=parent_response_id,
            persist_response=request.store is not False,
            replay_items=len(replay_items),
        )
        log_payload(
            logger,
            "response.store.prepared.payload",
            current_input_items=[item.model_dump(mode="json", exclude_none=True) for item in current_input_items],
            execution_request=execution_request.model_dump(mode="json", exclude_none=True),
            response_request=response_request.model_dump(mode="json", exclude_none=True),
        )
        return PreparedRequest(
            scope_id=scope_id,
            response_request=response_request,
            execution_request=execution_request,
            current_input_items=current_input_items,
            stored_input_items=stored_input_items,
            parent_response_id=parent_response_id,
            conversation_id=conversation_id,
            persist_response=request.store is not False,
        )

    async def begin_response(self, prepared: PreparedRequest, response: ResponseObject) -> None:
        if not prepared.persist_response:
            return
        input_items = self._stored_input_payloads(response.id, prepared.stored_input_items)
        log_debug(
            logger,
            "response.store.begin",
            input_item_count=len(input_items),
            response_id=response.id,
        )
        log_payload(
            logger,
            "response.store.begin.payload",
            input_items=input_items,
            response=response.model_dump(mode="json", exclude_none=True),
        )
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
        log_debug(
            logger,
            "response.store.append_output_item",
            output_index=output_index,
            response_id=response_id,
            type=item.get("type") if isinstance(item, dict) else None,
        )
        log_payload(
            logger,
            "response.store.append_output_item.payload",
            item=item,
            output_index=output_index,
            response_id=response_id,
        )
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

    async def replace_output_item(
        self,
        prepared: PreparedRequest,
        response_id: str,
        output_index: int,
        item: object,
    ) -> None:
        if not prepared.persist_response:
            return
        log_debug(
            logger,
            "response.store.replace_output_item",
            output_index=output_index,
            response_id=response_id,
            type=item.get("type") if isinstance(item, dict) else None,
        )
        log_payload(
            logger,
            "response.store.replace_output_item.payload",
            item=item,
            output_index=output_index,
            response_id=response_id,
        )
        async with self._database.connection_transaction() as connection:
            delete_result = await connection.execute(
                text(
                    """
                    delete from responses.response_output_items
                     where scope_id = :scope_id
                       and response_id = :response_id
                       and output_index = :output_index
                     returning output_index
                    """
                ),
                {
                    "scope_id": prepared.scope_id,
                    "response_id": response_id,
                    "output_index": output_index,
                },
            )
            deleted_output_index = delete_result.scalar_one_or_none()
            delete_result.close()
            if deleted_output_index is None:
                raise RuntimeError("response output item to replace was not found")
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
        log_debug(
            logger,
            "response.store.finish",
            output_count=len(response.output),
            response_id=response.id,
            status=response.status,
        )
        log_payload(
            logger,
            "response.store.finish.payload",
            response=response.model_dump(mode="json", exclude_none=True),
        )
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
                    text("select responses.move_conversation_head(:scope_id, :conversation_id, :response_id, :retention)"),
                    {
                        "scope_id": prepared.scope_id,
                        "conversation_id": prepared.conversation_id,
                        "response_id": response.id,
                        "retention": None,
                    },
                )
                move_head_result.scalar_one_or_none()
                move_head_result.close()

    async def fail_response(self, prepared: PreparedRequest, response_id: str) -> bool:
        if not prepared.persist_response:
            return False
        completed_at = datetime.now(UTC)
        log_debug(
            logger,
            "response.store.fail",
            response_id=response_id,
        )
        async with self._database.connection_transaction() as connection:
            update_result = await connection.execute(
                text(
                    """
                    update responses.response_records
                       set status = 'failed',
                           completed_at = :completed_at
                     where scope_id = :scope_id
                       and response_id = :response_id
                       and status in ('queued', 'in_progress')
                    returning response_id
                    """
                ),
                {
                    "completed_at": completed_at,
                    "scope_id": prepared.scope_id,
                    "response_id": response_id,
                },
            )
            failed_response_id = update_result.scalar_one_or_none()
            update_result.close()
            return failed_response_id is not None

    async def cancel_response(self, prepared: PreparedRequest, response: ResponseObject) -> bool:
        if not prepared.persist_response:
            return False
        log_debug(
            logger,
            "response.store.cancel",
            response_id=response.id,
            status=response.status,
        )
        log_payload(
            logger,
            "response.store.cancel.payload",
            response=response.model_dump(mode="json", exclude_none=True),
        )
        async with self._database.connection_transaction() as connection:
            update_result = await connection.execute(
                text(
                    """
                    update responses.response_records
                       set status = 'cancelled',
                           completed_at = :completed_at,
                           fields = cast(:fields as jsonb)
                     where scope_id = :scope_id
                       and response_id = :response_id
                       and status in ('queued', 'in_progress')
                    returning response_id
                    """
                ),
                {
                    "completed_at": self._completed_at(response),
                    "fields": self._json_text(self._response_fields(response)),
                    "scope_id": prepared.scope_id,
                    "response_id": response.id,
                },
            )
            cancelled_response_id = update_result.scalar_one_or_none()
            update_result.close()
            return cancelled_response_id is not None

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
            raise _input_items_order_error()
        page_limit = limit or 20
        async with self._database.connection() as connection:
            exists = await self._response_exists(connection, scope_id, response_id)
            if not exists:
                raise _response_not_found_error(response_id, action="input_items")
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
                raise _input_items_after_not_found_error()
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
                text("delete from responses.conversations where scope_id = :scope_id and current_response_id = :response_id"),
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

    @staticmethod
    def _is_implicit_reasoning_replay(item: RequestInputItem) -> bool:
        return isinstance(item, RequestReasoningItem) and item.id is not None and item.encrypted_content is None

    @classmethod
    def _needs_execution_resolution(cls, items: list[RequestInputItem]) -> bool:
        return any(isinstance(item, RequestItemReference) or cls._is_implicit_reasoning_replay(item) for item in items)

    @classmethod
    def _stored_current_input_items(cls, items: list[RequestInputItem]) -> list[RequestInputItem]:
        stored: list[RequestInputItem] = []
        for item in items:
            if cls._is_implicit_reasoning_replay(item):
                stored.append(RequestItemReference(id=item.id, type="item_reference"))
                continue
            stored.append(item)
        return stored

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
            raise _previous_response_not_found_error(response_id, param=param)

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

    async def _resolve_execution_items(
        self,
        connection: AsyncConnection,
        scope_id: UUID,
        items: list[RequestInputItem],
    ) -> list[RequestInputItem]:
        resolved: list[RequestInputItem] = []
        for item in items:
            if isinstance(item, RequestItemReference):
                resolved.append(await self._resolve_item_reference(connection, scope_id, item.id))
                continue
            if self._is_implicit_reasoning_replay(item):
                resolved_item = await self._resolve_implicit_reasoning_replay(connection, scope_id, item.id)
                if resolved_item is not None:
                    resolved.append(resolved_item)
                    continue
                resolved.append(item)
                continue
            resolved.append(item)
        return resolved

    async def _resolve_implicit_reasoning_replay(
        self,
        connection: AsyncConnection,
        scope_id: UUID,
        item_id: str | None,
    ) -> RequestInputItem | None:
        if item_id is None:
            return None
        payloads = await self._stored_item_payloads(connection, scope_id, item_id)
        if not payloads:
            return None
        if len(payloads) > 1:
            raise _item_reference_ambiguous_error(item_id)
        resolved = self._request_input_from_payload(payloads[0])
        if not isinstance(resolved, RequestReasoningItem):
            return None
        return resolved

    async def _resolve_item_reference(
        self,
        connection: AsyncConnection,
        scope_id: UUID,
        item_id: str,
    ) -> RequestInputItem:
        payloads = await self._stored_item_payloads(connection, scope_id, item_id)
        if not payloads:
            raise _item_reference_not_found_error(item_id)
        if len(payloads) > 1:
            raise _item_reference_ambiguous_error(item_id)
        return self._request_input_from_payload(payloads[0])

    async def _stored_item_payloads(
        self,
        connection: AsyncConnection,
        scope_id: UUID,
        item_id: str,
    ) -> list[PayloadObject]:
        result = await connection.execute(
            text(
                """
                select payload.payload_json
                  from responses.response_input_items item
                  join responses.payloads payload
                    on payload.scope_id = item.scope_id
                   and payload.payload_id = item.payload_id
                 where item.scope_id = :scope_id
                   and payload.payload_json ->> 'id' = :item_id
                   and coalesce(payload.payload_json ->> 'type', '') <> 'item_reference'
                   and not exists (
                     select 1
                       from responses.response_tombstones tombstone
                      where tombstone.scope_id = item.scope_id
                        and tombstone.response_id = item.response_id
                   )
                union all
                select payload.payload_json
                  from responses.response_output_items item
                  join responses.payloads payload
                    on payload.scope_id = item.scope_id
                   and payload.payload_id = item.payload_id
                 where item.scope_id = :scope_id
                   and payload.payload_json ->> 'id' = :item_id
                   and not exists (
                     select 1
                       from responses.response_tombstones tombstone
                      where tombstone.scope_id = item.scope_id
                        and tombstone.response_id = item.response_id
                   )
                """
            ),
            {"scope_id": scope_id, "item_id": item_id},
        )
        try:
            return [dict(cast(Mapping[str, object], payload)) for payload in result.scalars().all()]
        finally:
            result.close()

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
