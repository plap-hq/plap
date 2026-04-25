from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any
from uuid import UUID

from pydantic import TypeAdapter

from plap.responses.contracts import (
    RequestInputItem,
    ResponseCreateRequest,
    ResponseObject,
    ResponseOutputItem,
    ResponseStatus,
)
from plap.responses.contracts.items import ReasoningItem, ResponseCompactionItem
from plap.responses.state.repository import ResponseRepository
from plap.responses.state.types import NamespaceCursor, ResponseRecord, StateItem

_REQUEST_ITEM_ADAPTER = TypeAdapter(RequestInputItem)
_RESPONSE_ITEM_ADAPTER = TypeAdapter(ResponseOutputItem)

_FIELD_KEYS = {
    "conversation",
    "instructions",
    "max_output_tokens",
    "max_tool_calls",
    "metadata",
    "model",
    "parallel_tool_calls",
    "prompt",
    "prompt_cache_key",
    "reasoning",
    "safety_identifier",
    "service_tier",
    "temperature",
    "text",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "truncation",
    "user",
}


@dataclass(frozen=True, slots=True)
class StoredResponseItem:
    position: int
    namespace: str
    ordinal: int
    item: RequestInputItem | ResponseOutputItem


@dataclass(frozen=True, slots=True)
class StoredResponse:
    record: ResponseRecord
    items: list[StoredResponseItem]


class ResponseStore:
    def __init__(self, repository: ResponseRepository) -> None:
        self._repository = repository

    async def get_record(
        self,
        scope_id: UUID,
        response_id: str,
    ) -> ResponseRecord | None:
        return await self._repository.get_response_record(scope_id, response_id)

    async def list_state_items(
        self,
        scope_id: UUID,
        response_id: str,
    ) -> list[StoredResponseItem]:
        record = await self._repository.get_response_record(scope_id, response_id)
        if record is None:
            return []

        items = await self._repository.list_items(scope_id, record.state_root_id)
        return [self._stored_item(item) for item in items]

    async def retrieve_response(
        self,
        scope_id: UUID,
        response_id: str,
    ) -> ResponseObject | None:
        record = await self._repository.get_response_record(scope_id, response_id)
        if record is None:
            return None

        output_items = await self._list_output_items(scope_id, record)
        fields = dict(record.fields)
        return ResponseObject.model_validate(
            {
                **fields,
                "id": record.response_id,
                "created_at": record.created_at.timestamp(),
                "completed_at": record.completed_at.timestamp()
                if record.completed_at is not None
                else None,
                "object": "response",
                "output": [item.model_dump(mode="json") for item in output_items],
                "previous_response_id": record.previous_response_id,
                "status": record.status,
            }
        )

    async def create_response(
        self,
        scope_id: UUID,
        response_id: str,
        previous_response_id: str | None,
        output_items: Sequence[ResponseOutputItem],
        request: ResponseCreateRequest,
        *,
        conversation_id: str | None = None,
        retention: timedelta | None = timedelta(days=30),
        status: ResponseStatus = "completed",
    ) -> ResponseRecord:
        previous_cursors = await self._previous_cursors(scope_id, previous_response_id)
        state_items, namespace_cursors = self._state_items_from_output(
            output_items,
            previous_cursors,
        )
        fields = self._fields_from_request(request)
        result = await self._repository.append_response(
            scope_id,
            response_id,
            previous_response_id,
            state_items,
            namespace_cursors,
            retention=retention,
            status=status,
            fields=fields,
        )
        if conversation_id is not None:
            await self._repository.move_conversation_head(
                scope_id,
                conversation_id,
                response_id,
                retention=retention,
            )

        record = await self._repository.get_response_record(
            scope_id, result.response_id
        )
        if record is None:
            raise RuntimeError(f"created response {response_id!r} was not persisted")
        return record

    async def _previous_cursors(
        self,
        scope_id: UUID,
        previous_response_id: str | None,
    ) -> dict[str, int]:
        if previous_response_id is None:
            return {"m": 0, "r": 0, "s": 0}
        cursors = await self._repository.get_namespace_cursors(
            scope_id,
            previous_response_id,
        )
        return {cursor.namespace: cursor.next_ordinal for cursor in cursors}

    async def _list_output_items(
        self,
        scope_id: UUID,
        record: ResponseRecord,
    ) -> list[ResponseOutputItem]:
        items = await self._repository.list_items(scope_id, record.output_state_root_id)
        return [_RESPONSE_ITEM_ADAPTER.validate_python(item.payload) for item in items]

    @staticmethod
    def _fields_from_request(request: ResponseCreateRequest) -> dict[str, Any]:
        dumped = request.model_dump(mode="json", exclude_none=True)
        return {key: value for key, value in dumped.items() if key in _FIELD_KEYS}

    @staticmethod
    def _is_response_item(item: RequestInputItem | ResponseOutputItem) -> bool:
        try:
            _RESPONSE_ITEM_ADAPTER.validate_python(item.model_dump(mode="json"))
        except ValueError:
            return False
        return True

    @staticmethod
    def _state_namespace(item: ResponseOutputItem) -> str:
        if isinstance(item, ReasoningItem):
            return "r"
        if isinstance(item, ResponseCompactionItem):
            return "s"
        return "m"

    @classmethod
    def _state_items_from_output(
        cls,
        output_items: Sequence[ResponseOutputItem],
        previous_cursors: dict[str, int],
    ) -> tuple[list[StateItem], tuple[NamespaceCursor, ...]]:
        cursors = {
            "m": previous_cursors.get("m", 0),
            "r": previous_cursors.get("r", 0),
            "s": previous_cursors.get("s", 0),
        }
        state_items: list[StateItem] = []
        for item in output_items:
            namespace = cls._state_namespace(item)
            ordinal = cursors[namespace]
            cursors[namespace] += 1
            state_items.append(
                StateItem(
                    namespace=namespace,
                    ordinal=ordinal,
                    payload=item.model_dump(mode="json"),
                )
            )
        return state_items, tuple(
            NamespaceCursor(namespace=namespace, next_ordinal=next_ordinal)
            for namespace, next_ordinal in sorted(cursors.items())
        )

    @staticmethod
    def _stored_item(item: StateItem) -> StoredResponseItem:
        parsed: RequestInputItem | ResponseOutputItem
        try:
            parsed = _RESPONSE_ITEM_ADAPTER.validate_python(item.payload)
        except ValueError:
            parsed = _REQUEST_ITEM_ADAPTER.validate_python(item.payload)

        if item.position is None:
            raise ValueError("state repository returned an item without a position")
        return StoredResponseItem(
            position=item.position,
            namespace=item.namespace,
            ordinal=item.ordinal,
            item=parsed,
        )
