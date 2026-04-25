from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import TypeAdapter

from plap.responses.contracts import (
    ResponseCreateRequest,
    ResponseObject,
    ResponseOutputItem,
)
from plap.responses.contracts.items import ReasoningItem, ResponseCompactionItem
from plap.responses.state import (
    NamespaceCursor,
    ResponseOutputEntry,
    ResponseOutputManifestItem,
    ResponseRecord,
    ResponseRepository,
    StateItem,
)

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

_DESCRIPTOR_KEYS = {
    "call_id",
    "created_by",
    "id",
    "name",
    "namespace",
    "phase",
    "role",
    "status",
    "type",
}


def fields_from_request(request: ResponseCreateRequest) -> dict[str, Any]:
    dumped = request.model_dump(mode="json", exclude_none=True)
    return {key: value for key, value in dumped.items() if key in _FIELD_KEYS}


def encode_output_items(
    output_items: Sequence[ResponseOutputItem],
    previous_cursors: dict[str, int],
) -> tuple[
    list[StateItem],
    list[ResponseOutputManifestItem],
    tuple[NamespaceCursor, ...],
]:
    cursors = {
        "m": previous_cursors.get("m", 0),
        "r": previous_cursors.get("r", 0),
        "s": previous_cursors.get("s", 0),
    }
    state_items: list[StateItem] = []
    output_manifest: list[ResponseOutputManifestItem] = []

    for item in output_items:
        namespace = state_namespace(item)
        ordinal = cursors[namespace]
        cursors[namespace] += 1

        payload, descriptor = split_output_item(item)
        payload_hash = ResponseRepository.payload_hash(payload)
        state_items.append(
            StateItem(
                namespace=namespace,
                ordinal=ordinal,
                payload=payload,
                payload_hash=payload_hash,
            )
        )
        output_manifest.append(
            ResponseOutputManifestItem(
                type=descriptor["type"],
                namespace=namespace,
                ordinal=ordinal,
                payload_hash=payload_hash,
                descriptor=descriptor,
            )
        )

    namespace_cursors = tuple(
        NamespaceCursor(namespace=namespace, next_ordinal=next_ordinal)
        for namespace, next_ordinal in sorted(cursors.items())
    )
    return state_items, output_manifest, namespace_cursors


def decode_response(
    record: ResponseRecord,
    output_entries: Sequence[ResponseOutputEntry],
) -> ResponseObject:
    output_items = [decode_output_item(entry) for entry in output_entries]
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


def decode_output_item(entry: ResponseOutputEntry) -> ResponseOutputItem:
    if entry.payload is None:
        raise ValueError(f"response output {entry.output_index} has no payload")
    return _RESPONSE_ITEM_ADAPTER.validate_python({**entry.payload, **entry.descriptor})


def split_output_item(
    item: ResponseOutputItem,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dumped = item.model_dump(mode="json", exclude_none=True)
    descriptor = {
        key: value for key, value in dumped.items() if key in _DESCRIPTOR_KEYS
    }
    payload = {
        key: value for key, value in dumped.items() if key not in _DESCRIPTOR_KEYS
    }
    return payload, descriptor


def state_namespace(item: ResponseOutputItem) -> str:
    if isinstance(item, ReasoningItem):
        return "r"
    if isinstance(item, ResponseCompactionItem):
        return "s"
    return "m"
