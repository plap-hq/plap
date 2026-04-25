from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

type JSONPayload = dict[str, Any]


@dataclass(frozen=True, slots=True)
class StateItem:
    namespace: str
    ordinal: int
    payload: JSONPayload
    payload_hash: str | None = None
    position: int | None = None


@dataclass(frozen=True, slots=True)
class NamespaceCursor:
    namespace: str
    next_ordinal: int


@dataclass(frozen=True, slots=True)
class StateCheckpoint:
    state_root_id: int
    namespace_cursors: tuple[NamespaceCursor, ...]


@dataclass(frozen=True, slots=True)
class ResponseRecord:
    response_id: str
    previous_response_id: str | None
    state_root_id: int
    output_state_root_id: int
    status: str
    created_at: datetime
    completed_at: datetime | None
    fields: JSONPayload


@dataclass(frozen=True, slots=True)
class AppendResponseResult:
    response_id: str
    state_root_id: int
    output_state_root_id: int
