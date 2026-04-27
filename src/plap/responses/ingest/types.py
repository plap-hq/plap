from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import blake3
import msgspec

from plap.responses.tools import ToolPolicy

type Side = Literal["main", "reviewer", "arbitrator"]
type Namespace = Literal["m", "s"]

ChatMessage = dict[str, Any]


def chat_message_hash(message: ChatMessage) -> str:
    return blake3.blake3(
        msgspec.json.encode(message, order="deterministic")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ChatMessageWithOrdinal:
    namespace: Namespace
    ordinal: int
    message: ChatMessage
    content_hash: str = ""

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("message row ordinal must be non-negative")
        if self.namespace not in {"m", "s"}:
            raise ValueError("message row namespace is invalid")
        if not self.content_hash:
            object.__setattr__(self, "content_hash", chat_message_hash(self.message))


@dataclass(frozen=True, slots=True)
class SideMessage:
    message: ChatMessage
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            object.__setattr__(self, "content_hash", chat_message_hash(self.message))


@dataclass(frozen=True, slots=True)
class CompactionPayload:
    active: tuple[ChatMessageWithOrdinal, ...]
    source: tuple[ChatMessageWithOrdinal, ...]
    cursors: dict[str, int]


@dataclass(frozen=True, slots=True)
class ReasoningPayload:
    side: Side
    temp: bool
    messages: tuple[ChatMessage, ...]


@dataclass(frozen=True, slots=True)
class SealedCallID:
    side: Side
    content_hash_prefix: bytes
    tool_call_index: int
    upstream_tool_call_id: str


@dataclass(frozen=True, slots=True)
class IngestedQueues:
    main: tuple[ChatMessageWithOrdinal, ...]
    reviewer: tuple[SideMessage, ...]
    arbitrator: tuple[SideMessage, ...]
    compaction: CompactionPayload | None
    source: tuple[ChatMessageWithOrdinal, ...]
    cursors: dict[str, int]
    tool_policies: dict[str, ToolPolicy]
    diagnostics: tuple[str, ...] = field(default_factory=tuple)


class IngestionError(ValueError):
    pass
