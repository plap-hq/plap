from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import blake3
import msgspec

type Side = Literal["main", "reviewer", "arbitrator"]

ChatMessage = dict[str, Any]


def chat_message_hash(message: ChatMessage) -> str:
    return blake3.blake3(
        msgspec.json.encode(message, order="deterministic")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ChatMessageSpan:
    start: int
    end: int
    message: ChatMessage
    content_hash: str = ""

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < 0:
            raise ValueError("message span bounds must be non-negative")
        if self.start > self.end:
            raise ValueError("message span start must not exceed end")
        if not self.content_hash:
            object.__setattr__(self, "content_hash", chat_message_hash(self.message))

    @property
    def is_leaf(self) -> bool:
        return self.start == self.end

    @property
    def citation(self) -> str:
        if self.is_leaf:
            return f"[~{self.start}]"
        return f"[~{self.start}_{self.end}]"


@dataclass(frozen=True, slots=True)
class SideMessage:
    message: ChatMessage
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            object.__setattr__(self, "content_hash", chat_message_hash(self.message))


@dataclass(frozen=True, slots=True)
class CompactionPayload:
    active: tuple[ChatMessageSpan, ...]
    source: tuple[ChatMessageSpan, ...]
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
    main_context: tuple[ChatMessageSpan, ...]
    main_transcript: tuple[ChatMessageSpan, ...]
    reviewer: tuple[SideMessage, ...]
    arbitrator: tuple[SideMessage, ...]
    continuation_side: Side
    in_temp_debate: bool
    compaction: CompactionPayload | None
    cursors: dict[str, int]
    diagnostics: tuple[str, ...] = field(default_factory=tuple)


class IngestionError(ValueError):
    pass
