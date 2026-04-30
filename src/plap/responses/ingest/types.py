from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import blake3
import msgspec

type Side = Literal["main", "reviewer", "arbitrator"]

ChatMessage = dict[str, Any]


def chat_message_hash(message: ChatMessage) -> str:
    return blake3.blake3(msgspec.json.encode(message, order="deterministic")).hexdigest()


@dataclass(frozen=True, slots=True)
class ChatMessageSpan:
    start: int
    end: int
    message: ChatMessage
    token_count: int
    content_hash: str = ""
    children_token_count: int = 0
    expanded_token_count: int = 0
    children: tuple[ChatMessageSpan, ...] = ()
    children_pruned: bool = False

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < 0:
            raise ValueError("message span bounds must be non-negative")
        if self.start > self.end:
            raise ValueError("message span start must not exceed end")
        if self.token_count <= 0:
            raise ValueError("message span token_count must be positive")
        for field_name, value in (
            ("children_token_count", self.children_token_count),
            ("expanded_token_count", self.expanded_token_count),
        ):
            if value < 0:
                raise ValueError(f"message span {field_name} must be non-negative")
        if not self.content_hash:
            object.__setattr__(self, "content_hash", chat_message_hash(self.message))
        if self.children:
            if not self.children_token_count:
                object.__setattr__(
                    self,
                    "children_token_count",
                    sum(child.token_count for child in self.children),
                )
            if not self.expanded_token_count:
                object.__setattr__(
                    self,
                    "expanded_token_count",
                    sum(child.expanded_token_count for child in self.children),
                )
        elif self.is_leaf and not self.expanded_token_count:
            object.__setattr__(self, "expanded_token_count", self.token_count)

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
    cursors: dict[str, int]


@dataclass(frozen=True, slots=True)
class ReasoningPayload:
    side: Side
    temp: bool
    messages: tuple[ChatMessage, ...]
    continuation_side: Side | None = None


@dataclass(frozen=True, slots=True)
class SealedCallID:
    side: Side
    content_hash_prefix: bytes
    tool_call_index: int
    upstream_tool_call_id: str


@dataclass(frozen=True, slots=True)
class IngestedQueues:
    main_context: tuple[ChatMessageSpan, ...]
    main_context_temp: tuple[ChatMessageSpan, ...]
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
