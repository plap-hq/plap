from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import ceil

import blake3
import msgspec

from plap.llms.chat import ChatMessage as LLMChatMessage
from plap.llms.chat import ChatRole, ChatUsage
from plap.llms.chat import ChatToolCall as LLMChatToolCall
from plap.llms.json_utils import JSONInvalidError, normalize_json_text_with_repair, parse_json_value_with_repair
from plap.responses.contracts import (
    ResponseUsage,
    ResponseUsageInputTokensDetails,
    ResponseUsageOutputTokensDetails,
)
from plap.settings import PublicUsageConfig

_LEADING_INTERNAL_CITATION_RE = re.compile(r"^\s*(?:(?:\[~\d+(?:_\d+)?\])\s+)+")


def strip_leading_internal_citations(text: str | None) -> str | None:
    if text is None:
        return None
    return _LEADING_INTERNAL_CITATION_RE.sub("", text, count=1)


class Side(StrEnum):
    MAIN = "main"
    DEFENDER = "defender"
    REVIEWER = "reviewer"
    ARBITRATOR = "arbitrator"


class Actor(StrEnum):
    MAIN = "main"
    DEFENDER = "defender"
    REVIEWER = "reviewer"
    ARBITRATOR = "arbitrator"


@dataclass(frozen=True, slots=True)
class StateToolCall:
    id: str
    name: str
    arguments: str

    def arguments_value(self) -> object:
        try:
            return parse_json_value_with_repair(self.arguments)
        except JSONInvalidError:
            return self.arguments

    def to_assistant_primitive(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }

    @classmethod
    def from_assistant_primitive(cls, value: object) -> StateToolCall:
        if not isinstance(value, dict):
            raise TypeError("tool call must be an object")
        call_id = value.get("id")
        name = value.get("name")
        arguments = value.get("arguments")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("tool call id is required")
        if not isinstance(name, str) or not name:
            raise ValueError("tool call name is required")
        if not isinstance(arguments, str):
            raise TypeError("tool call arguments is required")
        return cls(id=call_id, name=name, arguments=arguments)

    def to_chat_tool_call(self) -> LLMChatToolCall:
        return LLMChatToolCall(
            id=self.id,
            name=self.name,
            arguments=self.arguments,
        )

    def canonical_arguments(self) -> str:
        try:
            return normalize_json_text_with_repair(self.arguments)
        except JSONInvalidError:
            return self.arguments

    def deduplication_key(self) -> tuple[str, str]:
        return self.name, self.canonical_arguments()


@dataclass(slots=True)
class StateMessage:
    role: ChatRole
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[StateToolCall] = field(default_factory=list)
    reasoning_content: str | None = None
    reasoning_details: list[object] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.role = ChatRole(self.role)

    def content_hash(self) -> str:
        return blake3.blake3(msgspec.json.encode(self.to_primitive(), order="deterministic")).hexdigest()

    def content_text(self) -> str | None:
        return self.content

    def is_assistant(self) -> bool:
        return self.role == "assistant"

    def is_tool(self) -> bool:
        return self.role == "tool"

    def append_tool_call(self, tool_call: StateToolCall) -> None:
        self.tool_calls.append(tool_call)

    def tool_call_at(self, index: int) -> StateToolCall:
        return self.tool_calls[index]

    def to_primitive(self, *, include_reasoning: bool = True) -> dict[str, object]:
        value: dict[str, object] = {"role": self.role}
        if self.content is not None:
            value["content"] = self.content
        if self.name is not None:
            value["name"] = self.name
        if self.tool_call_id is not None:
            value["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            value["tool_calls"] = [call.to_assistant_primitive() for call in self.tool_calls]
        if include_reasoning and self.reasoning_content is not None:
            value["reasoning_content"] = self.reasoning_content
        if include_reasoning and self.reasoning_details:
            value["reasoning_details"] = list(self.reasoning_details)
        return value

    @classmethod
    def from_primitive(cls, value: object) -> StateMessage:
        if not isinstance(value, Mapping):
            raise TypeError("message must be an object")
        raw_role = value.get("role")
        try:
            role = ChatRole(raw_role)
        except ValueError as exc:
            raise ValueError("message role is invalid") from exc
        tool_calls_value = value.get("tool_calls")
        tool_calls: list[StateToolCall] = []
        if tool_calls_value is not None:
            if not isinstance(tool_calls_value, list):
                raise ValueError("message tool_calls must be an array")
            tool_calls = [StateToolCall.from_assistant_primitive(item) for item in tool_calls_value]
        reasoning_details_value = value.get("reasoning_details")
        reasoning_details: list[object] = []
        if reasoning_details_value is not None:
            if not isinstance(reasoning_details_value, list):
                raise ValueError("message reasoning_details must be an array")
            reasoning_details = list(reasoning_details_value)
        content = value.get("content")
        name = value.get("name")
        tool_call_id = value.get("tool_call_id")
        reasoning_content = value.get("reasoning_content")
        return cls(
            role=role,
            content=content if isinstance(content, str) else None,
            name=name if isinstance(name, str) else None,
            tool_call_id=tool_call_id if isinstance(tool_call_id, str) else None,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content if isinstance(reasoning_content, str) else None,
            reasoning_details=reasoning_details,
        )

    def to_chat_message(self, *, untrusted: bool = False, citation: str | None = None) -> LLMChatMessage:
        content = self.content_text() or ""
        if untrusted and self.role in {ChatRole.SYSTEM, ChatRole.DEVELOPER}:
            content = f"[^untrusted]\n{content}"
        role = self.role
        if citation is not None:
            content = f"{citation}\n{content}"
        return LLMChatMessage(
            role=role,
            content=content,
            name=self.name,
            tool_call_id=self.tool_call_id,
            tool_calls=[call.to_chat_tool_call() for call in self.tool_calls] or None,
            reasoning_content=self.reasoning_content,
            reasoning_details=list(self.reasoning_details) or None,
        )

    def to_transcript_message(self, *, untrusted: bool = False) -> TranscriptMessage:
        content = self.content_text()
        if untrusted and self.role in {ChatRole.SYSTEM, ChatRole.DEVELOPER}:
            content = f"[^untrusted]\n{content or ''}"
        if self.role == ChatRole.ASSISTANT:
            role = ChatRole.ASSISTANT
        elif self.role in {ChatRole.SYSTEM, ChatRole.DEVELOPER}:
            role = self.role
        else:
            role = ChatRole.USER
        return TranscriptMessage(
            role=role,
            content=content,
            tool_calls=tuple(
                TranscriptToolCall(
                    _id=call.id,
                    name=call.name,
                    arguments=call.arguments_value(),
                )
                for call in self.tool_calls
            ),
        )

    def duplicate_tool_call_ids(self, latest_call_id_by_key: Mapping[tuple[str, str], str]) -> set[str]:
        if not self.is_assistant() or not self.tool_calls:
            return set()
        return {call.id for call in self.tool_calls if latest_call_id_by_key.get(call.deduplication_key()) != call.id}

    def with_content(self, content: str | None) -> StateMessage:
        return StateMessage(
            role=self.role,
            content=content,
            name=self.name,
            tool_call_id=self.tool_call_id,
            tool_calls=list(self.tool_calls),
            reasoning_content=self.reasoning_content,
            reasoning_details=list(self.reasoning_details),
        )

    def without_reasoning(self) -> StateMessage:
        if self.reasoning_content is None and not self.reasoning_details:
            return self
        return StateMessage(
            role=self.role,
            content=self.content,
            name=self.name,
            tool_call_id=self.tool_call_id,
            tool_calls=list(self.tool_calls),
            reasoning_content=None,
            reasoning_details=[],
        )

    def with_duplicate_tool_output_tombstone(
        self,
        duplicate_call_ids: set[str],
        tombstone: str,
    ) -> StateMessage:
        if not self.is_tool() or self.tool_call_id not in duplicate_call_ids or self.content == tombstone:
            return self
        return self.with_content(tombstone)


@dataclass(frozen=True, slots=True)
class ReasoningMessagePatch:
    content_hash: str
    role: ChatRole | None = None
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[StateToolCall, ...] | None = None
    reasoning_content: str | None = None
    reasoning_details: tuple[object, ...] | None = None

    def apply_to(self, message: StateMessage) -> None:
        if self.role is not None:
            message.role = ChatRole(self.role)
        if self.content is not None:
            message.content = self.content
        if self.name is not None:
            message.name = self.name
        if self.tool_call_id is not None:
            message.tool_call_id = self.tool_call_id
        if self.tool_calls is not None:
            message.tool_calls = list(self.tool_calls)
        if self.reasoning_content is not None:
            message.reasoning_content = self.reasoning_content
        if self.reasoning_details is not None:
            message.reasoning_details = list(self.reasoning_details)

    def to_primitive(self) -> dict[str, object]:
        value: dict[str, object] = {"content_hash": self.content_hash}
        if self.role is not None:
            value["role"] = self.role
        if self.content is not None:
            value["content"] = self.content
        if self.name is not None:
            value["name"] = self.name
        if self.tool_call_id is not None:
            value["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            value["tool_calls"] = [call.to_assistant_primitive() for call in self.tool_calls]
        if self.reasoning_content is not None:
            value["reasoning_content"] = self.reasoning_content
        if self.reasoning_details is not None:
            value["reasoning_details"] = list(self.reasoning_details)
        return value

    @classmethod
    def from_primitive(cls, value: object) -> ReasoningMessagePatch:
        if not isinstance(value, Mapping):
            raise TypeError("reasoning patch must be an object")
        content_hash = value.get("content_hash")
        if not isinstance(content_hash, str) or not content_hash:
            raise ValueError("reasoning patch content_hash is required")
        raw_role = value.get("role")
        role = None
        if raw_role is not None:
            try:
                role = ChatRole(raw_role)
            except ValueError as exc:
                raise ValueError("reasoning patch role is invalid") from exc
        tool_calls_value = value.get("tool_calls")
        tool_calls = None
        if tool_calls_value is not None:
            if not isinstance(tool_calls_value, list):
                raise ValueError("reasoning patch tool_calls must be an array")
            tool_calls = tuple(StateToolCall.from_assistant_primitive(item) for item in tool_calls_value)
        reasoning_details_value = value.get("reasoning_details")
        reasoning_details = None
        if reasoning_details_value is not None:
            if not isinstance(reasoning_details_value, list):
                raise ValueError("reasoning patch reasoning_details must be an array")
            reasoning_details = tuple(reasoning_details_value)
        content = value.get("content")
        name = value.get("name")
        tool_call_id = value.get("tool_call_id")
        reasoning_content = value.get("reasoning_content")
        return cls(
            content_hash=content_hash,
            role=role,
            content=content if isinstance(content, str) else None,
            name=name if isinstance(name, str) else None,
            tool_call_id=tool_call_id if isinstance(tool_call_id, str) else None,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content if isinstance(reasoning_content, str) else None,
            reasoning_details=reasoning_details,
        )


@dataclass(frozen=True, slots=True)
class TranscriptToolCall:
    _id: str | None
    name: str
    arguments: object
    output: str | None = None

    def to_primitive(self) -> dict[str, object]:
        value: dict[str, object] = {"name": self.name, "arguments": self.arguments}
        if self.output is not None:
            value["output"] = self.output
        return value

    def without_id(self) -> TranscriptToolCall:
        return TranscriptToolCall(
            _id=None,
            name=self.name,
            arguments=self.arguments,
            output=self.output,
        )


@dataclass(frozen=True, slots=True)
class TranscriptMessage:
    role: ChatRole
    content: str | None = None
    tool_calls: tuple[TranscriptToolCall, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", ChatRole(self.role))

    def to_primitive(self) -> dict[str, object]:
        value: dict[str, object] = {"role": self.role}
        if self.content is not None:
            value["content"] = self.content
        if self.tool_calls:
            value["tool_calls"] = [call.to_primitive() for call in self.tool_calls]
        return value

    def without_ids(self) -> TranscriptMessage:
        return TranscriptMessage(
            role=self.role,
            content=self.content,
            tool_calls=tuple(call.without_id() for call in self.tool_calls),
        )


@dataclass(frozen=True, slots=True)
class ChatMessageSpan:
    start: int
    end: int
    message: StateMessage
    content_hash: str = ""
    children: tuple[ChatMessageSpan, ...] = ()

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < 0:
            raise ValueError("message span bounds must be non-negative")
        if self.start > self.end:
            raise ValueError("message span start must not exceed end")
        if not self.content_hash:
            object.__setattr__(self, "content_hash", self.message.content_hash())

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def citation(self) -> str:
        if self.start == self.end:
            return f"[~{self.start}]"
        return f"[~{self.start}_{self.end}]"

    def render_for_model(self, *, include_citation: bool) -> object:
        return self.message.to_chat_message(
            untrusted=True,
            citation=self.citation if include_citation else None,
        )

    def with_message(self, message: StateMessage) -> ChatMessageSpan:
        if message is self.message:
            return self
        return ChatMessageSpan(
            start=self.start,
            end=self.end,
            message=message,
            children=self.children,
        )

    def with_children(self, children: tuple[ChatMessageSpan, ...]) -> ChatMessageSpan:
        if children == self.children:
            return self
        return ChatMessageSpan(
            start=self.start,
            end=self.end,
            message=self.message,
            content_hash=self.content_hash,
            children=children,
        )

    def collect_latest_tool_call_ids(self, latest_call_id_by_key: dict[tuple[str, str], str]) -> None:
        if self.children:
            for child in self.children:
                child.collect_latest_tool_call_ids(latest_call_id_by_key)
            return
        for call in self.message.tool_calls:
            latest_call_id_by_key[call.deduplication_key()] = call.id

    def collect_duplicate_tool_call_ids(
        self,
        latest_call_id_by_key: Mapping[tuple[str, str], str],
        duplicate_call_ids: set[str],
    ) -> None:
        if self.children:
            for child in self.children:
                child.collect_duplicate_tool_call_ids(latest_call_id_by_key, duplicate_call_ids)
            return
        duplicate_call_ids.update(self.message.duplicate_tool_call_ids(latest_call_id_by_key))

    def collect_duplicate_tool_call_ids_before(
        self,
        latest_call_id_by_key: Mapping[tuple[str, str], str],
        duplicate_call_ids: set[str],
        *,
        before_start: int,
    ) -> None:
        if self.end < before_start:
            self.collect_duplicate_tool_call_ids(latest_call_id_by_key, duplicate_call_ids)
            return
        if self.start >= before_start or not self.children:
            return
        for child in self.children:
            child.collect_duplicate_tool_call_ids_before(
                latest_call_id_by_key,
                duplicate_call_ids,
                before_start=before_start,
            )

    def with_duplicate_tool_output_tombstones(
        self,
        duplicate_call_ids: set[str],
        tombstone: str,
    ) -> ChatMessageSpan:
        if self.children:
            children = tuple(child.with_duplicate_tool_output_tombstones(duplicate_call_ids, tombstone) for child in self.children)
            return self.with_children(children)
        return self.with_message(self.message.with_duplicate_tool_output_tombstone(duplicate_call_ids, tombstone))

    @classmethod
    def deduplicate_tool_call_outputs(
        cls,
        spans: list[ChatMessageSpan],
        *,
        tombstone: str,
        before_start: int | None = None,
    ) -> list[ChatMessageSpan]:
        latest_call_id_by_key: dict[tuple[str, str], str] = {}
        for span in spans:
            span.collect_latest_tool_call_ids(latest_call_id_by_key)
        duplicate_call_ids: set[str] = set()
        for span in spans:
            if before_start is None:
                span.collect_duplicate_tool_call_ids(latest_call_id_by_key, duplicate_call_ids)
                continue
            span.collect_duplicate_tool_call_ids_before(
                latest_call_id_by_key,
                duplicate_call_ids,
                before_start=before_start,
            )
        if not duplicate_call_ids:
            return spans
        return [span.with_duplicate_tool_output_tombstones(duplicate_call_ids, tombstone) for span in spans]

    def to_primitive(self) -> dict[str, object]:
        value: dict[str, object] = {
            "start": self.start,
            "end": self.end,
            "message": self.message.to_primitive(),
        }
        if self.content_hash:
            value["content_hash"] = self.content_hash
        if self.children:
            value["children"] = [child.to_primitive() for child in self.children]
        return value

    @classmethod
    def from_primitive(cls, value: object) -> ChatMessageSpan:
        if not isinstance(value, Mapping):
            raise TypeError("message span must be an object")
        children_value = value.get("children")
        children: tuple[ChatMessageSpan, ...] = ()
        if children_value is not None:
            if not isinstance(children_value, list):
                raise ValueError("message span children must be an array")
            children = tuple(cls.from_primitive(child) for child in children_value)
        message = StateMessage.from_primitive(value.get("message"))
        content_hash = value.get("content_hash")
        return cls(
            start=_required_int(value, "start"),
            end=_required_int(value, "end"),
            message=message,
            content_hash=content_hash if isinstance(content_hash, str) else "",
            children=children,
        )


@dataclass(frozen=True, slots=True)
class SideMessage:
    message: StateMessage
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            object.__setattr__(self, "content_hash", self.message.content_hash())


@dataclass(frozen=True, slots=True)
class CompactionPayload:
    active: tuple[ChatMessageSpan, ...]
    cursors: dict[str, int]

    def to_primitive(self) -> dict[str, object]:
        return {
            "active": [row.to_primitive() for row in self.active],
            "cursors": dict(self.cursors),
        }

    @classmethod
    def from_primitive(cls, value: object) -> CompactionPayload:
        if not isinstance(value, Mapping):
            raise TypeError("compaction payload must be an object")
        active_value = value.get("active")
        if not isinstance(active_value, list):
            raise TypeError("compaction active rows are required")
        cursors_value = value.get("cursors")
        if not isinstance(cursors_value, Mapping):
            raise TypeError("compaction cursors are required")
        return cls(
            active=tuple(ChatMessageSpan.from_primitive(row) for row in active_value),
            cursors={"m": _required_int(cursors_value, "m")},
        )


@dataclass(frozen=True, slots=True)
class ReasoningPayload:
    side: Side
    temp: bool
    messages: tuple[StateMessage | ReasoningMessagePatch, ...]
    continuation_side: Side

    def to_primitive(self) -> dict[str, object]:
        return {
            "side": self.side,
            "temp": self.temp,
            "messages": [message.to_primitive() for message in self.messages],
            "continuation_side": self.continuation_side,
        }

    @classmethod
    def from_primitive(cls, value: object) -> ReasoningPayload:
        if not isinstance(value, Mapping):
            raise TypeError("reasoning payload must be an object")
        raw_side = value.get("side")
        try:
            side = Side(raw_side)
        except ValueError as exc:
            raise ValueError("reasoning side is invalid") from exc
        temp = value.get("temp")
        if not isinstance(temp, bool):
            raise TypeError("reasoning temp flag is required")
        messages_value = value.get("messages")
        if not isinstance(messages_value, list):
            raise TypeError("reasoning messages must be an array")
        messages: list[StateMessage | ReasoningMessagePatch] = []
        for item in messages_value:
            if isinstance(item, Mapping) and isinstance(item.get("content_hash"), str):
                messages.append(ReasoningMessagePatch.from_primitive(item))
            else:
                messages.append(StateMessage.from_primitive(item))
        raw_continuation_side = value.get("continuation_side")
        if raw_continuation_side is None:
            raise TypeError("reasoning continuation_side is required")
        try:
            continuation_side = Side(raw_continuation_side)
        except ValueError as exc:
            raise ValueError("reasoning continuation_side is invalid") from exc
        return cls(
            side=side,
            temp=temp,
            messages=tuple(messages),
            continuation_side=continuation_side,
        )


@dataclass(frozen=True, slots=True)
class SealedCallID:
    side: Side
    temp: bool
    content_hash_prefix: bytes
    tool_call_index: int
    upstream_tool_call_id: str


@dataclass(frozen=True, slots=True)
class IngestedQueues:
    main_context: tuple[ChatMessageSpan, ...]
    defender: tuple[SideMessage, ...]
    reviewer: tuple[SideMessage, ...]
    arbitrator: tuple[SideMessage, ...]
    continuation_side: Side
    cursors: dict[str, int]


@dataclass(frozen=True, slots=True)
class DefenderParts:
    held_candidate: SideMessage | None
    held_hidden_tool_rows: tuple[SideMessage, ...]
    remaining_temp_rows: tuple[SideMessage, ...]


def append_main_context_row(
    rows: list[ChatMessageSpan],
    cursors: dict[str, int],
    message: StateMessage,
    *,
    content_hash: str = "",
) -> ChatMessageSpan:
    if message.is_tool():
        if not rows:
            raise ValueError("tool output requires a preceding main context segment")
        ordinal = rows[-1].end
    else:
        ordinal = cursors["m"]
        cursors["m"] = ordinal + 1
    row = ChatMessageSpan(
        start=ordinal,
        end=ordinal,
        message=message,
        content_hash=content_hash,
    )
    rows.append(row)
    return row


@dataclass(slots=True)
class MutableQueues:
    main_context: list[ChatMessageSpan]
    defender: list[SideMessage]
    reviewer: list[SideMessage]
    arbitrator: list[SideMessage]
    cursors: dict[str, int]
    continuation_side: Side

    @classmethod
    def from_ingested(cls, queues: IngestedQueues) -> MutableQueues:
        return cls(
            main_context=list(queues.main_context),
            defender=list(queues.defender),
            reviewer=list(queues.reviewer),
            arbitrator=list(queues.arbitrator),
            cursors=dict(queues.cursors),
            continuation_side=queues.continuation_side,
        )

    def current_actor(self) -> Actor:
        if self.continuation_side == Side.REVIEWER:
            return Actor.REVIEWER
        if self.continuation_side == Side.ARBITRATOR:
            return Actor.ARBITRATOR
        if self.continuation_side == Side.DEFENDER:
            return Actor.DEFENDER
        return Actor.MAIN

    def append_main(self, message: StateMessage, *, content_hash: str = "") -> ChatMessageSpan:
        return append_main_context_row(self.main_context, self.cursors, message, content_hash=content_hash)

    def append_side(self, side: Side, message: StateMessage, *, content_hash: str = "") -> SideMessage:
        side = Side(side)
        if side == Side.MAIN:
            raise ValueError("append_side does not accept main; use append_main")
        row = SideMessage(message=message, content_hash=content_hash)
        if side == Side.DEFENDER:
            self.defender.append(row)
            return row
        if side == Side.REVIEWER:
            self.reviewer.append(row)
        else:
            self.arbitrator.append(row)
        return row

    def defender_parts(self) -> DefenderParts:
        if not self.defender:
            return DefenderParts(held_candidate=None, held_hidden_tool_rows=(), remaining_temp_rows=())
        held_candidate = self.defender[0]
        if not held_candidate.message.is_assistant():
            return DefenderParts(
                held_candidate=None,
                held_hidden_tool_rows=(),
                remaining_temp_rows=tuple(self.defender),
            )
        hidden_end = 1
        while hidden_end < len(self.defender) and self.defender[hidden_end].message.is_tool():
            hidden_end += 1
        return DefenderParts(
            held_candidate=held_candidate,
            held_hidden_tool_rows=tuple(self.defender[1:hidden_end]),
            remaining_temp_rows=tuple(self.defender[hidden_end:]),
        )

    def set_continuation(self, side: Side) -> None:
        self.continuation_side = side

    def clear_debate(self) -> None:
        self.defender.clear()
        self.reviewer.clear()
        self.arbitrator.clear()
        self.continuation_side = Side.MAIN


def build_response_usage(
    *,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    reasoning_tokens: int,
    reasoning_to_output: float,
) -> ResponseUsage:
    scaled_reasoning_tokens = ceil(reasoning_tokens * reasoning_to_output)
    scaled_output_tokens = max(0, output_tokens + scaled_reasoning_tokens - reasoning_tokens)
    return ResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details=ResponseUsageInputTokensDetails(cached_tokens=cached_tokens),
        output_tokens=scaled_output_tokens,
        output_tokens_details=ResponseUsageOutputTokensDetails(reasoning_tokens=scaled_reasoning_tokens),
        total_tokens=input_tokens + scaled_output_tokens,
    )


@dataclass(slots=True)
class UsageLedger:
    budget: int | None
    reasoning_to_output: float = 1.0
    hidden: list[tuple[PublicUsageConfig, ChatUsage]] = field(default_factory=list)
    anchor: ChatUsage | None = None

    def remaining(self) -> int | None:
        return self.budget

    def cap_for(self, config: PublicUsageConfig) -> int | None:
        return config.cap_from_budget(self.budget)

    def record_hidden(self, config: PublicUsageConfig, usage: ChatUsage | None) -> int | None:
        if usage is None:
            return None
        if self.budget is not None:
            self.budget -= config.hidden_debit(usage)
        self.hidden.append((config, usage))
        return len(self.hidden) - 1

    def use_hidden_as_anchor(self, index: int) -> None:
        if self.anchor is not None:
            raise ValueError("usage anchor is already set")
        _, usage = self.hidden.pop(index)
        self.anchor = usage

    def set_anchor(self, usage: ChatUsage | None) -> None:
        if usage is None:
            return
        if self.anchor is not None:
            raise ValueError("usage anchor is already set")
        self.anchor = usage

    def to_response_usage(self) -> ResponseUsage | None:
        if self.anchor is None:
            return None

        hidden_equivalent_output = sum(config.hidden_debit(usage) for config, usage in self.hidden)
        cached_tokens = self.anchor.cached_tokens or 0
        reasoning_tokens = (self.anchor.reasoning_tokens or 0) + hidden_equivalent_output
        output_tokens = self.anchor.output_tokens + hidden_equivalent_output
        return build_response_usage(
            input_tokens=self.anchor.input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            reasoning_to_output=self.reasoning_to_output,
        )


def _required_int(value: Mapping[str, object], key: str) -> int:
    raw = value.get(key)
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise TypeError(f"{key} must be an integer")
    return raw


def _required_positive_int(value: Mapping[str, object], key: str) -> int:
    raw = _required_int(value, key)
    if raw <= 0:
        raise ValueError(f"{key} must be positive")
    return raw


def _optional_non_negative_int(value: Mapping[str, object], key: str) -> int:
    raw = value.get(key)
    if raw is None:
        return 0
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return raw
