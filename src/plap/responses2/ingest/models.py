from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from plap.llms.completions.chat import ChatRole

type JSONValue = object
type JSONPatch = list[dict[str, JSONValue]]


def _required_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _required_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is required")
    return value


def _optional_string(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def _required_patch(value: object, *, label: str) -> JSONPatch:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    patch: JSONPatch = []
    for index, operation in enumerate(value):
        if not isinstance(operation, Mapping):
            raise TypeError(f"{label}[{index}] must be an object")
        patch.append(dict(operation))
    return patch


class Side(StrEnum):
    MAIN = "main"
    DEFENDER = "defender"
    REVIEWER = "reviewer"
    ARBITRATOR = "arbitrator"


NON_MAIN_SIDES: tuple[Side, ...] = tuple(side for side in Side if side != Side.MAIN)


def _validate_known_side_keys(item: Mapping[str, object], *, label: str) -> None:
    allowed = {Side.MAIN.value, *(side.value for side in NON_MAIN_SIDES)}
    unknown = set(item) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"{label} contains unknown side keys: {names}")


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def to_primitive(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }

    @classmethod
    def from_primitive(cls, value: object) -> ToolCall:
        item = _required_mapping(value, label="tool call")
        return cls(
            id=_required_string(item.get("id"), label="tool call id"),
            name=_required_string(item.get("name"), label="tool call name"),
            arguments=_required_string(item.get("arguments"), label="tool call arguments"),
        )


@dataclass(frozen=True, slots=True)
class MessagePatch:
    content_hash: str
    tool_calls: list[ToolCall] | None = None
    reasoning_content: str | None = None
    reasoning_details: list[object] | None = None

    def __post_init__(self) -> None:
        if self.tool_calls is None and self.reasoning_content is None and self.reasoning_details is None:
            raise ValueError("message patch must change hidden assistant fields")

    def to_primitive(self) -> dict[str, object]:
        value: dict[str, object] = {"content_hash": self.content_hash}
        if self.tool_calls is not None:
            value["tool_calls"] = [call.to_primitive() for call in self.tool_calls]
        if self.reasoning_content is not None:
            value["reasoning_content"] = self.reasoning_content
        if self.reasoning_details is not None:
            value["reasoning_details"] = list(self.reasoning_details)
        return value

    @classmethod
    def from_primitive(cls, value: object) -> MessagePatch:
        item = _required_mapping(value, label="message patch")
        tool_calls_value = item.get("tool_calls")
        tool_calls: list[ToolCall] | None = None
        if tool_calls_value is not None:
            if not isinstance(tool_calls_value, list):
                raise TypeError("message patch tool_calls must be an array")
            tool_calls = [ToolCall.from_primitive(call) for call in tool_calls_value]
        reasoning_details_value = item.get("reasoning_details")
        reasoning_details: list[object] | None = None
        if reasoning_details_value is not None:
            if not isinstance(reasoning_details_value, list):
                raise TypeError("message patch reasoning_details must be an array")
            reasoning_details = list(reasoning_details_value)
        return cls(
            content_hash=_required_string(item.get("content_hash"), label="message patch content_hash"),
            tool_calls=tool_calls,
            reasoning_content=_optional_string(item.get("reasoning_content"), label="message patch reasoning_content"),
            reasoning_details=reasoning_details,
        )


@dataclass(slots=True)
class Message:
    role: ChatRole
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_content: str | None = None
    reasoning_details: list[object] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.role = ChatRole(self.role)

    def is_assistant(self) -> bool:
        return self.role == ChatRole.ASSISTANT

    def is_tool(self) -> bool:
        return self.role == ChatRole.TOOL

    def append_tool_call(self, tool_call: ToolCall) -> None:
        self.tool_calls.append(tool_call)

    def tool_call_at(self, index: int) -> ToolCall:
        return self.tool_calls[index]

    def to_primitive(self) -> dict[str, object]:
        value: dict[str, object] = {"role": self.role}
        if self.content is not None:
            value["content"] = self.content
        if self.name is not None:
            value["name"] = self.name
        if self.tool_call_id is not None:
            value["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            value["tool_calls"] = [call.to_primitive() for call in self.tool_calls]
        if self.reasoning_content is not None:
            value["reasoning_content"] = self.reasoning_content
        if self.reasoning_details:
            value["reasoning_details"] = list(self.reasoning_details)
        return value

    @classmethod
    def from_primitive(cls, value: object) -> Message:
        item = _required_mapping(value, label="message")
        tool_calls_value = item.get("tool_calls")
        tool_calls: list[ToolCall] = []
        if tool_calls_value is not None:
            if not isinstance(tool_calls_value, list):
                raise TypeError("message tool_calls must be an array")
            tool_calls = [ToolCall.from_primitive(call) for call in tool_calls_value]
        reasoning_details_value = item.get("reasoning_details")
        reasoning_details: list[object] = []
        if reasoning_details_value is not None:
            if not isinstance(reasoning_details_value, list):
                raise TypeError("message reasoning_details must be an array")
            reasoning_details = list(reasoning_details_value)
        return cls(
            role=ChatRole(_required_string(item.get("role"), label="message role")),
            content=_optional_string(item.get("content"), label="message content"),
            name=_optional_string(item.get("name"), label="message name"),
            tool_call_id=_optional_string(item.get("tool_call_id"), label="message tool_call_id"),
            tool_calls=tool_calls,
            reasoning_content=_optional_string(item.get("reasoning_content"), label="message reasoning_content"),
            reasoning_details=reasoning_details,
        )


def _required_message_list(value: object, *, label: str) -> list[Message]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    return [Message.from_primitive(item) for item in value]


type MainUpdate = Message | MessagePatch


def _required_main_update_list(value: object, *, label: str) -> list[MainUpdate]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    updates: list[MainUpdate] = []
    for index, item in enumerate(value):
        if isinstance(item, Mapping) and isinstance(item.get("content_hash"), str):
            updates.append(MessagePatch.from_primitive(item))
            continue
        try:
            updates.append(Message.from_primitive(item))
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"{label}[{index}]: {exc}") from exc
    return updates


def _tool_call_ids(tool_calls: list[ToolCall], *, label: str) -> list[str]:
    call_ids = [tool_call.id for tool_call in tool_calls]
    if len(call_ids) != len(set(call_ids)):
        raise ValueError(f"{label} must not contain duplicate tool call ids")
    return call_ids


def _validate_closed_prefix_messages(messages: list[Message], *, label: str) -> None:
    pending_call_ids: set[str] = set()
    for index, message in enumerate(messages):
        if message.is_tool():
            if message.tool_call_id is None or message.tool_call_id not in pending_call_ids:
                raise ValueError(f"{label}[{index}] has no pending tool call to satisfy")
            pending_call_ids.remove(message.tool_call_id)
            continue
        if pending_call_ids:
            raise ValueError(f"{label}[{index}] cannot appear before earlier tool calls are satisfied")
        if not message.is_assistant() or not message.tool_calls:
            continue
        pending_call_ids.update(_tool_call_ids(message.tool_calls, label=f"{label}[{index}] tool_calls"))
    if pending_call_ids:
        raise ValueError(f"{label} must satisfy all prefix tool calls before the anchor")


def _anchor_open_call_ids(anchor: MainUpdate) -> set[str]:
    if isinstance(anchor, MessagePatch):
        if anchor.tool_calls is None:
            return set()
        return set(_tool_call_ids(anchor.tool_calls, label="message patch tool_calls"))
    return set(_tool_call_ids(anchor.tool_calls, label="anchor tool_calls"))


def _validate_main_updates(main: list[MainUpdate]) -> None:
    if not main:
        return
    patch_indices = [index for index, update in enumerate(main) if isinstance(update, MessagePatch)]
    if len(patch_indices) > 1:
        raise ValueError("sides update main may contain at most one message patch")
    anchor_index = len(main) - 1
    while anchor_index >= 0 and isinstance(main[anchor_index], Message) and main[anchor_index].is_tool():
        anchor_index -= 1
    if anchor_index < 0:
        raise ValueError("sides update main must contain an assistant anchor or message patch")
    anchor = main[anchor_index]
    if isinstance(anchor, MessagePatch):
        if patch_indices != [anchor_index]:
            raise ValueError("message patch must be the last non-tool main update")
    elif not anchor.is_assistant():
        raise ValueError("sides update main anchor must be an assistant message or message patch")
    prefix = main[:anchor_index]
    if any(isinstance(update, MessagePatch) for update in prefix):
        raise ValueError("message patch must be the last non-tool main update")
    _validate_closed_prefix_messages([update for update in prefix if isinstance(update, Message)], label="sides update main prefix")
    pending_anchor_call_ids = _anchor_open_call_ids(anchor)
    for index, update in enumerate(main[anchor_index + 1 :], start=anchor_index + 1):
        if not isinstance(update, Message) or not update.is_tool() or update.tool_call_id is None:
            raise ValueError(f"sides update main[{index}] must be a tool message with tool_call_id after the anchor")
        if update.tool_call_id not in pending_anchor_call_ids:
            raise ValueError(f"sides update main[{index}] does not match an unresolved anchor tool call")
        pending_anchor_call_ids.remove(update.tool_call_id)


@dataclass(slots=True)
class Sides:
    main: list[Message] = field(default_factory=list)
    others: dict[Side, list[Message]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[Side, list[Message]] = {}
        for raw_side, messages in self.others.items():
            side = Side(raw_side)
            if side == Side.MAIN:
                raise ValueError("sides others must not include main")
            normalized[side] = list(messages)
        for side in NON_MAIN_SIDES:
            normalized.setdefault(side, [])
        self.others = normalized

    def messages(self, side: Side) -> list[Message]:
        if side == Side.MAIN:
            return self.main
        return self.others[Side(side)]

    def to_primitive(self) -> dict[str, object]:
        value: dict[str, object] = {
            "main": [message.to_primitive() for message in self.main],
        }
        for side in NON_MAIN_SIDES:
            value[side.value] = [message.to_primitive() for message in self.others[side]]
        return value

    @classmethod
    def from_primitive(cls, value: object) -> Sides:
        item = _required_mapping(value, label="sides")
        _validate_known_side_keys(item, label="sides")
        return cls(
            main=_required_message_list(item.get("main", []), label="sides.main"),
            others={
                side: _required_message_list(item.get(side.value, []), label=f"sides.{side.value}")
                for side in NON_MAIN_SIDES
            },
        )


@dataclass(frozen=True, slots=True)
class SidesUpdate:
    main: list[MainUpdate] = field(default_factory=list)
    others: dict[Side, JSONPatch] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_main_updates(self.main)
        normalized: dict[Side, JSONPatch] = {}
        for raw_side, patch in self.others.items():
            side = Side(raw_side)
            if side == Side.MAIN:
                raise ValueError("sides update others must not include main")
            normalized[side] = list(patch)
        for side in NON_MAIN_SIDES:
            normalized.setdefault(side, [])
        object.__setattr__(self, "others", normalized)

    def is_empty(self) -> bool:
        return not self.main and not any(self.others[side] for side in NON_MAIN_SIDES)

    def to_primitive(self) -> dict[str, object]:
        value: dict[str, object] = {
            "main": [message.to_primitive() for message in self.main],
        }
        for side in NON_MAIN_SIDES:
            value[side.value] = list(self.others[side])
        return value

    @classmethod
    def from_primitive(cls, value: object) -> SidesUpdate:
        item = _required_mapping(value, label="sides update")
        _validate_known_side_keys(item, label="sides update")
        return cls(
            main=_required_main_update_list(item.get("main"), label="sides update main"),
            others={
                side: _required_patch(item.get(side.value), label=f"sides update {side.value}")
                for side in NON_MAIN_SIDES
            },
        )


@dataclass(frozen=True, slots=True)
class ReasoningPayload:
    machine: JSONPatch
    sides: SidesUpdate

    def __post_init__(self) -> None:
        if not self.machine and self.sides.is_empty():
            raise ValueError("reasoning payload must change machine or sides")

    def to_primitive(self) -> dict[str, object]:
        return {
            "machine": list(self.machine),
            "sides": self.sides.to_primitive(),
        }

    @classmethod
    def from_primitive(cls, value: object) -> ReasoningPayload:
        item = _required_mapping(value, label="reasoning payload")
        return cls(
            machine=_required_patch(item.get("machine"), label="reasoning payload machine"),
            sides=SidesUpdate.from_primitive(item.get("sides")),
        )


@dataclass(frozen=True, slots=True)
class CompactionPayload:
    machine: dict[str, JSONValue]
    sides: Sides

    def to_primitive(self) -> dict[str, object]:
        return {
            "machine": dict(self.machine),
            "sides": self.sides.to_primitive(),
        }

    @classmethod
    def from_primitive(cls, value: object) -> CompactionPayload:
        item = _required_mapping(value, label="compaction payload")
        machine_value = item.get("machine")
        if not isinstance(machine_value, Mapping):
            raise TypeError("compaction payload machine must be an object")
        return cls(
            machine=dict(machine_value),
            sides=Sides.from_primitive(item.get("sides")),
        )


@dataclass(frozen=True, slots=True)
class CallID:
    side: Side
    content_hash_prefix: bytes
    tool_call_index: int
    upstream_tool_call_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", Side(self.side))


@dataclass(slots=True)
class Ingested:
    machine: dict[str, JSONValue]
    sides: Sides
    last_side: Side | None
