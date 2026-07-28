from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from plap.llms.completions.chat import ChatMessage as Message
from plap.llms.completions.chat import ChatToolCall as ToolCall
from plap.responses.ingest.patch import JSONPatch, JSONValue


def _required_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _required_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is required")
    return value


def _optional_non_empty_string(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, label=label)


def _required_patch(value: object, *, label: str) -> JSONPatch:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    patch: JSONPatch = []
    for index, operation in enumerate(value):
        if not isinstance(operation, Mapping):
            raise TypeError(f"{label}[{index}] must be an object")
        patch.append(dict(operation))
    return patch


type Side = str

MAIN_SIDE: Side = "main"


def _validate_main_source(source: Message) -> None:
    if not source.is_assistant():
        raise ValueError("main tail source must be an assistant message")


@dataclass(frozen=True, slots=True)
class HiddenMainTail:
    source: Message

    def __post_init__(self) -> None:
        _validate_main_source(self.source)


@dataclass(frozen=True, slots=True)
class PublicMainTail:
    source: Message | None

    def __post_init__(self) -> None:
        if self.source is not None:
            _validate_main_source(self.source)


@dataclass(frozen=True, slots=True)
class CompactedMainTail:
    source: Message

    def __post_init__(self) -> None:
        _validate_main_source(self.source)


type MainTail = HiddenMainTail | PublicMainTail | CompactedMainTail


def _required_side(value: object, *, label: str) -> Side:
    return _required_string(value, label=label)


def _required_side_set(value: object, *, label: str) -> set[Side]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    sides = {_required_side(side, label=f"{label} item") for side in value}
    if len(sides) != len(value):
        raise ValueError(f"{label} must not contain duplicates")
    return sides


@dataclass(frozen=True, slots=True)
class MessagePatch:
    message: Message

    def __post_init__(self) -> None:
        if not self.message.is_assistant():
            raise ValueError("message patch must wrap an assistant message")

    def to_primitive(self) -> dict[str, object]:
        return {"message": self.message.to_primitive()}

    @classmethod
    def from_primitive(cls, value: object) -> MessagePatch:
        item = _required_mapping(value, label="message patch")
        allowed = {"message"}
        unknown = set(item) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"message patch contains unknown keys: {names}")
        if "message" not in item:
            raise ValueError("message patch is missing key: message")
        return cls(message=Message.from_primitive(item["message"]))


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
        if isinstance(item, Mapping) and "message" in item and "role" not in item:
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


def _anchor_open_call_ids(anchor: Message) -> set[str]:
    return set(_tool_call_ids(anchor.tool_calls, label="anchor tool_calls"))


def split_tail(items: list[Message]) -> tuple[list[Message], Message | None, list[Message], list[Message]]:
    anchor_index: int | None = None
    for index in range(len(items) - 1, -1, -1):
        candidate = items[index]
        if candidate.is_assistant():
            anchor_index = index
            break
    if anchor_index is None:
        return [], None, [], list(items)

    prefix = list(items[:anchor_index])
    anchor = items[anchor_index]
    suffix: list[Message] = []
    after: list[Message] = []
    after_started = False
    for item in items[anchor_index + 1 :]:
        if not after_started and item.is_tool():
            suffix.append(item)
            continue
        after_started = True
        after.append(item)
    return prefix, anchor, suffix, after


@dataclass(slots=True)
class Sides:
    active: set[Side] = field(default_factory=lambda: {MAIN_SIDE})
    messages: dict[Side, list[Message]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.active = {_required_side(side, label="active side") for side in self.active}
        normalized: dict[Side, list[Message]] = {}
        for raw_side, messages in self.messages.items():
            side = _required_side(raw_side, label="sides key")
            normalized[side] = list(messages)
        self.messages = normalized

    def get(self, side: Side, default: list[Message] | None = None) -> list[Message] | None:
        return self.messages.get(_required_side(side, label="side"), default)

    def __getitem__(self, side: Side) -> list[Message]:
        return self.messages[_required_side(side, label="side")]

    def __setitem__(self, side: Side, messages: list[Message]) -> None:
        key = _required_side(side, label="side")
        self.messages[key] = list(messages)

    def setdefault(self, side: Side, default: list[Message] | None = None) -> list[Message]:
        key = _required_side(side, label="side")
        if key not in self.messages:
            self.messages[key] = [] if default is None else list(default)
        return self.messages[key]

    def items(self):
        return self.messages.items()

    def to_primitive(self) -> dict[str, object]:
        return {
            "active": sorted(self.active),
            "messages": {side: [message.to_primitive() for message in self.messages[side]] for side in sorted(self.messages)},
        }

    @classmethod
    def from_primitive(cls, value: object) -> Sides:
        item = _required_mapping(value, label="sides")
        allowed = {"active", "messages"}
        unknown = set(item) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"sides contains unknown keys: {names}")
        missing = allowed - set(item)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"sides is missing keys: {names}")
        messages_value = _required_mapping(item["messages"], label="sides messages")
        return cls(
            active=_required_side_set(item["active"], label="sides active"),
            messages={
                _required_side(side, label="sides key"): _required_message_list(messages, label=f"sides.{side}")
                for side, messages in messages_value.items()
            },
        )


def split_main_updates(
    updates: list[MainUpdate],
) -> tuple[list[Message], list[Message], Message | None, list[Message], list[Message], MessagePatch | None]:
    patch_indices = [index for index, update in enumerate(updates) if isinstance(update, MessagePatch)]
    if len(patch_indices) > 1:
        raise ValueError("reasoning main may contain at most one message patch")
    patch: MessagePatch | None = None
    if patch_indices:
        candidate = updates[patch_indices[0]]
        if not isinstance(candidate, MessagePatch):  # pragma: no cover - narrowed by patch_indices
            raise TypeError("expected message patch")
        patch = candidate
    if patch_indices and patch_indices[0] != len(updates) - 1:
        raise ValueError("message patch must be the final main update")

    messages = [update for update in updates if isinstance(update, Message)]
    leading_end = 0
    while leading_end < len(messages) and messages[leading_end].is_tool():
        leading_end += 1
    leading_outputs = messages[:leading_end]
    local = messages[leading_end:]
    prefix, anchor, suffix, after = split_tail(local)
    if anchor is None:
        _validate_closed_prefix_messages(after, label="reasoning main")
        if patch is not None and after:
            raise ValueError("message patch target may not have a trailing non-assistant tail")
        return leading_outputs, [], None, [], after, patch

    anchor_index = next(index for index, update in enumerate(local) if update is anchor)
    _validate_closed_prefix_messages(prefix, label="reasoning main prefix")
    pending_anchor_call_ids = _anchor_open_call_ids(anchor)
    for offset, update in enumerate(suffix, start=leading_end + anchor_index + 1):
        if update.tool_call_id is None:
            raise ValueError(f"reasoning main[{offset}] must be a tool message with tool_call_id after the anchor")
        if update.tool_call_id not in pending_anchor_call_ids:
            raise ValueError(f"reasoning main[{offset}] does not match an unresolved anchor tool call")
        pending_anchor_call_ids.remove(update.tool_call_id)
    if pending_anchor_call_ids and after:
        raise ValueError("reasoning main with unresolved anchor tool calls may not have trailing non-assistant tail")
    for offset, update in enumerate(after, start=leading_end + anchor_index + 1 + len(suffix)):
        if update.is_assistant() or update.is_tool():
            raise ValueError(f"reasoning main[{offset}] must be a closed non-assistant tail message")
    if patch is not None and after:
        raise ValueError("message patch target may not have a trailing non-assistant tail")
    return leading_outputs, prefix, anchor, suffix, after, patch


@dataclass(frozen=True, slots=True)
class ReasoningCheckpoint:
    durable: dict[str, JSONValue]
    active: set[Side]
    sides: dict[Side, list[Message]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "durable", dict(self.durable))
        object.__setattr__(self, "active", {_required_side(side, label="active side") for side in self.active})
        normalized: dict[Side, list[Message]] = {}
        for raw_side, messages in self.sides.items():
            side = _required_side(raw_side, label="reasoning checkpoint sides key")
            if side == MAIN_SIDE:
                raise ValueError("reasoning checkpoint sides may not contain main")
            normalized[side] = list(messages)
        object.__setattr__(self, "sides", normalized)

    def to_primitive(self) -> dict[str, object]:
        return {
            "type": "checkpoint",
            "durable": dict(self.durable),
            "active": sorted(self.active),
            "sides": {side: [message.to_primitive() for message in self.sides[side]] for side in sorted(self.sides)},
        }

    @classmethod
    def from_primitive(cls, value: object) -> ReasoningCheckpoint:
        item = _required_mapping(value, label="reasoning checkpoint")
        allowed = {"type", "durable", "active", "sides"}
        unknown = set(item) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"reasoning checkpoint contains unknown keys: {names}")
        if item.get("type") != "checkpoint":
            raise ValueError("reasoning checkpoint type must be 'checkpoint'")
        durable = _required_mapping(item.get("durable"), label="reasoning checkpoint durable state")
        sides = _required_mapping(item.get("sides"), label="reasoning checkpoint sides")
        return cls(
            durable=dict(durable),
            active=_required_side_set(item.get("active"), label="reasoning checkpoint active"),
            sides={
                _required_side(side, label="reasoning checkpoint sides key"): _required_message_list(
                    messages,
                    label=f"reasoning checkpoint sides.{side}",
                )
                for side, messages in sides.items()
            },
        )


@dataclass(frozen=True, slots=True)
class ReasoningPatch:
    durable: JSONPatch
    active: set[Side] | None = None
    sides: dict[Side, JSONPatch] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "durable", _required_patch(self.durable, label="reasoning durable patch"))
        if self.active is not None:
            object.__setattr__(self, "active", {_required_side(side, label="active side") for side in self.active})
        normalized: dict[Side, JSONPatch] = {}
        for raw_side, patch in self.sides.items():
            side = _required_side(raw_side, label="reasoning patch sides key")
            if side == MAIN_SIDE:
                raise ValueError("reasoning patch sides may not target main")
            normalized[side] = _required_patch(patch, label=f"reasoning patch sides.{side}")
        object.__setattr__(self, "sides", normalized)

    def to_primitive(self) -> dict[str, object]:
        return {
            "type": "patch",
            "durable": list(self.durable),
            "active": None if self.active is None else sorted(self.active),
            "sides": {side: list(patch) for side, patch in sorted(self.sides.items())},
        }

    @classmethod
    def from_primitive(cls, value: object) -> ReasoningPatch:
        item = _required_mapping(value, label="reasoning patch")
        allowed = {"type", "durable", "active", "sides"}
        unknown = set(item) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"reasoning patch contains unknown keys: {names}")
        if item.get("type") != "patch":
            raise ValueError("reasoning patch type must be 'patch'")
        sides = _required_mapping(item.get("sides"), label="reasoning patch sides")
        active = item.get("active")
        return cls(
            durable=_required_patch(item.get("durable"), label="reasoning durable patch"),
            active=None if active is None else _required_side_set(active, label="reasoning patch active"),
            sides={
                _required_side(side, label="reasoning patch sides key"): _required_patch(
                    patch,
                    label=f"reasoning patch sides.{side}",
                )
                for side, patch in sides.items()
            },
        )


type ReasoningState = ReasoningCheckpoint | ReasoningPatch


def _reasoning_state_from_primitive(value: object) -> ReasoningState:
    item = _required_mapping(value, label="reasoning state")
    state_type = item.get("type")
    if state_type == "checkpoint":
        return ReasoningCheckpoint.from_primitive(item)
    if state_type == "patch":
        return ReasoningPatch.from_primitive(item)
    raise ValueError("reasoning state type must be 'checkpoint' or 'patch'")


@dataclass(frozen=True, slots=True)
class ReasoningPayload:
    id: str
    previous_reasoning_id: str | None
    previous_compaction_id: str | None
    state: ReasoningState
    main: list[MainUpdate] = field(default_factory=list)

    def __post_init__(self) -> None:
        _required_string(self.id, label="reasoning payload id")
        _optional_non_empty_string(self.previous_reasoning_id, label="reasoning payload previous_reasoning_id")
        _optional_non_empty_string(self.previous_compaction_id, label="reasoning payload previous_compaction_id")
        normalized = list(self.main)
        split_main_updates(normalized)
        object.__setattr__(self, "main", normalized)

    def to_primitive(self) -> dict[str, object]:
        return {
            "id": self.id,
            "previous_reasoning_id": self.previous_reasoning_id,
            "previous_compaction_id": self.previous_compaction_id,
            "state": self.state.to_primitive(),
            "main": [update.to_primitive() for update in self.main],
        }

    @classmethod
    def from_primitive(cls, value: object) -> ReasoningPayload:
        item = _required_mapping(value, label="reasoning payload")
        allowed = {"id", "previous_reasoning_id", "previous_compaction_id", "state", "main"}
        unknown = set(item) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"reasoning payload contains unknown keys: {names}")
        return cls(
            id=_required_string(item.get("id"), label="reasoning payload id"),
            previous_reasoning_id=_optional_non_empty_string(
                item.get("previous_reasoning_id"), label="reasoning payload previous_reasoning_id"
            ),
            previous_compaction_id=_optional_non_empty_string(
                item.get("previous_compaction_id"), label="reasoning payload previous_compaction_id"
            ),
            state=_reasoning_state_from_primitive(item.get("state")),
            main=_required_main_update_list(item.get("main", []), label="reasoning main"),
        )


@dataclass(frozen=True, slots=True)
class CompactionPayload:
    id: str
    durable: dict[str, JSONValue]
    sides: Sides

    def __post_init__(self) -> None:
        _required_string(self.id, label="compaction payload id")

    def to_primitive(self) -> dict[str, object]:
        return {
            "id": self.id,
            "durable": dict(self.durable),
            "sides": self.sides.to_primitive(),
        }

    @classmethod
    def from_primitive(cls, value: object) -> CompactionPayload:
        item = _required_mapping(value, label="compaction payload")
        durable_value = item.get("durable")
        if not isinstance(durable_value, Mapping):
            raise TypeError("compaction payload durable state must be an object")
        return cls(
            id=_required_string(item.get("id"), label="compaction payload id"),
            durable=dict(durable_value),
            sides=Sides.from_primitive(item.get("sides")),
        )


@dataclass(frozen=True, slots=True)
class CallID:
    side: Side
    upstream_tool_call_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", _required_side(self.side, label="function call side"))


@dataclass(slots=True)
class Ingested:
    durable: dict[str, JSONValue]
    sides: Sides
    main_tail: MainTail | None
    last_reasoning_id: str | None
    last_compaction_id: str | None = None
    checkpoint_required: bool = False
