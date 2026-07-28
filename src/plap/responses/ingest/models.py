from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from plap.llms.completions.chat import ChatMessage as Message
from plap.llms.completions.chat import ChatToolCall as ToolCall
from plap.responses.ingest.patch import JSONPatch, JSONValue
from plap.responses.ingest.shape import shape


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


def _required_shape(value: object, *, label: str) -> JSONValue:
    if value is None:
        return None
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    return value


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

    def shape(self, side: Side) -> JSONValue:
        messages = self.get(side)
        if messages is None:
            return None
        return shape(messages)

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


@dataclass(frozen=True, slots=True)
class GuardedPatch:
    shape: JSONValue
    patch: JSONPatch | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", _required_shape(self.shape, label="guarded patch shape"))
        if self.patch is not None:
            object.__setattr__(self, "patch", _required_patch(self.patch, label="guarded patch patch"))

    def to_primitive(self) -> dict[str, object]:
        return {
            "shape": self.shape,
            "patch": None if self.patch is None else list(self.patch),
        }

    @classmethod
    def from_primitive(cls, value: object) -> GuardedPatch:
        item = _required_mapping(value, label="guarded patch")
        allowed = {"shape", "patch"}
        unknown = set(item) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"guarded patch contains unknown keys: {names}")
        missing = allowed - set(item)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"guarded patch is missing keys: {names}")
        return cls(
            shape=_required_shape(item["shape"], label="guarded patch shape"),
            patch=None if item["patch"] is None else _required_patch(item["patch"], label="guarded patch patch"),
        )


@dataclass(frozen=True, slots=True)
class SidesUpdate:
    active: set[Side] | None = None
    main: list[MainUpdate] = field(default_factory=list)
    patches: dict[Side, GuardedPatch] = field(default_factory=dict)

    def split_main(
        self,
    ) -> tuple[list[Message], list[Message], Message | None, list[Message], list[Message], MessagePatch | None]:
        patch_indices = [index for index, update in enumerate(self.main) if isinstance(update, MessagePatch)]
        if len(patch_indices) > 1:
            raise ValueError("sides update main may contain at most one message patch")
        patch: MessagePatch | None = None
        if patch_indices:
            candidate = self.main[patch_indices[0]]
            if not isinstance(candidate, MessagePatch):  # pragma: no cover - narrowed by patch_indices
                raise TypeError("expected message patch")
            patch = candidate
        if patch_indices and patch_indices[0] != len(self.main) - 1:
            raise ValueError("message patch must be the final main update")

        messages = [update for update in self.main if isinstance(update, Message)]
        leading_end = 0
        while leading_end < len(messages) and messages[leading_end].is_tool():
            leading_end += 1
        leading_outputs = messages[:leading_end]
        local = messages[leading_end:]
        prefix, anchor, suffix, after = split_tail(local)
        if anchor is None:
            _validate_closed_prefix_messages(after, label="sides update main")
            if patch is not None and after:
                raise ValueError("message patch target may not have a trailing non-assistant tail")
            return leading_outputs, [], None, [], after, patch

        anchor_index = next(index for index, update in enumerate(local) if update is anchor)
        _validate_closed_prefix_messages(prefix, label="sides update main prefix")
        pending_anchor_call_ids = _anchor_open_call_ids(anchor)
        for offset, update in enumerate(suffix, start=leading_end + anchor_index + 1):
            if update.tool_call_id is None:
                raise ValueError(f"sides update main[{offset}] must be a tool message with tool_call_id after the anchor")
            if update.tool_call_id not in pending_anchor_call_ids:
                raise ValueError(f"sides update main[{offset}] does not match an unresolved anchor tool call")
            pending_anchor_call_ids.remove(update.tool_call_id)
        if pending_anchor_call_ids and after:
            raise ValueError("sides update main with unresolved anchor tool calls may not have trailing non-assistant tail")
        for offset, update in enumerate(after, start=leading_end + anchor_index + 1 + len(suffix)):
            if update.is_assistant() or update.is_tool():
                raise ValueError(f"sides update main[{offset}] must be a closed non-assistant tail message")
        if patch is not None and after:
            raise ValueError("message patch target may not have a trailing non-assistant tail")
        return leading_outputs, prefix, anchor, suffix, after, patch

    def __post_init__(self) -> None:
        self.split_main()
        if self.active is not None:
            object.__setattr__(self, "active", {_required_side(side, label="active side") for side in self.active})
        normalized: dict[Side, GuardedPatch] = {}
        for raw_side, guarded in self.patches.items():
            side = _required_side(raw_side, label="sides update patches key")
            normalized[side] = guarded
        object.__setattr__(self, "patches", normalized)

    def to_primitive(self) -> dict[str, object]:
        value: dict[str, object] = {
            "active": None if self.active is None else sorted(self.active),
            "main": [message.to_primitive() for message in self.main],
            "patches": {side: guarded.to_primitive() for side, guarded in sorted(self.patches.items())},
        }
        return value

    @classmethod
    def from_primitive(cls, value: object) -> SidesUpdate:
        item = _required_mapping(value, label="sides update")
        allowed = {"active", "main", "patches"}
        unknown = set(item) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"sides update contains unknown keys: {names}")
        patches_value = _required_mapping(item.get("patches", {}), label="sides update patches")
        active_value = item.get("active")
        return cls(
            active=None if active_value is None else _required_side_set(active_value, label="sides update active"),
            main=_required_main_update_list(item.get("main", []), label="sides update main"),
            patches={
                _required_side(side, label="sides update patches key"): GuardedPatch.from_primitive(guarded)
                for side, guarded in patches_value.items()
            },
        )


@dataclass(frozen=True, slots=True)
class ReasoningPayload:
    id: str
    previous_reasoning_id: str | None
    previous_compaction_id: str | None
    machine: JSONPatch
    sides: SidesUpdate

    def __post_init__(self) -> None:
        _required_string(self.id, label="reasoning payload id")
        _optional_non_empty_string(self.previous_reasoning_id, label="reasoning payload previous_reasoning_id")
        previous_compaction_id = _optional_non_empty_string(
            self.previous_compaction_id,
            label="reasoning payload previous_compaction_id",
        )
        if previous_compaction_id is not None:
            raise ValueError("reasoning payload previous_compaction_id is no longer supported")

    def to_primitive(self) -> dict[str, object]:
        return {
            "id": self.id,
            "previous_reasoning_id": self.previous_reasoning_id,
            "previous_compaction_id": self.previous_compaction_id,
            "machine": list(self.machine),
            "sides": self.sides.to_primitive(),
        }

    @classmethod
    def from_primitive(cls, value: object) -> ReasoningPayload:
        item = _required_mapping(value, label="reasoning payload")
        return cls(
            id=_required_string(item.get("id"), label="reasoning payload id"),
            previous_reasoning_id=_optional_non_empty_string(
                item.get("previous_reasoning_id"), label="reasoning payload previous_reasoning_id"
            ),
            previous_compaction_id=_optional_non_empty_string(
                item.get("previous_compaction_id"), label="reasoning payload previous_compaction_id"
            ),
            machine=_required_patch(item.get("machine"), label="reasoning payload machine"),
            sides=SidesUpdate.from_primitive(item.get("sides")),
        )


@dataclass(frozen=True, slots=True)
class CompactionPayload:
    id: str
    machine: dict[str, JSONValue]
    sides: Sides

    def __post_init__(self) -> None:
        _required_string(self.id, label="compaction payload id")

    def to_primitive(self) -> dict[str, object]:
        return {
            "id": self.id,
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
            id=_required_string(item.get("id"), label="compaction payload id"),
            machine=dict(machine_value),
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
    machine: dict[str, JSONValue]
    sides: Sides
    last_reasoning_id: str | None
