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


def _required_thread_name(value: object, *, label: str) -> str:
    return _required_string(value, label=label)


def _required_thread_name_set(value: object, *, label: str) -> set[str]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be an array")
    threads = {_required_thread_name(thread, label=f"{label} item") for thread in value}
    if len(threads) != len(value):
        raise ValueError(f"{label} must not contain duplicates")
    return threads


def _required_thread_name_mapping(value: object, *, label: str) -> dict[str, int]:
    item = _required_mapping(value, label=label)
    normalized: dict[str, int] = {}
    for raw_thread, raw_count in item.items():
        thread = _required_thread_name(raw_thread, label=f"{label} key")
        if not isinstance(raw_count, int) or raw_count < 0:
            raise ValueError(f"{label}.{thread} must be a non-negative integer")
        normalized[thread] = raw_count
    return normalized


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
class Threads:
    active: set[str] = field(default_factory=lambda: {"main"})
    blocking: set[str] = field(default_factory=set)
    messages: dict[str, list[Message]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.active = {_required_thread_name(thread, label="active thread") for thread in self.active}
        self.blocking = {_required_thread_name(thread, label="blocking thread") for thread in self.blocking}
        if self.blocking:
            self.active.discard("main")
        else:
            self.active.add("main")
        normalized: dict[str, list[Message]] = {}
        for raw_thread, messages in self.messages.items():
            thread = _required_thread_name(raw_thread, label="threads key")
            normalized[thread] = list(messages)
        self.messages = normalized

    def _thread_key(self, thread: str) -> str:
        return _required_thread_name(thread, label="thread")

    def _thread_active(self, thread: str) -> bool:
        if thread == "main":
            return not self.blocking
        return thread in self.active

    def _set_thread_active(self, thread: str, value: bool) -> None:
        if thread == "main":
            if value:
                self.blocking.discard("main")
                if not self.blocking:
                    self.active.add("main")
                return
            self.blocking.add("main")
            self.active.discard("main")
            return
        if value:
            self.active.add(thread)
            return
        self.active.discard(thread)

    def _thread_blocking(self, thread: str) -> bool:
        return thread in self.blocking

    def _set_thread_blocking(self, thread: str, value: bool) -> None:
        if value:
            self.blocking.add(thread)
            self.active.discard("main")
            return
        self.blocking.discard(thread)
        if not self.blocking:
            self.active.add("main")

    def _thread_messages(self, thread: str) -> list[Message]:
        return self.messages[thread]

    def _set_thread_messages(self, thread: str, messages: list[Message]) -> None:
        self.messages[thread] = list(messages)

    def get(self, thread: str, default: Thread | list[Message] | None = None) -> Thread | list[Message] | None:
        key = self._thread_key(thread)
        if key not in self.messages and key not in self.active and key not in self.blocking:
            return default
        return Thread(name=key, threads=self)

    def __getitem__(self, thread: str) -> Thread:
        key = self._thread_key(thread)
        return Thread(name=key, threads=self)

    def __setitem__(self, thread: str, value: Thread | list[Message]) -> None:
        key = self._thread_key(thread)
        if isinstance(value, Thread):
            if value.name != key:
                raise ValueError("thread wrapper name does not match assigned key")
            self.messages[key] = list(value.messages)
            if key == "main":
                self._set_thread_active(key, value.active)
                self._set_thread_blocking(key, value.blocking)
                return
            self._set_thread_active(key, value.active)
            self._set_thread_blocking(key, value.blocking)
            return
        self.messages[key] = list(value)

    def setdefault(self, thread: str, default: list[Message] | None = None) -> Thread:
        key = self._thread_key(thread)
        if key not in self.messages:
            self.messages[key] = [] if default is None else list(default)
        return Thread(name=key, threads=self)

    def items(self):
        return self.messages.items()

    def to_primitive(self) -> dict[str, object]:
        return {
            "active": sorted(self.active),
            "blocking": sorted(self.blocking),
            "messages": {thread: [message.to_primitive() for message in self.messages[thread]] for thread in sorted(self.messages)},
        }

    @classmethod
    def from_primitive(cls, value: object) -> Threads:
        item = _required_mapping(value, label="threads")
        allowed = {"active", "blocking", "messages"}
        unknown = set(item) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"threads contains unknown keys: {names}")
        missing = allowed - set(item)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"threads is missing keys: {names}")
        messages_value = _required_mapping(item["messages"], label="threads messages")
        return cls(
            active=_required_thread_name_set(item["active"], label="threads active"),
            blocking=_required_thread_name_set(item["blocking"], label="threads blocking"),
            messages={
                _required_thread_name(thread, label="threads key"): _required_message_list(messages, label=f"threads.{thread}")
                for thread, messages in messages_value.items()
            },
        )


@dataclass(slots=True, eq=False)
class Thread:
    name: str
    threads: Threads

    __hash__ = object.__hash__

    def _message_items(self) -> list[Message]:
        return self.messages

    def __bool__(self) -> bool:
        return bool(self._message_items())

    def __iter__(self):
        return iter(self._message_items())

    def __len__(self) -> int:
        return len(self._message_items())

    def __getitem__(self, index: int | slice) -> Message | list[Message]:
        return self._message_items()[index]

    def __setitem__(self, index: int | slice, value: Message | list[Message]) -> None:
        self._message_items()[index] = value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Thread):
            return self.name == other.name and self.messages == other.messages
        if isinstance(other, list):
            return self.messages == other
        return NotImplemented

    @property
    def active(self) -> bool:
        return self.threads._thread_active(self.name)

    @active.setter
    def active(self, value: bool) -> None:
        self.threads._set_thread_active(self.name, value)

    @property
    def blocking(self) -> bool:
        return self.threads._thread_blocking(self.name)

    @blocking.setter
    def blocking(self, value: bool) -> None:
        self.threads._set_thread_blocking(self.name, value)

    @property
    def messages(self) -> list[Message]:
        return self.threads._thread_messages(self.name)

    @messages.setter
    def messages(self, value: list[Message]) -> None:
        self.threads._set_thread_messages(self.name, value)

    def append(self, message: Message) -> None:
        self.messages.append(message)

    def extend(self, messages: list[Message]) -> None:
        self.messages.extend(messages)

    def block(self) -> None:
        self.blocking = True

    def unblock(self) -> None:
        self.blocking = False


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
    memory: dict[str, JSONValue]
    active: set[str]
    blocking: set[str]
    threads: dict[str, list[Message]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory", dict(self.memory))
        object.__setattr__(
            self,
            "active",
            {_required_thread_name(thread, label="active thread") for thread in self.active},
        )
        object.__setattr__(
            self,
            "blocking",
            {_required_thread_name(thread, label="blocking thread") for thread in self.blocking},
        )
        normalized: dict[str, list[Message]] = {}
        for raw_thread, messages in self.threads.items():
            thread = _required_thread_name(raw_thread, label="reasoning checkpoint threads key")
            if thread == "main":
                raise ValueError("reasoning checkpoint threads may not contain main")
            normalized[thread] = list(messages)
        object.__setattr__(self, "threads", normalized)

    def to_primitive(self) -> dict[str, object]:
        return {
            "type": "checkpoint",
            "memory": dict(self.memory),
            "active": sorted(self.active),
            "blocking": sorted(self.blocking),
            "threads": {thread: [message.to_primitive() for message in self.threads[thread]] for thread in sorted(self.threads)},
        }

    @classmethod
    def from_primitive(cls, value: object) -> ReasoningCheckpoint:
        item = _required_mapping(value, label="reasoning checkpoint")
        allowed = {"type", "memory", "active", "blocking", "threads"}
        unknown = set(item) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"reasoning checkpoint contains unknown keys: {names}")
        if item.get("type") != "checkpoint":
            raise ValueError("reasoning checkpoint type must be 'checkpoint'")
        memory = _required_mapping(item.get("memory"), label="reasoning checkpoint memory")
        threads = _required_mapping(item.get("threads"), label="reasoning checkpoint threads")
        return cls(
            memory=dict(memory),
            active=_required_thread_name_set(item.get("active"), label="reasoning checkpoint active"),
            blocking=_required_thread_name_set(item.get("blocking"), label="reasoning checkpoint blocking"),
            threads={
                _required_thread_name(thread, label="reasoning checkpoint threads key"): _required_message_list(
                    messages,
                    label=f"reasoning checkpoint threads.{thread}",
                )
                for thread, messages in threads.items()
            },
        )


@dataclass(frozen=True, slots=True)
class ReasoningPatch:
    memory: JSONPatch
    active: set[str] | None = None
    blocking: set[str] | None = None
    threads: dict[str, JSONPatch] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory", _required_patch(self.memory, label="reasoning memory patch"))
        if self.active is not None:
            object.__setattr__(
                self,
                "active",
                {_required_thread_name(thread, label="active thread") for thread in self.active},
            )
        if self.blocking is not None:
            object.__setattr__(
                self,
                "blocking",
                {_required_thread_name(thread, label="blocking thread") for thread in self.blocking},
            )
        normalized: dict[str, JSONPatch] = {}
        for raw_thread, patch in self.threads.items():
            thread = _required_thread_name(raw_thread, label="reasoning patch threads key")
            if thread == "main":
                raise ValueError("reasoning patch threads may not target main")
            normalized[thread] = _required_patch(patch, label=f"reasoning patch threads.{thread}")
        object.__setattr__(self, "threads", normalized)

    def to_primitive(self) -> dict[str, object]:
        return {
            "type": "patch",
            "memory": list(self.memory),
            "active": None if self.active is None else sorted(self.active),
            "blocking": None if self.blocking is None else sorted(self.blocking),
            "threads": {thread: list(patch) for thread, patch in sorted(self.threads.items())},
        }

    @classmethod
    def from_primitive(cls, value: object) -> ReasoningPatch:
        item = _required_mapping(value, label="reasoning patch")
        allowed = {"type", "memory", "active", "blocking", "threads"}
        unknown = set(item) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"reasoning patch contains unknown keys: {names}")
        if item.get("type") != "patch":
            raise ValueError("reasoning patch type must be 'patch'")
        threads = _required_mapping(item.get("threads"), label="reasoning patch threads")
        active = item.get("active")
        blocking = item.get("blocking")
        return cls(
            memory=_required_patch(item.get("memory"), label="reasoning memory patch"),
            active=None if active is None else _required_thread_name_set(active, label="reasoning patch active"),
            blocking=None if blocking is None else _required_thread_name_set(blocking, label="reasoning patch blocking"),
            threads={
                _required_thread_name(thread, label="reasoning patch threads key"): _required_patch(
                    patch,
                    label=f"reasoning patch threads.{thread}",
                )
                for thread, patch in threads.items()
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
    memory: dict[str, JSONValue]
    threads: Threads

    def __post_init__(self) -> None:
        _required_string(self.id, label="compaction payload id")

    def to_primitive(self) -> dict[str, object]:
        return {
            "id": self.id,
            "memory": dict(self.memory),
            "threads": self.threads.to_primitive(),
        }

    @classmethod
    def from_primitive(cls, value: object) -> CompactionPayload:
        item = _required_mapping(value, label="compaction payload")
        memory_value = item.get("memory")
        if not isinstance(memory_value, Mapping):
            raise TypeError("compaction payload memory must be an object")
        return cls(
            id=_required_string(item.get("id"), label="compaction payload id"),
            memory=dict(memory_value),
            threads=Threads.from_primitive(item.get("threads")),
        )


@dataclass(frozen=True, slots=True)
class CallID:
    thread: str
    upstream_tool_call_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "thread", _required_thread_name(self.thread, label="function call thread"))


@dataclass(slots=True)
class Ingested:
    memory: dict[str, JSONValue]
    threads: Threads
    main_tail: MainTail | None
    last_reasoning_id: str | None
    last_compaction_id: str | None = None
    checkpoint_required: bool = False
