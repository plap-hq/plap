from __future__ import annotations

import secrets
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import cast

import svcs
from pydantic import BaseModel, ConfigDict

from plap.keyring import SealingKeyring
from plap.llms.completions.chat import ChatToolCall
from plap.responses.contracts import ResponseFunctionCallItem, ResponseMessageItem
from plap.responses.ingest import content
from plap.responses.ingest.models import (
    MAIN_SIDE,
    CallID,
    HiddenMainTail,
    Ingested,
    MainTail,
    Message,
    MessagePatch,
    ReasoningCheckpoint,
    ReasoningPatch,
    ReasoningState,
    Side,
    Sides,
    split_tail,
)
from plap.responses.ingest.patch import JSONPatch, JSONValue, diff
from plap.responses.ingest.sealing import seal_call_id
from plap.responses.store import PreparedRequest
from plap.responses.streaming import StreamCoordinator

INTERRUPTED_TOOL_OUTPUT = "Tool call aborted because the response was interrupted."


class DurableState(BaseModel):
    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_primitive(cls, value: dict[str, JSONValue]) -> DurableState:
        return cls.model_validate(value)

    def to_primitive(self) -> dict[str, JSONValue]:
        dumped = self.model_dump(mode="json", exclude_none=True)
        if not isinstance(dumped, dict):  # pragma: no cover
            raise TypeError("durable state dump must be an object")
        return cast(dict[str, JSONValue], dumped)


@dataclass(slots=True)
class State:
    prepared: PreparedRequest
    svcs: svcs.Container
    coordinator: StreamCoordinator
    _sealing_keyring: SealingKeyring
    _side_codes: Mapping[str, int]
    _base_durable: DurableState
    _base_sides: Sides
    _base_main_tail: MainTail | None
    _reasoning_id: str | None
    _checkpoint_required: bool

    durable: DurableState
    sides: Sides

    @classmethod
    def from_ingested(
        cls,
        *,
        ingested: Ingested,
        prepared: PreparedRequest,
        svcs: svcs.Container,
        coordinator: StreamCoordinator,
        sealing_keyring: SealingKeyring,
        side_codes: Mapping[str, int],
    ) -> State:
        base_durable = DurableState.from_primitive(ingested.durable)
        base_sides = deepcopy(ingested.sides)
        base_sides.setdefault(MAIN_SIDE)
        return cls(
            prepared=prepared,
            svcs=svcs,
            coordinator=coordinator,
            _sealing_keyring=sealing_keyring,
            _side_codes=dict(side_codes),
            _base_durable=base_durable,
            _base_sides=base_sides,
            _base_main_tail=deepcopy(ingested.main_tail),
            _reasoning_id=None,
            _checkpoint_required=ingested.checkpoint_required,
            durable=base_durable.model_copy(deep=True),
            sides=deepcopy(base_sides),
        )

    def _split_tail(
        self,
        messages: list[Message],
        *,
        label: str,
    ) -> tuple[list[Message], Message | None, list[Message], list[Message], list[ChatToolCall]]:
        prefix, anchor, suffix, after = split_tail(messages)
        if anchor is None:
            if any(message.is_tool() for message in after):
                raise RuntimeError(f"{label} without an assistant anchor may not contain tool outputs")
            return prefix, None, suffix, after, []

        if self._split_tail(prefix, label=f"{label} prefix")[4]:
            raise RuntimeError(f"{label} has unresolved tool calls before its final anchor")
        if not isinstance(anchor, Message):  # pragma: no cover
            raise TypeError(f"{label}: expected a message anchor")
        pending = {tool_call.id: tool_call for tool_call in anchor.tool_calls}
        for index, tool_message in enumerate(suffix, start=len(prefix) + 1):
            if not tool_message.is_tool():
                raise RuntimeError(f"{label}[{index}] must be a tool output after the final anchor")
            if tool_message.tool_call_id is None or tool_message.tool_call_id not in pending:
                raise RuntimeError(f"{label}[{index}] does not match an open tool call on the final anchor")
            pending.pop(tool_message.tool_call_id)
        open_calls = [tool_call for tool_call in anchor.tool_calls if tool_call.id in pending]
        if open_calls and after:
            raise RuntimeError(f"{label} with unresolved tool calls may not have trailing non-assistant tail")
        for index, message in enumerate(after, start=len(prefix) + 1 + len(suffix)):
            if message.is_assistant() or message.is_tool():
                raise RuntimeError(f"{label}[{index}] must be a closed non-assistant tail message")
        return prefix, anchor, suffix, after, open_calls

    def _stubbed(self, messages: list[Message], *, label: str) -> list[Message]:
        prefix, anchor, suffix, after, open_calls = self._split_tail(messages, label=label)
        if anchor is None:
            return [*deepcopy(prefix), *deepcopy(after)]
        shadow = [*deepcopy(prefix), anchor, *deepcopy(suffix), *deepcopy(after)]
        shadow.extend(Message(role="tool", tool_call_id=tool_call.id, content=INTERRUPTED_TOOL_OUTPUT) for tool_call in open_calls)
        return shadow

    def _message_item(self, message: Message) -> ResponseMessageItem | None:
        if not message.is_assistant():
            return None
        output = content.assistant_output(message)
        if not output:
            return None
        return ResponseMessageItem(
            content=output,
            id=f"msg_{secrets.token_urlsafe(18)}",
            role="assistant",
            status="completed",
            type="message",
        )

    def _function_items(self, side: str, tool_calls: list[ChatToolCall]) -> list[ResponseFunctionCallItem]:
        items: list[ResponseFunctionCallItem] = []
        for tool_call in tool_calls:
            sealed_call_id = seal_call_id(
                CallID(side=side, upstream_tool_call_id=tool_call.id),
                keyring=self._sealing_keyring,
                side_codes=self._side_codes,
            )
            items.append(
                ResponseFunctionCallItem(
                    arguments=tool_call.arguments,
                    call_id=sealed_call_id,
                    id=f"fc_{secrets.token_urlsafe(18)}",
                    name=tool_call.name,
                    status="completed",
                    type="function_call",
                )
            )
        return items

    def _main_suffix(self, sides: Sides) -> list[Message]:
        base = self._base_sides[MAIN_SIDE]
        current = sides[MAIN_SIDE]
        if len(current) < len(base) or current[: len(base)] != base:
            raise RuntimeError("persisted main history is immutable; replace only the current response suffix")
        return current[len(base) :]

    def _build_update(
        self,
        *,
        durable: DurableState,
        sides: Sides,
    ) -> ReasoningState:
        self._main_suffix(sides)

        if self._checkpoint_required:
            return ReasoningCheckpoint(
                durable=durable.to_primitive(),
                active=set(sides.active),
                sides={side: deepcopy(messages) for side, messages in sides.items() if side != MAIN_SIDE},
            )

        durable_patch = diff(self._base_durable.to_primitive(), durable.to_primitive())
        patches: dict[str, JSONPatch] = {}
        for side in sorted(set(self._base_sides.messages) | set(sides.messages)):
            if side == MAIN_SIDE:
                continue
            base_present = side in self._base_sides.messages
            current_present = side in sides.messages
            base_messages = self._base_sides.get(side)
            current_messages = sides.get(side)
            if base_present == current_present and base_messages == current_messages:
                continue
            patches[side] = diff(
                [] if base_messages is None else [message.to_primitive() for message in base_messages],
                [] if current_messages is None else [message.to_primitive() for message in current_messages],
            )
        active = None if self._base_sides.active == sides.active else set(sides.active)
        return ReasoningPatch(durable=durable_patch, active=active, sides=patches)

    def _validate_main(self) -> None:
        self._main_suffix(self.sides)
        self._split_tail(self.sides[MAIN_SIDE], label="main history")

    def _logical_main(self, *, sides: Sides | None = None) -> Message | None:
        current_sides = self.sides if sides is None else sides
        _, anchor, suffix, after, open_calls = self._split_tail(current_sides[MAIN_SIDE], label="main history")
        if anchor is None or after:
            return None
        if suffix and not open_calls:
            return None
        return anchor

    def _main_publication(
        self,
        *,
        sides: Sides,
        main: list[Message],
    ) -> tuple[Message | None, Message | None]:
        if MAIN_SIDE not in sides.active:
            return None, None
        visible = self._logical_main(sides=sides)
        if visible is None:
            return None, None
        if any(message is visible for message in main):
            return visible, None
        if not isinstance(self._base_main_tail, HiddenMainTail):
            return None, None
        return visible, self._base_main_tail.source

    def _main_update(
        self,
        main: list[Message],
        visible: Message | None,
        visible_message: ResponseMessageItem | None,
        *,
        has_visible_calls: bool,
        persisted_source: Message | None,
    ) -> list[Message | MessagePatch]:
        if visible is None or (visible_message is None and not has_visible_calls):
            return deepcopy(main)

        visible_index = next((index for index, message in enumerate(main) if message is visible), None)
        if visible_index is None:
            if persisted_source is None:  # pragma: no cover - _main_publication supplies persisted provenance
                raise RuntimeError("persisted main publication requires an authenticated source")
            return [*deepcopy(main), MessagePatch(message=deepcopy(persisted_source))]

        if visible_message is None:
            return deepcopy(main)
        if visible == content.assistant_message(list(visible_message.content)):
            return [*deepcopy(main[:visible_index]), *deepcopy(main[visible_index + 1 :])]
        return [*deepcopy(main), MessagePatch(message=deepcopy(visible))]

    def _update_empty(self, state: ReasoningState, main: list[Message | MessagePatch]) -> bool:
        if main or isinstance(state, ReasoningCheckpoint):
            return False
        return not state.durable and state.active is None and not state.sides

    def _staged_snapshot(self) -> tuple[DurableState, Sides, list[Message]]:
        self._validate_main()
        shadow_durable = self.durable.model_copy(deep=True)
        shadow_sides = deepcopy(self.sides)
        for side in list(shadow_sides.messages):
            if side != MAIN_SIDE and side in shadow_sides.active:
                shadow_sides[side] = self._stubbed(shadow_sides[side], label=f"{side} side")
        if MAIN_SIDE in shadow_sides.active:
            open_calls = self._split_tail(shadow_sides[MAIN_SIDE], label="main history")[4]
            shadow_sides[MAIN_SIDE].extend(
                Message(role="tool", tool_call_id=tool_call.id, content=INTERRUPTED_TOOL_OUTPUT) for tool_call in open_calls
            )
        return shadow_durable, shadow_sides, self._main_suffix(shadow_sides)

    def _commit_snapshot(self) -> tuple[DurableState, Sides, list[Message]]:
        self._validate_main()
        for side, messages in self.sides.items():
            if side == MAIN_SIDE:
                continue
            self._split_tail(messages, label=f"{side} side")
        sides = deepcopy(self.sides)
        return self.durable.model_copy(deep=True), sides, self._main_suffix(sides)

    def activate(self, side: Side) -> None:
        if side not in self._side_codes:
            raise ValueError(f"cannot activate unconfigured side {side!r}")
        self.sides.active.add(side)

    def deactivate(self, side: Side) -> None:
        if side not in self._side_codes:
            raise ValueError(f"cannot deactivate unconfigured side {side!r}")
        self.sides.active.discard(side)

    def open_calls(self, side: Side) -> list[ChatToolCall]:
        return self._split_tail(self.sides.get(side, []) or [], label=f"{side} history")[4]

    async def stage(self) -> None:
        durable, sides, main = self._staged_snapshot()
        state = self._build_update(durable=durable, sides=sides)
        main_update = deepcopy(main)
        if self._reasoning_id is None:
            if self._update_empty(state, main_update):
                return
            self._reasoning_id = await self.coordinator.begin_reasoning(state=state, main=main_update)
            return
        await self.coordinator.replace_reasoning(state=state, main=main_update)

    async def ensure_staged(self) -> None:
        if self._reasoning_id is None:
            await self.stage()

    async def commit(self) -> None:
        durable, sides, main = self._commit_snapshot()
        visible_main, persisted_source = self._main_publication(sides=sides, main=main)
        visible_message = None if visible_main is None else self._message_item(visible_main)
        visible_calls: list[ResponseFunctionCallItem] = []
        main_open_calls: list[ChatToolCall] = []
        if visible_main is not None:
            main_open_calls = self._split_tail(sides[MAIN_SIDE], label="main history")[4]
            visible_calls.extend(self._function_items(MAIN_SIDE, main_open_calls))
        main_update = self._main_update(
            main,
            visible_main,
            visible_message,
            has_visible_calls=bool(main_open_calls),
            persisted_source=persisted_source,
        )

        state = self._build_update(durable=durable, sides=sides)

        for side in sorted(sides.messages):
            if side == MAIN_SIDE or side not in sides.active:
                continue
            messages = sides[side]
            visible_calls.extend(self._function_items(side, self._split_tail(messages, label=f"{side} side")[4]))

        if self._reasoning_id is None and not self._update_empty(state, main_update):
            self._reasoning_id = await self.coordinator.begin_reasoning(state=state, main=main_update)
        if self._reasoning_id is not None:
            await self.coordinator.finish_reasoning(state=state, main=main_update)
            self._reasoning_id = None
            self._checkpoint_required = False
        if visible_message is not None:
            await self.coordinator.emit(visible_message)
        for item in visible_calls:
            await self.coordinator.emit(item)
