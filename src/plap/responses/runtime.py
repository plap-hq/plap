from __future__ import annotations

import secrets
from copy import deepcopy
from dataclasses import dataclass, field
from math import ceil
from typing import cast

import anyio
import structlog
from pydantic import BaseModel, ConfigDict

from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.keyring import SealingKeyring
from plap.llms.completions.chat import ChatToolCall, ChatUsage, IChatCompletionClient
from plap.responses.contracts import (
    OutputTextContent,
    ResponseFunctionCallItem,
    ResponseMessageItem,
    ResponseUsage,
    ResponseUsageInputTokensDetails,
    ResponseUsageOutputTokensDetails,
)
from plap.responses.ingest.models import MAIN_SIDE, CallID, GuardedPatch, Ingested, Message, MessagePatch, Sides, SidesUpdate, split_tail
from plap.responses.ingest.sealing import seal_call_id
from plap.responses.patch import JSONPatch, JSONValue, diff
from plap.responses.store import PreparedRequest
from plap.responses.streaming import StreamCoordinator
from plap.settings import PublicUsageConfig, RuntimeSelector, Settings
from plap.tools import IToolCallPolicyResolver, IToolPolicyResolver
from plap.tools.mcp import IMCPToolProvider

logger = structlog.get_logger(__name__)
_INTERRUPTED_TOOL_OUTPUT_TEXT = "Tool output unavailable because the response was interrupted."


class Machine(BaseModel):
    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_primitive(cls, value: dict[str, JSONValue]) -> Machine:
        return cls.model_validate(value)

    def to_primitive(self) -> dict[str, JSONValue]:
        dumped = self.model_dump(mode="json", exclude_none=True)
        if not isinstance(dumped, dict):  # pragma: no cover
            raise TypeError("machine dump must be an object")
        return cast(dict[str, JSONValue], dumped)


def _cached_input_tokens(usage: ChatUsage) -> int:
    return min(usage.cached_tokens or 0, usage.input_tokens)


def _visible_output_tokens(usage: ChatUsage) -> int:
    return max(0, usage.output_tokens - (usage.reasoning_tokens or 0))


def _output_equivalent_tokens(usage: ChatUsage, *, reasoning_to_output: float) -> int:
    return _visible_output_tokens(usage) + ceil((usage.reasoning_tokens or 0) * reasoning_to_output)


def _output_debit(config: PublicUsageConfig, usage: ChatUsage, *, reasoning_to_output: float) -> int:
    return ceil(config.output_to_output * _output_equivalent_tokens(usage, reasoning_to_output=reasoning_to_output))


def _hidden_debit(config: PublicUsageConfig, usage: ChatUsage, *, reasoning_to_output: float) -> int:
    cached_input = _cached_input_tokens(usage)
    uncached_input = usage.input_tokens - cached_input
    debit = (
        uncached_input * config.uncached_input_to_output
        + cached_input * config.cached_input_to_output
        + config.output_to_output * _output_equivalent_tokens(usage, reasoning_to_output=reasoning_to_output)
    )
    return ceil(debit)


def _build_response_usage(
    *,
    input_tokens: int,
    cached_tokens: int,
    visible_tokens: int,
    normalized_output_tokens: int,
) -> ResponseUsage:
    output_tokens = max(visible_tokens, normalized_output_tokens)
    reasoning_tokens = output_tokens - visible_tokens
    return ResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details=ResponseUsageInputTokensDetails(cached_tokens=cached_tokens),
        output_tokens=output_tokens,
        output_tokens_details=ResponseUsageOutputTokensDetails(reasoning_tokens=reasoning_tokens),
        total_tokens=input_tokens + output_tokens,
    )


@dataclass(slots=True)
class UsageLedger:
    budget: int | None
    reasoning_to_output: float
    hidden: list[tuple[PublicUsageConfig, ChatUsage]] = field(default_factory=list)
    output: list[tuple[PublicUsageConfig, ChatUsage]] = field(default_factory=list)
    hidden_output: list[int] = field(default_factory=list)
    input_anchor: ChatUsage | None = None

    def remaining(self) -> int | None:
        return self.budget

    def cap_for(self, config: PublicUsageConfig) -> int | None:
        return config.cap_from_budget(self.budget)

    def record_hidden(self, config: PublicUsageConfig, usage: ChatUsage | None) -> int | None:
        if usage is None:
            return None
        if self.budget is not None:
            self.budget -= _hidden_debit(config, usage, reasoning_to_output=self.reasoning_to_output)
        self.hidden.append((config, usage))
        return len(self.hidden) - 1

    def record_output(self, config: PublicUsageConfig, usage: ChatUsage | None) -> None:
        if usage is None:
            return
        if self.budget is not None:
            self.budget -= _output_debit(config, usage, reasoning_to_output=self.reasoning_to_output)
        self.output.append((config, usage))

    def promote_hidden_to_output(self, index: int) -> None:
        if index < 0 or index >= len(self.hidden):
            raise IndexError(index)
        if index in self.hidden_output:
            raise ValueError("hidden usage is already visible output")
        self.hidden_output.append(index)

    def set_input_anchor(self, usage: ChatUsage | None) -> None:
        if usage is None:
            return
        if self.input_anchor is not None:
            raise ValueError("input usage anchor is already set")
        self.input_anchor = usage

    def to_response_usage(self) -> ResponseUsage | None:
        if self.input_anchor is None:
            return None

        visible_tokens = sum(_visible_output_tokens(usage) for _, usage in self.output)
        visible_tokens += sum(_visible_output_tokens(self.hidden[index][1]) for index in self.hidden_output)
        normalized_output_tokens = sum(
            _output_debit(config, usage, reasoning_to_output=self.reasoning_to_output) for config, usage in self.output
        )
        normalized_output_tokens += sum(
            _hidden_debit(config, usage, reasoning_to_output=self.reasoning_to_output) for config, usage in self.hidden
        )
        return _build_response_usage(
            input_tokens=self.input_anchor.input_tokens,
            cached_tokens=_cached_input_tokens(self.input_anchor),
            visible_tokens=visible_tokens,
            normalized_output_tokens=normalized_output_tokens,
        )


@dataclass(slots=True)
class State:
    _coordinator: StreamCoordinator
    _sealing_keyring: SealingKeyring
    _base_machine: Machine
    _base_sides: Sides
    _reasoning_id: str | None

    machine: Machine
    sides: Sides
    main: list[Message]

    @classmethod
    def from_ingested(
        cls,
        coordinator: StreamCoordinator,
        sealing_keyring: SealingKeyring,
        ingested: Ingested,
    ) -> State:
        base_machine = Machine.from_primitive(ingested.machine)
        base_sides = deepcopy(ingested.sides)
        return cls(
            _coordinator=coordinator,
            _sealing_keyring=sealing_keyring,
            _base_machine=base_machine,
            _base_sides=base_sides,
            _reasoning_id=None,
            machine=base_machine.model_copy(deep=True),
            sides=deepcopy(base_sides),
            main=[],
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

    def _assert_closed(self, messages: list[Message], *, label: str) -> None:
        if self._split_tail(messages, label=label)[4]:
            raise RuntimeError(f"{label} contains unresolved tool calls")

    def _closed_and_residue(
        self,
        messages: list[Message],
        *,
        label: str,
    ) -> tuple[list[Message], list[Message]]:
        prefix, anchor, suffix, after, open_calls = self._split_tail(messages, label=label)
        if anchor is None:
            return [*deepcopy(prefix), *deepcopy(after)], []
        if not open_calls:
            return deepcopy(messages), []
        return prefix, [anchor, *suffix]

    def _stubbed(self, messages: list[Message], *, label: str) -> list[Message]:
        prefix, anchor, suffix, after, open_calls = self._split_tail(messages, label=label)
        if anchor is None:
            return [*deepcopy(prefix), *deepcopy(after)]
        shadow = [*deepcopy(prefix), anchor, *deepcopy(suffix), *deepcopy(after)]
        shadow.extend(
            Message(role="tool", tool_call_id=tool_call.id, content=_INTERRUPTED_TOOL_OUTPUT_TEXT)
            for tool_call in open_calls
        )
        return shadow

    def _message_patch(self, message: Message) -> MessagePatch | None:
        if message.content is None:
            return None
        tool_calls = list(message.tool_calls) or None
        reasoning_details = list(message.reasoning_details) or None
        if tool_calls is None and message.reasoning_content is None and reasoning_details is None:
            return None
        visible = Message(role="assistant", content=message.content)
        return MessagePatch(
            content_hash=visible.content_hash(),
            tool_calls=tool_calls,
            reasoning_content=message.reasoning_content,
            reasoning_details=reasoning_details,
        )

    def _message_item(self, message: Message) -> ResponseMessageItem | None:
        if not message.is_assistant() or not message.content:
            return None
        return ResponseMessageItem(
            content=[OutputTextContent(text=message.content, type="output_text")],
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

    def _build_update(
        self,
        *,
        machine: Machine,
        sides: Sides,
        main_update: list[Message | MessagePatch],
    ) -> tuple[JSONPatch, SidesUpdate]:
        machine_patch = diff(self._base_machine.to_primitive(), machine.to_primitive())
        patches: dict[str, GuardedPatch] = {}
        for side in sorted(set(self._base_sides.messages) | set(sides.messages)):
            base_present = side in self._base_sides.messages
            current_present = side in sides.messages
            base_messages = self._base_sides.get(side)
            current_messages = sides.get(side)
            if base_present == current_present and base_messages == current_messages:
                continue
            patches[side] = GuardedPatch(
                shape=self._base_sides.shape(side),
                patch=diff(
                    [] if base_messages is None else [message.to_primitive() for message in base_messages],
                    [] if current_messages is None else [message.to_primitive() for message in current_messages],
                ),
            )
        return machine_patch, SidesUpdate(main=main_update, patches=patches)

    def _shadow_snapshot(self) -> tuple[Machine, Sides, list[Message]]:
        main_patch_lane = self.sides.get(MAIN_SIDE)
        self._assert_closed([] if main_patch_lane is None else main_patch_lane, label="main patch lane")
        shadow_machine = self.machine.model_copy(deep=True)
        shadow_sides = deepcopy(self.sides)
        for side in list(shadow_sides.messages):
            if side == MAIN_SIDE:
                continue
            shadow_sides[side] = self._stubbed(shadow_sides[side], label=f"{side} side")
        shadow_main = self._stubbed(self.main, label="main append lane")
        return shadow_machine, shadow_sides, shadow_main

    def _live_snapshot(self) -> tuple[Machine, Sides, list[Message]]:
        main_patch_lane = self.sides.get(MAIN_SIDE)
        self._assert_closed([] if main_patch_lane is None else main_patch_lane, label="main patch lane")
        self._split_tail(self.main, label="main append lane")
        for side, messages in self.sides.items():
            if side == MAIN_SIDE:
                continue
            self._split_tail(messages, label=f"{side} side")
        return self.machine.model_copy(deep=True), deepcopy(self.sides), deepcopy(self.main)

    async def flush(self) -> None:
        machine, sides, main = self._shadow_snapshot()
        machine_patch, sides_update = self._build_update(machine=machine, sides=sides, main_update=deepcopy(main))
        if self._reasoning_id is None:
            if not machine_patch and not sides_update.main and not sides_update.patches:
                return
            self._reasoning_id = await self._coordinator.begin_reasoning(machine=machine_patch, sides=sides_update)
            return
        await self._coordinator.replace_reasoning(machine=machine_patch, sides=sides_update)

    async def finalize(self) -> None:
        machine, sides, main = self._live_snapshot()
        visible_message: ResponseMessageItem | None = None
        visible_calls: list[ResponseFunctionCallItem] = []
        main_update: list[Message | MessagePatch] = []

        if main:
            prefix, anchor, suffix, after, open_calls = self._split_tail(main, label="main append lane")
            if anchor is None:
                main_update = [*deepcopy(prefix), *deepcopy(after)]
            else:
                if after:
                    main_update = [*deepcopy(prefix), anchor, *deepcopy(suffix)]
                    main_update.extend(deepcopy(after))
                else:
                    patch = self._message_patch(anchor)
                    visible_message = self._message_item(anchor)
                    if visible_message is None:
                        main_update = [*deepcopy(prefix), anchor, *deepcopy(suffix)]
                    elif patch is not None or suffix:
                        if patch is None:  # pragma: no cover
                            raise RuntimeError("assistant patch is required when hidden fields are present")
                        main_update = [*deepcopy(prefix), patch, *deepcopy(suffix)]
                    else:
                        main_update = deepcopy(prefix)
                    visible_calls.extend(self._function_items(MAIN_SIDE, open_calls))

        machine_patch, sides_update = self._build_update(machine=machine, sides=sides, main_update=main_update)

        for side, messages in sides.items():
            if side == MAIN_SIDE:
                continue
            visible_calls.extend(self._function_items(side, self._split_tail(messages, label=f"{side} side")[4]))

        if self._reasoning_id is None and (machine_patch or sides_update.main or sides_update.patches):
            self._reasoning_id = await self._coordinator.begin_reasoning(machine=machine_patch, sides=sides_update)
        if self._reasoning_id is not None:
            await self._coordinator.finish_reasoning(machine=machine_patch, sides=sides_update)
            self._reasoning_id = None
        if visible_message is not None:
            await self._coordinator.emit(visible_message)
        for item in visible_calls:
            await self._coordinator.emit(item)

    async def compaction(self, *, created_by: str | None = None) -> None:
        if self._reasoning_id is not None:
            await self._coordinator.finish_reasoning(machine=[], sides=SidesUpdate())
            self._reasoning_id = None

        full_machine = self.machine.model_copy(deep=True)
        full_sides = deepcopy(self.sides)
        residue_sides: dict[str, list[Message]] = {}

        for side, messages in list(full_sides.items()):
            if side == MAIN_SIDE:
                self._assert_closed(messages, label="main patch lane")
                continue
            closed, residue = self._closed_and_residue(messages, label=f"{side} side")
            full_sides[side] = closed
            if residue:
                residue_sides[side] = residue

        closed_main, residue_main = self._closed_and_residue(
            self.main,
            label="main append lane",
        )
        if closed_main:
            full_sides[MAIN_SIDE] = [*(full_sides.get(MAIN_SIDE) or []), *closed_main]

        await self._coordinator.emit_compaction(
            machine=full_machine.to_primitive(),
            sides=full_sides,
            created_by=created_by,
        )

        self._base_machine = full_machine
        self._base_sides = deepcopy(full_sides)
        self.machine = full_machine.model_copy(deep=True)
        self.sides = deepcopy(full_sides)
        for side, residue in residue_sides.items():
            self.sides[side] = [*(self.sides.get(side) or []), *residue]
        self.main = residue_main

        if residue_main or residue_sides:
            await self.flush()


def _selector_from_request(request) -> RuntimeSelector:
    reasoning = request.reasoning
    return RuntimeSelector(
        service_tier=request.service_tier,
        reasoning_effort=reasoning.effort if reasoning is not None else None,
    )


def _not_implemented_error() -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=501,
            type="server_error",
            code="not_implemented",
            message="responses execution is not implemented yet",
        ),
        private=PrivateError(
            event="response.not_implemented",
            reason="responses_execution_not_implemented",
            message="responses execution is not implemented yet",
            level=ErrorLevel.WARNING,
        ),
    )


def _unexpected_public_error() -> PublicError:
    return PublicError(
        status_code=500,
        type="server_error",
        code="internal_error",
        message="An unexpected error occurred.",
    )


async def _execute_not_implemented(
    *,
    state: State,
    prepared: PreparedRequest,
    profile,
    chat_completion_client: IChatCompletionClient,
    tool_policy_resolver: IToolPolicyResolver,
    tool_call_policy_resolver: IToolCallPolicyResolver,
    mcp_tool_providers: tuple[IMCPToolProvider, ...],
) -> None:
    _ = (
        state,
        prepared,
        profile,
        chat_completion_client,
        tool_policy_resolver,
        tool_call_policy_resolver,
        mcp_tool_providers,
    )
    raise _not_implemented_error()


async def run_response(
    *,
    prepared: PreparedRequest,
    ingested: Ingested,
    coordinator: StreamCoordinator,
    sealing_keyring: SealingKeyring,
    settings: Settings,
    chat_completion_client: IChatCompletionClient,
    tool_policy_resolver: IToolPolicyResolver,
    tool_call_policy_resolver: IToolCallPolicyResolver,
    mcp_tool_providers: tuple[IMCPToolProvider, ...],
) -> None:
    created = False
    state: State | None = None
    try:
        await anyio.sleep(0)
        profile = settings.resolve_runtime_model_profile(
            prepared.response_request.model,
            selector=_selector_from_request(prepared.response_request),
        )
        await anyio.sleep(0)
        with anyio.CancelScope(shield=True):
            await coordinator.created()
        created = True
        await coordinator.in_progress()
        state = State.from_ingested(coordinator, sealing_keyring, ingested)
        await _execute_not_implemented(
            state=state,
            prepared=prepared,
            profile=profile,
            chat_completion_client=chat_completion_client,
            tool_policy_resolver=tool_policy_resolver,
            tool_call_policy_resolver=tool_call_policy_resolver,
            mcp_tool_providers=mcp_tool_providers,
        )
    except anyio.get_cancelled_exc_class():
        if created:
            if state is not None:
                with anyio.CancelScope(shield=True):
                    await state.flush()
            with anyio.CancelScope(shield=True):
                await coordinator.cancelled()
        raise
    except PlapError as exc:
        public = exc.public or _unexpected_public_error()
        exc.log(
            logger,
            response_id=coordinator.response_id,
            failure_code=public.code,
            failure_type=public.type,
            status_code=public.status_code,
        )
        with anyio.CancelScope(shield=True):
            await coordinator.fail(public)
        raise
    except Exception:
        public = _unexpected_public_error()
        logger.exception(
            "responses execution failed",
            response_id=coordinator.response_id,
            failure_code=public.code,
            failure_type=public.type,
            status_code=public.status_code,
        )
        with anyio.CancelScope(shield=True):
            await coordinator.fail(public)
        raise
