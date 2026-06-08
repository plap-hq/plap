from __future__ import annotations

import secrets
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from math import ceil
from typing import cast

import structlog
from pydantic import BaseModel, ConfigDict

from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.keyring import SealingKeyring
from plap.llms import ChatReasoningSummarizer, RetryLimitExceededError, SummaryDelta, SummaryDone, with_summary
from plap.llms.completions.chat import (
    ChatCompletionRequest,
    ChatFinishReason,
    ChatFunctionTool,
    ChatMessage,
    ChatResponseFormat,
    ChatStreamOptions,
    ChatTool,
    ChatToolCall,
    ChatToolChoiceFunction,
    ChatUsage,
    IChatCompletionClient,
)
from plap.llms.retry import retry_on_tool_choice_mismatch, retry_on_unusable_tool_calls
from plap.llms.retry import stream as retry_stream
from plap.logging import log_debug, log_payload
from plap.responses.contracts import (
    FunctionTool,
    OutputTextContent,
    ResponseFunctionCallItem,
    ResponseMessageItem,
    ResponseUsage,
    ResponseUsageInputTokensDetails,
    ResponseUsageOutputTokensDetails,
    ToolChoiceFunction,
)
from plap.responses.ingest.models import MAIN_SIDE, CallID, GuardedPatch, Ingested, Message, MessagePatch, Sides, SidesUpdate, split_tail
from plap.responses.ingest.patch import JSONPatch, JSONValue, diff
from plap.responses.ingest.sealing import seal_call_id
from plap.responses.store import PreparedRequest
from plap.responses.streaming import StreamCoordinator
from plap.settings import PublicUsageConfig, RuntimeModelProfileConfig
from plap.tools import IMCPToolProvider, IToolCallPolicyResolver, IToolPolicyResolver

INTERRUPTED_TOOL_OUTPUT = "Tool output unavailable because the response was interrupted."
DEVELOPER_PROMPT_TEMPLATE = "You are {model_name}, an AI assistant."
logger = structlog.get_logger(__name__)


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
            Message(role="tool", tool_call_id=tool_call.id, content=INTERRUPTED_TOOL_OUTPUT)
            for tool_call in open_calls
        )
        return shadow

    def _message_patch(self, message: Message) -> MessagePatch | None:
        if message.content is None:
            return None
        tool_calls = list(message.tool_calls) or None
        if tool_calls is None and message.reasoning_content is None:
            return None
        visible = Message(role="assistant", content=message.content)
        return MessagePatch(
            content_hash=visible.content_hash(),
            tool_calls=tool_calls,
            reasoning_content=message.reasoning_content,
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


def _summary_mode(prepared: PreparedRequest) -> str | None:
    reasoning = prepared.execution_request.reasoning
    if reasoning is None:
        return None
    return reasoning.summary or reasoning.generate_summary


def _response_format(prepared: PreparedRequest) -> ChatResponseFormat | None:
    text = prepared.execution_request.text
    if text is None or text.format is None:
        return None
    format_ = text.format
    if format_.type == "json_schema":
        return ChatResponseFormat(
            type=format_.type,
            name=format_.name,
            schema=format_.schema_,
            strict=format_.strict,
            description=format_.description,
        )
    return ChatResponseFormat(type=format_.type)


def _tool_choice(prepared: PreparedRequest) -> str | ChatToolChoiceFunction | None:
    tool_choice = prepared.execution_request.tool_choice
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, ToolChoiceFunction):  # pragma: no cover
        raise TypeError(f"unsupported tool choice: {type(tool_choice).__name__}")
    return ChatToolChoiceFunction(name=tool_choice.name)


def _base_request(
    prepared: PreparedRequest,
    state: State,
    profile: RuntimeModelProfileConfig,
) -> ChatCompletionRequest:
    request = prepared.execution_request
    top_logprobs = profile.main.map_top_logprobs(request.top_logprobs)
    instructions = [
        ChatMessage(
            role="developer",
            content=DEVELOPER_PROMPT_TEMPLATE.format(model_name=profile.display_name),
        )
    ]
    if request.instructions is not None:
        instructions.append(ChatMessage(role="developer", content=request.instructions))
    tools = [
        ChatTool(
            function=ChatFunctionTool(
                name=tool.name,
                parameters=tool.parameters,
                strict=tool.strict,
                description=tool.description,
            )
        )
        for tool in request.tools or []
        if isinstance(tool, FunctionTool)
    ]
    return ChatCompletionRequest(
        model=profile.main.model,
        messages=[*instructions, *deepcopy(state.sides.get(MAIN_SIDE) or [])],
        tools=tools,
        tool_choice=_tool_choice(prepared),
        parallel_tool_calls=request.parallel_tool_calls,
        response_format=_response_format(prepared),
        temperature=profile.main.map_temperature(request.temperature),
        top_p=profile.main.map_top_p(request.top_p),
        logprobs=True if top_logprobs is not None else None,
        top_logprobs=top_logprobs,
        reasoning_effort=profile.main.reasoning_effort,
        stream_options=ChatStreamOptions(include_usage=True),
        user=request.user,
        prompt_cache_key=request.prompt_cache_key,
        metadata=request.metadata,
        service_tier=profile.main.service_tier,
    )


def _retry_limit_error() -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=500,
            type="server_error",
            code="internal_error",
            message="An unexpected error occurred.",
        ),
        private=PrivateError(
            event="response.retry_limit_exceeded",
            reason="retry_limit_exceeded",
            message="responses execution exhausted retry attempts after unusable tool calls",
            level=ErrorLevel.ERROR,
        ),
    )


async def execute(
    *,
    state: State,
    prepared: PreparedRequest,
    profile: RuntimeModelProfileConfig,
    chat_completion_client: IChatCompletionClient,
    tool_policy_resolver: IToolPolicyResolver,
    tool_call_policy_resolver: IToolCallPolicyResolver,
    mcp_tool_providers: tuple[IMCPToolProvider, ...],
) -> None:
    _ = tool_policy_resolver, tool_call_policy_resolver, mcp_tool_providers

    base_request = _base_request(prepared, state, profile)
    summary_mode = _summary_mode(prepared)
    ledger = UsageLedger(
        budget=prepared.execution_request.max_output_tokens,
        reasoning_to_output=profile.reasoning_to_output,
    )
    hidden_results_accounted = 0
    input_anchor_seen = False
    budget_exhausted = False
    latest_snapshot = None

    log_debug(
        logger,
        "response.runtime.turn",
        continuation_side=MAIN_SIDE,
        main_model=base_request.model,
        reasoning_summary_mode=summary_mode,
        tool_count=len(base_request.tools),
    )

    def next_request(history):
        nonlocal hidden_results_accounted, input_anchor_seen, budget_exhausted
        for result in history.results[hidden_results_accounted:]:
            if not input_anchor_seen:
                ledger.set_input_anchor(result.usage)
                input_anchor_seen = True
            ledger.record_hidden(profile.main.public_usage, result.usage)
            hidden_results_accounted += 1
        attempt_index = hidden_results_accounted + 1
        attempt_budget = ledger.cap_for(profile.main.public_usage)
        attempt_cap = profile.main.cap_max_completion_tokens(attempt_budget)
        if attempt_cap == 0:
            budget_exhausted = True
            log_debug(
                logger,
                "response.runtime.main_request.skipped",
                attempt_budget=attempt_budget,
                attempt_index=attempt_index,
                hidden_history_messages=len(history.messages),
                hidden_history_results=len(history.results),
                remaining_budget=ledger.remaining(),
                reason="budget_exhausted",
            )
            return None
        attempt_request = replace(
            base_request,
            messages=[*base_request.messages, *history.messages],
            max_completion_tokens=attempt_cap,
        )
        log_debug(
            logger,
            "response.runtime.main_request",
            attempt_budget=attempt_budget,
            attempt_index=attempt_index,
            hidden_history_messages=len(history.messages),
            hidden_history_results=len(history.results),
            main_cap=attempt_cap,
            remaining_budget=ledger.remaining(),
        )
        log_payload(
            logger,
            "response.runtime.main_request.payload",
            attempt_index=attempt_index,
            request=asdict(attempt_request),
        )
        return attempt_request

    source = retry_stream(
        chat_completion_client,
        next_request=next_request,
        validators=(retry_on_tool_choice_mismatch, retry_on_unusable_tool_calls),
    )

    try:
        if summary_mode is None:
            async for snapshot in source:
                latest_snapshot = snapshot
                state.main = list(snapshot.messages)
        else:
            summarizer = ChatReasoningSummarizer(
                client=chat_completion_client,
                model=profile.reasoning_summarizer.model,
                prompt_cache_key=prepared.execution_request.prompt_cache_key,
                reasoning_effort=profile.reasoning_summarizer.reasoning_effort,
                service_tier=profile.reasoning_summarizer.service_tier,
            )
            async with with_summary(source, mode=summary_mode, summarizer=summarizer) as items:
                async for item in items:
                    if isinstance(item, SummaryDelta):
                        if state._reasoning_id is None:
                            await state.flush()
                        await state._coordinator.summary_delta(item)
                        continue
                    if isinstance(item, SummaryDone):
                        await state._coordinator.summary_done(item)
                        await state.flush()
                        continue
                    latest_snapshot = item
                    state.main = list(item.messages)
    except* RetryLimitExceededError as exc:
        if latest_snapshot is not None:
            state.main = list(latest_snapshot.messages)
        await state.finalize()
        raise _retry_limit_error() from exc.exceptions[0]

    if latest_snapshot is not None:
        state.main = list(latest_snapshot.messages)
    await state.finalize()

    if latest_snapshot is None:
        if budget_exhausted:
            await state._coordinator.incomplete(usage=ledger.to_response_usage())
            return
        raise RuntimeError("response stream produced no snapshots")

    accepted_results = list(latest_snapshot.results[hidden_results_accounted:])
    if accepted_results:
        final_result = accepted_results[-1]
        if not input_anchor_seen:
            ledger.set_input_anchor(final_result.usage)
            input_anchor_seen = True
        ledger.record_output(profile.main.public_usage, final_result.usage)
        usage = ledger.to_response_usage()
        if final_result.finish_reason == ChatFinishReason.LENGTH:
            await state._coordinator.incomplete(service_tier=final_result.service_tier, usage=usage)
            return
        await state._coordinator.completed(service_tier=final_result.service_tier, usage=usage)
        return

    if budget_exhausted:
        service_tier = latest_snapshot.results[-1].service_tier if latest_snapshot.results else None
        await state._coordinator.incomplete(service_tier=service_tier, usage=ledger.to_response_usage())
        return

    raise RuntimeError("response stream ended without an accepted final result")
