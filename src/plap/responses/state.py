from __future__ import annotations

import secrets
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

import svcs

from plap.config import CueBox
from plap.keyring import SealingKeyring
from plap.llms.completions.chat import ChatToolCall
from plap.responses.contracts import ResponseCreateRequest, ResponseFunctionCallItem, ResponseMessageItem
from plap.responses.ingest import content
from plap.responses.ingest.models import (
    CallID,
    HiddenMainTail,
    Ingested,
    MainTail,
    Message,
    MessagePatch,
    ReasoningCheckpoint,
    ReasoningPatch,
    ReasoningState,
    Threads,
    split_tail,
)
from plap.responses.ingest.patch import JSONPatch, JSONValue, diff
from plap.responses.ingest.sealing import seal_call_id
from plap.responses.streaming import StreamCoordinator

INTERRUPTED_TOOL_OUTPUT = "Tool call aborted because the response was interrupted."


@dataclass(slots=True)
class State:
    request: ResponseCreateRequest
    config: CueBox
    svcs: svcs.Container

    memory: dict[str, JSONValue]
    threads: Threads

    _thread_codes: Mapping[str, int]
    _base_memory: dict[str, JSONValue]
    _base_threads: Threads
    _base_main_tail: MainTail | None
    _reasoning_id: str | None
    checkpoint_required: bool

    @classmethod
    def from_ingested(
        cls,
        *,
        ingested: Ingested,
        request: ResponseCreateRequest,
        config: CueBox,
        svcs: svcs.Container,
        thread_codes: Mapping[str, int],
    ) -> State:
        base_memory = deepcopy(ingested.memory)
        base_threads = deepcopy(ingested.threads)
        base_threads.setdefault("main")
        return cls(
            request=request,
            config=config,
            svcs=svcs,
            _thread_codes=dict(thread_codes),
            _base_memory=base_memory,
            _base_threads=base_threads,
            _base_main_tail=deepcopy(ingested.main_tail),
            _reasoning_id=None,
            checkpoint_required=ingested.checkpoint_required,
            memory=deepcopy(base_memory),
            threads=deepcopy(base_threads),
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

    def _function_items(self, thread: str, tool_calls: list[ChatToolCall]) -> list[ResponseFunctionCallItem]:
        items: list[ResponseFunctionCallItem] = []
        for tool_call in tool_calls:
            sealed_call_id = seal_call_id(
                CallID(thread=thread, upstream_tool_call_id=tool_call.id),
                keyring=self.svcs.get(SealingKeyring),
                thread_codes=self._thread_codes,
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

    def _main_suffix(self, threads: Threads) -> list[Message]:
        base = self._base_threads["main"]
        current = threads["main"]
        if len(current) < len(base) or current[: len(base)] != base:
            raise RuntimeError("persisted main history is immutable; replace only the current response suffix")
        return current[len(base) :]

    def _build_update(
        self,
        *,
        memory: dict[str, JSONValue],
        threads: Threads,
    ) -> ReasoningState:
        self._main_suffix(threads)

        if self.checkpoint_required:
            return ReasoningCheckpoint(
                memory=deepcopy(memory),
                active=set(threads.active),
                threads={thread: deepcopy(messages) for thread, messages in threads.items() if thread != "main"},
            )

        memory_patch = diff(self._base_memory, memory)
        patches: dict[str, JSONPatch] = {}
        for thread in sorted(set(self._base_threads.messages) | set(threads.messages)):
            if thread == "main":
                continue
            base_present = thread in self._base_threads.messages
            current_present = thread in threads.messages
            base_messages = self._base_threads.get(thread)
            current_messages = threads.get(thread)
            if base_present == current_present and base_messages == current_messages:
                continue
            patches[thread] = diff(
                [] if base_messages is None else [message.to_primitive() for message in base_messages],
                [] if current_messages is None else [message.to_primitive() for message in current_messages],
            )
        active = None if self._base_threads.active == threads.active else set(threads.active)
        return ReasoningPatch(memory=memory_patch, active=active, threads=patches)

    def _validate_threads(self) -> None:
        unknown = (self.threads.active | set(self.threads.messages)) - set(self._thread_codes)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"state contains unconfigured threads: {names}")

    def _validate_main(self) -> None:
        self._validate_threads()
        self._main_suffix(self.threads)
        self._split_tail(self.threads["main"], label="main history")

    def _logical_main(self, *, threads: Threads | None = None) -> Message | None:
        current_threads = self.threads if threads is None else threads
        _, anchor, suffix, after, open_calls = self._split_tail(current_threads["main"], label="main history")
        if anchor is None or after:
            return None
        if suffix and not open_calls:
            return None
        return anchor

    def _main_publication(
        self,
        *,
        threads: Threads,
        main: list[Message],
    ) -> tuple[Message | None, Message | None]:
        if "main" not in threads.active:
            return None, None
        visible = self._logical_main(threads=threads)
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
        return not state.memory and state.active is None and not state.threads

    def _progress_snapshot(self) -> tuple[dict[str, JSONValue], Threads, list[Message]]:
        self._validate_main()
        shadow_memory = deepcopy(self.memory)
        shadow_threads = deepcopy(self.threads)
        for thread in list(shadow_threads.messages):
            if thread != "main" and thread in shadow_threads.active:
                shadow_threads[thread] = self._stubbed(shadow_threads[thread], label=f"{thread} thread")
        if "main" in shadow_threads.active:
            open_calls = self._split_tail(shadow_threads["main"], label="main history")[4]
            shadow_threads["main"].extend(
                Message(role="tool", tool_call_id=tool_call.id, content=INTERRUPTED_TOOL_OUTPUT) for tool_call in open_calls
            )
        return shadow_memory, shadow_threads, self._main_suffix(shadow_threads)

    def _commit_snapshot(self) -> tuple[dict[str, JSONValue], Threads, list[Message]]:
        self._validate_main()
        for thread, messages in self.threads.items():
            if thread == "main":
                continue
            self._split_tail(messages, label=f"{thread} thread")
        threads = deepcopy(self.threads)
        return deepcopy(self.memory), threads, self._main_suffix(threads)

    def open_calls(self, thread: str) -> list[ChatToolCall]:
        return self._split_tail(self.threads.get(thread, []) or [], label=f"{thread} history")[4]

    async def save_progress(self) -> None:
        coordinator = self.svcs.get(StreamCoordinator)
        memory, threads, main = self._progress_snapshot()
        state = self._build_update(memory=memory, threads=threads)
        main_update = deepcopy(main)
        if self._reasoning_id is None:
            if self._update_empty(state, main_update):
                return
            self._reasoning_id = await coordinator.begin_reasoning(state=state, main=main_update)
            return
        await coordinator.replace_reasoning(state=state, main=main_update)

    async def ensure_progress(self) -> None:
        if self._reasoning_id is None:
            await self.save_progress()

    async def commit(self) -> None:
        coordinator = self.svcs.get(StreamCoordinator)
        memory, threads, main = self._commit_snapshot()
        visible_main, persisted_source = self._main_publication(threads=threads, main=main)
        visible_message = None if visible_main is None else self._message_item(visible_main)
        visible_calls: list[ResponseFunctionCallItem] = []
        main_open_calls: list[ChatToolCall] = []
        if visible_main is not None:
            main_open_calls = self._split_tail(threads["main"], label="main history")[4]
            visible_calls.extend(self._function_items("main", main_open_calls))
        main_update = self._main_update(
            main,
            visible_main,
            visible_message,
            has_visible_calls=bool(main_open_calls),
            persisted_source=persisted_source,
        )

        state = self._build_update(memory=memory, threads=threads)

        for thread in sorted(threads.messages):
            if thread == "main" or thread not in threads.active:
                continue
            messages = threads[thread]
            visible_calls.extend(self._function_items(thread, self._split_tail(messages, label=f"{thread} thread")[4]))

        if self._reasoning_id is None and not self._update_empty(state, main_update):
            self._reasoning_id = await coordinator.begin_reasoning(state=state, main=main_update)
        if self._reasoning_id is not None:
            await coordinator.finish_reasoning(state=state, main=main_update)
            self._reasoning_id = None
            self.checkpoint_required = False
        if visible_message is not None:
            await coordinator.emit(visible_message)
        for item in visible_calls:
            await coordinator.emit(item)
