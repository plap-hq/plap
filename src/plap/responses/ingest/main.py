from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from plap.responses.contracts import RequestFunctionCallItem, RequestFunctionCallOutputItem
from plap.responses.ingest import content
from plap.responses.ingest.errors import reasoning_replay_error, tool_replay_error
from plap.responses.ingest.models import (
    CompactedMainTail,
    HiddenMainTail,
    MainTail,
    MainUpdate,
    Message,
    MessagePatch,
    PublicMainTail,
    ToolCall,
    split_main_updates,
)


class CallPhase(StrEnum):
    DECLARED = "declared"
    OPEN = "open"
    CLOSED = "closed"


def _copy_assistant(
    private: Message,
    *,
    public: Message | None = None,
    tool_calls: list[ToolCall] | None = None,
) -> Message:
    return Message(
        role="assistant",
        content=private.content if public is None else public.content,
        name=private.name,
        refusal=private.refusal if public is None else public.refusal,
        tool_calls=list(private.tool_calls if tool_calls is None else tool_calls),
        reasoning_content=private.reasoning_content,
    )


@dataclass(slots=True)
class _Call:
    id: str
    name: str
    arguments: str
    order: int
    phase: CallPhase = CallPhase.DECLARED

    @classmethod
    def declared(cls, tool_call: ToolCall, *, order: int) -> _Call:
        return cls(
            id=tool_call.id,
            name=tool_call.name,
            arguments=tool_call.arguments,
            order=order,
        )

    @classmethod
    def opened(cls, item: RequestFunctionCallItem, *, call_id: str, order: int) -> _Call:
        return cls(
            id=call_id,
            name=item.name,
            arguments=item.arguments,
            order=order,
            phase=CallPhase.OPEN,
        )

    def open(self) -> None:
        if self.phase == CallPhase.CLOSED:
            raise tool_replay_error(
                reason="function_call_already_satisfied",
                private_message="function_call already has a tool output in history",
            )
        if self.phase == CallPhase.OPEN:
            raise tool_replay_error(
                reason="duplicate_pending_function_call",
                private_message="duplicate pending function_call",
            )
        self.phase = CallPhase.OPEN

    def settle_hidden(self) -> None:
        if self.phase == CallPhase.CLOSED:
            raise tool_replay_error(
                reason="function_call_output_without_pending_function_call",
                private_message="function_call already has a tool output",
            )
        self.phase = CallPhase.CLOSED

    def close(self) -> None:
        if self.phase != CallPhase.OPEN:
            raise tool_replay_error(
                reason="function_call_output_without_pending_function_call",
                private_message="function_call_output has no pending function_call",
            )
        self.phase = CallPhase.CLOSED

    def to_tool_call(self) -> ToolCall:
        return ToolCall(id=self.id, name=self.name, arguments=self.arguments)


@dataclass(slots=True)
class _AssistantBundle:
    private: Message
    source: Message | None
    calls: list[_Call] = field(default_factory=list)
    outputs: list[Message] = field(default_factory=list)
    public: Message | None = None
    snapshot: bool = False

    def call(self, call_id: str) -> _Call | None:
        for call in self.calls:
            if call.id == call_id:
                return call
        return None

    def remove_call(self, call: _Call) -> None:
        self.calls.remove(call)

    def append_call(self, call: _Call) -> None:
        self.calls.append(call)

    def assistant(self) -> Message:
        calls = [call.to_tool_call() for call in sorted(self.calls, key=lambda call: call.order)]
        return _copy_assistant(self.private, public=self.public, tool_calls=calls)

    def render(self) -> list[Message]:
        return [self.assistant(), *self.outputs]

    def tail(self) -> MainTail:
        if self.public is not None:
            return PublicMainTail(source=self.source)
        if self.snapshot:
            if self.source is None:  # pragma: no cover - snapshots are authenticated
                raise RuntimeError("compacted main tail has no authenticated source")
            return CompactedMainTail(source=self.source)
        if self.source is None:  # pragma: no cover - hidden bundles are authenticated
            raise RuntimeError("hidden main tail has no authenticated source")
        return HiddenMainTail(source=self.source)


type _Node = Message | _AssistantBundle


@dataclass(slots=True)
class MainReplay:
    nodes: list[_Node] = field(default_factory=list)
    authenticated_tail: _AssistantBundle | None = None
    pending_public: _AssistantBundle | None = None
    _next_call_order: int = 0

    def _new_call_order(self) -> int:
        order = self._next_call_order
        self._next_call_order += 1
        return order

    def _bundles(self) -> list[_AssistantBundle]:
        return [node for node in self.nodes if isinstance(node, _AssistantBundle)]

    def _find_call(self, call_id: str) -> tuple[_AssistantBundle, _Call] | None:
        bundles = self._bundles()
        if self.pending_public is not None:
            bundles.append(self.pending_public)
        for bundle in reversed(bundles):
            call = bundle.call(call_id)
            if call is not None:
                return bundle, call
        return None

    def _new_bundle(
        self,
        message: Message,
        *,
        source: Message | None,
        public: Message | None = None,
        snapshot: bool = False,
    ) -> _AssistantBundle:
        bundle = _AssistantBundle(
            private=message,
            source=source,
            public=public,
            snapshot=snapshot,
        )
        for tool_call in message.tool_calls:
            if self._find_call(tool_call.id) is not None:
                raise tool_replay_error(
                    reason="duplicate_tool_call_id_in_history",
                    private_message="tool call id appears more than once in main history",
                )
            bundle.calls.append(_Call.declared(tool_call, order=self._new_call_order()))
        return bundle

    def _last_assistant(self) -> _AssistantBundle | None:
        for node in reversed(self.nodes):
            if isinstance(node, _AssistantBundle):
                return node
        return None

    def _apply_hidden_output(self, bundle: _AssistantBundle, message: Message) -> None:
        call_id = message.tool_call_id
        if call_id is None:
            raise reasoning_replay_error(
                reason="reasoning_message_invalid",
                private_message="hidden main tool output must include tool_call_id",
            )
        call = bundle.call(call_id)
        if call is None:
            raise reasoning_replay_error(
                reason="reasoning_message_invalid",
                private_message="hidden main tool output does not match the target assistant",
            )
        call.settle_hidden()
        bundle.outputs.append(message)

    def _append_hidden_message(self, message: Message, *, snapshot: bool = False) -> _AssistantBundle | None:
        if message.is_tool():
            owner = self._find_call(message.tool_call_id or "")
            if owner is None:
                raise reasoning_replay_error(
                    reason="reasoning_message_invalid",
                    private_message="hidden main tool output has no declaration",
                )
            self._apply_hidden_output(owner[0], message)
            return None
        if message.is_assistant():
            bundle = self._new_bundle(message, source=message, snapshot=snapshot)
            self.nodes.append(bundle)
            self.authenticated_tail = bundle
            return bundle
        self.nodes.append(message)
        return None

    def _stage_patch(self, patch: MessagePatch) -> _AssistantBundle:
        source = self.authenticated_tail
        if source is not None and source.source == patch.message:
            if source in self.nodes:
                self.nodes.remove(source)
            elif source is not self.pending_public:  # pragma: no cover - one patch per reasoning item
                raise RuntimeError("authenticated main source is not present")
            bundle = source
        else:
            bundle = self._new_bundle(patch.message, source=patch.message)
        self.authenticated_tail = bundle
        return bundle

    def _finish_patch(self, bundle: _AssistantBundle) -> None:
        if content.assistant_output(bundle.private):
            if self.pending_public is not None:
                raise reasoning_replay_error(
                    reason="reasoning_message_patch_invalid",
                    private_message="main assistant is already pending public materialization",
                )
            self.pending_public = bundle
            return
        self.nodes.append(bundle)

    def _append_local_update(
        self,
        *,
        prefix: list[Message],
        anchor: Message | None,
        suffix: list[Message],
        after: list[Message],
    ) -> _AssistantBundle | None:
        for message in prefix:
            self._append_hidden_message(message)
        local_anchor = None if anchor is None else self._append_hidden_message(anchor)
        if anchor is not None and local_anchor is None:  # pragma: no cover - split_main_updates guarantees assistant anchor
            raise TypeError("reasoning main anchor must be an assistant")
        if local_anchor is not None:
            for message in suffix:
                self._apply_hidden_output(local_anchor, message)
        for message in after:
            self._append_hidden_message(message)
        return local_anchor

    def assert_no_pending_patch(self) -> None:
        if self.pending_public is not None:
            raise reasoning_replay_error(
                reason="main_message_patch_target_missing",
                private_message="message patch target is missing",
            )

    def load_snapshot(self, messages: list[Message]) -> None:
        self.nodes = []
        self.authenticated_tail = None
        self.pending_public = None
        self._next_call_order = 0
        for message in messages:
            self._append_hidden_message(message, snapshot=True)

    def patch_matches_authenticated(self, updates: list[MainUpdate]) -> bool:
        patch = split_main_updates(updates)[5]
        return patch is not None and self.authenticated_tail is not None and self.authenticated_tail.source == patch.message

    def apply_update(self, updates: list[MainUpdate]) -> None:
        if self.pending_public is not None:
            raise reasoning_replay_error(
                reason="main_message_patch_target_missing",
                private_message="message patch target is missing before the next reasoning item",
            )
        leading_outputs, prefix, anchor, suffix, after, patch = split_main_updates(updates)
        baseline = self.authenticated_tail
        local_anchor = self._append_local_update(prefix=prefix, anchor=anchor, suffix=suffix, after=after)

        target = local_anchor
        if patch is not None:
            target = self._stage_patch(patch)
        if leading_outputs:
            leading_target = baseline if prefix or anchor is not None or after else target
            if leading_target is None:
                leading_target = self.authenticated_tail
            if leading_target is None:
                raise reasoning_replay_error(
                    reason="reasoning_message_invalid",
                    private_message="leading main outputs require an authenticated assistant",
                )
            for output in leading_outputs:
                self._apply_hidden_output(leading_target, output)
        if patch is not None:
            if target is None:  # pragma: no cover - _stage_patch always returns a bundle
                raise RuntimeError("message patch requires a target bundle")
            self._finish_patch(target)

    def append_message(self, message: Message) -> None:
        if message.is_assistant() and self.pending_public is not None:
            bundle = self.pending_public
            bundle.public = message
            self.nodes.append(bundle)
            self.pending_public = None
            return
        if message.is_assistant():
            self.nodes.append(self._new_bundle(message, source=None, public=message))
            return
        self.nodes.append(message)

    def add_call(self, item: RequestFunctionCallItem, *, call_id: str) -> None:
        if self.pending_public is not None:
            raise reasoning_replay_error(
                reason="main_message_patch_target_missing",
                private_message="message patch target is missing before function_call",
            )
        owner = self._last_assistant()
        if owner is None:
            raise tool_replay_error(
                reason="sealed_function_call_without_attachment_owner",
                private_message="function_call has no previous assistant message",
            )
        existing = self._find_call(call_id)
        if existing is None:
            owner.append_call(_Call.opened(item, call_id=call_id, order=self._new_call_order()))
            return
        old_owner, call = existing
        call.open()
        if old_owner is owner:
            return
        old_owner.remove_call(call)
        owner.append_call(call)

    def add_output(self, item: RequestFunctionCallOutputItem, *, call_id: str) -> None:
        owner = self._find_call(call_id)
        if owner is None:
            raise tool_replay_error(
                reason="function_call_output_without_pending_function_call",
                private_message="function_call_output has no pending function_call",
            )
        bundle, call = owner
        call.close()
        bundle.outputs.append(Message(role="tool", tool_call_id=call_id, content=content.tool_output(item)))

    def phases(self) -> dict[str, CallPhase]:
        return {call.id: call.phase for bundle in self._bundles() for call in bundle.calls}

    def interrupt_declared(self, *, output: str) -> None:
        if self.pending_public is not None:
            raise reasoning_replay_error(
                reason="main_message_patch_target_missing",
                private_message="cannot interrupt calls before public patch attachment",
            )
        for bundle in self._bundles():
            for call in bundle.calls:
                if call.phase != CallPhase.DECLARED:
                    continue
                call.settle_hidden()
                bundle.outputs.append(Message(role="tool", tool_call_id=call.id, content=output))

    def current_messages(self) -> list[Message]:
        rendered: list[Message] = []
        for node in self.nodes:
            if isinstance(node, _AssistantBundle):
                rendered.extend(node.render())
                continue
            rendered.append(node)
        return rendered

    def tail(self) -> MainTail | None:
        bundle = self._last_assistant()
        return None if bundle is None else bundle.tail()


__all__ = ["CallPhase", "MainReplay"]
