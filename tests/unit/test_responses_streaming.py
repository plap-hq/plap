from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

import pytest
from pydantic import TypeAdapter

from plap.keyring import SealingKeyring
from plap.responses.contracts import (
    OutputRefusalContent,
    ResponseCreateRequest,
    ResponseMessageItem,
    ResponseStreamEvent,
    SummaryTextContent,
)
from plap.responses.ingest.models import Message, ReasoningCheckpoint, ReasoningPatch
from plap.responses.ingest.sealing import open_reasoning_payload
from plap.responses.store import PreparedRequest
from plap.responses.streaming import StreamCoordinator
from plap.responses.summary import SummaryDelta, SummaryDone

_STREAM_EVENT_ADAPTER = TypeAdapter(ResponseStreamEvent)


class _RecordingChannels:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, object]]] = []

    async def wait_published(self, data: dict[str, object], channels: str | Sequence[str]) -> None:
        channel_names = [channels] if isinstance(channels, str) else list(channels)
        for channel_name in channel_names:
            self.published.append((channel_name, data))


class _RecordingStore:
    def __init__(self) -> None:
        self.begin_calls = 0
        self.append_calls = 0
        self.replace_calls = 0
        self.finish_calls = 0
        self.cancel_calls = 0
        self.fail_calls = 0

    async def begin_response(self, prepared: PreparedRequest, response) -> None:
        _ = prepared, response
        self.begin_calls += 1

    async def append_output_item(self, prepared: PreparedRequest, response_id: str, output_index: int, item: object) -> None:
        _ = prepared, response_id, output_index, item
        self.append_calls += 1

    async def replace_output_item(self, prepared: PreparedRequest, response_id: str, output_index: int, item: object) -> None:
        _ = prepared, response_id, output_index, item
        self.replace_calls += 1

    async def finish_response(self, prepared: PreparedRequest, response) -> None:
        _ = prepared, response
        self.finish_calls += 1

    async def cancel_response(self, prepared: PreparedRequest, response) -> bool:
        _ = prepared, response
        self.cancel_calls += 1
        return True

    async def fail_response(self, prepared: PreparedRequest, response_id: str) -> bool:
        _ = prepared, response_id
        self.fail_calls += 1
        return True


def _keyring() -> SealingKeyring:
    return SealingKeyring(roots=(b"i" * 32,))


def _request() -> ResponseCreateRequest:
    return ResponseCreateRequest(model="plap/test", input="hello")


def _prepared() -> PreparedRequest:
    request = _request()
    return PreparedRequest(
        scope_id=uuid4(),
        response_request=request,
        execution_request=request,
        current_input_items=[],
        stored_input_items=[],
        parent_response_id=None,
        conversation_id=None,
        persist_response=True,
    )


def _reasoning_args(label: str) -> dict[str, object]:
    return {
        "state": ReasoningPatch(memory=[]),
        "main": [Message(role="assistant", content=label)],
    }


def _checkpoint_args(label: str) -> dict[str, object]:
    return {
        "state": ReasoningCheckpoint(memory={"checkpoint": True}, active={"main"}, threads={}),
        "main": [Message(role="assistant", content=label)],
    }


def _last_output_item(coordinator: StreamCoordinator):
    return coordinator.current_response().output[-1]


def _open_reasoning_item(item) -> object:
    return open_reasoning_payload(item.encrypted_content, keyring=_keyring())


def _published_event_types(channels: _RecordingChannels) -> list[str]:
    return [_STREAM_EVENT_ADAPTER.validate_python(payload).type for _, payload in channels.published]


async def test_begin_then_finish_reasoning_chains_previous_reasoning_id() -> None:
    channels = _RecordingChannels()
    coordinator = StreamCoordinator(request=_request(), channels=channels, sealing_keyring=_keyring())

    first_id = await coordinator.begin_reasoning(**_reasoning_args("first"))
    await coordinator.finish_reasoning(**_reasoning_args("first"))
    first_item = _last_output_item(coordinator)
    first_payload = _open_reasoning_item(first_item)

    second_id = await coordinator.begin_reasoning(**_reasoning_args("second"))
    await coordinator.finish_reasoning(**_reasoning_args("second"))
    second_item = _last_output_item(coordinator)
    second_payload = _open_reasoning_item(second_item)

    assert first_id == first_item.id == first_payload.id
    assert first_payload.previous_reasoning_id is None
    assert first_payload.previous_compaction_id is None
    assert second_id == second_item.id == second_payload.id
    assert second_payload.previous_reasoning_id == first_payload.id
    assert second_payload.previous_compaction_id is None


async def test_reasoning_item_lineage_stays_stable_across_replace_and_finish() -> None:
    channels = _RecordingChannels()
    coordinator = StreamCoordinator(
        request=_request(),
        channels=channels,
        sealing_keyring=_keyring(),
        last_compaction_id="cmp_seed",
    )

    reasoning_id = await coordinator.begin_reasoning(**_reasoning_args("draft"))
    begun = _open_reasoning_item(_last_output_item(coordinator))

    await coordinator.replace_reasoning(**_reasoning_args("draft replace"))
    replaced = _open_reasoning_item(_last_output_item(coordinator))

    await coordinator.summary_delta(SummaryDelta(text="summary part"))
    await coordinator.summary_done(SummaryDone())
    await coordinator.finish_reasoning(**_reasoning_args("draft final"))
    finished_item = _last_output_item(coordinator)
    finished = _open_reasoning_item(finished_item)

    await coordinator.begin_reasoning(**_reasoning_args("next"))
    await coordinator.finish_reasoning(**_reasoning_args("next"))
    next_payload = _open_reasoning_item(_last_output_item(coordinator))

    assert begun.id == replaced.id == finished.id == finished_item.id == reasoning_id
    assert begun.previous_reasoning_id == replaced.previous_reasoning_id == finished.previous_reasoning_id
    assert begun.previous_compaction_id == replaced.previous_compaction_id == finished.previous_compaction_id == "cmp_seed"
    assert finished_item.summary == [SummaryTextContent(text="summary part", type="summary_text")]
    assert next_payload.previous_reasoning_id == finished.id


async def test_reasoning_summary_deltas_publish_expected_event_types() -> None:
    channels = _RecordingChannels()
    coordinator = StreamCoordinator(request=_request(), channels=channels, sealing_keyring=_keyring())

    await coordinator.begin_reasoning(**_reasoning_args("draft"))
    await coordinator.summary_delta(SummaryDelta(text="part"))
    await coordinator.summary_done(SummaryDone())

    assert _published_event_types(channels) == [
        "response.output_item.added",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_summary_part.done",
    ]


async def test_finish_reasoning_rejects_pending_summary_text() -> None:
    channels = _RecordingChannels()
    coordinator = StreamCoordinator(request=_request(), channels=channels, sealing_keyring=_keyring())

    await coordinator.begin_reasoning(**_reasoning_args("draft"))
    await coordinator.summary_delta(SummaryDelta(text="part"))

    try:
        await coordinator.finish_reasoning(**_reasoning_args("draft final"))
    except RuntimeError as exc:
        assert "pending" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected finish_reasoning to reject pending summary text")


async def test_begin_then_finish_reasoning_uses_seeded_chain_state() -> None:
    channels = _RecordingChannels()
    coordinator = StreamCoordinator(
        request=_request(),
        channels=channels,
        sealing_keyring=_keyring(),
        last_reasoning_id="rs_seed",
        last_compaction_id="cmp_seed",
    )

    await coordinator.begin_reasoning(**_reasoning_args("seeded"))
    await coordinator.finish_reasoning(**_reasoning_args("seeded"))
    payload = _open_reasoning_item(_last_output_item(coordinator))

    assert payload.previous_reasoning_id == "rs_seed"
    assert payload.previous_compaction_id == "cmp_seed"


async def test_checkpoint_resets_seeded_chain_and_next_patch_chains_to_checkpoint() -> None:
    channels = _RecordingChannels()
    coordinator = StreamCoordinator(
        request=_request(),
        channels=channels,
        sealing_keyring=_keyring(),
        last_reasoning_id="rs_seed",
        last_compaction_id="cmp_seed",
    )

    await coordinator.begin_reasoning(**_checkpoint_args("checkpoint"))
    checkpoint_id = await coordinator.finish_reasoning(**_checkpoint_args("checkpoint"))
    checkpoint = _open_reasoning_item(_last_output_item(coordinator))
    await coordinator.begin_reasoning(**_reasoning_args("patch"))
    await coordinator.finish_reasoning(**_reasoning_args("patch"))
    patch = _open_reasoning_item(_last_output_item(coordinator))

    assert checkpoint.previous_reasoning_id is None
    assert checkpoint.previous_compaction_id == "cmp_seed"
    assert patch.previous_reasoning_id == checkpoint_id
    assert patch.previous_compaction_id == "cmp_seed"


async def test_cancelled_flushes_active_reasoning_item_without_completing_chain() -> None:
    channels = _RecordingChannels()
    coordinator = StreamCoordinator(request=_request(), channels=channels, sealing_keyring=_keyring())

    reasoning_id = await coordinator.begin_reasoning(**_reasoning_args("draft"))
    await coordinator.summary_delta(SummaryDelta(text="part"))
    await coordinator.summary_done(SummaryDone())
    await coordinator.cancelled()

    response = coordinator.current_response()
    item = response.output[-1]
    payload = _open_reasoning_item(item)

    assert response.status == "cancelled"
    assert item.id == reasoning_id
    assert item.summary == [SummaryTextContent(text="part", type="summary_text")]
    assert payload.previous_reasoning_id is None


async def test_cancelled_checkpoint_preserves_variant_and_null_predecessor() -> None:
    channels = _RecordingChannels()
    coordinator = StreamCoordinator(
        request=_request(),
        channels=channels,
        sealing_keyring=_keyring(),
        last_reasoning_id="rs_seed",
        last_compaction_id="cmp_seed",
    )

    await coordinator.begin_reasoning(**_checkpoint_args("draft"))
    await coordinator.replace_reasoning(**_checkpoint_args("replacement"))
    await coordinator.cancelled()
    payload = _open_reasoning_item(_last_output_item(coordinator))

    assert isinstance(payload.state, ReasoningCheckpoint)
    assert payload.previous_reasoning_id is None
    assert payload.previous_compaction_id == "cmp_seed"
    assert payload.main == [Message(role="assistant", content="replacement")]


async def test_active_reasoning_draft_cannot_change_variant() -> None:
    channels = _RecordingChannels()
    coordinator = StreamCoordinator(request=_request(), channels=channels, sealing_keyring=_keyring())

    await coordinator.begin_reasoning(**_checkpoint_args("checkpoint"))

    with pytest.raises(RuntimeError, match="state type cannot change"):
        await coordinator.replace_reasoning(**_reasoning_args("patch"))


async def test_summary_done_does_not_write_store() -> None:
    channels = _RecordingChannels()
    store = _RecordingStore()
    coordinator = StreamCoordinator(
        request=_request(),
        channels=channels,
        prepared=_prepared(),
        response_store=store,
        sealing_keyring=_keyring(),
    )

    await coordinator.begin_reasoning(**_reasoning_args("draft"))
    assert store.append_calls == 1

    await coordinator.summary_delta(SummaryDelta(text="part"))
    await coordinator.summary_done(SummaryDone())

    assert store.replace_calls == 0

    await coordinator.finish_reasoning(**_reasoning_args("done"))

    assert store.replace_calls == 1


async def test_emit_message_with_refusal_publishes_refusal_events() -> None:
    channels = _RecordingChannels()
    coordinator = StreamCoordinator(request=_request(), channels=channels, sealing_keyring=_keyring())

    await coordinator.emit(
        ResponseMessageItem(
            content=[OutputRefusalContent(refusal="nope", type="refusal")],
            id="msg_refusal",
            role="assistant",
            status="completed",
            type="message",
        )
    )

    assert _published_event_types(channels) == [
        "response.output_item.added",
        "response.content_part.added",
        "response.refusal.delta",
        "response.refusal.done",
        "response.content_part.done",
        "response.output_item.done",
    ]
