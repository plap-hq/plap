from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

import pytest
import svcs

from plap.auth import AuthContext
from plap.bus import EventBus
from plap.config import CueBox
from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.keyring import SealingKeyring
from plap.responses.contracts import ResponseCreateRequest
from plap.responses.ingest.models import MAIN_SIDE, Ingested, Sides
from plap.responses.routes import _prepare_create, _run_stream
from plap.responses.store import PreparedRequest, ResponseStore
from plap.responses.streaming import StreamCoordinator


class _RecordingChannels:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, object]]] = []

    async def wait_published(self, data: dict[str, object], channels: str | Sequence[str]) -> None:
        channel_names = [channels] if isinstance(channels, str) else list(channels)
        for channel_name in channel_names:
            self.published.append((channel_name, data))


class _RecordingStore:
    def __init__(self, prepared: PreparedRequest) -> None:
        self._prepared = prepared
        self.prepare_calls = 0
        self.begin_calls = 0

    async def prepare_request(self, auth_context: AuthContext, request: ResponseCreateRequest) -> PreparedRequest:
        _ = auth_context, request
        self.prepare_calls += 1
        return self._prepared

    async def begin_response(self, prepared: PreparedRequest, response) -> None:
        _ = prepared, response
        self.begin_calls += 1


def _auth_context() -> AuthContext:
    return AuthContext(
        api_key_id=uuid4(),
        organization_id=None,
        user_id=uuid4(),
    )


def _keyring() -> SealingKeyring:
    return SealingKeyring(roots=(b"i" * 32,))


def _loaded() -> CueBox:
    return CueBox(
        {
            "plap": CueBox(
                {
                    "config": CueBox(
                        {
                            "sides": {"main": 0},
                        },
                        frozen_box=True,
                    )
                },
                frozen_box=True,
            )
        },
        frozen_box=True,
    )


def _request() -> ResponseCreateRequest:
    return ResponseCreateRequest(model="plap/test", input="hello")


def _prepared(request: ResponseCreateRequest | None = None) -> PreparedRequest:
    response_request = request or _request()
    return PreparedRequest(
        scope_id=uuid4(),
        response_request=response_request,
        execution_request=response_request,
        current_input_items=[],
        stored_input_items=[],
        parent_response_id=None,
        conversation_id=None,
        persist_response=True,
    )


def _coordinator() -> StreamCoordinator:
    return StreamCoordinator(request=_request(), channels=_RecordingChannels())


def _plap_error() -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code="test_error",
            message="test error",
        ),
        private=PrivateError(
            event="test.error",
            reason="test_error",
            message="test error",
            level=ErrorLevel.WARNING,
        ),
    )


def _svcs() -> svcs.Container:
    registry = svcs.Registry()
    registry.register_value(AuthContext, _auth_context())
    registry.register_value(CueBox, _loaded())
    registry.register_value(SealingKeyring, _keyring())
    store = _RecordingStore(_prepared())
    registry.register_value(_RecordingStore, store)

    class FakeResponseStore:
        def __init__(self, inner: _RecordingStore) -> None:
            self._inner = inner

        async def prepare_request(self, auth_context, request):
            return await self._inner.prepare_request(auth_context, request)

        async def begin_response(self, prepared, response):
            await self._inner.begin_response(prepared, response)

    registry.register_value(FakeResponseStore, FakeResponseStore(store))
    registry.register_value(ResponseStore, FakeResponseStore(store))

    return svcs.Container(registry)


async def test_prepare_create_prepares_and_stops_before_created() -> None:
    request = _request()
    store = _RecordingStore(_prepared(request))
    registry = svcs.Registry()
    registry.register_value(AuthContext, _auth_context())
    registry.register_value(CueBox, _loaded())
    registry.register_value(SealingKeyring, _keyring())
    registry.register_value(ResponseStore, store)
    container = svcs.Container(registry)

    prepared, ingested, coordinator = await _prepare_create(
        svcs=container,
        auth_context=_auth_context(),
        request=request,
        channels=_RecordingChannels(),
    )

    assert store.prepare_calls == 1
    assert store.begin_calls == 0
    assert prepared.response_request is request
    assert prepared.execution_request is request
    assert ingested.machine == {}
    assert ingested.sides.messages.keys() == {"main"}
    assert ingested.sides["main"][0].role == "user"
    assert ingested.sides["main"][0].content == "hello"
    assert ingested.sides.active == {MAIN_SIDE}
    assert ingested.last_reasoning_id is None
    assert ingested.current_compaction_id is None
    assert coordinator.current_response().model == request.model


async def test_run_stream_swallows_runtime_plap_error(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = EventBus()

    @bus.listen("response.start")
    async def _boom(*, next, state) -> None:
        _ = next, state
        raise _plap_error()

    monkeypatch.setattr("plap.responses.routes.bus", bus)

    registry = svcs.Registry()
    registry.register_value(CueBox, _loaded())
    registry.register_value(SealingKeyring, _keyring())
    container = svcs.Container(registry)

    await _run_stream(
        prepared=_prepared(),
        ingested=Ingested(machine={}, sides=Sides(), last_reasoning_id=None, current_compaction_id=None),
        coordinator=_coordinator(),
        svcs=container,
    )
