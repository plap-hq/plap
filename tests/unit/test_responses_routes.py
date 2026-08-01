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
from plap.llms.completions.budget import CompletionBudget
from plap.responses.contracts import (
    RequestCompactionItem,
    RequestMessageItem,
    RequestReasoningItem,
    ResponseCreateRequest,
    ResponseReasoningItem,
)
from plap.responses.ingest.models import (
    CompactionPayload,
    Ingested,
    ReasoningCheckpoint,
    ReasoningPatch,
    ReasoningPayload,
    Threads,
)
from plap.responses.ingest.sealing import (
    open_reasoning_payload,
    seal_compaction_payload,
    seal_reasoning_payload,
)
from plap.responses.routes import _accept_response, _execute_response, _model_object, _prepare_create, responses_socket
from plap.responses.state import State
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

    async def append_output_item(self, prepared: PreparedRequest, response_id: str, output_index: int, item: object) -> None:
        _ = prepared, response_id, output_index, item

    async def replace_output_item(self, prepared: PreparedRequest, response_id: str, output_index: int, item: object) -> None:
        _ = prepared, response_id, output_index, item


def _auth_context() -> AuthContext:
    return AuthContext(
        api_key_id=uuid4(),
        organization_id=None,
        user_id=uuid4(),
    )


def _keyring() -> SealingKeyring:
    return SealingKeyring(roots=(b"i" * 32,))


class _Config(CueBox):
    def resolve(self, request: dict[str, object] | None = None, /, **kwargs: object) -> _Config:
        _ = request, kwargs
        return self


def _config() -> _Config:
    return _Config(
        {
            "threads": {"main": 0},
            "overlays": {"model": {"plap/test": {}}},
            "main": {
                "output_equivalence": {
                    "uncached_input_to_output": 0.25,
                    "cached_input_to_output": 0.05,
                    "output_to_output": 1.0,
                },
            },
            "reasoning_to_output": 1.0,
        },
        frozen_box=True,
    )


def _request() -> ResponseCreateRequest:
    return ResponseCreateRequest(model="plap/test", input="hello")


def test_model_object_uses_configured_creation_time() -> None:
    config = CueBox(
        {
            "model_info": {
                "created": 1777849810,
                "provider": "plap",
            }
        },
        frozen_box=True,
    )

    model = _model_object(config, model="plap/test")

    assert model.created == config.model_info.created


def _prepared(request: ResponseCreateRequest | None = None) -> PreparedRequest:
    response_request = request or _request()
    return PreparedRequest(
        scope_id=uuid4(),
        response_request=response_request,
        execution_request=response_request,
        stored_input_items=[],
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
    registry.register_value(CueBox, _config())
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


def _state() -> State:
    config = _config()
    container = _svcs()
    container.register_local_value(CompletionBudget, CompletionBudget(None, reasoning_to_output=1.0))
    container.register_local_value(StreamCoordinator, _coordinator())
    return State.from_ingested(
        ingested=Ingested(memory={}, threads=Threads(), main_tail=None, last_reasoning_id=None),
        request=_request(),
        config=config,
        svcs=container,
        thread_codes={"main": 0},
    )


async def test_prepare_create_prepares_and_stops_before_created() -> None:
    request = _request()
    config = _config()
    store = _RecordingStore(_prepared(request))
    registry = svcs.Registry()
    registry.register_value(AuthContext, _auth_context())
    registry.register_value(CueBox, config)
    registry.register_value(SealingKeyring, _keyring())
    registry.register_value(ResponseStore, store)
    container = svcs.Container(registry)

    state = await _prepare_create(
        svcs=container,
        auth_context=_auth_context(),
        request=request,
        channels=_RecordingChannels(),
    )

    assert store.prepare_calls == 1
    assert store.begin_calls == 0
    assert state.request is request
    assert state.config is config
    assert state.svcs.get(CompletionBudget).remaining is None
    assert state.memory == {}
    assert state.threads.messages.keys() == {"main"}
    assert state.threads["main"][0].role == "user"
    assert state.threads["main"][0].content == "hello"
    assert state.threads.active == {"main"}
    assert state.svcs.get(StreamCoordinator).current_response().model == request.model


async def test_prepare_create_preserves_stored_user_boundary_and_resets_lineage() -> None:
    keyring = _keyring()
    stored = ReasoningPayload(
        id="rs_stored",
        previous_reasoning_id=None,
        previous_compaction_id=None,
        state=ReasoningPatch(memory=[]),
    )
    execution_request = ResponseCreateRequest(
        model="plap/test",
        input=[
            RequestReasoningItem(
                encrypted_content=seal_reasoning_payload(stored, keyring=keyring),
                id=stored.id,
                summary=[],
                type="reasoning",
            ),
            RequestMessageItem(content="new turn", role="user", type="message"),
        ],
    )
    response_request = _request()
    prepared = PreparedRequest(
        scope_id=uuid4(),
        response_request=response_request,
        execution_request=execution_request,
        stored_input_items=[],
    )
    store = _RecordingStore(prepared)
    registry = svcs.Registry()
    registry.register_value(AuthContext, _auth_context())
    registry.register_value(CueBox, _config())
    registry.register_value(SealingKeyring, keyring)
    registry.register_value(ResponseStore, store)
    container = svcs.Container(registry)

    state = await _prepare_create(
        svcs=container,
        auth_context=_auth_context(),
        request=response_request,
        channels=_RecordingChannels(),
    )
    coordinator = state.svcs.get(StreamCoordinator)
    checkpoint = ReasoningCheckpoint(memory={"turn": 2}, active={"main"}, threads={})
    checkpoint_id = await coordinator.begin_reasoning(state=checkpoint, main=[])
    await coordinator.finish_reasoning(state=checkpoint, main=[])
    patch = ReasoningPatch(memory=[{"op": "add", "path": "/tool", "value": True}])
    await coordinator.begin_reasoning(state=patch, main=[])
    await coordinator.finish_reasoning(state=patch, main=[])
    output = coordinator.current_response().output
    checkpoint_item = output[-2]
    patch_item = output[-1]

    assert isinstance(checkpoint_item, ResponseReasoningItem)
    assert isinstance(patch_item, ResponseReasoningItem)
    checkpoint_payload = open_reasoning_payload(checkpoint_item.encrypted_content, keyring=keyring)
    patch_payload = open_reasoning_payload(patch_item.encrypted_content, keyring=keyring)
    assert checkpoint_payload.previous_reasoning_id is None
    assert checkpoint_payload.previous_compaction_id is None
    assert patch_payload.previous_reasoning_id == checkpoint_id
    assert patch_payload.previous_compaction_id is None


async def test_prepare_create_echoes_inbound_compaction_anchor_into_reasoning() -> None:
    keyring = _keyring()
    compaction = CompactionPayload(id="cmp_stored", memory={"generation": 1}, threads=Threads())
    execution_request = ResponseCreateRequest(
        model="plap/test",
        input=[
            RequestCompactionItem(
                encrypted_content=seal_compaction_payload(compaction, keyring=keyring),
                id=compaction.id,
                type="compaction",
            ),
            RequestMessageItem(content="new turn", role="user", type="message"),
        ],
    )
    response_request = _request()
    prepared = PreparedRequest(
        scope_id=uuid4(),
        response_request=response_request,
        execution_request=execution_request,
        stored_input_items=[],
    )
    store = _RecordingStore(prepared)
    registry = svcs.Registry()
    registry.register_value(AuthContext, _auth_context())
    registry.register_value(CueBox, _config())
    registry.register_value(SealingKeyring, keyring)
    registry.register_value(ResponseStore, store)
    container = svcs.Container(registry)

    state = await _prepare_create(
        svcs=container,
        auth_context=_auth_context(),
        request=response_request,
        channels=_RecordingChannels(),
    )
    coordinator = state.svcs.get(StreamCoordinator)
    checkpoint = ReasoningCheckpoint(memory={"generation": 2}, active={"main"}, threads={})
    checkpoint_id = await coordinator.begin_reasoning(state=checkpoint, main=[])
    await coordinator.finish_reasoning(state=checkpoint, main=[])
    patch = ReasoningPatch(memory=[{"op": "add", "path": "/tool", "value": True}])
    await coordinator.begin_reasoning(state=patch, main=[])
    await coordinator.finish_reasoning(state=patch, main=[])
    checkpoint_item, patch_item = coordinator.current_response().output[-2:]

    assert isinstance(checkpoint_item, ResponseReasoningItem)
    assert isinstance(patch_item, ResponseReasoningItem)
    checkpoint_payload = open_reasoning_payload(checkpoint_item.encrypted_content, keyring=keyring)
    patch_payload = open_reasoning_payload(patch_item.encrypted_content, keyring=keyring)
    assert checkpoint_payload.previous_reasoning_id is None
    assert checkpoint_payload.previous_compaction_id == compaction.id
    assert patch_payload.previous_reasoning_id == checkpoint_id
    assert patch_payload.previous_compaction_id == compaction.id


async def test_response_start_return_without_terminalization_fails_response(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = EventBus()

    @bus.emit("response.start")
    async def _body(state: State) -> None:
        _ = state

    monkeypatch.setattr("plap.responses.routes.bus", bus)
    state = _state()

    await _accept_response(state)
    await _execute_response(state)

    response = state.svcs.get(StreamCoordinator).current_response()
    assert response.status == "failed"
    assert response.error is not None
    assert response.error.code == "server_error"


async def test_response_start_error_after_terminalization_preserves_response(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = EventBus()

    @bus.emit("response.start")
    async def _body(state: State) -> None:
        await state.svcs.get(StreamCoordinator).completed()

    @bus.listen("response.start")
    async def _boom(state: State, *, next) -> None:
        await next(state=state)
        raise _plap_error()

    monkeypatch.setattr("plap.responses.routes.bus", bus)
    state = _state()

    await _accept_response(state)
    await _execute_response(state)

    assert state.svcs.get(StreamCoordinator).current_response().status == "completed"


async def test_safe_progress_failure_does_not_fabricate_failed_response(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = EventBus()

    @bus.emit("response.start")
    async def _boom(state: State) -> None:
        _ = state
        raise RuntimeError("execution failed")

    async def _fail_save_progress(self: State) -> None:
        raise RuntimeError("progress persistence failed")

    monkeypatch.setattr("plap.responses.routes.bus", bus)
    monkeypatch.setattr(State, "save_progress", _fail_save_progress)
    state = _state()

    await _accept_response(state)
    with pytest.raises(RuntimeError, match="progress persistence failed"):
        await _execute_response(state)

    assert state.svcs.get(StreamCoordinator).current_response().status == "in_progress"


async def test_config_failure_raises_during_preparation() -> None:
    class FailingConfig(_Config):
        def resolve(self, request: dict[str, object] | None = None, /, **kwargs: object) -> _Config:
            _ = request, kwargs
            raise _plap_error()

    registry = svcs.Registry()
    registry.register_value(CueBox, FailingConfig(_config().to_dict(), frozen_box=True))
    registry.register_value(SealingKeyring, _keyring())
    store = _RecordingStore(_prepared())
    registry.register_value(ResponseStore, store)
    container = svcs.Container(registry)

    with pytest.raises(PlapError) as exc_info:
        await _prepare_create(
            svcs=container,
            auth_context=_auth_context(),
            request=_request(),
            channels=_RecordingChannels(),
        )

    assert exc_info.value.public is not None
    assert exc_info.value.public.code == "test_error"
    assert store.prepare_calls == 0


async def test_websocket_response_containers_close_after_preparation_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class Container:
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    class Plugins:
        def get(self, plugin_type):
            _ = plugin_type
            return object()

    class AppState:
        svcs_registry = object()

    class App:
        plugins = Plugins()
        state = AppState()

    class Socket:
        app = App()

        def __init__(self) -> None:
            self.sent: list[dict[str, object]] = []
            self.received = 0

        async def accept(self) -> None:
            return None

        async def receive_json(self) -> dict[str, object]:
            if self.received == 2:
                raise RuntimeError("socket closed")
            self.received += 1
            return {
                "type": "response.create",
                "response": {"model": "plap/test", "input": "hello"},
            }

        async def send_json(self, payload: dict[str, object]) -> None:
            self.sent.append(payload)

    async def fail_prepare(**kwargs):
        _ = kwargs
        raise RuntimeError("preparation failed")

    containers: list[Container] = []

    def build_container(registry) -> Container:
        _ = registry
        container = Container()
        containers.append(container)
        return container

    monkeypatch.setattr("plap.responses.routes.Container", build_container)
    monkeypatch.setattr("plap.responses.routes._prepare_create", fail_prepare)

    socket = Socket()
    await responses_socket.fn(socket, _auth_context())

    assert len(containers) == 2
    assert all(container.closed for container in containers)
    assert [payload["type"] for payload in socket.sent] == ["error", "error"]
