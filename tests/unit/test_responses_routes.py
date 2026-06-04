from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

import pytest

from plap.auth import AuthContext
from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.keyring import SealingKeyring
from plap.responses.contracts import ResponseCreateRequest
from plap.responses.ingest.models import Ingested, Sides
from plap.responses.routes import _prepare_create, _run_stream
from plap.responses.store import PreparedRequest
from plap.responses.streaming import StreamCoordinator
from plap.settings import Settings
from plap.tools import StaticToolCallPolicyResolver, StaticToolPolicyResolver


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


def _settings() -> Settings:
    return Settings(api_key_pepper="pepper", database_url="postgres://example", sealing_keys=["key"])


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


async def test_prepare_create_prepares_and_stops_before_created() -> None:
    request = _request()
    store = _RecordingStore(_prepared(request))

    prepared, ingested, coordinator = await _prepare_create(
        auth_context=_auth_context(),
        request=request,
        response_store=store,
        sealing_keyring=_keyring(),
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
    assert ingested.last_side == "main"
    assert ingested.last_reasoning_id is None
    assert ingested.current_compaction_id is None
    assert coordinator.current_response().model == request.model


async def test_run_stream_swallows_runtime_plap_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _boom(**kwargs) -> None:
        _ = kwargs
        raise _plap_error()

    monkeypatch.setattr("plap.responses.routes.run_response", _boom)

    await _run_stream(
        prepared=_prepared(),
        ingested=Ingested(machine={}, sides=Sides(), last_side=None, last_reasoning_id=None, current_compaction_id=None),
        coordinator=_coordinator(),
        sealing_keyring=_keyring(),
        settings=_settings(),
        chat_completion_client=object(),
        tool_policy_resolver=StaticToolPolicyResolver(),
        tool_call_policy_resolver=StaticToolCallPolicyResolver(),
        mcp_tool_providers=(),
    )
