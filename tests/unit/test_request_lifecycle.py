from __future__ import annotations

from typing import cast

import pytest
import svcs
from litestar.types import Message, ReceiveMessage, Scope

import plap.telemetry as telemetry
from plap.app import svcs_lifecycle_middleware


class _RecordingLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, object]]] = []

    def info(self, event: str, **context: object) -> None:
        self.records.append((event, context))


class _RecordingContainer(svcs.Container):
    def __init__(self) -> None:
        super().__init__(svcs.Registry())
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True
        await super().aclose()


def _http_scope() -> Scope:
    return cast(
        Scope,
        {
            "method": "POST",
            "path": "/v1/responses",
            "state": {},
            "type": "http",
        },
    )


async def test_request_context_logs_disconnect_without_sse_terminus(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = _RecordingLogger()
    monkeypatch.setattr(telemetry, "_ACCESS_LOGGER", logger)

    async def app(scope, receive, send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"partial", "more_body": True})
        assert (await receive())["type"] == "http.disconnect"

    scope = _http_scope()

    async def receive() -> ReceiveMessage:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        telemetry.emit_access_log(message, scope)

    await telemetry.request_context_middleware(app)(scope, receive, send)

    assert len(logger.records) == 1
    assert logger.records[0][0] == "http.request.completed"
    assert logger.records[0][1]["outcome"] == "disconnected"
    assert logger.records[0][1]["status_code"] == 200


async def test_request_context_logs_post_start_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = _RecordingLogger()
    monkeypatch.setattr(telemetry, "_ACCESS_LOGGER", logger)

    async def app(scope, receive, send) -> None:
        _ = scope, receive
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"partial", "more_body": True})
        raise RuntimeError("stream failed")

    scope = _http_scope()

    async def receive() -> ReceiveMessage:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        telemetry.emit_access_log(message, scope)

    with pytest.raises(RuntimeError, match="stream failed"):
        await telemetry.request_context_middleware(app)(scope, receive, send)

    assert logger.records[0][0] == "http.request.completed"
    assert logger.records[0][1]["outcome"] == "aborted"
    assert logger.records[0][1]["status_code"] == 200


async def test_svcs_lifecycle_closes_container_after_post_start_failure() -> None:
    container = _RecordingContainer()

    async def app(scope, receive, send) -> None:
        _ = receive
        scope["state"]["svcs_container"] = container
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"partial", "more_body": True})
        raise RuntimeError("stream failed")

    scope = _http_scope()

    async def receive() -> ReceiveMessage:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        _ = message

    with pytest.raises(RuntimeError, match="stream failed"):
        await svcs_lifecycle_middleware(app)(scope, receive, send)

    assert container.closed
    assert "svcs_container" not in scope["state"]
