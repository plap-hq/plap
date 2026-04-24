from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass

import pytest
import uvicorn

from plap.app import create_app
from plap.settings import Settings


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(slots=True)
class LiveServer:
    base_url: str
    websocket_base_url: str


@pytest.fixture
def test_app(test_settings: Settings):
    return create_app(test_settings)


@pytest.fixture
def live_server(test_settings: Settings) -> LiveServer:
    port = _find_free_port()
    app = create_app(test_settings)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 10
    while time.time() < deadline:
        if server.started:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("Uvicorn test server did not start")

    yield LiveServer(
        base_url=f"http://127.0.0.1:{port}",
        websocket_base_url=f"ws://127.0.0.1:{port}",
    )

    server.should_exit = True
    thread.join(timeout=10)
