from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import AsyncIterator
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import svcs
import uvicorn

from plap.app import create_app
from plap.llms.completions.budget import BudgetedChatCompletionClient, CompletionBudget
from plap.llms.completions.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatMessage,
    ChatToolCall,
    IChatCompletionClient,
)
from tests.pytest_plugins.database import TestConfig


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(slots=True)
class LiveServer:
    base_url: str
    websocket_base_url: str


@contextmanager
def _with_test_env(test_config: TestConfig):
    updates = {
        "PLAP_API_KEY_PEPPER": test_config.api_key_pepper,
        "PLAP_DATABASE_URL": test_config.database_url,
        "PLAP_SEALING_KEYS": ",".join(test_config.sealing_keys),
    }
    saved = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _build_test_app(
    test_config: TestConfig,
    *,
    chat_completion_client: IChatCompletionClient | None = None,
):
    with _with_test_env(test_config):
        app = create_app()
    client = chat_completion_client or _StaticChatCompletionClient()

    def client_factory(svcs_container: svcs.Container) -> IChatCompletionClient:
        return BudgetedChatCompletionClient(client, svcs_container.get(CompletionBudget))

    app.state.svcs_registry.register_factory(
        IChatCompletionClient,
        client_factory,
        on_registry_close=client.aclose,
    )
    return app


@pytest.fixture
def test_app_factory(test_config: TestConfig):
    def build(
        *,
        chat_completion_client: IChatCompletionClient | None = None,
    ):
        return _build_test_app(
            test_config,
            chat_completion_client=chat_completion_client,
        )

    return build


@pytest.fixture
def test_app(test_app_factory):
    return test_app_factory()


@pytest.fixture
def live_server_factory(test_config: TestConfig):
    @contextmanager
    def build(
        *,
        chat_completion_client: IChatCompletionClient | None = None,
    ):
        port = _find_free_port()
        app = _build_test_app(
            test_config,
            chat_completion_client=chat_completion_client,
        )

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

        try:
            yield LiveServer(
                base_url=f"http://127.0.0.1:{port}",
                websocket_base_url=f"ws://127.0.0.1:{port}",
            )
        finally:
            server.should_exit = True
            thread.join(timeout=10)

    return build


@pytest.fixture
def live_server(live_server_factory) -> LiveServer:
    with live_server_factory() as server:
        yield server


class _StaticChatCompletionClient(IChatCompletionClient):
    async def complete(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResult:
        if len(request.tools) == 1 and request.tools[0].function.name == "compact":
            return ChatCompletionResult(
                id="chatcmpl_test",
                model=request.model,
                created_at=None,
                message=ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ChatToolCall(
                            id="compact_call_1",
                            name="compact",
                            arguments='{"ranges":[{"start":"[~0]","end":"[~0]","summary":"brief"}]}',
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        return ChatCompletionResult(
            id="chatcmpl_test",
            model=request.model,
            created_at=None,
            message=ChatMessage(role="assistant", content="test response"),
            finish_reason="stop",
        )

    async def stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionDelta]:
        yield ChatCompletionDelta(
            id="chatcmpl_test",
            model=request.model,
            created_at=None,
            choice_index=0,
            content_delta="test response",
        )
        yield ChatCompletionDelta(
            id="chatcmpl_test",
            model=request.model,
            created_at=None,
            choice_index=0,
            finish_reason="stop",
        )

    async def aclose(self) -> None:
        return None
