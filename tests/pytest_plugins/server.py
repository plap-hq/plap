from __future__ import annotations

import socket
import threading
import time
from collections.abc import AsyncIterator
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import uvicorn

from plap.app import create_app
from plap.llms.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatMessage,
    ChatToolCall,
    IChatCompletionClient,
)
from plap.settings import Settings


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass(slots=True)
class LiveServer:
    base_url: str
    websocket_base_url: str


def _build_test_app(
    settings: Settings,
    *,
    chat_completion_client: IChatCompletionClient | None = None,
    tool_classifier: object | None = None,
):
    app = create_app(settings)
    app.state.chat_completion_client = chat_completion_client or _StaticChatCompletionClient()
    if tool_classifier is not None:
        app.state.tool_classifier = tool_classifier
    return app


@pytest.fixture
def test_app_factory(test_settings: Settings):
    def build(
        *,
        chat_completion_client: IChatCompletionClient | None = None,
        tool_classifier: object | None = None,
    ):
        return _build_test_app(
            test_settings,
            chat_completion_client=chat_completion_client,
            tool_classifier=tool_classifier,
        )

    return build


@pytest.fixture
def test_app(test_app_factory):
    return test_app_factory()


@pytest.fixture
def live_server_factory(test_settings: Settings):
    @contextmanager
    def build(
        *,
        chat_completion_client: IChatCompletionClient | None = None,
        tool_classifier: object | None = None,
    ):
        port = _find_free_port()

        def app_factory():
            return _build_test_app(
                test_settings,
                chat_completion_client=chat_completion_client,
                tool_classifier=tool_classifier,
            )

        config = uvicorn.Config(app_factory, host="127.0.0.1", port=port, log_level="warning", factory=True)
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
                            arguments=(
                                '{"action":"apply","ranges":[{"start":"[~0]","end":"[~0]","summary":"brief",'
                                '"summary_fidelity":5}]}'
                            ),
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
