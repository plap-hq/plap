from __future__ import annotations

from typing import Any

from litestar import Request
from litestar.connection import WebSocket

from plap.llms.completions.chat import IChatCompletionClient


def provide_request_chat_completion_client(
    request: Request[Any, Any, Any],
) -> IChatCompletionClient:
    return request.app.state.chat_completion_client


def provide_socket_chat_completion_client(
    socket: WebSocket,
) -> IChatCompletionClient:
    return socket.app.state.chat_completion_client
