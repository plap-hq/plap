from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from litestar import Request
from litestar.connection import WebSocket
from sqlalchemy.ext.asyncio import AsyncSession


async def provide_request_db_session(
    request: Request[Any, Any, Any],
) -> AsyncIterator[AsyncSession]:
    async with request.app.state.session_maker() as session:
        yield session


async def provide_socket_db_session(socket: WebSocket) -> AsyncIterator[AsyncSession]:
    async with socket.app.state.session_maker() as session:
        yield session
