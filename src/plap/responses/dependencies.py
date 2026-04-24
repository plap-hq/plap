from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from litestar import Request
from litestar.connection import WebSocket
from litestar.di import Provide
from litestar.exceptions import NotAuthorizedException
from sqlalchemy.ext.asyncio import AsyncSession

from plap.auth import APIKeyManager, AuthContext, AuthError
from plap.settings import Settings


def provide_settings(request: Request[Any, Any, Any]) -> Settings:
    return request.app.state.settings


def provide_request_api_key_manager(request: Request[Any, Any, Any]) -> APIKeyManager:
    return request.app.state.api_key_manager


async def provide_request_db_session(
    request: Request[Any, Any, Any],
) -> AsyncIterator[AsyncSession]:
    async with request.app.state.session_maker() as session:
        yield session


async def provide_request_auth_context(
    request: Request[Any, Any, Any],
    db_session: AsyncSession,
    api_key_manager: APIKeyManager,
) -> AuthContext:
    try:
        auth_context = await api_key_manager.authenticate_bearer_token(
            db_session,
            request.headers.get("authorization"),
        )
    except AuthError as exc:
        raise NotAuthorizedException(
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    request.state.auth_context = auth_context
    request.scope["user"] = str(auth_context.user_id)
    return auth_context


def provide_socket_api_key_manager(socket: WebSocket) -> APIKeyManager:
    return socket.app.state.api_key_manager


async def provide_socket_db_session(socket: WebSocket) -> AsyncIterator[AsyncSession]:
    async with socket.app.state.session_maker() as session:
        yield session


async def provide_socket_auth_context(
    socket: WebSocket,
    db_session: AsyncSession,
    api_key_manager: APIKeyManager,
) -> AuthContext:
    try:
        auth_context = await api_key_manager.authenticate_bearer_token(
            db_session,
            socket.headers.get("authorization"),
        )
    except AuthError as exc:
        raise NotAuthorizedException(
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    socket.state.auth_context = auth_context
    socket.scope["user"] = str(auth_context.user_id)
    return auth_context


HTTP_ROUTE_DEPENDENCIES = {
    "api_key_manager": Provide(
        provide_request_api_key_manager,
        use_cache=True,
        sync_to_thread=False,
    ),
    "auth_context": Provide(provide_request_auth_context),
    "db_session": Provide(provide_request_db_session),
    "settings": Provide(provide_settings, use_cache=True, sync_to_thread=False),
}


WEBSOCKET_ROUTE_DEPENDENCIES = {
    "api_key_manager": Provide(
        provide_socket_api_key_manager,
        use_cache=True,
        sync_to_thread=False,
    ),
    "auth_context": Provide(provide_socket_auth_context),
    "db_session": Provide(provide_socket_db_session),
}
