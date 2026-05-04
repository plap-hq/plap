from __future__ import annotations

from typing import Any

from litestar import Request
from litestar.connection import WebSocket
from litestar.exceptions import NotAuthorizedException

from plap.auth.api_keys import APIKeyManager, AuthContext, AuthError


def provide_request_api_key_manager(request: Request[Any, Any, Any]) -> APIKeyManager:
    return request.app.state.api_key_manager


async def provide_request_auth_context(
    request: Request[Any, Any, Any],
    api_key_manager: APIKeyManager,
) -> AuthContext:
    try:
        async with request.app.state.database.session() as session:
            auth_context = await api_key_manager.authenticate_bearer_token(
                session,
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


async def provide_socket_auth_context(
    socket: WebSocket,
    api_key_manager: APIKeyManager,
) -> AuthContext:
    try:
        async with socket.app.state.database.session() as session:
            auth_context = await api_key_manager.authenticate_bearer_token(
                session,
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
