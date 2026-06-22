from __future__ import annotations

from typing import Any

from litestar import Request
from litestar.connection import WebSocket
from litestar.exceptions import NotAuthorizedException
from litestar.types import ASGIApp, Receive, Scope, Send

from plap.auth.api_keys import APIKeyManager, AuthContext, AuthError


def provide_request_auth_context(request: Request[Any, Any, Any]) -> AuthContext:
    auth_context = request.state.auth_context
    if not isinstance(auth_context, AuthContext):
        raise TypeError("request auth context is missing")
    return auth_context


def provide_socket_auth_context(socket: WebSocket) -> AuthContext:
    auth_context = socket.state.auth_context
    if not isinstance(auth_context, AuthContext):
        raise TypeError("socket auth context is missing")
    return auth_context


def auth_middleware(app: ASGIApp) -> ASGIApp:
    async def middleware(scope: Scope, receive: Receive, send: Send) -> None:
        scope_type = scope.get("type")
        if scope_type not in {"http", "websocket"}:
            await app(scope, receive, send)
            return

        connection = Request[Any, Any, Any](scope, receive, send) if scope_type == "http" else WebSocket(scope, receive, send)
        api_key_manager: APIKeyManager = connection.app.state.api_key_manager
        try:
            async with connection.app.state.database.session() as session:
                auth_context = await api_key_manager.authenticate_bearer_token(
                    session,
                    connection.headers.get("authorization"),
                )
        except AuthError as exc:
            raise NotAuthorizedException(
                detail=str(exc),
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

        connection.state.auth_context = auth_context
        connection.scope["user"] = str(auth_context.user_id)
        await app(scope, receive, send)

    return middleware
