from __future__ import annotations

from typing import Any

from litestar import Litestar, Request, Response
from litestar.datastructures import State
from litestar.exceptions import (
    HTTPException,
    NotAuthorizedException,
    ValidationException,
)

from plap.auth import APIKeyManager
from plap.persistence import create_database_engine, create_session_maker
from plap.responses import RESPONSE_ROUTE_HANDLERS
from plap.settings import Settings, get_settings


def _error_body(
    *,
    message: str,
    error_type: str,
    code: str | None = None,
    param: str | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "param": param,
            "type": error_type,
        }
    }


def _error_response(
    *,
    message: str,
    status_code: int,
    error_type: str,
    code: str | None = None,
    param: str | None = None,
    headers: dict[str, str] | None = None,
) -> Response[dict[str, Any]]:
    return Response(
        _error_body(
            message=message,
            error_type=error_type,
            code=code,
            param=param,
        ),
        headers=headers,
        media_type="application/json",
        status_code=status_code,
    )


def handle_auth_exception(
    _: Request[Any, Any, Any],
    exc: NotAuthorizedException,
) -> Response[dict[str, Any]]:
    return _error_response(
        message=exc.detail or "Not authorized",
        status_code=401,
        error_type="authentication_error",
        code="invalid_api_key",
        headers={"WWW-Authenticate": "Bearer"},
    )


def handle_validation_exception(
    _: Request[Any, Any, Any],
    exc: ValidationException,
) -> Response[dict[str, Any]]:
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail or exc)
    return _error_response(
        message=detail,
        status_code=400,
        error_type="invalid_request_error",
        code="invalid_request",
    )


def handle_http_exception(
    _: Request[Any, Any, Any],
    exc: HTTPException,
) -> Response[dict[str, Any]]:
    return _error_response(
        message=exc.detail or "Request failed",
        status_code=exc.status_code,
        error_type="invalid_request_error",
        code="request_error",
    )


async def _shutdown_database(app: Litestar) -> None:
    if app.state.owns_engine:
        await app.state.db_engine.dispose()


def create_app(settings: Settings | None = None) -> Litestar:
    resolved_settings = settings or get_settings()
    db_engine = create_database_engine(resolved_settings.database_url)
    session_maker = create_session_maker(db_engine)
    state = State(
        {
            "api_key_manager": APIKeyManager(pepper=resolved_settings.api_key_pepper),
            "db_engine": db_engine,
            "owns_engine": True,
            "session_maker": session_maker,
            "settings": resolved_settings,
        }
    )

    return Litestar(
        route_handlers=RESPONSE_ROUTE_HANDLERS,
        exception_handlers={
            HTTPException: handle_http_exception,
            NotAuthorizedException: handle_auth_exception,
            ValidationException: handle_validation_exception,
        },
        on_shutdown=[_shutdown_database],
        state=state,
    )
