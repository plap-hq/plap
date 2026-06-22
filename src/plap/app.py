from __future__ import annotations

import importlib
import sys
from functools import partial
from importlib.metadata import EntryPoint, entry_points
from typing import Any

import anyio
import structlog
import svcs
from litestar import Litestar, Request, Response
from litestar.channels import ChannelsPlugin
from litestar.channels.backends.memory import MemoryChannelsBackend
from litestar.datastructures import State
from litestar.di import Provide
from litestar.exceptions import HTTPException, NotAuthorizedException, ValidationException

from plap.auth import APIKeyManager
from plap.auth.dependencies import auth_middleware, provide_request_auth_context
from plap.bus import bus
from plap.config import CueBox, load
from plap.errors import PlapError, PublicError
from plap.keyring import SealingKeyring
from plap.logging import configure_logging
from plap.persistence import Database
from plap.responses.store import ResponseStore
from plap.telemetry import build_telemetry, emit_access_log, request_context_middleware, shutdown_telemetry

logger = structlog.stdlib.get_logger(__name__)


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
        _error_body(message=message, error_type=error_type, code=code, param=param),
        headers=headers,
        media_type="application/json",
        status_code=status_code,
    )


def handle_plap_error(request: Request[Any, Any, Any], exc: PlapError) -> Response[dict[str, Any]]:
    public = exc.public or PublicError(
        status_code=500,
        type="server_error",
        code="server_error",
        message="Response generation failed.",
    )
    exc.log(
        logger,
        failure_code=public.code,
        failure_type=public.type,
        method=request.method,
        path=request.url.path,
        status_code=public.status_code,
    )
    return _error_response(
        message=public.message,
        status_code=public.status_code,
        error_type=public.type,
        code=public.code,
        param=public.param,
        headers=public.headers,
    )


def handle_validation_exception(
    request: Request[Any, Any, Any],
    exc: ValidationException,
) -> Response[dict[str, Any]]:
    logger.warning(
        "response.validation_failed",
        exc_info=exc,
        method=request.method,
        path=request.url.path,
        status_code=400,
    )
    return _error_response(
        message="Invalid request.",
        status_code=400,
        error_type="invalid_request_error",
        code="invalid_request",
    )


def handle_auth_exception(request: Request[Any, Any, Any], exc: NotAuthorizedException) -> Response[dict[str, Any]]:
    logger.warning(
        "response.auth_failed",
        exc_info=exc,
        method=request.method,
        path=request.url.path,
        status_code=401,
    )
    return _error_response(
        message="Not authorized.",
        status_code=401,
        error_type="authentication_error",
        code="invalid_api_key",
        headers={"WWW-Authenticate": "Bearer"},
    )


def handle_http_exception(request: Request[Any, Any, Any], exc: HTTPException) -> Response[dict[str, Any]]:
    logger.warning(
        "response.http_failed",
        exc_info=exc,
        method=request.method,
        path=request.url.path,
        status_code=exc.status_code,
    )
    return _error_response(
        message="Request failed.",
        status_code=exc.status_code,
        error_type="invalid_request_error",
        code="request_error",
    )


def handle_unexpected_exception(request: Request[Any, Any, Any], exc: Exception) -> Response[dict[str, Any]]:
    logger.error(
        "response.unhandled_failed",
        exc_info=exc,
        method=request.method,
        path=request.url.path,
        status_code=500,
    )
    return _error_response(
        message="Response generation failed.",
        status_code=500,
        error_type="server_error",
        code="server_error",
    )


def _plugins() -> dict[str, EntryPoint]:
    discovered: dict[str, EntryPoint] = {}
    for entrypoint in entry_points().select(group="plap.plugin"):
        if entrypoint.name in discovered:
            raise RuntimeError(f"duplicate plugin entrypoint: {entrypoint.name!r}")
        discovered[entrypoint.name] = entrypoint
    return discovered


def _import_plugin(entrypoint: EntryPoint) -> object:
    module_name = entrypoint.module
    module = sys.modules.get(module_name)
    if module is None:
        return importlib.import_module(module_name)
    return importlib.reload(module)


@bus.emit("config.collect")
async def _collect_config(paths: tuple[str, ...]) -> tuple[str, ...]:
    return paths


@bus.emit("routes.collect")
async def _collect_routes(routes: tuple[object, ...], loaded: CueBox) -> tuple[object, ...]:
    _ = loaded
    return routes


@bus.emit("svcs.collect")
async def _collect_svcs(registry: svcs.Registry, loaded: CueBox) -> None:
    _ = registry, loaded


@bus.emit("shutdown.collect")
async def _collect_shutdown(hooks: tuple[object, ...], loaded: CueBox) -> tuple[object, ...]:
    _ = loaded
    return hooks


def _sealing_keys(config: CueBox) -> list[str]:
    raw = config.sealing_keys
    if isinstance(raw, list):
        return [str(value) for value in raw if str(value).strip()]
    if isinstance(raw, str):
        return [part.strip() for part in raw.split(",") if part.strip()]
    raise TypeError(f"config.sealing_keys must be a string or list, got {type(raw).__name__}")


def provide_svcs(request: Request[Any, Any, Any]) -> svcs.Container:
    registry: svcs.Registry = request.app.state.svcs_registry
    container = svcs.Container(registry)
    request.state.svcs_container = container
    return container


async def cleanup_svcs(message: Any, scope: dict[str, Any]) -> None:
    scope_type = scope.get("type")
    if scope_type == "http":
        if message.get("type") != "http.response.body" or bool(message.get("more_body", False)):
            return
    elif scope_type == "websocket":
        if message.get("type") != "websocket.close":
            return
    else:
        return

    state = scope.get("state")
    if not isinstance(state, dict):
        return
    container = state.get("svcs_container")
    if isinstance(container, svcs.Container):
        await container.aclose()


async def _shutdown_database(app: Litestar) -> None:
    await app.state.database.dispose_all()


async def _shutdown_svcs_registry(app: Litestar) -> None:
    await app.state.svcs_registry.aclose()


def create_app() -> Litestar:
    discovered = _plugins()
    if "core" not in discovered:
        raise RuntimeError("missing required plugin entrypoint 'core'")

    # Clear any listeners left from prior imports so the bus reflects only the
    # bootstrap plugin set.
    bus.reset()
    _import_plugin(discovered["core"])

    core_paths = anyio.run(partial(_collect_config, paths=()))
    loaded = load(*core_paths)
    if "plap" not in loaded:
        raise RuntimeError("config load did not produce package 'plap'")
    config = loaded.plap.config

    plugin_names = list(config.plugins)
    if "core" not in plugin_names:
        raise RuntimeError("config.plugins must include 'core'")

    # Boostrap only imported core so we could read the plugin allowlist. Rebuild
    # the bus from the canonical config order for the final config load.
    bus.reset()
    for name in plugin_names:
        entrypoint = discovered.get(name)
        if entrypoint is None:
            raise RuntimeError(f"config requested unknown plugin: {name!r}")
        _import_plugin(entrypoint)

    final_paths = anyio.run(partial(_collect_config, paths=()))
    loaded = load(*final_paths)
    if "plap" not in loaded:
        raise RuntimeError("config load did not produce package 'plap'")
    config = loaded.plap.config

    telemetry = build_telemetry()
    configure_logging(config, handlers=(telemetry.log_handler,))

    logger.info(
        "app.startup",
        foreign_log_level=config.foreign_log_level,
        log_level=config.log_level,
        plugins=plugin_names,
    )

    database = Database(config.database_url)
    keyring = SealingKeyring.from_encoded(_sealing_keys(config))
    api_key_manager = APIKeyManager(pepper=config.api_key_pepper)

    registry = svcs.Registry()
    registry.register_value(CueBox, loaded)
    registry.register_value(Database, database)
    registry.register_value(SealingKeyring, keyring)
    registry.register_factory(ResponseStore, lambda svcs_container: ResponseStore(svcs_container.get(Database)))
    anyio.run(partial(_collect_svcs, registry=registry, loaded=loaded))

    routes = anyio.run(partial(_collect_routes, routes=(), loaded=loaded))
    shutdown_hooks = anyio.run(partial(_collect_shutdown, hooks=(), loaded=loaded))

    state = State(
        {
            "api_key_manager": api_key_manager,
            "config": loaded,
            "database": database,
            "sealing_keyring": keyring,
            "svcs_registry": registry,
        }
    )

    return Litestar(
        route_handlers=routes,
        before_send=[emit_access_log, cleanup_svcs],
        logging_config=None,
        middleware=[request_context_middleware, auth_middleware],
        plugins=[
            telemetry.plugin,
            ChannelsPlugin(
                backend=MemoryChannelsBackend(),
                arbitrary_channels_allowed=True,
                create_ws_route_handlers=False,
            ),
        ],
        dependencies={
            "svcs": Provide(provide_svcs, use_cache=True, sync_to_thread=False),
            "auth_context": Provide(provide_request_auth_context, use_cache=True, sync_to_thread=False),
        },
        exception_handlers={
            HTTPException: handle_http_exception,
            NotAuthorizedException: handle_auth_exception,
            PlapError: handle_plap_error,
            ValidationException: handle_validation_exception,
            Exception: handle_unexpected_exception,
        },
        on_shutdown=[
            *shutdown_hooks,
            _shutdown_database,
            shutdown_telemetry,
            _shutdown_svcs_registry,
        ],
        state=state,
    )
