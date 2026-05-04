from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import structlog
from cachetools import LRUCache
from litestar import Litestar, Request, Response
from litestar.datastructures import State
from litestar.exceptions import HTTPException, NotAuthorizedException, ValidationException

from plap.auth import APIKeyManager
from plap.keyring import SealingKeyring
from plap.llms.chat import IChatCompletionClient
from plap.llms.crof import CrofChatCompletionClient
from plap.llms.fireworks import FireworksChatCompletionClient
from plap.llms.lightning import LightningChatCompletionClient
from plap.llms.novita import NovitaChatCompletionClient
from plap.llms.router import (
    ModelRoute,
    RoutingChatCompletionClient,
    UnavailableChatCompletionClient,
)
from plap.persistence import Database
from plap.responses.errors import ResponseError
from plap.responses.reasoning import LLMReasoningSummarizer
from plap.responses.routes import RESPONSE_ROUTE_HANDLERS
from plap.responses.tools import (
    TOOL_CALL_EFFECT_CLASSIFIER_MODEL,
    TOOL_EFFECT_CLASSIFIER_MODEL,
    IToolCallClassifier,
    IToolClassifier,
    LLMToolCallClassifier,
    LLMToolClassifier,
)
from plap.responses.tools.mcp import (
    IMCPToolProvider,
    MCPToolProvider,
)
from plap.settings import MCPServerConfig, Settings, get_settings

logger = structlog.get_logger(__name__)


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


def handle_response_error(
    request: Request[Any, Any, Any],
    exc: ResponseError,
) -> Response[dict[str, Any]]:
    exc.log(
        logger,
        failure_code=exc.public.code,
        failure_type=exc.public.type,
        method=request.method,
        path=request.url.path,
        status_code=exc.public.status_code,
    )
    return _error_response(
        message=exc.public.message,
        status_code=exc.public.status_code,
        error_type=exc.public.type,
        code=exc.public.code,
        param=exc.public.param,
        headers=exc.public.headers,
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


def handle_auth_exception(
    request: Request[Any, Any, Any],
    exc: NotAuthorizedException,
) -> Response[dict[str, Any]]:
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


def handle_http_exception(
    request: Request[Any, Any, Any],
    exc: HTTPException,
) -> Response[dict[str, Any]]:
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


def handle_unexpected_exception(
    request: Request[Any, Any, Any],
    exc: Exception,
) -> Response[dict[str, Any]]:
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


async def _shutdown_database(app: Litestar) -> None:
    await app.state.database.dispose_all()


def _create_chat_completion_client(settings: Settings) -> IChatCompletionClient:
    routes = list(_chat_completion_routes(settings))
    if not routes:
        return UnavailableChatCompletionClient()
    return RoutingChatCompletionClient(routes)


def _chat_completion_routes(settings: Settings) -> Iterable[ModelRoute]:
    if settings.llm_lightning_api_key:
        client = LightningChatCompletionClient(api_key=settings.llm_lightning_api_key)
        yield ModelRoute(prefix="lightning/", client=client)

    if settings.llm_novita_api_key:
        client = NovitaChatCompletionClient(api_key=settings.llm_novita_api_key)
        yield ModelRoute(prefix="novita/", client=client)

    if settings.llm_fireworks_api_key:
        client = FireworksChatCompletionClient(api_key=settings.llm_fireworks_api_key)
        yield ModelRoute(prefix="fireworks/", client=client)

    if settings.llm_crof_api_key:
        client = CrofChatCompletionClient(api_key=settings.llm_crof_api_key)
        yield ModelRoute(prefix="crof/", client=client)


def _create_tool_classifier(
    settings: Settings,
    chat_completion_client: IChatCompletionClient,
) -> IToolClassifier:
    if not _has_configured_chat_completion_route(
        settings,
        TOOL_EFFECT_CLASSIFIER_MODEL,
    ):
        raise ValueError(f"tool effect classifier model does not match any configured LLM route: {TOOL_EFFECT_CLASSIFIER_MODEL!r}")
    return LLMToolClassifier(
        client=chat_completion_client,
        classifier_model=TOOL_EFFECT_CLASSIFIER_MODEL,
        max_concurrency=settings.tool_classifier_max_concurrency,
    )


def _create_tool_call_classifier(
    settings: Settings,
    chat_completion_client: IChatCompletionClient,
) -> IToolCallClassifier:
    if not _has_configured_chat_completion_route(
        settings,
        TOOL_CALL_EFFECT_CLASSIFIER_MODEL,
    ):
        raise ValueError(f"tool call classifier model does not match any configured LLM route: {TOOL_CALL_EFFECT_CLASSIFIER_MODEL!r}")
    return LLMToolCallClassifier(
        client=chat_completion_client,
        classifier_model=TOOL_CALL_EFFECT_CLASSIFIER_MODEL,
        max_concurrency=settings.tool_classifier_max_concurrency,
    )


def _create_mcp_tool_providers(settings: Settings) -> tuple[IMCPToolProvider, ...]:
    return tuple(_create_mcp_tool_provider(server) for server in settings.mcp_servers)


def _create_mcp_tool_provider(server: MCPServerConfig) -> IMCPToolProvider:
    transport = server.url if server.url is not None else server.config
    if transport is None:
        raise ValueError("mcp server config requires a transport")
    return MCPToolProvider(transport, tool_names=server.tool_names)


def _validate_runtime_model_profiles(settings: Settings) -> None:
    for name, profile in settings.runtime_model_profiles.items():
        for model in profile.all_models():
            if not _has_configured_chat_completion_route(settings, model):
                raise ValueError(f"runtime model profile references an unconfigured LLM route: {name!r} -> {model!r}")


def _has_configured_chat_completion_route(settings: Settings, model: str) -> bool:
    return any(model.startswith(prefix) for prefix in _configured_chat_completion_prefixes(settings))


def _configured_chat_completion_prefixes(settings: Settings) -> Iterable[str]:
    if settings.llm_lightning_api_key:
        yield "lightning/"
    if settings.llm_novita_api_key:
        yield "novita/"
    if settings.llm_fireworks_api_key:
        yield "fireworks/"
    if settings.llm_crof_api_key:
        yield "crof/"


def create_app(settings: Settings | None = None) -> Litestar:
    resolved_settings = settings or get_settings()
    chat_completion_client = _create_chat_completion_client(resolved_settings)
    reasoning_summarizer = LLMReasoningSummarizer(chat_completion_client)
    tool_classifier = _create_tool_classifier(
        resolved_settings,
        chat_completion_client,
    )
    tool_call_classifier = _create_tool_call_classifier(
        resolved_settings,
        chat_completion_client,
    )
    mcp_tool_providers = _create_mcp_tool_providers(resolved_settings)
    _validate_runtime_model_profiles(resolved_settings)
    state = State(
        {
            "api_key_manager": APIKeyManager(pepper=resolved_settings.api_key_pepper),
            "chat_completion_client": chat_completion_client,
            "database": Database(resolved_settings.database_url),
            "runtime_model_profiles": resolved_settings.runtime_model_profiles,
            "reasoning_summarizer": reasoning_summarizer,
            "sealing_keyring": SealingKeyring.from_encoded(resolved_settings.sealing_keys),
            "settings": resolved_settings,
            "tool_call_classifier": tool_call_classifier,
            "tool_call_policy_l1_cache": LRUCache(maxsize=resolved_settings.tool_call_policy_l1_maxsize),
            "tool_classifier": tool_classifier,
            "tool_policy_l1_cache": LRUCache(maxsize=resolved_settings.tool_policy_l1_maxsize),
            "mcp_tool_providers": mcp_tool_providers,
        }
    )

    return Litestar(
        route_handlers=RESPONSE_ROUTE_HANDLERS,
        exception_handlers={
            HTTPException: handle_http_exception,
            NotAuthorizedException: handle_auth_exception,
            ResponseError: handle_response_error,
            ValidationException: handle_validation_exception,
            Exception: handle_unexpected_exception,
        },
        on_shutdown=[_shutdown_database],
        state=state,
    )
