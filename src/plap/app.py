from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from cachetools import LRUCache
from litestar import Litestar, Request, Response
from litestar.datastructures import State
from litestar.exceptions import (
    HTTPException,
    NotAuthorizedException,
    ValidationException,
)

from plap.auth import APIKeyManager
from plap.keyring import SealingKeyring
from plap.llms.chat import IChatCompletionClient
from plap.llms.fireworks import FireworksChatCompletionClient
from plap.llms.lightning import LightningChatCompletionClient
from plap.llms.novita import NovitaChatCompletionClient
from plap.llms.router import (
    ModelRoute,
    RoutingChatCompletionClient,
    UnavailableChatCompletionClient,
)
from plap.persistence import create_database_engine, create_session_maker
from plap.responses import RESPONSE_ROUTE_HANDLERS
from plap.responses.tools import (
    IToolCallClassifier,
    IToolClassifier,
    LLMToolCallClassifier,
    LLMToolClassifier,
)
from plap.responses.tools.web_search import (
    IWebSearchToolProvider,
    MCPWebSearchToolProvider,
)
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


def _create_chat_completion_client(settings: Settings) -> IChatCompletionClient:
    routes = list(_chat_completion_routes(settings))
    if not routes:
        return UnavailableChatCompletionClient()
    return RoutingChatCompletionClient(routes)


def _chat_completion_routes(settings: Settings) -> Iterable[ModelRoute]:
    if settings.llm_lightning_api_key:
        client = LightningChatCompletionClient(api_key=settings.llm_lightning_api_key)
        for prefix in settings.llm_lightning_model_prefixes:
            yield ModelRoute(prefix=prefix, client=client)

    if settings.llm_novita_api_key:
        client = NovitaChatCompletionClient(api_key=settings.llm_novita_api_key)
        for prefix in settings.llm_novita_model_prefixes:
            yield ModelRoute(prefix=prefix, client=client)

    if settings.llm_fireworks_api_key:
        client = FireworksChatCompletionClient(api_key=settings.llm_fireworks_api_key)
        for prefix in settings.llm_fireworks_model_prefixes:
            yield ModelRoute(prefix=prefix, client=client)


def _create_tool_classifier(
    settings: Settings,
    chat_completion_client: IChatCompletionClient,
) -> IToolClassifier | None:
    if settings.tool_classifier_model is None:
        return None
    if not _has_configured_chat_completion_route(
        settings, settings.tool_classifier_model
    ):
        raise ValueError(
            "tool_classifier_model does not match any configured LLM route: "
            f"{settings.tool_classifier_model!r}"
        )
    return LLMToolClassifier(
        client=chat_completion_client,
        classifier_model=settings.tool_classifier_model,
        max_concurrency=settings.tool_classifier_max_concurrency,
    )


def _create_tool_call_classifier(
    settings: Settings,
    chat_completion_client: IChatCompletionClient,
) -> IToolCallClassifier | None:
    classifier_model = (
        settings.tool_call_classifier_model or settings.tool_classifier_model
    )
    if classifier_model is None:
        return None
    if not _has_configured_chat_completion_route(settings, classifier_model):
        raise ValueError(
            "tool_call_classifier_model does not match any configured LLM route: "
            f"{classifier_model!r}"
        )
    return LLMToolCallClassifier(
        client=chat_completion_client,
        classifier_model=classifier_model,
        max_concurrency=settings.tool_classifier_max_concurrency,
    )


def _create_web_search_tool_provider(
    settings: Settings,
) -> IWebSearchToolProvider | None:
    if settings.web_search_mcp_url:
        return MCPWebSearchToolProvider(
            settings.web_search_mcp_url,
            tool_names=settings.web_search_mcp_tool_names,
        )
    if settings.web_search_brave_api_key:
        return MCPWebSearchToolProvider(
            _brave_mcp_config(settings),
            tool_names=settings.web_search_mcp_tool_names,
        )
    return None


def _brave_mcp_config(settings: Settings) -> dict[str, Any]:
    env = {"BRAVE_API_KEY": settings.web_search_brave_api_key}
    if settings.web_search_mcp_tool_names:
        env["BRAVE_MCP_ENABLED_TOOLS"] = ",".join(
            settings.web_search_mcp_tool_names
        )
    return {
        "mcpServers": {
            "brave": {
                "command": settings.web_search_mcp_command,
                "args": settings.web_search_mcp_args,
                "env": env,
            }
        }
    }


def _has_configured_chat_completion_route(settings: Settings, model: str) -> bool:
    return any(
        model.startswith(prefix)
        for prefix in _configured_chat_completion_prefixes(settings)
    )


def _configured_chat_completion_prefixes(settings: Settings) -> Iterable[str]:
    if settings.llm_lightning_api_key:
        yield from settings.llm_lightning_model_prefixes
    if settings.llm_novita_api_key:
        yield from settings.llm_novita_model_prefixes
    if settings.llm_fireworks_api_key:
        yield from settings.llm_fireworks_model_prefixes


def create_app(settings: Settings | None = None) -> Litestar:
    resolved_settings = settings or get_settings()
    db_engine = create_database_engine(resolved_settings.database_url)
    session_maker = create_session_maker(db_engine)
    chat_completion_client = _create_chat_completion_client(resolved_settings)
    tool_classifier = _create_tool_classifier(
        resolved_settings,
        chat_completion_client,
    )
    tool_call_classifier = _create_tool_call_classifier(
        resolved_settings,
        chat_completion_client,
    )
    web_search_tool_provider = _create_web_search_tool_provider(resolved_settings)
    state = State(
        {
            "api_key_manager": APIKeyManager(pepper=resolved_settings.api_key_pepper),
            "chat_completion_client": chat_completion_client,
            "db_engine": db_engine,
            "owns_engine": True,
            "session_maker": session_maker,
            "sealing_keyring": SealingKeyring.from_encoded(
                resolved_settings.sealing_keys
            ),
            "settings": resolved_settings,
            "tool_call_classifier": tool_call_classifier,
            "tool_call_policy_l1_cache": LRUCache(
                maxsize=resolved_settings.tool_call_policy_l1_maxsize
            ),
            "tool_classifier": tool_classifier,
            "tool_policy_l1_cache": LRUCache(
                maxsize=resolved_settings.tool_policy_l1_maxsize
            ),
            "web_search_tool_provider": web_search_tool_provider,
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
