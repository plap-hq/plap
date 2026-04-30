from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import structlog
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
from plap.llms.crof import CrofChatCompletionClient
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
from plap.responses.errors import ResponseOperationUnsupportedError
from plap.responses.ingest import IngestionError
from plap.responses.reasoning import LLMReasoningSummarizer
from plap.responses.tools import (
    IToolCallClassifier,
    IToolClassifier,
    LLMToolCallClassifier,
    LLMToolClassifier,
    ToolPolicyError,
)
from plap.responses.tools.mcp import (
    IMCPToolProvider,
    MCPToolProvider,
)
from plap.settings import RuntimeModelProfileConfig, Settings, get_settings

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PublicError:
    status_code: int
    message: str
    error_type: str
    code: str
    headers: dict[str, str] | None = None
    log_event: str = "request.failed"


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


def public_error_for(exc: Exception) -> PublicError:
    if isinstance(exc, NotAuthorizedException):
        return PublicError(
            message="Not authorized.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
            headers={"WWW-Authenticate": "Bearer"},
            log_event="request.auth_failed",
        )
    if isinstance(exc, ValidationException | IngestionError):
        return PublicError(
            message="Invalid request.",
            status_code=400,
            error_type="invalid_request_error",
            code="invalid_request",
            log_event="request.validation_failed",
        )
    if isinstance(exc, ToolPolicyError):
        return PublicError(
            message="Invalid request.",
            status_code=400,
            error_type="invalid_request_error",
            code="invalid_tool",
            log_event="request.tool_validation_failed",
        )
    if isinstance(exc, ResponseOperationUnsupportedError):
        return PublicError(
            message="Operation is not supported.",
            status_code=exc.status_code,
            error_type="invalid_request_error",
            code="unsupported_operation",
            log_event="request.unsupported_operation",
        )
    if isinstance(exc, HTTPException):
        return PublicError(
            message="Request failed.",
            status_code=exc.status_code,
            error_type="invalid_request_error",
            code="request_error",
            log_event="request.http_failed",
        )
    return PublicError(
        message="Response generation failed.",
        status_code=500,
        error_type="server_error",
        code="server_error",
        log_event="request.unhandled_failed",
    )


def handle_public_exception(
    request: Request[Any, Any, Any],
    exc: Exception,
) -> Response[dict[str, Any]]:
    public_error = public_error_for(exc)
    log = logger.error if public_error.status_code >= 500 else logger.warning
    log(
        public_error.log_event,
        exc_info=True,
        exception_type=type(exc).__name__,
        method=request.method,
        path=request.url.path,
        status_code=public_error.status_code,
    )
    return _error_response(
        message=public_error.message,
        status_code=public_error.status_code,
        error_type=public_error.error_type,
        code=public_error.code,
        headers=public_error.headers,
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


def _create_mcp_tool_provider(
    settings: Settings,
) -> IMCPToolProvider | None:
    if settings.web_search_mcp_url:
        return MCPToolProvider(
            settings.web_search_mcp_url,
            tool_names=settings.web_search_mcp_tool_names,
        )
    if settings.web_search_mcp_config:
        return MCPToolProvider(
            settings.web_search_mcp_config,
            tool_names=settings.web_search_mcp_tool_names,
        )
    return None


def _validate_runtime_model_profiles(settings: Settings) -> None:
    for name, profile in settings.runtime_model_profiles.items():
        for model in _runtime_profile_models(profile):
            if not _has_configured_chat_completion_route(settings, model):
                raise ValueError(
                    "runtime model profile references an unconfigured LLM route: "
                    f"{name!r} -> {model!r}"
                )


def _resolve_runtime_model_profile(
    settings: Settings,
    model: str | None,
    service_tier: str | None = None,
) -> RuntimeModelProfileConfig:
    if model is None:
        raise ValueError("model is required")
    profile = settings.runtime_model_profiles.get(model)
    if profile is None:
        raise ValueError(f"unknown runtime model: {model!r}")
    return profile.for_service_tier(service_tier)


def _runtime_profile_models(profile: RuntimeModelProfileConfig) -> Iterable[str]:
    yield profile.main_model
    yield profile.main_debate_model
    yield profile.reviewer_model
    yield profile.arbitrator_model
    yield profile.reasoning_summarizer_model
    for override in profile.service_tier_overrides.values():
        if override.main_model is not None:
            yield override.main_model
        if override.main_debate_model is not None:
            yield override.main_debate_model
        if override.reviewer_model is not None:
            yield override.reviewer_model
        if override.arbitrator_model is not None:
            yield override.arbitrator_model
        if override.reasoning_summarizer_model is not None:
            yield override.reasoning_summarizer_model


def _has_configured_chat_completion_route(settings: Settings, model: str) -> bool:
    return any(
        model.startswith(prefix)
        for prefix in _configured_chat_completion_prefixes(settings)
    )


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
    db_engine = create_database_engine(resolved_settings.database_url)
    session_maker = create_session_maker(db_engine)
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
    mcp_tool_provider = _create_mcp_tool_provider(resolved_settings)
    _validate_runtime_model_profiles(resolved_settings)
    state = State(
        {
            "api_key_manager": APIKeyManager(pepper=resolved_settings.api_key_pepper),
            "chat_completion_client": chat_completion_client,
            "db_engine": db_engine,
            "owns_engine": True,
            "runtime_model_profiles": resolved_settings.runtime_model_profiles,
            "reasoning_summarizer": reasoning_summarizer,
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
            "mcp_tool_provider": mcp_tool_provider,
        }
    )

    return Litestar(
        route_handlers=RESPONSE_ROUTE_HANDLERS,
        exception_handlers={
            HTTPException: handle_public_exception,
            IngestionError: handle_public_exception,
            NotAuthorizedException: handle_public_exception,
            ResponseOperationUnsupportedError: handle_public_exception,
            ToolPolicyError: handle_public_exception,
            ValidationException: handle_public_exception,
            Exception: handle_public_exception,
        },
        on_shutdown=[_shutdown_database],
        state=state,
    )
