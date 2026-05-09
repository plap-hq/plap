from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import structlog
from cachetools import LRUCache
from litestar import Litestar, Request, Response
from litestar.datastructures import State
from litestar.exceptions import HTTPException, NotAuthorizedException, ValidationException

from plap.auth import APIKeyManager
from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.keyring import SealingKeyring
from plap.llms.chat import ChatCompletionRequest, ChatFunctionTool, ChatTool, IChatCompletionClient
from plap.llms.chat import ChatMessage as LLMChatMessage
from plap.llms.crof import CrofChatCompletionClient
from plap.llms.fireworks import FireworksChatCompletionClient
from plap.llms.lightning import LightningChatCompletionClient
from plap.llms.novita import NovitaChatCompletionClient
from plap.llms.openrouter import OpenRouterChatCompletionClient
from plap.llms.router import (
    ModelRoute,
    RoutingChatCompletionClient,
    UnavailableChatCompletionClient,
)
from plap.logging import configure_logging, log_debug
from plap.persistence import Database
from plap.responses.reasoning import LLMReasoningSummarizer
from plap.responses.routes import RESPONSE_ROUTE_HANDLERS
from plap.responses.tokens import measure_request_tokens
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


def handle_plap_error(
    request: Request[Any, Any, Any],
    exc: PlapError,
) -> Response[dict[str, Any]]:
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

    if settings.llm_openrouter_api_key:
        client = OpenRouterChatCompletionClient(api_key=settings.llm_openrouter_api_key)
        yield ModelRoute(prefix="openrouter/", client=client)


def _create_tool_classifier(
    settings: Settings,
    chat_completion_client: IChatCompletionClient,
) -> IToolClassifier:
    if not _has_configured_chat_completion_route(
        settings,
        TOOL_EFFECT_CLASSIFIER_MODEL,
    ):
        raise PlapError(
            public=None,
            private=PrivateError(
                event="app.startup_invalid",
                reason="tool_effect_classifier_route_unconfigured",
                message=f"tool effect classifier model does not match any configured LLM route: {TOOL_EFFECT_CLASSIFIER_MODEL!r}",
                level=ErrorLevel.ERROR,
                context={"model": TOOL_EFFECT_CLASSIFIER_MODEL},
            ),
        )
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
        raise PlapError(
            public=None,
            private=PrivateError(
                event="app.startup_invalid",
                reason="tool_call_classifier_route_unconfigured",
                message=f"tool call classifier model does not match any configured LLM route: {TOOL_CALL_EFFECT_CLASSIFIER_MODEL!r}",
                level=ErrorLevel.ERROR,
                context={"model": TOOL_CALL_EFFECT_CLASSIFIER_MODEL},
            ),
        )
    return LLMToolCallClassifier(
        client=chat_completion_client,
        classifier_model=TOOL_CALL_EFFECT_CLASSIFIER_MODEL,
        max_concurrency=settings.tool_classifier_max_concurrency,
    )


def _create_mcp_tool_providers(settings: Settings) -> tuple[IMCPToolProvider, ...]:
    return tuple(_create_mcp_tool_provider(server) for server in settings.mcp_servers)


def _create_mcp_tool_provider(server: MCPServerConfig) -> IMCPToolProvider:
    try:
        transport = server.mcp_config()
    except (TypeError, ValueError) as exc:
        raise PlapError(
            public=None,
            private=PrivateError(
                event="app.startup_invalid",
                reason="mcp_transport_invalid",
                message=f"invalid MCP server config for {server.name!r}: {exc}",
                level=ErrorLevel.ERROR,
                context={"server_name": server.name},
            ),
        ) from exc
    return MCPToolProvider(server.name, transport, tools=server.tools)


def _validate_runtime_model_profiles(settings: Settings) -> None:
    for name, profile in settings.runtime_model_profiles.items():
        for model in profile.all_models():
            if not _has_configured_chat_completion_route(settings, model):
                raise PlapError(
                    public=None,
                    private=PrivateError(
                        event="app.startup_invalid",
                        reason="runtime_profile_route_unconfigured",
                        message=f"runtime model profile references an unconfigured LLM route: {name!r} -> {model!r}",
                        level=ErrorLevel.ERROR,
                        context={"model": model, "runtime_model_profile": name},
                    ),
                )


def _runtime_profile_actors(settings: Settings) -> Iterable[tuple[str, str, object]]:
    for profile_name, profile in settings.runtime_model_profiles.items():
        yield profile_name, "main", profile.main
        yield profile_name, "compactor", profile.compactor
        yield profile_name, "main_debate", profile.main_debate
        yield profile_name, "reviewer", profile.reviewer
        yield profile_name, "arbitrator", profile.arbitrator
        yield profile_name, "reasoning_summarizer", profile.reasoning_summarizer


def _validate_runtime_profile_tokenizers(settings: Settings) -> None:
    validated: set[tuple[str, str | None, bool]] = set()
    probe_request = ChatCompletionRequest(
        model="tokenizer-probe",
        messages=[
            LLMChatMessage(role="developer", content="Tokenization probe."),
            LLMChatMessage(role="user", content="hello"),
        ],
        tools=[
            ChatTool(
                function=ChatFunctionTool(
                    name="probe_tool",
                    description="Probe tool",
                    parameters={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                    },
                )
            )
        ],
    )
    for profile_name, actor_name, actor_config in _runtime_profile_actors(settings):
        if actor_config.tokenizer_hf_repo is None:
            continue
        tokenizer_key = (
            actor_config.tokenizer_hf_repo,
            actor_config.tokenizer_revision,
            actor_config.tokenizer_trust_remote_code,
        )
        if tokenizer_key in validated:
            continue
        try:
            measure_request_tokens(probe_request, actor_config=actor_config)
        except Exception as exc:
            raise PlapError(
                public=None,
                private=PrivateError(
                    event="app.startup_invalid",
                    reason="runtime_profile_tokenizer_invalid",
                    message=(
                        "runtime model profile tokenizer validation failed: "
                        f"{profile_name!r}.{actor_name} -> {actor_config.tokenizer_hf_repo!r}"
                    ),
                    level=ErrorLevel.ERROR,
                    cause=exc,
                    context={
                        "actor": actor_name,
                        "runtime_model_profile": profile_name,
                        "tokenizer_hf_repo": actor_config.tokenizer_hf_repo,
                        "tokenizer_revision": actor_config.tokenizer_revision,
                        "tokenizer_trust_remote_code": actor_config.tokenizer_trust_remote_code,
                    },
                ),
            ) from exc
        validated.add(tokenizer_key)


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
    if settings.llm_openrouter_api_key:
        yield "openrouter/"


def create_app(settings: Settings | None = None) -> Litestar:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings)
    try:
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
        _validate_runtime_profile_tokenizers(resolved_settings)
    except PlapError as exc:
        exc.log(logger)
        raise
    log_debug(
        logger,
        "app.startup",
        debug_payloads=resolved_settings.debug_payloads,
        log_file=resolved_settings.log_file,
        log_json=resolved_settings.log_json,
        mcp_servers=[server.name for server in resolved_settings.mcp_servers],
        provider_routes=sorted(_configured_chat_completion_prefixes(resolved_settings)),
        runtime_models=sorted(resolved_settings.runtime_model_profiles),
    )
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
            PlapError: handle_plap_error,
            ValidationException: handle_validation_exception,
            Exception: handle_unexpected_exception,
        },
        on_shutdown=[_shutdown_database],
        state=state,
    )
