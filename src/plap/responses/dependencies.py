from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from litestar import Request
from litestar.connection import WebSocket
from litestar.di import Provide

from plap.auth.dependencies import (
    provide_request_api_key_manager,
    provide_request_auth_context,
    provide_socket_api_key_manager,
    provide_socket_auth_context,
)
from plap.keyring import SealingKeyring
from plap.llms.dependencies import (
    provide_request_chat_completion_client,
    provide_socket_chat_completion_client,
)
from plap.persistence.dependencies import (
    provide_request_db_session,
    provide_socket_db_session,
)
from plap.responses.reasoning import IReasoningSummarizer
from plap.responses.tools import (
    CachedToolCallPolicyResolver,
    CachedToolPolicyResolver,
    IToolCallPolicyResolver,
    IToolPolicyResolver,
    StaticToolCallPolicyResolver,
    StaticToolPolicyResolver,
)
from plap.responses.tools.mcp import IMCPToolProvider
from plap.responses.tools.repository import ToolClassificationRepository
from plap.settings import Settings


def provide_settings(request: Request[Any, Any, Any]) -> Settings:
    return request.app.state.settings


def provide_socket_settings(socket: WebSocket) -> Settings:
    return socket.app.state.settings


def provide_sealing_keyring(
    request: Request[Any, Any, Any],
) -> SealingKeyring:
    return request.app.state.sealing_keyring


def provide_socket_sealing_keyring(socket: WebSocket) -> SealingKeyring:
    return socket.app.state.sealing_keyring


def provide_mcp_tool_providers(
    request: Request[Any, Any, Any],
) -> tuple[IMCPToolProvider, ...]:
    return request.app.state.mcp_tool_providers


def provide_socket_mcp_tool_providers(
    socket: WebSocket,
) -> tuple[IMCPToolProvider, ...]:
    return socket.app.state.mcp_tool_providers


def provide_reasoning_summarizer(
    request: Request[Any, Any, Any],
) -> IReasoningSummarizer:
    return request.app.state.reasoning_summarizer


def provide_socket_reasoning_summarizer(
    socket: WebSocket,
) -> IReasoningSummarizer:
    return socket.app.state.reasoning_summarizer


async def provide_tool_policy_resolver(
    request: Request[Any, Any, Any] | WebSocket,
) -> AsyncIterator[IToolPolicyResolver]:
    async for resolver in _provide_tool_policy_resolver(request):
        yield resolver


async def provide_socket_tool_policy_resolver(
    socket: WebSocket,
) -> AsyncIterator[IToolPolicyResolver]:
    async for resolver in _provide_tool_policy_resolver(socket):
        yield resolver


async def _provide_tool_policy_resolver(
    connection: Request[Any, Any, Any] | WebSocket,
) -> AsyncIterator[IToolPolicyResolver]:
    classifier = connection.app.state.tool_classifier
    if classifier is None:
        yield StaticToolPolicyResolver()
        return

    async with connection.app.state.session_maker.begin() as session:
        resolver = CachedToolPolicyResolver(
            ToolClassificationRepository(session),
            classifier,
            classification_l1=connection.app.state.tool_policy_l1_cache,
        )
        yield resolver


async def provide_tool_call_policy_resolver(
    request: Request[Any, Any, Any] | WebSocket,
) -> AsyncIterator[IToolCallPolicyResolver]:
    async for resolver in _provide_tool_call_policy_resolver(request):
        yield resolver


async def provide_socket_tool_call_policy_resolver(
    socket: WebSocket,
) -> AsyncIterator[IToolCallPolicyResolver]:
    async for resolver in _provide_tool_call_policy_resolver(socket):
        yield resolver


async def _provide_tool_call_policy_resolver(
    connection: Request[Any, Any, Any] | WebSocket,
) -> AsyncIterator[IToolCallPolicyResolver]:
    classifier = connection.app.state.tool_call_classifier
    if classifier is None:
        yield StaticToolCallPolicyResolver()
        return

    async with connection.app.state.session_maker.begin() as session:
        resolver = CachedToolCallPolicyResolver(
            ToolClassificationRepository(session),
            classifier,
            classification_l1=connection.app.state.tool_call_policy_l1_cache,
        )
        yield resolver


HTTP_ROUTE_DEPENDENCIES = {
    "api_key_manager": Provide(
        provide_request_api_key_manager,
        use_cache=True,
        sync_to_thread=False,
    ),
    "auth_context": Provide(provide_request_auth_context),
    "db_session": Provide(provide_request_db_session),
    "settings": Provide(provide_settings, use_cache=True, sync_to_thread=False),
    "sealing_keyring": Provide(
        provide_sealing_keyring,
        use_cache=True,
        sync_to_thread=False,
    ),
    "chat_completion_client": Provide(
        provide_request_chat_completion_client,
        use_cache=True,
        sync_to_thread=False,
    ),
    "tool_policy_resolver": Provide(
        provide_tool_policy_resolver,
    ),
    "tool_call_policy_resolver": Provide(
        provide_tool_call_policy_resolver,
    ),
    "mcp_tool_providers": Provide(
        provide_mcp_tool_providers,
        use_cache=True,
        sync_to_thread=False,
    ),
    "reasoning_summarizer": Provide(
        provide_reasoning_summarizer,
        use_cache=True,
        sync_to_thread=False,
    ),
}


WEBSOCKET_ROUTE_DEPENDENCIES = {
    "api_key_manager": Provide(
        provide_socket_api_key_manager,
        use_cache=True,
        sync_to_thread=False,
    ),
    "auth_context": Provide(provide_socket_auth_context),
    "db_session": Provide(provide_socket_db_session),
    "settings": Provide(provide_socket_settings, use_cache=True, sync_to_thread=False),
    "sealing_keyring": Provide(
        provide_socket_sealing_keyring,
        use_cache=True,
        sync_to_thread=False,
    ),
    "chat_completion_client": Provide(
        provide_socket_chat_completion_client,
        use_cache=True,
        sync_to_thread=False,
    ),
    "tool_policy_resolver": Provide(
        provide_socket_tool_policy_resolver,
    ),
    "tool_call_policy_resolver": Provide(
        provide_socket_tool_call_policy_resolver,
    ),
    "mcp_tool_providers": Provide(
        provide_socket_mcp_tool_providers,
        use_cache=True,
        sync_to_thread=False,
    ),
    "reasoning_summarizer": Provide(
        provide_socket_reasoning_summarizer,
        use_cache=True,
        sync_to_thread=False,
    ),
}
