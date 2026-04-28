from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from litestar import Request
from litestar.di import Provide

from plap.auth.dependencies import (
    provide_request_api_key_manager,
    provide_request_auth_context,
    provide_socket_api_key_manager,
    provide_socket_auth_context,
)
from plap.llms.dependencies import (
    provide_request_chat_completion_client,
    provide_socket_chat_completion_client,
)
from plap.persistence.dependencies import (
    provide_request_db_session,
    provide_socket_db_session,
)
from plap.responses.tools import (
    CachedToolCallPolicyResolver,
    CachedToolPolicyResolver,
    IToolCallPolicyResolver,
    IToolPolicyResolver,
    StaticToolCallPolicyResolver,
    StaticToolPolicyResolver,
)
from plap.responses.tools.repository import ToolClassificationRepository
from plap.settings import Settings


def provide_settings(request: Request[Any, Any, Any]) -> Settings:
    return request.app.state.settings


async def provide_tool_policy_resolver(
    request: Request[Any, Any, Any],
) -> AsyncIterator[IToolPolicyResolver]:
    classifier = request.app.state.tool_classifier
    if classifier is None:
        yield StaticToolPolicyResolver()
        return

    async with request.app.state.session_maker.begin() as session:
        resolver = CachedToolPolicyResolver(
            ToolClassificationRepository(session),
            classifier,
            classification_l1=request.app.state.tool_policy_l1_cache,
        )
        yield resolver


async def provide_tool_call_policy_resolver(
    request: Request[Any, Any, Any],
) -> AsyncIterator[IToolCallPolicyResolver]:
    classifier = request.app.state.tool_call_classifier
    if classifier is None:
        yield StaticToolCallPolicyResolver()
        return

    async with request.app.state.session_maker.begin() as session:
        resolver = CachedToolCallPolicyResolver(
            ToolClassificationRepository(session),
            classifier,
            classification_l1=request.app.state.tool_call_policy_l1_cache,
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
}


WEBSOCKET_ROUTE_DEPENDENCIES = {
    "api_key_manager": Provide(
        provide_socket_api_key_manager,
        use_cache=True,
        sync_to_thread=False,
    ),
    "auth_context": Provide(provide_socket_auth_context),
    "db_session": Provide(provide_socket_db_session),
    "chat_completion_client": Provide(
        provide_socket_chat_completion_client,
        use_cache=True,
        sync_to_thread=False,
    ),
    "tool_policy_resolver": Provide(
        provide_tool_policy_resolver,
    ),
    "tool_call_policy_resolver": Provide(
        provide_tool_call_policy_resolver,
    ),
}
