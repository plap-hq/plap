from __future__ import annotations

from typing import Any

from litestar import Request
from litestar.di import Provide

from plap.auth.dependencies import (
    provide_request_api_key_manager,
    provide_request_auth_context,
    provide_socket_api_key_manager,
    provide_socket_auth_context,
)
from plap.persistence.dependencies import (
    provide_request_db_session,
    provide_socket_db_session,
)
from plap.responses.tools import IToolPolicyResolver
from plap.settings import Settings


def provide_settings(request: Request[Any, Any, Any]) -> Settings:
    return request.app.state.settings


def provide_tool_policy_resolver(
    request: Request[Any, Any, Any],
) -> IToolPolicyResolver:
    return request.app.state.tool_policy_resolver


HTTP_ROUTE_DEPENDENCIES = {
    "api_key_manager": Provide(
        provide_request_api_key_manager,
        use_cache=True,
        sync_to_thread=False,
    ),
    "auth_context": Provide(provide_request_auth_context),
    "db_session": Provide(provide_request_db_session),
    "settings": Provide(provide_settings, use_cache=True, sync_to_thread=False),
    "tool_policy_resolver": Provide(
        provide_tool_policy_resolver,
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
}
