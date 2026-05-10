from __future__ import annotations

from typing import Any

from plap.llms.chat import ChatCompletionRequest
from plap.llms.openai import (
    OPENAI_CHAT_FIELDS,
    ChatProviderProfile,
    OpenAICompatibleChatCompletionClient,
    build_chat_params,
)

CANOPYWAVE_OPENAI_BASE_URL = "https://inference.canopywave.io/v1"

CANOPYWAVE_CHAT_PROVIDER_PROFILE = ChatProviderProfile(
    developer_role="system",
    passthrough_fields=OPENAI_CHAT_FIELDS,
)


class CanopyWaveChatCompletionClient(OpenAICompatibleChatCompletionClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url or CANOPYWAVE_OPENAI_BASE_URL,
            client=client,
            developer_role="system",
        )

    def _chat_params(
        self,
        request: ChatCompletionRequest,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        return to_canopywave_chat_params(request, stream=stream)


def _strip_schema_patterns(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_schema_patterns(child) for key, child in value.items() if key != "pattern"}
    if isinstance(value, list):
        return [_strip_schema_patterns(child) for child in value]
    return value


def _strip_tool_schema_patterns(params: dict[str, Any]) -> None:
    tools = params.get("tools")
    if not isinstance(tools, list):
        return
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        parameters = function.get("parameters")
        if parameters is not None:
            function["parameters"] = _strip_schema_patterns(parameters)


def to_canopywave_chat_params(
    request: ChatCompletionRequest,
    *,
    stream: bool,
) -> dict[str, Any]:
    params = build_chat_params(
        request,
        stream=stream,
        profile=CANOPYWAVE_CHAT_PROVIDER_PROFILE,
    )
    _strip_tool_schema_patterns(params)
    return params
