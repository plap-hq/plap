from __future__ import annotations

from typing import Any

from plap.llms.chat import ChatCompletionRequest
from plap.llms.openai import (
    OPENAI_CHAT_FIELDS,
    ChatProviderProfile,
    OpenAICompatibleChatCompletionClient,
    build_chat_params,
)

OPENROUTER_OPENAI_BASE_URL = "https://openrouter.ai/api/v1"

OPENROUTER_CHAT_PROVIDER_PROFILE = ChatProviderProfile(
    developer_role="system",
    passthrough_fields=OPENAI_CHAT_FIELDS,
)


class OpenRouterChatCompletionClient(OpenAICompatibleChatCompletionClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url or OPENROUTER_OPENAI_BASE_URL,
            client=client,
            developer_role="system",
        )

    def _chat_params(
        self,
        request: ChatCompletionRequest,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        return to_openrouter_chat_params(request, stream=stream)


def to_openrouter_chat_params(
    request: ChatCompletionRequest,
    *,
    stream: bool,
) -> dict[str, Any]:
    return build_chat_params(
        request,
        stream=stream,
        profile=OPENROUTER_CHAT_PROVIDER_PROFILE,
    )
