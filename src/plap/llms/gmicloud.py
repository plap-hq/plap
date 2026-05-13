from __future__ import annotations

from typing import Any

from plap.llms.chat import ChatCompletionRequest
from plap.llms.openai import (
    COMMON_CHAT_FIELDS,
    ChatProviderProfile,
    OpenAICompatibleChatCompletionClient,
    _set,
    build_chat_params,
)

GMICLOUD_OPENAI_BASE_URL = "https://api.gmi-serving.com/v1"

GMICLOUD_CHAT_FIELDS = (*COMMON_CHAT_FIELDS, "top_k")

GMICLOUD_CHAT_PROVIDER_PROFILE = ChatProviderProfile(
    developer_role="system",
    passthrough_fields=GMICLOUD_CHAT_FIELDS,
)


def to_gmicloud_chat_params(
    request: ChatCompletionRequest,
    *,
    stream: bool,
) -> dict[str, Any]:
    params = build_chat_params(
        request,
        stream=stream,
        profile=GMICLOUD_CHAT_PROVIDER_PROFILE,
    )
    max_completion_tokens = params.pop("max_completion_tokens", None)
    _set(params, "max_tokens", max_completion_tokens)
    params["context_length_exceeded_behavior"] = "error"
    return params


class GMICloudChatCompletionClient(OpenAICompatibleChatCompletionClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url or GMICLOUD_OPENAI_BASE_URL,
            client=client,
            developer_role="system",
        )

    def _chat_params(
        self,
        request: ChatCompletionRequest,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        return to_gmicloud_chat_params(request, stream=stream)
