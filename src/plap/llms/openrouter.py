from __future__ import annotations

from typing import Any

from plap.llms.chat import ChatCompletionRequest
from plap.llms.errors import ChatCompletionUnsupportedRequestError
from plap.llms.openai import (
    OPENAI_CHAT_FIELDS,
    ChatProviderProfile,
    OpenAICompatibleChatCompletionClient,
    build_chat_params,
)

OPENROUTER_OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_SPECIAL_MODEL_SUFFIXES = frozenset(
    {
        "exacto",
        "extended",
        "floor",
        "free",
        "nitro",
        "online",
        "thinking",
    }
)

OPENROUTER_CHAT_PROVIDER_PROFILE = ChatProviderProfile(
    developer_role="system",
    passthrough_fields=OPENAI_CHAT_FIELDS,
)


def _openrouter_model_and_provider_preferences(
    model: str,
) -> tuple[str, dict[str, Any] | None]:
    segments = model.split(":")
    base_model = segments[0]
    if not base_model:
        raise ChatCompletionUnsupportedRequestError(f"OpenRouter model {model!r} is missing a base model slug")

    provider_order: list[str] = []
    model_segments = [base_model]
    for segment in segments[1:]:
        if not segment:
            raise ChatCompletionUnsupportedRequestError(f"OpenRouter model {model!r} contains an empty suffix segment")
        if segment in OPENROUTER_SPECIAL_MODEL_SUFFIXES:
            model_segments.append(segment)
            continue
        provider_order.append(segment)

    provider_preferences = None
    if provider_order:
        provider_preferences = {
            "order": provider_order,
            "allow_fallbacks": False,
        }
    return ":".join(model_segments), provider_preferences


def to_openrouter_chat_params(
    request: ChatCompletionRequest,
    *,
    stream: bool,
) -> dict[str, Any]:
    params = build_chat_params(
        request,
        stream=stream,
        profile=OPENROUTER_CHAT_PROVIDER_PROFILE,
    )
    params["model"], provider_preferences = _openrouter_model_and_provider_preferences(request.model)
    if provider_preferences is None:
        return params

    extra_body = params.get("extra_body")
    if extra_body is None:
        params["extra_body"] = {"provider": provider_preferences}
        return params

    params["extra_body"] = {**extra_body, "provider": provider_preferences}
    return params


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
