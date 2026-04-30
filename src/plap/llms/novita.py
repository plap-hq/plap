from __future__ import annotations

from typing import Any

from plap.llms.chat import (
    ChatCompletionRequest,
    ChatToolChoiceFunction,
)
from plap.llms.errors import ChatCompletionUnsupportedRequestError
from plap.llms.openai import (
    COMMON_CHAT_FIELDS,
    ChatProviderProfile,
    OpenAICompatibleChatCompletionClient,
    _set,
    build_chat_params,
)

NOVITA_OPENAI_BASE_URL = "https://api.novita.ai/openai"
NOVITA_FORCED_TOOL_CHOICE_QUIRK_MODELS = frozenset({"deepseek/deepseek-v4-flash"})
NOVITA_THINKING_CONTROL_MODELS = frozenset({"deepseek/deepseek-v4-flash"})

NOVITA_CHAT_FIELDS = (*COMMON_CHAT_FIELDS, "logprobs", "reasoning_effort")

NOVITA_CHAT_PROVIDER_PROFILE = ChatProviderProfile(
    developer_role="system",
    passthrough_fields=NOVITA_CHAT_FIELDS,
)


class NovitaChatCompletionClient(OpenAICompatibleChatCompletionClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url or NOVITA_OPENAI_BASE_URL,
            client=client,
            developer_role="system",
        )

    def _chat_params(
        self,
        request: ChatCompletionRequest,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        return to_novita_chat_params(request, stream=stream)


def to_novita_chat_params(
    request: ChatCompletionRequest,
    *,
    stream: bool,
) -> dict[str, Any]:
    params = build_chat_params(
        request,
        stream=stream,
        profile=NOVITA_CHAT_PROVIDER_PROFILE,
    )
    max_completion_tokens = params.pop("max_completion_tokens", None)
    _set(params, "max_tokens", max_completion_tokens)
    _apply_novita_tool_choice_quirks(params, request)
    _set(params, "extra_body", _deepseek_v4_thinking_extra_body(request))

    return params


def _deepseek_v4_thinking_extra_body(
    request: ChatCompletionRequest,
) -> dict[str, Any] | None:
    if request.model not in NOVITA_THINKING_CONTROL_MODELS or request.reasoning_effort is None:
        return None
    thinking_type = "disabled" if request.reasoning_effort == "none" else "enabled"
    return {"thinking": {"type": thinking_type}}


def _apply_novita_tool_choice_quirks(
    params: dict[str, Any],
    request: ChatCompletionRequest,
) -> None:
    if request.model not in NOVITA_FORCED_TOOL_CHOICE_QUIRK_MODELS:
        return
    if not isinstance(request.tool_choice, ChatToolChoiceFunction):
        return
    if len(request.tools) == 1 and request.tools[0].function.name == request.tool_choice.name:
        params["tool_choice"] = "required"
        return
    raise ChatCompletionUnsupportedRequestError(
        "Novita model rejects forced function tool_choice objects; use tool_choice='required' or provide exactly one matching tool"
    )
