from __future__ import annotations

from typing import Any

from plap.llms.chat import ChatCompletionRequest, ChatToolChoiceFunction
from plap.llms.errors import ChatCompletionUnsupportedRequestError
from plap.llms.json_utils import JSONInvalidError, parse_json_value_with_repair
from plap.llms.openai import (
    COMMON_CHAT_FIELDS,
    ChatProviderProfile,
    OpenAICompatibleChatCompletionClient,
    _set,
    build_chat_params,
)

GMICLOUD_OPENAI_BASE_URL = "https://api.gmi-serving.com/v1"
GMICLOUD_DEEPSEEK_MODELS = frozenset({"deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Pro"})

GMICLOUD_CHAT_FIELDS = (*COMMON_CHAT_FIELDS, "top_k")

GMICLOUD_CHAT_PROVIDER_PROFILE = ChatProviderProfile(
    developer_role="system",
    passthrough_fields=GMICLOUD_CHAT_FIELDS,
)


def _gmicloud_parse_tool_input(arguments: str) -> dict[str, Any]:
    try:
        value = parse_json_value_with_repair(arguments)
    except JSONInvalidError:
        return {"arguments": arguments}
    if isinstance(value, dict):
        return value
    return {"arguments": value}


def _gmicloud_is_tool_result_message(message: dict[str, Any]) -> bool:
    if message.get("role") != "user":
        return False
    content = message.get("content")
    return isinstance(content, list) and all(
        isinstance(block, dict) and block.get("type") == "tool_result" for block in content
    )


def _gmicloud_deepseek_assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        transformed: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
        _set(transformed, "reasoning_content", message.get("reasoning_content"))
        _set(transformed, "reasoning_details", message.get("reasoning_details"))
        return transformed

    content_blocks: list[dict[str, Any]] = []
    reasoning_content = message.get("reasoning_content")
    if reasoning_content is not None:
        content_blocks.append({"type": "thinking", "thinking": reasoning_content})
    content = message.get("content")
    if isinstance(content, str) and content:
        content_blocks.append({"type": "text", "text": content})
    for tool_call in tool_calls:
        function = tool_call.get("function") if isinstance(tool_call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        arguments = function.get("arguments") if isinstance(function, dict) else None
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id"),
                "name": name,
                "input": _gmicloud_parse_tool_input(arguments if isinstance(arguments, str) else ""),
            }
        )
    return {"role": "assistant", "content": content_blocks}


def _gmicloud_deepseek_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transformed: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            tool_result = {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id"),
                "content": message.get("content") or "",
            }
            if transformed and _gmicloud_is_tool_result_message(transformed[-1]):
                transformed[-1]["content"].append(tool_result)
            else:
                transformed.append({"role": "user", "content": [tool_result]})
            continue
        if role == "assistant":
            transformed.append(_gmicloud_deepseek_assistant_message(message))
            continue
        if role == "system":
            transformed.append({"role": "user", "content": message.get("content") or ""})
            continue
        transformed.append({"role": role, "content": message.get("content") or ""})
    return transformed


def _gmicloud_messages(request: ChatCompletionRequest, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if request.model in GMICLOUD_DEEPSEEK_MODELS:
        return _gmicloud_deepseek_messages(messages)
    return messages


def _gmicloud_thinking_control(request: ChatCompletionRequest) -> dict[str, str] | None:
    if request.model not in GMICLOUD_DEEPSEEK_MODELS or request.reasoning_effort is None:
        return None
    thinking_type = "disabled" if request.reasoning_effort == "none" else "enabled"
    return {"type": thinking_type}


def _gmicloud_reasoning_effort(request: ChatCompletionRequest) -> str | None:
    effort = request.reasoning_effort
    if effort is None:
        return None
    if effort == "none":
        return None
    return effort


def _apply_gmicloud_tool_choice_quirks(
    params: dict[str, Any],
    request: ChatCompletionRequest,
) -> None:
    if request.model not in GMICLOUD_DEEPSEEK_MODELS:
        return
    if not isinstance(request.tool_choice, ChatToolChoiceFunction):
        return
    if len(request.tools) == 1 and request.tools[0].function.name == request.tool_choice.name:
        params["tool_choice"] = "required"
        return
    raise ChatCompletionUnsupportedRequestError(
        "GMICloud DeepSeek rejects forced function tool_choice objects; use tool_choice='required' or provide exactly one matching tool"
    )


def _gmicloud_extra_body(request: ChatCompletionRequest) -> dict[str, Any]:
    body: dict[str, Any] = {"context_length_exceeded_behavior": "error"} #does not work on OpenAI models, do not pass
    thinking = _gmicloud_thinking_control(request)
    if thinking is not None:
        body["thinking"] = thinking
    return body


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
    params["messages"] = _gmicloud_messages(request, params["messages"])
    max_completion_tokens = params.pop("max_completion_tokens", None)
    _set(params, "max_tokens", max_completion_tokens) #does not work on OpenAI models, max_completion_tokens is used instead
    _apply_gmicloud_tool_choice_quirks(params, request)
    _set(params, "reasoning_effort", _gmicloud_reasoning_effort(request))
    params["extra_body"] = _gmicloud_extra_body(request)
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
