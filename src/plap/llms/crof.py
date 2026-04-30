from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import msgspec
from jsonschema import Draft202012Validator, SchemaError, ValidationError

from plap.llms.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatResponseFormat,
)
from plap.llms.errors import ChatCompletionInvalidRequestError
from plap.llms.openai import (
    COMMON_CHAT_FIELDS,
    ChatProviderProfile,
    OpenAICompatibleChatCompletionClient,
    _set,
    build_chat_params,
)

CROF_OPENAI_BASE_URL = "https://beta.crof.ai/v1"
CROF_REASONING_CONTENT_RESPONSE_FORMAT_MODELS = frozenset({"glm-4.7-flash"})
CROF_THINKING_CONTROL_MODELS = frozenset({"glm-4.7-flash"})

CROF_CHAT_FIELDS = (
    *COMMON_CHAT_FIELDS,
    "reasoning_effort",
)

CROF_CHAT_PROVIDER_PROFILE = ChatProviderProfile(
    developer_role="system",
    passthrough_fields=CROF_CHAT_FIELDS,
)


class CrofChatCompletionClient(OpenAICompatibleChatCompletionClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=CROF_OPENAI_BASE_URL,
            client=client,
            developer_role="system",
        )

    def _chat_params(
        self,
        request: ChatCompletionRequest,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        return to_crof_chat_params(request, stream=stream)

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        result = await super().complete(request)
        return _normalize_crof_response_format_result(request, result)

    async def stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionDelta]:
        async for delta in super().stream(request):
            yield delta


def to_crof_chat_params(
    request: ChatCompletionRequest,
    *,
    stream: bool,
) -> dict[str, Any]:
    params = build_chat_params(
        request,
        stream=stream,
        profile=CROF_CHAT_PROVIDER_PROFILE,
    )
    max_completion_tokens = params.pop("max_completion_tokens", None)
    _set(params, "max_tokens", max_completion_tokens)
    _set(params, "extra_body", _crof_thinking_extra_body(request))
    return params


def _crof_thinking_extra_body(
    request: ChatCompletionRequest,
) -> dict[str, Any] | None:
    if request.model not in CROF_THINKING_CONTROL_MODELS:
        return None
    if request.reasoning_effort is not None and request.reasoning_effort != "none":
        return None
    return {"thinking": {"type": "disabled"}}


def _normalize_crof_response_format_result(
    request: ChatCompletionRequest,
    result: ChatCompletionResult,
) -> ChatCompletionResult:
    response_format = request.response_format
    reasoning_content = result.message.reasoning_content
    if (
        request.model not in CROF_REASONING_CONTENT_RESPONSE_FORMAT_MODELS
        or response_format is None
        or result.message.content
        or not reasoning_content
        or _response_format_validation_error(reasoning_content, response_format) is not None
    ):
        return result

    return replace(
        result,
        message=replace(result.message, content=reasoning_content),
    )


def _response_format_validation_error(
    content: str | None,
    response_format: ChatResponseFormat,
) -> str | None:
    if not content or not content.strip():
        return "response content is empty"
    try:
        value = msgspec.json.decode(content.encode())
    except msgspec.DecodeError as exc:
        return f"invalid JSON: {exc}"

    if response_format.type == "json_object" and not isinstance(value, dict):
        return "top-level JSON value is not an object"
    if response_format.type == "json_schema":
        return _json_schema_validation_error(value, response_format)
    return None


def _json_schema_validation_error(
    value: object,
    response_format: ChatResponseFormat,
) -> str | None:
    schema = response_format.schema or {}
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ChatCompletionInvalidRequestError(f"Invalid JSON schema for response_format: {exc.message}") from exc
    validator = Draft202012Validator(schema)
    try:
        validator.validate(value)
    except ValidationError as exc:
        return f"response JSON does not match schema: {exc.message}"
    return None
