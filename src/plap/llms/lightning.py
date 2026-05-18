from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from typing import Any

import msgspec
from jsonschema import Draft202012Validator, SchemaError, ValidationError

from plap.llms.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatMessage,
    ChatResponseFormat,
)
from plap.llms.errors import (
    ChatCompletionInvalidRequestError,
    ChatCompletionProviderError,
)
from plap.llms.json_utils import JSONInvalidError, normalize_json_text_with_repair, parse_json_value_with_repair
from plap.llms.openai import (
    COMMON_CHAT_FIELDS,
    ChatProviderProfile,
    OpenAICompatibleChatCompletionClient,
    build_chat_params,
)

LIGHTNING_OPENAI_BASE_URL = "https://lightning.ai/api/v1"
LIGHTNING_RESPONSE_FORMAT_FALLBACK_MODELS = frozenset({"lightning-ai/gpt-oss-120b"})

LIGHTNING_CHAT_FIELDS = (
    *COMMON_CHAT_FIELDS,
    "logprobs",
    "top_logprobs",
    "reasoning_effort",
    "user",
    "metadata",
)

LIGHTNING_CHAT_PROVIDER_PROFILE = ChatProviderProfile(
    developer_role="system",
    passthrough_fields=LIGHTNING_CHAT_FIELDS,
)


class LightningChatCompletionClient(OpenAICompatibleChatCompletionClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url or LIGHTNING_OPENAI_BASE_URL,
            client=client,
            developer_role="system",
        )

    def _chat_params(
        self,
        request: ChatCompletionRequest,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        return to_lightning_chat_params(request, stream=stream)

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        if _uses_response_format_fallback(request):
            return await self._complete_with_response_format_fallback(request)
        return await super().complete(request)

    async def stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionDelta]:
        if _uses_response_format_fallback(request):
            result = await self._complete_with_response_format_fallback(request)
            for delta in response_format_fallback_stream_deltas(result):
                yield delta
            return
        async for delta in super().stream(request):
            yield delta

    async def _complete_with_response_format_fallback(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResult:
        return await complete_with_response_format_fallback(
            request,
            complete=super().complete,
            provider_name="Lightning",
        )


def to_lightning_chat_params(
    request: ChatCompletionRequest,
    *,
    stream: bool,
) -> dict[str, Any]:
    return build_chat_params(
        request,
        stream=stream,
        profile=LIGHTNING_CHAT_PROVIDER_PROFILE,
    )


def _uses_response_format_fallback(request: ChatCompletionRequest) -> bool:
    return (
        request.model in LIGHTNING_RESPONSE_FORMAT_FALLBACK_MODELS
        and request.response_format is not None
        and request.response_format.type in {"json_object", "json_schema"}
    )


def _validated_response_format_result(
    result: ChatCompletionResult,
    response_format: ChatResponseFormat,
) -> tuple[ChatCompletionResult, str | None]:
    normalized_content, validation_error = _validated_response_format_content(result.message.content, response_format)
    if validation_error is not None or normalized_content is None:
        return result, validation_error
    if normalized_content == result.message.content:
        return result, None
    return replace(result, message=replace(result.message, content=normalized_content)), None


async def complete_with_response_format_fallback(
    request: ChatCompletionRequest,
    *,
    complete: Callable[[ChatCompletionRequest], Awaitable[ChatCompletionResult]],
    provider_name: str,
) -> ChatCompletionResult:
    response_format = request.response_format
    if response_format is None:
        return await complete(request)

    result, validation_error = _validated_response_format_result(
        await complete(response_format_fallback_request(request)),
        response_format,
    )
    if validation_error is None:
        return result

    retry_result, retry_validation_error = _validated_response_format_result(
        await complete(
            response_format_fallback_request(
                request,
                invalid_content=result.message.content,
                validation_error=validation_error,
            )
        ),
        response_format,
    )
    if retry_validation_error is None:
        return retry_result

    raise ChatCompletionProviderError(f"{provider_name} response_format fallback returned invalid JSON: {retry_validation_error}")


def response_format_fallback_stream_deltas(
    result: ChatCompletionResult,
) -> tuple[ChatCompletionDelta, ChatCompletionDelta]:
    return (
        ChatCompletionDelta(
            id=result.id,
            model=result.model,
            created_at=result.created_at,
            choice_index=0,
            content_delta=result.message.content,
            refusal_delta=result.message.refusal,
            reasoning_delta=result.message.reasoning_content,
            reasoning_details_delta=result.message.reasoning_details,
        ),
        ChatCompletionDelta(
            id=result.id,
            model=result.model,
            created_at=result.created_at,
            choice_index=0,
            finish_reason=result.finish_reason,
            usage=result.usage,
            system_fingerprint=result.system_fingerprint,
            service_tier=result.service_tier,
        ),
    )


def response_format_fallback_request(
    request: ChatCompletionRequest,
    *,
    invalid_content: str | None = None,
    validation_error: str | None = None,
) -> ChatCompletionRequest:
    response_format = request.response_format
    if response_format is None:
        return request

    messages = [
        ChatMessage(
            role="system",
            content=response_format_instruction(response_format),
        ),
        *request.messages,
    ]
    if validation_error is not None:
        if invalid_content:
            messages.append(ChatMessage(role="assistant", content=invalid_content))
        messages.append(
            ChatMessage(
                role="user",
                content=(
                    f"The previous response did not satisfy the requested JSON format: {validation_error}. Return only corrected JSON."
                ),
            )
        )
    return replace(request, messages=messages, response_format=None)


def response_format_instruction(response_format: ChatResponseFormat) -> str:
    lines = [
        "Return exactly one valid JSON value.",
        "Do not include prose, markdown fences, comments, or extra text.",
    ]
    if response_format.type == "json_object":
        lines.append("The top-level JSON value must be an object.")
    elif response_format.type == "json_schema":
        if response_format.name:
            lines.append(f"Schema name: {response_format.name}.")
        if response_format.description:
            lines.append(response_format.description)
        lines.append("The JSON must validate against this JSON Schema:")
        lines.append(
            msgspec.json.encode(
                response_format.schema or {},
                order="deterministic",
            ).decode()
        )
    return "\n".join(lines)


def _validated_response_format_content(
    content: str | None,
    response_format: ChatResponseFormat,
) -> tuple[str | None, str | None]:
    if not content or not content.strip():
        return None, "response content is empty"
    try:
        value = parse_json_value_with_repair(content)
    except JSONInvalidError as exc:
        return None, f"invalid JSON: {exc}"
    if response_format.type == "json_object" and not isinstance(value, dict):
        return None, "top-level JSON value is not an object"
    if response_format.type == "json_schema":
        validation_error = _json_schema_validation_error(value, response_format)
        if validation_error is not None:
            return None, validation_error
    return normalize_json_text_with_repair(content), None


def response_format_validation_error(
    content: str | None,
    response_format: ChatResponseFormat,
) -> str | None:
    _, validation_error = _validated_response_format_content(content, response_format)
    return validation_error


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
