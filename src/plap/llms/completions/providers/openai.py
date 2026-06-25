from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)
from openai.resources.chat.completions.completions import AsyncCompletions

from plap.llms.completions.client import Call, Provider, Quirk
from plap.llms.completions.common import close_any_object, close_stream_object, to_data
from plap.llms.completions.errors import (
    ChatCompletionAuthenticationError,
    ChatCompletionContextLengthExceededError,
    ChatCompletionInvalidRequestError,
    ChatCompletionProviderError,
    ChatCompletionRateLimitError,
    ChatCompletionTimeoutError,
    is_context_length_exceeded_code,
    is_context_length_exceeded_error,
)
from plap.llms.completions.providers._transport import (
    log_transport_error,
    timeout_config,
    timeout_error_message,
)
from plap.llms.completions.quirks import (
    Drop,
    DropIf,
    DropMessageField,
    DropToolFunctionField,
    EnsureAssistantReasoningContent,
    ExtraBody,
    ForceNamedToolChoice,
    ForceRequiredToolChoice,
    Move,
    MoveMessageField,
    MoveOutput,
    Only,
    RateLimit,
    RejectMessageField,
    RejectResponseFormat,
    Set,
    SystemRole,
)

logger = structlog.stdlib.get_logger(__name__)
OPENAI_SDK_TOP_LEVEL_FIELDS = frozenset(
    name
    for name in inspect.signature(AsyncCompletions.create).parameters
    if name not in {"self", "extra_headers", "extra_query", "extra_body", "timeout"}
)


def _normalize_openai_sdk_body(body: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    extra_body = dict(body.get("extra_body")) if isinstance(body.get("extra_body"), dict) else {}
    for key, value in body.items():
        if key == "extra_body":
            continue
        if key in OPENAI_SDK_TOP_LEVEL_FIELDS:
            normalized[key] = value
            continue
        extra_body[key] = value
    if extra_body:
        normalized["extra_body"] = extra_body
    return normalized


def _openai_context_length_exceeded_error(exc: BadRequestError) -> ChatCompletionContextLengthExceededError | None:
    body = exc.body if isinstance(exc.body, dict) else None
    error = body.get("error") if isinstance(body, dict) and isinstance(body.get("error"), dict) else None
    code_candidates = (
        error.get("code") if isinstance(error, dict) else None,
        error.get("type") if isinstance(error, dict) else None,
        body.get("code") if isinstance(body, dict) else None,
        body.get("type") if isinstance(body, dict) else None,
    )
    if any(isinstance(code, str) and is_context_length_exceeded_code(code) for code in code_candidates):
        return ChatCompletionContextLengthExceededError(str(exc))
    if is_context_length_exceeded_error(exc):
        return ChatCompletionContextLengthExceededError(str(exc))
    return None


def _log_transport_error(*, provider: str, client: Any, call: Call, exc: Exception, streaming: bool) -> None:
    timeout = timeout_config(getattr(client, "timeout", None))
    base_url = getattr(client, "base_url", None)
    log_transport_error(
        logger=logger,
        provider=provider,
        base_url=str(base_url) if base_url is not None else None,
        client_max_retries=getattr(client, "max_retries", None),
        call=call,
        exc=exc,
        streaming=streaming,
        should_log=isinstance(exc, (APITimeoutError, APIConnectionError, httpx.RequestError)),
        extra_context={
            "timeout_connect_seconds": timeout.get("connect") if timeout is not None else None,
            "timeout_read_seconds": timeout.get("read") if timeout is not None else None,
            "timeout_write_seconds": timeout.get("write") if timeout is not None else None,
            "timeout_pool_seconds": timeout.get("pool") if timeout is not None else None,
        },
    )


def normalize_openai_error(exc: Exception) -> ChatCompletionProviderError:
    if isinstance(exc, ChatCompletionProviderError):
        return exc
    if isinstance(exc, (APITimeoutError, httpx.TimeoutException)):
        return ChatCompletionTimeoutError(timeout_error_message(exc))
    if isinstance(exc, AuthenticationError):
        return ChatCompletionAuthenticationError(str(exc))
    if isinstance(exc, RateLimitError):
        return ChatCompletionRateLimitError(str(exc))
    if isinstance(exc, BadRequestError):
        context_length_error = _openai_context_length_exceeded_error(exc)
        if context_length_error is not None:
            return context_length_error
        return ChatCompletionInvalidRequestError(str(exc))
    if isinstance(exc, APIStatusError):
        return ChatCompletionProviderError(str(exc))
    return ChatCompletionProviderError(str(exc))


class OpenAIProvider(Provider):
    def __init__(
        self,
        *,
        name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        quirks=(),
        models: dict[str, tuple] | None = None,
    ) -> None:
        super().__init__(name=name, quirks=quirks, models=models)
        # Keep retry ownership in our router so provider fallback and timing stay explicit.
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)

    async def complete(self, call: Call) -> dict[str, Any]:
        try:
            response = await self._client.chat.completions.create(**_normalize_openai_sdk_body(call.body))
        except Exception as exc:
            _log_transport_error(provider=self.name, client=self._client, call=call, exc=exc, streaming=False)
            raise normalize_openai_error(exc) from exc
        return to_data(response)

    def stream(self, call: Call) -> AsyncIterator[dict[str, Any]]:
        async def run() -> AsyncIterator[dict[str, Any]]:
            stream: Any | None = None
            try:
                stream = await self._client.chat.completions.create(**_normalize_openai_sdk_body(call.body))
                async for chunk in stream:
                    yield to_data(chunk)
            except Exception as exc:
                _log_transport_error(provider=self.name, client=self._client, call=call, exc=exc, streaming=True)
                raise normalize_openai_error(exc) from exc
            finally:
                if stream is not None:
                    await close_stream_object(stream)

        return run()

    async def aclose(self) -> None:
        await close_any_object(self._client)


LIGHTNING_OPENAI_BASE_URL = "https://lightning.ai/api/v1"
GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
CEREBRAS_OPENAI_BASE_URL = "https://api.cerebras.ai/v1"
GMICLOUD_OPENAI_BASE_URL = "https://api.gmi-serving.com/v1"
NOVITA_OPENAI_BASE_URL = "https://api.novita.ai/openai"
CROF_OPENAI_BASE_URL = "https://crof.ai/v1"
QUBRID_OPENAI_BASE_URL = "https://platform.qubrid.com/v1"

LIGHTNING_FIELDS = (
    "model",
    "messages",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "stop",
    "seed",
    "n",
    "logprobs",
    "top_logprobs",
    "reasoning_effort",
    "user",
    "metadata",
)
NOVITA_FIELDS = (
    "model",
    "messages",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "stop",
    "seed",
    "n",
    "logprobs",
    "reasoning_effort",
)
GROQ_FIELDS = (
    "model",
    "messages",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "seed",
    "n",
    "reasoning_effort",
    "user",
    "service_tier",
)
CEREBRAS_FIELDS = (
    "model",
    "messages",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "stop",
    "seed",
    "n",
    "reasoning_effort",
    "user",
    "metadata",
)
GMICLOUD_FIELDS = (
    "model",
    "messages",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "stop",
    "seed",
    "n",
    "reasoning_effort",
)
CROF_FIELDS = (
    "model",
    "messages",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "stop",
    "seed",
    "n",
    "reasoning_effort",
)
QUBRID_FIELDS = (
    "model",
    "messages",
    "stream",
    "stream_options",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "stop",
    "seed",
    "n",
    "reasoning_effort",
    "user",
    "metadata",
)


def _request_limits(*windows: tuple[int, float]) -> tuple[Quirk, ...]:
    return tuple(RateLimit(limit=limit, window_seconds=window_seconds) for limit, window_seconds in windows)


GROQ_NO_PARALLEL_TOOL_CALLS: tuple[Quirk, ...] = (Set("parallel_tool_calls", False),)
GROQ_INCLUDE_REASONING: tuple[Quirk, ...] = (ExtraBody({"include_reasoning": True}),)
LIGHTNING_MODELS: dict[str, tuple[Quirk, ...]] = {
    "lightning-ai/gpt-oss-20b": _request_limits((15, 60)),
    "lightning-ai/gpt-oss-120b": (RejectResponseFormat(), *_request_limits((15, 60))),
    "lightning-ai/nvidia-nemotron-3-super-120b-a12b": (),
    "lightning-ai/nvidia-nemotron-3-nano-omni-30b-a3b": (),
}
GROQ_MODELS: dict[str, tuple[Quirk, ...]] = {
    "openai/gpt-oss-20b": (*GROQ_NO_PARALLEL_TOOL_CALLS, *GROQ_INCLUDE_REASONING, *_request_limits((30, 60), (1000, 86400))),
    "openai/gpt-oss-safeguard-20b": (
        *GROQ_NO_PARALLEL_TOOL_CALLS,
        *GROQ_INCLUDE_REASONING,
        *_request_limits((30, 60), (1000, 86400)),
    ),
    "openai/gpt-oss-120b": (*GROQ_NO_PARALLEL_TOOL_CALLS, *GROQ_INCLUDE_REASONING, *_request_limits((30, 60), (1000, 86400))),
    "meta-llama/llama-4-scout-17b-16e-instruct": _request_limits((30, 60), (1000, 86400)),
    "qwen/qwen3-32b": (RejectResponseFormat("json_schema"), *GROQ_INCLUDE_REASONING, *_request_limits((60, 60), (1000, 86400))),
    "llama-3.3-70b-versatile": (RejectResponseFormat("json_schema"), *_request_limits((30, 60), (1000, 86400))),
    "llama-3.1-8b-instant": (RejectResponseFormat("json_schema"), *_request_limits((30, 60), (14400, 86400))),
}
CEREBRAS_MODELS: dict[str, tuple[Quirk, ...]] = {
    "gpt-oss-120b": _request_limits((5, 60)),
    "zai-glm-4.7": (ExtraBody({"clear_thinking": False}), *_request_limits((5, 60))),
}
NOVITA_MODELS: dict[str, tuple[Quirk, ...]] = {
    "deepseek/deepseek-v4-flash": (ForceRequiredToolChoice(),),
    "openai/gpt-oss-20b": (),
    "openai/gpt-oss-120b": (),
}
GMICLOUD_MODELS: dict[str, tuple[Quirk, ...]] = {
    "XiaomiMiMo/MiMo-V2.5": (),
    "XiaomiMiMo/MiMo-V2.5-Pro": (),
    "openai/gpt-oss-20b": (),
    "openai/gpt-oss-120b": (),
}
CROF_MODELS: dict[str, tuple[Quirk, ...]] = {
    "deepseek-v4-flash": (RejectResponseFormat("json_schema"),),
    "deepseek-v4-pro": (RejectResponseFormat("json_schema"),),
    "deepseek-v3.2": (RejectResponseFormat(),),
    "gemma-4-31b-it": (RejectResponseFormat(),),
    "greg": (RejectResponseFormat("json_object"),),
    "glm-4.7": (RejectResponseFormat(),),
    "glm-4.7-flash": (RejectResponseFormat(),),
    "glm-5": (RejectResponseFormat(),),
    "glm-5.1": (RejectResponseFormat("json_schema"),),
    "kimi-k2.5": (RejectResponseFormat(),),
    "kimi-k2.5-lightning": (),
    "kimi-k2.6": (RejectResponseFormat("json_schema"),),
    "minimax-m2.5": (RejectResponseFormat(),),
    "mimo-v2.5-pro": (RejectResponseFormat("json_schema"),),
    "qwen3.6-27b": (RejectResponseFormat(),),
    "qwen3.5-397b-a17b": (RejectResponseFormat(),),
    "qwen3.5-9b": (),
}
QUBRID_MODELS: dict[str, tuple[Quirk, ...]] = {
    "deepseek-ai/DeepSeek-V4-Flash": (ForceRequiredToolChoice(),),
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8": (
        Drop("parallel_tool_calls"),
        DropToolFunctionField("strict"),
        ForceNamedToolChoice(),
    ),
}


def build_lightning_provider(*, api_key: str) -> Provider:
    return OpenAIProvider(
        name="lightning",
        api_key=api_key,
        base_url=LIGHTNING_OPENAI_BASE_URL,
        quirks=(
            SystemRole(),
            Only(*LIGHTNING_FIELDS),
            RejectMessageField(("file", "file_id"), content_type="file"),
            RejectMessageField(("file", "file_url"), content_type="file"),
        ),
        models=LIGHTNING_MODELS,
    )


def build_gmicloud_provider(*, api_key: str) -> Provider:
    return OpenAIProvider(
        name="gmicloud",
        api_key=api_key,
        base_url=GMICLOUD_OPENAI_BASE_URL,
        quirks=(
            SystemRole(),
            Only(*GMICLOUD_FIELDS),
            RejectMessageField(("file", "file_id"), content_type="file"),
            RejectMessageField(("file", "file_url"), content_type="file"),
            EnsureAssistantReasoningContent(),
            Move("max_completion_tokens", "max_tokens"),
            DropIf("reasoning_effort", "none"),  # gmi does not have a real thinking disable switch
            ExtraBody({"context_length_exceeded_behavior": "error"}),
        ),
        models=GMICLOUD_MODELS,
    )


def build_groq_provider(*, api_key: str) -> Provider:
    return OpenAIProvider(
        name="groq",
        api_key=api_key,
        base_url=GROQ_OPENAI_BASE_URL,
        quirks=(
            SystemRole(),
            Only(*GROQ_FIELDS),
            RejectMessageField(("file", "file_id"), content_type="file"),
            RejectMessageField(("file", "file_url"), content_type="file"),
            DropMessageField("name"),
            MoveMessageField("reasoning_content", "reasoning", role="assistant"),
            MoveOutput("reasoning", "reasoning_content"),
        ),
        models=GROQ_MODELS,
    )


def build_cerebras_provider(*, api_key: str) -> Provider:
    return OpenAIProvider(
        name="cerebras",
        api_key=api_key,
        base_url=CEREBRAS_OPENAI_BASE_URL,
        quirks=(
            SystemRole(),
            Only(*CEREBRAS_FIELDS),
            RejectMessageField(("file", "file_id"), content_type="file"),
            RejectMessageField(("file", "file_url"), content_type="file"),
            MoveMessageField("reasoning_content", "reasoning", role="assistant"),
            MoveOutput("reasoning", "reasoning_content"),
        ),
        models=CEREBRAS_MODELS,
    )


def build_novita_provider(*, api_key: str) -> Provider:
    return OpenAIProvider(
        name="novita",
        api_key=api_key,
        base_url=NOVITA_OPENAI_BASE_URL,
        quirks=(
            SystemRole(),
            Only(*NOVITA_FIELDS),
            RejectMessageField(("file", "file_id"), content_type="file"),
            RejectMessageField(("file", "file_url"), content_type="file"),
            Move("max_completion_tokens", "max_tokens"),
        ),
        models=NOVITA_MODELS,
    )


def build_crof_provider(*, api_key: str) -> Provider:
    return OpenAIProvider(
        name="crof",
        api_key=api_key,
        base_url=CROF_OPENAI_BASE_URL,
        quirks=(
            Only(*CROF_FIELDS),
            RejectMessageField(("file", "file_id"), content_type="file"),
            RejectMessageField(("file", "file_url"), content_type="file"),
            Move("max_completion_tokens", "max_tokens"),
        ),
        models=CROF_MODELS,
    )


def build_qubrid_provider(*, api_key: str) -> Provider:
    return OpenAIProvider(
        name="qubrid",
        api_key=api_key,
        base_url=QUBRID_OPENAI_BASE_URL,
        quirks=(
            SystemRole(),
            Only(*QUBRID_FIELDS),
            RejectMessageField(("file", "file_id"), content_type="file"),
            RejectMessageField(("file", "file_url"), content_type="file"),
        ),
        models=QUBRID_MODELS,
    )


__all__ = [
    "CEREBRAS_OPENAI_BASE_URL",
    "CROF_OPENAI_BASE_URL",
    "GMICLOUD_OPENAI_BASE_URL",
    "GROQ_OPENAI_BASE_URL",
    "LIGHTNING_OPENAI_BASE_URL",
    "NOVITA_OPENAI_BASE_URL",
    "OPENAI_SDK_TOP_LEVEL_FIELDS",
    "QUBRID_OPENAI_BASE_URL",
    "OpenAIProvider",
    "build_cerebras_provider",
    "build_crof_provider",
    "build_gmicloud_provider",
    "build_groq_provider",
    "build_lightning_provider",
    "build_novita_provider",
    "build_qubrid_provider",
    "normalize_openai_error",
]
