from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openai import (
    APIStatusError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from plap.llms.completions.client import Call, Provider, Quirk
from plap.llms.completions.common import close_stream_object, to_data
from plap.llms.completions.errors import (
    ChatCompletionAuthenticationError,
    ChatCompletionContextLengthExceededError,
    ChatCompletionInvalidRequestError,
    ChatCompletionProviderError,
    ChatCompletionRateLimitError,
    is_context_length_exceeded_code,
    is_context_length_exceeded_error,
)
from plap.llms.completions.quirks import (
    DropMessageName,
    DropIf,
    EnsureAssistantReasoningContent,
    ExtraBody,
    ForceRequiredTool,
    Only,
    RejectResponseFormat,
    Rename,
    RenameOutput,
    Set,
    SystemRole,
)


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


def normalize_openai_error(exc: Exception) -> ChatCompletionProviderError:
    if isinstance(exc, ChatCompletionProviderError):
        return exc
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
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(self, call: Call) -> dict[str, Any]:
        try:
            response = await self._client.chat.completions.create(**call.body)
        except Exception as exc:
            raise normalize_openai_error(exc) from exc
        return to_data(response)

    def stream(self, call: Call) -> AsyncIterator[dict[str, Any]]:
        async def run() -> AsyncIterator[dict[str, Any]]:
            stream: Any | None = None
            try:
                stream = await self._client.chat.completions.create(**call.body)
                async for chunk in stream:
                    yield to_data(chunk)
            except Exception as exc:
                raise normalize_openai_error(exc) from exc
            finally:
                if stream is not None:
                    await close_stream_object(stream)

        return run()

LIGHTNING_OPENAI_BASE_URL = "https://lightning.ai/api/v1"
GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
GMICLOUD_OPENAI_BASE_URL = "https://api.gmi-serving.com/v1"
NOVITA_OPENAI_BASE_URL = "https://api.novita.ai/openai"
CROF_OPENAI_BASE_URL = "https://crof.ai/v1"

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

LIGHTNING_MODELS: dict[str, tuple[Quirk, ...]] = {
    "lightning-ai/gpt-oss-20b": (),
    "lightning-ai/gpt-oss-120b": (RejectResponseFormat(),),
    "lightning-ai/nvidia-nemotron-3-super-120b-a12b": (),
    "lightning-ai/nvidia-nemotron-3-nano-omni-30b-a3b": (),
}
GROQ_NO_PARALLEL_TOOL_CALLS: tuple[Quirk, ...] = (Set("parallel_tool_calls", False),)
GROQ_MODELS: dict[str, tuple[Quirk, ...]] = {
    "openai/gpt-oss-20b": GROQ_NO_PARALLEL_TOOL_CALLS,
    "openai/gpt-oss-safeguard-20b": GROQ_NO_PARALLEL_TOOL_CALLS,
    "openai/gpt-oss-120b": GROQ_NO_PARALLEL_TOOL_CALLS,
    "meta-llama/llama-4-scout-17b-16e-instruct": (),
    "qwen/qwen3-32b": (RejectResponseFormat("json_schema"),),
    "llama-3.3-70b-versatile": (RejectResponseFormat("json_schema"),),
    "llama-3.1-8b-instant": (RejectResponseFormat("json_schema"),),
}
NOVITA_MODELS: dict[str, tuple[Quirk, ...]] = {
    "deepseek/deepseek-v4-flash": (ForceRequiredTool(),),
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
    "deepseek-v4-pro-precision": (RejectResponseFormat("json_schema"),),
    "deepseek-v3.2": (RejectResponseFormat(),),
    "gemma-4-31b-it": (RejectResponseFormat(),),
    "greg": (RejectResponseFormat("json_object"),),
    "glm-4.7": (RejectResponseFormat(),),
    "glm-4.7-flash": (RejectResponseFormat(),),
    "glm-5": (RejectResponseFormat(),),
    "glm-5.1": (RejectResponseFormat("json_schema"),),
    "glm-5.1-precision": (RejectResponseFormat("json_schema"),),
    "kimi-k2.5": (RejectResponseFormat(),),
    "kimi-k2.5-lightning": (),
    "kimi-k2.6": (RejectResponseFormat("json_schema"),),
    "kimi-k2.6-precision": (RejectResponseFormat("json_schema"),),
    "minimax-m2.5": (RejectResponseFormat(),),
    "mimo-v2.5-pro": (RejectResponseFormat("json_schema"),),
    "mimo-v2.5-pro-precision": (RejectResponseFormat("json_schema"),),
    "qwen3.6-27b": (RejectResponseFormat(),),
    "qwen3.5-397b-a17b": (RejectResponseFormat(),),
    "qwen3.5-9b": (),
}


def build_lightning_provider(settings: Any) -> Provider | None:
    if not settings.llm_lightning_api_key:
        return None
    return OpenAIProvider(
        name="lightning",
        api_key=settings.llm_lightning_api_key,
        base_url=LIGHTNING_OPENAI_BASE_URL,
        quirks=(SystemRole(), Only(*LIGHTNING_FIELDS)),
        models=LIGHTNING_MODELS,
    )


def build_gmicloud_provider(settings: Any) -> Provider | None:
    if not settings.llm_gmicloud_api_key:
        return None
    return OpenAIProvider(
        name="gmicloud",
        api_key=settings.llm_gmicloud_api_key,
        base_url=GMICLOUD_OPENAI_BASE_URL,
        quirks=(
            SystemRole(),
            Only(*GMICLOUD_FIELDS),
            EnsureAssistantReasoningContent(),
            Rename("max_completion_tokens", "max_tokens"),
            DropIf("reasoning_effort", "none"),
            ExtraBody({"context_length_exceeded_behavior": "error"}),
        ),
        models=GMICLOUD_MODELS,
    )


def build_groq_provider(settings: Any) -> Provider | None:
    if not settings.llm_groq_api_key:
        return None
    return OpenAIProvider(
        name="groq",
        api_key=settings.llm_groq_api_key,
        base_url=GROQ_OPENAI_BASE_URL,
        quirks=(
            SystemRole(),
            Only(*GROQ_FIELDS),
            DropMessageName(),
            ExtraBody({"include_reasoning": True}),
            RenameOutput("reasoning", "reasoning_content"),
        ),
        models=GROQ_MODELS,
    )


def build_novita_provider(settings: Any) -> Provider | None:
    if not settings.llm_novita_api_key:
        return None
    return OpenAIProvider(
        name="novita",
        api_key=settings.llm_novita_api_key,
        base_url=NOVITA_OPENAI_BASE_URL,
        quirks=(SystemRole(), Only(*NOVITA_FIELDS), Rename("max_completion_tokens", "max_tokens")),
        models=NOVITA_MODELS,
    )


def build_crof_provider(settings: Any) -> Provider | None:
    if not settings.llm_crof_api_key:
        return None
    return OpenAIProvider(
        name="crof",
        api_key=settings.llm_crof_api_key,
        base_url=CROF_OPENAI_BASE_URL,
        quirks=(SystemRole(), Only(*CROF_FIELDS), Rename("max_completion_tokens", "max_tokens")),
        models=CROF_MODELS,
    )


__all__ = [
    "CROF_OPENAI_BASE_URL",
    "GMICLOUD_OPENAI_BASE_URL",
    "GROQ_OPENAI_BASE_URL",
    "LIGHTNING_OPENAI_BASE_URL",
    "NOVITA_OPENAI_BASE_URL",
    "OpenAIProvider",
    "build_crof_provider",
    "build_gmicloud_provider",
    "build_groq_provider",
    "build_lightning_provider",
    "build_novita_provider",
    "normalize_openai_error",
]
