from __future__ import annotations

from typing import Any

from plap.llms.client import Provider, Quirk
from plap.llms.errors import ChatCompletionUnsupportedRequestError
from plap.llms.providers.openai import OpenAIProvider
from plap.llms.quirks import ExtraBody, Only, RenameOutput, Set, SystemRole

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
OPENROUTER_FIELDS = (
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
    "top_logprobs",
    "reasoning_effort",
    "user",
    "prompt_cache_key",
    "metadata",
    "service_tier",
    "prediction",
)
OPENROUTER_MODELS: dict[str, tuple[Quirk, ...]] = {
    "deepseek/deepseek-v4-flash": (),
    "meta-llama/llama-3.3-70b-instruct": (),
    "openai/gpt-oss-20b": (),
    "openai/gpt-oss-120b": (),
    "stepfun/step-3.5-flash": (),
}


def _openrouter_model_parts(name: str) -> tuple[str, str, list[str]]:
    segments = name.split(":")
    base = segments[0]
    if not base:
        raise ChatCompletionUnsupportedRequestError(
            f"OpenRouter model {name!r} is missing a base model slug"
        )

    provider_order: list[str] = []
    model_segments = [base]
    for segment in segments[1:]:
        if not segment:
            raise ChatCompletionUnsupportedRequestError(
                f"OpenRouter model {name!r} contains an empty suffix segment"
            )
        if segment in OPENROUTER_SPECIAL_MODEL_SUFFIXES:
            model_segments.append(segment)
            continue
        provider_order.append(segment)
    return base, ":".join(model_segments), provider_order


class OpenRouterProvider(OpenAIProvider):
    def lookup(self, name: str) -> tuple[Quirk, ...]:
        base, wire_model, provider_order = _openrouter_model_parts(name)
        model_quirks = super().lookup(base)
        quirks: list[Quirk] = []
        if wire_model != name:
            quirks.append(Set("model", wire_model))
        if provider_order:
            quirks.append(
                ExtraBody(
                    {
                        "provider": {
                            "order": provider_order,
                            "allow_fallbacks": False,
                        }
                    }
                )
            )
        return (*quirks, *model_quirks)


def build_openrouter_provider(settings: Any) -> Provider | None:
    if not settings.llm_openrouter_api_key:
        return None
    return OpenRouterProvider(
        name="openrouter",
        api_key=settings.llm_openrouter_api_key,
        base_url=OPENROUTER_OPENAI_BASE_URL,
        quirks=(SystemRole(), Only(*OPENROUTER_FIELDS), RenameOutput("reasoning", "reasoning_content")),
        models=OPENROUTER_MODELS,
    )


__all__ = [
    "OPENROUTER_OPENAI_BASE_URL",
    "OpenRouterProvider",
    "build_openrouter_provider",
]
