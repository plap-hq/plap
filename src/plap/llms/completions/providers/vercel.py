from __future__ import annotations

from plap.llms.completions.client import Provider, Quirk
from plap.llms.completions.errors import ChatCompletionUnsupportedRequestError
from plap.llms.completions.providers.openai import OpenAIProvider
from plap.llms.completions.quirks import (
    ExtraBody,
    Move,
    MoveMessageField,
    MoveOutput,
    Only,
    PromoteOutput,
    Set,
    SystemRole,
)

VERCEL_OPENAI_BASE_URL = "https://ai-gateway.vercel.sh/v1"
VERCEL_FIELDS = (
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
VERCEL_MODELS: dict[str, tuple[Quirk, ...]] = {
    "deepseek/deepseek-v4-flash": (),
    "openai/gpt-oss-20b": (),
    "openai/gpt-oss-120b": (),
    "xiaomi/mimo-v2.5-pro": (),
}


def _vercel_model_parts(name: str) -> tuple[str, list[str]]:
    segments = name.split(":")
    base_model = segments[0]
    if not base_model:
        raise ChatCompletionUnsupportedRequestError(f"Vercel model {name!r} is missing a base model slug")

    provider_order: list[str] = []
    for segment in segments[1:]:
        if not segment:
            raise ChatCompletionUnsupportedRequestError(f"Vercel model {name!r} contains an empty suffix segment")
        provider_order.append(segment)
    return base_model, provider_order


class VercelProvider(OpenAIProvider):
    def lookup(self, name: str) -> tuple[Quirk, ...]:
        base_model, provider_order = _vercel_model_parts(name)
        model_quirks = super().lookup(base_model)
        quirks: list[Quirk] = []
        if base_model != name:
            quirks.append(Set("model", base_model))
        if provider_order:
            quirks.append(
                ExtraBody(
                    {
                        "providerOptions": {
                            "gateway": {
                                "only": provider_order,
                                "order": provider_order,
                            }
                        }
                    }
                )
            )
        return (*quirks, *model_quirks)


def build_vercel_provider(*, api_key: str) -> Provider:
    return VercelProvider(
        name="vercel",
        api_key=api_key,
        base_url=VERCEL_OPENAI_BASE_URL,
        quirks=(
            SystemRole(),
            Only(*VERCEL_FIELDS),
            MoveMessageField(("file", "file_data"), ("file", "data"), content_type="file"),
            Move("max_completion_tokens", "max_tokens"),
            Move("reasoning_effort", "extra_body", "reasoning", "effort"),
            MoveMessageField("reasoning_content", "reasoning", role="assistant"),
            MoveOutput("reasoning", "reasoning_content"),
            PromoteOutput("service_tier", "provider_metadata", "gateway", "serviceTier"),
        ),
        models=VERCEL_MODELS,
    )


__all__ = [
    "VERCEL_OPENAI_BASE_URL",
    "VercelProvider",
    "build_vercel_provider",
]
