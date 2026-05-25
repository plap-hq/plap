from __future__ import annotations

from collections.abc import Callable

from plap.llms.completions.client import Provider
from plap.llms.completions.providers.fireworks import build_fireworks_provider
from plap.llms.completions.providers.openai import (
    CEREBRAS_OPENAI_BASE_URL,
    CROF_OPENAI_BASE_URL,
    GMICLOUD_OPENAI_BASE_URL,
    GROQ_OPENAI_BASE_URL,
    LIGHTNING_OPENAI_BASE_URL,
    NOVITA_OPENAI_BASE_URL,
    QUBRID_OPENAI_BASE_URL,
    build_cerebras_provider,
    build_crof_provider,
    build_gmicloud_provider,
    build_groq_provider,
    build_lightning_provider,
    build_novita_provider,
    build_qubrid_provider,
)
from plap.llms.completions.providers.openrouter import (
    OPENROUTER_OPENAI_BASE_URL,
    OpenRouterProvider,
    build_openrouter_provider,
)


type ProviderBuilder = Callable[..., Provider]


PROVIDER_BUILDERS: dict[str, ProviderBuilder] = {
    "lightning": build_lightning_provider,
    "cerebras": build_cerebras_provider,
    "groq": build_groq_provider,
    "gmicloud": build_gmicloud_provider,
    "novita": build_novita_provider,
    "fireworks": build_fireworks_provider,
    "crof": build_crof_provider,
    "qubrid": build_qubrid_provider,
    "openrouter": build_openrouter_provider,
}


def build_providers(settings) -> dict[str, Provider]:
    providers: dict[str, Provider] = {}

    for slug, build in PROVIDER_BUILDERS.items():
        api_key = settings.llm_api_keys.get(slug)
        if not api_key:
            continue
        providers[f"{slug}/"] = build(api_key=api_key)
    return providers


__all__ = [
    "CEREBRAS_OPENAI_BASE_URL",
    "CROF_OPENAI_BASE_URL",
    "GMICLOUD_OPENAI_BASE_URL",
    "GROQ_OPENAI_BASE_URL",
    "LIGHTNING_OPENAI_BASE_URL",
    "NOVITA_OPENAI_BASE_URL",
    "QUBRID_OPENAI_BASE_URL",
    "OPENROUTER_OPENAI_BASE_URL",
    "OpenRouterProvider",
    "PROVIDER_BUILDERS",
    "build_cerebras_provider",
    "build_crof_provider",
    "build_fireworks_provider",
    "build_gmicloud_provider",
    "build_groq_provider",
    "build_lightning_provider",
    "build_novita_provider",
    "build_openrouter_provider",
    "build_qubrid_provider",
    "build_providers",
]
