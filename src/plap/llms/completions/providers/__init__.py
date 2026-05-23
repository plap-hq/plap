from __future__ import annotations

from typing import Any

from plap.llms.completions.client import Provider
from plap.llms.completions.providers.fireworks import build_fireworks_provider
from plap.llms.completions.providers.openai import (
    CROF_OPENAI_BASE_URL,
    GMICLOUD_OPENAI_BASE_URL,
    GROQ_OPENAI_BASE_URL,
    LIGHTNING_OPENAI_BASE_URL,
    NOVITA_OPENAI_BASE_URL,
    build_crof_provider,
    build_gmicloud_provider,
    build_groq_provider,
    build_lightning_provider,
    build_novita_provider,
)
from plap.llms.completions.providers.openrouter import (
    OPENROUTER_OPENAI_BASE_URL,
    OpenRouterProvider,
    build_openrouter_provider,
)


def build_providers(settings: Any) -> dict[str, Provider]:
    providers: dict[str, Provider] = {}

    lightning = build_lightning_provider(settings)
    if lightning is not None:
        providers["lightning/"] = lightning

    groq = build_groq_provider(settings)
    if groq is not None:
        providers["groq/"] = groq

    gmicloud = build_gmicloud_provider(settings)
    if gmicloud is not None:
        providers["gmicloud/"] = gmicloud

    novita = build_novita_provider(settings)
    if novita is not None:
        providers["novita/"] = novita

    fireworks = build_fireworks_provider(settings)
    if fireworks is not None:
        providers["fireworks/"] = fireworks

    crof = build_crof_provider(settings)
    if crof is not None:
        providers["crof/"] = crof

    openrouter = build_openrouter_provider(settings)
    if openrouter is not None:
        providers["openrouter/"] = openrouter

    return providers


__all__ = [
    "CROF_OPENAI_BASE_URL",
    "GMICLOUD_OPENAI_BASE_URL",
    "GROQ_OPENAI_BASE_URL",
    "LIGHTNING_OPENAI_BASE_URL",
    "NOVITA_OPENAI_BASE_URL",
    "OPENROUTER_OPENAI_BASE_URL",
    "OpenRouterProvider",
    "build_crof_provider",
    "build_fireworks_provider",
    "build_gmicloud_provider",
    "build_groq_provider",
    "build_lightning_provider",
    "build_novita_provider",
    "build_openrouter_provider",
    "build_providers",
]
