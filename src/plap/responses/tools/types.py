from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

type EffectClass = Literal["safe", "mutation", "unknown"]
type ToolSource = Literal["client", "server"]


@dataclass(frozen=True, slots=True)
class ToolSignature:
    signature_hash: bytes
    signature: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolClassification:
    signature_hash: bytes
    classifier: str
    classifier_model: str
    prompt_hash: bytes
    effect_class: EffectClass
    confidence: float
    rationale: str
    raw_output: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    name: str
    source: ToolSource
    effect_class: EffectClass
    classification: ToolClassification | None = None


class ToolPolicyResolver(Protocol):
    async def resolve(self, tools: object) -> dict[str, ToolPolicy]: ...
