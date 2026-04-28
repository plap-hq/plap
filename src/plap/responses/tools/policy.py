from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable
from uuid import UUID

import blake3
import msgspec
from cachetools import LRUCache

from plap.responses.contracts import FunctionTool, SupportedTool, WebSearchTool

type EffectClass = Literal["safe", "mutation", "contextual", "unknown"]
type ToolCallEffectClass = Literal["safe", "mutation", "unknown"]
type ToolSource = Literal["client", "server"]
type _ClassificationL1Key = tuple[bytes, str, str, bytes]


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


type ClassificationL1Cache = MutableMapping[_ClassificationL1Key, ToolClassification]


@dataclass(frozen=True, slots=True)
class ToolCallClassification:
    scope_id: UUID
    signature_hash: bytes
    arguments_hash: bytes
    classifier: str
    classifier_model: str
    prompt_hash: bytes
    effect_class: ToolCallEffectClass
    confidence: float
    rationale: str
    raw_output: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    name: str
    source: ToolSource
    effect_class: EffectClass
    classification: ToolClassification | None = None


@runtime_checkable
class IToolPolicyResolver(Protocol):
    async def resolve(
        self, tools: Sequence[SupportedTool]
    ) -> dict[str, ToolPolicy]: ...


@runtime_checkable
class IToolClassifier(Protocol):
    classifier: str
    classifier_model: str
    prompt_hash: bytes

    async def classify_many(
        self, signatures: list[ToolSignature]
    ) -> dict[bytes, ToolClassification]: ...


class ToolPolicyError(ValueError):
    pass


SERVER_TOOL_POLICIES: dict[str, ToolPolicy] = {
    "web_search": ToolPolicy(
        name="web_search",
        source="server",
        effect_class="safe",
    )
}


def normalize_function_tool(tool: FunctionTool) -> dict[str, object]:
    return {
        "description": tool.description,
        "name": tool.name,
        "parameters": tool.parameters,
        "strict": tool.strict,
        "type": "function",
    }


def function_tool_signature(tool: FunctionTool) -> ToolSignature:
    signature = normalize_function_tool(tool)
    return ToolSignature(
        signature_hash=blake3.blake3(
            msgspec.json.encode(signature, order="deterministic")
        ).digest(),
        signature=signature,
    )


def signature_hash_hex(signature_hash: bytes) -> str:
    return signature_hash.hex()


def get_server_tool_policy(name: str) -> ToolPolicy | None:
    return SERVER_TOOL_POLICIES.get(name)


class CachedToolPolicyResolver(IToolPolicyResolver):
    def __init__(
        self,
        repository: Any,
        classifier: IToolClassifier,
        *,
        l1_maxsize: int = 4096,
        classification_l1: ClassificationL1Cache | None = None,
    ) -> None:
        self._repository = repository
        self._classifier = classifier
        self._classifications_l1 = classification_l1 or LRUCache(maxsize=l1_maxsize)

    async def resolve(self, tools: Sequence[SupportedTool]) -> dict[str, ToolPolicy]:
        policies: dict[str, ToolPolicy] = {}
        signatures_by_name: dict[str, bytes] = {}
        client_signatures_by_name: dict[str, ToolSignature] = {}
        for tool in tools:
            if isinstance(tool, WebSearchTool):
                policy = get_server_tool_policy(tool.type)
                if policy is None:
                    raise ToolPolicyError(f"unknown server tool: {tool.type}")
                policies[tool.type] = policy
                continue

            if isinstance(tool, FunctionTool):
                signature = function_tool_signature(tool)
                previous_hash = signatures_by_name.get(tool.name)
                if (
                    previous_hash is not None
                    and previous_hash != signature.signature_hash
                ):
                    raise ToolPolicyError(
                        "duplicate function tool name with different signature: "
                        f"{tool.name}"
                    )
                signatures_by_name[tool.name] = signature.signature_hash
                if (
                    tool.name not in policies
                    and tool.name not in client_signatures_by_name
                ):
                    client_signatures_by_name[tool.name] = signature
        classifications: dict[bytes, ToolClassification] = {}
        l2_signatures: list[ToolSignature] = []
        for signature in client_signatures_by_name.values():
            cached = self._classifications_l1.get(
                self._classification_l1_key(signature.signature_hash)
            )
            if cached is None:
                l2_signatures.append(signature)
                continue
            classifications[signature.signature_hash] = cached
        if l2_signatures:
            await self._repository.get_or_create_signatures(l2_signatures)
            l2_cached = await self._repository.get_classifications(
                [signature.signature_hash for signature in l2_signatures],
                classifier=self._classifier.classifier,
                classifier_model=self._classifier.classifier_model,
                prompt_hash=self._classifier.prompt_hash,
            )
            for classification in l2_cached.values():
                self._classifications_l1[
                    self._classification_l1_key(classification.signature_hash)
                ] = classification
            classifications.update(l2_cached)
            missing = [
                signature
                for signature in l2_signatures
                if signature.signature_hash not in l2_cached
            ]
            if missing:
                new_classifications = await self._classifier.classify_many(missing)
                stored = await self._repository.store_classifications(
                    list(new_classifications.values())
                )
                for classification in stored.values():
                    self._classifications_l1[
                        self._classification_l1_key(classification.signature_hash)
                    ] = classification
                classifications.update(stored)
        for tool_name, signature in client_signatures_by_name.items():
            classification = classifications[signature.signature_hash]
            policies[tool_name] = ToolPolicy(
                name=tool_name,
                source="client",
                effect_class=classification.effect_class,
                classification=classification,
            )
        return policies

    def _classification_l1_key(
        self, signature_hash: bytes
    ) -> _ClassificationL1Key:
        return (
            signature_hash,
            self._classifier.classifier,
            self._classifier.classifier_model,
            self._classifier.prompt_hash,
        )


class StaticToolPolicyResolver(IToolPolicyResolver):
    async def resolve(self, tools: Sequence[SupportedTool]) -> dict[str, ToolPolicy]:
        policies: dict[str, ToolPolicy] = {}
        signatures_by_name: dict[str, bytes] = {}
        for tool in tools:
            if isinstance(tool, WebSearchTool):
                policy = get_server_tool_policy(tool.type)
                if policy is not None:
                    policies[tool.type] = policy
                continue
            if isinstance(tool, FunctionTool):
                signature = function_tool_signature(tool)
                previous_hash = signatures_by_name.get(tool.name)
                if (
                    previous_hash is not None
                    and previous_hash != signature.signature_hash
                ):
                    raise ToolPolicyError(
                        "duplicate function tool name with different signature: "
                        f"{tool.name}"
                    )
                signatures_by_name[tool.name] = signature.signature_hash
                policies.setdefault(
                    tool.name,
                    ToolPolicy(
                        name=tool.name,
                        source="client",
                        effect_class="unknown",
                    ),
                )
        return policies
