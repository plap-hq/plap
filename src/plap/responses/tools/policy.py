from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from uuid import UUID

import blake3
import msgspec

from plap.responses.contracts import FunctionTool, SupportedTool, WebSearchTool

type EffectClass = Literal["safe", "mutation", "contextual", "unknown"]
type ToolCallEffectClass = Literal["safe", "mutation", "unknown"]
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


class ToolPolicyResolver(Protocol):
    async def resolve(self, tools: object) -> dict[str, ToolPolicy]: ...


class ToolClassifier(Protocol):
    classifier: str
    classifier_model: str
    prompt_hash: bytes

    async def classify_many(
        self, signatures: list[ToolSignature]
    ) -> dict[bytes, ToolClassification]: ...


class ToolClassificationCache(Protocol):
    async def get_or_create_signatures(
        self, signatures: list[ToolSignature]
    ) -> list[ToolSignature]: ...

    async def get_classifications(
        self,
        signature_hashes: list[bytes],
        *,
        classifier: str,
        classifier_model: str,
        prompt_hash: bytes,
    ) -> dict[bytes, ToolClassification]: ...

    async def store_classifications(
        self, classifications: list[ToolClassification]
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


class CachedToolPolicyResolver:
    def __init__(
        self,
        repository: ToolClassificationCache,
        classifier: ToolClassifier,
    ) -> None:
        self._repository = repository
        self._classifier = classifier

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
        signatures = list(client_signatures_by_name.values())
        await self._repository.get_or_create_signatures(signatures)
        cached = await self._repository.get_classifications(
            [signature.signature_hash for signature in signatures],
            classifier=self._classifier.classifier,
            classifier_model=self._classifier.classifier_model,
            prompt_hash=self._classifier.prompt_hash,
        )
        missing = [
            signature
            for signature in signatures
            if signature.signature_hash not in cached
        ]
        stored: dict[bytes, ToolClassification] = {}
        if missing:
            new_classifications = await self._classifier.classify_many(missing)
            stored = await self._repository.store_classifications(
                list(new_classifications.values())
            )
        classifications = {**cached, **stored}
        for tool_name, signature in client_signatures_by_name.items():
            classification = classifications[signature.signature_hash]
            policies[tool_name] = ToolPolicy(
                name=tool_name,
                source="client",
                effect_class=classification.effect_class,
                classification=classification,
            )
        return policies


class StaticToolPolicyResolver:
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
