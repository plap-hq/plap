from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import blake3
import msgspec
from cachetools import LRUCache

from plap.responses.contracts import FunctionTool

type EffectClass = Literal["safe", "visible", "mutation", "contextual", "unknown"]
type ToolCallEffectClass = Literal["safe", "mutation", "unknown"]
type ToolSource = Literal["client", "server"]
type _ClassificationL1Key = tuple[bytes, str, str, bytes]
type _ToolCallClassificationL1Key = tuple[bytes, bytes, str, str, bytes]


@dataclass(frozen=True, slots=True)
class ToolSignature:
    signature: dict[str, Any]

    @property
    def signature_hash(self) -> bytes:
        return blake3.blake3(
            msgspec.json.encode(self.signature, order="deterministic")
        ).digest()


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
    signature_hash: bytes
    arguments_hash: bytes
    classifier: str
    classifier_model: str
    prompt_hash: bytes
    effect_class: ToolCallEffectClass
    confidence: float
    rationale: str
    raw_output: dict[str, Any]


type ToolCallClassificationL1Cache = MutableMapping[
    _ToolCallClassificationL1Key,
    ToolCallClassification,
]


@dataclass(frozen=True, slots=True)
class ToolCall:
    tool: FunctionTool
    policy: ToolPolicy
    arguments: str


@dataclass(frozen=True, slots=True)
class ToolCallSignature:
    signature: ToolSignature
    arguments: dict[str, Any]

    @property
    def signature_hash(self) -> bytes:
        return self.signature.signature_hash

    @property
    def arguments_hash(self) -> bytes:
        return tool_arguments_hash(self.arguments)

    @property
    def classification_key(self) -> tuple[bytes, bytes]:
        return (self.signature_hash, self.arguments_hash)


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    name: str
    source: ToolSource
    effect_class: EffectClass
    classification: ToolClassification | ToolCallClassification | None = None


@runtime_checkable
class IToolPolicyResolver(Protocol):
    async def resolve(
        self, tools: Sequence[FunctionTool]
    ) -> dict[str, ToolPolicy]: ...


@runtime_checkable
class IToolClassifier(Protocol):
    classifier: str
    classifier_model: str
    prompt_hash: bytes

    async def classify_many(
        self, signatures: list[ToolSignature]
    ) -> dict[bytes, ToolClassification]: ...


@runtime_checkable
class IToolCallClassifier(Protocol):
    classifier: str
    classifier_model: str
    prompt_hash: bytes

    async def classify_many(
        self, calls: list[ToolCallSignature]
    ) -> dict[tuple[bytes, bytes], ToolCallClassification]: ...


@runtime_checkable
class IToolCallPolicyResolver(Protocol):
    async def resolve(
        self, calls: Sequence[ToolCall]
    ) -> tuple[ToolPolicy, ...]: ...


class ToolPolicyError(ValueError):
    pass


def normalize_function_tool(tool: FunctionTool) -> dict[str, object]:
    return {
        "description": tool.description,
        "name": tool.name,
        "parameters": tool.parameters,
        "strict": tool.strict,
        "type": "function",
    }


def function_tool_signature(tool: FunctionTool) -> ToolSignature:
    return ToolSignature(signature=normalize_function_tool(tool))


def signature_hash_hex(signature_hash: bytes) -> str:
    return signature_hash.hex()


def canonical_tool_arguments(arguments: str) -> dict[str, Any]:
    try:
        value = msgspec.json.decode(arguments.encode())
    except msgspec.DecodeError as exc:
        raise ToolPolicyError("function call arguments must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ToolPolicyError("function call arguments must be a JSON object")
    return value


def function_tool_call_signature(
    *,
    tool: FunctionTool,
    arguments: str,
) -> ToolCallSignature:
    return ToolCallSignature(
        signature=function_tool_signature(tool),
        arguments=canonical_tool_arguments(arguments),
    )


def tool_arguments_hash(arguments: Any) -> bytes:
    return blake3.blake3(
        msgspec.json.encode(arguments, order="deterministic")
    ).digest()


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
        self._classifications_l1 = (
            classification_l1
            if classification_l1 is not None
            else LRUCache(maxsize=l1_maxsize)
        )

    async def resolve(self, tools: Sequence[FunctionTool]) -> dict[str, ToolPolicy]:
        policies: dict[str, ToolPolicy] = {}
        signatures_by_name: dict[str, bytes] = {}
        client_signatures_by_name: dict[str, ToolSignature] = {}
        for tool in tools:
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
            if tool.name not in client_signatures_by_name:
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


class CachedToolCallPolicyResolver(IToolCallPolicyResolver):
    def __init__(
        self,
        repository: Any,
        classifier: IToolCallClassifier,
        *,
        l1_maxsize: int = 4096,
        classification_l1: ToolCallClassificationL1Cache | None = None,
    ) -> None:
        self._repository = repository
        self._classifier = classifier
        self._classifications_l1 = (
            classification_l1
            if classification_l1 is not None
            else LRUCache(maxsize=l1_maxsize)
        )

    async def resolve(
        self, calls: Sequence[ToolCall]
    ) -> tuple[ToolPolicy, ...]:
        resolved: list[ToolPolicy | None] = []
        contextual_by_index: dict[int, ToolCallSignature] = {}
        contextual_by_key: dict[tuple[bytes, bytes], ToolCallSignature] = {}
        classifications: dict[tuple[bytes, bytes], ToolCallClassification] = {}

        for call in calls:
            if call.policy.effect_class != "contextual":
                resolved.append(call.policy)
                continue

            call_signature = function_tool_call_signature(
                tool=call.tool,
                arguments=call.arguments,
            )
            cached = self._classifications_l1.get(
                self._classification_l1_key(
                    call_signature.signature_hash,
                    call_signature.arguments_hash,
                )
            )
            resolved.append(None)
            index = len(resolved) - 1
            if cached is not None:
                classifications[call_signature.classification_key] = cached
                contextual_by_index[index] = call_signature
                continue
            contextual_by_index[index] = call_signature
            contextual_by_key.setdefault(
                call_signature.classification_key,
                call_signature,
            )

        if contextual_by_key:
            signatures_by_hash = {
                call_signature.signature_hash: call_signature.signature
                for call_signature in contextual_by_key.values()
            }
            await self._repository.get_or_create_signatures(
                list(signatures_by_hash.values())
            )
            l2_cached = await self._repository.get_tool_call_classifications(
                list(contextual_by_key),
                classifier=self._classifier.classifier,
                classifier_model=self._classifier.classifier_model,
                prompt_hash=self._classifier.prompt_hash,
            )
            for classification in l2_cached.values():
                self._classifications_l1[
                    self._classification_l1_key(
                        classification.signature_hash,
                        classification.arguments_hash,
                    )
                ] = classification
            classifications.update(l2_cached)

            missing = [
                call_signature
                for key, call_signature in contextual_by_key.items()
                if key not in l2_cached
            ]
            if missing:
                new_classifications = await self._classifier.classify_many(missing)
                stored = await self._repository.store_tool_call_classifications(
                    list(new_classifications.values())
                )
                for classification in stored.values():
                    self._classifications_l1[
                        self._classification_l1_key(
                            classification.signature_hash,
                            classification.arguments_hash,
                        )
                    ] = classification
                classifications.update(stored)

        for index, call_signature in contextual_by_index.items():
            classification = classifications[call_signature.classification_key]
            call = calls[index]
            resolved[index] = ToolPolicy(
                name=call.tool.name,
                source=call.policy.source,
                effect_class=classification.effect_class,
                classification=classification,
            )

        if any(policy is None for policy in resolved):
            raise RuntimeError(
                "tool call policy resolution did not produce all outputs"
            )
        return tuple(policy for policy in resolved if policy is not None)

    def _classification_l1_key(
        self,
        signature_hash: bytes,
        arguments_hash: bytes,
    ) -> _ToolCallClassificationL1Key:
        return (
            signature_hash,
            arguments_hash,
            self._classifier.classifier,
            self._classifier.classifier_model,
            self._classifier.prompt_hash,
        )


class StaticToolPolicyResolver(IToolPolicyResolver):
    async def resolve(self, tools: Sequence[FunctionTool]) -> dict[str, ToolPolicy]:
        policies: dict[str, ToolPolicy] = {}
        signatures_by_name: dict[str, bytes] = {}
        for tool in tools:
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


class StaticToolCallPolicyResolver(IToolCallPolicyResolver):
    async def resolve(
        self, calls: Sequence[ToolCall]
    ) -> tuple[ToolPolicy, ...]:
        policies: list[ToolPolicy] = []
        for call in calls:
            if call.policy.effect_class != "contextual":
                policies.append(call.policy)
                continue
            policies.append(
                ToolPolicy(
                    name=call.tool.name,
                    source=call.policy.source,
                    effect_class="unknown",
                )
            )
        return tuple(policies)
