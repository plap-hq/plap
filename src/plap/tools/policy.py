from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import blake3
import msgspec
import structlog
from cachetools import LRUCache

from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.llms.completions.chat import ChatToolCall
from plap.llms.json import decode_json_object_with_error
from plap.responses.contracts import FunctionTool

logger = structlog.stdlib.get_logger(__name__)


def _invalid_tool_arguments_error(*, reason: str, private_message: str, cause: BaseException | None = None) -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code="invalid_tool_arguments",
            message="Tool call arguments must be a valid JSON object.",
            param="input",
        ),
        private=PrivateError(
            event="tool.policy.invalid_request",
            reason=reason,
            message=private_message,
            level=ErrorLevel.WARNING,
            cause=cause,
        ),
    )


def _duplicate_tool_signature_error(tool_name: str) -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code="invalid_tool_definition",
            message=f"Tool '{tool_name}' is defined more than once with conflicting schemas.",
            param="input",
        ),
        private=PrivateError(
            event="tool.policy.invalid_request",
            reason="duplicate_function_tool_name",
            message=f"duplicate function tool name with different signature: {tool_name}",
            level=ErrorLevel.WARNING,
            context={"tool_name": tool_name},
        ),
    )


class EffectClass(StrEnum):
    SAFE = "safe"
    VISIBLE = "visible"
    MUTATION = "mutation"
    CONTEXTUAL = "contextual"


class ToolCallEffectClass(StrEnum):
    SAFE = "safe"
    VISIBLE = "visible"
    MUTATION = "mutation"
    UNKNOWN = "unknown"


class ToolSource(StrEnum):
    CLIENT = "client"
    SERVER = "server"


type ResolvedEffectClass = EffectClass | ToolCallEffectClass


type _ClassificationL1Key = tuple[bytes, str, str, bytes]
type _ToolCallClassificationL1Key = tuple[bytes, bytes, str, str, bytes]


def _resolved_effect_class(value: object) -> ResolvedEffectClass:
    if value == ToolCallEffectClass.UNKNOWN:
        return ToolCallEffectClass.UNKNOWN
    return EffectClass(value)


@dataclass(frozen=True, slots=True)
class ToolSignature:
    signature: dict[str, Any]

    @property
    def signature_hash(self) -> bytes:
        return blake3.blake3(msgspec.json.encode(self.signature, order="deterministic")).digest()


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
    persistable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "effect_class", EffectClass(self.effect_class))


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
    persistable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "effect_class", ToolCallEffectClass(self.effect_class))


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
    effect_class: ResolvedEffectClass
    classification: ToolClassification | ToolCallClassification | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", ToolSource(self.source))
        object.__setattr__(self, "effect_class", _resolved_effect_class(self.effect_class))


@runtime_checkable
class IToolPolicyResolver(Protocol):
    async def resolve(self, tools: Sequence[FunctionTool]) -> dict[str, ToolPolicy]: ...


@runtime_checkable
class IToolClassifier(Protocol):
    classifier: str
    classifier_model: str
    classifier_cache_model: str
    prompt_hash: bytes

    async def classify_many(self, signatures: list[ToolSignature]) -> dict[bytes, ToolClassification]: ...


@runtime_checkable
class IToolCallClassifier(Protocol):
    classifier: str
    classifier_model: str
    classifier_cache_model: str
    prompt_hash: bytes

    async def classify_many(self, calls: list[ToolCallSignature]) -> dict[tuple[bytes, bytes], ToolCallClassification]: ...


@runtime_checkable
class IToolCallPolicyResolver(Protocol):
    async def resolve(self, calls: Sequence[ToolCall]) -> tuple[ToolPolicy, ...]: ...


async def resolve_tool_call_policies(
    calls: Sequence[ChatToolCall],
    *,
    tools: Mapping[str, FunctionTool],
    tool_policies: Mapping[str, ToolPolicy],
    resolver: IToolCallPolicyResolver,
) -> tuple[ToolPolicy, ...]:
    if not calls:
        return ()

    calls_to_resolve: list[ToolCall] = []
    for call in calls:
        policy = tool_policies.get(call.name)
        if policy is None:
            raise ValueError(f"unknown tool call: {call.name}")

        tool = tools.get(call.name)
        if tool is None:
            raise ValueError(f"unknown tool definition for call: {call.name}")

        calls_to_resolve.append(
            ToolCall(
                tool=tool,
                policy=policy,
                arguments=call.arguments,
            )
        )

    return await resolver.resolve(calls_to_resolve)


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


def _strict_json_object(arguments: str) -> dict[str, Any]:
    decoded, error = decode_json_object_with_error(arguments)
    if error is not None:
        raise _invalid_tool_arguments_error(
            reason="tool_arguments_invalid_json",
            private_message="function call arguments must be valid JSON",
            cause=error,
        ) from error
    if decoded is None:
        raise _invalid_tool_arguments_error(
            reason="tool_arguments_not_object",
            private_message="function call arguments must be a JSON object",
        )
    return decoded


def canonical_tool_arguments(arguments: str) -> dict[str, Any]:
    return _strict_json_object(arguments)


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
    return blake3.blake3(msgspec.json.encode(arguments, order="deterministic")).digest()


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
        self._classifications_l1 = classification_l1 if classification_l1 is not None else LRUCache(maxsize=l1_maxsize)

    async def resolve(self, tools: Sequence[FunctionTool]) -> dict[str, ToolPolicy]:
        policies: dict[str, ToolPolicy] = {}
        signatures_by_name: dict[str, bytes] = {}
        client_signatures_by_name: dict[str, ToolSignature] = {}
        classification_sources: dict[bytes, str] = {}
        for tool in tools:
            signature = function_tool_signature(tool)
            previous_hash = signatures_by_name.get(tool.name)
            if previous_hash is not None and previous_hash != signature.signature_hash:
                raise _duplicate_tool_signature_error(tool.name)
            signatures_by_name[tool.name] = signature.signature_hash
            if tool.name not in client_signatures_by_name:
                client_signatures_by_name[tool.name] = signature
        classifications: dict[bytes, ToolClassification] = {}
        l2_signatures: list[ToolSignature] = []
        for signature in client_signatures_by_name.values():
            cached = self._classifications_l1.get(self._classification_l1_key(signature.signature_hash))
            if cached is None:
                l2_signatures.append(signature)
                continue
            classifications[signature.signature_hash] = cached
            classification_sources[signature.signature_hash] = "l1"
        if l2_signatures:
            await self._repository.get_or_create_signatures(l2_signatures)
            l2_cached = await self._repository.get_classifications(
                [signature.signature_hash for signature in l2_signatures],
                classifier=self._classifier.classifier,
                classifier_model=self._classifier.classifier_cache_model,
                prompt_hash=self._classifier.prompt_hash,
            )
            for classification in l2_cached.values():
                self._classifications_l1[self._classification_l1_key(classification.signature_hash)] = classification
                classification_sources[classification.signature_hash] = "l2"
            classifications.update(l2_cached)
            missing = [signature for signature in l2_signatures if signature.signature_hash not in l2_cached]
            if missing:
                new_classifications = await self._classifier.classify_many(missing)
                persistable = [classification for classification in new_classifications.values() if classification.persistable]
                fallback = [classification for classification in new_classifications.values() if not classification.persistable]
                stored = await self._repository.store_classifications(persistable) if persistable else {}
                for classification in stored.values():
                    self._classifications_l1[self._classification_l1_key(classification.signature_hash)] = classification
                    classification_sources[classification.signature_hash] = "fresh"
                classifications.update(stored)
                for classification in fallback:
                    classification_sources[classification.signature_hash] = "fallback"
                    classifications[classification.signature_hash] = classification
        for tool_name, signature in client_signatures_by_name.items():
            classification = classifications[signature.signature_hash]
            policies[tool_name] = ToolPolicy(
                name=tool_name,
                source="client",
                effect_class=classification.effect_class,
                classification=classification,
            )
            source = classification_sources.get(signature.signature_hash, "unknown")
            logger.info(
                "tool.policy.resolved",
                classifier=classification.classifier,
                classifier_model=classification.classifier_model,
                confidence=classification.confidence,
                effect_class=classification.effect_class,
                source=source,
                tool_name=tool_name,
            )
            logger.bind(log_channel="payload").info(
                "tool.policy.resolved.payload",
                raw_output=classification.raw_output,
                signature=signature.signature,
                source=source,
                tool_name=tool_name,
            )
        return policies

    def _classification_l1_key(self, signature_hash: bytes) -> _ClassificationL1Key:
        return (
            signature_hash,
            self._classifier.classifier,
            self._classifier.classifier_cache_model,
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
        self._classifications_l1 = classification_l1 if classification_l1 is not None else LRUCache(maxsize=l1_maxsize)

    async def resolve(self, calls: Sequence[ToolCall]) -> tuple[ToolPolicy, ...]:
        resolved: list[ToolPolicy | None] = []
        contextual_by_index: dict[int, ToolCallSignature] = {}
        contextual_by_key: dict[tuple[bytes, bytes], ToolCallSignature] = {}
        classifications: dict[tuple[bytes, bytes], ToolCallClassification] = {}
        classification_sources: dict[tuple[bytes, bytes], str] = {}

        for call in calls:
            if call.policy.effect_class != "contextual":
                resolved.append(call.policy)
                logger.info("tool.call_policy.preclassified", effect_class=call.policy.effect_class, tool_name=call.tool.name)
                logger.bind(log_channel="payload").info(
                    "tool.call_policy.preclassified.payload",
                    arguments=call.arguments,
                    tool=normalize_function_tool(call.tool),
                    tool_name=call.tool.name,
                )
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
                classification_sources[call_signature.classification_key] = "l1"
                continue
            contextual_by_index[index] = call_signature
            contextual_by_key.setdefault(
                call_signature.classification_key,
                call_signature,
            )

        if contextual_by_key:
            signatures_by_hash = {call_signature.signature_hash: call_signature.signature for call_signature in contextual_by_key.values()}
            await self._repository.get_or_create_signatures(list(signatures_by_hash.values()))
            l2_cached = await self._repository.get_tool_call_classifications(
                list(contextual_by_key),
                classifier=self._classifier.classifier,
                classifier_model=self._classifier.classifier_cache_model,
                prompt_hash=self._classifier.prompt_hash,
            )
            for classification in l2_cached.values():
                self._classifications_l1[
                    self._classification_l1_key(
                        classification.signature_hash,
                        classification.arguments_hash,
                    )
                ] = classification
                classification_sources[(classification.signature_hash, classification.arguments_hash)] = "l2"
            classifications.update(l2_cached)

            missing = [call_signature for key, call_signature in contextual_by_key.items() if key not in l2_cached]
            if missing:
                new_classifications = await self._classifier.classify_many(missing)
                persistable = [classification for classification in new_classifications.values() if classification.persistable]
                fallback = [classification for classification in new_classifications.values() if not classification.persistable]
                stored = await self._repository.store_tool_call_classifications(persistable) if persistable else {}
                for classification in stored.values():
                    self._classifications_l1[
                        self._classification_l1_key(
                            classification.signature_hash,
                            classification.arguments_hash,
                        )
                    ] = classification
                    classification_sources[(classification.signature_hash, classification.arguments_hash)] = "fresh"
                classifications.update(stored)
                for classification in fallback:
                    classification_sources[(classification.signature_hash, classification.arguments_hash)] = "fallback"
                    classifications[(classification.signature_hash, classification.arguments_hash)] = classification

        for index, call_signature in contextual_by_index.items():
            classification = classifications[call_signature.classification_key]
            call = calls[index]
            resolved[index] = ToolPolicy(
                name=call.tool.name,
                source=call.policy.source,
                effect_class=classification.effect_class,
                classification=classification,
            )
            source = classification_sources.get(call_signature.classification_key, "unknown")
            logger.info(
                "tool.call_policy.resolved",
                classifier=classification.classifier,
                classifier_model=classification.classifier_model,
                confidence=classification.confidence,
                effect_class=classification.effect_class,
                source=source,
                tool_name=call.tool.name,
            )
            logger.bind(log_channel="payload").info(
                "tool.call_policy.resolved.payload",
                arguments=call_signature.arguments,
                raw_output=classification.raw_output,
                signature=call_signature.signature.signature,
                source=source,
                tool_name=call.tool.name,
            )

        if any(policy is None for policy in resolved):
            raise RuntimeError("tool call policy resolution did not produce all outputs")
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
            self._classifier.classifier_cache_model,
            self._classifier.prompt_hash,
        )


class StaticToolPolicyResolver(IToolPolicyResolver):
    async def resolve(self, tools: Sequence[FunctionTool]) -> dict[str, ToolPolicy]:
        policies: dict[str, ToolPolicy] = {}
        signatures_by_name: dict[str, bytes] = {}
        for tool in tools:
            signature = function_tool_signature(tool)
            previous_hash = signatures_by_name.get(tool.name)
            if previous_hash is not None and previous_hash != signature.signature_hash:
                raise _duplicate_tool_signature_error(tool.name)
            signatures_by_name[tool.name] = signature.signature_hash
            policies.setdefault(
                tool.name,
                ToolPolicy(
                    name=tool.name,
                    source="client",
                    effect_class="contextual",
                ),
            )
            logger.info("tool.policy.static", effect_class="contextual", tool_name=tool.name)
            logger.bind(log_channel="payload").info(
                "tool.policy.static.payload",
                signature=signature.signature,
                tool_name=tool.name,
            )
        return policies


class StaticToolCallPolicyResolver(IToolCallPolicyResolver):
    async def resolve(self, calls: Sequence[ToolCall]) -> tuple[ToolPolicy, ...]:
        policies: list[ToolPolicy] = []
        for call in calls:
            if call.policy.effect_class != "contextual":
                policies.append(call.policy)
                logger.info("tool.call_policy.preclassified", effect_class=call.policy.effect_class, tool_name=call.tool.name)
                logger.bind(log_channel="payload").info(
                    "tool.call_policy.preclassified.payload",
                    arguments=call.arguments,
                    tool=normalize_function_tool(call.tool),
                    tool_name=call.tool.name,
                )
                continue
            logger.info("tool.call_policy.static", effect_class="unknown", tool_name=call.tool.name)
            logger.bind(log_channel="payload").info(
                "tool.call_policy.static.payload",
                arguments=call.arguments,
                tool=normalize_function_tool(call.tool),
                tool_name=call.tool.name,
            )
            policies.append(
                ToolPolicy(
                    name=call.tool.name,
                    source=call.policy.source,
                    effect_class="unknown",
                )
            )
        return tuple(policies)
