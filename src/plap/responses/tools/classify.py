from __future__ import annotations

from dataclasses import asdict
from typing import Any

import anyio
import blake3
import msgspec
import structlog

from plap.llms.chat import (
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatFunctionTool,
    ChatMessage,
    ChatTool,
    IChatCompletionClient,
)
from plap.llms.json_utils import parse_json_object_with_repair
from plap.logging import log_debug, log_payload
from plap.responses.tools.policy import (
    EffectClass,
    IToolCallClassifier,
    IToolClassifier,
    ToolCallClassification,
    ToolCallEffectClass,
    ToolCallSignature,
    ToolClassification,
    ToolSignature,
)

logger = structlog.get_logger(__name__)

TOOL_EFFECT_CLASSIFIER_PROMPT = """Classify client-provided tools by side effects.

Call the `classify_tool_effect` tool exactly once.

Definitions:
- safe: read-only or exploratory; no file, client, repo, shell, or external mutation.
- visible: changes visible user-facing state or agent control flow, but does not
  mutate files, repositories, clients, services, or external systems.
- mutation: writes files, runs mutating commands, changes external state, or has
  irreversible side effects.
- contextual: can be safe or mutating depending on call arguments, such as shell,
  SQL, HTTP, or command execution tools. If the signature is too ambiguous or
  incomplete to decide safely from tool-level information alone, classify it as
  contextual so call-time classification can decide later.
"""
TOOL_EFFECT_CLASSIFIER_NAME = "llm_tool_effect_classifier"

TOOL_EFFECT_CLASSIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "effect_class": {
            "type": "string",
            "enum": ["safe", "visible", "mutation", "contextual"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": ["effect_class", "confidence", "rationale"],
    "additionalProperties": False,
}

TOOL_EFFECT_CLASSIFIER_TOOL = ChatTool(
    function=ChatFunctionTool(
        name="classify_tool_effect",
        parameters=TOOL_EFFECT_CLASSIFIER_SCHEMA,
        strict=True,
        description="Effect classification for a client-provided function tool.",
    )
)
TOOL_EFFECT_CLASSIFIER_MAX_TOKENS = 512

TOOL_CALL_EFFECT_CLASSIFIER_PROMPT = """Classify a concrete client tool call.

Call the `classify_tool_call_effect` tool exactly once.

Definitions:
- safe: this specific call is read-only or exploratory and should not mutate files,
  client state, repositories, shells, databases, services, or external systems.
- visible: this specific call changes visible user-facing state or agent control flow,
  but does not mutate files, repositories, clients, services, or external systems.
- mutation: this specific call writes, deletes, runs mutating commands, sends data,
  changes external state, or has irreversible side effects.
- unknown: the concrete arguments are ambiguous, incomplete, malformed, or not enough
  to decide safely.
"""
TOOL_CALL_EFFECT_CLASSIFIER_NAME = "llm_tool_call_effect_classifier"

TOOL_CALL_EFFECT_CLASSIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "effect_class": {
            "type": "string",
            "enum": ["safe", "visible", "mutation", "unknown"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": ["effect_class", "confidence", "rationale"],
    "additionalProperties": False,
}

TOOL_CALL_EFFECT_CLASSIFIER_TOOL = ChatTool(
    function=ChatFunctionTool(
        name="classify_tool_call_effect",
        parameters=TOOL_CALL_EFFECT_CLASSIFIER_SCHEMA,
        strict=True,
        description="Effect classification for a concrete client-provided function tool call.",
    )
)
TOOL_CALL_EFFECT_CLASSIFIER_MAX_TOKENS = 512


def _classifier_result_context(result: ChatCompletionResult) -> dict[str, object]:
    return {
        "finish_reason": result.finish_reason,
        "model": result.model,
        "service_tier": result.service_tier,
        "system_fingerprint": result.system_fingerprint,
        "usage": asdict(result.usage) if result.usage is not None else None,
    }


class LLMToolClassifier(IToolClassifier):
    def __init__(
        self,
        client: IChatCompletionClient,
        *,
        classifier_model: str,
        classifier_cache_model: str,
        classifier: str = TOOL_EFFECT_CLASSIFIER_NAME,
        prompt: str = TOOL_EFFECT_CLASSIFIER_PROMPT,
        max_concurrency: int = 4,
    ) -> None:
        self._client = client
        self.classifier = classifier
        self.classifier_model = classifier_model
        self.classifier_cache_model = classifier_cache_model
        self.prompt_hash = _prompt_hash(
            prompt,
            tool=TOOL_EFFECT_CLASSIFIER_TOOL,
            max_tokens=TOOL_EFFECT_CLASSIFIER_MAX_TOKENS,
        )
        self._prompt = prompt
        self._max_concurrency = max_concurrency

    async def classify(self, signature: ToolSignature) -> ToolClassification:
        return (await self.classify_many([signature]))[signature.signature_hash]

    async def classify_many(self, signatures: list[ToolSignature]) -> dict[bytes, ToolClassification]:
        signatures_by_hash = {signature.signature_hash: signature for signature in signatures}
        if not signatures_by_hash:
            return {}

        limiter = anyio.CapacityLimiter(self._max_concurrency)
        classifications: dict[bytes, ToolClassification] = {}

        async def classify_with_limit(
            signature: ToolSignature,
        ) -> None:
            async with limiter:
                classification = await self._classify_one(signature)
            classifications[classification.signature_hash] = classification

        async with anyio.create_task_group() as task_group:
            for signature in signatures_by_hash.values():
                task_group.start_soon(classify_with_limit, signature)

        return classifications

    async def _classify_one(self, signature: ToolSignature) -> ToolClassification:
        request = ChatCompletionRequest(
            model=self.classifier_model,
            messages=[
                ChatMessage(role="system", content=self._prompt),
                ChatMessage(
                    role="user",
                    content=msgspec.json.encode(
                        {"signature": signature.signature},
                        order="deterministic",
                    ).decode(),
                ),
            ],
            tools=[TOOL_EFFECT_CLASSIFIER_TOOL],
            tool_choice="required",
            parallel_tool_calls=False,
            max_completion_tokens=TOOL_EFFECT_CLASSIFIER_MAX_TOKENS,
            temperature=0,
        )
        log_payload(
            logger,
            "tool.classifier.request.payload",
            request=asdict(request),
            signature=signature.signature,
        )
        result: ChatCompletionResult | None = None
        try:
            result = await self._client.complete(request)
            raw = _parse_raw_output(result.message, expected_tool_name=TOOL_EFFECT_CLASSIFIER_TOOL.function.name)
            classification = _classification_from_raw(
                signature_hash=signature.signature_hash,
                classifier=self.classifier,
                classifier_model=self.classifier_cache_model,
                prompt_hash=self.prompt_hash,
                raw=raw,
            )
        except Exception as exc:
            log_debug(
                logger,
                "tool.classifier.failed",
                classifier=self.classifier,
                classifier_model=self.classifier_model,
                classifier_cache_model=self.classifier_cache_model,
                error_message=str(exc),
                error_type=type(exc).__name__,
                has_result=result is not None,
                **(_classifier_result_context(result) if result is not None else {}),
                signature_hash=signature.signature_hash.hex(),
            )
            log_payload(
                logger,
                "tool.classifier.failed.payload",
                signature=signature.signature,
            )
            if result is not None:
                log_payload(
                    logger,
                    "tool.classifier.failed.result.payload",
                    request=asdict(request),
                    result=asdict(result),
                    signature=signature.signature,
                )
            return _unknown_classification(
                signature_hash=signature.signature_hash,
                classifier=self.classifier,
                classifier_model=self.classifier_cache_model,
                prompt_hash=self.prompt_hash,
                rationale=f"classifier failed: {type(exc).__name__}",
                raw_output={},
            )
        else:
            log_debug(
                logger,
                "tool.classifier.fresh",
                classifier=self.classifier,
                classifier_model=self.classifier_model,
                classifier_cache_model=self.classifier_cache_model,
                confidence=classification.confidence,
                effect_class=classification.effect_class,
                signature_hash=signature.signature_hash.hex(),
            )
            log_payload(
                logger,
                "tool.classifier.fresh.payload",
                raw_output=raw,
                result=asdict(result),
                signature=signature.signature,
            )
            return classification


class LLMToolCallClassifier(IToolCallClassifier):
    def __init__(
        self,
        client: IChatCompletionClient,
        *,
        classifier_model: str,
        classifier_cache_model: str,
        classifier: str = TOOL_CALL_EFFECT_CLASSIFIER_NAME,
        prompt: str = TOOL_CALL_EFFECT_CLASSIFIER_PROMPT,
        max_concurrency: int = 4,
    ) -> None:
        self._client = client
        self.classifier = classifier
        self.classifier_model = classifier_model
        self.classifier_cache_model = classifier_cache_model
        self.prompt_hash = _prompt_hash(
            prompt,
            tool=TOOL_CALL_EFFECT_CLASSIFIER_TOOL,
            max_tokens=TOOL_CALL_EFFECT_CLASSIFIER_MAX_TOKENS,
        )
        self._prompt = prompt
        self._max_concurrency = max_concurrency

    async def classify(
        self,
        call: ToolCallSignature,
    ) -> ToolCallClassification:
        return (await self.classify_many([call]))[call.classification_key]

    async def classify_many(
        self,
        calls: list[ToolCallSignature],
    ) -> dict[tuple[bytes, bytes], ToolCallClassification]:
        calls_by_key = {call.classification_key: call for call in calls}
        if not calls_by_key:
            return {}

        limiter = anyio.CapacityLimiter(self._max_concurrency)
        classifications: dict[tuple[bytes, bytes], ToolCallClassification] = {}

        async def classify_with_limit(call: ToolCallSignature) -> None:
            async with limiter:
                classification = await self._classify_one(call)
            key = (classification.signature_hash, classification.arguments_hash)
            classifications[key] = classification

        async with anyio.create_task_group() as task_group:
            for call in calls_by_key.values():
                task_group.start_soon(classify_with_limit, call)

        return classifications

    async def _classify_one(
        self,
        call: ToolCallSignature,
    ) -> ToolCallClassification:
        request = ChatCompletionRequest(
            model=self.classifier_model,
            messages=[
                ChatMessage(role="system", content=self._prompt),
                ChatMessage(
                    role="user",
                    content=msgspec.json.encode(
                        {
                            "signature": call.signature.signature,
                            "arguments": call.arguments,
                        },
                        order="deterministic",
                    ).decode(),
                ),
            ],
            tools=[TOOL_CALL_EFFECT_CLASSIFIER_TOOL],
            tool_choice="required",
            parallel_tool_calls=False,
            max_completion_tokens=TOOL_CALL_EFFECT_CLASSIFIER_MAX_TOKENS,
            temperature=0,
        )
        log_payload(
            logger,
            "tool.call_classifier.request.payload",
            arguments=call.arguments,
            request=asdict(request),
            signature=call.signature.signature,
        )
        result: ChatCompletionResult | None = None
        try:
            result = await self._client.complete(request)
            raw = _parse_raw_output(result.message, expected_tool_name=TOOL_CALL_EFFECT_CLASSIFIER_TOOL.function.name)
            classification = _tool_call_classification_from_raw(
                signature_hash=call.signature_hash,
                arguments_hash=call.arguments_hash,
                classifier=self.classifier,
                classifier_model=self.classifier_cache_model,
                prompt_hash=self.prompt_hash,
                raw=raw,
            )
        except Exception as exc:
            log_debug(
                logger,
                "tool.call_classifier.failed",
                arguments_hash=call.arguments_hash.hex(),
                classifier=self.classifier,
                classifier_model=self.classifier_model,
                classifier_cache_model=self.classifier_cache_model,
                error_message=str(exc),
                error_type=type(exc).__name__,
                has_result=result is not None,
                **(_classifier_result_context(result) if result is not None else {}),
                signature_hash=call.signature_hash.hex(),
            )
            log_payload(
                logger,
                "tool.call_classifier.failed.payload",
                arguments=call.arguments,
                signature=call.signature.signature,
            )
            if result is not None:
                log_payload(
                    logger,
                    "tool.call_classifier.failed.result.payload",
                    arguments=call.arguments,
                    request=asdict(request),
                    result=asdict(result),
                    signature=call.signature.signature,
                )
            return _unknown_tool_call_classification(
                signature_hash=call.signature_hash,
                arguments_hash=call.arguments_hash,
                classifier=self.classifier,
                classifier_model=self.classifier_cache_model,
                prompt_hash=self.prompt_hash,
                rationale=f"classifier failed: {type(exc).__name__}",
                raw_output={},
            )
        else:
            log_debug(
                logger,
                "tool.call_classifier.fresh",
                arguments_hash=call.arguments_hash.hex(),
                classifier=self.classifier,
                classifier_model=self.classifier_model,
                classifier_cache_model=self.classifier_cache_model,
                confidence=classification.confidence,
                effect_class=classification.effect_class,
                signature_hash=call.signature_hash.hex(),
            )
            log_payload(
                logger,
                "tool.call_classifier.fresh.payload",
                arguments=call.arguments,
                raw_output=raw,
                result=asdict(result),
                signature=call.signature.signature,
            )
            return classification


def _parse_raw_output(message: ChatMessage, *, expected_tool_name: str) -> dict[str, Any]:
    tool_calls = message.tool_calls or ()
    if len(tool_calls) != 1:
        raise ValueError("classifier did not return exactly one tool call")
    tool_call = tool_calls[0]
    if tool_call.name != expected_tool_name:
        raise ValueError("classifier returned an unexpected tool call")
    return parse_json_object_with_repair(tool_call.arguments)


def _prompt_hash(
    prompt: str,
    *,
    tool: ChatTool,
    max_tokens: int,
) -> bytes:
    value = {
        "prompt": prompt,
        "tool": {
            "type": tool.type,
            "function": {
                "name": tool.function.name,
                "parameters": tool.function.parameters,
                "strict": tool.function.strict,
                "description": tool.function.description,
            },
        },
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "max_completion_tokens": max_tokens,
        "temperature": 0,
    }
    return blake3.blake3(msgspec.json.encode(value, order="deterministic")).digest()


def _classification_from_raw(
    *,
    signature_hash: bytes,
    classifier: str,
    classifier_model: str,
    prompt_hash: bytes,
    raw: dict[str, Any],
) -> ToolClassification:
    effect_class = _effect_class(raw.get("effect_class"))
    confidence = _confidence(raw.get("confidence"))
    rationale = raw.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("classifier rationale must be a non-empty string")
    return ToolClassification(
        signature_hash=signature_hash,
        classifier=classifier,
        classifier_model=classifier_model,
        prompt_hash=prompt_hash,
        effect_class=effect_class,
        confidence=confidence,
        rationale=rationale,
        raw_output=raw,
    )


def _unknown_classification(
    *,
    signature_hash: bytes,
    classifier: str,
    classifier_model: str,
    prompt_hash: bytes,
    rationale: str,
    raw_output: dict[str, Any],
) -> ToolClassification:
    return ToolClassification(
        signature_hash=signature_hash,
        classifier=classifier,
        classifier_model=classifier_model,
        prompt_hash=prompt_hash,
        effect_class="contextual",
        confidence=0.0,
        rationale=rationale,
        raw_output=raw_output,
        persistable=False,
    )


def _tool_call_classification_from_raw(
    *,
    signature_hash: bytes,
    arguments_hash: bytes,
    classifier: str,
    classifier_model: str,
    prompt_hash: bytes,
    raw: dict[str, Any],
) -> ToolCallClassification:
    effect_class = _tool_call_effect_class(raw.get("effect_class"))
    confidence = _confidence(raw.get("confidence"))
    rationale = raw.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("classifier rationale must be a non-empty string")
    return ToolCallClassification(
        signature_hash=signature_hash,
        arguments_hash=arguments_hash,
        classifier=classifier,
        classifier_model=classifier_model,
        prompt_hash=prompt_hash,
        effect_class=effect_class,
        confidence=confidence,
        rationale=rationale,
        raw_output=raw,
    )


def _unknown_tool_call_classification(
    *,
    signature_hash: bytes,
    arguments_hash: bytes,
    classifier: str,
    classifier_model: str,
    prompt_hash: bytes,
    rationale: str,
    raw_output: dict[str, Any],
) -> ToolCallClassification:
    return ToolCallClassification(
        signature_hash=signature_hash,
        arguments_hash=arguments_hash,
        classifier=classifier,
        classifier_model=classifier_model,
        prompt_hash=prompt_hash,
        effect_class="unknown",
        confidence=0.0,
        rationale=rationale,
        raw_output=raw_output,
        persistable=False,
    )


def _effect_class(value: object) -> EffectClass:
    try:
        return EffectClass(value)
    except ValueError as exc:
        raise ValueError("classifier effect_class is invalid") from exc


def _tool_call_effect_class(value: object) -> ToolCallEffectClass:
    try:
        return ToolCallEffectClass(value)
    except ValueError as exc:
        raise ValueError("classifier effect_class is invalid") from exc


def _confidence(value: object) -> float:
    if isinstance(value, int | float) and 0 <= value <= 1:
        return float(value)
    raise ValueError("classifier confidence is invalid")
