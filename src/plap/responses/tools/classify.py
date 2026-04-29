from __future__ import annotations

from typing import Any

import anyio
import blake3
import msgspec

from plap.llms.chat import (
    ChatCompletionRequest,
    ChatMessage,
    ChatResponseFormat,
    IChatCompletionClient,
)
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

TOOL_EFFECT_CLASSIFIER_PROMPT = """Classify client-provided tools by side effects.

Return only JSON matching this schema:
{"effect_class":"safe|visible|mutation|contextual|unknown","confidence":0.0,"rationale":"short"}

Definitions:
- safe: read-only or exploratory; no file, client, repo, shell, or external mutation.
- visible: changes visible user-facing state or agent control flow, but does not
  mutate files, repositories, clients, services, or external systems.
- mutation: writes files, runs mutating commands, changes external state, or has
  irreversible side effects.
- contextual: can be safe or mutating depending on call arguments, such as shell,
  SQL, HTTP, or command execution tools.
- unknown: ambiguous or insufficient information.
"""
TOOL_EFFECT_CLASSIFIER_NAME = "llm_tool_effect_classifier"

TOOL_EFFECT_CLASSIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "effect_class": {
            "type": "string",
            "enum": ["safe", "visible", "mutation", "contextual", "unknown"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": ["effect_class", "confidence", "rationale"],
    "additionalProperties": False,
}

TOOL_EFFECT_CLASSIFIER_RESPONSE_FORMAT = ChatResponseFormat(
    type="json_schema",
    name="tool_effect_classification",
    schema=TOOL_EFFECT_CLASSIFIER_SCHEMA,
    strict=True,
    description="Effect classification for a client-provided function tool.",
)
TOOL_EFFECT_CLASSIFIER_MAX_TOKENS = 512

TOOL_CALL_EFFECT_CLASSIFIER_PROMPT = """Classify a concrete client tool call.

Return only JSON matching this schema:
{"effect_class":"safe|mutation|unknown","confidence":0.0,"rationale":"short"}

Definitions:
- safe: this specific call is read-only or exploratory and should not mutate files,
  client state, repositories, shells, databases, services, or external systems.
- mutation: this specific call writes, deletes, runs mutating commands, sends data,
  changes external state, or has irreversible side effects.
- unknown: the concrete arguments are ambiguous, incomplete, malformed, or not enough
  to decide safely.

Do not return contextual. You are classifying this concrete call, not the tool family.
"""
TOOL_CALL_EFFECT_CLASSIFIER_NAME = "llm_tool_call_effect_classifier"

TOOL_CALL_EFFECT_CLASSIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "effect_class": {
            "type": "string",
            "enum": ["safe", "mutation", "unknown"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": ["effect_class", "confidence", "rationale"],
    "additionalProperties": False,
}

TOOL_CALL_EFFECT_CLASSIFIER_RESPONSE_FORMAT = ChatResponseFormat(
    type="json_schema",
    name="tool_call_effect_classification",
    schema=TOOL_CALL_EFFECT_CLASSIFIER_SCHEMA,
    strict=True,
    description="Effect classification for one concrete client tool call.",
)
TOOL_CALL_EFFECT_CLASSIFIER_MAX_TOKENS = 512


class LLMToolClassifier(IToolClassifier):
    def __init__(
        self,
        client: IChatCompletionClient,
        *,
        classifier_model: str,
        classifier: str = TOOL_EFFECT_CLASSIFIER_NAME,
        prompt: str = TOOL_EFFECT_CLASSIFIER_PROMPT,
        max_concurrency: int = 4,
    ) -> None:
        self._client = client
        self.classifier = classifier
        self.classifier_model = classifier_model
        self.prompt_hash = _prompt_hash(
            prompt,
            response_format=TOOL_EFFECT_CLASSIFIER_RESPONSE_FORMAT,
            max_tokens=TOOL_EFFECT_CLASSIFIER_MAX_TOKENS,
        )
        self._prompt = prompt
        self._max_concurrency = max_concurrency

    async def classify(self, signature: ToolSignature) -> ToolClassification:
        return (await self.classify_many([signature]))[signature.signature_hash]

    async def classify_many(
        self, signatures: list[ToolSignature]
    ) -> dict[bytes, ToolClassification]:
        signatures_by_hash = {
            signature.signature_hash: signature for signature in signatures
        }
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
        try:
            result = await self._client.complete(
                ChatCompletionRequest(
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
                    response_format=TOOL_EFFECT_CLASSIFIER_RESPONSE_FORMAT,
                    max_completion_tokens=TOOL_EFFECT_CLASSIFIER_MAX_TOKENS,
                    temperature=0,
                )
            )
            raw = _parse_raw_output(result.message.content)
            return _classification_from_raw(
                signature_hash=signature.signature_hash,
                classifier=self.classifier,
                classifier_model=self.classifier_model,
                prompt_hash=self.prompt_hash,
                raw=raw,
            )
        except Exception as exc:
            return _unknown_classification(
                signature_hash=signature.signature_hash,
                classifier=self.classifier,
                classifier_model=self.classifier_model,
                prompt_hash=self.prompt_hash,
                rationale=f"classifier failed: {type(exc).__name__}",
                raw_output={},
            )


class LLMToolCallClassifier(IToolCallClassifier):
    def __init__(
        self,
        client: IChatCompletionClient,
        *,
        classifier_model: str,
        classifier: str = TOOL_CALL_EFFECT_CLASSIFIER_NAME,
        prompt: str = TOOL_CALL_EFFECT_CLASSIFIER_PROMPT,
        max_concurrency: int = 4,
    ) -> None:
        self._client = client
        self.classifier = classifier
        self.classifier_model = classifier_model
        self.prompt_hash = _prompt_hash(
            prompt,
            response_format=TOOL_CALL_EFFECT_CLASSIFIER_RESPONSE_FORMAT,
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
        try:
            result = await self._client.complete(
                ChatCompletionRequest(
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
                    response_format=TOOL_CALL_EFFECT_CLASSIFIER_RESPONSE_FORMAT,
                    max_completion_tokens=TOOL_CALL_EFFECT_CLASSIFIER_MAX_TOKENS,
                    temperature=0,
                )
            )
            raw = _parse_raw_output(result.message.content)
            return _tool_call_classification_from_raw(
                signature_hash=call.signature_hash,
                arguments_hash=call.arguments_hash,
                classifier=self.classifier,
                classifier_model=self.classifier_model,
                prompt_hash=self.prompt_hash,
                raw=raw,
            )
        except Exception as exc:
            return _unknown_tool_call_classification(
                signature_hash=call.signature_hash,
                arguments_hash=call.arguments_hash,
                classifier=self.classifier,
                classifier_model=self.classifier_model,
                prompt_hash=self.prompt_hash,
                rationale=f"classifier failed: {type(exc).__name__}",
                raw_output={},
            )


def _parse_raw_output(content: str | None) -> dict[str, Any]:
    if not content:
        raise ValueError("classifier returned no content")
    value = msgspec.json.decode(content.encode())
    if not isinstance(value, dict):
        raise TypeError("classifier returned non-object JSON")
    return value


def _prompt_hash(
    prompt: str,
    *,
    response_format: ChatResponseFormat,
    max_tokens: int,
) -> bytes:
    value = {
        "prompt": prompt,
        "response_format": {
            "type": response_format.type,
            "name": response_format.name,
            "schema": response_format.schema,
            "strict": response_format.strict,
            "description": response_format.description,
        },
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
        effect_class="unknown",
        confidence=0.0,
        rationale=rationale,
        raw_output=raw_output,
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
    )


def _effect_class(value: object) -> EffectClass:
    if value in {"safe", "visible", "mutation", "contextual", "unknown"}:
        return value
    raise ValueError("classifier effect_class is invalid")


def _tool_call_effect_class(value: object) -> ToolCallEffectClass:
    if value in {"safe", "mutation", "unknown"}:
        return value
    raise ValueError("classifier effect_class is invalid")


def _confidence(value: object) -> float:
    if isinstance(value, int | float) and 0 <= value <= 1:
        return float(value)
    raise ValueError("classifier confidence is invalid")
