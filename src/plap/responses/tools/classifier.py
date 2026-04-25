from __future__ import annotations

from typing import Any, Protocol

import blake3
import msgspec

from plap.llms.chat import (
    ChatCompletionClient,
    ChatCompletionRequest,
    ChatMessage,
    ChatResponseFormat,
)
from plap.responses.tools.repository import ToolClassificationRepository
from plap.responses.tools.types import (
    EffectClass,
    ToolClassification,
    ToolSignature,
)

TOOL_EFFECT_CLASSIFIER_PROMPT = """Classify a client-provided tool by side effects.

Return only JSON matching this schema:
{"effect_class":"safe|mutation|unknown","confidence":0.0,"rationale":"short"}

Definitions:
- safe: read-only or exploratory; no file, client, repo, shell, or external mutation.
- mutation: writes files, runs mutating commands, changes external state, or has
  irreversible side effects.
- unknown: ambiguous or insufficient information.
"""

TOOL_EFFECT_CLASSIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "effect_class": {"type": "string", "enum": ["safe", "mutation", "unknown"]},
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
TOOL_EFFECT_CLASSIFIER_MAX_TOKENS = 128


class ToolClassifier(Protocol):
    classifier: str
    classifier_model: str
    prompt_hash: bytes

    async def classify(self, signature: ToolSignature) -> ToolClassification: ...


class LLMToolClassifier:
    def __init__(
        self,
        client: ChatCompletionClient,
        *,
        classifier: str,
        classifier_model: str,
        prompt: str = TOOL_EFFECT_CLASSIFIER_PROMPT,
    ) -> None:
        self._client = client
        self.classifier = classifier
        self.classifier_model = classifier_model
        self.prompt_hash = _prompt_hash(prompt)
        self._prompt = prompt

    async def classify(self, signature: ToolSignature) -> ToolClassification:
        try:
            result = await self._client.complete(
                ChatCompletionRequest(
                    model=self.classifier_model,
                    messages=[
                        ChatMessage(role="system", content=self._prompt),
                        ChatMessage(
                            role="user",
                            content=msgspec.json.encode(
                                signature.signature, order="deterministic"
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


class CachedToolClassifier:
    def __init__(
        self,
        repository: ToolClassificationRepository,
        classifier: ToolClassifier,
    ) -> None:
        self._repository = repository
        self._classifier = classifier

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

        unique_signatures = list(signatures_by_hash.values())
        await self._repository.get_or_create_signatures(unique_signatures)
        cached = await self._repository.get_classifications(
            list(signatures_by_hash),
            classifier=self._classifier.classifier,
            classifier_model=self._classifier.classifier_model,
            prompt_hash=self._classifier.prompt_hash,
        )
        missing = [
            signature
            for signature_hash, signature in signatures_by_hash.items()
            if signature_hash not in cached
        ]
        new_classifications = [
            await self._classifier.classify(signature) for signature in missing
        ]
        stored = await self._repository.store_classifications(new_classifications)
        return {**cached, **stored}


def _parse_raw_output(content: str | None) -> dict[str, Any]:
    if not content:
        raise ValueError("classifier returned no content")
    value = msgspec.json.decode(content.encode())
    if not isinstance(value, dict):
        raise TypeError("classifier returned non-object JSON")
    return value


def _prompt_hash(prompt: str) -> bytes:
    value = {
        "prompt": prompt,
        "response_format": {
            "type": TOOL_EFFECT_CLASSIFIER_RESPONSE_FORMAT.type,
            "name": TOOL_EFFECT_CLASSIFIER_RESPONSE_FORMAT.name,
            "schema": TOOL_EFFECT_CLASSIFIER_RESPONSE_FORMAT.schema,
            "strict": TOOL_EFFECT_CLASSIFIER_RESPONSE_FORMAT.strict,
            "description": TOOL_EFFECT_CLASSIFIER_RESPONSE_FORMAT.description,
        },
        "max_completion_tokens": TOOL_EFFECT_CLASSIFIER_MAX_TOKENS,
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


def _effect_class(value: object) -> EffectClass:
    if value in {"safe", "mutation", "unknown"}:
        return value
    raise ValueError("classifier effect_class is invalid")


def _confidence(value: object) -> float:
    if isinstance(value, int | float) and 0 <= value <= 1:
        return float(value)
    raise ValueError("classifier confidence is invalid")
