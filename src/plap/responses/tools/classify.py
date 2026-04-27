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
    IToolClassifier,
    ToolClassification,
    ToolSignature,
    signature_hash_hex,
)

TOOL_EFFECT_CLASSIFIER_PROMPT = """Classify client-provided tools by side effects.

Return only JSON matching this schema:
{"effect_class":"safe|mutation|contextual|unknown","confidence":0.0,"rationale":"short"}

Definitions:
- safe: read-only or exploratory; no file, client, repo, shell, or external mutation.
- mutation: writes files, runs mutating commands, changes external state, or has
  irreversible side effects.
- contextual: can be safe or mutating depending on call arguments, such as shell,
  SQL, HTTP, or command execution tools.
- unknown: ambiguous or insufficient information.
"""

TOOL_EFFECT_CLASSIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "effect_class": {
            "type": "string",
            "enum": ["safe", "mutation", "contextual", "unknown"],
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


class LLMToolClassifier(IToolClassifier):
    def __init__(
        self,
        client: IChatCompletionClient,
        *,
        classifier: str,
        classifier_model: str,
        prompt: str = TOOL_EFFECT_CLASSIFIER_PROMPT,
        max_concurrency: int = 4,
    ) -> None:
        self._client = client
        self.classifier = classifier
        self.classifier_model = classifier_model
        self.prompt_hash = _prompt_hash(prompt)
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
                                {
                                    "signature_hash": signature_hash_hex(
                                        signature.signature_hash
                                    ),
                                    "signature": signature.signature,
                                },
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
    if value in {"safe", "mutation", "contextual", "unknown"}:
        return value
    raise ValueError("classifier effect_class is invalid")


def _confidence(value: object) -> float:
    if isinstance(value, int | float) and 0 <= value <= 1:
        return float(value)
    raise ValueError("classifier confidence is invalid")
