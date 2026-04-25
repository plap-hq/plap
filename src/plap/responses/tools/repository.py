from __future__ import annotations

from typing import Any

import msgspec
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from plap.responses.tools.types import (
    EffectClass,
    ToolClassification,
    ToolSignature,
)


class ToolClassificationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_or_create_signature(self, signature: ToolSignature) -> ToolSignature:
        await self.get_or_create_signatures([signature])
        return signature

    async def get_or_create_signatures(
        self, signatures: list[ToolSignature]
    ) -> list[ToolSignature]:
        if not signatures:
            return []
        signatures_json = _json_string(
            [
                {
                    "signature_hash": signature.signature_hash.hex(),
                    "signature": signature.signature,
                }
                for signature in signatures
            ]
        )
        await self._session.execute(
            text(
                """
                insert into responses.tool_signatures (
                  signature_hash,
                  signature_json
                )
                select
                  decode(value ->> 'signature_hash', 'hex'),
                  value -> 'signature'
                from jsonb_array_elements(cast(:signatures as jsonb)) value
                on conflict (signature_hash) do nothing
                """
            ),
            {"signatures": signatures_json},
        )
        return signatures

    async def get_classification(
        self,
        signature_hash: bytes,
        *,
        classifier: str,
        classifier_model: str,
        prompt_hash: bytes,
    ) -> ToolClassification | None:
        result = await self._session.execute(
            text(
                """
                select
                  signature_hash,
                  classifier,
                  classifier_model,
                  prompt_hash,
                  effect_class,
                  confidence,
                  rationale,
                  raw_output
                from responses.tool_classifications
                where signature_hash = :signature_hash
                  and classifier = :classifier
                  and classifier_model = :classifier_model
                  and prompt_hash = :prompt_hash
                """
            ),
            {
                "signature_hash": signature_hash,
                "classifier": classifier,
                "classifier_model": classifier_model,
                "prompt_hash": prompt_hash,
            },
        )
        row = result.one_or_none()
        if row is None:
            return None
        return _classification_from_row(row)

    async def get_classifications(
        self,
        signature_hashes: list[bytes],
        *,
        classifier: str,
        classifier_model: str,
        prompt_hash: bytes,
    ) -> dict[bytes, ToolClassification]:
        if not signature_hashes:
            return {}
        signature_hashes_json = _json_string(
            [signature_hash.hex() for signature_hash in signature_hashes]
        )
        result = await self._session.execute(
            text(
                """
                with requested(signature_hash) as (
                  select decode(value #>> '{}', 'hex')
                  from jsonb_array_elements(cast(:signature_hashes as jsonb)) value
                )
                select
                  classifications.signature_hash,
                  classifications.classifier,
                  classifications.classifier_model,
                  classifications.prompt_hash,
                  classifications.effect_class,
                  classifications.confidence,
                  classifications.rationale,
                  classifications.raw_output
                from responses.tool_classifications classifications
                join requested
                  on requested.signature_hash = classifications.signature_hash
                where classifications.classifier = :classifier
                  and classifications.classifier_model = :classifier_model
                  and classifications.prompt_hash = :prompt_hash
                """
            ),
            {
                "signature_hashes": signature_hashes_json,
                "classifier": classifier,
                "classifier_model": classifier_model,
                "prompt_hash": prompt_hash,
            },
        )
        classifications = [_classification_from_row(row) for row in result]
        return {
            classification.signature_hash: classification
            for classification in classifications
        }

    async def store_classification(
        self, classification: ToolClassification
    ) -> ToolClassification:
        stored = await self.store_classifications([classification])
        return stored[classification.signature_hash]

    async def store_classifications(
        self, classifications: list[ToolClassification]
    ) -> dict[bytes, ToolClassification]:
        if not classifications:
            return {}
        first = classifications[0]
        if any(
            classification.classifier != first.classifier
            or classification.classifier_model != first.classifier_model
            or classification.prompt_hash != first.prompt_hash
            for classification in classifications
        ):
            raise ValueError("batched classifications must share classifier identity")
        classifications_json = _json_string(
            [
                {
                    "signature_hash": classification.signature_hash.hex(),
                    "classifier": classification.classifier,
                    "classifier_model": classification.classifier_model,
                    "prompt_hash": classification.prompt_hash.hex(),
                    "effect_class": classification.effect_class,
                    "confidence": classification.confidence,
                    "rationale": classification.rationale,
                    "raw_output": classification.raw_output,
                }
                for classification in classifications
            ]
        )
        await self._session.execute(
            text(
                """
                insert into responses.tool_classifications (
                  signature_hash,
                  classifier,
                  classifier_model,
                  prompt_hash,
                  effect_class,
                  confidence,
                  rationale,
                  raw_output
                )
                select
                  decode(value ->> 'signature_hash', 'hex'),
                  value ->> 'classifier',
                  value ->> 'classifier_model',
                  decode(value ->> 'prompt_hash', 'hex'),
                  value ->> 'effect_class',
                  (value ->> 'confidence')::numeric,
                  value ->> 'rationale',
                  value -> 'raw_output'
                from jsonb_array_elements(cast(:classifications as jsonb)) value
                on conflict (
                  signature_hash,
                  classifier,
                  classifier_model,
                  prompt_hash
                ) do nothing
                """
            ),
            {"classifications": classifications_json},
        )
        stored = await self.get_classifications(
            [classification.signature_hash for classification in classifications],
            classifier=first.classifier,
            classifier_model=first.classifier_model,
            prompt_hash=first.prompt_hash,
        )
        if len(stored) != len({item.signature_hash for item in classifications}):
            raise RuntimeError("tool classification insert did not produce a row")
        return stored


def _classification_from_row(row: Any) -> ToolClassification:
    return ToolClassification(
        signature_hash=bytes(row.signature_hash),
        classifier=row.classifier,
        classifier_model=row.classifier_model,
        prompt_hash=bytes(row.prompt_hash),
        effect_class=_effect_class(row.effect_class),
        confidence=float(row.confidence),
        rationale=row.rationale,
        raw_output=row.raw_output,
    )


def _effect_class(value: str) -> EffectClass:
    if value in {"safe", "mutation", "unknown"}:
        return value
    raise ValueError(f"unsupported effect class: {value}")


def _json_string(value: object) -> str:
    return msgspec.json.encode(value, order="deterministic").decode()
