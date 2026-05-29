from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import msgspec
from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from sqlalchemy.ext.asyncio import AsyncSession

from plap.persistence import Database
from plap.tools.policy import (
    EffectClass,
    ToolCallClassification,
    ToolCallEffectClass,
    ToolClassification,
    ToolSignature,
)


class ToolClassificationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_or_create_signature(self, signature: ToolSignature) -> ToolSignature:
        await self.get_or_create_signatures([signature])
        return signature

    async def get_or_create_signatures(self, signatures: list[ToolSignature]) -> list[ToolSignature]:
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
        result = await self._session.execute(
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
        result.close()
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
        try:
            row = result.mappings().one_or_none()
            if row is None:
                return None
            return _classification_from_row(row)
        finally:
            result.close()

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
        signature_hashes_json = _json_string([signature_hash.hex() for signature_hash in signature_hashes])
        result = await self._session.execute(
            text(
                """
                with requested(signature_hash) as (
                  select decode(value, 'hex')
                  from jsonb_array_elements_text(
                    cast(:signature_hashes as jsonb)
                  ) value
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
        try:
            classifications = [_classification_from_row(row) for row in result.mappings()]
            return {classification.signature_hash: classification for classification in classifications}
        finally:
            result.close()

    async def store_classification(self, classification: ToolClassification) -> ToolClassification:
        stored = await self.store_classifications([classification])
        return stored[classification.signature_hash]

    async def store_classifications(self, classifications: list[ToolClassification]) -> dict[bytes, ToolClassification]:
        if not classifications:
            return {}
        if any(not classification.persistable for classification in classifications):
            raise ValueError("non-persistable classifications must not be stored")
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
        result = await self._session.execute(
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
        result.close()
        stored = await self.get_classifications(
            [classification.signature_hash for classification in classifications],
            classifier=first.classifier,
            classifier_model=first.classifier_model,
            prompt_hash=first.prompt_hash,
        )
        if len(stored) != len({item.signature_hash for item in classifications}):
            raise RuntimeError("tool classification insert did not produce a row")
        return stored

    async def get_tool_call_classification(
        self,
        *,
        signature_hash: bytes,
        arguments_hash: bytes,
        classifier: str,
        classifier_model: str,
        prompt_hash: bytes,
    ) -> ToolCallClassification | None:
        classifications = await self.get_tool_call_classifications(
            [(signature_hash, arguments_hash)],
            classifier=classifier,
            classifier_model=classifier_model,
            prompt_hash=prompt_hash,
        )
        return classifications.get((signature_hash, arguments_hash))

    async def get_tool_call_classifications(
        self,
        keys: list[tuple[bytes, bytes]],
        *,
        classifier: str,
        classifier_model: str,
        prompt_hash: bytes,
    ) -> dict[tuple[bytes, bytes], ToolCallClassification]:
        if not keys:
            return {}
        keys_json = _json_string(
            [
                {
                    "signature_hash": signature_hash.hex(),
                    "arguments_hash": arguments_hash.hex(),
                }
                for signature_hash, arguments_hash in keys
            ]
        )
        result = await self._session.execute(
            text(
                """
                with requested(signature_hash, arguments_hash) as (
                  select
                    decode(value ->> 'signature_hash', 'hex'),
                    decode(value ->> 'arguments_hash', 'hex')
                  from jsonb_array_elements(cast(:keys as jsonb)) value
                )
                select
                  classifications.signature_hash,
                  classifications.arguments_hash,
                  classifications.classifier,
                  classifications.classifier_model,
                  classifications.prompt_hash,
                  classifications.effect_class,
                  classifications.confidence,
                  classifications.rationale,
                  classifications.raw_output
                from responses.tool_call_classifications classifications
                join requested
                  on requested.signature_hash = classifications.signature_hash
                 and requested.arguments_hash = classifications.arguments_hash
                where classifications.classifier = :classifier
                  and classifications.classifier_model = :classifier_model
                  and classifications.prompt_hash = :prompt_hash
                """
            ),
            {
                "keys": keys_json,
                "classifier": classifier,
                "classifier_model": classifier_model,
                "prompt_hash": prompt_hash,
            },
        )
        try:
            classifications = [_tool_call_classification_from_row(row) for row in result.mappings()]
            return {
                (
                    classification.signature_hash,
                    classification.arguments_hash,
                ): classification
                for classification in classifications
            }
        finally:
            result.close()

    async def store_tool_call_classification(self, classification: ToolCallClassification) -> ToolCallClassification:
        stored = await self.store_tool_call_classifications([classification])
        return stored[(classification.signature_hash, classification.arguments_hash)]

    async def store_tool_call_classifications(
        self, classifications: list[ToolCallClassification]
    ) -> dict[tuple[bytes, bytes], ToolCallClassification]:
        if not classifications:
            return {}
        if any(not classification.persistable for classification in classifications):
            raise ValueError("non-persistable tool call classifications must not be stored")
        first = classifications[0]
        if any(
            classification.classifier != first.classifier
            or classification.classifier_model != first.classifier_model
            or classification.prompt_hash != first.prompt_hash
            for classification in classifications
        ):
            raise ValueError("batched tool call classifications must share classifier identity")
        classifications_json = _json_string(
            [
                {
                    "signature_hash": classification.signature_hash.hex(),
                    "arguments_hash": classification.arguments_hash.hex(),
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
        result = await self._session.execute(
            text(
                """
                insert into responses.tool_call_classifications (
                  signature_hash,
                  arguments_hash,
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
                  decode(value ->> 'arguments_hash', 'hex'),
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
                  arguments_hash,
                  classifier,
                  classifier_model,
                  prompt_hash
                ) do nothing
                """
            ),
            {"classifications": classifications_json},
        )
        result.close()
        stored = await self.get_tool_call_classifications(
            [(classification.signature_hash, classification.arguments_hash) for classification in classifications],
            classifier=first.classifier,
            classifier_model=first.classifier_model,
            prompt_hash=first.prompt_hash,
        )
        if len(stored) != len({(item.signature_hash, item.arguments_hash) for item in classifications}):
            raise RuntimeError("tool call classification insert did not produce a row")
        return stored


class ToolClassificationStore:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def get_or_create_signature(self, signature: ToolSignature) -> ToolSignature:
        async with self._database.session_transaction() as session:
            return await ToolClassificationRepository(session).get_or_create_signature(signature)

    async def get_or_create_signatures(self, signatures: list[ToolSignature]) -> list[ToolSignature]:
        async with self._database.session_transaction() as session:
            return await ToolClassificationRepository(session).get_or_create_signatures(signatures)

    async def get_classification(
        self,
        signature_hash: bytes,
        *,
        classifier: str,
        classifier_model: str,
        prompt_hash: bytes,
    ) -> ToolClassification | None:
        async with self._database.session() as session:
            return await ToolClassificationRepository(session).get_classification(
                signature_hash,
                classifier=classifier,
                classifier_model=classifier_model,
                prompt_hash=prompt_hash,
            )

    async def get_classifications(
        self,
        signature_hashes: list[bytes],
        *,
        classifier: str,
        classifier_model: str,
        prompt_hash: bytes,
    ) -> dict[bytes, ToolClassification]:
        async with self._database.session() as session:
            return await ToolClassificationRepository(session).get_classifications(
                signature_hashes,
                classifier=classifier,
                classifier_model=classifier_model,
                prompt_hash=prompt_hash,
            )

    async def store_classification(self, classification: ToolClassification) -> ToolClassification:
        async with self._database.session_transaction() as session:
            return await ToolClassificationRepository(session).store_classification(classification)

    async def store_classifications(self, classifications: list[ToolClassification]) -> dict[bytes, ToolClassification]:
        async with self._database.session_transaction() as session:
            return await ToolClassificationRepository(session).store_classifications(classifications)

    async def get_tool_call_classification(
        self,
        *,
        signature_hash: bytes,
        arguments_hash: bytes,
        classifier: str,
        classifier_model: str,
        prompt_hash: bytes,
    ) -> ToolCallClassification | None:
        async with self._database.session() as session:
            return await ToolClassificationRepository(session).get_tool_call_classification(
                signature_hash=signature_hash,
                arguments_hash=arguments_hash,
                classifier=classifier,
                classifier_model=classifier_model,
                prompt_hash=prompt_hash,
            )

    async def get_tool_call_classifications(
        self,
        keys: list[tuple[bytes, bytes]],
        *,
        classifier: str,
        classifier_model: str,
        prompt_hash: bytes,
    ) -> dict[tuple[bytes, bytes], ToolCallClassification]:
        async with self._database.session() as session:
            return await ToolClassificationRepository(session).get_tool_call_classifications(
                keys,
                classifier=classifier,
                classifier_model=classifier_model,
                prompt_hash=prompt_hash,
            )

    async def store_tool_call_classification(self, classification: ToolCallClassification) -> ToolCallClassification:
        async with self._database.session_transaction() as session:
            return await ToolClassificationRepository(session).store_tool_call_classification(classification)

    async def store_tool_call_classifications(
        self,
        classifications: list[ToolCallClassification],
    ) -> dict[tuple[bytes, bytes], ToolCallClassification]:
        async with self._database.session_transaction() as session:
            return await ToolClassificationRepository(session).store_tool_call_classifications(classifications)


def _classification_from_row(row: RowMapping) -> ToolClassification:
    return ToolClassification(
        signature_hash=bytes(row["signature_hash"]),
        classifier=str(row["classifier"]),
        classifier_model=str(row["classifier_model"]),
        prompt_hash=bytes(row["prompt_hash"]),
        effect_class=_effect_class(str(row["effect_class"])),
        confidence=float(row["confidence"]),
        rationale=str(row["rationale"]),
        raw_output=dict(cast(Mapping[str, object], row["raw_output"])),
    )


def _tool_call_classification_from_row(row: RowMapping) -> ToolCallClassification:
    return ToolCallClassification(
        signature_hash=bytes(row["signature_hash"]),
        arguments_hash=bytes(row["arguments_hash"]),
        classifier=str(row["classifier"]),
        classifier_model=str(row["classifier_model"]),
        prompt_hash=bytes(row["prompt_hash"]),
        effect_class=_tool_call_effect_class(str(row["effect_class"])),
        confidence=float(row["confidence"]),
        rationale=str(row["rationale"]),
        raw_output=dict(cast(Mapping[str, object], row["raw_output"])),
    )


def _effect_class(value: str) -> EffectClass:
    try:
        return EffectClass(value)
    except ValueError as exc:
        raise ValueError(f"unsupported effect class: {value}") from exc


def _tool_call_effect_class(value: str) -> ToolCallEffectClass:
    try:
        return ToolCallEffectClass(value)
    except ValueError as exc:
        raise ValueError(f"unsupported tool call effect class: {value}") from exc


def _json_string(value: object) -> str:
    return msgspec.json.encode(value, order="deterministic").decode()
