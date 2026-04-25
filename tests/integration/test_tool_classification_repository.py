from __future__ import annotations

import pytest

from plap.responses.contracts import FunctionTool
from plap.responses.tools import CachedToolClassifier, function_tool_signature
from plap.responses.tools.repository import ToolClassificationRepository
from plap.responses.tools.types import ToolClassification, ToolSignature


@pytest.mark.asyncio
async def test_tool_classification_repository_roundtrips_signature_and_classification(
    db_session_maker,
) -> None:
    signature = function_tool_signature(_read_file_tool())
    classification = ToolClassification(
        signature_hash=signature.signature_hash,
        classifier="fake",
        classifier_model="fake/model",
        prompt_hash=b"p" * 32,
        effect_class="safe",
        confidence=0.95,
        rationale="read-only",
        raw_output={"effect_class": "safe"},
    )

    async with db_session_maker() as session:
        repository = ToolClassificationRepository(session)
        await repository.get_or_create_signature(signature)
        await repository.store_classification(classification)
        await session.commit()

    async with db_session_maker() as session:
        repository = ToolClassificationRepository(session)
        stored = await repository.get_classification(
            signature.signature_hash,
            classifier="fake",
            classifier_model="fake/model",
            prompt_hash=b"p" * 32,
        )

    assert stored == classification


@pytest.mark.asyncio
async def test_tool_classification_repository_cache_key_uses_prompt_hash(
    db_session_maker,
) -> None:
    signature = function_tool_signature(_read_file_tool())

    async with db_session_maker() as session:
        repository = ToolClassificationRepository(session)
        await repository.get_or_create_signature(signature)
        await repository.store_classification(
            ToolClassification(
                signature_hash=signature.signature_hash,
                classifier="fake",
                classifier_model="fake/model",
                prompt_hash=b"a" * 32,
                effect_class="safe",
                confidence=1.0,
                rationale="read-only",
                raw_output={},
            )
        )
        await session.commit()

    async with db_session_maker() as session:
        repository = ToolClassificationRepository(session)
        missing = await repository.get_classification(
            signature.signature_hash,
            classifier="fake",
            classifier_model="fake/model",
            prompt_hash=b"b" * 32,
        )

    assert missing is None


@pytest.mark.asyncio
async def test_cached_tool_classifier_uses_stored_classification(
    db_session_maker,
) -> None:
    signature = function_tool_signature(_read_file_tool())

    async with db_session_maker() as session:
        repository = ToolClassificationRepository(session)
        classifier = _CountingClassifier()
        cached = CachedToolClassifier(repository, classifier)

        first = await cached.classify(signature)
        second = await cached.classify(signature)
        await session.commit()

    assert first == second
    assert classifier.calls == 1


@pytest.mark.asyncio
async def test_cached_tool_classifier_batches_cache_lookup(
    db_session_maker,
) -> None:
    read_signature = function_tool_signature(_read_file_tool())
    list_signature = function_tool_signature(_list_files_tool())
    cached_read = ToolClassification(
        signature_hash=read_signature.signature_hash,
        classifier="fake",
        classifier_model="fake/model",
        prompt_hash=b"p" * 32,
        effect_class="safe",
        confidence=1.0,
        rationale="cached read-only",
        raw_output={},
    )

    async with db_session_maker() as session:
        repository = ToolClassificationRepository(session)
        await repository.get_or_create_signatures([read_signature, list_signature])
        await repository.store_classification(cached_read)
        classifier = _CountingClassifier()
        cached = CachedToolClassifier(repository, classifier)

        classifications = await cached.classify_many([read_signature, list_signature])
        await session.commit()

    assert classifications[read_signature.signature_hash] == cached_read
    assert classifications[list_signature.signature_hash].rationale == "read-only"
    assert classifier.calls == 1


def _read_file_tool() -> FunctionTool:
    return FunctionTool(
        description="Read a file without changing it.",
        name="read_file",
        parameters={"type": "object"},
        strict=True,
        type="function",
    )


def _list_files_tool() -> FunctionTool:
    return FunctionTool(
        description="List files without changing them.",
        name="list_files",
        parameters={"type": "object"},
        strict=True,
        type="function",
    )


class _CountingClassifier:
    classifier = "fake"
    classifier_model = "fake/model"
    prompt_hash = b"p" * 32

    def __init__(self) -> None:
        self.calls = 0

    async def classify(self, signature: ToolSignature) -> ToolClassification:
        self.calls += 1
        return ToolClassification(
            signature_hash=signature.signature_hash,
            classifier=self.classifier,
            classifier_model=self.classifier_model,
            prompt_hash=self.prompt_hash,
            effect_class="safe",
            confidence=1.0,
            rationale="read-only",
            raw_output={},
        )
