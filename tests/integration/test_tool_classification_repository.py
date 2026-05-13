import pytest

from plap.responses.contracts import FunctionTool
from plap.responses.tools import (
    CachedToolPolicyResolver,
    ToolCallClassification,
    ToolClassification,
    ToolSignature,
    function_tool_signature,
)
from plap.responses.tools.repository import ToolClassificationRepository


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


async def test_tool_classification_repository_roundtrips_contextual_classification(
    db_session_maker,
) -> None:
    signature = function_tool_signature(_bash_tool())
    classification = ToolClassification(
        signature_hash=signature.signature_hash,
        classifier="fake",
        classifier_model="fake/model",
        prompt_hash=b"p" * 32,
        effect_class="contextual",
        confidence=0.8,
        rationale="depends on shell command arguments",
        raw_output={"effect_class": "contextual"},
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


async def test_tool_classification_repository_rejects_non_persistable_classification(
    db_session_maker,
) -> None:
    signature = function_tool_signature(_read_file_tool())

    async with db_session_maker() as session:
        repository = ToolClassificationRepository(session)
        await repository.get_or_create_signature(signature)

        with pytest.raises(ValueError, match="non-persistable classifications must not be stored"):
            await repository.store_classification(
                ToolClassification(
                    signature_hash=signature.signature_hash,
                    classifier="fake",
                    classifier_model="fake/model",
                    prompt_hash=b"p" * 32,
                    effect_class="contextual",
                    confidence=0.0,
                    rationale="classifier failed: ChatCompletionRateLimitError",
                    raw_output={},
                    persistable=False,
                )
            )


async def test_cached_policy_resolver_uses_stored_classification(
    db_session_maker,
) -> None:
    async with db_session_maker() as session:
        repository = ToolClassificationRepository(session)
        classifier = _CountingClassifier()
        resolver = CachedToolPolicyResolver(repository, classifier)

        first = await resolver.resolve([_read_file_tool()])
        await session.commit()

    async with db_session_maker() as session:
        repository = ToolClassificationRepository(session)
        resolver = CachedToolPolicyResolver(repository, classifier)
        second = await resolver.resolve([_read_file_tool()])

    assert first == second
    assert classifier.calls == 1


async def test_cached_policy_resolver_batches_cache_lookup(
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
        resolver = CachedToolPolicyResolver(repository, classifier)

        policies = await resolver.resolve([_read_file_tool(), _list_files_tool()])
        await session.commit()

    assert policies["read_file"].classification == cached_read
    assert policies["list_files"].classification is not None
    assert policies["list_files"].classification.rationale == "read-only"
    assert classifier.calls == 1


async def test_tool_call_classification_repository_is_unscoped(
    db_session_maker,
) -> None:
    signature = function_tool_signature(_bash_tool())
    classification = ToolCallClassification(
        signature_hash=signature.signature_hash,
        arguments_hash=b"a" * 32,
        classifier="fake",
        classifier_model="fake/model",
        prompt_hash=b"p" * 32,
        effect_class="safe",
        confidence=0.7,
        rationale="read-only command",
        raw_output={"effect_class": "safe"},
    )

    async with db_session_maker() as session:
        repository = ToolClassificationRepository(session)
        await repository.get_or_create_signature(signature)
        await repository.store_tool_call_classification(classification)
        await session.commit()

    async with db_session_maker() as session:
        repository = ToolClassificationRepository(session)
        stored = await repository.get_tool_call_classification(
            signature_hash=signature.signature_hash,
            arguments_hash=b"a" * 32,
            classifier="fake",
            classifier_model="fake/model",
            prompt_hash=b"p" * 32,
        )

    assert stored == classification


async def test_tool_call_classification_repository_batches_lookup(
    db_session_maker,
) -> None:
    signature = function_tool_signature(_bash_tool())
    first = ToolCallClassification(
        signature_hash=signature.signature_hash,
        arguments_hash=b"a" * 32,
        classifier="fake",
        classifier_model="fake/model",
        prompt_hash=b"p" * 32,
        effect_class="safe",
        confidence=0.7,
        rationale="read-only command",
        raw_output={"effect_class": "safe"},
    )
    second = ToolCallClassification(
        signature_hash=signature.signature_hash,
        arguments_hash=b"b" * 32,
        classifier="fake",
        classifier_model="fake/model",
        prompt_hash=b"p" * 32,
        effect_class="mutation",
        confidence=0.8,
        rationale="mutating command",
        raw_output={"effect_class": "mutation"},
    )

    async with db_session_maker() as session:
        repository = ToolClassificationRepository(session)
        await repository.get_or_create_signature(signature)
        await repository.store_tool_call_classifications([first, second])
        await session.commit()

    async with db_session_maker() as session:
        repository = ToolClassificationRepository(session)
        stored = await repository.get_tool_call_classifications(
            [
                (signature.signature_hash, b"a" * 32),
                (signature.signature_hash, b"b" * 32),
                (signature.signature_hash, b"c" * 32),
            ],
            classifier="fake",
            classifier_model="fake/model",
            prompt_hash=b"p" * 32,
        )

    assert stored == {
        (signature.signature_hash, b"a" * 32): first,
        (signature.signature_hash, b"b" * 32): second,
    }


async def test_tool_call_classification_repository_rejects_non_persistable_classification(
    db_session_maker,
) -> None:
    signature = function_tool_signature(_bash_tool())

    async with db_session_maker() as session:
        repository = ToolClassificationRepository(session)
        await repository.get_or_create_signature(signature)

        with pytest.raises(ValueError, match="non-persistable tool call classifications must not be stored"):
            await repository.store_tool_call_classification(
                ToolCallClassification(
                    signature_hash=signature.signature_hash,
                    arguments_hash=b"a" * 32,
                    classifier="fake",
                    classifier_model="fake/model",
                    prompt_hash=b"p" * 32,
                    effect_class="unknown",
                    confidence=0.0,
                    rationale="classifier failed: ChatCompletionRateLimitError",
                    raw_output={},
                    persistable=False,
                )
            )


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


def _bash_tool() -> FunctionTool:
    return FunctionTool(
        description="Run a shell command.",
        name="bash",
        parameters={"type": "object"},
        strict=True,
        type="function",
    )


class _CountingClassifier:
    classifier = "fake"
    classifier_model = "fake/model"
    classifier_cache_model = "fake/model"
    prompt_hash = b"p" * 32

    def __init__(self) -> None:
        self.calls = 0

    async def classify_many(self, signatures: list[ToolSignature]) -> dict[bytes, ToolClassification]:
        self.calls += 1
        return {
            signature.signature_hash: ToolClassification(
                signature_hash=signature.signature_hash,
                classifier=self.classifier,
                classifier_model=self.classifier_cache_model,
                prompt_hash=self.prompt_hash,
                effect_class="safe",
                confidence=1.0,
                rationale="read-only",
                raw_output={},
            )
            for signature in signatures
        }
