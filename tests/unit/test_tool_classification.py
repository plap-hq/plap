from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from plap.llms.chat import (
    ChatAssistantMessage,
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResult,
    IChatCompletionClient,
)
from plap.responses.contracts import FunctionTool, WebSearchTool
from plap.responses.tools import (
    CachedToolPolicyResolver,
    IToolClassifier,
    LLMToolClassifier,
    StaticToolPolicyResolver,
    ToolClassification,
    ToolPolicyError,
    ToolSignature,
    function_tool_signature,
)
from plap.responses.tools.classify import (
    TOOL_EFFECT_CLASSIFIER_MAX_TOKENS,
)


def test_function_tool_signature_is_canonical_for_schema_key_order() -> None:
    first = FunctionTool(
        description="Read a file",
        name="read_file",
        parameters={"properties": {"path": {"type": "string"}}, "type": "object"},
        strict=True,
        type="function",
    )
    second = FunctionTool(
        description="Read a file",
        name="read_file",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        strict=True,
        type="function",
    )

    assert (
        function_tool_signature(first).signature_hash
        == function_tool_signature(second).signature_hash
    )


def test_function_tool_signature_changes_for_effectful_fields() -> None:
    read_file = FunctionTool(
        description="Read a file",
        name="read_file",
        parameters={"type": "object"},
        strict=True,
        type="function",
    )
    write_file = FunctionTool(
        description="Write a file",
        name="write_file",
        parameters={"type": "object"},
        strict=True,
        type="function",
    )

    assert (
        function_tool_signature(read_file).signature_hash
        != function_tool_signature(write_file).signature_hash
    )


async def test_llm_tool_classifier_parses_valid_json() -> None:
    signature = function_tool_signature(_read_file_tool())
    client = _FakeChatClient(
        '{"effect_class":"safe","confidence":0.9,"rationale":"Read-only."}'
    )
    classifier = LLMToolClassifier(
        client=client,
        classifier="fake",
        classifier_model="fake/model",
    )

    result = await classifier.classify(signature)

    assert result.effect_class == "safe"
    assert result.confidence == 0.9
    assert result.rationale == "Read-only."
    assert client.requests[0].messages[0].role == "system"
    assert "Return only JSON" in (client.requests[0].messages[0].content or "")
    assert client.requests[0].response_format is not None
    assert client.requests[0].response_format.type == "json_schema"
    assert client.requests[0].response_format.strict is True
    assert client.requests[0].response_format.schema is not None
    assert client.requests[0].response_format.schema["required"] == [
        "effect_class",
        "confidence",
        "rationale",
    ]
    assert client.requests[0].response_format.schema["properties"]["effect_class"][
        "enum"
    ] == [
        "safe",
        "mutation",
        "contextual",
        "unknown",
    ]
    assert "contextual" in (client.requests[0].messages[0].content or "")
    assert (
        "minLength"
        not in client.requests[0].response_format.schema["properties"]["rationale"]
    )
    assert client.requests[0].max_completion_tokens == TOOL_EFFECT_CLASSIFIER_MAX_TOKENS
    assert client.requests[0].temperature == 0
    assert (
        result.prompt_hash
        != LLMToolClassifier(
            client=_FakeChatClient("{}"),
            classifier="fake",
            classifier_model="fake/model",
            prompt="prompt-b",
        ).prompt_hash
    )


async def test_llm_tool_classifier_malformed_json_returns_unknown() -> None:
    classifier = LLMToolClassifier(
        client=_FakeChatClient("not json"),
        classifier="fake",
        classifier_model="fake/model",
    )

    result = await classifier.classify(function_tool_signature(_read_file_tool()))

    assert result.effect_class == "unknown"
    assert result.confidence == 0.0


async def test_llm_tool_classifier_fans_out_batch_as_isolated_requests() -> None:
    read_signature = function_tool_signature(_read_file_tool())
    list_signature = function_tool_signature(_list_files_tool())
    classifier = LLMToolClassifier(
        client=_SequenceChatClient(
            [
                '{"effect_class":"safe","confidence":0.9,"rationale":"Read-only."}',
                '{"effect_class":"safe","confidence":0.8,'
                '"rationale":"Read-only list."}',
            ]
        ),
        classifier="fake",
        classifier_model="fake/model",
    )

    classifications = await classifier.classify_many([read_signature, list_signature])

    assert set(classifications) == {
        read_signature.signature_hash,
        list_signature.signature_hash,
    }
    assert classifications[list_signature.signature_hash].confidence == 0.8
    assert len(classifier._client.requests) == 2
    first_request, second_request = classifier._client.requests
    assert read_signature.signature_hash.hex() in (
        first_request.messages[1].content or ""
    )
    assert list_signature.signature_hash.hex() in (
        second_request.messages[1].content or ""
    )


async def test_llm_tool_classifier_parses_contextual_json() -> None:
    signature = function_tool_signature(_bash_tool())
    classifier = LLMToolClassifier(
        client=_FakeChatClient(
            '{"effect_class":"contextual","confidence":0.8,'
            '"rationale":"Shell commands depend on arguments."}'
        ),
        classifier="fake",
        classifier_model="fake/model",
    )

    result = await classifier.classify(signature)

    assert result.effect_class == "contextual"
    assert result.confidence == 0.8


async def test_static_policy_resolver_uses_registry_and_unknown_client_tools() -> None:
    policies = await StaticToolPolicyResolver().resolve(
        [_read_file_tool(), WebSearchTool(type="web_search")]
    )

    assert policies["read_file"].source == "client"
    assert policies["read_file"].effect_class == "unknown"
    assert policies["web_search"].source == "server"
    assert policies["web_search"].effect_class == "safe"


async def test_policy_resolver_rejects_duplicate_names_with_different_signatures() -> (
    None
):
    resolver = StaticToolPolicyResolver()

    with pytest.raises(ToolPolicyError, match="duplicate function tool name"):
        await resolver.resolve(
            [
                _read_file_tool(),
                FunctionTool(
                    description="Mutate a file",
                    name="read_file",
                    parameters={"type": "object"},
                    strict=True,
                    type="function",
                ),
            ]
        )


async def test_cached_policy_resolver_classifies_client_tools() -> None:
    classifier = _RecordingClassifier()
    resolver = CachedToolPolicyResolver(_MemoryClassificationRepository(), classifier)

    policies = await resolver.resolve(
        [_read_file_tool(), _list_files_tool(), WebSearchTool(type="web_search")]
    )

    assert policies["read_file"].effect_class == "safe"
    assert policies["list_files"].effect_class == "safe"
    assert policies["read_file"].classification is not None
    assert policies["web_search"].source == "server"
    assert classifier.calls == 1


async def test_cached_policy_resolver_uses_l1_before_repository() -> None:
    repository = _MemoryClassificationRepository()
    classifier = _RecordingClassifier()
    resolver = CachedToolPolicyResolver(repository, classifier)

    first = await resolver.resolve([_read_file_tool()])
    second = await resolver.resolve([_read_file_tool()])

    assert second["read_file"].classification == first["read_file"].classification
    assert classifier.calls == 1
    assert repository.get_or_create_signatures_calls == 1
    assert repository.get_classifications_calls == 1
    assert repository.store_classifications_calls == 1


async def test_cached_policy_resolver_preserves_contextual_classification() -> None:
    resolver = CachedToolPolicyResolver(
        _MemoryClassificationRepository(),
        _ContextualClassifier(),
    )

    policies = await resolver.resolve([_bash_tool()])

    assert policies["bash"].effect_class == "contextual"
    assert policies["bash"].classification is not None


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


class _FakeChatClient(IChatCompletionClient):
    def __init__(self, content: str) -> None:
        self.content = content
        self.requests: list[ChatCompletionRequest] = []

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        self.requests.append(request)
        return ChatCompletionResult(
            id="chat_1",
            model=request.model,
            created_at=1.0,
            message=ChatAssistantMessage(content=self.content),
            finish_reason="stop",
        )

    def stream(
        self, _request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionDelta]:
        raise NotImplementedError


class _SequenceChatClient(IChatCompletionClient):
    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.requests: list[ChatCompletionRequest] = []

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        self.requests.append(request)
        return ChatCompletionResult(
            id=f"chat_{len(self.requests)}",
            model=request.model,
            created_at=1.0,
            message=ChatAssistantMessage(content=self.contents[len(self.requests) - 1]),
            finish_reason="stop",
        )

    def stream(
        self, _request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionDelta]:
        raise NotImplementedError


@dataclass(slots=True)
class _RecordingClassifier(IToolClassifier):
    calls: int = 0
    classifier: str = "fake"
    classifier_model: str = "fake/model"
    prompt_hash: bytes = b"p" * 32

    async def classify_many(
        self, signatures: list[ToolSignature]
    ) -> dict[bytes, ToolClassification]:
        self.calls += 1
        return {
            signature.signature_hash: ToolClassification(
                signature_hash=signature.signature_hash,
                classifier=self.classifier,
                classifier_model=self.classifier_model,
                prompt_hash=self.prompt_hash,
                effect_class="safe",
                confidence=1.0,
                rationale="read-only",
                raw_output={"effect_class": "safe"},
            )
            for signature in signatures
        }


class _ContextualClassifier(_RecordingClassifier):
    async def classify_many(
        self, signatures: list[ToolSignature]
    ) -> dict[bytes, ToolClassification]:
        self.calls += 1
        return {
            signature.signature_hash: ToolClassification(
                signature_hash=signature.signature_hash,
                classifier=self.classifier,
                classifier_model=self.classifier_model,
                prompt_hash=self.prompt_hash,
                effect_class="contextual",
                confidence=0.8,
                rationale="depends on arguments",
                raw_output={"effect_class": "contextual"},
            )
            for signature in signatures
        }


class _MemoryClassificationRepository:
    def __init__(self) -> None:
        self._cache: dict[bytes, ToolClassification] = {}
        self.get_or_create_signatures_calls = 0
        self.get_classifications_calls = 0
        self.store_classifications_calls = 0

    async def get_or_create_signatures(
        self, signatures: list[ToolSignature]
    ) -> list[ToolSignature]:
        self.get_or_create_signatures_calls += 1
        return signatures

    async def get_classifications(
        self,
        signature_hashes: list[bytes],
        *,
        classifier: str,
        classifier_model: str,
        prompt_hash: bytes,
    ) -> dict[bytes, ToolClassification]:
        self.get_classifications_calls += 1
        return {
            signature_hash: cached
            for signature_hash in signature_hashes
            if (cached := self._cache.get(signature_hash)) is not None
            and cached.classifier == classifier
            and cached.classifier_model == classifier_model
            and cached.prompt_hash == prompt_hash
        }

    async def store_classifications(
        self, classifications: list[ToolClassification]
    ) -> dict[bytes, ToolClassification]:
        self.store_classifications_calls += 1
        for classification in classifications:
            self._cache[classification.signature_hash] = classification
        return {
            classification.signature_hash: classification
            for classification in classifications
        }
