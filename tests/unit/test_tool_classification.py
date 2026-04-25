from __future__ import annotations

from dataclasses import dataclass

import pytest

from plap.llms.chat import (
    ChatAssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResult,
)
from plap.responses.contracts import FunctionTool, WebSearchTool
from plap.responses.tools import (
    CachedToolPolicyResolver,
    LLMToolClassifier,
    StaticToolPolicyResolver,
    function_tool_signature,
)
from plap.responses.tools.classifier import (
    TOOL_EFFECT_CLASSIFIER_MAX_TOKENS,
    ToolClassifier,
)
from plap.responses.tools.policy import ToolPolicyError
from plap.responses.tools.types import ToolClassification, ToolSignature


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
    client = _FakeChatClient(
        '{"effect_class":"safe","confidence":0.9,"rationale":"Read-only."}'
    )
    classifier = LLMToolClassifier(
        client=client,
        classifier="fake",
        classifier_model="fake/model",
    )
    signature = function_tool_signature(_read_file_tool())

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
    resolver = CachedToolPolicyResolver(_MemoryCachedClassifier(classifier))

    policies = await resolver.resolve(
        [_read_file_tool(), _list_files_tool(), WebSearchTool(type="web_search")]
    )

    assert policies["read_file"].effect_class == "safe"
    assert policies["list_files"].effect_class == "safe"
    assert policies["read_file"].classification is not None
    assert policies["web_search"].source == "server"
    assert classifier.calls == 2


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


class _FakeChatClient:
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


@dataclass(slots=True)
class _RecordingClassifier:
    calls: int = 0
    classifier: str = "fake"
    classifier_model: str = "fake/model"
    prompt_hash: bytes = b"p" * 32

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
            raw_output={"effect_class": "safe"},
        )


class _MemoryCachedClassifier:
    def __init__(self, classifier: ToolClassifier) -> None:
        self._classifier = classifier
        self._cache: dict[bytes, ToolClassification] = {}

    async def classify(self, signature: ToolSignature) -> ToolClassification:
        return (await self.classify_many([signature]))[signature.signature_hash]

    async def classify_many(
        self, signatures: list[ToolSignature]
    ) -> dict[bytes, ToolClassification]:
        classifications: dict[bytes, ToolClassification] = {}
        missing: list[ToolSignature] = []
        for signature in signatures:
            cached = self._cache.get(signature.signature_hash)
            if cached is not None:
                classifications[signature.signature_hash] = cached
            else:
                missing.append(signature)
        for signature in missing:
            classification = await self._classifier.classify(signature)
            self._cache[signature.signature_hash] = classification
            classifications[signature.signature_hash] = classification
        return classifications
