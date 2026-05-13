from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from plap.errors import PlapError
from plap.llms.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatMessage,
    IChatCompletionClient,
)
from plap.responses.contracts import FunctionTool
from plap.responses.tools import (
    CachedToolCallPolicyResolver,
    CachedToolPolicyResolver,
    IToolCallClassifier,
    IToolClassifier,
    LLMToolCallClassifier,
    LLMToolClassifier,
    StaticToolCallPolicyResolver,
    StaticToolPolicyResolver,
    ToolCall,
    ToolCallClassification,
    ToolCallSignature,
    ToolClassification,
    ToolPolicy,
    ToolSignature,
    canonical_tool_arguments,
    function_tool_signature,
    tool_arguments_hash,
)
from plap.responses.tools.classify import (
    TOOL_CALL_EFFECT_CLASSIFIER_MAX_TOKENS,
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

    assert function_tool_signature(first).signature_hash == function_tool_signature(second).signature_hash


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

    assert function_tool_signature(read_file).signature_hash != function_tool_signature(write_file).signature_hash


async def test_llm_tool_classifier_parses_valid_json() -> None:
    signature = function_tool_signature(_read_file_tool())
    client = _FakeChatClient('{"effect_class":"safe","confidence":0.9,"rationale":"Read-only."}')
    classifier = LLMToolClassifier(
        client=client,
        classifier="fake",
        classifier_model="fake/model",
        classifier_cache_model="fake/cache",
    )

    result = await classifier.classify(signature)

    assert result.effect_class == "safe"
    assert result.confidence == 0.9
    assert result.rationale == "Read-only."
    assert client.requests[0].messages[0].role == "system"
    assert client.requests[0].response_format is not None
    assert client.requests[0].response_format.type == "json_schema"
    assert client.requests[0].response_format.strict is True
    assert client.requests[0].response_format.schema is not None
    assert client.requests[0].response_format.schema["required"] == [
        "effect_class",
        "confidence",
        "rationale",
    ]
    assert client.requests[0].response_format.schema["properties"]["effect_class"]["enum"] == [
        "safe",
        "visible",
        "mutation",
        "contextual",
    ]
    assert "minLength" not in client.requests[0].response_format.schema["properties"]["rationale"]
    assert client.requests[0].max_completion_tokens == TOOL_EFFECT_CLASSIFIER_MAX_TOKENS
    assert client.requests[0].temperature == 0
    assert (
        result.prompt_hash
        != LLMToolClassifier(
            client=_FakeChatClient("{}"),
            classifier="fake",
            classifier_model="fake/model",
            classifier_cache_model="fake/cache",
            prompt="prompt-b",
        ).prompt_hash
    )


async def test_llm_tool_classifier_malformed_json_returns_contextual() -> None:
    classifier = LLMToolClassifier(
        client=_FakeChatClient("not json"),
        classifier="fake",
        classifier_model="fake/model",
        classifier_cache_model="fake/cache",
    )

    result = await classifier.classify(function_tool_signature(_read_file_tool()))

    assert result.effect_class == "contextual"
    assert result.confidence == 0.0


async def test_llm_tool_classifier_fans_out_batch_as_isolated_requests() -> None:
    read_signature = function_tool_signature(_read_file_tool())
    list_signature = function_tool_signature(_list_files_tool())
    classifier = LLMToolClassifier(
        client=_SequenceChatClient(
            [
                '{"effect_class":"safe","confidence":0.9,"rationale":"Read-only."}',
                '{"effect_class":"safe","confidence":0.8,"rationale":"Read-only list."}',
            ]
        ),
        classifier="fake",
        classifier_model="fake/model",
        classifier_cache_model="fake/cache",
    )

    classifications = await classifier.classify_many([read_signature, list_signature])

    assert set(classifications) == {
        read_signature.signature_hash,
        list_signature.signature_hash,
    }
    assert classifications[list_signature.signature_hash].confidence == 0.8
    assert len(classifier._client.requests) == 2
    first_request, second_request = classifier._client.requests
    assert "read_file" in (first_request.messages[1].content or "")
    assert "signature_hash" not in (first_request.messages[1].content or "")
    assert read_signature.signature_hash.hex() not in (first_request.messages[1].content or "")
    assert "list_files" in (second_request.messages[1].content or "")
    assert "signature_hash" not in (second_request.messages[1].content or "")
    assert list_signature.signature_hash.hex() not in (second_request.messages[1].content or "")


async def test_llm_tool_classifier_parses_contextual_json() -> None:
    signature = function_tool_signature(_bash_tool())
    classifier = LLMToolClassifier(
        client=_FakeChatClient('{"effect_class":"contextual","confidence":0.8,"rationale":"Shell commands depend on arguments."}'),
        classifier="fake",
        classifier_model="fake/model",
        classifier_cache_model="fake/cache",
    )

    result = await classifier.classify(signature)

    assert result.effect_class == "contextual"
    assert result.confidence == 0.8


async def test_llm_tool_classifier_parses_visible_json() -> None:
    signature = function_tool_signature(
        FunctionTool(
            description="Update the visible plan.",
            name="update_plan",
            parameters={"type": "object"},
            strict=True,
            type="function",
        )
    )
    classifier = LLMToolClassifier(
        client=_FakeChatClient('{"effect_class":"visible","confidence":0.85,"rationale":"Updates user-visible agent plan only."}'),
        classifier="fake",
        classifier_model="fake/model",
        classifier_cache_model="fake/cache",
    )

    result = await classifier.classify(signature)

    assert result.effect_class == "visible"
    assert result.confidence == 0.85


def test_tool_arguments_hash_is_canonical_for_key_order() -> None:
    first = canonical_tool_arguments('{"path":"README.md","limit":10}')
    second = canonical_tool_arguments('{"limit":10,"path":"README.md"}')

    assert first == second
    assert tool_arguments_hash(first) == tool_arguments_hash(second)


def test_tool_arguments_reject_non_object_json() -> None:
    with pytest.raises(PlapError) as exc_info:
        canonical_tool_arguments('["not", "object"]')

    assert exc_info.value.public is not None
    assert exc_info.value.public.code == "invalid_tool_arguments"
    assert exc_info.value.public.param == "input"


def test_tool_arguments_reject_malformed_json() -> None:
    with pytest.raises(PlapError) as exc_info:
        canonical_tool_arguments("not json")

    assert exc_info.value.public is not None
    assert exc_info.value.public.code == "invalid_tool_arguments"
    assert exc_info.value.public.param == "input"


async def test_llm_tool_call_classifier_parses_valid_json() -> None:
    signature = function_tool_signature(_bash_tool())
    arguments = {"command": "ls"}
    arguments_hash = tool_arguments_hash(arguments)
    classifier = LLMToolCallClassifier(
        client=_FakeChatClient('{"effect_class":"safe","confidence":0.9,"rationale":"Read-only."}'),
        classifier="fake",
        classifier_model="fake/model",
        classifier_cache_model="fake/cache",
    )

    result = await classifier.classify(
        ToolCallSignature(
            signature=signature,
            arguments=arguments,
        )
    )

    assert result.effect_class == "safe"
    assert result.arguments_hash == arguments_hash
    request = classifier._client.requests[0]
    assert request.response_format is not None
    assert request.response_format.type == "json_schema"
    assert request.response_format.schema is not None
    assert request.response_format.schema["properties"]["effect_class"]["enum"] == [
        "safe",
        "visible",
        "mutation",
        "unknown",
    ]
    assert "contextual" not in str(request.response_format.schema)
    assert request.max_completion_tokens == TOOL_CALL_EFFECT_CLASSIFIER_MAX_TOKENS
    assert "bash" in (request.messages[1].content or "")
    assert "command" in (request.messages[1].content or "")
    assert "arguments_hash" not in (request.messages[1].content or "")
    assert "signature_hash" not in (request.messages[1].content or "")
    assert arguments_hash.hex() not in (request.messages[1].content or "")
    assert signature.signature_hash.hex() not in (request.messages[1].content or "")


async def test_llm_tool_call_classifier_malformed_json_returns_unknown() -> None:
    signature = function_tool_signature(_bash_tool())
    arguments = {"command": "rm -rf tmp"}
    classifier = LLMToolCallClassifier(
        client=_FakeChatClient("not json"),
        classifier="fake",
        classifier_model="fake/model",
        classifier_cache_model="fake/cache",
    )

    result = await classifier.classify(
        ToolCallSignature(
            signature=signature,
            arguments=arguments,
        )
    )

    assert result.effect_class == "unknown"
    assert result.confidence == 0.0


async def test_llm_tool_call_classifier_parses_visible_json() -> None:
    signature = function_tool_signature(
        FunctionTool(
            description="Update the visible plan.",
            name="update_plan",
            parameters={"type": "object"},
            strict=True,
            type="function",
        )
    )
    classifier = LLMToolCallClassifier(
        client=_FakeChatClient('{"effect_class":"visible","confidence":0.85,"rationale":"Updates visible plan state only."}'),
        classifier="fake",
        classifier_model="fake/model",
        classifier_cache_model="fake/cache",
    )

    result = await classifier.classify(
        ToolCallSignature(
            signature=signature,
            arguments={"step": "Check config"},
        )
    )

    assert result.effect_class == "visible"
    assert result.confidence == 0.85


async def test_static_policy_resolver_returns_contextual_client_tools() -> None:
    policies = await StaticToolPolicyResolver().resolve([_read_file_tool()])

    assert policies["read_file"].source == "client"
    assert policies["read_file"].effect_class == "contextual"


async def test_policy_resolver_rejects_duplicate_names_with_different_signatures() -> None:
    resolver = StaticToolPolicyResolver()

    with pytest.raises(PlapError) as exc_info:
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

    assert exc_info.value.public is not None
    assert exc_info.value.public.code == "invalid_tool_definition"
    assert exc_info.value.public.param == "input"


async def test_cached_policy_resolver_classifies_client_tools() -> None:
    classifier = _RecordingClassifier()
    resolver = CachedToolPolicyResolver(_MemoryClassificationRepository(), classifier)

    policies = await resolver.resolve([_read_file_tool(), _list_files_tool()])

    assert policies["read_file"].effect_class == "safe"
    assert policies["list_files"].effect_class == "safe"
    assert policies["read_file"].classification is not None
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


async def test_cached_policy_resolver_does_not_store_non_persistable_fallbacks() -> None:
    repository = _MemoryClassificationRepository()
    classifier = _FallbackClassifier()
    resolver = CachedToolPolicyResolver(repository, classifier)

    first = await resolver.resolve([_read_file_tool()])
    second = await resolver.resolve([_read_file_tool()])

    assert first["read_file"].classification is not None
    assert first["read_file"].classification.persistable is False
    assert second["read_file"].classification is not None
    assert second["read_file"].classification.persistable is False
    assert classifier.calls == 2
    assert repository.store_classifications_calls == 0


async def test_cached_policy_resolver_uses_shared_empty_l1() -> None:
    repository = _MemoryClassificationRepository()
    classifier = _RecordingClassifier()
    shared_l1 = {}
    first = CachedToolPolicyResolver(
        repository,
        classifier,
        classification_l1=shared_l1,
    )
    second = CachedToolPolicyResolver(
        repository,
        classifier,
        classification_l1=shared_l1,
    )

    await first.resolve([_read_file_tool()])
    await second.resolve([_read_file_tool()])

    assert classifier.calls == 1
    assert repository.get_classifications_calls == 1


async def test_cached_policy_resolver_reuses_cache_model_across_execution_routes() -> None:
    repository = _MemoryClassificationRepository()
    first = _RecordingClassifier(
        classifier_model="lightning/lightning-ai/gpt-oss-20b",
        classifier_cache_model="gpt-oss-20b",
    )
    second = _RecordingClassifier(
        classifier_model="lightning/lightning-ai/gpt-oss-20b,openrouter/openai/gpt-oss-20b:deepinfra",
        classifier_cache_model="gpt-oss-20b",
    )

    await CachedToolPolicyResolver(repository, first).resolve([_read_file_tool()])
    await CachedToolPolicyResolver(repository, second).resolve([_read_file_tool()])

    assert first.calls == 1
    assert second.calls == 0
    assert repository.store_classifications_calls == 1


async def test_cached_policy_resolver_preserves_contextual_classification() -> None:
    resolver = CachedToolPolicyResolver(
        _MemoryClassificationRepository(),
        _ContextualClassifier(),
    )

    policies = await resolver.resolve([_bash_tool()])

    assert policies["bash"].effect_class == "contextual"
    assert policies["bash"].classification is not None


async def test_static_tool_call_policy_resolver_maps_contextual_to_unknown() -> None:
    resolver = StaticToolCallPolicyResolver()

    policies = await resolver.resolve(
        [
            ToolCall(
                tool=_bash_tool(),
                policy=ToolPolicy(name="bash", source="client", effect_class="contextual"),
                arguments='{"command":"ls"}',
            )
        ]
    )

    policy = policies[0]
    assert policy.effect_class == "unknown"
    assert policy.classification is None


async def test_cached_tool_call_policy_resolver_skips_non_contextual_policies() -> None:
    classifier = _RecordingToolCallClassifier()
    resolver = CachedToolCallPolicyResolver(
        _MemoryClassificationRepository(),
        classifier,
    )

    policies = await resolver.resolve(
        [
            ToolCall(
                tool=_read_file_tool(),
                policy=ToolPolicy(
                    name="read_file",
                    source="client",
                    effect_class="safe",
                ),
                arguments='{"path":"README.md"}',
            )
        ]
    )

    policy = policies[0]
    assert policy.effect_class == "safe"
    assert policy.classification is None
    assert classifier.calls == 0


async def test_cached_tool_call_policy_resolver_classifies_contextual_calls() -> None:
    repository = _MemoryClassificationRepository()
    classifier = _RecordingToolCallClassifier()
    resolver = CachedToolCallPolicyResolver(repository, classifier)

    calls = [
        ToolCall(
            tool=_bash_tool(),
            policy=ToolPolicy(name="bash", source="client", effect_class="contextual"),
            arguments='{"command":"ls"}',
        )
    ]

    first = await resolver.resolve(calls)
    second = await resolver.resolve(calls)

    assert first[0].effect_class == "safe"
    assert first[0].classification is not None
    assert second[0].classification == first[0].classification
    assert classifier.calls == 1
    assert repository.get_tool_call_classifications_calls == 1
    assert repository.store_tool_call_classifications_calls == 1


async def test_cached_tool_call_policy_resolver_does_not_store_non_persistable_fallbacks() -> None:
    repository = _MemoryClassificationRepository()
    classifier = _FallbackToolCallClassifier()
    resolver = CachedToolCallPolicyResolver(repository, classifier)
    calls = [
        ToolCall(
            tool=_bash_tool(),
            policy=ToolPolicy(name="bash", source="client", effect_class="contextual"),
            arguments='{"command":"ls"}',
        )
    ]

    first = await resolver.resolve(calls)
    second = await resolver.resolve(calls)

    assert first[0].classification is not None
    assert first[0].classification.persistable is False
    assert second[0].classification is not None
    assert second[0].classification.persistable is False
    assert classifier.calls == 2
    assert repository.store_tool_call_classifications_calls == 0


async def test_cached_tool_call_policy_resolver_preserves_order() -> None:
    repository = _MemoryClassificationRepository()
    classifier = _RecordingToolCallClassifier()
    resolver = CachedToolCallPolicyResolver(repository, classifier)

    policies = await resolver.resolve(
        [
            ToolCall(
                tool=_read_file_tool(),
                policy=ToolPolicy(
                    name="read_file",
                    source="client",
                    effect_class="safe",
                ),
                arguments='{"path":"README.md"}',
            ),
            ToolCall(
                tool=_bash_tool(),
                policy=ToolPolicy(name="bash", source="client", effect_class="contextual"),
                arguments='{"command":"ls"}',
            ),
            ToolCall(
                tool=_list_files_tool(),
                policy=ToolPolicy(
                    name="list_files",
                    source="client",
                    effect_class="mutation",
                ),
                arguments='{"path":"tmp"}',
            ),
        ]
    )

    assert [policy.name for policy in policies] == ["read_file", "bash", "list_files"]
    assert [policy.effect_class for policy in policies] == ["safe", "safe", "mutation"]


async def test_cached_tool_call_policy_resolver_uses_shared_empty_l1() -> None:
    repository = _MemoryClassificationRepository()
    classifier = _RecordingToolCallClassifier()
    shared_l1 = {}
    first = CachedToolCallPolicyResolver(
        repository,
        classifier,
        classification_l1=shared_l1,
    )
    second = CachedToolCallPolicyResolver(
        repository,
        classifier,
        classification_l1=shared_l1,
    )

    calls = [
        ToolCall(
            tool=_bash_tool(),
            policy=ToolPolicy(
                name="bash",
                source="client",
                effect_class="contextual",
            ),
            arguments='{"command":"ls"}',
        )
    ]
    await first.resolve(calls)
    await second.resolve(calls)

    assert classifier.calls == 1
    assert repository.get_tool_call_classifications_calls == 1


async def test_cached_tool_call_policy_resolver_reuses_cache_model_across_execution_routes() -> None:
    repository = _MemoryClassificationRepository()
    first = _RecordingToolCallClassifier(
        classifier_model="lightning/lightning-ai/gpt-oss-20b",
        classifier_cache_model="gpt-oss-20b",
    )
    second = _RecordingToolCallClassifier(
        classifier_model="lightning/lightning-ai/gpt-oss-20b,openrouter/openai/gpt-oss-20b:deepinfra",
        classifier_cache_model="gpt-oss-20b",
    )
    calls = [
        ToolCall(
            tool=_bash_tool(),
            policy=ToolPolicy(name="bash", source="client", effect_class="contextual"),
            arguments='{"command":"ls"}',
        )
    ]

    await CachedToolCallPolicyResolver(repository, first).resolve(calls)
    await CachedToolCallPolicyResolver(repository, second).resolve(calls)

    assert first.calls == 1
    assert second.calls == 0
    assert repository.store_tool_call_classifications_calls == 1


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
            message=ChatMessage(role="assistant", content=self.content),
            finish_reason="stop",
        )

    def stream(self, _request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionDelta]:
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
            message=ChatMessage(role="assistant", content=self.contents[len(self.requests) - 1]),
            finish_reason="stop",
        )

    def stream(self, _request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionDelta]:
        raise NotImplementedError


@dataclass(slots=True)
class _RecordingClassifier(IToolClassifier):
    calls: int = 0
    classifier: str = "fake"
    classifier_model: str = "fake/model"
    classifier_cache_model: str = "fake/cache"
    prompt_hash: bytes = b"p" * 32

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
                raw_output={"effect_class": "safe"},
            )
            for signature in signatures
        }


class _ContextualClassifier(_RecordingClassifier):
    async def classify_many(self, signatures: list[ToolSignature]) -> dict[bytes, ToolClassification]:
        self.calls += 1
        return {
            signature.signature_hash: ToolClassification(
                signature_hash=signature.signature_hash,
                classifier=self.classifier,
                classifier_model=self.classifier_cache_model,
                prompt_hash=self.prompt_hash,
                effect_class="contextual",
                confidence=0.8,
                rationale="depends on arguments",
                raw_output={"effect_class": "contextual"},
            )
            for signature in signatures
        }


class _FallbackClassifier(_RecordingClassifier):
    async def classify_many(self, signatures: list[ToolSignature]) -> dict[bytes, ToolClassification]:
        self.calls += 1
        return {
            signature.signature_hash: ToolClassification(
                signature_hash=signature.signature_hash,
                classifier=self.classifier,
                classifier_model=self.classifier_cache_model,
                prompt_hash=self.prompt_hash,
                effect_class="contextual",
                confidence=0.0,
                rationale="classifier failed: ChatCompletionRateLimitError",
                raw_output={},
                persistable=False,
            )
            for signature in signatures
        }


@dataclass(slots=True)
class _RecordingToolCallClassifier(IToolCallClassifier):
    calls: int = 0
    classifier: str = "fake"
    classifier_model: str = "fake/model"
    classifier_cache_model: str = "fake/cache"
    prompt_hash: bytes = b"c" * 32

    async def classify_many(self, calls: list[ToolCallSignature]) -> dict[tuple[bytes, bytes], ToolCallClassification]:
        self.calls += 1
        return {
            call.classification_key: ToolCallClassification(
                signature_hash=call.signature_hash,
                arguments_hash=call.arguments_hash,
                classifier=self.classifier,
                classifier_model=self.classifier_cache_model,
                prompt_hash=self.prompt_hash,
                effect_class="safe",
                confidence=1.0,
                rationale="read-only call",
                raw_output={"effect_class": "safe"},
            )
            for call in calls
        }


class _FallbackToolCallClassifier(_RecordingToolCallClassifier):
    async def classify_many(self, calls: list[ToolCallSignature]) -> dict[tuple[bytes, bytes], ToolCallClassification]:
        self.calls += 1
        return {
            call.classification_key: ToolCallClassification(
                signature_hash=call.signature_hash,
                arguments_hash=call.arguments_hash,
                classifier=self.classifier,
                classifier_model=self.classifier_cache_model,
                prompt_hash=self.prompt_hash,
                effect_class="unknown",
                confidence=0.0,
                rationale="classifier failed: ChatCompletionRateLimitError",
                raw_output={},
                persistable=False,
            )
            for call in calls
        }


class _MemoryClassificationRepository:
    def __init__(self) -> None:
        self._cache: dict[bytes, ToolClassification] = {}
        self._call_cache: dict[tuple[bytes, bytes], ToolCallClassification] = {}
        self.get_or_create_signatures_calls = 0
        self.get_classifications_calls = 0
        self.store_classifications_calls = 0
        self.get_tool_call_classifications_calls = 0
        self.store_tool_call_classifications_calls = 0

    async def get_or_create_signature(self, signature: ToolSignature) -> ToolSignature:
        self.get_or_create_signatures_calls += 1
        return signature

    async def get_or_create_signatures(self, signatures: list[ToolSignature]) -> list[ToolSignature]:
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

    async def store_classifications(self, classifications: list[ToolClassification]) -> dict[bytes, ToolClassification]:
        self.store_classifications_calls += 1
        for classification in classifications:
            self._cache[classification.signature_hash] = classification
        return {classification.signature_hash: classification for classification in classifications}

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
        self.get_tool_call_classifications_calls += 1
        return {
            key: cached
            for key in keys
            if (cached := self._call_cache.get(key)) is not None
            and cached.classifier == classifier
            and cached.classifier_model == classifier_model
            and cached.prompt_hash == prompt_hash
        }

    async def store_tool_call_classification(self, classification: ToolCallClassification) -> ToolCallClassification:
        stored = await self.store_tool_call_classifications([classification])
        return stored[(classification.signature_hash, classification.arguments_hash)]

    async def store_tool_call_classifications(
        self, classifications: list[ToolCallClassification]
    ) -> dict[tuple[bytes, bytes], ToolCallClassification]:
        self.store_tool_call_classifications_calls += 1
        for classification in classifications:
            self._call_cache[(classification.signature_hash, classification.arguments_hash)] = classification
        return {
            (
                classification.signature_hash,
                classification.arguments_hash,
            ): classification
            for classification in classifications
        }
