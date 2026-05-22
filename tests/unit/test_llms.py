from __future__ import annotations

from dataclasses import replace
from typing import Any

import anyio
import httpx
import pytest
from fireworks.client.error import InvalidRequestError
from openai import BadRequestError

import plap.llms.completions.router as router_module
from plap.llms.accumulator import Accumulator, Snapshot
from plap.llms.completions.chat import (
    ChatCompletionDelta,
    ChatCompletionResult,
    ChatCompletionRequest,
    ChatFunctionTool,
    ChatMessage,
    ChatPrediction,
    ChatResponseFormat,
    ChatStreamOptions,
    ChatTool,
    ChatToolCall,
    ChatToolCallDelta,
    ChatToolChoiceFunction,
    IChatCompletionClient,
)
from plap.llms.completions.client import Call, ChatCompletionClient
from plap.llms.completions.common import build_chat_body
from plap.llms.completions.errors import (
    ChatCompletionContextLengthExceededError,
    ChatCompletionProviderError,
    ChatCompletionUnsupportedRequestError,
    is_context_length_exceeded_error,
)
from plap.llms.completions.providers import (
    CROF_OPENAI_BASE_URL,
    GMICLOUD_OPENAI_BASE_URL,
    LIGHTNING_OPENAI_BASE_URL,
    NOVITA_OPENAI_BASE_URL,
    OPENROUTER_OPENAI_BASE_URL,
    OpenRouterProvider,
    build_crof_provider,
    build_fireworks_provider,
    build_gmicloud_provider,
    build_lightning_provider,
    build_novita_provider,
    build_openrouter_provider,
)
from plap.llms.completions.providers.fireworks import FireworksProvider
from plap.llms.completions.providers.openai import OpenAIProvider
from plap.llms.completions.router import ModelRoute, RoutingChatCompletionClient, UnavailableChatCompletionClient
from plap.llms.retry import RETRY_TOOL_PLACEHOLDER, complete as retry_complete, stream as retry_stream
from plap.settings import Settings


def _capture_router_logs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []

    def record(_logger: object, event: str, /, **context: object) -> None:
        events.append({"event": event, **context})

    monkeypatch.setattr(router_module, "log_debug", record)
    return events


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "api_key_pepper": "pepper",
        "database_url": "postgresql+asyncpg://example/test",
        "sealing_keys": ["a" * 43],
    }
    values.update(overrides)
    return Settings(**values)


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="model-a",
        messages=[
            ChatMessage(role="developer", content="be precise"),
            ChatMessage(role="user", content="hello", name="caller"),
        ],
        tools=[
            ChatTool(
                function=ChatFunctionTool(
                    name="lookup",
                    parameters={"type": "object"},
                    strict=True,
                    description="look something up",
                )
            )
        ],
        tool_choice=ChatToolChoiceFunction(name="lookup"),
        parallel_tool_calls=True,
        response_format=ChatResponseFormat(
            type="json_schema",
            name="answer",
            schema={"type": "object"},
            strict=True,
            description="answer schema",
        ),
        max_completion_tokens=128,
        temperature=0.2,
        top_p=0.9,
        top_k=17,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        logit_bias={"1": -10},
        logprobs=True,
        top_logprobs=3,
        stop=["END"],
        seed=7,
        n=1,
        reasoning_effort="low",
        stream_options=ChatStreamOptions(include_usage=True),
        user="user-1",
        prompt_cache_key="cache-a",
        metadata={"k": "v"},
        service_tier="flex",
        prediction=ChatPrediction(content="expected"),
    )


def _request_for_model(model: str) -> ChatCompletionRequest:
    return replace(_request(), model=model)


def _completion_response(
    *,
    model: str,
    content: str | None,
    finish_reason: str = "stop",
    reasoning_content: str | None = None,
    reasoning: str | None = None,
    reasoning_details: list[dict[str, Any]] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "content": content,
        "refusal": None,
        "reasoning_content": reasoning_content,
        "reasoning_details": reasoning_details,
        "tool_calls": tool_calls,
    }
    if reasoning is not None:
        message["reasoning"] = reasoning
    return {
        "id": "chatcmpl_1",
        "model": model,
        "created": 10,
        "system_fingerprint": "fp_1",
        "service_tier": "default",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": message,
            }
        ],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 5,
            "total_tokens": 8,
            "prompt_tokens_details": {"cached_tokens": 2},
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
    }


def _chunk(
    *,
    model: str,
    content: str | None = None,
    reasoning_content: str | None = None,
    reasoning: str | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {
        "content": content,
        "refusal": None,
        "reasoning_content": reasoning_content,
        "tool_calls": None,
    }
    if reasoning is not None:
        delta["reasoning"] = reasoning
    return {
        "id": "chatcmpl_1",
        "model": model,
        "created": 10,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "delta": delta,
            }
        ],
        "usage": None,
    }


class _AsyncListStream:
    def __init__(self, items: list[Any]) -> None:
        self._items = list(items)
        self.closed = False

    def __aiter__(self) -> _AsyncListStream:
        return self

    async def __anext__(self) -> Any:
        if not self._items:
            raise StopAsyncIteration
        item = self._items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def aclose(self) -> None:
        self.closed = True


def _next_result(results: list[Any]) -> Any:
    if not results:
        raise AssertionError("no fake result available")
    result = results.pop(0)
    if isinstance(result, Exception):
        raise result
    return result


class _FakeOpenAICompletions:
    def __init__(self, results: list[Any]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._results = list(results)

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return _next_result(self._results)


class _FakeOpenAIClient:
    def __init__(self, results: list[Any], *, base_url: str = "https://example.com/v1") -> None:
        self.chat = type("Chat", (), {"completions": _FakeOpenAICompletions(results)})()
        self.base_url = base_url


class _FakeFireworksCompletions:
    def __init__(self, *, complete_result: Any, stream_result: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self._complete_result = complete_result
        self._stream_result = stream_result

    def acreate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            if isinstance(self._stream_result, Exception):
                async def raise_stream() -> Any:
                    raise self._stream_result
                    yield
                return raise_stream()
            return _AsyncListStream(list(self._stream_result))

        async def complete() -> Any:
            if isinstance(self._complete_result, Exception):
                raise self._complete_result
            return self._complete_result

        return complete()


class _FakeFireworksClient:
    def __init__(self, *, complete_result: Any, stream_result: Any, base_url: str = "https://example.com/v1") -> None:
        self.chat = type("Chat", (), {"completions": _FakeFireworksCompletions(complete_result=complete_result, stream_result=stream_result)})()
        self.base_url = base_url


class _StubChatClient(IChatCompletionClient):
    def __init__(self, *, complete_result: Any = None, stream_result: list[Any] | None = None) -> None:
        self.complete_requests: list[ChatCompletionRequest] = []
        self.stream_requests: list[ChatCompletionRequest] = []
        self._complete_result = complete_result
        self._stream_result = list(stream_result or [])

    async def complete(self, request: ChatCompletionRequest):
        self.complete_requests.append(request)
        if isinstance(self._complete_result, Exception):
            raise self._complete_result
        return self._complete_result

    def stream(self, request: ChatCompletionRequest):
        self.stream_requests.append(request)

        async def run():
            for item in self._stream_result:
                if isinstance(item, Exception):
                    raise item
                yield item

        return run()


def _body_for(provider, request: ChatCompletionRequest, *, stream: bool) -> dict[str, Any]:
    quirks = (*provider.quirks, *provider.lookup(request.model))
    call = Call(request=request, body=build_chat_body(request, stream=stream))
    for quirk in quirks:
        quirk.request(call)
    return call.body


def _completion_result(model: str, content: str):
    from plap.llms.completions.chat import ChatCompletionResult, ChatMessage, ChatUsage

    return ChatCompletionResult(
        id="chatcmpl_test",
        model=model,
        created_at=None,
        message=ChatMessage(role="assistant", content=content),
        finish_reason="stop",
        usage=ChatUsage(input_tokens=1, output_tokens=1, total_tokens=2),
    )


def _delta(model: str, *, content_delta: str | None = None, finish_reason: str | None = None):
    from plap.llms.completions.chat import ChatCompletionDelta

    return ChatCompletionDelta(
        id="chatcmpl_test",
        model=model,
        created_at=None,
        choice_index=0,
        content_delta=content_delta,
        finish_reason=finish_reason,
    )


def test_chat_tool_call_keeps_raw_arguments() -> None:
    call = ChatToolCall(id="call_1", name="lookup", arguments="{'q':'x'}")

    assert call.arguments == "{'q':'x'}"


def test_accumulator_assembles_streamed_tool_call_and_final_result() -> None:
    from plap.llms.completions.chat import ChatCompletionDelta, ChatToolCallDelta, ChatUsage

    accumulator = Accumulator()
    first = accumulator.apply(
        ChatCompletionDelta(
            id="chatcmpl_1",
            model="model-a",
            created_at=10,
            choice_index=0,
            tool_call_delta=ChatToolCallDelta(
                index=0,
                id="call_1",
                name="lookup",
                arguments_delta="{'n':",
            ),
        )
    )
    second = accumulator.apply(
        ChatCompletionDelta(
            id="chatcmpl_1",
            model=None,
            created_at=10,
            choice_index=0,
            tool_call_delta=ChatToolCallDelta(
                index=0,
                arguments_delta="'4','x':1}",
            ),
            finish_reason="tool_calls",
            usage=ChatUsage(input_tokens=3, output_tokens=5, total_tokens=8),
        )
    )

    assert first == Snapshot(messages=first.messages, results=(), delta=first.delta)
    assert first.messages[0].tool_calls is not None
    assert first.messages[0].tool_calls[0].arguments == '{"n":""}'
    assert second.results
    assert second.messages[0].tool_calls is not None
    assert second.messages[0].tool_calls[0].arguments == '{"n":"4","x":1}'
    assert second.results[0].model == "model-a"
    assert second.results[0].finish_reason == "tool_calls"


def test_accumulator_apply_returns_snapshot_and_terminal_result() -> None:
    accumulator = Accumulator()

    first = accumulator.apply(
        ChatCompletionDelta(
            id="chatcmpl_1",
            model="model-a",
            created_at=10,
            choice_index=0,
            content_delta="hello",
        )
    )
    second = accumulator.apply(
        ChatCompletionDelta(
            id="chatcmpl_1",
            model="model-a",
            created_at=10,
            choice_index=0,
            finish_reason="stop",
        )
    )

    assert first == Snapshot(messages=first.messages, results=(), delta=first.delta)
    assert first.messages[0].content == "hello"
    assert second.results
    assert second.results[0].model == "model-a"
    assert second.results[0].message.content == "hello"


def test_accumulator_repairs_tool_call_arguments_automatically() -> None:
    tool = ChatTool(
        function=ChatFunctionTool(
            name="lookup",
            parameters={
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
                "additionalProperties": False,
            },
        )
    )
    accumulator = Accumulator(tools=(tool,))
    accumulator.apply(_delta(model="model-a", content_delta=""))
    final = accumulator.apply(
        ChatCompletionDelta(
            id="chatcmpl_1",
            model=None,
            created_at=10,
            choice_index=0,
            tool_call_delta=ChatToolCallDelta(
                index=0,
                id="call_1",
                name="lookup",
                arguments_delta="{'n':'4','x':1}",
            ),
            finish_reason="tool_calls",
        )
    )

    assert final.messages[0].tool_calls is not None
    assert final.messages[0].tool_calls[0].arguments == '{"n":4}'
    assert final.results
    assert final.results[0].message.tool_calls is not None
    assert final.results[0].message.tool_calls[0].arguments == '{"n":4}'


class _RetryStreamClient(IChatCompletionClient):
    def __init__(self, attempts: list[list[ChatCompletionDelta]]) -> None:
        self._attempts = [list(attempt) for attempt in attempts]
        self.requests: list[ChatCompletionRequest] = []

    async def complete(self, request: ChatCompletionRequest):
        raise AssertionError(f"unexpected complete for {request.model}")

    def stream(self, request: ChatCompletionRequest):
        self.requests.append(request)
        if not self._attempts:
            raise AssertionError("no retry attempt available")
        items = self._attempts.pop(0)

        async def run():
            for item in items:
                yield item

        return run()


async def test_retry_stream_retries_with_tool_stub_and_next_request() -> None:
    attempts = [
        [
            ChatCompletionDelta(
                id="chatcmpl_1",
                model="model-a",
                created_at=10,
                choice_index=0,
                tool_call_delta=ChatToolCallDelta(
                    index=0,
                    id="call_1",
                    name="lookup",
                    arguments_delta='{"q":"x"}',
                ),
                finish_reason="tool_calls",
            )
        ],
        [
            ChatCompletionDelta(
                id="chatcmpl_2",
                model="model-a",
                created_at=11,
                choice_index=0,
                content_delta="fixed",
            ),
            ChatCompletionDelta(
                id="chatcmpl_2",
                model="model-a",
                created_at=11,
                choice_index=0,
                finish_reason="stop",
            ),
        ],
    ]
    client = _RetryStreamClient(attempts)

    def validate(result: ChatCompletionResult) -> str | None:
        if result.finish_reason == "tool_calls":
            return "Your previous answer could not be used. Reply again without tool calls."
        return None

    def next_request(snapshot: Snapshot) -> ChatCompletionRequest | None:
        return ChatCompletionRequest(
            model="model-a",
            messages=[ChatMessage(role="developer", content="be precise"), *snapshot.messages],
            tools=[],
        )

    items = [
        item
        async for item in retry_stream(
            client,
            next_request=next_request,
            validate=validate,
            max_attempts=2,
        )
    ]

    assert len(items) == 4
    assert items[0].messages[-1].tool_calls is not None
    assert items[0].results and items[0].results[0].finish_reason == "tool_calls"
    assert items[1].results and items[1].results[0].finish_reason == "tool_calls"
    assert items[2].messages[-3].content == RETRY_TOOL_PLACEHOLDER
    assert items[2].messages[-2].role == "user"
    assert items[3].messages[-1].content == "fixed"
    assert len(items[3].results) == 2
    assert client.requests[1].messages[-3].tool_calls is not None
    assert client.requests[1].messages[-2].content == RETRY_TOOL_PLACEHOLDER
    assert client.requests[1].messages[-1].role == "user"


async def test_retry_complete_returns_final_snapshot() -> None:
    client = _RetryStreamClient(
        [[ChatCompletionDelta(id="chatcmpl_1", model="model-a", created_at=10, choice_index=0, finish_reason="stop")]]
    )

    def validate(result: ChatCompletionResult) -> str | None:
        _ = result
        return None

    def next_request(snapshot: Snapshot) -> ChatCompletionRequest | None:
        return ChatCompletionRequest(
            model="model-a",
            messages=[ChatMessage(role="developer", content="be precise"), *snapshot.messages],
        )

    final = await retry_complete(
        client,
        next_request=next_request,
        validate=validate,
    )

    assert final.results and final.results[0].finish_reason == "stop"


async def test_completions_client_fills_missing_result_and_delta_models() -> None:
    fake_client = _FakeOpenAIClient(
        [
            {
                **_completion_response(model="ignored-model", content="ok"),
                "model": None,
            },
            _AsyncListStream(
                [
                    {
                        **_chunk(model="ignored-model", content="ok"),
                        "model": None,
                    },
                    {
                        **_chunk(model="ignored-model", finish_reason="stop"),
                        "model": None,
                    },
                ]
            ),
        ],
        base_url=NOVITA_OPENAI_BASE_URL,
    )
    client = ChatCompletionClient(_novita_provider(client=fake_client))

    result = await client.complete(_request_for_model("openai/gpt-oss-20b"))
    deltas = [delta async for delta in client.stream(_request_for_model("openai/gpt-oss-20b"))]

    assert result.model == "openai/gpt-oss-20b"
    assert deltas[0].model == "openai/gpt-oss-20b"
    assert deltas[1].model == "openai/gpt-oss-20b"


def _lightning_provider(*, client: Any | None = None) -> OpenAIProvider:
    provider = build_lightning_provider(_settings(llm_lightning_api_key="lightning-key"))
    assert isinstance(provider, OpenAIProvider)
    if client is not None:
        provider._client = client
    return provider


def _gmicloud_provider(*, client: Any | None = None) -> OpenAIProvider:
    provider = build_gmicloud_provider(_settings(llm_gmicloud_api_key="gmicloud-key"))
    assert isinstance(provider, OpenAIProvider)
    if client is not None:
        provider._client = client
    return provider


def _novita_provider(*, client: Any | None = None) -> OpenAIProvider:
    provider = build_novita_provider(_settings(llm_novita_api_key="novita-key"))
    assert isinstance(provider, OpenAIProvider)
    if client is not None:
        provider._client = client
    return provider


def _crof_provider(*, client: Any | None = None) -> OpenAIProvider:
    provider = build_crof_provider(_settings(llm_crof_api_key="crof-key"))
    assert isinstance(provider, OpenAIProvider)
    if client is not None:
        provider._client = client
    return provider


def _openrouter_provider(*, client: Any | None = None) -> OpenRouterProvider:
    provider = build_openrouter_provider(_settings(llm_openrouter_api_key="openrouter-key"))
    assert isinstance(provider, OpenRouterProvider)
    if client is not None:
        provider._client = client
    return provider


def _fireworks_provider(*, client: Any | None = None) -> FireworksProvider:
    provider = build_fireworks_provider(_settings(llm_fireworks_api_key="fireworks-key"))
    assert isinstance(provider, FireworksProvider)
    if client is not None:
        provider._client = client
    return provider


def test_build_chat_body_preserves_full_request_shape() -> None:
    body = build_chat_body(_request(), stream=True)

    assert body["model"] == "model-a"
    assert body["messages"][0] == {"role": "developer", "content": "be precise"}
    assert body["messages"][1] == {"role": "user", "content": "hello", "name": "caller"}
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object"},
                "strict": True,
                "description": "look something up",
            },
        }
    ]
    assert body["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
    assert body["parallel_tool_calls"] is True
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {"type": "object"},
            "strict": True,
            "description": "answer schema",
        },
    }
    assert body["max_completion_tokens"] == 128
    assert body["top_k"] == 17
    assert body["reasoning_effort"] == "low"
    assert body["prompt_cache_key"] == "cache-a"
    assert body["metadata"] == {"k": "v"}
    assert body["service_tier"] == "flex"
    assert body["prediction"] == {"type": "content", "content": "expected"}
    assert body["stream_options"] == {"include_usage": True}


def test_lightning_request_quirks_keep_supported_fields_and_map_role() -> None:
    body = _body_for(_lightning_provider(), _request_for_model("lightning-ai/gpt-oss-20b"), stream=True)

    assert body["messages"][0] == {"role": "system", "content": "be precise"}
    assert body["max_completion_tokens"] == 128
    assert body["reasoning_effort"] == "low"
    assert "top_k" not in body
    assert "prompt_cache_key" not in body
    assert "service_tier" not in body
    assert "prediction" not in body


def test_gmicloud_request_quirks_map_max_tokens_and_drop_none_effort() -> None:
    request = replace(_request_for_model("XiaomiMiMo/MiMo-V2.5-Pro"), reasoning_effort="none")

    body = _body_for(_gmicloud_provider(), request, stream=True)

    assert body["messages"][0] == {"role": "system", "content": "be precise"}
    assert body["max_tokens"] == 128
    assert body["extra_body"] == {"context_length_exceeded_behavior": "error"}
    assert "max_completion_tokens" not in body
    assert "reasoning_effort" not in body
    assert "logprobs" not in body
    assert "service_tier" not in body


def test_gmicloud_provider_is_strict_and_rejects_deepseek_models() -> None:
    provider = _gmicloud_provider()

    with pytest.raises(ChatCompletionUnsupportedRequestError, match="unsupported gmicloud model"):
        provider.lookup("deepseek-ai/DeepSeek-V4-Flash")


def test_novita_request_quirks_map_forced_tool_choice() -> None:
    request = replace(_request_for_model("deepseek/deepseek-v4-flash"), reasoning_effort="high")

    body = _body_for(_novita_provider(), request, stream=False)

    assert body["messages"][0] == {"role": "system", "content": "be precise"}
    assert body["max_tokens"] == 128
    assert body["tool_choice"] == "required"
    assert "max_completion_tokens" not in body
    assert "extra_body" not in body


def test_crof_request_quirks_map_max_tokens_for_flash_model() -> None:
    request = replace(_request_for_model("glm-4.7-flash"), reasoning_effort="none", response_format=None)

    body = _body_for(_crof_provider(), request, stream=False)

    assert body["messages"][0] == {"role": "system", "content": "be precise"}
    assert body["max_tokens"] == 128
    assert "max_completion_tokens" not in body
    assert "extra_body" not in body


def test_openrouter_lookup_keeps_special_suffix_and_routes_provider_order() -> None:
    body = _body_for(
        _openrouter_provider(),
        _request_for_model("meta-llama/llama-3.3-70b-instruct:free:provider1"),
        stream=False,
    )

    assert body["model"] == "meta-llama/llama-3.3-70b-instruct:free"
    assert body["extra_body"] == {
        "provider": {
            "order": ["provider1"],
            "allow_fallbacks": False,
        }
    }


def test_openrouter_lookup_rejects_unknown_base_model_even_with_suffixes() -> None:
    provider = _openrouter_provider()

    with pytest.raises(ChatCompletionUnsupportedRequestError, match="unsupported openrouter model"):
        provider.lookup("unknown/model:nitro:provider1")


async def test_openrouter_client_aliases_reasoning_on_complete_and_stream() -> None:
    fake_client = _FakeOpenAIClient(
        [
            _completion_response(model="deepseek/deepseek-v4-flash", content="ok", reasoning="because"),
            _AsyncListStream(
                [
                    _chunk(model="deepseek/deepseek-v4-flash", content="ok", reasoning="because"),
                    _chunk(model="deepseek/deepseek-v4-flash", finish_reason="stop"),
                ]
            ),
        ],
        base_url=OPENROUTER_OPENAI_BASE_URL,
    )
    client = ChatCompletionClient(_openrouter_provider(client=fake_client))

    result = await client.complete(_request_for_model("deepseek/deepseek-v4-flash"))
    deltas = [delta async for delta in client.stream(_request_for_model("deepseek/deepseek-v4-flash"))]

    assert result.message.reasoning_content == "because"
    assert deltas[0].reasoning_delta == "because"
    assert deltas[1].finish_reason == "stop"


async def test_chat_completion_client_parses_openai_like_responses() -> None:
    fake_client = _FakeOpenAIClient(
        [
            _completion_response(
                model="openai/gpt-oss-20b",
                content="answer",
                reasoning_content="because",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {"name": "lookup", "arguments": {"q": "x"}},
                    }
                ],
            )
        ],
        base_url=NOVITA_OPENAI_BASE_URL,
    )
    client = ChatCompletionClient(_novita_provider(client=fake_client))

    result = await client.complete(_request_for_model("openai/gpt-oss-20b"))

    assert result.message.content == "answer"
    assert result.message.reasoning_content == "because"
    assert result.message.tool_calls == [ChatToolCall(id="call_1", name="lookup", arguments='{"q":"x"}')]
    assert result.usage is not None
    assert result.usage.cached_tokens == 2
    assert result.usage.reasoning_tokens == 1


async def test_openai_provider_normalizes_context_length_errors() -> None:
    error = BadRequestError(
        "This model's maximum context length is 128000 tokens. However, you requested 128001 tokens.",
        response=httpx.Response(400, request=httpx.Request("POST", "https://example.com/v1/chat/completions")),
        body={
            "error": {
                "code": "context_length_exceeded",
                "message": "This model's maximum context length is 128000 tokens. However, you requested 128001 tokens.",
                "type": "invalid_request_error",
            }
        },
    )
    fake_client = _FakeOpenAIClient([error], base_url=LIGHTNING_OPENAI_BASE_URL)
    client = ChatCompletionClient(_lightning_provider(client=fake_client))

    with pytest.raises(ChatCompletionContextLengthExceededError, match="maximum context length"):
        await client.complete(_request_for_model("lightning-ai/gpt-oss-20b"))


def test_context_length_classifier_matches_structured_codes_and_messages() -> None:
    assert is_context_length_exceeded_error({"error": {"type": "prompt-too-long"}})
    assert is_context_length_exceeded_error(
        {"detail": "Requested 128001 tokens, but the model's maximum context length is 128000 tokens."}
    )
    assert is_context_length_exceeded_error(
        {"response": {"body": {"message": "Input token count exceeds the maximum allowed token limit."}}}
    )
    assert not is_context_length_exceeded_error({"error": {"code": "payload_too_large", "message": "Payload too large."}})


async def test_lightning_gpt_oss_120b_rejects_response_format() -> None:
    fake_client = _FakeOpenAIClient([], base_url=LIGHTNING_OPENAI_BASE_URL)
    client = ChatCompletionClient(_lightning_provider(client=fake_client))
    request = ChatCompletionRequest(
        model="lightning-ai/gpt-oss-120b",
        messages=[ChatMessage(role="user", content='Return {"ok": true}.')],
        response_format=ChatResponseFormat(type="json_object"),
    )

    with pytest.raises(ChatCompletionUnsupportedRequestError, match="response_format is not supported"):
        await client.complete(request)

    assert fake_client.chat.completions.calls == []


async def test_lightning_gpt_oss_120b_stream_rejects_response_format() -> None:
    fake_client = _FakeOpenAIClient([], base_url=LIGHTNING_OPENAI_BASE_URL)
    client = ChatCompletionClient(_lightning_provider(client=fake_client))
    request = ChatCompletionRequest(
        model="lightning-ai/gpt-oss-120b",
        messages=[ChatMessage(role="user", content='Return {"ok": true}.')],
        response_format=ChatResponseFormat(type="json_object"),
    )

    with pytest.raises(ChatCompletionUnsupportedRequestError, match="response_format is not supported"):
        [delta async for delta in client.stream(request)]

    assert fake_client.chat.completions.calls == []


async def test_crof_glm_4_7_flash_rejects_response_format() -> None:
    fake_client = _FakeOpenAIClient([], base_url=CROF_OPENAI_BASE_URL)
    client = ChatCompletionClient(_crof_provider(client=fake_client))
    request = ChatCompletionRequest(
        model="glm-4.7-flash",
        messages=[ChatMessage(role="user", content="hello")],
        response_format=ChatResponseFormat(type="json_object"),
    )

    with pytest.raises(ChatCompletionUnsupportedRequestError, match="response_format is not supported"):
        await client.complete(request)

    assert fake_client.chat.completions.calls == []


async def test_fireworks_provider_uses_its_own_sdk_and_shared_parser() -> None:
    fake_client = _FakeFireworksClient(
        complete_result=_completion_response(model="accounts/fireworks/models/gpt-oss-20b", content="ok"),
        stream_result=[
            _chunk(model="accounts/fireworks/models/gpt-oss-20b", content="ok"),
            _chunk(model="accounts/fireworks/models/gpt-oss-20b", finish_reason="stop"),
        ],
    )
    provider = _fireworks_provider(client=fake_client)
    client = ChatCompletionClient(provider)

    result = await client.complete(_request_for_model("accounts/fireworks/models/gpt-oss-20b"))
    deltas = [
        delta
        async for delta in client.stream(
            replace(_request_for_model("accounts/fireworks/models/gpt-oss-20b"), stream_options=ChatStreamOptions(include_usage=True))
        )
    ]

    assert result.message.content == "ok"
    assert deltas[0].content_delta == "ok"
    assert deltas[1].finish_reason == "stop"
    assert fake_client.chat.completions.calls[0]["stream"] is False
    assert fake_client.chat.completions.calls[1]["stream"] is True


async def test_fireworks_provider_normalizes_context_length_errors() -> None:
    fake_client = _FakeFireworksClient(
        complete_result=InvalidRequestError("This prompt is too long for the model context window."),
        stream_result=[],
    )
    client = ChatCompletionClient(_fireworks_provider(client=fake_client))

    with pytest.raises(ChatCompletionContextLengthExceededError, match="context window"):
        await client.complete(_request_for_model("accounts/fireworks/models/gpt-oss-20b"))


async def test_router_complete_falls_back_before_success(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _capture_router_logs(monkeypatch)
    primary = _StubChatClient(complete_result=ChatCompletionProviderError("boom"))
    fallback = _StubChatClient(
        complete_result=_completion_result("gmicloud/XiaomiMiMo/MiMo-V2.5-Pro", "ok")
    )
    router = RoutingChatCompletionClient(
        [
            ModelRoute(prefix="crof/", client=primary),
            ModelRoute(prefix="gmicloud/", client=fallback),
        ]
    )

    result = await router.complete(
        ChatCompletionRequest(
            model="crof/qwen3.5-9b,gmicloud/XiaomiMiMo/MiMo-V2.5-Pro",
            messages=[ChatMessage(role="user", content="hello")],
        )
    )

    assert result.model == "gmicloud/XiaomiMiMo/MiMo-V2.5-Pro"
    assert primary.complete_requests[0].model == "qwen3.5-9b"
    assert fallback.complete_requests[0].model == "XiaomiMiMo/MiMo-V2.5-Pro"
    assert [event["event"] for event in events] == [
        "llm.router.attempt_failed",
        "llm.router.fallback_succeeded",
    ]


async def test_router_stream_falls_back_before_first_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _capture_router_logs(monkeypatch)
    primary = _StubChatClient(stream_result=[ChatCompletionProviderError("boom")])
    fallback = _StubChatClient(
        stream_result=[
            _delta("gmicloud/XiaomiMiMo/MiMo-V2.5-Pro", content_delta="ok"),
            _delta("gmicloud/XiaomiMiMo/MiMo-V2.5-Pro", finish_reason="stop"),
        ]
    )
    router = RoutingChatCompletionClient(
        [
            ModelRoute(prefix="crof/", client=primary),
            ModelRoute(prefix="gmicloud/", client=fallback),
        ]
    )

    deltas = [
        delta
        async for delta in router.stream(
            ChatCompletionRequest(
                model="crof/qwen3.5-9b,gmicloud/XiaomiMiMo/MiMo-V2.5-Pro",
                messages=[ChatMessage(role="user", content="hello")],
            )
        )
    ]

    assert deltas[0].model == "gmicloud/XiaomiMiMo/MiMo-V2.5-Pro"
    assert primary.stream_requests[0].model == "qwen3.5-9b"
    assert fallback.stream_requests[0].model == "XiaomiMiMo/MiMo-V2.5-Pro"
    assert [event["event"] for event in events] == [
        "llm.router.attempt_failed",
        "llm.router.fallback_succeeded",
    ]


async def test_router_stream_does_not_fallback_after_first_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _capture_router_logs(monkeypatch)
    primary = _StubChatClient(
        stream_result=[
            _delta("crof/qwen3.5-9b", content_delta="first"),
            ChatCompletionProviderError("boom"),
        ]
    )
    fallback = _StubChatClient(
        stream_result=[
            _delta("gmicloud/XiaomiMiMo/MiMo-V2.5-Pro", content_delta="fallback"),
            _delta("gmicloud/XiaomiMiMo/MiMo-V2.5-Pro", finish_reason="stop"),
        ]
    )
    router = RoutingChatCompletionClient(
        [
            ModelRoute(prefix="crof/", client=primary),
            ModelRoute(prefix="gmicloud/", client=fallback),
        ]
    )

    with pytest.raises(ChatCompletionProviderError, match="boom"):
        [
            delta
            async for delta in router.stream(
                ChatCompletionRequest(
                    model="crof/qwen3.5-9b,gmicloud/XiaomiMiMo/MiMo-V2.5-Pro",
                    messages=[ChatMessage(role="user", content="hello")],
                )
            )
        ]

    assert fallback.stream_requests == []
    assert events == []


async def test_router_stream_falls_back_after_first_delta_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _capture_router_logs(monkeypatch)

    class SlowClient(IChatCompletionClient):
        async def complete(self, request: ChatCompletionRequest):
            raise AssertionError(f"unexpected complete for {request.model}")

        def stream(self, request: ChatCompletionRequest):
            async def run():
                await anyio.sleep(0.05)
                yield _delta(request.model, content_delta="late")
            return run()

    fallback = _StubChatClient(
        stream_result=[
            _delta("gmicloud/XiaomiMiMo/MiMo-V2.5-Pro", content_delta="ok"),
            _delta("gmicloud/XiaomiMiMo/MiMo-V2.5-Pro", finish_reason="stop"),
        ]
    )
    router = RoutingChatCompletionClient(
        [
            ModelRoute(prefix="crof/", client=SlowClient()),
            ModelRoute(prefix="gmicloud/", client=fallback),
        ],
        stream_first_delta_timeout_seconds=0.01,
    )

    deltas = [
        delta
        async for delta in router.stream(
            ChatCompletionRequest(
                model="crof/qwen3.5-9b,gmicloud/XiaomiMiMo/MiMo-V2.5-Pro",
                messages=[ChatMessage(role="user", content="hello")],
            )
        )
    ]

    assert deltas[0].model == "gmicloud/XiaomiMiMo/MiMo-V2.5-Pro"
    assert [event["event"] for event in events] == [
        "llm.router.attempt_failed",
        "llm.router.fallback_succeeded",
    ]


async def test_unavailable_chat_client_rejects_unknown_models() -> None:
    client = UnavailableChatCompletionClient()

    with pytest.raises(ChatCompletionUnsupportedRequestError, match="No chat completion provider configured"):
        await client.complete(_request())
