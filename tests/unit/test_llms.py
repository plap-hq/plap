from __future__ import annotations

from dataclasses import replace
from typing import Any

import anyio
import httpx
import pytest
from fireworks.client.error import InvalidRequestError
from openai import APITimeoutError, BadRequestError

import plap.llms.completions.providers.openai as openai_provider_module
import plap.llms.completions.quirks as quirks_module
import plap.llms.completions.router as router_module
from plap.llms.accumulator import Accumulator, Snapshot
from plap.llms.completions.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResult,
    ChatFunctionTool,
    ChatMessage,
    ChatPrediction,
    ChatResponseFormat,
    ChatStreamOptions,
    ChatTool,
    ChatToolCall,
    ChatToolCallDelta,
    ChatToolChoiceFunction,
    ChatUsage,
    IChatCompletionClient,
)
from plap.llms.completions.client import Call, ChatCompletionClient, Provider
from plap.llms.completions.common import build_chat_body
from plap.llms.completions.errors import (
    ChatCompletionContextLengthExceededError,
    ChatCompletionInvalidRequestError,
    ChatCompletionProviderError,
    ChatCompletionRateLimitError,
    ChatCompletionTimeoutError,
    ChatCompletionUnsupportedRequestError,
    is_context_length_exceeded_error,
)
from plap.llms.completions.providers import (
    CEREBRAS_OPENAI_BASE_URL,
    CROF_OPENAI_BASE_URL,
    GROQ_OPENAI_BASE_URL,
    LIGHTNING_OPENAI_BASE_URL,
    NOVITA_OPENAI_BASE_URL,
    OPENROUTER_OPENAI_BASE_URL,
    OpenRouterProvider,
    build_cerebras_provider,
    build_crof_provider,
    build_fireworks_provider,
    build_gmicloud_provider,
    build_groq_provider,
    build_lightning_provider,
    build_novita_provider,
    build_openrouter_provider,
    build_qubrid_provider,
)
from plap.llms.completions.providers.fireworks import FireworksProvider
from plap.llms.completions.providers.openai import OpenAIProvider
from plap.llms.completions.router import ModelRoute, RoutingChatCompletionClient, UnavailableChatCompletionClient
from plap.llms.retry import RETRY_TOOL_PLACEHOLDER, RetryLimitExceededError, RetryToolSchemaError, retry_on_unusable_tool_calls
from plap.llms.retry import complete as retry_complete
from plap.llms.retry import stream as retry_stream
from plap.settings import Settings


def _capture_router_logs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []

    def record(_logger: object, event: str, /, **context: object) -> None:
        events.append({"event": event, **context})

    monkeypatch.setattr(router_module, "log_debug", record)
    return events


def _capture_router_retry_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def record(delay_seconds: float) -> None:
        delays.append(delay_seconds)

    monkeypatch.setattr(router_module, "_sleep_for_transient_retry", record)
    return delays


def _capture_openai_provider_logs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []

    def record(_logger: object, event: str, /, **context: object) -> None:
        events.append({"event": event, **context})

    monkeypatch.setattr(openai_provider_module, "log_debug", record)
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
        self.chat = type(
            "Chat", (), {"completions": _FakeFireworksCompletions(complete_result=complete_result, stream_result=stream_result)}
        )()
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


class _StaticProvider(Provider):
    def __init__(
        self,
        *,
        quirks: tuple[Any, ...] = (),
        models: dict[str, tuple[Any, ...]] | None = None,
        complete_raw: dict[str, Any] | None = None,
        stream_raw: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(name="static", quirks=quirks, models=models)
        self.complete_calls: list[Call] = []
        self.stream_calls: list[Call] = []
        self._complete_raw = complete_raw or _completion_response(model="model-a", content="ok")
        self._stream_raw = list(
            stream_raw
            or [
                _chunk(model="model-a", content="ok"),
                _chunk(model="model-a", finish_reason="stop"),
            ]
        )

    async def complete(self, call: Call) -> dict[str, Any]:
        self.complete_calls.append(call)
        return self._complete_raw

    def stream(self, call: Call):
        self.stream_calls.append(call)

        async def run():
            for item in self._stream_raw:
                yield item

        return run()


def _body_for(provider, request: ChatCompletionRequest, *, stream: bool) -> dict[str, Any]:
    quirks = (*provider.quirks, *provider.lookup(request.model))
    call = Call(request=request, body=build_chat_body(request, stream=stream))
    for quirk in quirks:
        quirk.request(call)
    return call.body


def _completion_result(model: str, content: str):
    return ChatCompletionResult(
        id="chatcmpl_test",
        model=model,
        created_at=None,
        message=ChatMessage(role="assistant", content=content),
        finish_reason="stop",
        usage=ChatUsage(input_tokens=1, output_tokens=1, total_tokens=2),
    )


def _delta(model: str, *, content_delta: str | None = None, finish_reason: str | None = None):
    return ChatCompletionDelta(
        id="chatcmpl_test",
        model=model,
        created_at=None,
        choice_index=0,
        content_delta=content_delta,
        finish_reason=finish_reason,
    )


def _simple_request(model: str = "model-a") -> ChatCompletionRequest:
    return ChatCompletionRequest(model=model, messages=[ChatMessage(role="user", content="hello")])


def test_chat_tool_call_keeps_raw_arguments() -> None:
    call = ChatToolCall(id="call_1", name="lookup", arguments="{'q':'x'}")

    assert call.arguments == "{'q':'x'}"


def test_accumulator_assembles_streamed_tool_call_and_final_result() -> None:
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
    def __init__(self, attempts: list[list[object]]) -> None:
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
                if isinstance(item, Exception):
                    raise item
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

    def validate(result: ChatCompletionResult, request: ChatCompletionRequest) -> str | None:
        _ = request
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
            validators=(validate,),
            max_attempts=2,
        )
    ]

    assert len(items) == 4
    assert items[0].messages[-1].tool_calls is not None
    assert items[0].results and items[0].results[0].finish_reason == "tool_calls"
    assert items[1].results and items[1].results[0].finish_reason == "tool_calls"
    assert items[1].messages[-2].content == RETRY_TOOL_PLACEHOLDER
    assert items[1].messages[-1].role == "user"
    assert items[3].messages[-1].content == "fixed"
    assert len(items[3].results) == 2
    assert client.requests[1].messages[-3].tool_calls is not None
    assert client.requests[1].messages[-2].content == RETRY_TOOL_PLACEHOLDER
    assert client.requests[1].messages[-1].role == "user"


async def test_retry_complete_returns_final_snapshot() -> None:
    client = _RetryStreamClient(
        [[ChatCompletionDelta(id="chatcmpl_1", model="model-a", created_at=10, choice_index=0, finish_reason="stop")]]
    )

    def validate(result: ChatCompletionResult, request: ChatCompletionRequest) -> str | None:
        _ = request
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
        validators=(validate,),
    )

    assert final.results and final.results[0].finish_reason == "stop"


async def test_retry_stream_retries_after_partial_stream_timeout_without_persisting_partial_message() -> None:
    client = _RetryStreamClient(
        [
            [
                ChatCompletionDelta(
                    id="chatcmpl_1",
                    model="model-a",
                    created_at=10,
                    choice_index=0,
                    content_delta="partial",
                ),
                ChatCompletionTimeoutError("idle timeout"),
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
    )

    def validate(result: ChatCompletionResult, request: ChatCompletionRequest) -> str | None:
        _ = request
        _ = result
        return None

    def next_request(snapshot: Snapshot) -> ChatCompletionRequest | None:
        return ChatCompletionRequest(
            model="model-a",
            messages=[ChatMessage(role="developer", content="be precise"), *snapshot.messages],
        )

    items = [
        item
        async for item in retry_stream(
            client,
            next_request=next_request,
            validators=(validate,),
            max_attempts=2,
        )
    ]

    assert items[0].messages[-1].content == "partial"
    assert items[1] == Snapshot(messages=(), results=(), delta=None)
    assert items[-1].results and items[-1].results[-1].finish_reason == "stop"
    assert items[-1].messages[-1].content == "fixed"
    assert client.requests[1].messages == [ChatMessage(role="developer", content="be precise")]


async def test_retry_complete_accepts_final_result_when_stream_times_out_after_finish_reason() -> None:
    client = _RetryStreamClient(
        [
            [
                ChatCompletionDelta(
                    id="chatcmpl_1",
                    model="model-a",
                    created_at=10,
                    choice_index=0,
                    content_delta="done",
                ),
                ChatCompletionDelta(
                    id="chatcmpl_1",
                    model="model-a",
                    created_at=10,
                    choice_index=0,
                    finish_reason="stop",
                ),
                ChatCompletionTimeoutError("idle timeout"),
            ]
        ]
    )

    def validate(result: ChatCompletionResult, request: ChatCompletionRequest) -> str | None:
        _ = request
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
        validators=(validate,),
    )

    assert final.results and final.results[-1].finish_reason == "stop"
    assert final.messages[-1].content == "done"
    assert len(client.requests) == 1


async def test_retry_stream_retries_on_non_object_tool_arguments() -> None:
    tool = ChatTool(
        function=ChatFunctionTool(
            name="read_file",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        )
    )
    client = _RetryStreamClient(
        [
            [
                ChatCompletionDelta(
                    id="chatcmpl_1",
                    model="model-a",
                    created_at=10,
                    choice_index=0,
                    tool_call_delta=ChatToolCallDelta(
                        index=0,
                        id="call_1",
                        name="read_file",
                        arguments_delta='["README.md"]',
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
    )

    def next_request(snapshot: Snapshot) -> ChatCompletionRequest | None:
        return ChatCompletionRequest(
            model="model-a",
            messages=[ChatMessage(role="developer", content="be precise"), *snapshot.messages],
            tools=[tool],
        )

    items = [
        item
        async for item in retry_stream(
            client,
            next_request=next_request,
            validators=(retry_on_unusable_tool_calls,),
            max_attempts=2,
        )
    ]

    assert len(items) == 4
    assert items[1].results and items[1].results[0].message.tool_calls is not None
    assert items[1].messages[-2].content == RETRY_TOOL_PLACEHOLDER
    assert items[1].messages[-1].role == "user"
    assert client.requests[1].messages[-3].tool_calls is not None
    assert client.requests[1].messages[-2].content == RETRY_TOOL_PLACEHOLDER
    assert client.requests[1].messages[-1].role == "user"
    assert items[-1].messages[-1].content == "fixed"


async def test_retry_stream_retries_on_strict_tool_schema_mismatch() -> None:
    tool = ChatTool(
        function=ChatFunctionTool(
            name="lookup",
            parameters={
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
                "additionalProperties": False,
            },
            strict=True,
        )
    )
    client = _RetryStreamClient(
        [
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
                        arguments_delta='{"n":null}',
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
    )

    def next_request(snapshot: Snapshot) -> ChatCompletionRequest | None:
        return ChatCompletionRequest(
            model="model-a",
            messages=[ChatMessage(role="developer", content="be precise"), *snapshot.messages],
            tools=[tool],
        )

    items = [
        item
        async for item in retry_stream(
            client,
            next_request=next_request,
            validators=(retry_on_unusable_tool_calls,),
            max_attempts=2,
        )
    ]

    assert len(items) == 4
    assert items[1].results and items[1].results[0].message.tool_calls is not None
    assert items[1].messages[-2].content == RETRY_TOOL_PLACEHOLDER
    assert items[1].messages[-1].role == "user"
    assert items[-1].messages[-1].content == "fixed"


async def test_retry_stream_raises_retry_limit_exceeded_when_attempts_are_exhausted() -> None:
    client = _RetryStreamClient(
        [
            [
                ChatCompletionDelta(
                    id="chatcmpl_1",
                    model="model-a",
                    created_at=10,
                    choice_index=0,
                    finish_reason="tool_calls",
                )
            ]
        ]
    )

    def validate(result: ChatCompletionResult, request: ChatCompletionRequest) -> str | None:
        _ = request
        if result.finish_reason == "tool_calls":
            return "Reply again without tool calls."
        return None

    def next_request(snapshot: Snapshot) -> ChatCompletionRequest | None:
        return ChatCompletionRequest(
            model="model-a",
            messages=[ChatMessage(role="developer", content="be precise"), *snapshot.messages],
        )

    stream = retry_stream(
        client,
        next_request=next_request,
        validators=(validate,),
        max_attempts=1,
    )
    items: list[Snapshot] = []
    with pytest.raises(RetryLimitExceededError, match="retry limit reached"):
        items.append(await anext(stream))
        items.append(await anext(stream))
        await anext(stream)

    assert len(items) == 2
    assert items[0].results and items[0].results[0].finish_reason == "tool_calls"
    assert items[1].messages[-1].role == "user"


def test_retry_on_unusable_tool_calls_returns_retry_message_for_unknown_tool_name() -> None:
    request = ChatCompletionRequest(
        model="model-a",
        messages=[ChatMessage(role="developer", content="be precise")],
        tools=[ChatTool(function=ChatFunctionTool(name="lookup", parameters={"type": "object"}))],
    )
    result = ChatCompletionResult(
        id="chatcmpl_1",
        model="model-a",
        created_at=10,
        message=ChatMessage(
            role="assistant",
            tool_calls=[ChatToolCall(id="call_1", name="read_file", arguments='{"path":"README.md"}')],
        ),
        finish_reason="tool_calls",
    )

    retry_message = retry_on_unusable_tool_calls(result, request)

    assert retry_message is not None
    assert "`read_file`" in retry_message
    assert "`lookup`" in retry_message


def test_retry_on_unusable_tool_calls_returns_retry_message_for_non_object_arguments() -> None:
    request = ChatCompletionRequest(
        model="model-a",
        messages=[ChatMessage(role="developer", content="be precise")],
        tools=[ChatTool(function=ChatFunctionTool(name="read_file", parameters={"type": "object"}))],
    )
    result = ChatCompletionResult(
        id="chatcmpl_1",
        model="model-a",
        created_at=10,
        message=ChatMessage(
            role="assistant",
            tool_calls=[ChatToolCall(id="call_1", name="read_file", arguments='["README.md"]')],
        ),
        finish_reason="tool_calls",
    )

    retry_message = retry_on_unusable_tool_calls(result, request)

    assert retry_message is not None
    assert "`read_file`" in retry_message
    assert "JSON object" in retry_message


def test_retry_on_unusable_tool_calls_returns_retry_message_for_strict_schema_mismatch() -> None:
    request = ChatCompletionRequest(
        model="model-a",
        messages=[ChatMessage(role="developer", content="be precise")],
        tools=[
            ChatTool(
                function=ChatFunctionTool(
                    name="lookup",
                    parameters={
                        "type": "object",
                        "properties": {"n": {"type": "integer"}},
                        "required": ["n"],
                        "additionalProperties": False,
                    },
                    strict=True,
                )
            )
        ],
    )
    result = ChatCompletionResult(
        id="chatcmpl_1",
        model="model-a",
        created_at=10,
        message=ChatMessage(
            role="assistant",
            tool_calls=[ChatToolCall(id="call_1", name="lookup", arguments='{"n":null}')],
        ),
        finish_reason="tool_calls",
    )

    retry_message = retry_on_unusable_tool_calls(result, request)

    assert retry_message is not None
    assert "`lookup`" in retry_message
    assert "data.n must be integer" in retry_message


def test_retry_on_unusable_tool_calls_raises_for_uncompilable_strict_schema() -> None:
    request = ChatCompletionRequest(
        model="model-a",
        messages=[ChatMessage(role="developer", content="be precise")],
        tools=[ChatTool(function=ChatFunctionTool(name="lookup", parameters={"type": "wat"}, strict=True))],
    )
    result = ChatCompletionResult(
        id="chatcmpl_1",
        model="model-a",
        created_at=10,
        message=ChatMessage(
            role="assistant",
            tool_calls=[ChatToolCall(id="call_1", name="lookup", arguments="{}")],
        ),
        finish_reason="tool_calls",
    )

    with pytest.raises(RetryToolSchemaError, match="strict tool schema"):
        retry_on_unusable_tool_calls(result, request)


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


async def test_completions_client_coerces_complete_finish_reason_to_tool_handoff_when_tool_calls_present() -> None:
    fake_client = _FakeOpenAIClient(
        [
            _completion_response(
                model="ignored-model",
                content=None,
                finish_reason="stop",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": {"q": "x"}},
                    }
                ],
            )
        ],
        base_url=NOVITA_OPENAI_BASE_URL,
    )
    client = ChatCompletionClient(_novita_provider(client=fake_client))

    result = await client.complete(_request_for_model("openai/gpt-oss-20b"))

    assert result.finish_reason == "tool_calls"
    assert result.message.tool_calls is not None


async def test_completions_client_coerces_complete_finish_reason_to_stop_when_no_tool_calls_exist() -> None:
    fake_client = _FakeOpenAIClient(
        [
            _completion_response(
                model="ignored-model",
                content="hello back",
                finish_reason="tool_calls",
                tool_calls=None,
            )
        ],
        base_url=NOVITA_OPENAI_BASE_URL,
    )
    client = ChatCompletionClient(_novita_provider(client=fake_client))

    result = await client.complete(_request_for_model("openai/gpt-oss-20b"))

    assert result.finish_reason == "stop"
    assert result.message.content == "hello back"
    assert result.message.tool_calls is None


async def test_completions_client_coerces_stream_finish_reason_to_tool_handoff_when_tool_calls_present() -> None:
    fake_client = _FakeOpenAIClient(
        [
            _AsyncListStream(
                [
                    {
                        "id": "chatcmpl_1",
                        "model": "ignored-model",
                        "created": 10,
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": None,
                                "delta": {
                                    "content": None,
                                    "refusal": None,
                                    "reasoning_content": None,
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "function": {"name": "lookup", "arguments": {"q": "x"}},
                                        }
                                    ],
                                },
                            }
                        ],
                        "usage": None,
                    },
                    {
                        "id": "chatcmpl_1",
                        "model": "ignored-model",
                        "created": 10,
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "delta": {},
                            }
                        ],
                        "usage": None,
                    },
                ]
            )
        ],
        base_url=NOVITA_OPENAI_BASE_URL,
    )
    client = ChatCompletionClient(_novita_provider(client=fake_client))

    deltas = [delta async for delta in client.stream(_request_for_model("openai/gpt-oss-20b"))]

    assert deltas[-1].finish_reason == "tool_calls"


async def test_completions_client_coerces_stream_finish_reason_to_stop_when_no_tool_calls_exist() -> None:
    fake_client = _FakeOpenAIClient(
        [
            _AsyncListStream(
                [
                    {
                        "id": "chatcmpl_1",
                        "model": "ignored-model",
                        "created": 10,
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": None,
                                "delta": {
                                    "content": "hello back",
                                    "refusal": None,
                                    "reasoning_content": None,
                                    "tool_calls": None,
                                },
                            }
                        ],
                        "usage": None,
                    },
                    {
                        "id": "chatcmpl_1",
                        "model": "ignored-model",
                        "created": 10,
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "tool_calls",
                                "delta": {},
                            }
                        ],
                        "usage": None,
                    },
                ]
            )
        ],
        base_url=NOVITA_OPENAI_BASE_URL,
    )
    client = ChatCompletionClient(_novita_provider(client=fake_client))

    deltas = [delta async for delta in client.stream(_request_for_model("openai/gpt-oss-20b"))]

    assert deltas[-1].finish_reason == "stop"


def _lightning_provider(*, client: Any | None = None) -> OpenAIProvider:
    provider = build_lightning_provider(api_key="lightning-key")
    if client is not None:
        provider._client = client
    return provider


def _cerebras_provider(*, client: Any | None = None) -> OpenAIProvider:
    provider = build_cerebras_provider(api_key="cerebras-key")
    if client is not None:
        provider._client = client
    return provider


def _gmicloud_provider(*, client: Any | None = None) -> OpenAIProvider:
    provider = build_gmicloud_provider(api_key="gmicloud-key")
    if client is not None:
        provider._client = client
    return provider


def _groq_provider(*, client: Any | None = None) -> OpenAIProvider:
    provider = build_groq_provider(api_key="groq-key")
    if client is not None:
        provider._client = client
    return provider


def _novita_provider(*, client: Any | None = None) -> OpenAIProvider:
    provider = build_novita_provider(api_key="novita-key")
    if client is not None:
        provider._client = client
    return provider


def _crof_provider(*, client: Any | None = None) -> OpenAIProvider:
    provider = build_crof_provider(api_key="crof-key")
    if client is not None:
        provider._client = client
    return provider


def _qubrid_provider(*, client: Any | None = None) -> OpenAIProvider:
    provider = build_qubrid_provider(api_key="qubrid-key")
    if client is not None:
        provider._client = client
    return provider


def _openrouter_provider(*, client: Any | None = None) -> OpenRouterProvider:
    provider = build_openrouter_provider(api_key="openrouter-key")
    if client is not None:
        provider._client = client
    return provider


def _fireworks_provider(*, client: Any | None = None) -> FireworksProvider:
    provider = build_fireworks_provider(api_key="fireworks-key")
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


def test_lightning_provider_accepts_nemotron_models() -> None:
    provider = _lightning_provider()

    assert provider.lookup("lightning-ai/nvidia-nemotron-3-super-120b-a12b") == ()
    assert provider.lookup("lightning-ai/nvidia-nemotron-3-nano-omni-30b-a3b") == ()


def test_cerebras_request_quirks_preserve_supported_fields_and_glm_thinking() -> None:
    body = _body_for(_cerebras_provider(), _request_for_model("zai-glm-4.7"), stream=True)

    assert body["messages"][0] == {"role": "system", "content": "be precise"}
    assert body["max_completion_tokens"] == 128
    assert body["reasoning_effort"] == "low"
    assert body["extra_body"] == {"clear_thinking": False}
    assert "service_tier" not in body
    assert "prediction" not in body


def test_cerebras_provider_accepts_supported_models() -> None:
    provider = _cerebras_provider()

    assert provider.lookup("gpt-oss-120b")
    assert provider.lookup("zai-glm-4.7")


def test_groq_request_quirks_enable_reasoning_for_supported_models() -> None:
    request = _request_for_model("openai/gpt-oss-20b")

    body = _body_for(_groq_provider(), request, stream=True)

    assert body["messages"][0] == {"role": "system", "content": "be precise"}
    assert body["messages"][1] == {"role": "user", "content": "hello"}
    assert body["parallel_tool_calls"] is False
    assert body["extra_body"] == {"include_reasoning": True}
    assert "logprobs" not in body
    assert "logit_bias" not in body
    assert "top_logprobs" not in body
    assert "prompt_cache_key" not in body
    assert "metadata" not in body


def test_groq_request_quirks_skip_include_reasoning_for_unsupported_models() -> None:
    request = replace(_request_for_model("llama-3.1-8b-instant"), response_format=None)

    body = _body_for(_groq_provider(), request, stream=True)

    assert body["messages"][0] == {"role": "system", "content": "be precise"}
    assert body["messages"][1] == {"role": "user", "content": "hello"}
    assert body["parallel_tool_calls"] is True
    assert "extra_body" not in body


@pytest.mark.parametrize(
    ("provider_factory", "model"),
    [
        pytest.param(_groq_provider, "openai/gpt-oss-20b", id="groq"),
        pytest.param(_cerebras_provider, "gpt-oss-120b", id="cerebras"),
    ],
)
def test_provider_request_quirks_rename_assistant_reasoning_content_for_replay(
    provider_factory,
    model: str,
) -> None:
    request = replace(
        _request_for_model(model),
        messages=[
            ChatMessage(role="developer", content="be precise"),
            ChatMessage(role="user", content="hello", name="caller"),
            ChatMessage(role="assistant", content="draft", reasoning_content="because"),
        ],
    )

    body = _body_for(provider_factory(), request, stream=False)

    assert body["messages"][0] == {"role": "system", "content": "be precise"}
    assert body["messages"][2] == {"role": "assistant", "content": "draft", "reasoning": "because"}
    assert "reasoning_content" not in body["messages"][2]


def test_groq_provider_accepts_supported_models() -> None:
    provider = _groq_provider()

    assert provider.lookup("openai/gpt-oss-20b")
    assert provider.lookup("openai/gpt-oss-safeguard-20b")
    assert provider.lookup("openai/gpt-oss-120b")
    assert provider.lookup("meta-llama/llama-4-scout-17b-16e-instruct")
    assert provider.lookup("qwen/qwen3-32b")
    assert provider.lookup("llama-3.3-70b-versatile")
    assert provider.lookup("llama-3.1-8b-instant")


def test_qubrid_request_quirks_force_required_tool_for_deepseek_v4_flash() -> None:
    body = _body_for(_qubrid_provider(), _request_for_model("deepseek-ai/DeepSeek-V4-Flash"), stream=False)

    assert body["messages"][0] == {"role": "system", "content": "be precise"}
    assert body["tool_choice"] == "required"
    assert body["parallel_tool_calls"] is True
    assert body["tools"][0]["function"]["strict"] is True
    assert "top_k" not in body
    assert "logprobs" not in body
    assert "top_logprobs" not in body
    assert "service_tier" not in body
    assert "prediction" not in body


def test_qubrid_request_quirks_map_required_tool_choice_for_nemotron() -> None:
    request = replace(
        _request_for_model("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"),
        tool_choice="required",
    )

    body = _body_for(_qubrid_provider(), request, stream=False)

    assert body["messages"][0] == {"role": "system", "content": "be precise"}
    assert body["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
    assert "parallel_tool_calls" not in body
    assert body["tools"][0]["function"] == {
        "name": "lookup",
        "parameters": {"type": "object"},
        "description": "look something up",
    }


def test_qubrid_provider_accepts_supported_models() -> None:
    provider = _qubrid_provider()

    assert provider.lookup("deepseek-ai/DeepSeek-V4-Flash")
    assert provider.lookup("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8")


async def test_rate_limit_quirk_blocks_n_plus_1_complete_within_window(monkeypatch: pytest.MonkeyPatch) -> None:
    current = 100.0
    monkeypatch.setattr(quirks_module.time, "monotonic", lambda: current)
    provider = _StaticProvider(models={"model-a": (quirks_module.RateLimit(2, 60),)})
    client = ChatCompletionClient(provider)

    await client.complete(_simple_request())
    await client.complete(_simple_request())

    with pytest.raises(ChatCompletionRateLimitError, match="local rate limit exceeded"):
        await client.complete(_simple_request())

    assert len(provider.complete_calls) == 2


async def test_rate_limit_quirk_provider_scope_is_shared_across_models(monkeypatch: pytest.MonkeyPatch) -> None:
    current = 200.0
    monkeypatch.setattr(quirks_module.time, "monotonic", lambda: current)
    provider = _StaticProvider(
        quirks=(quirks_module.RateLimit(1, 60),),
        models={"model-a": (), "model-b": ()},
    )
    client = ChatCompletionClient(provider)

    await client.complete(_simple_request("model-a"))

    with pytest.raises(ChatCompletionRateLimitError, match="local rate limit exceeded"):
        await client.complete(_simple_request("model-b"))

    assert len(provider.complete_calls) == 1


async def test_provider_deepcopies_stateful_quirks_per_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    current = 250.0
    monkeypatch.setattr(quirks_module.time, "monotonic", lambda: current)
    shared_models = {"model-a": (quirks_module.RateLimit(1, 60),)}
    first = ChatCompletionClient(_StaticProvider(models=shared_models))
    second = ChatCompletionClient(_StaticProvider(models=shared_models))

    await first.complete(_simple_request())
    await second.complete(_simple_request())


async def test_rate_limit_quirk_window_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    current = 300.0
    monkeypatch.setattr(quirks_module.time, "monotonic", lambda: current)
    provider = _StaticProvider(models={"model-a": (quirks_module.RateLimit(1, 60),)})
    client = ChatCompletionClient(provider)

    await client.complete(_simple_request())
    current = 361.0
    await client.complete(_simple_request())

    assert len(provider.complete_calls) == 2


async def test_rate_limit_quirk_blocks_n_plus_1_stream_within_window(monkeypatch: pytest.MonkeyPatch) -> None:
    current = 400.0
    monkeypatch.setattr(quirks_module.time, "monotonic", lambda: current)
    provider = _StaticProvider(models={"model-a": (quirks_module.RateLimit(1, 60),)})
    client = ChatCompletionClient(provider)

    deltas = [delta async for delta in client.stream(_simple_request())]

    with pytest.raises(ChatCompletionRateLimitError, match="local rate limit exceeded"):
        [delta async for delta in client.stream(_simple_request())]

    assert deltas[-1].finish_reason == "stop"
    assert len(provider.stream_calls) == 1


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


def test_gmicloud_request_quirks_fill_missing_assistant_reasoning_content() -> None:
    request = replace(
        _request_for_model("XiaomiMiMo/MiMo-V2.5-Pro"),
        messages=[
            ChatMessage(role="developer", content="be precise"),
            ChatMessage(role="user", content="hello"),
            ChatMessage(role="assistant", content="draft"),
            ChatMessage(role="assistant", content="with reasoning", reasoning_content="because"),
        ],
    )

    body = _body_for(_gmicloud_provider(), request, stream=True)

    assert body["messages"][0] == {"role": "system", "content": "be precise"}
    assert body["messages"][1] == {"role": "user", "content": "hello"}
    assert body["messages"][2] == {"role": "assistant", "content": "draft", "reasoning_content": ""}
    assert body["messages"][3] == {
        "role": "assistant",
        "content": "with reasoning",
        "reasoning_content": "because",
    }


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


async def test_groq_client_aliases_reasoning_on_complete_and_stream() -> None:
    fake_client = _FakeOpenAIClient(
        [
            _completion_response(model="openai/gpt-oss-20b", content="ok", reasoning="because"),
            _AsyncListStream(
                [
                    _chunk(model="openai/gpt-oss-20b", content="ok", reasoning="because"),
                    _chunk(model="openai/gpt-oss-20b", finish_reason="stop"),
                ]
            ),
        ],
        base_url=GROQ_OPENAI_BASE_URL,
    )
    client = ChatCompletionClient(_groq_provider(client=fake_client))

    result = await client.complete(_request_for_model("openai/gpt-oss-20b"))
    deltas = [delta async for delta in client.stream(_request_for_model("openai/gpt-oss-20b"))]

    assert result.message.reasoning_content == "because"
    assert deltas[0].reasoning_delta == "because"
    assert deltas[1].finish_reason == "stop"


async def test_cerebras_client_aliases_reasoning_on_complete_and_stream() -> None:
    fake_client = _FakeOpenAIClient(
        [
            _completion_response(model="gpt-oss-120b", content="ok", reasoning="because"),
            _AsyncListStream(
                [
                    _chunk(model="gpt-oss-120b", content="ok", reasoning="because"),
                    _chunk(model="gpt-oss-120b", finish_reason="stop"),
                ]
            ),
        ],
        base_url=CEREBRAS_OPENAI_BASE_URL,
    )
    client = ChatCompletionClient(_cerebras_provider(client=fake_client))

    result = await client.complete(_request_for_model("gpt-oss-120b"))
    deltas = [delta async for delta in client.stream(_request_for_model("gpt-oss-120b"))]

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


async def test_openai_provider_normalizes_timeout_errors_and_logs_timeout_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("POST", "https://lightning.ai/api/v1/chat/completions")
    error = APITimeoutError(request=request)
    error.__cause__ = httpx.ConnectTimeout("connect timed out", request=request)
    events = _capture_openai_provider_logs(monkeypatch)
    fake_client = _FakeOpenAIClient([error], base_url=LIGHTNING_OPENAI_BASE_URL)
    fake_client.max_retries = 2
    fake_client.timeout = httpx.Timeout(timeout=600, connect=5.0)
    client = ChatCompletionClient(_lightning_provider(client=fake_client))

    with pytest.raises(ChatCompletionTimeoutError, match=r"Request timed out\. \(phase: connect\)"):
        await client.complete(_request_for_model("lightning-ai/gpt-oss-20b"))

    assert len(events) == 1
    event = events[0]
    assert event["event"] == "llm.provider.request_error"
    assert event["provider"] == "lightning"
    assert event["base_url"] == LIGHTNING_OPENAI_BASE_URL
    assert event["stream"] is False
    assert event["request_model"] == "lightning-ai/gpt-oss-20b"
    assert event["wire_model"] == "lightning-ai/gpt-oss-20b"
    assert event["sdk_error_type"] == "APITimeoutError"
    assert event["sdk_error_message"] == "Request timed out."
    assert event["timeout_phase"] == "connect"
    assert event["cause_chain_types"] == ["APITimeoutError", "ConnectTimeout"]
    assert event["root_cause_type"] == "ConnectTimeout"
    assert event["root_cause_message"] == "connect timed out"
    assert event["request_method"] == "POST"
    assert event["request_url"] == "https://lightning.ai/api/v1/chat/completions"
    assert event["client_max_retries"] == 2
    assert event["timeout_connect_seconds"] == 5.0
    assert event["timeout_read_seconds"] == 600.0
    assert event["timeout_write_seconds"] == 600.0
    assert event["timeout_pool_seconds"] == 600.0
    assert event["message_count"] == 2
    assert event["tool_count"] == 1
    assert isinstance(event["request_body_bytes"], int)
    assert event["request_body_bytes"] > 0


async def test_openai_provider_defaults_sdk_retries_to_zero() -> None:
    provider = _lightning_provider()

    assert provider._client.max_retries == 0

    await provider._client.close()


def test_context_length_classifier_matches_structured_codes_and_messages() -> None:
    assert is_context_length_exceeded_error({"error": {"type": "prompt-too-long"}})
    assert is_context_length_exceeded_error({"detail": "Requested 128001 tokens, but the model's maximum context length is 128000 tokens."})
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


async def test_router_complete_retries_transient_errors_before_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _capture_router_logs(monkeypatch)
    delays = _capture_router_retry_sleeps(monkeypatch)
    primary = _StubChatClient(complete_result=ChatCompletionProviderError("boom"))
    fallback = _StubChatClient(complete_result=_completion_result("gmicloud/XiaomiMiMo/MiMo-V2.5-Pro", "ok"))
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
    assert [request.model for request in primary.complete_requests] == ["qwen3.5-9b", "qwen3.5-9b", "qwen3.5-9b"]
    assert fallback.complete_requests[0].model == "XiaomiMiMo/MiMo-V2.5-Pro"
    assert [event["event"] for event in events] == [
        "llm.router.attempt_retry",
        "llm.router.attempt_retry",
        "llm.router.attempt_failed",
        "llm.router.fallback_succeeded",
    ]
    assert len(delays) == 2
    assert 0.25 <= delays[0] <= 0.5
    assert 0.5 <= delays[1] <= 1.0


async def test_router_complete_does_not_retry_current_attempt_for_invalid_request(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _capture_router_logs(monkeypatch)
    primary = _StubChatClient(complete_result=ChatCompletionInvalidRequestError("bad request"))
    fallback = _StubChatClient(complete_result=_completion_result("gmicloud/XiaomiMiMo/MiMo-V2.5-Pro", "ok"))
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
    assert [request.model for request in primary.complete_requests] == ["qwen3.5-9b"]
    assert [event["event"] for event in events] == [
        "llm.router.attempt_failed",
        "llm.router.fallback_succeeded",
    ]


async def test_router_complete_falls_back_immediately_for_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _capture_router_logs(monkeypatch)
    delays = _capture_router_retry_sleeps(monkeypatch)
    primary = _StubChatClient(complete_result=ChatCompletionRateLimitError("rate limited"))
    fallback = _StubChatClient(complete_result=_completion_result("gmicloud/XiaomiMiMo/MiMo-V2.5-Pro", "ok"))
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
    assert [request.model for request in primary.complete_requests] == ["qwen3.5-9b"]
    assert delays == []
    assert [event["event"] for event in events] == [
        "llm.router.attempt_failed",
        "llm.router.fallback_succeeded",
    ]


async def test_router_stream_retries_transient_errors_before_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _capture_router_logs(monkeypatch)
    delays = _capture_router_retry_sleeps(monkeypatch)
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
    assert [request.model for request in primary.stream_requests] == ["qwen3.5-9b", "qwen3.5-9b", "qwen3.5-9b"]
    assert fallback.stream_requests[0].model == "XiaomiMiMo/MiMo-V2.5-Pro"
    assert [event["event"] for event in events] == [
        "llm.router.attempt_retry",
        "llm.router.attempt_retry",
        "llm.router.attempt_failed",
        "llm.router.fallback_succeeded",
    ]
    assert len(delays) == 2
    assert 0.25 <= delays[0] <= 0.5
    assert 0.5 <= delays[1] <= 1.0


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


async def test_router_stream_times_out_between_deltas_without_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _capture_router_logs(monkeypatch)
    monkeypatch.setattr(router_module, "DEFAULT_STREAM_IDLE_DELTA_TIMEOUT_SECONDS", 0.01)

    class SlowSecondDeltaClient(IChatCompletionClient):
        def __init__(self) -> None:
            self.requests: list[ChatCompletionRequest] = []

        async def complete(self, request: ChatCompletionRequest):
            raise AssertionError(f"unexpected complete for {request.model}")

        def stream(self, request: ChatCompletionRequest):
            self.requests.append(request)

            async def run():
                yield _delta(request.model, content_delta="first")
                await anyio.sleep(0.05)
                yield _delta(request.model, finish_reason="stop")

            return run()

    primary = SlowSecondDeltaClient()
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

    with pytest.raises(ChatCompletionTimeoutError, match="produced no delta within"):
        [
            delta
            async for delta in router.stream(
                ChatCompletionRequest(
                    model="crof/qwen3.5-9b,gmicloud/XiaomiMiMo/MiMo-V2.5-Pro",
                    messages=[ChatMessage(role="user", content="hello")],
                )
            )
        ]

    assert [request.model for request in primary.requests] == ["qwen3.5-9b"]
    assert fallback.stream_requests == []
    assert events == []


async def test_router_stream_falls_back_after_first_delta_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _capture_router_logs(monkeypatch)
    delays = _capture_router_retry_sleeps(monkeypatch)

    class SlowClient(IChatCompletionClient):
        def __init__(self) -> None:
            self.requests: list[ChatCompletionRequest] = []

        async def complete(self, request: ChatCompletionRequest):
            raise AssertionError(f"unexpected complete for {request.model}")

        def stream(self, request: ChatCompletionRequest):
            self.requests.append(request)

            async def run():
                await anyio.sleep(0.05)
                yield _delta(request.model, content_delta="late")

            return run()

    primary = SlowClient()
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
    assert [request.model for request in primary.requests] == ["qwen3.5-9b", "qwen3.5-9b", "qwen3.5-9b"]
    assert [event["event"] for event in events] == [
        "llm.router.attempt_retry",
        "llm.router.attempt_retry",
        "llm.router.attempt_failed",
        "llm.router.fallback_succeeded",
    ]
    assert len(delays) == 2
    assert 0.25 <= delays[0] <= 0.5
    assert 0.5 <= delays[1] <= 1.0


async def test_unavailable_chat_client_rejects_unknown_models() -> None:
    client = UnavailableChatCompletionClient()

    with pytest.raises(ChatCompletionUnsupportedRequestError, match="No chat completion provider configured"):
        await client.complete(_request())
