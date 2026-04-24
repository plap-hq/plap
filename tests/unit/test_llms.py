from __future__ import annotations

from types import SimpleNamespace

from plap.llms.chat import (
    ChatCompletionRequest,
    ChatFunctionTool,
    ChatMessage,
    ChatPrediction,
    ChatResponseFormat,
    ChatStreamOptions,
    ChatTool,
    ChatToolCall,
    ChatToolChoiceFunction,
)
from plap.llms.fireworks import FireworksChatCompletionClient, to_fireworks_chat_params
from plap.llms.lightning import LightningChatCompletionClient
from plap.llms.openai_compatible import (
    OpenAICompatibleChatCompletionClient,
    to_openai_chat_params,
)


def test_openai_compatible_params_preserve_chat_completion_controls() -> None:
    request = _request()

    params = to_openai_chat_params(request, stream=True)

    assert params["model"] == "model-a"
    assert params["messages"][0] == {"role": "developer", "content": "be precise"}
    assert params["messages"][1] == {
        "role": "user",
        "content": "hello",
        "name": "caller",
    }
    assert params["tools"] == [
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
    assert params["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
    assert params["parallel_tool_calls"] is True
    assert params["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {"type": "object"},
            "strict": True,
            "description": "answer schema",
        },
    }
    assert params["max_completion_tokens"] == 128
    assert params["prompt_cache_key"] == "cache-a"
    assert params["metadata"] == {"k": "v"}
    assert params["service_tier"] == "flex"
    assert params["prediction"] == {"type": "content", "content": "expected"}
    assert params["stream_options"] == {"include_usage": True}
    assert "prompt_cache_retention" not in params
    assert "safety_identifier" not in params
    assert "store" not in params
    assert "verbosity" not in params


def test_fireworks_params_preserve_supported_provider_hints() -> None:
    params = to_fireworks_chat_params(_request())

    assert params["max_completion_tokens"] == 128
    assert params["prompt_cache_key"] == "cache-a"
    assert params["metadata"] == {"k": "v"}
    assert params["service_tier"] == "flex"
    assert params["prediction"] == {"type": "content", "content": "expected"}


async def test_openai_compatible_client_normalizes_completion_result() -> None:
    fake_completion = _FakeOpenAICompletion(
        SimpleNamespace(
            id="chatcmpl_1",
            model="model-a",
            created=10,
            system_fingerprint="fp_1",
            service_tier="default",
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        content="answer",
                        refusal=None,
                        reasoning_content="because",
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(
                                    name="lookup", arguments={"q": "x"}
                                ),
                            )
                        ],
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=3,
                completion_tokens=5,
                total_tokens=8,
                prompt_tokens_details=SimpleNamespace(cached_tokens=2),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=1),
            ),
        )
    )
    client = OpenAICompatibleChatCompletionClient(
        client=_FakeOpenAIClient(fake_completion)
    )

    result = await client.complete(_request())

    assert result.id == "chatcmpl_1"
    assert result.message.content == "answer"
    assert result.message.reasoning_content == "because"
    assert result.message.tool_calls == [
        ChatToolCall(id="call_1", name="lookup", arguments='{"q":"x"}')
    ]
    assert result.finish_reason == "tool_calls"
    assert result.usage is not None
    assert result.usage.cached_tokens == 2
    assert result.usage.reasoning_tokens == 1


async def test_openai_compatible_client_normalizes_stream_chunks() -> None:
    fake_completion = _FakeOpenAICompletion(
        _async_iter(
            [
                SimpleNamespace(
                    id="chatcmpl_1",
                    model="model-a",
                    created=10,
                    system_fingerprint="fp_1",
                    service_tier="default",
                    choices=[
                        SimpleNamespace(
                            index=0,
                            finish_reason=None,
                            delta=SimpleNamespace(
                                content="hel",
                                refusal=None,
                                reasoning_content=None,
                                tool_calls=None,
                            ),
                        )
                    ],
                    usage=None,
                ),
                SimpleNamespace(
                    id="chatcmpl_1",
                    model="model-a",
                    created=10,
                    system_fingerprint="fp_1",
                    service_tier="default",
                    choices=[
                        SimpleNamespace(
                            index=0,
                            finish_reason="tool_calls",
                            delta=SimpleNamespace(
                                content=None,
                                refusal=None,
                                reasoning_content=None,
                                tool_calls=[
                                    SimpleNamespace(
                                        index=0,
                                        id="call_1",
                                        function=SimpleNamespace(
                                            name="lookup", arguments='{"q"'
                                        ),
                                    )
                                ],
                            ),
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=1, completion_tokens=2, total_tokens=3
                    ),
                ),
            ]
        )
    )
    client = OpenAICompatibleChatCompletionClient(
        client=_FakeOpenAIClient(fake_completion)
    )

    deltas = [delta async for delta in client.stream(_request())]

    assert deltas[0].content_delta == "hel"
    assert deltas[1].finish_reason == "tool_calls"
    assert deltas[1].tool_call_delta is not None
    assert deltas[1].tool_call_delta.name == "lookup"
    assert deltas[1].usage is not None
    assert deltas[1].usage.total_tokens == 3


async def test_fireworks_client_uses_acreate_and_normalizes_response() -> None:
    fireworks = _FakeFireworksClient(
        SimpleNamespace(
            id="fw_1",
            model="model-a",
            created=10,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="ok",
                        refusal=None,
                        reasoning_content=None,
                        tool_calls=None,
                    ),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
    )
    client = FireworksChatCompletionClient(client=fireworks)

    result = await client.complete(_request())

    assert result.id == "fw_1"
    assert result.message.content == "ok"
    assert fireworks.chat.completions.calls[0]["stream"] is False
    assert fireworks.chat.completions.calls[0]["max_completion_tokens"] == 128


def test_lightning_client_is_openai_compatible_wrapper() -> None:
    client = LightningChatCompletionClient(
        client=_FakeOpenAIClient(_FakeOpenAICompletion(None))
    )

    assert isinstance(client, OpenAICompatibleChatCompletionClient)


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
        frequency_penalty=0.1,
        presence_penalty=0.2,
        logit_bias={"1": -10},
        logprobs=True,
        top_logprobs=2,
        stop=["END"],
        seed=7,
        n=1,
        reasoning_effort=True,
        stream_options=ChatStreamOptions(include_usage=True),
        user="user-a",
        prompt_cache_key="cache-a",
        metadata={"k": "v"},
        service_tier="flex",
        prediction=ChatPrediction(content="expected"),
    )


class _FakeOpenAIClient:
    def __init__(self, completion: _FakeOpenAICompletion) -> None:
        self.chat = SimpleNamespace(completions=completion)


class _FakeOpenAICompletion:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.response


class _FakeFireworksClient:
    def __init__(self, response: object) -> None:
        self.chat = SimpleNamespace(completions=_FakeFireworksCompletions(response))


class _FakeFireworksCompletions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def acreate(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return _async_iter([self.response])
        return self._complete()

    async def _complete(self) -> object:
        return self.response


async def _async_iter(values: list[object]):
    for value in values:
        yield value
