from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from plap.llms.chat import (
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
    ChatToolChoiceFunction,
)
from plap.llms.crof import CrofChatCompletionClient, to_crof_chat_params
from plap.llms.errors import (
    ChatCompletionProviderError,
    ChatCompletionUnsupportedRequestError,
)
from plap.llms.fireworks import FireworksChatCompletionClient, to_fireworks_chat_params
from plap.llms.lightning import (
    LIGHTNING_OPENAI_BASE_URL,
    LightningChatCompletionClient,
    to_lightning_chat_params,
)
from plap.llms.novita import NovitaChatCompletionClient, to_novita_chat_params
from plap.llms.openai import (
    OpenAICompatibleChatCompletionClient,
    to_openai_chat_params,
)
from plap.llms.openrouter import OPENROUTER_OPENAI_BASE_URL, OpenRouterChatCompletionClient, to_openrouter_chat_params
from plap.llms.router import (
    ModelRoute,
    RoutingChatCompletionClient,
    UnavailableChatCompletionClient,
)


def test_openai_params_preserve_chat_completion_controls() -> None:
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
    assert params["reasoning_effort"] == "low"
    assert params["prompt_cache_key"] == "cache-a"
    assert params["metadata"] == {"k": "v"}
    assert params["service_tier"] == "flex"
    assert params["prediction"] == {"type": "content", "content": "expected"}
    assert params["stream_options"] == {"include_usage": True}
    assert "prompt_cache_retention" not in params
    assert "safety_identifier" not in params
    assert "store" not in params
    assert "verbosity" not in params


def test_openai_params_preserve_assistant_reasoning_metadata() -> None:
    reasoning_details = [
        {
            "type": "reasoning.text",
            "format": "openai-responses-v1",
            "index": 0,
            "text": "hidden chain",
        }
    ]
    request = ChatCompletionRequest(
        model="model-a",
        messages=[
            ChatMessage(
                role="assistant",
                content="answer",
                reasoning_content="hidden chain",
                reasoning_details=reasoning_details,
            )
        ],
    )

    params = to_openai_chat_params(request, stream=False)

    assert params["messages"] == [
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "hidden chain",
            "reasoning_details": reasoning_details,
        }
    ]


def test_openai_params_can_downgrade_developer_role() -> None:
    params = to_openai_chat_params(_request(), stream=False, developer_role="system")

    assert params["messages"][0] == {"role": "system", "content": "be precise"}
    assert "stream_options" not in params


def test_chat_completion_request_rejects_boolean_reasoning_effort() -> None:
    with pytest.raises(ValueError, match="reasoning_effort"):
        ChatCompletionRequest(
            model="model-a",
            messages=[ChatMessage(role="user", content="hello")],
            reasoning_effort=True,
        )


async def test_routing_client_strips_route_prefix_for_completion() -> None:
    client = _RecordingChatCompletionClient("crof")
    router = RoutingChatCompletionClient([ModelRoute(prefix="crof/", client=client)])

    result = await router.complete(
        ChatCompletionRequest(
            model="crof/qwen3.5-9b",
            messages=[ChatMessage(role="user", content="hello")],
        )
    )

    assert result.model == "crof/qwen3.5-9b"
    assert result.message.content == "crof"
    assert client.complete_requests[0].model == "qwen3.5-9b"


async def test_routing_client_strips_route_prefix_for_stream() -> None:
    client = _RecordingChatCompletionClient("fireworks")
    router = RoutingChatCompletionClient([ModelRoute(prefix="fireworks/", client=client)])

    deltas = [
        delta
        async for delta in router.stream(
            ChatCompletionRequest(
                model="fireworks/accounts/fireworks/models/gpt-oss-20b",
                messages=[ChatMessage(role="user", content="hello")],
            )
        )
    ]

    assert deltas[0].model == "fireworks/accounts/fireworks/models/gpt-oss-20b"
    assert deltas[0].content_delta == "fireworks"
    assert client.stream_requests[0].model == "accounts/fireworks/models/gpt-oss-20b"


async def test_routing_client_rejects_unmatched_model() -> None:
    router = RoutingChatCompletionClient([ModelRoute(prefix="openai/", client=_RecordingChatCompletionClient("openai"))])

    with pytest.raises(ChatCompletionUnsupportedRequestError, match="No chat"):
        await router.complete(
            ChatCompletionRequest(
                model="lightning-ai/gpt-oss-20b",
                messages=[ChatMessage(role="user", content="hello")],
            )
        )


async def test_routing_client_rejects_empty_provider_model() -> None:
    router = RoutingChatCompletionClient([ModelRoute(prefix="crof/", client=_RecordingChatCompletionClient("crof"))])

    with pytest.raises(ChatCompletionUnsupportedRequestError, match="No provider"):
        await router.complete(
            ChatCompletionRequest(
                model="crof/",
                messages=[ChatMessage(role="user", content="hello")],
            )
        )


async def test_unavailable_client_rejects_completion() -> None:
    client = UnavailableChatCompletionClient()

    with pytest.raises(ChatCompletionUnsupportedRequestError, match="No chat"):
        await client.complete(
            ChatCompletionRequest(
                model="lightning-ai/gpt-oss-20b",
                messages=[ChatMessage(role="user", content="hello")],
            )
        )


async def test_unavailable_client_rejects_stream() -> None:
    client = UnavailableChatCompletionClient()

    with pytest.raises(ChatCompletionUnsupportedRequestError, match="No chat"):
        async for _delta in client.stream(
            ChatCompletionRequest(
                model="lightning-ai/gpt-oss-20b",
                messages=[ChatMessage(role="user", content="hello")],
            )
        ):
            pass


def test_lightning_params_preserve_conformance_backed_fields() -> None:
    params = to_lightning_chat_params(_request(), stream=True)

    assert params["messages"][0] == {"role": "system", "content": "be precise"}
    assert params["stream"] is True
    assert params["stream_options"] == {"include_usage": True}
    assert params["max_completion_tokens"] == 128
    assert params["parallel_tool_calls"] is True
    assert params["reasoning_effort"] == "low"
    assert params["metadata"] == {"k": "v"}
    assert "prompt_cache_key" not in params
    assert "prediction" not in params
    assert "service_tier" not in params


def test_fireworks_params_preserve_supported_provider_hints() -> None:
    params = to_fireworks_chat_params(_request(), stream=True)

    assert params["messages"][0] == {"role": "system", "content": "be precise"}
    assert params["stream"] is True
    assert params["stream_options"] == {"include_usage": True}
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
    assert params["reasoning_effort"] == "low"
    assert params["prompt_cache_key"] == "cache-a"
    assert params["metadata"] == {"k": "v"}
    assert params["service_tier"] == "flex"
    assert params["prediction"] == {"type": "content", "content": "expected"}


def test_fireworks_params_omit_stream_options_without_stream() -> None:
    params = to_fireworks_chat_params(_request(), stream=False)

    assert params["stream"] is False
    assert "stream_options" not in params


def test_novita_params_map_supported_provider_fields() -> None:
    params = to_novita_chat_params(_request_for_model("openai/gpt-oss-20b"), stream=True)

    assert params["messages"][0] == {"role": "system", "content": "be precise"}
    assert params["stream"] is True
    assert params["stream_options"] == {"include_usage": True}
    assert params["max_tokens"] == 128
    assert params["parallel_tool_calls"] is True
    assert params["reasoning_effort"] == "low"
    assert "max_completion_tokens" not in params
    assert "extra_body" not in params
    assert "prompt_cache_key" not in params
    assert "metadata" not in params
    assert "prediction" not in params
    assert "service_tier" not in params
    assert "top_logprobs" not in params
    assert "user" not in params


def test_novita_params_omit_reasoning_controls_when_effort_is_none() -> None:
    request = ChatCompletionRequest(
        model="deepseek/deepseek-v4-flash",
        messages=[ChatMessage(role="user", content="hello")],
        reasoning_effort=None,
    )

    params = to_novita_chat_params(request, stream=False)

    assert "reasoning_effort" not in params
    assert "extra_body" not in params


def test_novita_params_pass_reasoning_effort_for_unknown_models() -> None:
    request = ChatCompletionRequest(
        model="model-a",
        messages=[ChatMessage(role="user", content="hello")],
        reasoning_effort="low",
    )

    params = to_novita_chat_params(request, stream=False)

    assert params["reasoning_effort"] == "low"
    assert "extra_body" not in params


def test_novita_gpt_oss_params_use_reasoning_effort() -> None:
    request = ChatCompletionRequest(
        model="openai/gpt-oss-20b",
        messages=[ChatMessage(role="user", content="hello")],
        parallel_tool_calls=True,
        reasoning_effort="high",
    )

    params = to_novita_chat_params(request, stream=False)

    assert params["parallel_tool_calls"] is True
    assert params["reasoning_effort"] == "high"
    assert "extra_body" not in params


def test_novita_gpt_oss_params_pass_reasoning_effort_without_validation() -> None:
    request = ChatCompletionRequest(
        model="openai/gpt-oss-20b",
        messages=[ChatMessage(role="user", content="hello")],
        reasoning_effort="xhigh",
    )

    params = to_novita_chat_params(request, stream=False)

    assert params["reasoning_effort"] == "xhigh"
    assert "extra_body" not in params


def test_novita_deepseek_v4_params_use_thinking_control() -> None:
    request = ChatCompletionRequest(
        model="deepseek/deepseek-v4-flash",
        messages=[ChatMessage(role="user", content="hello")],
        reasoning_effort="high",
    )

    params = to_novita_chat_params(request, stream=False)

    assert params["reasoning_effort"] == "high"
    assert params["extra_body"] == {"thinking": {"type": "enabled"}}


def test_novita_deepseek_v4_params_disable_thinking_for_none() -> None:
    request = ChatCompletionRequest(
        model="deepseek/deepseek-v4-flash",
        messages=[ChatMessage(role="user", content="hello")],
        reasoning_effort="none",
    )

    params = to_novita_chat_params(request, stream=False)

    assert params["reasoning_effort"] == "none"
    assert params["extra_body"] == {"thinking": {"type": "disabled"}}


def test_novita_deepseek_v4_maps_single_forced_tool_choice_to_required() -> None:
    params = to_novita_chat_params(_request_for_model("deepseek/deepseek-v4-flash"), stream=False)

    assert params["tool_choice"] == "required"


def test_novita_deepseek_v4_rejects_ambiguous_forced_tool_choice() -> None:
    request = replace(
        _request_for_model("deepseek/deepseek-v4-flash"),
        tools=[
            ChatTool(function=ChatFunctionTool(name="lookup")),
            ChatTool(function=ChatFunctionTool(name="calculate")),
        ],
    )

    with pytest.raises(ChatCompletionUnsupportedRequestError, match="forced function"):
        to_novita_chat_params(request, stream=False)


def test_crof_params_map_supported_provider_fields() -> None:
    request = replace(_request_for_model("glm-4.7-flash"), reasoning_effort=None)

    params = to_crof_chat_params(request, stream=True)

    assert params["messages"][0] == {"role": "system", "content": "be precise"}
    assert params["stream"] is True
    assert params["stream_options"] == {"include_usage": True}
    assert params["max_tokens"] == 128
    assert params["parallel_tool_calls"] is True
    assert "reasoning_effort" not in params
    assert "max_completion_tokens" not in params
    assert params["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "prompt_cache_key" not in params
    assert "metadata" not in params
    assert "prediction" not in params
    assert "service_tier" not in params
    assert "top_logprobs" not in params
    assert "user" not in params


def test_crof_params_keep_thinking_for_explicit_reasoning_effort() -> None:
    request = ChatCompletionRequest(
        model="glm-4.7-flash",
        messages=[ChatMessage(role="user", content="hello")],
        reasoning_effort="high",
    )

    params = to_crof_chat_params(request, stream=False)

    assert params["reasoning_effort"] == "high"
    assert "extra_body" not in params


def test_crof_params_disable_thinking_for_none_effort() -> None:
    request = ChatCompletionRequest(
        model="glm-4.7-flash",
        messages=[ChatMessage(role="user", content="hello")],
        reasoning_effort="none",
    )

    params = to_crof_chat_params(request, stream=False)

    assert params["reasoning_effort"] == "none"
    assert params["extra_body"] == {"thinking": {"type": "disabled"}}


def test_crof_qwen_params_use_native_response_format() -> None:
    params = to_crof_chat_params(_request_for_model("qwen3.5-9b"), stream=False)

    assert params["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {"type": "object"},
            "strict": True,
            "description": "answer schema",
        },
    }
    assert "extra_body" not in params


def test_crof_newer_glm_params_do_not_use_flash_thinking_quirk() -> None:
    request = ChatCompletionRequest(
        model="glm-5.1",
        messages=[ChatMessage(role="user", content="hello")],
    )

    params = to_crof_chat_params(request, stream=False)

    assert "extra_body" not in params


def test_openrouter_params_preserve_openai_compatible_fields() -> None:
    params = to_openrouter_chat_params(_request(), stream=True)

    assert params["messages"][0] == {"role": "system", "content": "be precise"}
    assert params["stream"] is True
    assert params["stream_options"] == {"include_usage": True}
    assert params["max_completion_tokens"] == 128
    assert params["parallel_tool_calls"] is True
    assert params["reasoning_effort"] == "low"
    assert params["prompt_cache_key"] == "cache-a"
    assert params["metadata"] == {"k": "v"}
    assert params["service_tier"] == "flex"
    assert params["prediction"] == {"type": "content", "content": "expected"}


def test_openrouter_client_defaults_to_openrouter_base_url() -> None:
    client = OpenRouterChatCompletionClient(api_key="test-key")

    assert str(client._client.base_url).rstrip("/") == OPENROUTER_OPENAI_BASE_URL


async def test_openai_client_normalizes_completion_result() -> None:
    reasoning_details = [
        {
            "type": "reasoning.text",
            "format": "openai-responses-v1",
            "index": 0,
            "text": "because",
        }
    ]
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
                        reasoning_details=reasoning_details,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(name="lookup", arguments={"q": "x"}),
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
    client = OpenAICompatibleChatCompletionClient(client=_FakeOpenAIClient(fake_completion))

    result = await client.complete(_request())

    assert result.id == "chatcmpl_1"
    assert result.message.role == "assistant"
    assert result.message.content == "answer"
    assert result.message.reasoning_content == "because"
    assert result.message.reasoning_details == reasoning_details
    assert result.message.tool_calls == [ChatToolCall(id="call_1", name="lookup", arguments='{"q":"x"}')]
    assert result.finish_reason == "tool_calls"
    assert result.usage is not None
    assert result.usage.cached_tokens == 2
    assert result.usage.reasoning_tokens == 1


async def test_openai_client_reads_top_level_reasoning_tokens() -> None:
    fake_completion = _FakeOpenAICompletion(
        SimpleNamespace(
            id="chatcmpl_1",
            model="model-a",
            created=10,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="answer",
                        refusal=None,
                        reasoning_content=None,
                        tool_calls=None,
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=3,
                completion_tokens=5,
                total_tokens=8,
                reasoning_tokens=4,
            ),
        )
    )
    client = OpenAICompatibleChatCompletionClient(client=_FakeOpenAIClient(fake_completion))

    result = await client.complete(_request())

    assert result.usage is not None
    assert result.usage.reasoning_tokens == 4


async def test_openai_client_normalizes_stream_chunks() -> None:
    reasoning_details = [
        {
            "type": "reasoning.text",
            "format": "openai-responses-v1",
            "index": 0,
            "text": "because",
        }
    ]
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
                                reasoning_content="because",
                                reasoning_details=reasoning_details,
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
                                        function=SimpleNamespace(name="lookup", arguments='{"q"'),
                                    )
                                ],
                            ),
                        )
                    ],
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3),
                ),
            ]
        )
    )
    client = OpenAICompatibleChatCompletionClient(client=_FakeOpenAIClient(fake_completion))

    deltas = [delta async for delta in client.stream(_request())]

    assert deltas[0].content_delta == "hel"
    assert deltas[0].reasoning_delta == "because"
    assert deltas[0].reasoning_details_delta == reasoning_details
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


async def test_lightning_client_maps_developer_role_to_system() -> None:
    fake_completion = _FakeOpenAICompletion(
        SimpleNamespace(
            id="chatcmpl_1",
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
            usage=None,
        )
    )
    fake_client = _FakeOpenAIClient(fake_completion)
    client = LightningChatCompletionClient(client=fake_client)

    await client.complete(_request())

    assert fake_completion.calls[0]["messages"][0] == {
        "role": "system",
        "content": "be precise",
    }


def test_lightning_client_defaults_to_lightning_openai_base_url() -> None:
    client = LightningChatCompletionClient(api_key="test-key")

    assert str(client._client.base_url).rstrip("/") == LIGHTNING_OPENAI_BASE_URL


async def test_lightning_120b_response_format_fallback_omits_native_field() -> None:
    fake_completion = _FakeOpenAICompletion(_completion_response(model="lightning-ai/gpt-oss-120b", content='{"ok":true}'))
    client = LightningChatCompletionClient(client=_FakeOpenAIClient(fake_completion))

    result = await client.complete(
        ChatCompletionRequest(
            model="lightning-ai/gpt-oss-120b",
            messages=[ChatMessage(role="user", content='Return {"ok": true}.')],
            response_format=ChatResponseFormat(type="json_object"),
            max_completion_tokens=128,
        )
    )

    assert result.message.content == '{"ok":true}'
    call = fake_completion.calls[0]
    assert "response_format" not in call
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][1] == {
        "role": "user",
        "content": 'Return {"ok": true}.',
    }


async def test_lightning_120b_response_format_fallback_retries_schema_errors() -> None:
    fake_completion = _FakeOpenAICompletion(
        [
            _completion_response(
                model="lightning-ai/gpt-oss-120b",
                content='{"ok":"no"}',
            ),
            _completion_response(
                model="lightning-ai/gpt-oss-120b",
                content='{"ok":true}',
            ),
        ]
    )
    client = LightningChatCompletionClient(client=_FakeOpenAIClient(fake_completion))

    result = await client.complete(
        ChatCompletionRequest(
            model="lightning-ai/gpt-oss-120b",
            messages=[ChatMessage(role="user", content='Return {"ok": true}.')],
            response_format=ChatResponseFormat(
                type="json_schema",
                name="ok_response",
                schema={
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
                strict=True,
            ),
            max_completion_tokens=128,
        )
    )

    assert result.message.content == '{"ok":true}'
    assert len(fake_completion.calls) == 2
    assert "response_format" not in fake_completion.calls[0]
    assert "response_format" not in fake_completion.calls[1]
    assert fake_completion.calls[1]["messages"][-2] == {
        "role": "assistant",
        "content": '{"ok":"no"}',
    }
    assert fake_completion.calls[1]["messages"][-1]["role"] == "user"


async def test_lightning_120b_response_format_fallback_raises_after_retry() -> None:
    fake_completion = _FakeOpenAICompletion(
        [
            _completion_response(
                model="lightning-ai/gpt-oss-120b",
                content="not json",
            ),
            _completion_response(
                model="lightning-ai/gpt-oss-120b",
                content="still not json",
            ),
        ]
    )
    client = LightningChatCompletionClient(client=_FakeOpenAIClient(fake_completion))

    with pytest.raises(ChatCompletionProviderError, match="invalid JSON"):
        await client.complete(
            ChatCompletionRequest(
                model="lightning-ai/gpt-oss-120b",
                messages=[ChatMessage(role="user", content='Return {"ok": true}.')],
                response_format=ChatResponseFormat(type="json_object"),
            )
        )


async def test_lightning_120b_response_format_fallback_streams_result() -> None:
    fake_completion = _FakeOpenAICompletion(
        _completion_response(
            model="lightning-ai/gpt-oss-120b",
            content='{"ok":true}',
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
    )
    client = LightningChatCompletionClient(client=_FakeOpenAIClient(fake_completion))

    deltas = [
        delta
        async for delta in client.stream(
            ChatCompletionRequest(
                model="lightning-ai/gpt-oss-120b",
                messages=[ChatMessage(role="user", content='Return {"ok": true}.')],
                response_format=ChatResponseFormat(type="json_object"),
            )
        )
    ]

    assert len(deltas) == 2
    assert deltas[0].content_delta == '{"ok":true}'
    assert deltas[0].finish_reason is None
    assert deltas[0].usage is None
    assert deltas[1].content_delta is None
    assert deltas[1].finish_reason == "stop"
    assert deltas[1].usage is not None
    assert deltas[1].usage.total_tokens == 8
    assert fake_completion.calls[0]["stream"] is False
    assert "response_format" not in fake_completion.calls[0]


async def test_lightning_120b_response_format_fallback_stream_retries() -> None:
    fake_completion = _FakeOpenAICompletion(
        [
            _completion_response(
                model="lightning-ai/gpt-oss-120b",
                content='{"ok":"no"}',
            ),
            _completion_response(
                model="lightning-ai/gpt-oss-120b",
                content='{"ok":true}',
            ),
        ]
    )
    client = LightningChatCompletionClient(client=_FakeOpenAIClient(fake_completion))

    deltas = [
        delta
        async for delta in client.stream(
            ChatCompletionRequest(
                model="lightning-ai/gpt-oss-120b",
                messages=[ChatMessage(role="user", content='Return {"ok": true}.')],
                response_format=ChatResponseFormat(
                    type="json_schema",
                    schema={
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    },
                ),
            )
        )
    ]

    assert [delta.content_delta for delta in deltas] == ['{"ok":true}', None]
    assert len(fake_completion.calls) == 2
    assert fake_completion.calls[1]["messages"][-2] == {
        "role": "assistant",
        "content": '{"ok":"no"}',
    }
    assert fake_completion.calls[1]["messages"][-1]["role"] == "user"


async def test_lightning_120b_fallback_stream_raises_before_yield() -> None:
    fake_completion = _FakeOpenAICompletion(
        [
            _completion_response(
                model="lightning-ai/gpt-oss-120b",
                content="not json",
            ),
            _completion_response(
                model="lightning-ai/gpt-oss-120b",
                content="still not json",
            ),
        ]
    )
    client = LightningChatCompletionClient(client=_FakeOpenAIClient(fake_completion))

    with pytest.raises(ChatCompletionProviderError, match="invalid JSON"):
        await anext(
            client.stream(
                ChatCompletionRequest(
                    model="lightning-ai/gpt-oss-120b",
                    messages=[ChatMessage(role="user", content='Return {"ok": true}.')],
                    response_format=ChatResponseFormat(type="json_object"),
                )
            )
        )


async def test_lightning_non_fallback_response_format_streams_from_provider() -> None:
    fake_completion = _FakeOpenAICompletion(
        _async_iter(
            [
                SimpleNamespace(
                    id="chatcmpl_1",
                    model="lightning-ai/gpt-oss-20b",
                    created=10,
                    system_fingerprint=None,
                    service_tier=None,
                    choices=[
                        SimpleNamespace(
                            index=0,
                            finish_reason=None,
                            delta=SimpleNamespace(
                                content='{"ok":true}',
                                refusal=None,
                                reasoning_content=None,
                                tool_calls=None,
                            ),
                        )
                    ],
                    usage=None,
                )
            ]
        )
    )
    client = LightningChatCompletionClient(client=_FakeOpenAIClient(fake_completion))

    deltas = [
        delta
        async for delta in client.stream(
            ChatCompletionRequest(
                model="lightning-ai/gpt-oss-20b",
                messages=[ChatMessage(role="user", content='Return {"ok": true}.')],
                response_format=ChatResponseFormat(type="json_object"),
            )
        )
    ]

    assert deltas[0].content_delta == '{"ok":true}'
    assert fake_completion.calls[0]["stream"] is True
    assert fake_completion.calls[0]["response_format"] == {"type": "json_object"}


async def test_crof_client_uses_openai_create_with_crof_params() -> None:
    fake_completion = _FakeOpenAICompletion(_completion_response(model="glm-4.7-flash", content="ok"))
    fake_client = _FakeOpenAIClient(fake_completion)
    client = CrofChatCompletionClient(client=fake_client)

    result = await client.complete(
        ChatCompletionRequest(
            model="glm-4.7-flash",
            messages=[ChatMessage(role="developer", content="be precise")],
            max_completion_tokens=128,
        )
    )

    assert fake_completion.calls[0]["stream"] is False
    assert fake_completion.calls[0]["max_tokens"] == 128
    assert fake_completion.calls[0]["messages"][0] == {
        "role": "system",
        "content": "be precise",
    }
    assert fake_completion.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "max_completion_tokens" not in fake_completion.calls[0]
    assert result.message.content == "ok"


async def test_crof_glm_4_7_flash_copies_json_from_reasoning_content() -> None:
    fake_completion = _FakeOpenAICompletion(
        _completion_response(
            model="glm-4.7-flash",
            content="",
            reasoning_content='{"ok":true}',
        )
    )
    client = CrofChatCompletionClient(client=_FakeOpenAIClient(fake_completion))

    result = await client.complete(
        ChatCompletionRequest(
            model="glm-4.7-flash",
            messages=[ChatMessage(role="user", content='Return {"ok": true}.')],
            response_format=ChatResponseFormat(type="json_object"),
            max_completion_tokens=128,
        )
    )

    assert result.message.content == '{"ok":true}'
    assert result.message.reasoning_content == '{"ok":true}'
    call = fake_completion.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}


async def test_novita_client_uses_openai_create_with_novita_params() -> None:
    reasoning_details = [
        {
            "type": "reasoning.text",
            "format": "openai-responses-v1",
            "index": 0,
            "text": "because",
        }
    ]
    fake_completion = _FakeOpenAICompletion(
        SimpleNamespace(
            id="novita_1",
            model="model-a",
            created=10,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="ok",
                        refusal=None,
                        reasoning_content=None,
                        reasoning_details=reasoning_details,
                        tool_calls=None,
                    ),
                )
            ],
            usage=None,
        )
    )
    fake_client = _FakeOpenAIClient(fake_completion)
    client = NovitaChatCompletionClient(client=fake_client)

    result = await client.complete(_request_for_model("openai/gpt-oss-20b"))

    assert fake_completion.calls[0]["stream"] is False
    assert fake_completion.calls[0]["max_tokens"] == 128
    assert fake_completion.calls[0]["messages"][0] == {
        "role": "system",
        "content": "be precise",
    }
    assert fake_completion.calls[0]["reasoning_effort"] == "low"
    assert "extra_body" not in fake_completion.calls[0]
    assert "max_completion_tokens" not in fake_completion.calls[0]
    assert result.id == "novita_1"
    assert result.message.reasoning_details == reasoning_details


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
        reasoning_effort="low",
        stream_options=ChatStreamOptions(include_usage=True),
        user="user-a",
        prompt_cache_key="cache-a",
        metadata={"k": "v"},
        service_tier="flex",
        prediction=ChatPrediction(content="expected"),
    )


def _request_for_model(model: str) -> ChatCompletionRequest:
    return replace(_request(), model=model)


class _FakeOpenAIClient:
    def __init__(self, completion: _FakeOpenAICompletion) -> None:
        self.chat = SimpleNamespace(completions=completion)


def _completion_response(
    *,
    model: str = "model-a",
    content: str = "ok",
    reasoning_content: str | None = None,
    usage: object | None = None,
) -> object:
    return SimpleNamespace(
        id="chatcmpl_1",
        model=model,
        created=10,
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content=content,
                    refusal=None,
                    reasoning_content=reasoning_content,
                    tool_calls=None,
                ),
            )
        ],
        usage=usage,
    )


class _FakeOpenAICompletion:
    def __init__(self, response: object | list[object]) -> None:
        self.responses = response if isinstance(response, list) else [response]
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


class _RecordingChatCompletionClient:
    def __init__(self, name: str) -> None:
        self.name = name
        self.complete_requests: list[ChatCompletionRequest] = []
        self.stream_requests: list[ChatCompletionRequest] = []

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResult:
        self.complete_requests.append(request)
        return ChatCompletionResult(
            id=self.name,
            model=request.model,
            created_at=None,
            message=ChatMessage(role="assistant", content=self.name),
            finish_reason="stop",
        )

    async def stream(self, request: ChatCompletionRequest):
        self.stream_requests.append(request)
        yield ChatCompletionDelta(
            id=self.name,
            model=request.model,
            created_at=None,
            choice_index=0,
            content_delta=self.name,
        )


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
