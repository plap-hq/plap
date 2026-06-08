from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from plap.llms.completions.chat import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatFunctionTool,
    ChatMessage,
    ChatResponseFormat,
    ChatStreamOptions,
    ChatTool,
    ChatToolChoiceFunction,
    IChatCompletionClient,
    ReasoningEffort,
)
from plap.llms.completions.client import ChatCompletionClient
from plap.llms.completions.errors import ChatCompletionProviderError
from plap.llms.completions.providers import (
    build_cerebras_provider,
    build_crof_provider,
    build_fireworks_provider,
    build_gmicloud_provider,
    build_groq_provider,
    build_lightning_provider,
    build_novita_provider,
)
from plap.llms.completions.router import ModelRoute, RoutingChatCompletionClient
from plap.settings import Settings

pytestmark = pytest.mark.money


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "api_key_pepper": "pepper",
        "database_url": "postgresql+asyncpg://example/test",
        "sealing_keys": ["a" * 43],
    }
    values.update(overrides)
    return Settings(**values)


def _lightning_client(api_key: str) -> IChatCompletionClient:
    provider = build_lightning_provider(api_key=api_key)
    assert provider is not None
    return RoutingChatCompletionClient(
        [
            ModelRoute(
                prefix="lightning/",
                client=ChatCompletionClient(provider),
            )
        ]
    )


def _cerebras_client(api_key: str) -> IChatCompletionClient:
    provider = build_cerebras_provider(api_key=api_key)
    assert provider is not None
    return RoutingChatCompletionClient(
        [
            ModelRoute(
                prefix="cerebras/",
                client=ChatCompletionClient(provider),
            )
        ]
    )


def _novita_client(api_key: str) -> IChatCompletionClient:
    provider = build_novita_provider(api_key=api_key)
    assert provider is not None
    return RoutingChatCompletionClient(
        [
            ModelRoute(
                prefix="novita/",
                client=ChatCompletionClient(provider),
            )
        ]
    )


def _gmicloud_client(api_key: str) -> IChatCompletionClient:
    provider = build_gmicloud_provider(api_key=api_key)
    assert provider is not None
    return RoutingChatCompletionClient(
        [
            ModelRoute(
                prefix="gmicloud/",
                client=ChatCompletionClient(provider),
            )
        ]
    )


def _groq_client(api_key: str) -> IChatCompletionClient:
    provider = build_groq_provider(api_key=api_key)
    assert provider is not None
    return RoutingChatCompletionClient(
        [
            ModelRoute(
                prefix="groq/",
                client=ChatCompletionClient(provider),
            )
        ]
    )


def _fireworks_client(api_key: str) -> IChatCompletionClient:
    provider = build_fireworks_provider(api_key=api_key)
    assert provider is not None
    return RoutingChatCompletionClient(
        [
            ModelRoute(
                prefix="fireworks/",
                client=ChatCompletionClient(provider),
            )
        ]
    )


def _crof_client(api_key: str) -> IChatCompletionClient:
    provider = build_crof_provider(api_key=api_key)
    assert provider is not None
    return RoutingChatCompletionClient([ModelRoute(prefix="crof/", client=ChatCompletionClient(provider))])


LIGHTNING_GPT_OSS_20B_MODEL = "lightning/lightning-ai/gpt-oss-20b"
LIGHTNING_GPT_OSS_120B_MODEL = "lightning/lightning-ai/gpt-oss-120b"
CEREBRAS_GPT_OSS_120B_MODEL = "cerebras/gpt-oss-120b"
CEREBRAS_GLM_4_7_MODEL = "cerebras/zai-glm-4.7"
GROQ_GPT_OSS_20B_MODEL = "groq/openai/gpt-oss-20b"
GROQ_GPT_OSS_120B_MODEL = "groq/openai/gpt-oss-120b"
NOVITA_GPT_OSS_120B_MODEL = "novita/openai/gpt-oss-120b"
GMICLOUD_MIMO_V25_PRO_MODEL = "gmicloud/XiaomiMiMo/MiMo-V2.5-Pro"
FIREWORKS_GPT_OSS_20B_MODEL = "fireworks/accounts/fireworks/models/gpt-oss-20b"
CROF_QWEN_3_5_9B_MODEL = "crof/qwen3.5-9b"
DEFAULT_REASONING_EFFORT: ReasoningEffort = "low"
_UNAVAILABLE_PROVIDER_REASONS: dict[str, str] = {}


@dataclass(frozen=True)
class ProviderCase:
    name: str
    api_key_env: str
    model: str
    client_factory: Callable[[str], IChatCompletionClient]


GPT_OSS_PROVIDERS = (
    pytest.param(
        ProviderCase(
            name="lightning",
            api_key_env="LIGHTNING_API_KEY",
            model=LIGHTNING_GPT_OSS_20B_MODEL,
            client_factory=_lightning_client,
        ),
        id="lightning-gpt-oss-20b",
    ),
    pytest.param(
        ProviderCase(
            name="cerebras",
            api_key_env="CEREBRAS_API_KEY",
            model=CEREBRAS_GPT_OSS_120B_MODEL,
            client_factory=_cerebras_client,
        ),
        id="cerebras-gpt-oss-120b",
    ),
    pytest.param(
        ProviderCase(
            name="groq",
            api_key_env="GROQ_API_KEY",
            model=GROQ_GPT_OSS_20B_MODEL,
            client_factory=_groq_client,
        ),
        id="groq-gpt-oss-20b",
    ),
    pytest.param(
        ProviderCase(
            name="novita",
            api_key_env="NOVITA_API_KEY",
            model=NOVITA_GPT_OSS_120B_MODEL,
            client_factory=_novita_client,
        ),
        id="novita-gpt-oss-120b",
    ),
    pytest.param(
        ProviderCase(
            name="fireworks",
            api_key_env="FIREWORKS_API_KEY",
            model=FIREWORKS_GPT_OSS_20B_MODEL,
            client_factory=_fireworks_client,
        ),
        id="fireworks-gpt-oss-20b",
    ),
)

# Lightning 20B passes JSON mode but emits JSON text instead of tool calls.
# Novita exposes GPT-OSS, but that model currently rejects function calling.
TOOL_PROVIDERS = (
    pytest.param(
        ProviderCase(
            name="lightning",
            api_key_env="LIGHTNING_API_KEY",
            model=LIGHTNING_GPT_OSS_120B_MODEL,
            client_factory=_lightning_client,
        ),
        id="lightning-gpt-oss-120b",
    ),
    pytest.param(
        ProviderCase(
            name="cerebras",
            api_key_env="CEREBRAS_API_KEY",
            model=CEREBRAS_GLM_4_7_MODEL,
            client_factory=_cerebras_client,
        ),
        id="cerebras-zai-glm-4.7",
    ),
    pytest.param(
        ProviderCase(
            name="groq",
            api_key_env="GROQ_API_KEY",
            model=GROQ_GPT_OSS_120B_MODEL,
            client_factory=_groq_client,
        ),
        id="groq-gpt-oss-120b",
    ),
    pytest.param(
        ProviderCase(
            name="gmicloud",
            api_key_env="GMICLOUD_API_KEY",
            model=GMICLOUD_MIMO_V25_PRO_MODEL,
            client_factory=_gmicloud_client,
        ),
        id="gmicloud-mimo-v2.5-pro",
    ),
    GPT_OSS_PROVIDERS[2],
    pytest.param(
        ProviderCase(
            name="crof",
            api_key_env="CROF_API_KEY",
            model=CROF_QWEN_3_5_9B_MODEL,
            client_factory=_crof_client,
        ),
        id="crof-qwen3.5-9b",
    ),
)

CROF_PROVIDER = ProviderCase(
    name="crof",
    api_key_env="CROF_API_KEY",
    model=CROF_QWEN_3_5_9B_MODEL,
    client_factory=_crof_client,
)


@pytest.mark.parametrize("provider", GPT_OSS_PROVIDERS)
async def test_live_basic_chat_completion(provider: ProviderCase) -> None:
    result = await _complete(
        provider,
        ChatCompletionRequest(
            model=provider.model,
            messages=[ChatMessage(role="user", content="Reply with exactly: pong")],
            reasoning_effort=DEFAULT_REASONING_EFFORT,
            max_completion_tokens=96,
            temperature=0,
        ),
    )

    assert _message_has_output(result.message)


@pytest.mark.parametrize("provider", GPT_OSS_PROVIDERS)
async def test_live_basic_chat_stream(provider: ProviderCase) -> None:
    request = ChatCompletionRequest(
        model=provider.model,
        messages=[ChatMessage(role="user", content="Reply with exactly: stream pong")],
        reasoning_effort=DEFAULT_REASONING_EFFORT,
        max_completion_tokens=96,
        stream_options=ChatStreamOptions(include_usage=True),
        temperature=0,
    )

    deltas = await _stream(provider, request)

    assert any(_delta_has_output(delta) for delta in deltas)


@pytest.mark.parametrize("provider", TOOL_PROVIDERS)
async def test_live_tool_call_with_parallel_tool_calls(provider: ProviderCase) -> None:
    result = await _complete(
        provider,
        ChatCompletionRequest(
            model=provider.model,
            messages=[
                ChatMessage(
                    role="user",
                    content="Call the record_answer tool with answer set to 4.",
                )
            ],
            tools=[
                ChatTool(
                    function=ChatFunctionTool(
                        name="record_answer",
                        description="Record a numeric answer.",
                        parameters={
                            "type": "object",
                            "properties": {"answer": {"type": "integer"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                    )
                )
            ],
            tool_choice=ChatToolChoiceFunction(name="record_answer"),
            parallel_tool_calls=True,
            reasoning_effort=_reasoning_effort_for_tool_provider(provider),
            max_completion_tokens=192,
            temperature=0,
        ),
    )

    tool_calls = result.message.tool_calls or []
    assert any(tool_call.name == "record_answer" for tool_call in tool_calls)


@pytest.mark.parametrize("provider", GPT_OSS_PROVIDERS)
async def test_live_json_object_response_format(provider: ProviderCase) -> None:
    result = await _complete(
        provider,
        ChatCompletionRequest(
            model=provider.model,
            messages=[
                ChatMessage(
                    role="user",
                    content='Return only a JSON object like {"ok": true}.',
                )
            ],
            response_format=ChatResponseFormat(type="json_object"),
            reasoning_effort=DEFAULT_REASONING_EFFORT,
            max_completion_tokens=128,
            temperature=0,
        ),
    )

    content = result.message.content or ""
    parsed = json.loads(content)
    assert isinstance(parsed, dict)


@pytest.mark.parametrize("provider", GPT_OSS_PROVIDERS)
async def test_live_gpt_oss_reasoning_request(provider: ProviderCase) -> None:
    result = await _complete(
        provider,
        ChatCompletionRequest(
            model=provider.model,
            messages=[
                ChatMessage(
                    role="user",
                    content="Solve 17 + 25. Keep the final answer short.",
                )
            ],
            reasoning_effort=DEFAULT_REASONING_EFFORT,
            max_completion_tokens=128,
            temperature=0,
        ),
    )

    assert _message_has_output(result.message)


@pytest.mark.parametrize(
    "response_format",
    [
        pytest.param(ChatResponseFormat(type="json_object"), id="json-object"),
        pytest.param(
            ChatResponseFormat(
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
            id="json-schema",
        ),
    ],
)
async def test_live_groq_gpt_oss_20b_response_format(
    response_format: ChatResponseFormat,
) -> None:
    provider = ProviderCase(
        name="groq",
        api_key_env="GROQ_API_KEY",
        model=GROQ_GPT_OSS_20B_MODEL,
        client_factory=_groq_client,
    )

    result = await _complete(
        provider,
        ChatCompletionRequest(
            model=provider.model,
            messages=[
                ChatMessage(
                    role="system",
                    content="You are a strict JSON API. Output valid JSON only.",
                ),
                ChatMessage(role="user", content='Return exactly {"ok": true}.'),
            ],
            response_format=response_format,
            reasoning_effort="none",
            max_completion_tokens=128,
            temperature=0,
        ),
    )

    parsed = json.loads(result.message.content or "")
    assert parsed == {"ok": True}


@pytest.mark.parametrize(
    "response_format",
    [
        pytest.param(ChatResponseFormat(type="json_object"), id="json-object"),
        pytest.param(
            ChatResponseFormat(
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
            id="json-schema",
        ),
    ],
)
async def test_live_cerebras_glm_4_7_response_format(
    response_format: ChatResponseFormat,
) -> None:
    provider = ProviderCase(
        name="cerebras",
        api_key_env="CEREBRAS_API_KEY",
        model=CEREBRAS_GLM_4_7_MODEL,
        client_factory=_cerebras_client,
    )

    result = await _complete(
        provider,
        ChatCompletionRequest(
            model=provider.model,
            messages=[
                ChatMessage(
                    role="system",
                    content="You are a strict JSON API. Output valid JSON only.",
                ),
                ChatMessage(role="user", content='Return exactly {"ok": true}.'),
            ],
            response_format=response_format,
            reasoning_effort="none",
            max_completion_tokens=128,
            temperature=0,
        ),
    )

    parsed = json.loads(result.message.content or "")
    assert parsed == {"ok": True}


async def test_live_cerebras_glm_4_7_reasoning_request() -> None:
    provider = ProviderCase(
        name="cerebras",
        api_key_env="CEREBRAS_API_KEY",
        model=CEREBRAS_GLM_4_7_MODEL,
        client_factory=_cerebras_client,
    )

    result = await _complete(
        provider,
        ChatCompletionRequest(
            model=provider.model,
            messages=[
                ChatMessage(
                    role="user",
                    content="Think through 17 + 25. Keep the final answer short.",
                )
            ],
            reasoning_effort="high",
            max_completion_tokens=256,
            temperature=0,
        ),
    )

    assert _message_has_output(result.message)
    assert result.message.reasoning_content


async def test_live_gmicloud_deepseek_v4_flash_basic_reasoning_request() -> None:
    provider = ProviderCase(
        name="gmicloud",
        api_key_env="GMICLOUD_API_KEY",
        model=GMICLOUD_DEEPSEEK_V4_FLASH_MODEL,
        client_factory=_gmicloud_client,
    )

    result = await _complete(
        provider,
        ChatCompletionRequest(
            model=provider.model,
            messages=[
                ChatMessage(
                    role="user",
                    content="Think through 23 * 7, then give only the final product.",
                )
            ],
            reasoning_effort="high",
            max_completion_tokens=256,
            temperature=0.6,
        ),
    )

    assert _message_has_output(result.message)


async def test_live_gmicloud_deepseek_v4_flash_xhigh_reasoning_request() -> None:
    provider = ProviderCase(
        name="gmicloud",
        api_key_env="GMICLOUD_API_KEY",
        model=GMICLOUD_DEEPSEEK_V4_FLASH_MODEL,
        client_factory=_gmicloud_client,
    )

    result = await _complete(
        provider,
        ChatCompletionRequest(
            model=provider.model,
            messages=[
                ChatMessage(
                    role="user",
                    content="Think through 19 * 9, then give only the final product.",
                )
            ],
            reasoning_effort="xhigh",
            max_completion_tokens=256,
            temperature=0.6,
        ),
    )

    assert _message_has_output(result.message)


async def test_live_gmicloud_deepseek_v4_flash_tool_replay_round_trip() -> None:
    provider = ProviderCase(
        name="gmicloud",
        api_key_env="GMICLOUD_API_KEY",
        model=GMICLOUD_DEEPSEEK_V4_FLASH_MODEL,
        client_factory=_gmicloud_client,
    )
    tool = ChatTool(
        function=ChatFunctionTool(
            name="record_answer",
            description="Record a numeric answer.",
            parameters={
                "type": "object",
                "properties": {"answer": {"type": "integer"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        )
    )
    first_request = ChatCompletionRequest(
        model=provider.model,
        messages=[
            ChatMessage(
                role="user",
                content="Call the record_answer tool with answer set to 4.",
            )
        ],
        tools=[tool],
        tool_choice="required",
        reasoning_effort="high",
        max_completion_tokens=192,
        temperature=0,
    )

    first = await _complete(provider, first_request)

    tool_calls = first.message.tool_calls or []
    assert tool_calls

    second = await _complete(
        provider,
        ChatCompletionRequest(
            model=provider.model,
            messages=[
                ChatMessage(role="user", content="Call the record_answer tool with answer set to 4."),
                ChatMessage(
                    role="assistant",
                    content=first.message.content or "",
                    reasoning_content=first.message.reasoning_content,
                    tool_calls=tool_calls,
                ),
                ChatMessage(role="tool", tool_call_id=tool_calls[0].id, content="recorded 4"),
                ChatMessage(role="user", content="Now tell me what happened in one sentence."),
            ],
            tools=[tool],
            reasoning_effort="high",
            max_completion_tokens=192,
            temperature=0,
        ),
    )

    assert _message_has_output(second.message)


async def test_live_gmicloud_mimo_none_reasoning_effort_request() -> None:
    provider = ProviderCase(
        name="gmicloud",
        api_key_env="GMICLOUD_API_KEY",
        model=GMICLOUD_MIMO_V25_PRO_MODEL,
        client_factory=_gmicloud_client,
    )

    result = await _complete(
        provider,
        ChatCompletionRequest(
            model=provider.model,
            messages=[ChatMessage(role="user", content="Reply with exactly: pong")],
            reasoning_effort="none",
            max_completion_tokens=96,
            temperature=0,
        ),
    )

    assert _message_has_output(result.message)


async def test_live_crof_qwen_3_5_9b_basic_completion() -> None:
    result = await _complete(
        CROF_PROVIDER,
        ChatCompletionRequest(
            model=CROF_PROVIDER.model,
            messages=[ChatMessage(role="user", content="Reply with exactly: pong")],
            max_completion_tokens=256,
            temperature=0,
        ),
    )

    assert (result.message.content or "").strip()


async def test_live_crof_qwen_3_5_9b_stream() -> None:
    deltas = await _stream(
        CROF_PROVIDER,
        ChatCompletionRequest(
            model=CROF_PROVIDER.model,
            messages=[ChatMessage(role="user", content="Reply with exactly: pong")],
            max_completion_tokens=256,
            stream_options=ChatStreamOptions(include_usage=True),
            temperature=0,
        ),
    )

    assert any((delta.content_delta or "").strip() for delta in deltas)
    assert any(delta.usage is not None for delta in deltas)


@pytest.mark.parametrize(
    "response_format",
    [
        pytest.param(ChatResponseFormat(type="json_object"), id="json-object"),
        pytest.param(
            ChatResponseFormat(
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
            id="json-schema",
        ),
    ],
)
async def test_live_crof_qwen_3_5_9b_response_format(
    response_format: ChatResponseFormat,
) -> None:
    result = await _complete(
        CROF_PROVIDER,
        ChatCompletionRequest(
            model=CROF_PROVIDER.model,
            messages=[ChatMessage(role="user", content='Return exactly {"ok": true}.')],
            response_format=response_format,
            max_completion_tokens=512,
            temperature=0,
        ),
    )

    parsed = json.loads(result.message.content or "")
    assert parsed == {"ok": True}


async def test_live_crof_qwen_3_5_9b_reasoning_request() -> None:
    result = await _complete(
        CROF_PROVIDER,
        ChatCompletionRequest(
            model=CROF_PROVIDER.model,
            messages=[
                ChatMessage(
                    role="user",
                    content="Think through 17 + 25. Keep the final answer short.",
                )
            ],
            reasoning_effort="low",
            max_completion_tokens=256,
            temperature=0,
        ),
    )

    assert result.message.reasoning_content


def _client(provider: ProviderCase) -> IChatCompletionClient:
    _load_money_env()
    api_key = os.getenv(provider.api_key_env)
    if not api_key:
        pytest.skip(f"{provider.api_key_env} is not set")
    if reason := _UNAVAILABLE_PROVIDER_REASONS.get(provider.name):
        pytest.skip(reason)
    return provider.client_factory(api_key)


async def _complete(
    provider: ProviderCase,
    request: ChatCompletionRequest,
):
    try:
        return await _client(provider).complete(request)
    except ChatCompletionProviderError as exc:
        _skip_if_provider_account_unavailable(provider, exc)
        raise


async def _stream(
    provider: ProviderCase,
    request: ChatCompletionRequest,
) -> list[ChatCompletionDelta]:
    try:
        return [delta async for delta in _client(provider).stream(request)]
    except ChatCompletionProviderError as exc:
        _skip_if_provider_account_unavailable(provider, exc)
        raise


def _reasoning_effort_for_tool_provider(
    provider: ProviderCase,
) -> ReasoningEffort | None:
    if "gpt-oss" in provider.model:
        return DEFAULT_REASONING_EFFORT
    return None


def _skip_if_provider_account_unavailable(
    provider: ProviderCase,
    exc: ChatCompletionProviderError,
) -> None:
    message = str(exc).lower()
    unavailable_terms = (
        "insufficient_balance",
        "insufficient balance",
        "does not have enough credits",
        "not have enough credits",
        "account suspended",
        "spending limit",
    )
    if any(term in message for term in unavailable_terms) or (provider.name == "fireworks" and "precondition failed" in message):
        reason = f"{provider.name} provider account is unavailable: {exc}"
        _UNAVAILABLE_PROVIDER_REASONS[provider.name] = reason
        pytest.skip(reason)


def _load_money_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        os.environ.setdefault(key.strip(), _unquote(value.strip()))


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _message_has_output(message: ChatMessage) -> bool:
    return bool((message.content or "").strip() or (message.reasoning_content or "").strip() or message.tool_calls)


def _delta_has_output(delta: ChatCompletionDelta) -> bool:
    return bool(
        (delta.content_delta or "").strip()
        or (delta.reasoning_delta or "").strip()
        or delta.tool_call_delta
    )
