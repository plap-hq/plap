from __future__ import annotations

import os
from pathlib import Path

import pytest

from plap.llms.client import ChatCompletionClient
from plap.llms.chat import ChatCompletionRequest, ChatMessage
from plap.llms.errors import ChatCompletionProviderError
from plap.llms.providers import build_fireworks_provider
from plap.settings import Settings

pytestmark = pytest.mark.expensive

FIREWORKS_GPT_OSS_20B_MODEL = "accounts/fireworks/models/gpt-oss-20b"


async def test_live_fireworks_non_streaming_exceeds_4096_output_tokens() -> None:
    result = await _complete(
        ChatCompletionRequest(
            model=FIREWORKS_GPT_OSS_20B_MODEL,
            messages=[
                ChatMessage(
                    role="user",
                    content=("Repeat the exact word ping 4300 times separated by single spaces. Output nothing else."),
                )
            ],
            max_completion_tokens=5200,
            temperature=0,
        ),
    )

    assert result.usage is not None
    assert result.usage.output_tokens > 4096
    assert result.message.content


async def test_live_fireworks_non_streaming_exceeds_4096_input_tokens() -> None:
    result = await _complete(
        ChatCompletionRequest(
            model=FIREWORKS_GPT_OSS_20B_MODEL,
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        "Read the following repeated marker text and reply exactly: "
                        "input-ok\n\n" + "marker " * 5200 + "\n\nRemember: reply exactly input-ok."
                    ),
                )
            ],
            max_completion_tokens=32,
            temperature=0,
        ),
    )

    assert result.usage is not None
    assert result.usage.input_tokens > 4096
    assert result.usage.output_tokens > 0


async def _complete(request: ChatCompletionRequest):
    try:
        return await _fireworks_client().complete(request)
    except ChatCompletionProviderError as exc:
        _skip_if_provider_account_unavailable(exc)
        raise


def _fireworks_client() -> ChatCompletionClient:
    _load_expensive_env()
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        pytest.skip("FIREWORKS_API_KEY is not set")
    provider = build_fireworks_provider(
        Settings(
            api_key_pepper="pepper",
            database_url="postgresql+asyncpg://example/test",
            sealing_keys=["a" * 43],
            llm_fireworks_api_key=api_key,
        )
    )
    assert provider is not None
    return ChatCompletionClient(provider)


def _skip_if_provider_account_unavailable(exc: ChatCompletionProviderError) -> None:
    message = str(exc).lower()
    unavailable_terms = (
        "insufficient_balance",
        "does not have enough credits",
        "not have enough credits",
        "account suspended",
        "spending limit",
        "precondition failed",
    )
    if any(term in message for term in unavailable_terms):
        pytest.skip(f"fireworks provider account is unavailable: {exc}")


def _load_expensive_env() -> None:
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
