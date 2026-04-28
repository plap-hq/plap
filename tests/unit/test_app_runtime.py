from __future__ import annotations

import pytest

from plap.app import (
    _create_chat_completion_client,
    _create_tool_call_classifier,
    _create_tool_classifier,
    _create_web_search_tool_provider,
)
from plap.llms.router import (
    RoutingChatCompletionClient,
    UnavailableChatCompletionClient,
)
from plap.responses.tools import (
    TOOL_CALL_EFFECT_CLASSIFIER_NAME,
    TOOL_EFFECT_CLASSIFIER_NAME,
    LLMToolCallClassifier,
    LLMToolClassifier,
)
from plap.responses.tools.web_search import MCPWebSearchToolProvider
from plap.settings import Settings


def test_app_runtime_uses_unavailable_chat_client_without_provider_keys() -> None:
    client = _create_chat_completion_client(_settings())

    assert isinstance(client, UnavailableChatCompletionClient)


def test_app_runtime_builds_router_from_provider_prefix_settings() -> None:
    client = _create_chat_completion_client(
        _settings(
            llm_lightning_api_key="lightning-key",
            llm_novita_api_key="novita-key",
        )
    )

    assert isinstance(client, RoutingChatCompletionClient)


def test_app_runtime_rejects_unrouted_tool_classifier_model() -> None:
    client = _create_chat_completion_client(_settings())

    with pytest.raises(ValueError, match="tool_classifier_model"):
        _create_tool_classifier(
            _settings(tool_classifier_model="lightning-ai/gpt-oss-20b"),
            client,
        )


def test_app_runtime_builds_tool_classifier_for_routed_model() -> None:
    settings = _settings(
        llm_lightning_api_key="lightning-key",
        tool_classifier_model="lightning-ai/gpt-oss-20b",
        tool_classifier_max_concurrency=2,
    )
    client = _create_chat_completion_client(settings)

    classifier = _create_tool_classifier(settings, client)

    assert isinstance(classifier, LLMToolClassifier)
    assert classifier.classifier == TOOL_EFFECT_CLASSIFIER_NAME
    assert classifier.classifier_model == "lightning-ai/gpt-oss-20b"


def test_app_runtime_builds_tool_call_classifier_from_tool_model() -> None:
    settings = _settings(
        llm_lightning_api_key="lightning-key",
        tool_classifier_model="lightning-ai/gpt-oss-20b",
    )
    client = _create_chat_completion_client(settings)

    classifier = _create_tool_call_classifier(settings, client)

    assert isinstance(classifier, LLMToolCallClassifier)
    assert classifier.classifier == TOOL_CALL_EFFECT_CLASSIFIER_NAME
    assert classifier.classifier_model == "lightning-ai/gpt-oss-20b"


def test_app_runtime_tool_call_classifier_model_can_override_tool_model() -> None:
    settings = _settings(
        llm_lightning_api_key="lightning-key",
        tool_classifier_model="lightning-ai/gpt-oss-20b",
        tool_call_classifier_model="lightning-ai/gpt-oss-120b",
    )
    client = _create_chat_completion_client(settings)

    classifier = _create_tool_call_classifier(settings, client)

    assert isinstance(classifier, LLMToolCallClassifier)
    assert classifier.classifier_model == "lightning-ai/gpt-oss-120b"


def test_app_runtime_rejects_unrouted_tool_call_classifier_model() -> None:
    client = _create_chat_completion_client(_settings())

    with pytest.raises(ValueError, match="tool_call_classifier_model"):
        _create_tool_call_classifier(
            _settings(tool_call_classifier_model="lightning-ai/gpt-oss-20b"),
            client,
        )


def test_app_runtime_omits_web_search_provider_without_config() -> None:
    assert _create_web_search_tool_provider(_settings()) is None


def test_app_runtime_builds_brave_web_search_provider_from_api_key() -> None:
    provider = _create_web_search_tool_provider(
        _settings(
            web_search_brave_api_key="brave-key",
            web_search_mcp_tool_names=["brave_web_search"],
        )
    )

    assert isinstance(provider, MCPWebSearchToolProvider)


def test_app_runtime_builds_web_search_provider_from_mcp_url() -> None:
    provider = _create_web_search_tool_provider(
        _settings(web_search_mcp_url="http://localhost:8765/mcp")
    )

    assert isinstance(provider, MCPWebSearchToolProvider)


def _settings(**overrides: object) -> Settings:
    values = {
        "api_key_pepper": "pepper",
        "database_url": "postgresql+asyncpg://example/test",
        "sealing_keys": ["a" * 43],
    }
    values.update(overrides)
    return Settings(**values)
