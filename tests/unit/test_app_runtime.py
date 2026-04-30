from __future__ import annotations

import pytest

from plap.app import (
    _create_chat_completion_client,
    _create_mcp_tool_provider,
    _create_tool_call_classifier,
    _create_tool_classifier,
    _validate_runtime_model_profiles,
)
from plap.llms.router import (
    RoutingChatCompletionClient,
    UnavailableChatCompletionClient,
)
from plap.responses.tools import (
    TOOL_CALL_EFFECT_CLASSIFIER_MODEL,
    TOOL_CALL_EFFECT_CLASSIFIER_NAME,
    TOOL_EFFECT_CLASSIFIER_MODEL,
    TOOL_EFFECT_CLASSIFIER_NAME,
    LLMToolCallClassifier,
    LLMToolClassifier,
)
from plap.responses.tools.mcp import MCPToolProvider
from plap.settings import (
    RuntimeModelProfileConfig,
    RuntimeModelProfileOverrideConfig,
    Settings,
)


def test_app_runtime_uses_unavailable_chat_client_without_provider_keys() -> None:
    client = _create_chat_completion_client(_settings())

    assert isinstance(client, UnavailableChatCompletionClient)


def test_app_runtime_builds_router_from_provider_prefix_settings() -> None:
    client = _create_chat_completion_client(
        _settings(
            llm_lightning_api_key="lightning-key",
            llm_novita_api_key="novita-key",
            llm_crof_api_key="crof-key",
        )
    )

    assert isinstance(client, RoutingChatCompletionClient)


def test_app_runtime_includes_wisp_nano_default_profile() -> None:
    settings = _settings(
        llm_crof_api_key="crof-key",
        llm_lightning_api_key="lightning-key",
    )

    _validate_runtime_model_profiles(settings)

    profile = settings.runtime_model_profiles["plap-ai/wisp-nano"]
    assert profile.display_name == "Wisp Nano"
    assert profile.main_model == "crof/qwen3.5-9b"
    assert profile.main_debate_model == "crof/qwen3.5-9b"
    assert profile.reviewer_model == "crof/qwen3.5-9b"
    assert profile.arbitrator_model == "crof/qwen3.5-9b"
    assert profile.reasoning_summarizer_model == ("lightning/lightning-ai/gpt-oss-120b")
    assert profile.transcript_token_budget == 200_000
    assert profile.compression_soft_token_budget == 100_000
    assert profile.compression_hard_token_budget == 150_000
    assert profile.compression_max_rounds == 3
    assert profile.debate_max_rounds == 2


def test_app_runtime_validates_crof_provider_prefix() -> None:
    settings = _settings(
        llm_crof_api_key="crof-key",
        runtime_model_profiles={
            "plap/glm": _profile_config(
                main_model="crof/qwen3.5-9b",
                main_debate_model="crof/qwen3.5-9b",
                reviewer_model="crof/glm-4.7-flash",
                arbitrator_model="crof/glm-4.7-flash",
                reasoning_summarizer_model="crof/qwen3.5-9b",
            )
        },
    )

    _validate_runtime_model_profiles(settings)


def test_app_runtime_rejects_unrouted_tool_classifier_route() -> None:
    client = _create_chat_completion_client(_settings())

    with pytest.raises(ValueError, match="tool effect classifier model"):
        _create_tool_classifier(_settings(), client)


def test_app_runtime_builds_tool_classifier_for_routed_model() -> None:
    settings = _settings(
        llm_lightning_api_key="lightning-key",
        tool_classifier_max_concurrency=2,
    )
    client = _create_chat_completion_client(settings)

    classifier = _create_tool_classifier(settings, client)

    assert isinstance(classifier, LLMToolClassifier)
    assert classifier.classifier == TOOL_EFFECT_CLASSIFIER_NAME
    assert classifier.classifier_model == TOOL_EFFECT_CLASSIFIER_MODEL


def test_app_runtime_builds_tool_call_classifier_for_routed_model() -> None:
    settings = _settings(llm_lightning_api_key="lightning-key")
    client = _create_chat_completion_client(settings)

    classifier = _create_tool_call_classifier(settings, client)

    assert isinstance(classifier, LLMToolCallClassifier)
    assert classifier.classifier == TOOL_CALL_EFFECT_CLASSIFIER_NAME
    assert classifier.classifier_model == TOOL_CALL_EFFECT_CLASSIFIER_MODEL


def test_app_runtime_rejects_unrouted_tool_call_classifier_route() -> None:
    client = _create_chat_completion_client(_settings())

    with pytest.raises(ValueError, match="tool call classifier model"):
        _create_tool_call_classifier(_settings(), client)


def test_app_runtime_omits_mcp_provider_without_config() -> None:
    assert _create_mcp_tool_provider(_settings()) is None


def test_app_runtime_builds_mcp_provider_from_config() -> None:
    provider = _create_mcp_tool_provider(
        _settings(
            web_search_mcp_config={
                "mcpServers": {
                    "search": {
                        "command": "search-server",
                        "args": ["--stdio"],
                    }
                }
            },
            web_search_mcp_tool_names=["search_web"],
        )
    )

    assert isinstance(provider, MCPToolProvider)


def test_app_runtime_builds_mcp_provider_from_url() -> None:
    provider = _create_mcp_tool_provider(_settings(web_search_mcp_url="http://localhost:8765/mcp"))

    assert isinstance(provider, MCPToolProvider)


def test_app_runtime_validates_synthetic_model_profiles() -> None:
    settings = _settings(
        llm_lightning_api_key="lightning-key",
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="lightning/lightning-ai/gpt-oss-20b",
                main_debate_model="lightning/lightning-ai/gpt-oss-120b",
                reviewer_model="lightning/lightning-ai/gpt-oss-20b",
                arbitrator_model="lightning/lightning-ai/gpt-oss-120b",
                reasoning_summarizer_model="lightning/lightning-ai/llama-3.3-70b",
            )
        },
    )

    _validate_runtime_model_profiles(settings)

    profile = settings.runtime_model_profiles["plap/standard"]
    assert profile.display_name == "Test Model"
    assert profile.main_model == "lightning/lightning-ai/gpt-oss-20b"
    assert profile.main_debate_model == "lightning/lightning-ai/gpt-oss-120b"
    assert profile.reasoning_summarizer_model == "lightning/lightning-ai/llama-3.3-70b"


def test_app_runtime_rejects_runtime_profile_with_unrouted_model() -> None:
    settings = _settings(
        llm_lightning_api_key="lightning-key",
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="lightning/lightning-ai/gpt-oss-20b",
                main_debate_model="openai/gpt-oss-120b",
                reviewer_model="lightning/lightning-ai/gpt-oss-20b",
                arbitrator_model="lightning/lightning-ai/gpt-oss-120b",
                reasoning_summarizer_model="lightning/lightning-ai/llama-3.3-70b",
            )
        },
    )

    with pytest.raises(ValueError, match="unconfigured LLM route"):
        _validate_runtime_model_profiles(settings)


def test_app_runtime_validates_service_tier_profile_overrides() -> None:
    settings = _settings(
        llm_crof_api_key="crof-key",
        llm_lightning_api_key="lightning-key",
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="crof/qwen3.5-9b",
                main_debate_model="crof/qwen3.5-9b",
                reviewer_model="crof/qwen3.5-9b",
                arbitrator_model="crof/qwen3.5-9b",
                reasoning_summarizer_model="crof/qwen3.5-9b",
                service_tier_overrides={
                    "priority": RuntimeModelProfileOverrideConfig(
                        main_model="lightning/lightning-ai/gpt-oss-120b",
                        reviewer_model="lightning/lightning-ai/gpt-oss-120b",
                        transcript_token_budget=4096,
                        compression_soft_token_budget=4096,
                        compression_hard_token_budget=8192,
                        compression_max_rounds=1,
                        debate_max_rounds=1,
                    )
                },
            )
        },
    )

    _validate_runtime_model_profiles(settings)


def test_app_runtime_rejects_unrouted_service_tier_override() -> None:
    settings = _settings(
        llm_crof_api_key="crof-key",
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="crof/qwen3.5-9b",
                main_debate_model="crof/qwen3.5-9b",
                reviewer_model="crof/qwen3.5-9b",
                arbitrator_model="crof/qwen3.5-9b",
                reasoning_summarizer_model="crof/qwen3.5-9b",
                service_tier_overrides={"priority": RuntimeModelProfileOverrideConfig(main_model="lightning/lightning-ai/gpt-oss-120b")},
            )
        },
    )

    with pytest.raises(ValueError, match="unconfigured LLM route"):
        _validate_runtime_model_profiles(settings)


def test_app_runtime_resolves_only_explicit_synthetic_models() -> None:
    settings = _settings(
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="lightning/lightning-ai/gpt-oss-20b",
                main_debate_model="lightning/lightning-ai/gpt-oss-120b",
                reviewer_model="lightning/lightning-ai/gpt-oss-20b",
                arbitrator_model="lightning/lightning-ai/gpt-oss-120b",
                reasoning_summarizer_model="lightning/lightning-ai/llama-3.3-70b",
                transcript_token_budget=1024,
            )
        },
    )

    profile = settings.resolve_runtime_model_profile("plap/standard")

    assert profile is settings.runtime_model_profiles["plap/standard"]
    assert profile.display_name == "Test Model"
    assert profile.main_model == "lightning/lightning-ai/gpt-oss-20b"
    assert profile.transcript_token_budget == 1024
    assert settings.resolve_runtime_model_profile("plap/standard", "default") is profile
    assert settings.resolve_runtime_model_profile("plap/standard", "auto") is profile
    with pytest.raises(ValueError, match="model is required"):
        settings.resolve_runtime_model_profile(None)
    with pytest.raises(ValueError, match="unknown runtime model"):
        settings.resolve_runtime_model_profile("lightning/lightning-ai/gpt-oss-20b")


def test_app_runtime_resolves_service_tier_overrides() -> None:
    settings = _settings(
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="crof/qwen3.5-9b",
                main_debate_model="crof/qwen3.5-9b",
                reviewer_model="crof/qwen3.5-9b",
                arbitrator_model="crof/qwen3.5-9b",
                reasoning_summarizer_model="crof/qwen3.5-9b",
                compression_soft_token_budget=2000,
                compression_hard_token_budget=3000,
                compression_max_rounds=2,
                debate_max_rounds=2,
                service_tier_overrides={
                    "priority": RuntimeModelProfileOverrideConfig(
                        main_model="lightning/lightning-ai/gpt-oss-120b",
                        reviewer_model="lightning/lightning-ai/gpt-oss-120b",
                        transcript_token_budget=8192,
                        compression_soft_token_budget=5000,
                        compression_hard_token_budget=6000,
                        compression_max_rounds=1,
                        debate_max_rounds=1,
                    )
                },
            )
        },
    )

    base = settings.resolve_runtime_model_profile("plap/standard")
    priority = settings.resolve_runtime_model_profile(
        "plap/standard",
        "priority",
    )
    flex = settings.resolve_runtime_model_profile("plap/standard", "flex")

    assert priority is not base
    assert priority.display_name == "Test Model"
    assert priority.main_model == "lightning/lightning-ai/gpt-oss-120b"
    assert priority.reviewer_model == "lightning/lightning-ai/gpt-oss-120b"
    assert priority.main_debate_model == "crof/qwen3.5-9b"
    assert priority.arbitrator_model == "crof/qwen3.5-9b"
    assert priority.transcript_token_budget == 8192
    assert priority.compression_soft_token_budget == 5000
    assert priority.compression_hard_token_budget == 6000
    assert priority.compression_max_rounds == 1
    assert priority.debate_max_rounds == 1
    assert base.compression_soft_token_budget == 2000
    assert base.compression_hard_token_budget == 3000
    assert base.compression_max_rounds == 2
    assert base.debate_max_rounds == 2
    assert flex is base


def test_runtime_profile_rejects_invalid_compression_budgets() -> None:
    with pytest.raises(ValueError, match="hard token budget"):
        _profile_config(
            main_model="crof/qwen3.5-9b",
            main_debate_model="crof/qwen3.5-9b",
            reviewer_model="crof/qwen3.5-9b",
            arbitrator_model="crof/qwen3.5-9b",
            reasoning_summarizer_model="crof/qwen3.5-9b",
            compression_soft_token_budget=1000,
            compression_hard_token_budget=1000,
        )


def _settings(**overrides: object) -> Settings:
    values = {
        "api_key_pepper": "pepper",
        "database_url": "postgresql+asyncpg://example/test",
        "sealing_keys": ["a" * 43],
    }
    values.update(overrides)
    return Settings(**values)


def _profile_config(
    *,
    display_name: str = "Test Model",
    main_model: str,
    main_debate_model: str,
    reviewer_model: str,
    arbitrator_model: str,
    reasoning_summarizer_model: str,
    transcript_token_budget: int = 0,
    compression_soft_token_budget: int | None = None,
    compression_hard_token_budget: int | None = None,
    compression_max_rounds: int = 3,
    debate_max_rounds: int = 2,
    service_tier_overrides: dict[
        str,
        RuntimeModelProfileOverrideConfig,
    ]
    | None = None,
) -> RuntimeModelProfileConfig:
    return RuntimeModelProfileConfig(
        display_name=display_name,
        main_model=main_model,
        main_debate_model=main_debate_model,
        reviewer_model=reviewer_model,
        arbitrator_model=arbitrator_model,
        reasoning_summarizer_model=reasoning_summarizer_model,
        transcript_token_budget=transcript_token_budget,
        compression_soft_token_budget=compression_soft_token_budget,
        compression_hard_token_budget=compression_hard_token_budget,
        compression_max_rounds=compression_max_rounds,
        debate_max_rounds=debate_max_rounds,
        service_tier_overrides=service_tier_overrides or {},
    )
