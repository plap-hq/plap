from __future__ import annotations

import pytest

from plap.app import (
    _create_chat_completion_client,
    _create_mcp_tool_providers,
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
    MCPServerConfig,
    PublicUsageConfig,
    RuntimeActorConfig,
    RuntimeActorOverride,
    RuntimeModelInfoConfig,
    RuntimeModelInfoOverride,
    RuntimeModelPricingConfig,
    RuntimeModelPricingOverride,
    RuntimeModelProfileConfig,
    RuntimeProfileOverride,
    RuntimeSelector,
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
    assert profile.main.model == "crof/qwen3.5-9b"
    assert profile.main_debate.model == "crof/qwen3.5-9b"
    assert profile.reviewer.model == "crof/qwen3.5-9b"
    assert profile.arbitrator.model == "crof/qwen3.5-9b"
    assert profile.reasoning_summarizer.model == "lightning/lightning-ai/gpt-oss-120b"
    assert profile.reviewer_transcript_token_budget == 200_000
    assert profile.arbitrator_transcript_token_budget == 200_000
    assert profile.compression_soft_token_budget == 100_000
    assert profile.compression_hard_token_budget == 150_000
    assert profile.compression_max_rounds == 3
    assert profile.debate_max_rounds == 2
    assert profile.model_info.provider == "plap"
    assert profile.model_info.max_input_tokens == 200_000
    assert profile.model_info.description == "General-purpose plap responses model for text and tool use."
    assert profile.main.public_usage == PublicUsageConfig()
    assert "service_tier" not in profile.model_info.supported_parameters
    assert profile.by_service_tier == {}
    assert profile.by_reasoning_effort["high"].main is not None
    assert profile.by_reasoning_effort["high"].main.reasoning_effort == "high"


def test_app_runtime_includes_wisp_default_profile() -> None:
    settings = _settings(
        llm_crof_api_key="crof-key",
        llm_lightning_api_key="lightning-key",
    )

    _validate_runtime_model_profiles(settings)

    profile = settings.runtime_model_profiles["plap-ai/wisp"]
    assert profile.display_name == "Wisp"
    assert profile.main.model == "crof/glm-5.1"
    assert profile.main_debate.model == "crof/qwen3.5-397b-a17b"
    assert profile.reviewer.model == "crof/deepseek-v4-flash"
    assert profile.arbitrator.model == "crof/deepseek-v4-flash"
    assert profile.reasoning_summarizer.model == "lightning/lightning-ai/gpt-oss-120b"
    assert profile.reviewer_transcript_token_budget == 500_000
    assert profile.arbitrator_transcript_token_budget == 500_000
    assert profile.model_info.max_input_tokens == 200_000
    assert profile.model_info.max_output_tokens == 32_768


def test_app_runtime_wisp_nano_rejects_service_tier() -> None:
    settings = _settings(
        llm_crof_api_key="crof-key",
        llm_lightning_api_key="lightning-key",
    )

    with pytest.raises(ValueError, match="unsupported request parameters: service_tier"):
        settings.resolve_runtime_model_profile("plap-ai/wisp-nano", selector=RuntimeSelector(service_tier="priority"))


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
    assert _create_mcp_tool_providers(_settings()) == ()


def test_app_runtime_builds_mcp_providers_from_config_list() -> None:
    providers = _create_mcp_tool_providers(
        _settings(
            mcp_servers=[
                MCPServerConfig(
                    name="search",
                    config={
                        "mcpServers": {
                            "search": {
                                "command": "search-server",
                                "args": ["--stdio"],
                            }
                        }
                    },
                    tool_names=["search_web"],
                )
            ],
        )
    )

    assert len(providers) == 1
    assert isinstance(providers[0], MCPToolProvider)


def test_app_runtime_builds_mcp_provider_from_url() -> None:
    providers = _create_mcp_tool_providers(
        _settings(mcp_servers=[MCPServerConfig(name="remote", url="http://localhost:8765/mcp")])
    )

    assert len(providers) == 1
    assert isinstance(providers[0], MCPToolProvider)


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
    assert profile.main.model == "lightning/lightning-ai/gpt-oss-20b"
    assert profile.main_debate.model == "lightning/lightning-ai/gpt-oss-120b"
    assert profile.reasoning_summarizer.model == "lightning/lightning-ai/llama-3.3-70b"


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


def test_app_runtime_validates_runtime_profile_variants() -> None:
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
                by_service_tier={
                    "priority": RuntimeProfileOverride(
                        main=RuntimeActorOverride(model="lightning/lightning-ai/gpt-oss-120b"),
                        reviewer=RuntimeActorOverride(model="lightning/lightning-ai/gpt-oss-120b"),
                        reviewer_transcript_token_budget=4096,
                        arbitrator_transcript_token_budget=2048,
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


def test_app_runtime_rejects_unrouted_runtime_profile_variant() -> None:
    settings = _settings(
        llm_crof_api_key="crof-key",
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="crof/qwen3.5-9b",
                main_debate_model="crof/qwen3.5-9b",
                reviewer_model="crof/qwen3.5-9b",
                arbitrator_model="crof/qwen3.5-9b",
                reasoning_summarizer_model="crof/qwen3.5-9b",
                by_service_tier={
                    "priority": RuntimeProfileOverride(
                        main=RuntimeActorOverride(model="lightning/lightning-ai/gpt-oss-120b")
                    )
                },
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
                reviewer_transcript_token_budget=1024,
                arbitrator_transcript_token_budget=768,
            )
        },
    )

    profile = settings.resolve_runtime_model_profile("plap/standard")

    assert profile is settings.runtime_model_profiles["plap/standard"]
    assert profile.display_name == "Test Model"
    assert profile.main.model == "lightning/lightning-ai/gpt-oss-20b"
    assert profile.reviewer_transcript_token_budget == 1024
    assert profile.arbitrator_transcript_token_budget == 768
    assert settings.resolve_runtime_model_profile("plap/standard", selector=RuntimeSelector(service_tier="default")) is profile
    assert settings.resolve_runtime_model_profile("plap/standard", selector=RuntimeSelector(service_tier="auto")) is profile
    with pytest.raises(ValueError, match="model is required"):
        settings.resolve_runtime_model_profile(None)
    with pytest.raises(ValueError, match="unknown runtime model"):
        settings.resolve_runtime_model_profile("lightning/lightning-ai/gpt-oss-20b")


def test_app_runtime_rejects_unsupported_service_tier() -> None:
    settings = _settings(
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="crof/qwen3.5-9b",
                main_debate_model="crof/qwen3.5-9b",
                reviewer_model="crof/qwen3.5-9b",
                arbitrator_model="crof/qwen3.5-9b",
                reasoning_summarizer_model="crof/qwen3.5-9b",
                supported_parameters=["tools", "response_format"],
            )
        },
    )

    with pytest.raises(ValueError, match="unsupported request parameters: service_tier"):
        settings.resolve_runtime_model_profile("plap/standard", selector=RuntimeSelector(service_tier="priority"))


def test_app_runtime_rejects_unsupported_reasoning_effort() -> None:
    settings = _settings(
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="crof/qwen3.5-9b",
                main_debate_model="crof/qwen3.5-9b",
                reviewer_model="crof/qwen3.5-9b",
                arbitrator_model="crof/qwen3.5-9b",
                reasoning_summarizer_model="crof/qwen3.5-9b",
                supported_parameters=["tools", "response_format"],
            )
        },
    )

    with pytest.raises(ValueError, match="unsupported request parameters: reasoning_effort"):
        settings.resolve_runtime_model_profile("plap/standard", selector=RuntimeSelector(reasoning_effort="high"))


def test_runtime_profile_rejects_service_tier_overrides_without_supported_parameter() -> None:
    with pytest.raises(ValueError, match="service_tier overrides require"):
        _profile_config(
            main_model="crof/qwen3.5-9b",
            main_debate_model="crof/qwen3.5-9b",
            reviewer_model="crof/qwen3.5-9b",
            arbitrator_model="crof/qwen3.5-9b",
            reasoning_summarizer_model="crof/qwen3.5-9b",
            supported_parameters=["tools", "response_format"],
            by_service_tier={
                "priority": RuntimeProfileOverride(main=RuntimeActorOverride(service_tier="priority")),
            },
        )


def test_runtime_profile_rejects_reasoning_effort_overrides_without_supported_parameter() -> None:
    with pytest.raises(ValueError, match="reasoning_effort overrides require"):
        _profile_config(
            main_model="crof/qwen3.5-9b",
            main_debate_model="crof/qwen3.5-9b",
            reviewer_model="crof/qwen3.5-9b",
            arbitrator_model="crof/qwen3.5-9b",
            reasoning_summarizer_model="crof/qwen3.5-9b",
            supported_parameters=["tools", "response_format"],
            by_reasoning_effort={
                "high": RuntimeProfileOverride(main=RuntimeActorOverride(reasoning_effort="high")),
            },
        )


def test_app_runtime_resolves_runtime_profile_variants() -> None:
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
                by_service_tier={
                    "priority": RuntimeProfileOverride(
                        main=RuntimeActorOverride(model="lightning/lightning-ai/gpt-oss-120b"),
                        reviewer=RuntimeActorOverride(model="lightning/lightning-ai/gpt-oss-120b"),
                        reviewer_transcript_token_budget=8192,
                        arbitrator_transcript_token_budget=4096,
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
    priority = settings.resolve_runtime_model_profile("plap/standard", selector=RuntimeSelector(service_tier="priority"))
    flex = settings.resolve_runtime_model_profile("plap/standard", selector=RuntimeSelector(service_tier="flex"))

    assert priority is not base
    assert priority.display_name == "Test Model"
    assert priority.main.model == "lightning/lightning-ai/gpt-oss-120b"
    assert priority.reviewer.model == "lightning/lightning-ai/gpt-oss-120b"
    assert priority.main_debate.model == "crof/qwen3.5-9b"
    assert priority.arbitrator.model == "crof/qwen3.5-9b"
    assert priority.reviewer_transcript_token_budget == 8192
    assert priority.arbitrator_transcript_token_budget == 4096
    assert priority.compression_soft_token_budget == 5000
    assert priority.compression_hard_token_budget == 6000
    assert priority.compression_max_rounds == 1
    assert priority.debate_max_rounds == 1
    assert base.compression_soft_token_budget == 2000
    assert base.compression_hard_token_budget == 3000
    assert base.compression_max_rounds == 2
    assert base.debate_max_rounds == 2
    assert flex is base


def test_app_runtime_resolves_reasoning_effort_variant() -> None:
    settings = _settings(
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="crof/qwen3.5-9b",
                main_debate_model="crof/qwen3.5-9b",
                reviewer_model="crof/qwen3.5-9b",
                arbitrator_model="crof/qwen3.5-9b",
                reasoning_summarizer_model="crof/qwen3.5-9b",
                by_reasoning_effort={
                    "high": RuntimeProfileOverride(
                        main=RuntimeActorOverride(model="lightning/lightning-ai/gpt-oss-120b"),
                        main_debate=RuntimeActorOverride(model="lightning/lightning-ai/gpt-oss-120b"),
                    )
                },
            )
        },
    )

    profile = settings.resolve_runtime_model_profile("plap/standard", selector=RuntimeSelector(reasoning_effort="high"))

    assert profile.main.model == "lightning/lightning-ai/gpt-oss-120b"
    assert profile.main_debate.model == "lightning/lightning-ai/gpt-oss-120b"
    assert profile.reviewer.model == "crof/qwen3.5-9b"


def test_app_runtime_resolves_model_info_overrides() -> None:
    settings = _settings(
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="crof/qwen3.5-9b",
                main_debate_model="crof/qwen3.5-9b",
                reviewer_model="crof/qwen3.5-9b",
                arbitrator_model="crof/qwen3.5-9b",
                reasoning_summarizer_model="crof/qwen3.5-9b",
                by_service_tier={
                    "priority": RuntimeProfileOverride(
                        model_info=RuntimeModelInfoOverride(
                            display_name="Priority Model",
                            description="Priority profile",
                            mode="responses",
                            input_modalities=["text"],
                            output_modalities=["text"],
                            max_input_tokens=4096,
                            max_output_tokens=1024,
                            supported_parameters=["tools"],
                            pricing=RuntimeModelPricingOverride(input_per_token=0.0, output_per_token=0.0),
                            provider="plap",
                            deprecated=False,
                        )
                    )
                },
            )
        }
    )

    base = settings.resolve_runtime_model_profile("plap/standard")
    priority = settings.resolve_runtime_model_profile("plap/standard", selector=RuntimeSelector(service_tier="priority"))

    assert base.model_info.description == "Test profile"
    assert priority.model_info.display_name == "Priority Model"
    assert priority.model_info.description == "Priority profile"
    assert priority.model_info.max_input_tokens == 4096


def test_runtime_profile_rejects_conflicting_service_and_reasoning_overrides() -> None:
    with pytest.raises(ValueError, match="both set"):
        _profile_config(
            main_model="crof/qwen3.5-9b",
            main_debate_model="crof/qwen3.5-9b",
            reviewer_model="crof/qwen3.5-9b",
            arbitrator_model="crof/qwen3.5-9b",
            reasoning_summarizer_model="crof/qwen3.5-9b",
            by_service_tier={
                "priority": RuntimeProfileOverride(main=RuntimeActorOverride(model="lightning/lightning-ai/gpt-oss-120b"))
            },
            by_reasoning_effort={
                "high": RuntimeProfileOverride(main=RuntimeActorOverride(model="crof/glm-4.7-flash"))
            },
        )


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
    reviewer_transcript_token_budget: int = 0,
    arbitrator_transcript_token_budget: int = 0,
    compression_soft_token_budget: int | None = None,
    compression_hard_token_budget: int | None = None,
    compression_max_rounds: int = 3,
    debate_max_rounds: int = 2,
    by_service_tier: dict[str, RuntimeProfileOverride] | None = None,
    by_reasoning_effort: dict[str, RuntimeProfileOverride] | None = None,
    supported_parameters: list[str] | None = None,
) -> RuntimeModelProfileConfig:
    return RuntimeModelProfileConfig(
        display_name=display_name,
        model_info=RuntimeModelInfoConfig(
            display_name=display_name,
            description="Test profile",
            mode="responses",
            input_modalities=["text"],
            output_modalities=["text"],
            max_input_tokens=8192,
            max_output_tokens=2048,
            supported_parameters=supported_parameters
            or [
                "context_management",
                "tools",
                "tool_choice",
                "parallel_tool_calls",
                "response_format",
                "max_output_tokens",
                "reasoning_effort",
                "service_tier",
                "stream",
                "temperature",
                "top_p",
                "top_logprobs",
            ],
            pricing=RuntimeModelPricingConfig(input_per_token=0.0, output_per_token=0.0),
            provider="plap",
            deprecated=False,
        ),
        main=RuntimeActorConfig(model=main_model),
        main_debate=RuntimeActorConfig(model=main_debate_model),
        reviewer=RuntimeActorConfig(model=reviewer_model),
        arbitrator=RuntimeActorConfig(model=arbitrator_model),
        reasoning_summarizer=RuntimeActorConfig(model=reasoning_summarizer_model),
        reviewer_transcript_token_budget=reviewer_transcript_token_budget,
        arbitrator_transcript_token_budget=arbitrator_transcript_token_budget,
        compression_soft_token_budget=compression_soft_token_budget,
        compression_hard_token_budget=compression_hard_token_budget,
        compression_max_rounds=compression_max_rounds,
        debate_max_rounds=debate_max_rounds,
        by_service_tier=by_service_tier or {},
        by_reasoning_effort=by_reasoning_effort or {},
    )
