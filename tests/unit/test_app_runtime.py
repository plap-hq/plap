from __future__ import annotations

import pytest

from plap.app import (
    _create_chat_completion_client,
    _create_mcp_tool_providers,
    _create_tool_call_classifier,
    _create_tool_classifier,
    _validate_runtime_model_profiles,
    _validate_runtime_profile_tokenizers,
)
from plap.errors import PlapError
from plap.llms.completions.router import (
    RoutingChatCompletionClient,
    UnavailableChatCompletionClient,
)
from plap.responses.tools import (
    TOOL_CALL_EFFECT_CLASSIFIER_NAME,
    TOOL_EFFECT_CLASSIFIER_NAME,
    LLMToolCallClassifier,
    LLMToolClassifier,
)
from plap.responses.tools.mcp import MCPToolProvider
from plap.settings import (
    MCPServerConfig,
    MCPToolConfig,
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
            llm_cerebras_api_key="cerebras-key",
            llm_groq_api_key="groq-key",
            llm_gmicloud_api_key="gmicloud-key",
            llm_novita_api_key="novita-key",
            llm_crof_api_key="crof-key",
            llm_openrouter_api_key="openrouter-key",
        )
    )

    assert isinstance(client, RoutingChatCompletionClient)


def test_app_runtime_builds_gmicloud_route_from_provider_prefix_setting() -> None:
    client = _create_chat_completion_client(_settings(llm_gmicloud_api_key="gmicloud-key"))

    assert isinstance(client, RoutingChatCompletionClient)
    assert [route.prefix for route in client._routes] == ["gmicloud/"]


def test_app_runtime_builds_cerebras_route_from_provider_prefix_setting() -> None:
    client = _create_chat_completion_client(_settings(llm_cerebras_api_key="cerebras-key"))

    assert isinstance(client, RoutingChatCompletionClient)
    assert [route.prefix for route in client._routes] == ["cerebras/"]


def test_app_runtime_builds_groq_route_from_provider_prefix_setting() -> None:
    client = _create_chat_completion_client(_settings(llm_groq_api_key="groq-key"))

    assert isinstance(client, RoutingChatCompletionClient)
    assert [route.prefix for route in client._routes] == ["groq/"]


def test_runtime_actor_config_rejects_tokenizer_revision_without_repo() -> None:
    with pytest.raises(ValueError, match="tokenizer_revision requires tokenizer_hf_repo"):
        RuntimeActorConfig(model="crof/qwen3.5-9b", tokenizer_revision="main")


def test_runtime_actor_config_rejects_trust_remote_code_without_repo() -> None:
    with pytest.raises(ValueError, match="tokenizer_trust_remote_code requires tokenizer_hf_repo"):
        RuntimeActorConfig(model="crof/qwen3.5-9b", tokenizer_trust_remote_code=True)


def test_app_runtime_includes_wisp_mini_default_profile() -> None:
    settings = _settings(
        llm_crof_api_key="crof-key",
        llm_cerebras_api_key="cerebras-key",
        llm_groq_api_key="groq-key",
        llm_gmicloud_api_key="gmicloud-key",
        llm_lightning_api_key="lightning-key",
        llm_novita_api_key="novita-key",
        llm_openrouter_api_key="openrouter-key",
    )

    _validate_runtime_model_profiles(settings)

    profile = settings.runtime_model_profiles["plap-ai/wisp-mini"]
    assert profile.compact_max_rounds == 0
    assert profile.debate_max_rounds > 0
    assert profile.main.public_usage == PublicUsageConfig()
    assert {"tools", "response_format", "max_output_tokens", "service_tier", "stream"}.issubset(
        profile.model_info.supported_parameters
    )


def test_app_runtime_includes_wisp_default_profile() -> None:
    settings = _settings(
        llm_crof_api_key="crof-key",
        llm_cerebras_api_key="cerebras-key",
        llm_groq_api_key="groq-key",
        llm_gmicloud_api_key="gmicloud-key",
        llm_lightning_api_key="lightning-key",
        llm_novita_api_key="novita-key",
        llm_openrouter_api_key="openrouter-key",
    )

    _validate_runtime_model_profiles(settings)

    profile = settings.runtime_model_profiles["plap-ai/wisp"]
    assert profile.compact_max_rounds == 0
    assert profile.main.public_usage == PublicUsageConfig()
    assert {"tools", "response_format", "max_output_tokens", "service_tier", "stream"}.issubset(
        profile.model_info.supported_parameters
    )


def test_app_runtime_validates_runtime_profile_tokenizers(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings()
    seen: list[tuple[str, str | None, bool]] = []

    def fake_measure_request_tokens(request, *, actor_config):
        assert request.messages[0].role == "developer"
        assert request.messages[1].role == "user"
        seen.append(
            (
                actor_config.tokenizer_hf_repo,
                actor_config.tokenizer_revision,
                actor_config.tokenizer_trust_remote_code,
            )
        )
        return 1

    monkeypatch.setattr("plap.app.measure_request_tokens", fake_measure_request_tokens)

    _validate_runtime_profile_tokenizers(settings)

    assert set(seen) == {
        ("deepseek-ai/DeepSeek-V4-Flash", "6976c7ff1b30a1b2cb7805021b8ba4684041f136", False),
        ("XiaomiMiMo/MiMo-V2.5-Pro", "a75207db63de3c320950fe6fcfa9ff60f341b7a2", False),
    }


def test_app_runtime_rejects_invalid_runtime_profile_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings()

    def fake_measure_request_tokens(request, *, actor_config):
        _ = request
        if actor_config.tokenizer_hf_repo == "deepseek-ai/DeepSeek-V4-Flash":
            raise AttributeError("missing max_position_embeddings")
        return 1

    monkeypatch.setattr("plap.app.measure_request_tokens", fake_measure_request_tokens)

    with pytest.raises(PlapError) as exc_info:
        _validate_runtime_profile_tokenizers(settings)

    assert exc_info.value.private.reason == "runtime_profile_tokenizer_invalid"


def test_app_runtime_validates_crof_provider_prefix() -> None:
    settings = _settings(
        llm_crof_api_key="crof-key",
        runtime_model_profiles={
            "plap/glm": _profile_config(
                main_model="crof/qwen3.5-9b",
                defender_model="crof/qwen3.5-9b",
                reviewer_model="crof/glm-4.7-flash",
                arbitrator_model="crof/glm-4.7-flash",
                reasoning_summarizer_model="crof/qwen3.5-9b",
            )
        },
    )

    _validate_runtime_model_profiles(settings)


def test_app_runtime_validates_gmicloud_provider_prefix() -> None:
    settings = _settings(
        llm_gmicloud_api_key="gmicloud-key",
        runtime_model_profiles={
            "plap/gmi": _profile_config(
                main_model="gmicloud/openai/gpt-oss-120b",
                defender_model="gmicloud/openai/gpt-oss-120b",
                reviewer_model="gmicloud/openai/gpt-oss-120b",
                arbitrator_model="gmicloud/openai/gpt-oss-120b",
                reasoning_summarizer_model="gmicloud/openai/gpt-oss-120b",
            )
        },
    )

    _validate_runtime_model_profiles(settings)


def test_app_runtime_rejects_unrouted_tool_classifier_route() -> None:
    client = _create_chat_completion_client(_settings())

    with pytest.raises(PlapError) as exc_info:
        _create_tool_classifier(_settings(), client)

    assert exc_info.value.public is None
    assert exc_info.value.private.reason == "tool_effect_classifier_route_unconfigured"


def test_app_runtime_builds_tool_classifier_for_routed_model() -> None:
    settings = _settings(
        llm_lightning_api_key="lightning-key",
        llm_openrouter_api_key="openrouter-key",
        tool_classifier_max_concurrency=2,
    )
    client = _create_chat_completion_client(settings)

    classifier = _create_tool_classifier(settings, client)

    assert isinstance(classifier, LLMToolClassifier)
    assert classifier.classifier == TOOL_EFFECT_CLASSIFIER_NAME
    assert classifier.classifier_model == settings.tool_effect_classifier_model
    assert classifier.classifier_cache_model == settings.tool_effect_classifier_cache_model


def test_app_runtime_builds_tool_classifier_for_configured_model() -> None:
    settings = _settings(
        llm_crof_api_key="crof-key",
        tool_effect_classifier_model="crof/qwen3.5-9b",
        tool_effect_classifier_cache_model="qwen3.5-9b",
    )
    client = _create_chat_completion_client(settings)

    classifier = _create_tool_classifier(settings, client)

    assert isinstance(classifier, LLMToolClassifier)
    assert classifier.classifier == TOOL_EFFECT_CLASSIFIER_NAME
    assert classifier.classifier_model == "crof/qwen3.5-9b"
    assert classifier.classifier_cache_model == "qwen3.5-9b"


def test_app_runtime_builds_tool_call_classifier_for_routed_model() -> None:
    settings = _settings(
        llm_lightning_api_key="lightning-key",
        llm_openrouter_api_key="openrouter-key",
    )
    client = _create_chat_completion_client(settings)

    classifier = _create_tool_call_classifier(settings, client)

    assert isinstance(classifier, LLMToolCallClassifier)
    assert classifier.classifier == TOOL_CALL_EFFECT_CLASSIFIER_NAME
    assert classifier.classifier_model == settings.tool_call_effect_classifier_model
    assert classifier.classifier_cache_model == settings.tool_call_effect_classifier_cache_model


def test_app_runtime_builds_tool_call_classifier_for_configured_model() -> None:
    settings = _settings(
        llm_gmicloud_api_key="gmicloud-key",
        tool_call_effect_classifier_model="gmicloud/XiaomiMiMo/MiMo-V2.5-Pro",
        tool_call_effect_classifier_cache_model="mimo-v2.5-pro",
    )
    client = _create_chat_completion_client(settings)

    classifier = _create_tool_call_classifier(settings, client)

    assert isinstance(classifier, LLMToolCallClassifier)
    assert classifier.classifier == TOOL_CALL_EFFECT_CLASSIFIER_NAME
    assert classifier.classifier_model == "gmicloud/XiaomiMiMo/MiMo-V2.5-Pro"
    assert classifier.classifier_cache_model == "mimo-v2.5-pro"


def test_app_runtime_rejects_unrouted_tool_call_classifier_route() -> None:
    client = _create_chat_completion_client(_settings())

    with pytest.raises(PlapError) as exc_info:
        _create_tool_call_classifier(_settings(), client)

    assert exc_info.value.public is None
    assert exc_info.value.private.reason == "tool_call_classifier_route_unconfigured"


def test_app_runtime_omits_mcp_provider_without_config() -> None:
    assert _create_mcp_tool_providers(_settings()) == ()


def test_app_runtime_builds_mcp_providers_from_config_list() -> None:
    providers = _create_mcp_tool_providers(
        _settings(
            mcp_servers=[
                MCPServerConfig(
                    name="search",
                    config={
                        "command": "search-server",
                        "args": ["--stdio"],
                    },
                    tools={"search_web": MCPToolConfig(type="web_search")},
                )
            ],
        )
    )

    assert len(providers) == 1
    assert isinstance(providers[0], MCPToolProvider)


def test_app_runtime_merges_top_level_mcp_env_into_stdio_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLAP_TEST_REMOTE_TOKEN", "secret-token")

    providers = _create_mcp_tool_providers(
        _settings(
            mcp_servers=[
                MCPServerConfig(
                    name="search",
                    config={
                        "command": "search-server",
                        "args": ["--token=${MCP_TOKEN}"],
                        "env": {"MODE": "test"},
                    },
                    env={"MCP_TOKEN": "${PLAP_TEST_REMOTE_TOKEN}"},
                    tools={"search_web": MCPToolConfig(type="web_search")},
                )
            ],
        )
    )

    assert len(providers) == 1
    assert isinstance(providers[0], MCPToolProvider)
    assert providers[0]._transport == {
        "mcpServers": {
            "search": {
                "command": "search-server",
                "args": ["--token=secret-token"],
                "env": {
                    "MCP_TOKEN": "secret-token",
                    "MODE": "test",
                },
            }
        }
    }


def test_app_runtime_builds_mcp_provider_from_url() -> None:
    providers = _create_mcp_tool_providers(
        _settings(mcp_servers=[MCPServerConfig(name="remote", config={"url": "http://localhost:8765/mcp"})])
    )

    assert len(providers) == 1
    assert isinstance(providers[0], MCPToolProvider)


def test_app_runtime_builds_mcp_provider_from_url_with_headers_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLAP_TEST_JINA_TOKEN", "jina-secret")

    providers = _create_mcp_tool_providers(
        _settings(
            mcp_servers=[
                MCPServerConfig(
                    name="jina",
                    config={
                        "url": "https://mcp.jina.ai/v1?include_tools=${JINA_TOOLS}",
                        "headers": {"Authorization": "Bearer ${JINA_TOKEN}"},
                    },
                    env={
                        "JINA_TOKEN": "${PLAP_TEST_JINA_TOKEN}",
                        "JINA_TOOLS": "read_url,search_web",
                    },
                    tools={"search_web": MCPToolConfig(type="web_search")},
                )
            ],
        )
    )

    assert len(providers) == 1
    assert isinstance(providers[0], MCPToolProvider)
    assert providers[0]._transport == {
        "mcpServers": {
            "jina": {
                "url": "https://mcp.jina.ai/v1?include_tools=read_url,search_web",
                "headers": {"Authorization": "Bearer jina-secret"},
            }
        }
    }


def test_app_runtime_rejects_invalid_mcp_provider_config() -> None:
    with pytest.raises(PlapError) as exc_info:
        _create_mcp_tool_providers(
            _settings(
                mcp_servers=[
                    MCPServerConfig(
                        name="remote",
                        config={"url": "https://example.com/${MISSING_TOKEN}"},
                    )
                ],
            )
        )

    assert exc_info.value.public is None
    assert exc_info.value.private.reason == "mcp_transport_invalid"


def test_app_runtime_rejects_nested_mcp_servers_config() -> None:
    with pytest.raises(ValueError, match="single server"):
        MCPServerConfig(
            name="remote",
            config={
                "mcpServers": {
                    "remote": {
                        "url": "https://example.com/mcp",
                    }
                }
            },
        )


def test_app_runtime_rejects_unknown_mcp_tool_effect_class() -> None:
    with pytest.raises(ValueError, match="effect_class"):
        MCPToolConfig(type="web_search", effect_class="unknown")


def test_app_runtime_includes_builtin_jina_provider_when_api_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JINA_API_KEY", "builtin-jina-key")

    settings = Settings(
        api_key_pepper="pepper",
        database_url="postgresql+asyncpg://example/test",
        sealing_keys=["a" * 43],
    )
    providers = _create_mcp_tool_providers(settings)

    assert len(providers) == 1
    assert isinstance(providers[0], MCPToolProvider)
    assert set(providers[0].tool_configs) == {
        "read_url",
        "search_web",
        "search_arxiv",
        "search_ssrn",
        "search_bibtex",
    }
    assert providers[0].tool_configs["search_web"].argument_adapter == "web_search_user_location"
    assert providers[0].tool_configs["read_url"].argument_adapter is None
    assert providers[0]._transport == {
        "mcpServers": {
            "jina": {
                "url": (
                    "https://mcp.jina.ai/v1?include_tools="
                    "read_url,search_web,search_arxiv,search_ssrn,search_bibtex"
                ),
                "headers": {"Authorization": "Bearer builtin-jina-key"},
            }
        }
    }


def test_app_runtime_validates_synthetic_model_profiles() -> None:
    settings = _settings(
        llm_lightning_api_key="lightning-key",
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="lightning/lightning-ai/gpt-oss-20b",
                defender_model="lightning/lightning-ai/gpt-oss-120b",
                reviewer_model="lightning/lightning-ai/gpt-oss-20b",
                arbitrator_model="lightning/lightning-ai/gpt-oss-120b",
                reasoning_summarizer_model="lightning/lightning-ai/gpt-oss-20b",
            )
        },
    )

    _validate_runtime_model_profiles(settings)

    assert "plap/standard" in settings.runtime_model_profiles


def test_app_runtime_validates_runtime_profile_fallback_chain() -> None:
    settings = _settings(
        llm_crof_api_key="crof-key",
        llm_gmicloud_api_key="gmicloud-key",
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="crof/qwen3.5-9b,gmicloud/XiaomiMiMo/MiMo-V2.5-Pro",
                defender_model="crof/qwen3.5-9b",
                reviewer_model="gmicloud/XiaomiMiMo/MiMo-V2.5-Pro",
                arbitrator_model="crof/qwen3.5-9b",
                reasoning_summarizer_model="crof/qwen3.5-9b",
            )
        },
    )

    _validate_runtime_model_profiles(settings)


def test_app_runtime_rejects_runtime_profile_with_unrouted_model() -> None:
    settings = _settings(
        llm_lightning_api_key="lightning-key",
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="lightning/lightning-ai/gpt-oss-20b",
                defender_model="openai/gpt-oss-120b",
                reviewer_model="lightning/lightning-ai/gpt-oss-20b",
                arbitrator_model="lightning/lightning-ai/gpt-oss-120b",
                reasoning_summarizer_model="lightning/lightning-ai/gpt-oss-20b",
            )
        },
    )

    with pytest.raises(PlapError) as exc_info:
        _validate_runtime_model_profiles(settings)

    assert exc_info.value.public is None
    assert exc_info.value.private.reason == "runtime_profile_route_unconfigured"


def test_app_runtime_rejects_unrouted_runtime_profile_fallback_entry() -> None:
    settings = _settings(
        llm_crof_api_key="crof-key",
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="crof/qwen3.5-9b,lightning/lightning-ai/gpt-oss-20b",
                defender_model="crof/qwen3.5-9b",
                reviewer_model="crof/qwen3.5-9b",
                arbitrator_model="crof/qwen3.5-9b",
                reasoning_summarizer_model="crof/qwen3.5-9b",
            )
        },
    )

    with pytest.raises(PlapError) as exc_info:
        _validate_runtime_model_profiles(settings)

    assert exc_info.value.public is None
    assert exc_info.value.private.reason == "runtime_profile_route_unconfigured"


def test_app_runtime_validates_runtime_profile_variants() -> None:
    settings = _settings(
        llm_crof_api_key="crof-key",
        llm_lightning_api_key="lightning-key",
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="crof/qwen3.5-9b",
                defender_model="crof/qwen3.5-9b",
                reviewer_model="crof/qwen3.5-9b",
                arbitrator_model="crof/qwen3.5-9b",
                reasoning_summarizer_model="crof/qwen3.5-9b",
                by_service_tier={
                    "priority": RuntimeProfileOverride(
                        main=RuntimeActorOverride(model="lightning/lightning-ai/gpt-oss-120b"),
                        reviewer=RuntimeActorOverride(model="lightning/lightning-ai/gpt-oss-120b"),
                        reviewer_max_transcript_tokens=4096,
                        arbitrator_max_transcript_tokens=2048,
                        compact_threshold=8192,
                        compact_max_rounds=1,
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
                defender_model="crof/qwen3.5-9b",
                reviewer_model="crof/qwen3.5-9b",
                arbitrator_model="crof/qwen3.5-9b",
                reasoning_summarizer_model="crof/qwen3.5-9b",
                by_service_tier={
                    "priority": RuntimeProfileOverride(main=RuntimeActorOverride(model="lightning/lightning-ai/gpt-oss-120b"))
                },
            )
        },
    )

    with pytest.raises(PlapError) as exc_info:
        _validate_runtime_model_profiles(settings)

    assert exc_info.value.public is None
    assert exc_info.value.private.reason == "runtime_profile_route_unconfigured"


def test_app_runtime_resolves_only_explicit_synthetic_models() -> None:
    settings = _settings(
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="lightning/lightning-ai/gpt-oss-20b",
                defender_model="lightning/lightning-ai/gpt-oss-120b",
                reviewer_model="lightning/lightning-ai/gpt-oss-20b",
                arbitrator_model="lightning/lightning-ai/gpt-oss-120b",
                reasoning_summarizer_model="lightning/lightning-ai/gpt-oss-20b",
                reviewer_max_transcript_tokens=1024,
                arbitrator_max_transcript_tokens=768,
            )
        },
    )

    profile = settings.resolve_runtime_model_profile("plap/standard")

    assert profile is settings.runtime_model_profiles["plap/standard"]
    assert profile.reviewer_max_transcript_tokens == 1024
    assert profile.arbitrator_max_transcript_tokens == 768
    assert settings.resolve_runtime_model_profile("plap/standard", selector=RuntimeSelector(service_tier="default")) is profile
    assert settings.resolve_runtime_model_profile("plap/standard", selector=RuntimeSelector(service_tier="auto")) is profile
    with pytest.raises(PlapError) as exc_info:
        settings.resolve_runtime_model_profile(None)

    assert exc_info.value.public is not None
    assert exc_info.value.public.code == "missing_required_parameter"
    assert exc_info.value.public.param == "model"

    with pytest.raises(PlapError) as exc_info:
        settings.resolve_runtime_model_profile("lightning/lightning-ai/gpt-oss-20b")

    assert exc_info.value.public is not None
    assert exc_info.value.public.code == "model_not_found"
    assert exc_info.value.public.param == "model"


def test_app_runtime_rejects_unsupported_service_tier() -> None:
    settings = _settings(
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="crof/qwen3.5-9b",
                defender_model="crof/qwen3.5-9b",
                reviewer_model="crof/qwen3.5-9b",
                arbitrator_model="crof/qwen3.5-9b",
                reasoning_summarizer_model="crof/qwen3.5-9b",
                supported_parameters=["tools", "response_format"],
            )
        },
    )

    with pytest.raises(PlapError) as exc_info:
        settings.resolve_runtime_model_profile("plap/standard", selector=RuntimeSelector(service_tier="priority"))

    assert exc_info.value.public is not None
    assert exc_info.value.public.code == "unsupported_service_tier"
    assert exc_info.value.public.param == "service_tier"


def test_app_runtime_rejects_unsupported_reasoning_effort() -> None:
    settings = _settings(
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="crof/qwen3.5-9b",
                defender_model="crof/qwen3.5-9b",
                reviewer_model="crof/qwen3.5-9b",
                arbitrator_model="crof/qwen3.5-9b",
                reasoning_summarizer_model="crof/qwen3.5-9b",
                supported_parameters=["tools", "response_format"],
            )
        },
    )

    with pytest.raises(PlapError) as exc_info:
        settings.resolve_runtime_model_profile("plap/standard", selector=RuntimeSelector(reasoning_effort="high"))

    assert exc_info.value.public is not None
    assert exc_info.value.public.code == "unsupported_reasoning_effort"
    assert exc_info.value.public.param == "reasoning.effort"


def test_runtime_profile_rejects_service_tier_overrides_without_supported_parameter() -> None:
    with pytest.raises(ValueError, match="service_tier overrides require"):
        _profile_config(
            main_model="crof/qwen3.5-9b",
            defender_model="crof/qwen3.5-9b",
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
            defender_model="crof/qwen3.5-9b",
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
                defender_model="crof/qwen3.5-9b",
                reviewer_model="crof/qwen3.5-9b",
                arbitrator_model="crof/qwen3.5-9b",
                reasoning_summarizer_model="crof/qwen3.5-9b",
                compact_threshold=3000,
                compact_max_rounds=2,
                debate_max_rounds=2,
                by_service_tier={
                    "priority": RuntimeProfileOverride(
                        main=RuntimeActorOverride(model="lightning/lightning-ai/gpt-oss-120b"),
                        reviewer=RuntimeActorOverride(model="lightning/lightning-ai/gpt-oss-120b"),
                        reviewer_max_transcript_tokens=8192,
                        arbitrator_max_transcript_tokens=4096,
                        compact_threshold=6000,
                        compact_max_rounds=1,
                        debate_max_rounds=1,
                    )
                },
            )
        },
    )

    base = settings.resolve_runtime_model_profile("plap/standard")
    priority = settings.resolve_runtime_model_profile("plap/standard", selector=RuntimeSelector(service_tier="priority"))

    assert priority is not base
    assert priority.main.model == "lightning/lightning-ai/gpt-oss-120b"
    assert priority.reviewer.model == "lightning/lightning-ai/gpt-oss-120b"
    assert priority.compactor.model == "crof/qwen3.5-9b"
    assert priority.defender.model == "crof/qwen3.5-9b"
    assert priority.arbitrator.model == "crof/qwen3.5-9b"
    assert priority.reviewer_max_transcript_tokens == 8192
    assert priority.arbitrator_max_transcript_tokens == 4096
    assert priority.compact_threshold == 6000
    assert priority.compact_max_rounds == 1
    assert priority.debate_max_rounds == 1
    assert base.compact_threshold == 3000
    assert base.compact_max_rounds == 2
    assert base.debate_max_rounds == 2
    with pytest.raises(PlapError) as exc_info:
        settings.resolve_runtime_model_profile("plap/standard", selector=RuntimeSelector(service_tier="flex"))

    assert exc_info.value.public is not None
    assert exc_info.value.public.code == "unsupported_service_tier"
    assert exc_info.value.public.param == "service_tier"


def test_app_runtime_resolves_reasoning_effort_variant() -> None:
    settings = _settings(
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="crof/qwen3.5-9b",
                defender_model="crof/qwen3.5-9b",
                reviewer_model="crof/qwen3.5-9b",
                arbitrator_model="crof/qwen3.5-9b",
                reasoning_summarizer_model="crof/qwen3.5-9b",
                by_reasoning_effort={
                    "high": RuntimeProfileOverride(
                        main=RuntimeActorOverride(model="lightning/lightning-ai/gpt-oss-120b"),
                        defender=RuntimeActorOverride(model="lightning/lightning-ai/gpt-oss-120b"),
                    )
                },
            )
        },
    )

    profile = settings.resolve_runtime_model_profile("plap/standard", selector=RuntimeSelector(reasoning_effort="high"))

    assert profile.main.model == "lightning/lightning-ai/gpt-oss-120b"
    assert profile.defender.model == "lightning/lightning-ai/gpt-oss-120b"
    assert profile.reviewer.model == "crof/qwen3.5-9b"


def test_app_runtime_resolves_default_reasoning_effort_variant() -> None:
    settings = _settings(
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="crof/qwen3.5-9b",
                defender_model="crof/qwen3.5-9b",
                reviewer_model="crof/qwen3.5-9b",
                arbitrator_model="crof/qwen3.5-9b",
                reasoning_summarizer_model="crof/qwen3.5-9b",
                default_reasoning_effort="medium",
                by_reasoning_effort={
                    "medium": RuntimeProfileOverride(main=RuntimeActorOverride(model="lightning/lightning-ai/gpt-oss-120b")),
                    "high": RuntimeProfileOverride(main=RuntimeActorOverride(model="gmicloud/XiaomiMiMo/MiMo-V2.5-Pro")),
                },
            )
        },
    )

    default_profile = settings.resolve_runtime_model_profile("plap/standard")
    high_profile = settings.resolve_runtime_model_profile("plap/standard", selector=RuntimeSelector(reasoning_effort="high"))

    assert default_profile.main.model == "lightning/lightning-ai/gpt-oss-120b"
    assert high_profile.main.model == "gmicloud/XiaomiMiMo/MiMo-V2.5-Pro"


def test_app_runtime_rejects_missing_reasoning_effort_variant() -> None:
    settings = _settings(
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="crof/qwen3.5-9b",
                defender_model="crof/qwen3.5-9b",
                reviewer_model="crof/qwen3.5-9b",
                arbitrator_model="crof/qwen3.5-9b",
                reasoning_summarizer_model="crof/qwen3.5-9b",
                default_reasoning_effort="medium",
                by_reasoning_effort={
                    "medium": RuntimeProfileOverride(main=RuntimeActorOverride(model="lightning/lightning-ai/gpt-oss-120b"))
                },
            )
        },
    )

    with pytest.raises(PlapError) as exc_info:
        settings.resolve_runtime_model_profile("plap/standard", selector=RuntimeSelector(reasoning_effort="high"))

    assert exc_info.value.public is not None
    assert exc_info.value.public.code == "unsupported_reasoning_effort"
    assert exc_info.value.public.param == "reasoning.effort"


def test_runtime_profile_rejects_default_reasoning_effort_without_matching_override() -> None:
    with pytest.raises(ValueError, match="default_reasoning_effort"):
        _profile_config(
            main_model="crof/qwen3.5-9b",
            defender_model="crof/qwen3.5-9b",
            reviewer_model="crof/qwen3.5-9b",
            arbitrator_model="crof/qwen3.5-9b",
            reasoning_summarizer_model="crof/qwen3.5-9b",
            default_reasoning_effort="medium",
        )


def test_app_runtime_resolves_model_info_overrides() -> None:
    settings = _settings(
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="crof/qwen3.5-9b",
                defender_model="crof/qwen3.5-9b",
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

    priority = settings.resolve_runtime_model_profile("plap/standard", selector=RuntimeSelector(service_tier="priority"))

    assert priority.model_info.max_input_tokens == 4096


def test_app_runtime_resolves_reasoning_to_output_override() -> None:
    settings = _settings(
        runtime_model_profiles={
            "plap/standard": _profile_config(
                main_model="crof/qwen3.5-9b",
                defender_model="crof/qwen3.5-9b",
                reviewer_model="crof/qwen3.5-9b",
                arbitrator_model="crof/qwen3.5-9b",
                reasoning_summarizer_model="crof/qwen3.5-9b",
                by_service_tier={"priority": RuntimeProfileOverride(reasoning_to_output=1.5)},
            )
        }
    )

    base = settings.resolve_runtime_model_profile("plap/standard")
    priority = settings.resolve_runtime_model_profile("plap/standard", selector=RuntimeSelector(service_tier="priority"))

    assert base.reasoning_to_output == 1.0
    assert priority.reasoning_to_output == 1.5


def test_runtime_profile_rejects_conflicting_service_and_reasoning_overrides() -> None:
    with pytest.raises(ValueError, match="both set"):
        _profile_config(
            main_model="crof/qwen3.5-9b",
            defender_model="crof/qwen3.5-9b",
            reviewer_model="crof/qwen3.5-9b",
            arbitrator_model="crof/qwen3.5-9b",
            reasoning_summarizer_model="crof/qwen3.5-9b",
            by_service_tier={"priority": RuntimeProfileOverride(main=RuntimeActorOverride(model="lightning/lightning-ai/gpt-oss-120b"))},
            by_reasoning_effort={"high": RuntimeProfileOverride(main=RuntimeActorOverride(model="crof/glm-4.7-flash"))},
        )


def test_runtime_profile_rejects_conflicting_reasoning_to_output_overrides() -> None:
    with pytest.raises(ValueError, match="reasoning_to_output"):
        _profile_config(
            main_model="crof/qwen3.5-9b",
            defender_model="crof/qwen3.5-9b",
            reviewer_model="crof/qwen3.5-9b",
            arbitrator_model="crof/qwen3.5-9b",
            reasoning_summarizer_model="crof/qwen3.5-9b",
            by_service_tier={"priority": RuntimeProfileOverride(reasoning_to_output=1.2)},
            by_reasoning_effort={"high": RuntimeProfileOverride(reasoning_to_output=1.4)},
        )


def _settings(**overrides: object) -> Settings:
    values = {
        "api_key_pepper": "pepper",
        "database_url": "postgresql+asyncpg://example/test",
        "mcp_servers": [],
        "sealing_keys": ["a" * 43],
    }
    values.update(overrides)
    return Settings(**values)


def _profile_config(
    *,
    display_name: str = "Test Model",
    main_model: str,
    compactor_model: str | None = None,
    defender_model: str,
    reviewer_model: str,
    arbitrator_model: str,
    reasoning_summarizer_model: str,
    reviewer_max_transcript_tokens: int = 0,
    arbitrator_max_transcript_tokens: int = 0,
    compact_threshold: int | None = None,
    compact_max_rounds: int = 3,
    debate_max_rounds: int = 2,
    default_reasoning_effort: str | None = None,
    by_service_tier: dict[str, RuntimeProfileOverride] | None = None,
    by_reasoning_effort: dict[str, RuntimeProfileOverride] | None = None,
    supported_parameters: list[str] | None = None,
    reasoning_to_output: float = 1.0,
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
        compactor=RuntimeActorConfig(model=compactor_model or main_model),
        defender=RuntimeActorConfig(model=defender_model),
        reviewer=RuntimeActorConfig(model=reviewer_model),
        arbitrator=RuntimeActorConfig(model=arbitrator_model),
        reasoning_summarizer=RuntimeActorConfig(model=reasoning_summarizer_model),
        reviewer_max_transcript_tokens=reviewer_max_transcript_tokens,
        arbitrator_max_transcript_tokens=arbitrator_max_transcript_tokens,
        compact_threshold=compact_threshold,
        compact_max_rounds=compact_max_rounds,
        debate_max_rounds=debate_max_rounds,
        default_reasoning_effort=default_reasoning_effort,
        by_service_tier=by_service_tier or {},
        by_reasoning_effort=by_reasoning_effort or {},
        reasoning_to_output=reasoning_to_output,
    )
