from __future__ import annotations

from functools import lru_cache
from math import ceil, floor
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from plap.llms.chat import ReasoningEffort, ServiceTier
from plap.responses.contracts import ModelInfoObject, ModelInfoPricingObject, ModelObject


def _default_runtime_model_profiles() -> dict[str, RuntimeModelProfileConfig]:
    return {
        "plap-ai/wisp-nano": RuntimeModelProfileConfig(
            display_name="Wisp Nano",
            model_info=RuntimeModelInfoConfig(
                display_name="Wisp Nano",
                description="General-purpose plap responses model for text and tool use.",
                mode="responses",
                input_modalities=["text"],
                output_modalities=["text"],
                max_input_tokens=200_000,
                max_output_tokens=32_768,
                supported_parameters=[
                    "context_management",
                    "temperature",
                    "top_p",
                    "tools",
                    "tool_choice",
                    "parallel_tool_calls",
                    "response_format",
                    "max_output_tokens",
                    "reasoning_effort",
                    "stream",
                ],
                pricing=RuntimeModelPricingConfig(
                    input_per_token=0.0,
                    output_per_token=0.0,
                ),
                provider="plap",
                deprecated=False,
            ),
            main=RuntimeActorConfig(model="crof/qwen3.5-9b"),
            main_debate=RuntimeActorConfig(model="crof/qwen3.5-9b"),
            reviewer=RuntimeActorConfig(model="crof/qwen3.5-9b"),
            arbitrator=RuntimeActorConfig(model="crof/qwen3.5-9b"),
            reasoning_summarizer=RuntimeActorConfig(model="lightning/lightning-ai/gpt-oss-120b"),
            transcript_token_budget=200_000,
            compression_soft_token_budget=100_000,
            compression_hard_token_budget=150_000,
            compression_max_rounds=3,
            debate_max_rounds=2,
            by_reasoning_effort=_default_reasoning_effort_overrides(),
        )
    }


def _validate_compression_budgets(soft_budget: int | None, hard_budget: int | None) -> None:
    if soft_budget is not None and hard_budget is not None and hard_budget <= soft_budget:
        raise ValueError("compression hard token budget must exceed the soft token budget")


def _default_reasoning_effort_overrides() -> dict[ReasoningEffort, RuntimeProfileOverride]:
    return {
        effort: RuntimeProfileOverride(
            main=RuntimeActorOverride(reasoning_effort=effort),
            main_debate=RuntimeActorOverride(reasoning_effort=effort),
            reviewer=RuntimeActorOverride(reasoning_effort=effort),
            arbitrator=RuntimeActorOverride(reasoning_effort=effort),
        )
        for effort in ReasoningEffort
    }


class RuntimeSelector(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    service_tier: ServiceTier | None = None
    reasoning_effort: ReasoningEffort | None = None


class RuntimeModelPricingConfig(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    input_per_token: float = Field(ge=0)
    output_per_token: float = Field(ge=0)

    def to_contract(self) -> ModelInfoPricingObject:
        return ModelInfoPricingObject(
            input_per_token=self.input_per_token,
            output_per_token=self.output_per_token,
        )


class RuntimeModelPricingOverride(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    input_per_token: float | None = Field(default=None, ge=0)
    output_per_token: float | None = Field(default=None, ge=0)

    def apply_override(self, pricing: RuntimeModelPricingConfig) -> RuntimeModelPricingConfig:
        return pricing.model_copy(update=self.model_dump(exclude_none=True))

    def overridden_fields(self, prefix: str) -> set[str]:
        fields: set[str] = set()
        if self.input_per_token is not None:
            fields.add(f"{prefix}.input_per_token")
        if self.output_per_token is not None:
            fields.add(f"{prefix}.output_per_token")
        return fields


class RuntimeModelInfoConfig(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    display_name: str
    description: str
    mode: str
    input_modalities: list[str]
    output_modalities: list[str]
    max_input_tokens: int = Field(ge=1)
    max_output_tokens: int = Field(ge=1)
    supported_parameters: list[str]
    pricing: RuntimeModelPricingConfig
    provider: str
    deprecated: bool = False

    def supports_parameter(self, name: str) -> bool:
        return name in self.supported_parameters

    def apply_override(self, override: RuntimeModelInfoOverride | None) -> RuntimeModelInfoConfig:
        if override is None:
            return self
        updates = override.model_dump(exclude_none=True)
        pricing_override = override.pricing
        if pricing_override is not None:
            updates["pricing"] = pricing_override.apply_override(self.pricing)
        return self.model_copy(update=updates)

    def to_contract(self, *, model: str) -> ModelInfoObject:
        return ModelInfoObject(
            id=model,
            display_name=self.display_name,
            description=self.description,
            mode=self.mode,
            input_modalities=list(self.input_modalities),
            output_modalities=list(self.output_modalities),
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=self.max_output_tokens,
            supported_parameters=list(self.supported_parameters),
            pricing=self.pricing.to_contract(),
            provider=self.provider,
            deprecated=self.deprecated,
        )


class RuntimeModelInfoOverride(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    display_name: str | None = None
    description: str | None = None
    mode: str | None = None
    input_modalities: list[str] | None = None
    output_modalities: list[str] | None = None
    max_input_tokens: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    supported_parameters: list[str] | None = None
    pricing: RuntimeModelPricingOverride | None = None
    provider: str | None = None
    deprecated: bool | None = None

    def overridden_fields(self, prefix: str) -> set[str]:
        fields: set[str] = set()
        for name in (
            "display_name",
            "description",
            "mode",
            "input_modalities",
            "output_modalities",
            "max_input_tokens",
            "max_output_tokens",
            "supported_parameters",
            "provider",
            "deprecated",
        ):
            if getattr(self, name) is not None:
                fields.add(f"{prefix}.{name}")
        if self.pricing is not None:
            fields.update(self.pricing.overridden_fields(f"{prefix}.pricing"))
        return fields


class PublicUsageConfig(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    uncached_input_to_output: float = Field(default=0.25, ge=0)
    cached_input_to_output: float = Field(default=0.05, ge=0)
    output_to_output: float = Field(default=1.0, gt=0)

    def cap_from_budget(self, budget: int | None) -> int | None:
        if budget is None:
            return None
        if budget <= 0:
            return 0
        return floor(budget / self.output_to_output)

    def hidden_debit(self, usage) -> int:
        cached_input = min(usage.cached_tokens or 0, usage.input_tokens)
        uncached_input = usage.input_tokens - cached_input
        debit = (
            uncached_input * self.uncached_input_to_output
            + cached_input * self.cached_input_to_output
            + usage.output_tokens * self.output_to_output
        )
        return ceil(debit)


class RuntimeActorConfig(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    model: str
    reasoning_effort: ReasoningEffort | None = None
    service_tier: ServiceTier | None = None
    public_usage: PublicUsageConfig = Field(default_factory=PublicUsageConfig)

    def apply_override(self, override: RuntimeActorOverride | None) -> RuntimeActorConfig:
        if override is None:
            return self
        return self.model_copy(update=override.model_dump(exclude_none=True))


class RuntimeActorOverride(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    model: str | None = None
    reasoning_effort: ReasoningEffort | None = None
    service_tier: ServiceTier | None = None

    def overridden_fields(self, prefix: str) -> set[str]:
        fields: set[str] = set()
        if self.model is not None:
            fields.add(f"{prefix}.model")
        if self.reasoning_effort is not None:
            fields.add(f"{prefix}.reasoning_effort")
        if self.service_tier is not None:
            fields.add(f"{prefix}.service_tier")
        return fields


class RuntimeProfileOverride(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    display_name: str | None = None
    model_info: RuntimeModelInfoOverride | None = None
    main: RuntimeActorOverride | None = None
    main_debate: RuntimeActorOverride | None = None
    reviewer: RuntimeActorOverride | None = None
    arbitrator: RuntimeActorOverride | None = None
    reasoning_summarizer: RuntimeActorOverride | None = None
    transcript_token_budget: int | None = Field(default=None, ge=0)
    compression_soft_token_budget: int | None = Field(default=None, ge=0)
    compression_hard_token_budget: int | None = Field(default=None, ge=0)
    compression_max_rounds: int | None = Field(default=None, ge=0)
    debate_max_rounds: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_compression_config(self) -> RuntimeProfileOverride:
        _validate_compression_budgets(self.compression_soft_token_budget, self.compression_hard_token_budget)
        return self

    def apply_to(self, profile: RuntimeModelProfileConfig) -> RuntimeModelProfileConfig:
        updates: dict[str, object] = {}
        if self.display_name is not None:
            updates["display_name"] = self.display_name
        if self.model_info is not None:
            updates["model_info"] = profile.model_info.apply_override(self.model_info)
        if self.main is not None:
            updates["main"] = profile.main.apply_override(self.main)
        if self.main_debate is not None:
            updates["main_debate"] = profile.main_debate.apply_override(self.main_debate)
        if self.reviewer is not None:
            updates["reviewer"] = profile.reviewer.apply_override(self.reviewer)
        if self.arbitrator is not None:
            updates["arbitrator"] = profile.arbitrator.apply_override(self.arbitrator)
        if self.reasoning_summarizer is not None:
            updates["reasoning_summarizer"] = profile.reasoning_summarizer.apply_override(self.reasoning_summarizer)
        if self.transcript_token_budget is not None:
            updates["transcript_token_budget"] = self.transcript_token_budget
        if self.compression_soft_token_budget is not None:
            updates["compression_soft_token_budget"] = self.compression_soft_token_budget
        if self.compression_hard_token_budget is not None:
            updates["compression_hard_token_budget"] = self.compression_hard_token_budget
        if self.compression_max_rounds is not None:
            updates["compression_max_rounds"] = self.compression_max_rounds
        if self.debate_max_rounds is not None:
            updates["debate_max_rounds"] = self.debate_max_rounds
        return profile.model_copy(update=updates)

    def overridden_fields(self) -> set[str]:
        fields: set[str] = set()
        if self.display_name is not None:
            fields.add("display_name")
        if self.model_info is not None:
            fields.update(self.model_info.overridden_fields("model_info"))
        if self.main is not None:
            fields.update(self.main.overridden_fields("main"))
        if self.main_debate is not None:
            fields.update(self.main_debate.overridden_fields("main_debate"))
        if self.reviewer is not None:
            fields.update(self.reviewer.overridden_fields("reviewer"))
        if self.arbitrator is not None:
            fields.update(self.arbitrator.overridden_fields("arbitrator"))
        if self.reasoning_summarizer is not None:
            fields.update(self.reasoning_summarizer.overridden_fields("reasoning_summarizer"))
        if self.transcript_token_budget is not None:
            fields.add("transcript_token_budget")
        if self.compression_soft_token_budget is not None:
            fields.add("compression_soft_token_budget")
        if self.compression_hard_token_budget is not None:
            fields.add("compression_hard_token_budget")
        if self.compression_max_rounds is not None:
            fields.add("compression_max_rounds")
        if self.debate_max_rounds is not None:
            fields.add("debate_max_rounds")
        return fields

    def conflicts_with(self, other: RuntimeProfileOverride) -> set[str]:
        return self.overridden_fields() & other.overridden_fields()

    def all_models(self) -> tuple[str, ...]:
        return tuple(
            actor.model
            for actor in (self.main, self.main_debate, self.reviewer, self.arbitrator, self.reasoning_summarizer)
            if actor is not None and actor.model is not None
        )


class RuntimeModelProfileConfig(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    display_name: str
    model_info: RuntimeModelInfoConfig
    main: RuntimeActorConfig
    main_debate: RuntimeActorConfig
    reviewer: RuntimeActorConfig
    arbitrator: RuntimeActorConfig
    reasoning_summarizer: RuntimeActorConfig
    transcript_token_budget: int = Field(default=0, ge=0)
    compression_soft_token_budget: int | None = Field(default=None, ge=0)
    compression_hard_token_budget: int | None = Field(default=None, ge=0)
    compression_max_rounds: int = Field(default=3, ge=0)
    debate_max_rounds: int = Field(default=2, ge=0)
    by_service_tier: dict[ServiceTier, RuntimeProfileOverride] = Field(default_factory=dict)
    by_reasoning_effort: dict[ReasoningEffort, RuntimeProfileOverride] = Field(default_factory=dict)

    @field_validator("by_service_tier")
    @classmethod
    def validate_service_tier_overrides(cls, value: dict[ServiceTier, RuntimeProfileOverride]) -> dict[ServiceTier, RuntimeProfileOverride]:
        for key in value:
            if not isinstance(key, str) or not key:
                raise ValueError("service tier override keys must be non-empty strings")
        return value

    @model_validator(mode="after")
    def validate_profile(self) -> RuntimeModelProfileConfig:
        _validate_compression_budgets(self.compression_soft_token_budget, self.compression_hard_token_budget)

        if self.by_service_tier and not self.supports_parameter("service_tier"):
            raise ValueError("service_tier overrides require model_info.supported_parameters to include service_tier")
        if self.by_reasoning_effort and not self.supports_parameter("reasoning_effort"):
            raise ValueError(
                "reasoning_effort overrides require model_info.supported_parameters to include reasoning_effort"
            )

        for override in self.by_service_tier.values():
            resolved = override.apply_to(self)
            _validate_compression_budgets(resolved.compression_soft_token_budget, resolved.compression_hard_token_budget)

        for override in self.by_reasoning_effort.values():
            resolved = override.apply_to(self)
            _validate_compression_budgets(resolved.compression_soft_token_budget, resolved.compression_hard_token_budget)

        for service_override in self.by_service_tier.values():
            for reasoning_override in self.by_reasoning_effort.values():
                conflicts = service_override.conflicts_with(reasoning_override)
                if conflicts:
                    fields = ", ".join(sorted(conflicts))
                    raise ValueError(f"service_tier and reasoning_effort overrides both set: {fields}")
                resolved = reasoning_override.apply_to(service_override.apply_to(self))
                _validate_compression_budgets(resolved.compression_soft_token_budget, resolved.compression_hard_token_budget)

        return self

    def supports_parameter(self, name: str) -> bool:
        return self.model_info.supports_parameter(name)

    def validate_requested_parameters(self, parameters: set[str]) -> None:
        unsupported = sorted(parameter for parameter in parameters if not self.supports_parameter(parameter))
        if unsupported:
            joined = ", ".join(unsupported)
            raise ValueError(f"unsupported request parameters: {joined}")

    def resolve(self, selector: RuntimeSelector | None = None) -> RuntimeModelProfileConfig:
        resolved = self
        selector = selector or RuntimeSelector()

        if selector.service_tier not in {None, "default", "auto"}:
            if not self.supports_parameter("service_tier"):
                raise ValueError("unsupported request parameters: service_tier")
            override = self.by_service_tier.get(selector.service_tier)
            if override is not None:
                resolved = override.apply_to(resolved)

        if selector.reasoning_effort is not None:
            if not self.supports_parameter("reasoning_effort"):
                raise ValueError("unsupported request parameters: reasoning_effort")
            override = self.by_reasoning_effort.get(selector.reasoning_effort)
            if override is not None:
                resolved = override.apply_to(resolved)

        _validate_compression_budgets(resolved.compression_soft_token_budget, resolved.compression_hard_token_budget)
        return resolved

    def all_models(self) -> tuple[str, ...]:
        models = [
            self.main.model,
            self.main_debate.model,
            self.reviewer.model,
            self.arbitrator.model,
            self.reasoning_summarizer.model,
        ]
        for override in self.by_service_tier.values():
            models.extend(override.all_models())
        for override in self.by_reasoning_effort.values():
            models.extend(override.all_models())
        return tuple(model for model in models if "/" in model)

    def to_model_object(self, *, model: str) -> ModelObject:
        return ModelObject(
            id=model,
            created=0,
            owned_by="plap",
        )


class MCPServerConfig(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    name: str
    url: str | None = None
    config: dict[str, Any] | None = None
    tool_names: list[str] | None = None

    @model_validator(mode="after")
    def validate_transport(self) -> MCPServerConfig:
        if bool(self.url) == bool(self.config):
            raise ValueError("mcp server config requires exactly one of url or config")
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PLAP_", extra="ignore")

    api_key_pepper: str
    database_url: str
    sealing_keys: list[str]
    llm_lightning_api_key: str | None = None
    llm_novita_api_key: str | None = None
    llm_fireworks_api_key: str | None = None
    llm_crof_api_key: str | None = None
    tool_classifier_max_concurrency: int = 4
    tool_policy_l1_maxsize: int = 4096
    tool_call_policy_l1_maxsize: int = 4096
    runtime_model_profiles: dict[str, RuntimeModelProfileConfig] = Field(default_factory=_default_runtime_model_profiles)
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)

    @field_validator("sealing_keys", mode="before")
    @classmethod
    def split_sealing_keys(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    def resolve_runtime_model_profile(
        self,
        model: str | None,
        *,
        selector: RuntimeSelector | None = None,
    ) -> RuntimeModelProfileConfig:
        if model is None:
            raise ValueError("model is required")
        profile = self.runtime_model_profiles.get(model)
        if profile is None:
            raise ValueError(f"unknown runtime model: {model!r}")
        return profile.resolve(selector)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
