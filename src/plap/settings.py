from __future__ import annotations

from functools import lru_cache
from math import ceil, floor
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from plap.errors import ErrorLevel, PlapError, PrivateError, PublicError
from plap.llms.chat import ReasoningEffort, ServiceTier
from plap.responses.contracts import ModelInfoObject, ModelInfoPricingObject, ModelObject


def _default_runtime_model_profiles() -> dict[str, RuntimeModelProfileConfig]:
    return {
        "plap-ai/wisp-mini": RuntimeModelProfileConfig(
            display_name="Wisp Mini",
            model_info=RuntimeModelInfoConfig(
                display_name="Wisp Mini",
                description="General-purpose plap responses model for text and tool use.",
                mode="responses",
                input_modalities=["text"],
                output_modalities=["text"],
                max_input_tokens=262_144,
                max_output_tokens=262_144,
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
            main=RuntimeActorConfig(
                model="openrouter/stepfun/step-3.5-flash:nitro",
                tokenizer_hf_repo="stepfun-ai/Step-3.5-Flash",
                tokenizer_revision="ab446a3de5e171ea341227e24bb1f090e1b771f7",
            ),
            compactor=RuntimeActorConfig(
                model="openrouter/deepseek/deepseek-v4-flash:nitro",
                tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash",
                tokenizer_revision="6976c7ff1b30a1b2cb7805021b8ba4684041f136",
            ),
            main_debate=RuntimeActorConfig(
                model="openrouter/stepfun/step-3.5-flash:nitro",
                tokenizer_hf_repo="stepfun-ai/Step-3.5-Flash",
                tokenizer_revision="ab446a3de5e171ea341227e24bb1f090e1b771f7",
            ),
            reviewer=RuntimeActorConfig(
                model="openrouter/deepseek/deepseek-v4-flash:nitro",
                tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash",
                tokenizer_revision="6976c7ff1b30a1b2cb7805021b8ba4684041f136",
            ),
            arbitrator=RuntimeActorConfig(
                model="openrouter/deepseek/deepseek-v4-flash:nitro",
                tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash",
                tokenizer_revision="6976c7ff1b30a1b2cb7805021b8ba4684041f136",
            ),
            reasoning_summarizer=RuntimeActorConfig(model="lightning/lightning-ai/gpt-oss-120b"),
            reviewer_transcript_token_budget=800_000,
            arbitrator_transcript_token_budget=300_000,
            soft_compact_threshold=200_000,
            compact_threshold=250_000,
            compact_max_rounds=3,
            debate_max_rounds=2,
            default_reasoning_effort=ReasoningEffort.MEDIUM,
            by_reasoning_effort={
                ReasoningEffort.MINIMAL: RuntimeProfileOverride(
                    main=RuntimeActorOverride(reasoning_effort=ReasoningEffort.NONE),
                    main_debate=RuntimeActorOverride(reasoning_effort=ReasoningEffort.NONE),
                    reviewer=RuntimeActorOverride(reasoning_effort=ReasoningEffort.NONE),
                    arbitrator=RuntimeActorOverride(reasoning_effort=ReasoningEffort.NONE),
                ),
                ReasoningEffort.LOW: RuntimeProfileOverride(
                    main=RuntimeActorOverride(reasoning_effort=ReasoningEffort.LOW),
                    main_debate=RuntimeActorOverride(reasoning_effort=ReasoningEffort.LOW),
                    reviewer=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                    arbitrator=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                ),
                ReasoningEffort.MEDIUM: RuntimeProfileOverride(
                    main=RuntimeActorOverride(reasoning_effort=ReasoningEffort.MEDIUM),
                    main_debate=RuntimeActorOverride(reasoning_effort=ReasoningEffort.MEDIUM),
                    reviewer=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                    arbitrator=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                ),
                ReasoningEffort.HIGH: RuntimeProfileOverride(
                    main=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                    main_debate=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                    reviewer=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                    arbitrator=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                ),
                ReasoningEffort.XHIGH: RuntimeProfileOverride(
                    main=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                    main_debate=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                    reviewer=RuntimeActorOverride(reasoning_effort=ReasoningEffort.XHIGH),
                    arbitrator=RuntimeActorOverride(reasoning_effort=ReasoningEffort.XHIGH),
                ),
            },
        ),
        "plap-ai/wisp": RuntimeModelProfileConfig(
            display_name="Wisp",
            model_info=RuntimeModelInfoConfig(
                display_name="Wisp",
                description="Higher-quality plap responses model for text and tool use.",
                mode="responses",
                input_modalities=["text"],
                output_modalities=["text"],
                max_input_tokens=1000_000,
                max_output_tokens=1000_000,
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
            main=RuntimeActorConfig(
                model="crof/mimo-v2.5-pro",
                tokenizer_hf_repo="XiaomiMiMo/MiMo-V2.5-Pro",
                tokenizer_revision="a75207db63de3c320950fe6fcfa9ff60f341b7a2",
            ),
            compactor=RuntimeActorConfig(
                model="openrouter/deepseek/deepseek-v4-flash:nitro",
                tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash",
                tokenizer_revision="6976c7ff1b30a1b2cb7805021b8ba4684041f136",
            ),
            main_debate=RuntimeActorConfig(
                model="crof/mimo-v2.5-pro",
                tokenizer_hf_repo="XiaomiMiMo/MiMo-V2.5-Pro",
                tokenizer_revision="a75207db63de3c320950fe6fcfa9ff60f341b7a2",
            ),
            reviewer=RuntimeActorConfig(
                model="openrouter/deepseek/deepseek-v4-flash:nitro",
                tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash",
                tokenizer_revision="6976c7ff1b30a1b2cb7805021b8ba4684041f136",
            ),
            arbitrator=RuntimeActorConfig(
                model="openrouter/deepseek/deepseek-v4-flash:nitro",
                tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash",
                tokenizer_revision="6976c7ff1b30a1b2cb7805021b8ba4684041f136",
            ),
            reasoning_summarizer=RuntimeActorConfig(model="lightning/lightning-ai/gpt-oss-120b"),
            reviewer_transcript_token_budget=800_000,
            arbitrator_transcript_token_budget=300_000,
            soft_compact_threshold=200_000,
            compact_threshold=250_000,
            compact_max_rounds=3,
            debate_max_rounds=2,
            default_reasoning_effort=ReasoningEffort.MEDIUM,
            by_reasoning_effort={
                ReasoningEffort.MINIMAL: RuntimeProfileOverride(
                    main=RuntimeActorOverride(reasoning_effort=ReasoningEffort.NONE),
                    main_debate=RuntimeActorOverride(reasoning_effort=ReasoningEffort.NONE),
                    reviewer=RuntimeActorOverride(reasoning_effort=ReasoningEffort.NONE),
                    arbitrator=RuntimeActorOverride(reasoning_effort=ReasoningEffort.NONE),
                ),
                ReasoningEffort.LOW: RuntimeProfileOverride(
                    main=RuntimeActorOverride(reasoning_effort=ReasoningEffort.LOW),
                    main_debate=RuntimeActorOverride(reasoning_effort=ReasoningEffort.LOW),
                    reviewer=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                    arbitrator=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                ),
                ReasoningEffort.MEDIUM: RuntimeProfileOverride(
                    main=RuntimeActorOverride(reasoning_effort=ReasoningEffort.MEDIUM),
                    main_debate=RuntimeActorOverride(reasoning_effort=ReasoningEffort.MEDIUM),
                    reviewer=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                    arbitrator=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                ),
                ReasoningEffort.HIGH: RuntimeProfileOverride(
                    main=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                    main_debate=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                    reviewer=RuntimeActorOverride(reasoning_effort=ReasoningEffort.XHIGH),
                    arbitrator=RuntimeActorOverride(reasoning_effort=ReasoningEffort.XHIGH),
                ),
                ReasoningEffort.XHIGH: RuntimeProfileOverride(
                    main=RuntimeActorOverride(reasoning_effort=ReasoningEffort.XHIGH),
                    main_debate=RuntimeActorOverride(reasoning_effort=ReasoningEffort.XHIGH),
                    reviewer=RuntimeActorOverride(reasoning_effort=ReasoningEffort.XHIGH),
                    arbitrator=RuntimeActorOverride(reasoning_effort=ReasoningEffort.XHIGH),
                ),
            },
        ),
    }


def _validate_compact_thresholds(soft_threshold: int | None, threshold: int | None) -> None:
    if soft_threshold is not None and threshold is not None and threshold <= soft_threshold:
        raise ValueError("compact threshold must exceed the soft compact threshold")


def _missing_model_error() -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code="missing_required_parameter",
            message="Missing required parameter: 'model'.",
            param="model",
        ),
        private=PrivateError(
            event="runtime_profile.invalid_request",
            reason="missing_model",
            message="model is required",
            level=ErrorLevel.WARNING,
        ),
    )


def _unknown_runtime_model_error(model: str) -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=404,
            type="not_found_error",
            code="model_not_found",
            message=f"Model '{model}' not found.",
            param="model",
        ),
        private=PrivateError(
            event="runtime_profile.invalid_request",
            reason="unknown_runtime_model",
            message=f"unknown runtime model: {model!r}",
            level=ErrorLevel.WARNING,
            context={"model": model},
        ),
    )


def _unsupported_parameters_error(model: str, unsupported: list[str]) -> PlapError:
    if len(unsupported) == 1:
        parameter = unsupported[0]
        message = f"Parameter '{parameter}' is not supported for model '{model}'."
        param = parameter
    else:
        joined = ", ".join(f"'{parameter}'" for parameter in unsupported)
        message = f"Parameters {joined} are not supported for model '{model}'."
        param = None
    joined_private = ", ".join(unsupported)
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code="unsupported_parameter",
            message=message,
            param=param,
        ),
        private=PrivateError(
            event="runtime_profile.invalid_request",
            reason="unsupported_request_parameter",
            message=f"unsupported request parameters: {joined_private}",
            level=ErrorLevel.WARNING,
            context={"model": model, "parameters": unsupported},
        ),
    )


def _unsupported_service_tier_error(model: str, service_tier: ServiceTier, *, reason: str, private_message: str) -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code="unsupported_service_tier",
            message=f"Service tier '{service_tier}' is not supported for model '{model}'.",
            param="service_tier",
        ),
        private=PrivateError(
            event="runtime_profile.invalid_request",
            reason=reason,
            message=private_message,
            level=ErrorLevel.WARNING,
            context={"model": model, "service_tier": service_tier},
        ),
    )


def _unsupported_reasoning_effort_error(model: str, effort: ReasoningEffort, *, reason: str, private_message: str) -> PlapError:
    return PlapError(
        public=PublicError(
            status_code=400,
            type="invalid_request_error",
            code="unsupported_reasoning_effort",
            message=f"Reasoning effort '{effort}' is not supported for model '{model}'.",
            param="reasoning.effort",
        ),
        private=PrivateError(
            event="runtime_profile.invalid_request",
            reason=reason,
            message=private_message,
            level=ErrorLevel.WARNING,
            context={"model": model, "reasoning_effort": effort},
        ),
    )


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
    tokenizer_hf_repo: str | None = None
    tokenizer_revision: str | None = None
    tokenizer_trust_remote_code: bool = False
    reasoning_effort: ReasoningEffort | None = None
    service_tier: ServiceTier | None = None
    public_usage: PublicUsageConfig = Field(default_factory=PublicUsageConfig)

    @model_validator(mode="after")
    def validate_tokenizer(self) -> RuntimeActorConfig:
        if self.tokenizer_hf_repo is None:
            if self.tokenizer_revision is not None:
                raise ValueError("tokenizer_revision requires tokenizer_hf_repo")
            if self.tokenizer_trust_remote_code:
                raise ValueError("tokenizer_trust_remote_code requires tokenizer_hf_repo")
        return self

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
    compactor: RuntimeActorOverride | None = None
    main_debate: RuntimeActorOverride | None = None
    reviewer: RuntimeActorOverride | None = None
    arbitrator: RuntimeActorOverride | None = None
    reasoning_summarizer: RuntimeActorOverride | None = None
    reviewer_transcript_token_budget: int | None = Field(default=None, ge=0)
    arbitrator_transcript_token_budget: int | None = Field(default=None, ge=0)
    soft_compact_threshold: int | None = Field(default=None, ge=0)
    compact_threshold: int | None = Field(default=None, ge=0)
    compact_max_rounds: int | None = Field(default=None, ge=0)
    debate_max_rounds: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_compaction_config(self) -> RuntimeProfileOverride:
        _validate_compact_thresholds(self.soft_compact_threshold, self.compact_threshold)
        return self

    def apply_to(self, profile: RuntimeModelProfileConfig) -> RuntimeModelProfileConfig:
        updates: dict[str, object] = {}
        if self.display_name is not None:
            updates["display_name"] = self.display_name
        if self.model_info is not None:
            updates["model_info"] = profile.model_info.apply_override(self.model_info)
        if self.main is not None:
            updates["main"] = profile.main.apply_override(self.main)
        if self.compactor is not None:
            updates["compactor"] = profile.compactor.apply_override(self.compactor)
        if self.main_debate is not None:
            updates["main_debate"] = profile.main_debate.apply_override(self.main_debate)
        if self.reviewer is not None:
            updates["reviewer"] = profile.reviewer.apply_override(self.reviewer)
        if self.arbitrator is not None:
            updates["arbitrator"] = profile.arbitrator.apply_override(self.arbitrator)
        if self.reasoning_summarizer is not None:
            updates["reasoning_summarizer"] = profile.reasoning_summarizer.apply_override(self.reasoning_summarizer)
        if self.reviewer_transcript_token_budget is not None:
            updates["reviewer_transcript_token_budget"] = self.reviewer_transcript_token_budget
        if self.arbitrator_transcript_token_budget is not None:
            updates["arbitrator_transcript_token_budget"] = self.arbitrator_transcript_token_budget
        if self.soft_compact_threshold is not None:
            updates["soft_compact_threshold"] = self.soft_compact_threshold
        if self.compact_threshold is not None:
            updates["compact_threshold"] = self.compact_threshold
        if self.compact_max_rounds is not None:
            updates["compact_max_rounds"] = self.compact_max_rounds
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
        if self.compactor is not None:
            fields.update(self.compactor.overridden_fields("compactor"))
        if self.main_debate is not None:
            fields.update(self.main_debate.overridden_fields("main_debate"))
        if self.reviewer is not None:
            fields.update(self.reviewer.overridden_fields("reviewer"))
        if self.arbitrator is not None:
            fields.update(self.arbitrator.overridden_fields("arbitrator"))
        if self.reasoning_summarizer is not None:
            fields.update(self.reasoning_summarizer.overridden_fields("reasoning_summarizer"))
        if self.reviewer_transcript_token_budget is not None:
            fields.add("reviewer_transcript_token_budget")
        if self.arbitrator_transcript_token_budget is not None:
            fields.add("arbitrator_transcript_token_budget")
        if self.soft_compact_threshold is not None:
            fields.add("soft_compact_threshold")
        if self.compact_threshold is not None:
            fields.add("compact_threshold")
        if self.compact_max_rounds is not None:
            fields.add("compact_max_rounds")
        if self.debate_max_rounds is not None:
            fields.add("debate_max_rounds")
        return fields

    def conflicts_with(self, other: RuntimeProfileOverride) -> set[str]:
        return self.overridden_fields() & other.overridden_fields()

    def all_models(self) -> tuple[str, ...]:
        return tuple(
            actor.model
            for actor in (self.main, self.compactor, self.main_debate, self.reviewer, self.arbitrator, self.reasoning_summarizer)
            if actor is not None and actor.model is not None
        )


class RuntimeModelProfileConfig(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    display_name: str
    model_info: RuntimeModelInfoConfig
    main: RuntimeActorConfig
    compactor: RuntimeActorConfig
    main_debate: RuntimeActorConfig
    reviewer: RuntimeActorConfig
    arbitrator: RuntimeActorConfig
    reasoning_summarizer: RuntimeActorConfig
    reviewer_transcript_token_budget: int = Field(default=0, ge=0)
    arbitrator_transcript_token_budget: int = Field(default=0, ge=0)
    transcript_recount_margin: int = Field(default=16384, ge=0)
    soft_compact_threshold: int | None = Field(default=None, ge=0)
    compact_threshold: int | None = Field(default=None, ge=0)
    compact_max_rounds: int = Field(default=3, ge=0)
    debate_max_rounds: int = Field(default=2, ge=0)
    default_reasoning_effort: ReasoningEffort | None = None
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
        _validate_compact_thresholds(self.soft_compact_threshold, self.compact_threshold)

        if self.by_service_tier and not self.supports_parameter("service_tier"):
            raise ValueError("service_tier overrides require model_info.supported_parameters to include service_tier")
        if self.by_reasoning_effort and not self.supports_parameter("reasoning_effort"):
            raise ValueError("reasoning_effort overrides require model_info.supported_parameters to include reasoning_effort")
        if self.default_reasoning_effort is not None:
            if not self.supports_parameter("reasoning_effort"):
                raise ValueError("default_reasoning_effort requires model_info.supported_parameters to include reasoning_effort")
            if self.default_reasoning_effort not in self.by_reasoning_effort:
                raise ValueError("default_reasoning_effort must reference a configured reasoning_effort override")

        for override in self.by_service_tier.values():
            resolved = override.apply_to(self)
            _validate_compact_thresholds(resolved.soft_compact_threshold, resolved.compact_threshold)

        for override in self.by_reasoning_effort.values():
            resolved = override.apply_to(self)
            _validate_compact_thresholds(resolved.soft_compact_threshold, resolved.compact_threshold)

        for service_override in self.by_service_tier.values():
            for reasoning_override in self.by_reasoning_effort.values():
                conflicts = service_override.conflicts_with(reasoning_override)
                if conflicts:
                    fields = ", ".join(sorted(conflicts))
                    raise ValueError(f"service_tier and reasoning_effort overrides both set: {fields}")
                resolved = reasoning_override.apply_to(service_override.apply_to(self))
                _validate_compact_thresholds(resolved.soft_compact_threshold, resolved.compact_threshold)

        return self

    def supports_parameter(self, name: str) -> bool:
        return self.model_info.supports_parameter(name)

    def validate_requested_parameters(self, parameters: set[str], *, model: str) -> None:
        unsupported = sorted(parameter for parameter in parameters if not self.supports_parameter(parameter))
        if unsupported:
            raise _unsupported_parameters_error(model, unsupported)

    def resolve(self, selector: RuntimeSelector | None = None, *, model: str) -> RuntimeModelProfileConfig:
        resolved = self
        selector = selector or RuntimeSelector()

        if selector.service_tier not in {None, "default", "auto"}:
            if not self.supports_parameter("service_tier"):
                raise _unsupported_service_tier_error(
                    model,
                    selector.service_tier,
                    reason="unsupported_service_tier",
                    private_message="unsupported request parameters: service_tier",
                )
            override = self.by_service_tier.get(selector.service_tier)
            if override is None:
                raise _unsupported_service_tier_error(
                    model,
                    selector.service_tier,
                    reason="missing_service_tier_override",
                    private_message=f"missing runtime profile override for service_tier: {selector.service_tier}",
                )
            resolved = override.apply_to(resolved)

        effective_reasoning_effort = selector.reasoning_effort if selector.reasoning_effort is not None else self.default_reasoning_effort
        if effective_reasoning_effort is not None:
            if not self.supports_parameter("reasoning_effort"):
                raise _unsupported_reasoning_effort_error(
                    model,
                    effective_reasoning_effort,
                    reason="unsupported_reasoning_effort",
                    private_message="unsupported request parameters: reasoning_effort",
                )
            override = self.by_reasoning_effort.get(effective_reasoning_effort)
            if override is None:
                raise _unsupported_reasoning_effort_error(
                    model,
                    effective_reasoning_effort,
                    reason="missing_reasoning_effort_override",
                    private_message=f"missing runtime profile override for reasoning_effort: {effective_reasoning_effort}",
                )
            resolved = override.apply_to(resolved)

        _validate_compact_thresholds(resolved.soft_compact_threshold, resolved.compact_threshold)
        return resolved

    def all_models(self) -> tuple[str, ...]:
        models = [
            self.main.model,
            self.compactor.model,
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


class MCPToolConfig(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    type: str
    effect_class: Literal["safe", "visible", "mutation", "contextual", "unknown"] = "safe"


class MCPServerConfig(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    name: str
    url: str | None = None
    config: dict[str, Any] | None = None
    tools: dict[str, MCPToolConfig] = Field(default_factory=dict)

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
    debug: bool = False
    debug_payloads: bool = False
    log_json: bool = False
    log_file: str | None = None
    llm_lightning_api_key: str | None = None
    llm_novita_api_key: str | None = None
    llm_fireworks_api_key: str | None = None
    llm_crof_api_key: str | None = None
    llm_openrouter_api_key: str | None = None
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
            raise _missing_model_error()
        profile = self.runtime_model_profiles.get(model)
        if profile is None:
            raise _unknown_runtime_model_error(model)
        return profile.resolve(selector, model=model)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
