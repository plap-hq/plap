from __future__ import annotations

import os
import re
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
                    "service_tier",
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
                model="openrouter/deepseek/deepseek-v4-flash:novita,openrouter/deepseek/deepseek-v4-flash:atlas-cloud",
                max_completion_tokens=393_216,
                tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash",
                tokenizer_revision="6976c7ff1b30a1b2cb7805021b8ba4684041f136",
            ),
            compactor=RuntimeActorConfig(
                model="openrouter/deepseek/deepseek-v4-flash:novita,openrouter/deepseek/deepseek-v4-flash:atlas-cloud",
                max_completion_tokens=393_216,
                tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash",
                tokenizer_revision="6976c7ff1b30a1b2cb7805021b8ba4684041f136",
                reasoning_effort=ReasoningEffort.HIGH,
            ),
            main_debate=RuntimeActorConfig(
                model="openrouter/deepseek/deepseek-v4-flash:novita,openrouter/deepseek/deepseek-v4-flash:atlas-cloud",
                max_completion_tokens=393_216,
                tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash",
                tokenizer_revision="6976c7ff1b30a1b2cb7805021b8ba4684041f136",
            ),
            reviewer=RuntimeActorConfig(
                model="openrouter/deepseek/deepseek-v4-flash:novita,openrouter/deepseek/deepseek-v4-flash:atlas-cloud",
                max_completion_tokens=393_216,
                tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash",
                tokenizer_revision="6976c7ff1b30a1b2cb7805021b8ba4684041f136",
            ),
            arbitrator=RuntimeActorConfig(
                model="openrouter/deepseek/deepseek-v4-flash:novita,openrouter/deepseek/deepseek-v4-flash:atlas-cloud",
                max_completion_tokens=393_216,
                tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash",
                tokenizer_revision="6976c7ff1b30a1b2cb7805021b8ba4684041f136",
            ),
            reasoning_summarizer=RuntimeActorConfig(model="lightning/lightning-ai/gpt-oss-20b,openrouter/openai/gpt-oss-20b:deepinfra"),
            reviewer_transcript_token_budget=800_000,
            arbitrator_transcript_token_budget=300_000,
            soft_compact_threshold=200_000,
            compact_threshold=250_000,
            compact_max_rounds=0,
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
                    main=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                    main_debate=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                    reviewer=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                    arbitrator=RuntimeActorOverride(reasoning_effort=ReasoningEffort.NONE),
                ),
                ReasoningEffort.MEDIUM: RuntimeProfileOverride(
                    main=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                    main_debate=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                    reviewer=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                    arbitrator=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                ),
                ReasoningEffort.HIGH: RuntimeProfileOverride(
                    main=RuntimeActorOverride(reasoning_effort=ReasoningEffort.XHIGH),
                    main_debate=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                    reviewer=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                    arbitrator=RuntimeActorOverride(reasoning_effort=ReasoningEffort.HIGH),
                ),
                ReasoningEffort.XHIGH: RuntimeProfileOverride(
                    main=RuntimeActorOverride(reasoning_effort=ReasoningEffort.XHIGH),
                    main_debate=RuntimeActorOverride(reasoning_effort=ReasoningEffort.XHIGH),
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
                    "service_tier",
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
                model="gmicloud/XiaomiMiMo/MiMo-V2.5-Pro,crof/mimo-v2.5-pro-precision",
                max_completion_tokens=131_072,
                tokenizer_hf_repo="XiaomiMiMo/MiMo-V2.5-Pro",
                tokenizer_revision="a75207db63de3c320950fe6fcfa9ff60f341b7a2",
            ),
            compactor=RuntimeActorConfig(
                model="openrouter/deepseek/deepseek-v4-flash:novita,openrouter/deepseek/deepseek-v4-flash:atlas-cloud",
                max_completion_tokens=393_216,
                tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash",
                tokenizer_revision="6976c7ff1b30a1b2cb7805021b8ba4684041f136",
                reasoning_effort=ReasoningEffort.HIGH,
            ),
            main_debate=RuntimeActorConfig(
                model="gmicloud/XiaomiMiMo/MiMo-V2.5-Pro,crof/mimo-v2.5-pro-precision",
                max_completion_tokens=131_072,
                tokenizer_hf_repo="XiaomiMiMo/MiMo-V2.5-Pro",
                tokenizer_revision="a75207db63de3c320950fe6fcfa9ff60f341b7a2",
            ),
            reviewer=RuntimeActorConfig(
                model="openrouter/deepseek/deepseek-v4-flash:novita,openrouter/deepseek/deepseek-v4-flash:atlas-cloud",
                max_completion_tokens=393_216,
                tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash",
                tokenizer_revision="6976c7ff1b30a1b2cb7805021b8ba4684041f136",
            ),
            arbitrator=RuntimeActorConfig(
                model="openrouter/deepseek/deepseek-v4-flash:novita,openrouter/deepseek/deepseek-v4-flash:atlas-cloud",
                max_completion_tokens=393_216,
                tokenizer_hf_repo="deepseek-ai/DeepSeek-V4-Flash",
                tokenizer_revision="6976c7ff1b30a1b2cb7805021b8ba4684041f136",
            ),
            reasoning_summarizer=RuntimeActorConfig(model="lightning/lightning-ai/gpt-oss-20b,openrouter/openai/gpt-oss-20b:deepinfra"),
            reviewer_transcript_token_budget=800_000,
            arbitrator_transcript_token_budget=300_000,
            soft_compact_threshold=200_000,
            compact_threshold=250_000,
            compact_max_rounds=0,
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


_MCP_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _mcp_env_placeholders(value: str) -> set[str]:
    return {match.group(1) for match in _MCP_ENV_PATTERN.finditer(value)}


def _interpolate_mcp_string(value: str, *, variables: dict[str, str], server_name: str) -> str:
    missing = sorted(name for name in _mcp_env_placeholders(value) if name not in variables)
    if missing:
        raise ValueError(f"mcp server {server_name!r} references unset environment variables: {', '.join(missing)}")

    def replace(match: re.Match[str]) -> str:
        return variables[match.group(1)]

    return _MCP_ENV_PATTERN.sub(replace, value)


def _resolved_mcp_env(raw_env: dict[str, str], *, server_name: str) -> dict[str, str]:
    resolved: dict[str, str] = {}
    pending = dict(raw_env)
    pending_names = set(pending)
    while pending:
        progress = False
        variables = dict(os.environ)
        variables.update(resolved)
        for key, value in tuple(pending.items()):
            missing = _mcp_env_placeholders(value) - set(variables)
            if missing & pending_names:
                continue
            resolved[key] = _interpolate_mcp_string(value, variables=variables, server_name=server_name)
            del pending[key]
            pending_names.remove(key)
            progress = True
        if progress:
            continue
        raise ValueError(f"mcp server {server_name!r} has circular environment references: {', '.join(sorted(pending))}")
    return resolved


def _interpolate_mcp_value(value: Any, *, variables: dict[str, str], server_name: str) -> Any:
    if isinstance(value, str):
        return _interpolate_mcp_string(value, variables=variables, server_name=server_name)
    if isinstance(value, list):
        return [_interpolate_mcp_value(item, variables=variables, server_name=server_name) for item in value]
    if isinstance(value, dict):
        return {key: _interpolate_mcp_value(item, variables=variables, server_name=server_name) for key, item in value.items()}
    return value


def _merged_mcp_stdio_env(config: dict[str, Any], *, env: dict[str, str], server_name: str) -> dict[str, Any]:
    if not env:
        return config
    if (
        "command" not in config
        and config.get("transport") != "stdio"
        and config.get("type") != "stdio"
    ):
        return config
    existing_env = config.get("env")
    if existing_env is None:
        return {**config, "env": dict(env)}
    if not isinstance(existing_env, dict):
        raise TypeError(f"mcp server {server_name!r} has a non-object stdio env")
    return {**config, "env": {**env, **existing_env}}


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


def _validate_runtime_transform_bounds(
    min_value: float | int | None,
    max_value: float | int | None,
    *,
    label: str,
) -> None:
    if min_value is not None and max_value is not None and min_value > max_value:
        raise ValueError(f"{label} min_value cannot exceed max_value")


class RuntimeFloatTransform(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    disabled: bool = False
    fixed: float | None = None
    default: float | None = None
    scale: float = 1.0
    offset: float = 0.0
    min_value: float | None = None
    max_value: float | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> RuntimeFloatTransform:
        _validate_runtime_transform_bounds(self.min_value, self.max_value, label="float transform")
        return self

    def apply(
        self,
        value: float | None,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float | None:
        if self.disabled:
            return None
        if self.fixed is not None:
            resolved = self.fixed
        elif value is None:
            resolved = self.default
        else:
            resolved = value * self.scale + self.offset
        if resolved is None:
            return None
        if self.min_value is not None:
            resolved = max(resolved, self.min_value)
        if self.max_value is not None:
            resolved = min(resolved, self.max_value)
        if minimum is not None:
            resolved = max(resolved, minimum)
        if maximum is not None:
            resolved = min(resolved, maximum)
        return resolved


class RuntimeIntTransform(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    disabled: bool = False
    fixed: int | None = None
    default: int | None = None
    min_value: int | None = None
    max_value: int | None = None

    @model_validator(mode="after")
    def validate_bounds(self) -> RuntimeIntTransform:
        _validate_runtime_transform_bounds(self.min_value, self.max_value, label="int transform")
        return self

    def apply(
        self,
        value: int | None,
        *,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int | None:
        if self.disabled:
            return None
        resolved = self.fixed if self.fixed is not None else self.default if value is None else value
        if resolved is None:
            return None
        if self.min_value is not None:
            resolved = max(resolved, self.min_value)
        if self.max_value is not None:
            resolved = min(resolved, self.max_value)
        if minimum is not None:
            resolved = max(resolved, minimum)
        if maximum is not None:
            resolved = min(resolved, maximum)
        return resolved


class RuntimeActorSamplingConfig(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    temperature: RuntimeFloatTransform | None = None
    top_p: RuntimeFloatTransform | None = None
    top_logprobs: RuntimeIntTransform | None = None

    def apply_temperature(self, value: float | None) -> float | None:
        if self.temperature is None:
            return value
        return self.temperature.apply(value, minimum=0, maximum=2)

    def apply_top_p(self, value: float | None) -> float | None:
        if self.top_p is None:
            return value
        return self.top_p.apply(value, minimum=0, maximum=1)

    def apply_top_logprobs(self, value: int | None) -> int | None:
        if self.top_logprobs is None:
            return value
        return self.top_logprobs.apply(value, minimum=0, maximum=20)


class RuntimeActorConfig(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    model: str
    max_completion_tokens: int | None = Field(default=None, ge=1)
    tokenizer_hf_repo: str | None = None
    tokenizer_revision: str | None = None
    tokenizer_trust_remote_code: bool = False
    reasoning_effort: ReasoningEffort | None = None
    service_tier: ServiceTier | None = None
    sampling: RuntimeActorSamplingConfig = Field(default_factory=RuntimeActorSamplingConfig)
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

    def cap_max_completion_tokens(self, value: int | None) -> int | None:
        if self.max_completion_tokens is None:
            return value
        if value is None:
            return self.max_completion_tokens
        return min(value, self.max_completion_tokens)

    def map_temperature(self, value: float | None) -> float | None:
        return self.sampling.apply_temperature(value)

    def map_top_p(self, value: float | None) -> float | None:
        return self.sampling.apply_top_p(value)

    def map_top_logprobs(self, value: int | None) -> int | None:
        return self.sampling.apply_top_logprobs(value)


class RuntimeActorOverride(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    model: str | None = None
    max_completion_tokens: int | None = Field(default=None, ge=1)
    reasoning_effort: ReasoningEffort | None = None
    service_tier: ServiceTier | None = None

    def overridden_fields(self, prefix: str) -> set[str]:
        fields: set[str] = set()
        if self.model is not None:
            fields.add(f"{prefix}.model")
        if self.max_completion_tokens is not None:
            fields.add(f"{prefix}.max_completion_tokens")
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
    reasoning_to_output: float | None = Field(default=None, ge=0)

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
        if self.reasoning_to_output is not None:
            updates["reasoning_to_output"] = self.reasoning_to_output
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
        if self.reasoning_to_output is not None:
            fields.add("reasoning_to_output")
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
    reasoning_to_output: float = Field(default=1.0, ge=0)
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

    argument_adapter: Literal["web_search_user_location"] | None = None
    type: str
    effect_class: Literal["safe", "visible", "mutation", "contextual"] = "safe"


class MCPServerConfig(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    name: str
    config: dict[str, Any]
    env: dict[str, str] = Field(default_factory=dict)
    tools: dict[str, MCPToolConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_config(self) -> MCPServerConfig:
        if not self.config:
            raise ValueError("mcp server config is required")
        if "mcpServers" in self.config:
            raise ValueError("mcp server config must describe a single server, not an mcpServers mapping")
        return self

    def mcp_config(self) -> dict[str, Any]:
        env = _resolved_mcp_env(self.env, server_name=self.name)
        variables = dict(os.environ)
        variables.update(env)
        config = _merged_mcp_stdio_env(self.config, env=env, server_name=self.name)
        return {
            "mcpServers": {
                self.name: _interpolate_mcp_value(config, variables=variables, server_name=self.name),
            }
        }


def _default_jina_mcp_server() -> MCPServerConfig | None:
    if not os.environ.get("JINA_API_KEY"):
        return None
    tool_names = (
        "read_url",
        "search_web",
        "search_arxiv",
        "search_ssrn",
        "search_bibtex",
    )
    return MCPServerConfig(
        name="jina",
        config={
            "url": f"https://mcp.jina.ai/v1?include_tools={','.join(tool_names)}",
            "headers": {"Authorization": "Bearer ${JINA_API_KEY}"},
        },
        tools={
            tool_name: MCPToolConfig(
                type="web_search",
                argument_adapter="web_search_user_location" if tool_name == "search_web" else None,
            )
            for tool_name in tool_names
        },
    )


def _default_mcp_servers() -> list[MCPServerConfig]:
    jina_server = _default_jina_mcp_server()
    if jina_server is None:
        return []
    return [jina_server]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PLAP_", extra="ignore")

    api_key_pepper: str
    database_url: str
    sealing_keys: list[str]
    debug: bool = False
    debug_payloads: bool = False
    debug_debate_summaries: bool = False
    log_json: bool = False
    log_file: str | None = None
    llm_lightning_api_key: str | None = None
    llm_canopywave_api_key: str | None = None
    llm_gmicloud_api_key: str | None = None
    llm_novita_api_key: str | None = None
    llm_fireworks_api_key: str | None = None
    llm_crof_api_key: str | None = None
    llm_openrouter_api_key: str | None = None
    tool_effect_classifier_model: str = "lightning/lightning-ai/gpt-oss-20b,openrouter/openai/gpt-oss-20b:deepinfra"
    tool_call_effect_classifier_model: str = "lightning/lightning-ai/gpt-oss-20b,openrouter/openai/gpt-oss-20b:deepinfra"
    tool_effect_classifier_cache_model: str = "gpt-oss-20b"
    tool_call_effect_classifier_cache_model: str = "gpt-oss-20b"
    tool_classifier_max_concurrency: int = 2
    tool_policy_l1_maxsize: int = 4096
    tool_call_policy_l1_maxsize: int = 4096
    runtime_model_profiles: dict[str, RuntimeModelProfileConfig] = Field(default_factory=_default_runtime_model_profiles)
    mcp_servers: list[MCPServerConfig] = Field(default_factory=_default_mcp_servers)

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
