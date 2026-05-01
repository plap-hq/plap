from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

type RuntimeParameterValue = str | int | float | bool | None


def _default_runtime_model_profiles() -> dict[str, RuntimeModelProfileConfig]:
    return {
        "plap-ai/wisp-nano": RuntimeModelProfileConfig(
            display_name="Wisp Nano",
            main_model="crof/qwen3.5-9b",
            main_debate_model="crof/qwen3.5-9b",
            reviewer_model="crof/qwen3.5-9b",
            arbitrator_model="crof/qwen3.5-9b",
            reasoning_summarizer_model="lightning/lightning-ai/gpt-oss-120b",
            transcript_token_budget=200_000,
            compression_soft_token_budget=100_000,
            compression_hard_token_budget=150_000,
            compression_max_rounds=3,
            debate_max_rounds=2,
        )
    }


def _validate_compression_budgets(
    soft_budget: int | None,
    hard_budget: int | None,
) -> None:
    if soft_budget is not None and hard_budget is not None and hard_budget <= soft_budget:
        raise ValueError("compression hard token budget must exceed the soft token budget")


def _validate_runtime_profile_parameters(parameters: Mapping[str, RuntimeParameterValue]) -> None:
    for key, value in parameters.items():
        if not isinstance(key, str) or not key:
            raise ValueError("runtime profile variant parameter keys must be non-empty strings")
        if value is None or isinstance(value, str | bool | int | float):
            continue
        raise ValueError(f"runtime profile variant parameter {key!r} must be a scalar value")


class RuntimeParameterBag:
    def __init__(self, values: Mapping[str, RuntimeParameterValue] | None = None) -> None:
        self.values = dict(values or {})
        _validate_runtime_profile_parameters(self.values)

    def get(self, key: str) -> RuntimeParameterValue:
        return self.values.get(key)


class RuntimeModelProfilePatch(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    display_name: str | None = None
    main_model: str | None = None
    main_debate_model: str | None = None
    reviewer_model: str | None = None
    arbitrator_model: str | None = None
    reasoning_summarizer_model: str | None = None
    transcript_token_budget: int | None = Field(default=None, ge=0)
    compression_soft_token_budget: int | None = Field(default=None, ge=0)
    compression_hard_token_budget: int | None = Field(default=None, ge=0)
    compression_max_rounds: int | None = Field(default=None, ge=0)
    debate_max_rounds: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_compression_config(self) -> RuntimeModelProfilePatch:
        _validate_compression_budgets(
            self.compression_soft_token_budget,
            self.compression_hard_token_budget,
        )
        return self


class RuntimeModelProfileVariant(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    when: dict[str, RuntimeParameterValue] = Field(default_factory=dict)
    patch: RuntimeModelProfilePatch

    @model_validator(mode="after")
    def validate_variant(self) -> RuntimeModelProfileVariant:
        _validate_runtime_profile_parameters(self.when)
        return self


class RuntimeModelProfileConfig(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    display_name: str
    main_model: str
    main_debate_model: str
    reviewer_model: str
    arbitrator_model: str
    reasoning_summarizer_model: str
    transcript_token_budget: int = Field(default=0, ge=0)
    compression_soft_token_budget: int | None = Field(default=None, ge=0)
    compression_hard_token_budget: int | None = Field(default=None, ge=0)
    compression_max_rounds: int = Field(default=3, ge=0)
    debate_max_rounds: int = Field(default=2, ge=0)
    variants: list[RuntimeModelProfileVariant] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_compression_config(self) -> RuntimeModelProfileConfig:
        _validate_compression_budgets(
            self.compression_soft_token_budget,
            self.compression_hard_token_budget,
        )
        for variant in self.variants:
            patch = variant.patch
            soft_budget = (
                patch.compression_soft_token_budget
                if patch.compression_soft_token_budget is not None
                else self.compression_soft_token_budget
            )
            hard_budget = (
                patch.compression_hard_token_budget
                if patch.compression_hard_token_budget is not None
                else self.compression_hard_token_budget
            )
            _validate_compression_budgets(soft_budget, hard_budget)
        return self

    def resolve(
        self,
        parameters: Mapping[str, RuntimeParameterValue] | None = None,
    ) -> RuntimeModelProfileConfig:
        resolved = self
        normalized_parameters = RuntimeParameterBag(values=dict(parameters or {}))
        for variant in self.variants:
            if all(normalized_parameters.get(key) == value for key, value in variant.when.items()):
                resolved = resolved.model_copy(update=variant.patch.model_dump(exclude_none=True))
        _validate_compression_budgets(
            resolved.compression_soft_token_budget,
            resolved.compression_hard_token_budget,
        )
        return resolved

    def all_models(self) -> tuple[str, ...]:
        models = [
            self.main_model,
            self.main_debate_model,
            self.reviewer_model,
            self.arbitrator_model,
            self.reasoning_summarizer_model,
        ]
        for variant in self.variants:
            patch = variant.patch
            if patch.main_model is not None:
                models.append(patch.main_model)
            if patch.main_debate_model is not None:
                models.append(patch.main_debate_model)
            if patch.reviewer_model is not None:
                models.append(patch.reviewer_model)
            if patch.arbitrator_model is not None:
                models.append(patch.arbitrator_model)
            if patch.reasoning_summarizer_model is not None:
                models.append(patch.reasoning_summarizer_model)
        return tuple(model for model in models if "/" in model)


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
        parameters: Mapping[str, RuntimeParameterValue] | None = None,
    ) -> RuntimeModelProfileConfig:
        if model is None:
            raise ValueError("model is required")
        profile = self.runtime_model_profiles.get(model)
        if profile is None:
            raise ValueError(f"unknown runtime model: {model!r}")
        return profile.resolve(parameters)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
