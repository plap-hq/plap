from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

type RuntimeServiceTier = Literal["auto", "default", "priority", "flex"]


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
        )
    }


def _validate_compression_budgets(
    soft_budget: int | None,
    hard_budget: int | None,
) -> None:
    if soft_budget is not None and hard_budget is not None and hard_budget <= soft_budget:
        raise ValueError("compression hard token budget must exceed the soft token budget")


class RuntimeModelProfileOverrideConfig(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    main_model: str | None = None
    main_debate_model: str | None = None
    reviewer_model: str | None = None
    arbitrator_model: str | None = None
    reasoning_summarizer_model: str | None = None
    transcript_token_budget: int | None = Field(default=None, ge=0)
    compression_soft_token_budget: int | None = Field(default=None, ge=0)
    compression_hard_token_budget: int | None = Field(default=None, ge=0)
    compression_max_rounds: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_compression_config(self) -> RuntimeModelProfileOverrideConfig:
        _validate_compression_budgets(
            self.compression_soft_token_budget,
            self.compression_hard_token_budget,
        )
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
    service_tier_overrides: dict[
        RuntimeServiceTier,
        RuntimeModelProfileOverrideConfig,
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_compression_config(self) -> RuntimeModelProfileConfig:
        _validate_compression_budgets(
            self.compression_soft_token_budget,
            self.compression_hard_token_budget,
        )
        for override in self.service_tier_overrides.values():
            soft_budget = (
                override.compression_soft_token_budget
                if override.compression_soft_token_budget is not None
                else self.compression_soft_token_budget
            )
            hard_budget = (
                override.compression_hard_token_budget
                if override.compression_hard_token_budget is not None
                else self.compression_hard_token_budget
            )
            _validate_compression_budgets(soft_budget, hard_budget)
        return self

    def for_service_tier(
        self,
        service_tier: RuntimeServiceTier | str | None,
    ) -> RuntimeModelProfileConfig:
        if service_tier in (None, "auto", "default"):
            return self
        override = self.service_tier_overrides.get(service_tier)  # type: ignore[arg-type]
        if override is None:
            return self
        profile = self.model_copy(update=override.model_dump(exclude_none=True))
        _validate_compression_budgets(
            profile.compression_soft_token_budget,
            profile.compression_hard_token_budget,
        )
        return profile


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
    web_search_mcp_url: str | None = None
    web_search_mcp_config: dict[str, Any] | None = None
    web_search_mcp_tool_names: list[str] | None = None

    @field_validator("sealing_keys", mode="before")
    @classmethod
    def split_sealing_keys(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator(
        "web_search_mcp_tool_names",
        mode="before",
    )
    @classmethod
    def split_comma_separated_values(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
