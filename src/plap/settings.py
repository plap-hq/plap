from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

type RuntimeServiceTier = Literal["auto", "default", "priority", "flex"]


def _validate_compression_thresholds(
    thresholds: list[int],
    hard_budget: int | None,
) -> None:
    if any(value < 0 for value in thresholds):
        raise ValueError("compression token thresholds must be non-negative")
    if len(set(thresholds)) != len(thresholds) or thresholds != sorted(thresholds):
        raise ValueError("compression token thresholds must be strictly increasing")
    if hard_budget is not None and thresholds and hard_budget <= thresholds[-1]:
        raise ValueError(
            "compression hard token budget must exceed the last token threshold"
        )


class RuntimeModelProfileOverrideConfig(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    main_model: str | None = None
    main_debate_model: str | None = None
    reviewer_model: str | None = None
    arbitrator_model: str | None = None
    reasoning_summarizer_model: str | None = None
    transcript_token_budget: int | None = Field(default=None, ge=0)
    compression_token_thresholds: list[int] | None = None
    compression_hard_token_budget: int | None = Field(default=None, ge=0)
    compression_max_rounds: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_compression_config(self) -> RuntimeModelProfileOverrideConfig:
        if self.compression_token_thresholds is not None:
            _validate_compression_thresholds(
                self.compression_token_thresholds,
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
    compression_token_thresholds: list[int] = Field(default_factory=list)
    compression_hard_token_budget: int | None = Field(default=None, ge=0)
    compression_max_rounds: int = Field(default=3, ge=0)
    service_tier_overrides: dict[
        RuntimeServiceTier,
        RuntimeModelProfileOverrideConfig,
    ] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_compression_config(self) -> RuntimeModelProfileConfig:
        _validate_compression_thresholds(
            self.compression_token_thresholds,
            self.compression_hard_token_budget,
        )
        for override in self.service_tier_overrides.values():
            thresholds = (
                override.compression_token_thresholds
                if override.compression_token_thresholds is not None
                else self.compression_token_thresholds
            )
            hard_budget = (
                override.compression_hard_token_budget
                if override.compression_hard_token_budget is not None
                else self.compression_hard_token_budget
            )
            _validate_compression_thresholds(thresholds, hard_budget)
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
        _validate_compression_thresholds(
            profile.compression_token_thresholds,
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
    tool_classifier_model: str | None = None
    tool_call_classifier_model: str | None = None
    tool_classifier_max_concurrency: int = 4
    tool_policy_l1_maxsize: int = 4096
    tool_call_policy_l1_maxsize: int = 4096
    runtime_model_profiles: dict[str, RuntimeModelProfileConfig] = Field(
        default_factory=dict
    )
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
