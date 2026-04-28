from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

type RuntimeServiceTier = Literal["auto", "default", "priority", "flex"]


class RuntimeModelProfileOverrideConfig(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    main_model: str | None = None
    main_debate_model: str | None = None
    reviewer_model: str | None = None
    arbitrator_model: str | None = None
    reasoning_summarizer_model: str | None = None
    transcript_token_budget: int | None = Field(default=None, ge=0)


class RuntimeModelProfileConfig(BaseModel):
    model_config = SettingsConfigDict(extra="forbid")

    main_model: str
    main_debate_model: str
    reviewer_model: str
    arbitrator_model: str
    reasoning_summarizer_model: str
    transcript_token_budget: int = Field(default=0, ge=0)
    service_tier_overrides: dict[
        RuntimeServiceTier,
        RuntimeModelProfileOverrideConfig,
    ] = Field(default_factory=dict)

    def for_service_tier(
        self,
        service_tier: RuntimeServiceTier | str | None,
    ) -> RuntimeModelProfileConfig:
        if service_tier in (None, "auto", "default"):
            return self
        override = self.service_tier_overrides.get(service_tier)  # type: ignore[arg-type]
        if override is None:
            return self
        return self.model_copy(update=override.model_dump(exclude_none=True))


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
