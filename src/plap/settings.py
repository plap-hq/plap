from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PLAP_", extra="ignore")

    api_key_pepper: str
    database_url: str
    sealing_keys: list[str]
    llm_lightning_api_key: str | None = None
    llm_novita_api_key: str | None = None
    llm_fireworks_api_key: str | None = None
    llm_lightning_model_prefixes: list[str] = Field(
        default_factory=lambda: ["lightning-ai/"]
    )
    llm_novita_model_prefixes: list[str] = Field(
        default_factory=lambda: [
            "deepseek/",
            "minimax/",
            "moonshotai/",
            "openai/",
            "qwen/",
            "zai-org/",
        ]
    )
    llm_fireworks_model_prefixes: list[str] = Field(
        default_factory=lambda: ["accounts/fireworks/"]
    )
    tool_classifier_model: str | None = None
    tool_call_classifier_model: str | None = None
    tool_classifier_max_concurrency: int = 4
    tool_policy_l1_maxsize: int = 4096
    tool_call_policy_l1_maxsize: int = 4096
    web_search_brave_api_key: str | None = None
    web_search_mcp_url: str | None = None
    web_search_mcp_command: str = "npx"
    web_search_mcp_args: list[str] = Field(
        default_factory=lambda: ["-y", "@brave/brave-search-mcp-server"]
    )
    web_search_mcp_tool_names: list[str] | None = None

    @field_validator("sealing_keys", mode="before")
    @classmethod
    def split_sealing_keys(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator(
        "llm_lightning_model_prefixes",
        "llm_novita_model_prefixes",
        "llm_fireworks_model_prefixes",
        "web_search_mcp_args",
        "web_search_mcp_tool_names",
        mode="before",
    )
    @classmethod
    def split_model_prefixes(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
