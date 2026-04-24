from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PLAP_", extra="ignore")

    api_key_pepper: str
    database_url: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
