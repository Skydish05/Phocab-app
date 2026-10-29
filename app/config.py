from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All config comes from env vars (or .env locally), never from code."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    # Default matches infra/docker-compose.yml's local-only credentials, so
    # uvicorn works without a .env; the server sets DATABASE_URL.
    database_url: str = "postgresql+psycopg://phocab:phocab@localhost:5432/phocab"


# Cached so env parsing happens once; tests can call get_settings.cache_clear().
@lru_cache
def get_settings() -> Settings:
    return Settings()
