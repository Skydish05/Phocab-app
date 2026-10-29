from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All config comes from env vars (or .env locally), never from code."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    # Default matches infra/docker-compose.yml's local-only credentials, so
    # uvicorn works without a .env; the server sets DATABASE_URL.
    database_url: str = "postgresql+psycopg://phocab:phocab@localhost:5432/phocab"

    # No default on purpose: a default secret would silently become the production
    # secret on some deploy. Missing or short (guessable) secrets fail at startup.
    # SecretStr keeps the value out of reprs, logs, and tracebacks.
    jwt_secret: SecretStr = Field(min_length=32)
    # One short-lived access token and no refresh tokens (out of scope), so this
    # is the window a leaked token stays usable.
    jwt_expire_minutes: int = Field(default=60, gt=0)


# Cached so env parsing happens once; tests can call get_settings.cache_clear().
@lru_cache
def get_settings() -> Settings:
    return Settings()
