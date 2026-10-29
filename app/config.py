from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


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

    model_service_url: str = "http://localhost:8001"
    # The same file the model service classifies over, so every label it can
    # return has a curated Korean word here (rule 1: vocabulary comes from the CSV).
    vocab_path: Path = REPO_ROOT / "model_service" / "labels_ko.csv"

    # Required like the JWT secret: a wrong or missing bucket should stop the app
    # at startup, not fail a user's first upload. AWS credentials are not config:
    # boto3 finds them itself (the EC2 instance role in production, ~/.aws locally).
    s3_bucket: str
    aws_region: str = "ap-northeast-2"

    gemini_api_key: SecretStr
    # An env var, so a model retirement or a cheaper model is a config change.
    gemini_model: str = "gemini-3.5-flash-lite"


# Cached so env parsing happens once; tests can call get_settings.cache_clear().
@lru_cache
def get_settings() -> Settings:
    return Settings()
