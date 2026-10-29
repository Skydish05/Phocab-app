"""Shared route dependencies."""

from functools import lru_cache
from typing import Annotated

import boto3
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import User
from app.security import decode_access_token
from app.services.cache import Vocabulary, load_vocabulary
from app.services.card_generator import LLMClient
from app.services.llm import GeminiClient
from app.services.recognizer import Recognizer
from app.services.storage import PhotoStorage

# auto_error=False: we raise the 401 ourselves, since FastAPI versions disagree on
# 401 vs 403 for a missing header. HTTPBearer still adds the /docs "Authorize" box.
_bearer = HTTPBearer(auto_error=False)

DbSession = Annotated[Session, Depends(get_db)]


def get_current_user(
    db: DbSession,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> User:
    # One message for every failure, so a caller can't tell a forged token from
    # an expired one or from a deleted account.
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None:
        raise unauthorized
    user_id = decode_access_token(credentials.credentials)
    # Load the user on every request (one primary-key lookup) so a deleted
    # account's still-unexpired token stops working immediately.
    user = db.get(User, user_id) if user_id is not None else None
    if user is None:
        raise unauthorized
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


# External clients as cached dependencies: built once on first use and shared by
# every request (connection reuse), and replaceable with fakes in tests through
# app.dependency_overrides, with no monkeypatching.
@lru_cache
def get_vocabulary() -> Vocabulary:
    return load_vocabulary(get_settings().vocab_path)


@lru_cache
def get_recognizer() -> Recognizer:
    return Recognizer(get_settings().model_service_url)


@lru_cache
def get_storage() -> PhotoStorage:
    settings = get_settings()
    return PhotoStorage(settings.s3_bucket, boto3.client("s3", region_name=settings.aws_region))


@lru_cache
def get_llm() -> LLMClient:
    settings = get_settings()
    return GeminiClient(settings.gemini_api_key.get_secret_value(), settings.gemini_model)
