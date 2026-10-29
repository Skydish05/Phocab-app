"""Pydantic request/response models for the JSON API."""

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, EmailStr, Field

# Lowercased so "A@x.com" and "a@x.com" are one account: the unique constraint
# on users.email is case-sensitive.
Email = Annotated[EmailStr, AfterValidator(str.lower)]


class RegisterRequest(BaseModel):
    email: Email
    # The upper bound stops a multi-megabyte "password" from making the server
    # spend CPU and memory hashing it.
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: Email
    # No min_length here: a too-short password should be a plain 401, not a 422
    # that hints at the password rules.
    password: str = Field(max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str


class MeOut(BaseModel):
    email: str
