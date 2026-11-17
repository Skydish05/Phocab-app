"""Pydantic request/response models for the JSON API."""

from datetime import datetime
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


class CardOut(BaseModel):
    card_id: int
    label_en: str
    korean: str
    romanization: str
    topik_level: int
    example: str
    # The model's score for this label, shown so a low-confidence card can be
    # flagged in the UI and reported with the "this is wrong" button.
    confidence: float


class DeckCard(BaseModel):
    """A card as the deck shows it: the word plus the user's Leitner state."""

    card_id: int
    label_en: str
    korean: str
    romanization: str
    topik_level: int
    # Nullable like words.example: a vocabulary fix voids the sentence until it
    # is regenerated, and the card should still be listed meanwhile.
    example: str | None
    box: int
    next_review_at: datetime
    created_at: datetime


class CardList(BaseModel):
    items: list[DeckCard]
    # The full count (not just this page), so a client can show "12 cards" and
    # know when to stop paging.
    total: int


class ReviewNext(BaseModel):
    # None when nothing is due; the client then shows next_due_at instead.
    card: DeckCard | None
    due_count: int
    # When the soonest card becomes due; None only for an empty deck.
    next_due_at: datetime | None


class ReviewAnswer(BaseModel):
    correct: bool


class ReviewResult(BaseModel):
    card_id: int
    box: int
    next_review_at: datetime
    times_correct: int
    times_wrong: int
