"""ORM tables. Shared, cacheable content (words) is split from per-user state
(user_cards), so one generated card serves every user who photographs that object."""

import enum
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    MetaData,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Deterministic constraint names. Without them Postgres invents names, and a later
# Alembic migration can't reliably drop or alter a constraint it can't name.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def created_at_column() -> Mapped[datetime]:
    # The DB clock sets it, so every writer (app, scripts, psql) agrees, and
    # timezone-aware so the server's local timezone never leaks into the data.
    return mapped_column(DateTime(timezone=True), server_default=func.now())


def str_enum(enum_cls: type[enum.StrEnum], name: str) -> Enum:
    # VARCHAR + CHECK instead of a native Postgres ENUM: Alembic autogenerate
    # doesn't detect new values in native enums, while changing a CHECK is a
    # one-line migration. values_callable stores "wrong_object", not the member name.
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=max(len(m.value) for m in enum_cls),
        values_callable=lambda cls: [m.value for m in cls],
    )


class WordSource(enum.StrEnum):
    HF = "hf"  # zero-shot model picked it from labels_ko.csv
    VLM = "vlm"  # hosted vision-language fallback named it


class FeedbackReason(enum.StrEnum):
    WRONG_OBJECT = "wrong_object"
    WRONG_KOREAN = "wrong_korean"
    OTHER = "other"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    # 320 = the longest valid email address (64 local + @ + 255 domain).
    email: Mapped[str] = mapped_column(String(320), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    # Placeholder default; the rate-limit item decides the real number.
    daily_quota: Mapped[int] = mapped_column(server_default="20")
    created_at: Mapped[datetime] = created_at_column()


class Word(Base):
    """The card cache: one row per object label, shared by all users.

    Vocabulary fields (korean, romanization, topik_level) come from labels_ko.csv;
    only `example` is LLM-written. `example` is nullable so a vocabulary row can
    exist before its sentence is generated (or after a Korean-word fix voids it).
    """

    __tablename__ = "words"
    __table_args__ = (
        # TOPIK has 6 levels.
        CheckConstraint("topik_level BETWEEN 1 AND 6", name="topik_level_range"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # The UNIQUE constraint already builds the index PLAN.md asks for on label_en;
    # a separate index would duplicate it and slow down writes.
    label_en: Mapped[str] = mapped_column(String(100), unique=True)
    korean: Mapped[str] = mapped_column(String(100))
    romanization: Mapped[str] = mapped_column(String(100))
    topik_level: Mapped[int] = mapped_column(SmallInteger)
    example: Mapped[str | None] = mapped_column(Text)
    audio_url: Mapped[str | None] = mapped_column(String(1024))
    source: Mapped[WordSource] = mapped_column(
        str_enum(WordSource, "word_source"), server_default=WordSource.HF.value
    )
    created_at: Mapped[datetime] = created_at_column()


class Photo(Base):
    """Every upload and which model path handled it (input for evaluation)."""

    __tablename__ = "photos"
    __table_args__ = (Index(None, "user_id", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    # CASCADE: deleting an account deletes the user's data (needed for PIPA requests).
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    s3_key: Mapped[str] = mapped_column(String(1024), unique=True)
    # Nullable: recognition can fail, and the upload is still worth recording.
    word_id: Mapped[int | None] = mapped_column(ForeignKey("words.id"))
    # Nullable: the VLM fallback names an object without a comparable score.
    confidence: Mapped[float | None]
    # The model id (e.g. the SigLIP checkpoint or the VLM name), not just "hf"/"vlm",
    # so eval can tell model versions apart.
    model_used: Mapped[str] = mapped_column(String(200))
    latency_ms: Mapped[int]
    created_at: Mapped[datetime] = created_at_column()

    word: Mapped["Word | None"] = relationship()


class UserCard(Base):
    """A user's copy of a word in their deck, with its Leitner state."""

    __tablename__ = "user_cards"
    __table_args__ = (
        # One card per word per user: a second photo of a cup shouldn't add a second cup card.
        UniqueConstraint("user_id", "word_id"),
        CheckConstraint("box BETWEEN 1 AND 5", name="box_range"),
        CheckConstraint("times_correct >= 0", name="times_correct_nonnegative"),
        CheckConstraint("times_wrong >= 0", name="times_wrong_nonnegative"),
        # Serves "next due card for this user" without scanning every user's cards.
        Index(None, "user_id", "next_review_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    # No ondelete on words: they're shared, so deleting one must fail loudly
    # rather than silently remove cards from every user's deck.
    word_id: Mapped[int] = mapped_column(ForeignKey("words.id"))
    # SET NULL: photos are deleted after a retention period, but the card stays.
    photo_id: Mapped[int | None] = mapped_column(ForeignKey("photos.id", ondelete="SET NULL"))
    box: Mapped[int] = mapped_column(SmallInteger, server_default="1")
    next_review_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    times_correct: Mapped[int] = mapped_column(server_default="0")
    times_wrong: Mapped[int] = mapped_column(server_default="0")
    created_at: Mapped[datetime] = created_at_column()

    word: Mapped[Word] = relationship()
    photo: Mapped[Photo | None] = relationship()


class Feedback(Base):
    """'This is wrong' reports: the corrections dataset for evaluation."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    # SET NULL for the same retention reason as user_cards.photo_id.
    photo_id: Mapped[int | None] = mapped_column(ForeignKey("photos.id", ondelete="SET NULL"))
    word_id: Mapped[int | None] = mapped_column(ForeignKey("words.id"))
    reason: Mapped[FeedbackReason] = mapped_column(str_enum(FeedbackReason, "feedback_reason"))
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at_column()
