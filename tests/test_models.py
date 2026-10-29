import pytest
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Feedback, FeedbackReason, Photo, User, UserCard, Word, WordSource


def make_user(session: Session, email: str = "learner@example.com") -> User:
    user = User(email=email, password_hash="not-a-real-hash")
    session.add(user)
    session.flush()
    return user


def make_word(session: Session, label_en: str = "cup", **overrides) -> Word:
    fields = {"korean": "컵", "romanization": "keop", "topik_level": 1} | overrides
    word = Word(label_en=label_en, **fields)
    session.add(word)
    session.flush()
    return word


def make_photo(session: Session, user: User, word: Word) -> Photo:
    photo = Photo(
        user_id=user.id,
        s3_key=f"photos/{user.id}/{word.label_en}.jpg",
        word_id=word.id,
        confidence=0.9,
        model_used="test-model",
        latency_ms=120,
    )
    session.add(photo)
    session.flush()
    return photo


def count(session: Session, model) -> int:
    return session.scalar(select(func.count()).select_from(model))


def test_new_user_gets_database_defaults(db_session: Session):
    user = make_user(db_session)
    db_session.refresh(user)

    assert user.daily_quota == 20
    assert user.created_at.tzinfo is not None


def test_new_card_starts_in_box_1_and_is_due(db_session: Session):
    user, word = make_user(db_session), make_word(db_session)
    card = UserCard(user_id=user.id, word_id=word.id)
    db_session.add(card)
    db_session.flush()
    db_session.refresh(card)

    assert (card.box, card.times_correct, card.times_wrong) == (1, 0, 0)
    assert card.next_review_at <= db_session.scalar(select(func.now()))


def test_word_defaults_to_hf_source_and_no_example(db_session: Session):
    word = make_word(db_session)
    db_session.refresh(word)

    assert word.source is WordSource.HF
    assert word.example is None


def test_enum_columns_store_the_value_not_the_member_name(db_session: Session):
    user = make_user(db_session)
    db_session.add(Feedback(user_id=user.id, reason=FeedbackReason.WRONG_KOREAN))
    db_session.flush()

    stored = db_session.scalar(text("SELECT reason FROM feedback"))

    assert stored == "wrong_korean"


def test_duplicate_email_is_rejected(db_session: Session):
    make_user(db_session)

    with pytest.raises(IntegrityError, match="uq_users_email"):
        make_user(db_session)


def test_duplicate_label_is_rejected(db_session: Session):
    make_word(db_session)

    with pytest.raises(IntegrityError, match="uq_words_label_en"):
        make_word(db_session)


def test_second_card_for_the_same_word_is_rejected(db_session: Session):
    user, word = make_user(db_session), make_word(db_session)
    db_session.add(UserCard(user_id=user.id, word_id=word.id))
    db_session.flush()

    db_session.add(UserCard(user_id=user.id, word_id=word.id))
    with pytest.raises(IntegrityError, match="uq_user_cards_user_id_word_id"):
        db_session.flush()


@pytest.mark.parametrize("box", [0, 6])
def test_box_outside_1_to_5_is_rejected(db_session: Session, box: int):
    user, word = make_user(db_session), make_word(db_session)

    db_session.add(UserCard(user_id=user.id, word_id=word.id, box=box))
    with pytest.raises(IntegrityError, match="ck_user_cards_box_range"):
        db_session.flush()


def test_negative_answer_count_is_rejected(db_session: Session):
    user, word = make_user(db_session), make_word(db_session)

    db_session.add(UserCard(user_id=user.id, word_id=word.id, times_wrong=-1))
    with pytest.raises(IntegrityError, match="ck_user_cards_times_wrong_nonnegative"):
        db_session.flush()


@pytest.mark.parametrize("level", [0, 7])
def test_topik_level_outside_1_to_6_is_rejected(db_session: Session, level: int):
    with pytest.raises(IntegrityError, match="ck_words_topik_level_range"):
        make_word(db_session, topik_level=level)


# Raw SQL on purpose: the ORM refuses unknown enum values before they reach the
# database, but the CHECK must also stop scripts and psql sessions.
def test_unknown_feedback_reason_is_rejected_by_the_database(db_session: Session):
    user = make_user(db_session)

    with pytest.raises(IntegrityError, match="ck_feedback_feedback_reason"):
        db_session.execute(
            text("INSERT INTO feedback (user_id, reason) VALUES (:user_id, 'bogus')"),
            {"user_id": user.id},
        )


def test_unknown_word_source_is_rejected_by_the_database(db_session: Session):
    with pytest.raises(IntegrityError, match="ck_words_word_source"):
        db_session.execute(
            text(
                "INSERT INTO words (label_en, korean, romanization, topik_level, source) "
                "VALUES ('cup', '컵', 'keop', 1, 'gpt')"
            )
        )


def test_deleting_a_user_deletes_their_photos_cards_and_feedback(db_session: Session):
    user, word = make_user(db_session), make_word(db_session)
    photo = make_photo(db_session, user, word)
    db_session.add_all(
        [
            UserCard(user_id=user.id, word_id=word.id, photo_id=photo.id),
            Feedback(user_id=user.id, photo_id=photo.id, reason=FeedbackReason.OTHER),
        ]
    )
    db_session.flush()

    db_session.execute(delete(User).where(User.id == user.id))

    assert [count(db_session, m) for m in (Photo, UserCard, Feedback)] == [0, 0, 0]
    # Words are shared across users, so they survive.
    assert count(db_session, Word) == 1


def test_deleting_a_photo_keeps_the_card_and_feedback(db_session: Session):
    user, word = make_user(db_session), make_word(db_session)
    photo = make_photo(db_session, user, word)
    card = UserCard(user_id=user.id, word_id=word.id, photo_id=photo.id)
    feedback = Feedback(user_id=user.id, photo_id=photo.id, reason=FeedbackReason.WRONG_OBJECT)
    db_session.add_all([card, feedback])
    db_session.flush()

    db_session.execute(delete(Photo).where(Photo.id == photo.id))
    db_session.refresh(card)
    db_session.refresh(feedback)

    assert card.photo_id is None
    assert feedback.photo_id is None


def test_deleting_a_word_in_someones_deck_is_refused(db_session: Session):
    user, word = make_user(db_session), make_word(db_session)
    db_session.add(UserCard(user_id=user.id, word_id=word.id))
    db_session.flush()

    with pytest.raises(IntegrityError, match="fk_user_cards_word_id_words"):
        db_session.execute(delete(Word).where(Word.id == word.id))
