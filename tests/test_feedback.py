from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Feedback, FeedbackReason, Photo, User, UserCard
from tests.test_deck import add_card, login


def add_photo(db: Session, user: User, card: UserCard | None = None) -> Photo:
    photo = Photo(
        user_id=user.id,
        s3_key=f"photos/{user.id}/{db.scalar(select(func.count()).select_from(Photo))}.jpg",
        word_id=card.word_id if card else None,
        confidence=0.4,
        model_used="fake-siglip",
        latency_ms=40,
    )
    db.add(photo)
    db.flush()
    return photo


def test_report_saves_feedback_with_the_photos_word(client: TestClient, db_session: Session):
    headers, me = login(client, db_session, "me@example.com")
    photo = add_photo(db_session, me, add_card(db_session, me, "cup"))

    response = client.post(
        "/feedback",
        json={"photo_id": photo.id, "reason": "wrong_object", "note": "it's a mug"},
        headers=headers,
    )

    assert response.status_code == 201
    saved = db_session.scalars(select(Feedback)).one()
    assert (saved.user_id, saved.photo_id, saved.word_id) == (me.id, photo.id, photo.word_id)
    assert (saved.reason, saved.note) == (FeedbackReason.WRONG_OBJECT, "it's a mug")
    assert response.json() == {
        "id": saved.id,
        "photo_id": photo.id,
        "word_id": photo.word_id,
        "reason": "wrong_object",
        "note": "it's a mug",
    }


def test_blank_note_is_stored_as_null(client: TestClient, db_session: Session):
    headers, me = login(client, db_session, "me@example.com")
    photo = add_photo(db_session, me)

    client.post(
        "/feedback", json={"photo_id": photo.id, "reason": "other", "note": "  "}, headers=headers
    )

    assert db_session.scalars(select(Feedback.note)).one() is None


def test_cannot_report_someone_elses_photo(client: TestClient, db_session: Session):
    headers, _ = login(client, db_session, "me@example.com")
    _, other = login(client, db_session, "other@example.com")
    theirs = add_photo(db_session, other)

    response = client.post(
        "/feedback", json={"photo_id": theirs.id, "reason": "other"}, headers=headers
    )

    assert response.status_code == 404
    assert db_session.scalar(select(func.count()).select_from(Feedback)) == 0


def test_report_rejects_unknown_reasons_and_long_notes(client: TestClient, db_session: Session):
    headers, me = login(client, db_session, "me@example.com")
    photo = add_photo(db_session, me)

    bad_reason = client.post(
        "/feedback", json={"photo_id": photo.id, "reason": "ugly"}, headers=headers
    )
    long_note = client.post(
        "/feedback",
        json={"photo_id": photo.id, "reason": "other", "note": "x" * 501},
        headers=headers,
    )

    assert (bad_reason.status_code, long_note.status_code) == (422, 422)


def test_htmx_report_offers_to_remove_the_card(client: TestClient, db_session: Session):
    headers, me = login(client, db_session, "me@example.com")
    card = add_card(db_session, me, "cup")
    photo = add_photo(db_session, me, card)

    response = client.post(
        "/feedback",
        # The page's form sends strings (json-enc); the schema still parses them.
        json={"photo_id": str(photo.id), "reason": "wrong_object", "note": ""},
        headers=headers | {"HX-Request": "true"},
    )

    assert response.status_code == 201
    assert "Thanks, reported" in response.text
    assert f'hx-delete="/cards/{card.id}"' in response.text


def test_htmx_report_without_a_card_has_no_remove_button(client: TestClient, db_session: Session):
    headers, me = login(client, db_session, "me@example.com")
    photo = add_photo(db_session, me)  # no word: nothing in the deck to remove

    response = client.post(
        "/feedback",
        json={"photo_id": photo.id, "reason": "wrong_object"},
        headers=headers | {"HX-Request": "true"},
    )

    assert "Thanks, reported" in response.text and "hx-delete" not in response.text


def test_remove_card_keeps_the_photo_and_feedback(client: TestClient, db_session: Session):
    headers, me = login(client, db_session, "me@example.com")
    card = add_card(db_session, me, "cup")
    photo = add_photo(db_session, me, card)
    client.post("/feedback", json={"photo_id": photo.id, "reason": "wrong_object"}, headers=headers)

    response = client.delete(f"/cards/{card.id}", headers=headers)

    assert response.status_code == 204
    assert db_session.get(UserCard, card.id) is None
    assert db_session.get(Photo, photo.id) is not None
    assert db_session.scalar(select(func.count()).select_from(Feedback)) == 1


def test_cannot_remove_someone_elses_card(client: TestClient, db_session: Session):
    headers, _ = login(client, db_session, "me@example.com")
    _, other = login(client, db_session, "other@example.com")
    theirs = add_card(db_session, other, "cup")

    response = client.delete(f"/cards/{theirs.id}", headers=headers)

    assert response.status_code == 404
    assert db_session.get(UserCard, theirs.id) is not None


def test_htmx_remove_returns_a_note_to_show(client: TestClient, db_session: Session):
    headers, me = login(client, db_session, "me@example.com")
    card = add_card(db_session, me, "cup")

    response = client.delete(f"/cards/{card.id}", headers=headers | {"HX-Request": "true"})

    assert response.status_code == 200
    assert "Removed from your deck" in response.text


def test_feedback_and_remove_require_login(client: TestClient):
    assert client.post("/feedback", json={"photo_id": 1, "reason": "other"}).status_code == 401
    assert client.delete("/cards/1").status_code == 401
