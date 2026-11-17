from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import User, UserCard, Word
from app.templating import due_label

NOW = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)


def login(client: TestClient, db: Session, email: str) -> tuple[dict[str, str], User]:
    credentials = {"email": email, "password": "correct horse battery"}
    client.post("/auth/register", json=credentials)
    token = client.post("/auth/login", json=credentials).json()["access_token"]
    user = db.scalars(select(User).where(User.email == email)).one()
    return {"Authorization": f"Bearer {token}"}, user


def add_card(db: Session, user: User, label: str, minutes_ago: int = 0, box: int = 1) -> UserCard:
    word = db.scalars(select(Word).where(Word.label_en == label)).one_or_none()
    if word is None:
        word = Word(
            label_en=label, korean=f"{label}-ko", romanization=f"{label}-rom", topik_level=1
        )
        db.add(word)
        db.flush()
    created = NOW - timedelta(minutes=minutes_ago)
    card = UserCard(
        user_id=user.id, word_id=word.id, box=box, created_at=created, next_review_at=created
    )
    db.add(card)
    db.flush()
    return card


def test_deck_lists_only_my_cards_newest_first(client: TestClient, db_session: Session):
    headers, me = login(client, db_session, "me@example.com")
    _, other = login(client, db_session, "other@example.com")
    add_card(db_session, me, "cup", minutes_ago=10)
    add_card(db_session, me, "banana", minutes_ago=5)
    add_card(db_session, other, "cup")

    body = client.get("/cards", headers=headers).json()

    assert body["total"] == 2
    assert [c["label_en"] for c in body["items"]] == ["banana", "cup"]
    assert body["items"][0] | {"card_id": None, "next_review_at": None, "created_at": None} == {
        "card_id": None,
        "label_en": "banana",
        "korean": "banana-ko",
        "romanization": "banana-rom",
        "topik_level": 1,
        "example": None,
        "box": 1,
        "next_review_at": None,
        "created_at": None,
    }


def test_deck_pages_and_filters_by_box(client: TestClient, db_session: Session):
    headers, me = login(client, db_session, "me@example.com")
    for i, label in enumerate(["a", "b", "c", "d", "e"]):
        add_card(db_session, me, label, minutes_ago=i, box=2 if label in "bd" else 1)

    page = client.get("/cards?limit=2&offset=2", headers=headers).json()
    box_2 = client.get("/cards?box=2", headers=headers).json()

    # Newest first: a (0 min ago), b, c, d, e.
    assert ([c["label_en"] for c in page["items"]], page["total"]) == (["c", "d"], 5)
    assert ([c["label_en"] for c in box_2["items"]], box_2["total"]) == (["b", "d"], 2)


@pytest.mark.parametrize("query", ["limit=0", "limit=101", "offset=-1", "box=0", "box=6"])
def test_deck_rejects_out_of_range_paging(client: TestClient, db_session: Session, query: str):
    headers, _ = login(client, db_session, "me@example.com")

    assert client.get(f"/cards?{query}", headers=headers).status_code == 422


def test_deck_requires_login(client: TestClient):
    assert client.get("/cards").status_code == 401


def test_htmx_deck_is_html_with_a_show_more_button(client: TestClient, db_session: Session):
    headers, me = login(client, db_session, "me@example.com")
    for i, label in enumerate(["a", "b", "c"]):
        add_card(db_session, me, label, minutes_ago=i)
    headers |= {"HX-Request": "true"}

    first = client.get("/cards?limit=2", headers=headers)
    last = client.get("/cards?limit=2&offset=2", headers=headers)

    assert first.headers["content-type"].startswith("text/html")
    assert "3 cards" in first.text and "a-ko" in first.text and "b-ko" in first.text
    assert 'hx-get="/cards?offset=2&limit=2"' in first.text
    # A later page is just more list items: no header, and no button after the last page.
    assert "c-ko" in last.text and "3 cards" not in last.text and "Show more" not in last.text


def test_htmx_show_more_keeps_the_box_filter(client: TestClient, db_session: Session):
    headers, me = login(client, db_session, "me@example.com")
    for label in ["a", "b"]:
        add_card(db_session, me, label, box=3)

    response = client.get("/cards?limit=1&box=3", headers=headers | {"HX-Request": "true"})

    assert 'hx-get="/cards?offset=1&limit=1&box=3"' in response.text


def test_htmx_empty_deck_says_so(client: TestClient, db_session: Session):
    headers, _ = login(client, db_session, "me@example.com")

    response = client.get("/cards", headers=headers | {"HX-Request": "true"})

    assert "Your deck is empty" in response.text


@pytest.mark.parametrize(
    ("delta", "label"),
    [
        (timedelta(minutes=-1), "due now"),
        (timedelta(0), "due now"),
        (timedelta(minutes=30), "due in 1 h"),
        (timedelta(hours=5), "due in 5 h"),
        (timedelta(days=1), "due tomorrow"),
        (timedelta(days=3, hours=1), "due in 4 days"),
    ],
)
def test_due_label(delta: timedelta, label: str):
    assert due_label(NOW + delta, now=NOW) == label
