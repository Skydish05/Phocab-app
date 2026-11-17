from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.deps import get_now
from app.models import UserCard
from tests.test_deck import NOW, add_card, login


@pytest.fixture(autouse=True)
def frozen_clock(client: TestClient):
    client.app.dependency_overrides[get_now] = lambda: NOW


def make_due(card: UserCard, db: Session, delta: timedelta) -> UserCard:
    card.next_review_at = NOW + delta
    db.flush()
    return card


def test_next_is_the_most_overdue_of_my_cards(client: TestClient, db_session: Session):
    headers, me = login(client, db_session, "me@example.com")
    _, other = login(client, db_session, "other@example.com")
    make_due(add_card(db_session, me, "cup"), db_session, timedelta(hours=-1))
    make_due(add_card(db_session, me, "banana"), db_session, timedelta(days=-2))
    make_due(add_card(db_session, me, "balloon"), db_session, timedelta(days=1))
    make_due(add_card(db_session, other, "dog"), db_session, timedelta(days=-9))

    body = client.get("/review/next", headers=headers).json()

    assert body["card"]["label_en"] == "banana"
    assert body["due_count"] == 2


def test_next_when_nothing_is_due_says_when(client: TestClient, db_session: Session):
    headers, me = login(client, db_session, "me@example.com")
    make_due(add_card(db_session, me, "cup"), db_session, timedelta(days=4))
    make_due(add_card(db_session, me, "banana"), db_session, timedelta(days=1))

    body = client.get("/review/next", headers=headers).json()

    assert (body["card"], body["due_count"]) == (None, 0)
    assert body["next_due_at"].startswith((NOW + timedelta(days=1)).date().isoformat())


def test_next_on_an_empty_deck(client: TestClient, db_session: Session):
    headers, _ = login(client, db_session, "me@example.com")

    assert client.get("/review/next", headers=headers).json() == {
        "card": None,
        "due_count": 0,
        "next_due_at": None,
    }


@pytest.mark.parametrize(("correct", "box", "counts"), [(True, 3, (1, 0)), (False, 1, (0, 1))])
def test_answer_updates_box_due_date_and_counts(
    client: TestClient, db_session: Session, correct: bool, box: int, counts: tuple[int, int]
):
    headers, me = login(client, db_session, "me@example.com")
    card = make_due(add_card(db_session, me, "cup", box=2), db_session, timedelta(0))

    response = client.post(f"/review/{card.id}", json={"correct": correct}, headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert (body["box"], body["times_correct"], body["times_wrong"]) == (box, *counts)
    db_session.refresh(card)
    # The same schedule the Leitner unit tests pin down, applied at the frozen clock.
    expected_days = {3: 4, 1: 1}[box]
    assert (card.box, card.next_review_at) == (box, NOW + timedelta(days=expected_days))


def test_answer_for_a_card_not_yet_due_is_rejected(client: TestClient, db_session: Session):
    headers, me = login(client, db_session, "me@example.com")
    card = make_due(add_card(db_session, me, "cup"), db_session, timedelta(0))

    first = client.post(f"/review/{card.id}", json={"correct": True}, headers=headers)
    # A double tap: the first answer already pushed the due date into the future.
    second = client.post(f"/review/{card.id}", json={"correct": True}, headers=headers)

    assert (first.status_code, second.status_code) == (200, 409)
    db_session.refresh(card)
    assert (card.box, card.times_correct) == (2, 1)


def test_cannot_review_someone_elses_card(client: TestClient, db_session: Session):
    headers, _ = login(client, db_session, "me@example.com")
    _, other = login(client, db_session, "other@example.com")
    theirs = make_due(add_card(db_session, other, "cup"), db_session, timedelta(0))

    response = client.post(f"/review/{theirs.id}", json={"correct": True}, headers=headers)

    assert response.status_code == 404
    db_session.refresh(theirs)
    assert theirs.box == 1


def test_review_requires_login(client: TestClient):
    assert client.get("/review/next").status_code == 401
    assert client.post("/review/1", json={"correct": True}).status_code == 401


def test_htmx_review_shows_the_front_and_triggers_a_reload(client: TestClient, db_session: Session):
    headers, me = login(client, db_session, "me@example.com")
    card = make_due(add_card(db_session, me, "cup"), db_session, timedelta(0))
    headers |= {"HX-Request": "true"}

    page = client.get("/review/next", headers=headers)
    answer = client.post(f"/review/{card.id}", json={"correct": True}, headers=headers)
    done = client.get("/review/next", headers=headers)

    assert page.headers["content-type"].startswith("text/html")
    assert "1 card due" in page.text and "cup" in page.text
    assert f'hx-post="/review/{card.id}"' in page.text
    assert answer.headers["HX-Trigger"] == "card-reviewed"
    assert "All done for now" in done.text and "due in 2 days" in done.text


def test_json_answer_has_no_htmx_header(client: TestClient, db_session: Session):
    headers, me = login(client, db_session, "me@example.com")
    card = make_due(add_card(db_session, me, "cup"), db_session, timedelta(0))

    response = client.post(f"/review/{card.id}", json={"correct": True}, headers=headers)

    assert "HX-Trigger" not in response.headers
