import io
import json
from pathlib import Path
from types import SimpleNamespace

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.deps import get_llm, get_recognizer, get_storage, get_vocabulary
from app.models import Photo, UserCard, Word
from app.services.cache import load_vocabulary
from app.services.card_generator import LLMError, Message, VocabEntry
from app.services.images import MAX_UPLOAD_BYTES
from app.services.recognizer import Recognition, RecognitionError
from app.services.storage import PhotoStorage

BUCKET = "phocab-test-photos"
REGION = "ap-northeast-2"
# A small fixed vocabulary, so these tests don't break when labels_ko.csv is edited.
VOCAB = {
    e.label_en: e
    for e in (
        VocabEntry("cup", "컵", "keop", 1),
        VocabEntry("id card", "신분증", "sinbunjeung", 2),
    )
}
SENTENCES = {"컵": "컵에 물이 있어요.", "신분증": "신분증을 보여 주세요."}


class FakeRecognizer:
    def __init__(self) -> None:
        self.label = "cup"
        self.confidence = 0.87
        # The runner-up guesses after the top label; "tire" isn't in VOCAB.
        self.others: tuple[tuple[str, float], ...] = (("ID card", 0.0002), ("tire", 0.0001))
        self.error: Exception | None = None
        self.seen: list[bytes] = []

    def recognize(self, jpeg: bytes) -> Recognition:
        self.seen.append(jpeg)
        if self.error:
            raise self.error
        return Recognition(
            self.label,
            confidence=self.confidence,
            model_id="fake-siglip",
            latency_ms=42,
            candidates=((self.label, self.confidence), *self.others),
        )


class FakeLLM:
    """Replies with a valid sentence for whichever word it was asked about."""

    def __init__(self) -> None:
        self.calls = 0
        self.reply: str | None = None  # set to force a specific (e.g. invalid) reply
        self.error: Exception | None = None

    async def complete(self, messages: list[Message]) -> str:
        self.calls += 1
        if self.error:
            raise self.error
        if self.reply is not None:
            return self.reply
        word = next(w for w in SENTENCES if f"Korean word: {w}" in messages[1]["content"])
        return json.dumps({"example": SENTENCES[word]}, ensure_ascii=False)


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION}
        )
        yield client


@pytest.fixture
def fakes(client: TestClient, s3) -> SimpleNamespace:
    fakes = SimpleNamespace(
        recognizer=FakeRecognizer(), llm=FakeLLM(), storage=PhotoStorage(BUCKET, s3), s3=s3
    )
    overrides = client.app.dependency_overrides
    overrides[get_recognizer] = lambda: fakes.recognizer
    overrides[get_llm] = lambda: fakes.llm
    overrides[get_storage] = lambda: fakes.storage
    overrides[get_vocabulary] = lambda: VOCAB
    return fakes


def auth_headers(client: TestClient, email: str = "learner@example.com") -> dict[str, str]:
    credentials = {"email": email, "password": "correct horse battery"}
    client.post("/auth/register", json=credentials)
    token = client.post("/auth/login", json=credentials).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def photo_bytes(size: tuple[int, int] = (640, 480), fmt: str = "JPEG") -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, "white").save(out, format=fmt)
    return out.getvalue()


def upload(client: TestClient, headers: dict[str, str], data: bytes | None = None):
    files = {"file": ("photo.jpg", photo_bytes() if data is None else data, "image/jpeg")}
    return client.post("/cards/from-photo", files=files, headers=headers)


def count(db: Session, model) -> int:
    return db.scalar(select(func.count()).select_from(model))


def stored_keys(s3) -> list[str]:
    return [obj["Key"] for obj in s3.list_objects_v2(Bucket=BUCKET).get("Contents", [])]


def test_cache_miss_generates_the_card_and_saves_everything(
    client: TestClient, fakes: SimpleNamespace, db_session: Session
):
    response = upload(client, auth_headers(client))

    assert response.status_code == 201
    body = response.json()
    assert body | {"card_id": None, "photo_id": None} == {
        "card_id": None,
        "photo_id": None,
        "label_en": "cup",
        "korean": "컵",
        "romanization": "keop",
        "topik_level": 1,
        "example": "컵에 물이 있어요.",
        "confidence": 0.87,
    }
    assert fakes.llm.calls == 1
    card = db_session.get(UserCard, body["card_id"])
    photo = db_session.get(Photo, card.photo_id)
    assert (photo.confidence, photo.model_used, photo.latency_ms) == (0.87, "fake-siglip", 42)
    assert photo.word_id == card.word_id
    assert photo.s3_key.startswith(f"photos/{card.user_id}/")
    assert stored_keys(fakes.s3) == [photo.s3_key]


def test_cache_hit_skips_the_llm(client: TestClient, fakes: SimpleNamespace, db_session: Session):
    first = upload(client, auth_headers(client, "a@example.com"))
    second = upload(client, auth_headers(client, "b@example.com"))

    assert first.status_code == second.status_code == 201
    assert second.json()["example"] == first.json()["example"]
    assert fakes.llm.calls == 1
    assert count(db_session, Word) == 1
    assert count(db_session, UserCard) == 2


def test_cached_word_with_outdated_vocabulary_is_regenerated(
    client: TestClient, fakes: SimpleNamespace, db_session: Session
):
    # Cached before the CSV changed 학생증 to 신분증.
    db_session.add(
        Word(
            label_en="id card",
            korean="학생증",
            romanization="haksaengjeung",
            topik_level=2,
            example="학생증이 있어요.",
        )
    )
    db_session.flush()
    fakes.recognizer.label = "ID card"  # as the model returns it; the cache key is normalized

    body = upload(client, auth_headers(client)).json()

    assert (body["korean"], body["example"]) == ("신분증", "신분증을 보여 주세요.")
    assert fakes.llm.calls == 1
    assert count(db_session, Word) == 1


def test_same_object_twice_returns_the_existing_card(
    client: TestClient, fakes: SimpleNamespace, db_session: Session
):
    headers = auth_headers(client)

    first = upload(client, headers)
    second = upload(client, headers)

    assert (first.status_code, second.status_code) == (201, 200)
    assert second.json()["card_id"] == first.json()["card_id"]
    assert count(db_session, UserCard) == 1
    # Every upload is still recorded, for evaluation.
    assert count(db_session, Photo) == 2


def test_htmx_upload_gets_the_card_as_an_html_fragment(
    client: TestClient, fakes: SimpleNamespace, db_session: Session
):
    headers = auth_headers(client) | {"HX-Request": "true"}

    first = upload(client, headers)
    second = upload(client, headers)

    # Same status codes as the JSON API, so both clients see one contract.
    assert (first.status_code, second.status_code) == (201, 200)
    assert first.headers["content-type"].startswith("text/html")
    for text in ("컵", "keop", "cup", "TOPIK 1", "컵에 물이 있어요.", "This is wrong"):
        assert text in first.text
    # The raw score isn't shown: a correct guess can score 0.01 and read as "1%".
    assert "87%" not in first.text
    photo_id = db_session.scalars(select(Photo.id).order_by(Photo.id)).first()
    assert f'name="photo_id" value="{photo_id}"' in first.text
    # Listen buttons for the word and the sentence.
    assert 'data-speak="컵"' in first.text
    assert 'data-speak="컵에 물이 있어요."' in first.text
    assert "Already in your deck" not in first.text
    assert "Already in your deck" in second.text


def test_htmx_errors_stay_json_for_the_page_to_show(client: TestClient, fakes: SimpleNamespace):
    headers = auth_headers(client) | {"HX-Request": "true"}

    response = upload(client, headers, data=b"not an image")

    assert response.status_code == 400
    assert "detail" in response.json()


def test_the_resized_jpeg_is_what_gets_stored_and_classified(
    client: TestClient, fakes: SimpleNamespace
):
    upload(client, auth_headers(client), data=photo_bytes((3000, 2000), "PNG"))

    stored = fakes.s3.get_object(Bucket=BUCKET, Key=stored_keys(fakes.s3)[0])
    data = stored["Body"].read()
    assert stored["ContentType"] == "image/jpeg"
    assert Image.open(io.BytesIO(data)).size == (1024, 683)
    assert fakes.recognizer.seen == [data]


@pytest.mark.parametrize(
    "data, status",
    [
        (b"definitely not an image", 400),
        (photo_bytes(fmt="GIF"), 400),
        (b"\0" * (MAX_UPLOAD_BYTES + 1), 413),
    ],
    ids=["not-an-image", "gif", "over-5mb"],
)
def test_bad_uploads_are_rejected_before_any_work(
    client: TestClient, fakes: SimpleNamespace, data: bytes, status: int
):
    response = upload(client, auth_headers(client), data=data)

    assert response.status_code == status
    assert fakes.recognizer.seen == []
    assert stored_keys(fakes.s3) == []


def test_upload_requires_login(client: TestClient, fakes: SimpleNamespace):
    assert upload(client, headers={}).status_code == 401


def break_recognizer(fakes):
    fakes.recognizer.error = RecognitionError("model service down")


def return_unknown_label(fakes):
    fakes.recognizer.label = "spaceship"


def return_invalid_llm_replies(fakes):
    fakes.llm.reply = "not json"


def break_llm(fakes):
    fakes.llm.error = LLMError("gemini down")


@pytest.mark.parametrize(
    "break_something",
    [break_recognizer, return_unknown_label, return_invalid_llm_replies, break_llm],
)
def test_upstream_failures_return_502_and_store_nothing(
    client: TestClient, fakes: SimpleNamespace, db_session: Session, break_something
):
    headers = auth_headers(client)
    break_something(fakes)

    response = upload(client, headers)

    assert response.status_code == 502
    assert stored_keys(fakes.s3) == []
    assert count(db_session, Photo) == count(db_session, UserCard) == count(db_session, Word) == 0


def test_storage_failure_returns_502_but_keeps_the_generated_word(
    client: TestClient, fakes: SimpleNamespace, db_session: Session
):
    fakes.storage = PhotoStorage("bucket-that-does-not-exist", fakes.s3)

    response = upload(client, auth_headers(client))

    assert response.status_code == 502
    assert count(db_session, Photo) == count(db_session, UserCard) == 0
    # The LLM call was already paid for, so its result stays cached.
    assert count(db_session, Word) == 1


def test_db_failure_after_upload_deletes_the_photo(
    client: TestClient, fakes: SimpleNamespace, db_session: Session, monkeypatch
):
    headers = auth_headers(client)
    upload(client, headers)  # warm the cache, so the failing request's only commit is the last
    assert len(stored_keys(fakes.s3)) == 1

    def failing_commit():
        raise RuntimeError("database went away")

    monkeypatch.setattr(db_session, "commit", failing_commit)
    with pytest.raises(RuntimeError, match="database went away"):
        upload(client, headers)

    assert len(stored_keys(fakes.s3)) == 1  # only the first request's photo


def test_vocabulary_labels_are_normalized(tmp_path: Path):
    csv_file = tmp_path / "labels.csv"
    csv_file.write_text(
        "label_en,korean,romanization,topik_level\n ID  Card ,신분증,sinbunjeung,2\n",
        encoding="utf-8",
    )

    assert load_vocabulary(csv_file) == {
        "id card": VocabEntry("id card", "신분증", "sinbunjeung", 2)
    }


def test_labels_that_collide_after_normalizing_are_rejected(tmp_path: Path):
    csv_file = tmp_path / "labels.csv"
    csv_file.write_text(
        "label_en,korean,romanization,topik_level\nCup,컵,keop,1\ncup,컵,keop,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate label"):
        load_vocabulary(csv_file)


def test_the_real_vocabulary_file_loads():
    vocab = load_vocabulary(get_settings().vocab_path)

    assert vocab["cup"] == VocabEntry("cup", "컵", "keop", 1)


# --- "Couldn't recognize this": top score below RECOGNITION_MIN_SCORE (0.001) ---


def test_low_score_makes_no_card_and_offers_the_guesses(
    client: TestClient, fakes: SimpleNamespace, db_session: Session
):
    fakes.recognizer.confidence = 0.00003  # what an off-list object scores

    response = upload(client, auth_headers(client))

    assert response.status_code == 422
    body = response.json()
    photo = db_session.get(Photo, body["photo_id"])
    assert body == {
        "detail": "Couldn't recognize this object",
        "photo_id": photo.id,
        # "tire" isn't in the vocabulary, so it isn't offered.
        "candidates": [
            {"label_en": "cup", "korean": "컵", "romanization": "keop", "score": 0.00003},
            {
                "label_en": "id card",
                "korean": "신분증",
                "romanization": "sinbunjeung",
                "score": 0.0002,
            },
        ],
    }
    # No card and no LLM call, but the photo is kept (eval data, and for the pick).
    assert (count(db_session, UserCard), fakes.llm.calls) == (0, 0)
    assert (photo.word_id, photo.confidence) == (None, 0.00003)
    assert stored_keys(fakes.s3) == [photo.s3_key]


def test_score_at_the_threshold_still_makes_a_card(client: TestClient, fakes: SimpleNamespace):
    fakes.recognizer.confidence = 0.001

    assert upload(client, auth_headers(client)).status_code == 201


def test_threshold_comes_from_settings(client: TestClient, fakes: SimpleNamespace):
    strict = get_settings().model_copy(update={"recognition_min_score": 0.9})
    client.app.dependency_overrides[get_settings] = lambda: strict

    assert upload(client, auth_headers(client)).status_code == 422  # 0.87 < 0.9


def test_htmx_low_score_shows_the_guesses_as_buttons(client: TestClient, fakes: SimpleNamespace):
    fakes.recognizer.confidence = 0.00003

    response = upload(client, auth_headers(client) | {"HX-Request": "true"})

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("text/html")
    assert "Couldn't recognize this" in response.text
    assert "컵" in response.text and "신분증" in response.text and "None of these" in response.text
    assert response.text.count('hx-post="/photos/') == 2


def unrecognized_photo_id(client: TestClient, fakes: SimpleNamespace, headers) -> int:
    fakes.recognizer.confidence = 0.00003
    return upload(client, headers).json()["photo_id"]


def test_picking_a_word_makes_the_card(
    client: TestClient, fakes: SimpleNamespace, db_session: Session
):
    headers = auth_headers(client)
    photo_id = unrecognized_photo_id(client, fakes, headers)

    response = client.post(
        f"/photos/{photo_id}/label", json={"label_en": "ID card"}, headers=headers
    )

    assert response.status_code == 201
    body = response.json()
    assert (body["korean"], body["example"], body["confidence"]) == (
        "신분증",
        SENTENCES["신분증"],
        None,
    )
    photo = db_session.get(Photo, photo_id)
    card = db_session.get(UserCard, body["card_id"])
    assert photo.word_id == card.word_id and card.photo_id == photo_id
    # Marked as the user's label, so evaluation doesn't count it as a model prediction.
    assert photo.model_used == "user-picked:fake-siglip"
    assert fakes.llm.calls == 1


def test_htmx_pick_returns_the_card_fragment(client: TestClient, fakes: SimpleNamespace):
    headers = auth_headers(client)
    photo_id = unrecognized_photo_id(client, fakes, headers)

    response = client.post(
        f"/photos/{photo_id}/label",
        json={"label_en": "cup"},
        headers=headers | {"HX-Request": "true"},
    )

    assert response.status_code == 201
    assert "컵" in response.text and "This is wrong" in response.text


def test_a_photo_can_be_labeled_only_once(client: TestClient, fakes: SimpleNamespace):
    headers = auth_headers(client)
    photo_id = unrecognized_photo_id(client, fakes, headers)

    first = client.post(f"/photos/{photo_id}/label", json={"label_en": "cup"}, headers=headers)
    second = client.post(f"/photos/{photo_id}/label", json={"label_en": "ID card"}, headers=headers)

    assert (first.status_code, second.status_code) == (201, 409)


def test_a_recognized_photo_cannot_be_relabeled(client: TestClient, fakes: SimpleNamespace):
    headers = auth_headers(client)
    photo_id = upload(client, headers).json()["photo_id"]  # recognized as cup

    response = client.post(
        f"/photos/{photo_id}/label", json={"label_en": "ID card"}, headers=headers
    )

    assert response.status_code == 409


def test_cannot_label_someone_elses_photo(client: TestClient, fakes: SimpleNamespace):
    photo_id = unrecognized_photo_id(client, fakes, auth_headers(client, "a@example.com"))

    response = client.post(
        f"/photos/{photo_id}/label",
        json={"label_en": "cup"},
        headers=auth_headers(client, "b@example.com"),
    )

    assert response.status_code == 404


def test_picking_a_word_outside_the_vocabulary_is_rejected(
    client: TestClient, fakes: SimpleNamespace, db_session: Session
):
    headers = auth_headers(client)
    photo_id = unrecognized_photo_id(client, fakes, headers)

    response = client.post(f"/photos/{photo_id}/label", json={"label_en": "tire"}, headers=headers)

    assert response.status_code == 422
    assert db_session.get(Photo, photo_id).word_id is None
    assert count(db_session, UserCard) == 0
