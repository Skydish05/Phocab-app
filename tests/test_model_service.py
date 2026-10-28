import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from model_service.classifier import (
    Prediction,
    load_labels,
    prepare_image,
    top_k_predictions,
)
from model_service.main import Settings, create_app


class FakeClassifier:
    """Stands in for SiglipClassifier so tests never download model weights."""

    model_id = "fake/model"

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, int], int]] = []

    def classify(self, image: Image.Image, top_k: int = 3) -> list[Prediction]:
        self.calls.append((image.size, top_k))
        return [Prediction("cup", 0.9), Prediction("bottle", 0.05), Prediction("bowl", 0.01)][
            :top_k
        ]


def image_bytes(size=(32, 16), fmt="PNG", exif=None) -> bytes:
    buf = io.BytesIO()
    kwargs = {"exif": exif} if exif is not None else {}
    Image.new("RGB", size, "red").save(buf, format=fmt, **kwargs)
    return buf.getvalue()


@pytest.fixture
def fake():
    return FakeClassifier()


@pytest.fixture
def client(fake):
    # `with` runs the lifespan, which is where the classifier gets loaded.
    with TestClient(create_app(classifier_factory=lambda: fake)) as c:
        yield c


def test_bundled_labels_file_is_valid():
    labels = load_labels(Settings().labels_path)

    assert len(labels) >= 300


def test_load_labels_rejects_duplicates(tmp_path: Path):
    path = tmp_path / "labels.csv"
    path.write_text("label_en,korean,romanization,topik_level\ncup,컵,keop,1\ncup,컵,keop,1\n")

    with pytest.raises(ValueError, match="duplicate"):
        load_labels(path)


def test_load_labels_rejects_blank_label(tmp_path: Path):
    path = tmp_path / "labels.csv"
    path.write_text("label_en,korean,romanization,topik_level\n ,컵,keop,1\n")

    with pytest.raises(ValueError, match="blank"):
        load_labels(path)


def test_top_k_predictions_sorted_by_score_and_truncated():
    result = top_k_predictions(["a", "b", "c", "d"], [0.1, 0.7, 0.3, 0.5], k=3)

    assert result == [Prediction("b", 0.7), Prediction("d", 0.5), Prediction("c", 0.3)]


def test_prepare_image_applies_exif_rotation():
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90° clockwise, as phones write it

    image = prepare_image(image_bytes(size=(32, 16), fmt="JPEG", exif=exif))

    assert image.size == (16, 32)
    assert image.mode == "RGB"


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_classify_returns_top_labels(client, fake):
    response = client.post("/classify", files={"file": ("cup.png", image_bytes(), "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == "fake/model"
    assert [p["label_en"] for p in body["predictions"]] == ["cup", "bottle", "bowl"]
    assert body["latency_ms"] >= 0
    assert fake.calls == [((32, 16), 3)]


def test_classify_passes_top_k(client, fake):
    response = client.post(
        "/classify?top_k=1", files={"file": ("cup.png", image_bytes(), "image/png")}
    )

    assert [p["label_en"] for p in response.json()["predictions"]] == ["cup"]
    assert fake.calls[-1][1] == 1


def test_classify_rejects_non_image(client, fake):
    response = client.post("/classify", files={"file": ("notes.txt", b"hello", "text/plain")})

    assert response.status_code == 400
    assert fake.calls == []


def test_classify_rejects_out_of_range_top_k(client):
    response = client.post(
        "/classify?top_k=0", files={"file": ("cup.png", image_bytes(), "image/png")}
    )

    assert response.status_code == 422
