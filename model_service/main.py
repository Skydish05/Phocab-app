import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from PIL import UnidentifiedImageError
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from model_service.classifier import Classifier, SiglipClassifier, load_labels, prepare_image


class Settings(BaseSettings):
    # protected_namespaces=() because pydantic reserves the "model_" prefix,
    # which would otherwise warn on model_id.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    # Apache-2.0 on the Hub. openai/clip-* has no license tag and its model card
    # puts deployed use out of scope, which rules it out for a deployed app.
    model_id: str = "google/siglip2-base-patch16-224"
    labels_path: Path = Path(__file__).parent / "labels_ko.csv"


class LabelScore(BaseModel):
    label_en: str
    score: float


class ClassifyResponse(BaseModel):
    predictions: list[LabelScore]
    model_id: str
    # Measured here, not by the caller, so it excludes network and S3 time.
    # The API stores it in photos.latency_ms.
    latency_ms: float


def build_classifier() -> Classifier:
    settings = Settings()
    return SiglipClassifier(settings.model_id, load_labels(settings.labels_path))


def create_app(classifier_factory: Callable[[], Classifier] = build_classifier) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Load once at startup so a slow or broken model fails the container on
        # boot instead of on a user's first request.
        app.state.classifier = classifier_factory()
        yield

    app = FastAPI(title="Phocab model service", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # A plain `def` (not async) so FastAPI runs it in a worker thread and the
    # CPU-bound inference doesn't block the event loop.
    @app.post("/classify")
    def classify(
        request: Request, file: UploadFile, top_k: int = Query(3, ge=1, le=10)
    ) -> ClassifyResponse:
        classifier: Classifier = request.app.state.classifier
        start = time.perf_counter()
        try:
            image = prepare_image(file.file.read())
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(status_code=400, detail="File is not a readable image") from exc
        predictions = classifier.classify(image, top_k=top_k)
        return ClassifyResponse(
            predictions=[LabelScore(label_en=p.label_en, score=p.score) for p in predictions],
            model_id=classifier.model_id,
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    return app


app = create_app()
