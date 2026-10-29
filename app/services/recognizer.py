"""HTTP client for model_service's POST /classify.

No VLM fallback yet: the confidence threshold for it gets tuned on the eval set
(PLAN.md days 9-10), so for now the top label is always used and its score stored.
"""

from dataclasses import dataclass

import httpx2
from pydantic import BaseModel, Field, ValidationError

# Classification on a 2-vCPU box is about 30x faster than this (see
# eval/results/bench_2vcpu.md); the slack covers a cold or busy model container.
# The short connect timeout makes "container is down" fail fast.
TIMEOUT = httpx2.Timeout(10.0, connect=2.0)


# Copied from model_service's response model rather than imported: the services
# deploy separately, and the API image doesn't ship model_service's code.
class _LabelScore(BaseModel):
    label_en: str
    score: float


class _ClassifyResponse(BaseModel):
    predictions: list[_LabelScore] = Field(min_length=1)
    model_id: str
    latency_ms: float


@dataclass(frozen=True)
class Recognition:
    label_en: str
    confidence: float
    model_id: str
    latency_ms: int


class RecognitionError(Exception):
    """The model service failed or replied with something unusable. Maps to a 502."""


class Recognizer:
    def __init__(self, base_url: str, client: httpx2.Client | None = None) -> None:
        # One long-lived client so requests reuse connections to the model service.
        self._client = client or httpx2.Client(base_url=base_url, timeout=TIMEOUT)

    def recognize(self, jpeg: bytes) -> Recognition:
        try:
            response = self._client.post(
                "/classify", files={"file": ("photo.jpg", jpeg, "image/jpeg")}
            )
            response.raise_for_status()
            body = _ClassifyResponse.model_validate_json(response.content)
        except (httpx2.HTTPError, ValidationError) as exc:
            raise RecognitionError(f"model service call failed: {exc}") from exc
        top = body.predictions[0]  # model_service returns them best first
        return Recognition(
            label_en=top.label_en,
            confidence=top.score,
            model_id=body.model_id,
            latency_ms=round(body.latency_ms),
        )
