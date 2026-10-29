from collections.abc import Callable

import httpx2
import pytest

from app.services.recognizer import Recognition, RecognitionError, Recognizer

OK_BODY = {
    "predictions": [{"label_en": "cup", "score": 0.91}, {"label_en": "mug", "score": 0.4}],
    "model_id": "google/siglip2-base-patch16-224",
    "latency_ms": 312.6,
}


def recognizer(handler: Callable[[httpx2.Request], httpx2.Response]) -> Recognizer:
    transport = httpx2.MockTransport(handler)
    return Recognizer("unused", httpx2.Client(transport=transport, base_url="http://model"))


def test_returns_the_top_prediction():
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=OK_BODY)

    result = recognizer(handler).recognize(b"jpeg-bytes")

    assert result == Recognition("cup", 0.91, "google/siglip2-base-patch16-224", 313)
    assert seen[0].url.path == "/classify"
    assert b"jpeg-bytes" in seen[0].read()  # sent as the multipart "file" field


def raise_connect_error(request: httpx2.Request) -> httpx2.Response:
    raise httpx2.ConnectError("connection refused", request=request)


@pytest.mark.parametrize(
    "handler",
    [
        lambda request: httpx2.Response(500, text="boom"),
        lambda request: httpx2.Response(200, text="not json"),
        lambda request: httpx2.Response(200, json=OK_BODY | {"predictions": []}),
        raise_connect_error,
    ],
    ids=["http-500", "not-json", "no-predictions", "unreachable"],
)
def test_model_service_failures_raise_recognition_error(handler):
    with pytest.raises(RecognitionError):
        recognizer(handler).recognize(b"jpeg-bytes")
