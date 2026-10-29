from types import SimpleNamespace

import pytest
from google.genai import errors

from app.services.card_generator import LLMError, Message
from app.services.llm import GeminiClient

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeModels:
    """Stands in for the SDK's client.aio.models and records each request."""

    def __init__(self, text: str | None = "{}", error: Exception | None = None) -> None:
        self.text, self.error = text, error
        self.requests: list[dict] = []

    async def generate_content(self, **kwargs) -> SimpleNamespace:
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.text)


def gemini(models: FakeModels) -> GeminiClient:
    return GeminiClient(
        "unused", "test-model", client=SimpleNamespace(aio=SimpleNamespace(models=models))
    )


MESSAGES: list[Message] = [
    {"role": "system", "content": "be a flashcard writer"},
    {"role": "user", "content": "Korean word: 컵"},
    {"role": "assistant", "content": "bad reply"},
    {"role": "user", "content": "fix it"},
]


async def test_messages_are_mapped_to_gemini_roles_and_config():
    models = FakeModels(text='{"example": "컵이 있어요."}')

    reply = await gemini(models).complete(MESSAGES)

    assert reply == '{"example": "컵이 있어요."}'
    request = models.requests[0]
    assert request["model"] == "test-model"
    assert request["config"].system_instruction == "be a flashcard writer"
    assert request["config"].response_mime_type == "application/json"
    # The system prompt is config, not a turn; "assistant" becomes "model".
    assert [(c.role, c.parts[0].text) for c in request["contents"]] == [
        ("user", "Korean word: 컵"),
        ("model", "bad reply"),
        ("user", "fix it"),
    ]


async def test_empty_or_blocked_reply_becomes_empty_string():
    assert await gemini(FakeModels(text=None)).complete(MESSAGES) == ""


async def test_api_error_becomes_llm_error():
    error = errors.ServerError(503, {"error": {"message": "overloaded", "status": "UNAVAILABLE"}})

    with pytest.raises(LLMError):
        await gemini(FakeModels(error=error)).complete(MESSAGES)
