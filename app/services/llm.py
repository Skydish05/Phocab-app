"""Google Gemini implementation of card_generator's LLMClient."""

from typing import Any

import httpx2
from google import genai
from google.genai import errors, types

from app.services.card_generator import LLMError, Message

# A one-sentence reply is quick; this bounds how long a stuck call holds the request.
TIMEOUT_MS = 20_000
# The SDK never retries unless asked. 3 attempts (with its exponential backoff)
# rides out a brief 429/5xx without making a user wait through a long outage.
RETRY = types.HttpRetryOptions(attempts=3)


class GeminiClient:
    def __init__(self, api_key: str, model: str, client: Any = None) -> None:
        # `client` lets tests pass a fake instead of the real SDK client.
        self._client = client or genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=TIMEOUT_MS, retry_options=RETRY),
        )
        self._model = model

    async def complete(self, messages: list[Message]) -> str:
        # Gemini takes the system prompt as a config field, not as a message, and
        # calls the assistant role "model".
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
            if m["role"] != "system"
        ]
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            # JSON mode: the model can't wrap the reply in markdown fences or prose,
            # which removes the most common reason for a validation retry.
            # card_generator still validates the content itself.
            response_mime_type="application/json",
        )
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model, contents=contents, config=config
            )
        except (errors.APIError, httpx2.HTTPError) as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc
        # .text is None when the reply was blocked or empty; "" fails validation
        # and goes through card_generator's normal retry path.
        return response.text or ""
