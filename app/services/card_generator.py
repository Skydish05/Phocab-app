"""Generate the LLM-written parts of a vocabulary card, validated and retried.

The vocabulary fields (Korean word, romanization, TOPIK level) come from
labels_ko.csv and are passed in; the LLM only writes one Korean example
sentence. The final Card copies the vocabulary fields from the input entry, so
nothing the LLM returns can change them.
"""

import json
import re
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)

# 1 first attempt + 2 retries, per the project rules. Each retry costs a full
# LLM call, so more retries would mostly add latency and cost to a bad model.
MAX_RETRIES = 2

LATIN_LETTER = re.compile(r"[A-Za-z]")


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMClient(Protocol):
    """The only thing the generator needs from an LLM provider.

    A Protocol keeps provider SDKs out of this module, so tests use a fake and
    swapping providers doesn't touch the validation/retry logic. Transport
    errors (timeouts, 5xx) are the client's job; provider SDKs already retry them.
    """

    async def complete(self, messages: list[Message]) -> str: ...


@dataclass(frozen=True)
class VocabEntry:
    """One row of labels_ko.csv: the source of truth for vocabulary fields."""

    label_en: str
    korean: str
    romanization: str
    topik_level: int


class LLMCardContent(BaseModel):
    """The exact JSON the LLM must return.

    extra="forbid" rejects any extra keys, so a model that tries to "correct"
    the word or romanization fails validation instead of being silently trusted.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # Length cap stops a rambling or looping model from filling the card.
    example: str = Field(min_length=1, max_length=120)

    @field_validator("example")
    @classmethod
    def example_uses_the_word(cls, example: str, info: ValidationInfo) -> str:
        # A substring check is enough because every vocabulary entry is a noun,
        # and Korean particles attach after the noun unchanged (컵 -> 컵을, 컵이).
        word = (info.context or {}).get("korean")
        if word and word not in example:
            raise ValueError(f"example must contain the word {word!r}")
        # Models often append a translation like "(There is a cup.)" even when told
        # not to; rejecting Latin letters turns that into a retry instead of a bad card.
        if LATIN_LETTER.search(example):
            raise ValueError("example must be Korean only, with no English")
        return example


class Card(BaseModel):
    """A finished card: vocabulary fields from the CSV plus the LLM-written example."""

    label_en: str
    korean: str
    romanization: str
    topik_level: int
    example: str


class CardGenerationError(Exception):
    """The LLM never produced valid output. The API maps this to a 502."""

    def __init__(self, label_en: str, attempts: int, last_error: str) -> None:
        super().__init__(
            f"card generation for {label_en!r} failed after {attempts} attempts: {last_error}"
        )
        self.attempts = attempts
        self.last_error = last_error


SYSTEM_PROMPT = """\
You write content for Korean vocabulary flashcards for beginner learners.
Reply with ONLY a JSON object, no markdown fences or extra text, matching this JSON schema:
{schema}
Rules:
- Write exactly 1 short, natural example sentence in Korean at beginner level.
- The sentence must contain the given Korean word exactly as written
  (particles after it are fine).
- Write the sentence in Korean only, with no English translation.
- Do not include the word, romanization, or level in your reply; they are fixed."""


def build_messages(entry: VocabEntry) -> list[Message]:
    schema = json.dumps(LLMCardContent.model_json_schema(), ensure_ascii=False)
    user = (
        f"English: {entry.label_en}\n"
        f"Korean word: {entry.korean}\n"
        f"Romanization: {entry.romanization}\n"
        f"TOPIK level: {entry.topik_level}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(schema=schema)},
        {"role": "user", "content": user},
    ]


def format_errors(error: ValidationError) -> str:
    # Drop input values and doc URLs: the model already has its own reply, and
    # the shorter message keeps retry prompts (and their token cost) small.
    lines = []
    for err in error.errors(include_url=False, include_input=False):
        loc = ".".join(str(part) for part in err["loc"]) or "(root)"
        lines.append(f"- {loc}: {err['msg']}")
    return "\n".join(lines)


async def generate_card(entry: VocabEntry, client: LLMClient) -> Card:
    """Ask the LLM for an example sentence and validate it, retrying up to MAX_RETRIES times.

    On a validation failure, the bad reply and the errors are appended to the
    conversation, so the model can fix the specific problem instead of guessing again.
    """
    messages = build_messages(entry)
    last_error = ""
    for _ in range(MAX_RETRIES + 1):
        raw = await client.complete(messages)
        try:
            # model_validate_json also turns malformed JSON into a ValidationError,
            # so bad JSON and wrong fields share one retry path.
            content = LLMCardContent.model_validate_json(raw, context={"korean": entry.korean})
        except ValidationError as e:
            last_error = format_errors(e)
            messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": "Your reply failed validation:\n"
                    f"{last_error}\nReply again with only the corrected JSON object.",
                },
            ]
            continue
        return Card(
            label_en=entry.label_en,
            korean=entry.korean,
            romanization=entry.romanization,
            topik_level=entry.topik_level,
            example=content.example,
        )
    raise CardGenerationError(entry.label_en, MAX_RETRIES + 1, last_error)
