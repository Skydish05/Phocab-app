import json

import pytest

from app.services.card_generator import (
    MAX_RETRIES,
    CardGenerationError,
    Message,
    VocabEntry,
    generate_card,
)

# anyio's pytest plugin ships with FastAPI's deps, so no pytest-asyncio needed.
pytestmark = pytest.mark.anyio

CUP = VocabEntry(label_en="cup", korean="컵", romanization="keop", topik_level=1)


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeLLM:
    """Returns canned replies in order and records every conversation it was sent."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[list[Message]] = []

    async def complete(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        return self.replies.pop(0)


def reply(**overrides) -> str:
    body = {"example": "컵에 물이 있어요."}
    body.update(overrides)
    return json.dumps(body, ensure_ascii=False)


async def test_valid_reply_returns_card_on_first_call():
    llm = FakeLLM(reply())

    card = await generate_card(CUP, llm)

    assert len(llm.calls) == 1
    assert card.korean == "컵"
    assert card.romanization == "keop"
    assert card.topik_level == 1
    assert card.example == "컵에 물이 있어요."


async def test_prompt_contains_vocabulary_from_entry():
    llm = FakeLLM(reply())

    await generate_card(CUP, llm)

    user_msg = llm.calls[0][-1]["content"]
    assert "컵" in user_msg and "keop" in user_msg and "cup" in user_msg


async def test_llm_cannot_override_vocabulary_fields():
    # Extra keys are rejected, so a reply that "corrects" the word fails
    # validation instead of reaching the card.
    llm = FakeLLM(reply(korean="잔", romanization="jan"), reply())

    card = await generate_card(CUP, llm)

    assert len(llm.calls) == 2
    assert (card.korean, card.romanization) == ("컵", "keop")


async def test_extra_fields_like_mnemonic_are_retried():
    llm = FakeLLM(reply(mnemonic="Keep your cup."), reply())

    card = await generate_card(CUP, llm)

    assert len(llm.calls) == 2
    assert "mnemonic" not in card.model_dump()


async def test_invalid_json_is_retried_with_error_feedback():
    llm = FakeLLM("Sure! Here is your card: {", reply())

    card = await generate_card(CUP, llm)

    assert card.korean == "컵"
    retry = llm.calls[1]
    assert retry[-2] == {"role": "assistant", "content": "Sure! Here is your card: {"}
    assert retry[-1]["role"] == "user"
    assert "failed validation" in retry[-1]["content"]


async def test_example_without_the_word_is_retried():
    llm = FakeLLM(reply(example="물을 마셔요."), reply())

    await generate_card(CUP, llm)

    assert len(llm.calls) == 2
    assert "example must contain the word '컵'" in llm.calls[1][-1]["content"]


async def test_example_with_english_translation_is_retried():
    llm = FakeLLM(reply(example="컵에 물이 있어요. (There is water in the cup.)"), reply())

    card = await generate_card(CUP, llm)

    assert len(llm.calls) == 2
    assert "Korean only" in llm.calls[1][-1]["content"]
    assert card.example == "컵에 물이 있어요."


async def test_multi_word_vocabulary_is_matched():
    suitcase = VocabEntry("suitcase", "여행 가방", "yeohaeng gabang", 2)
    llm = FakeLLM(reply(example="여행 가방이 무거워요."))

    card = await generate_card(suitcase, llm)

    assert card.example == "여행 가방이 무거워요."


async def test_gives_up_after_max_retries():
    llm = FakeLLM(*["not json"] * (MAX_RETRIES + 1))

    with pytest.raises(CardGenerationError) as exc_info:
        await generate_card(CUP, llm)

    assert len(llm.calls) == MAX_RETRIES + 1 == 3
    assert exc_info.value.attempts == 3
    assert exc_info.value.last_error
