"""The words cache: one generated card per object label, shared by every user."""

import csv
from pathlib import Path

import anyio.from_thread
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models import Word, WordSource
from app.services.card_generator import LLMClient, VocabEntry, generate_card

Vocabulary = dict[str, VocabEntry]


class UnknownLabelError(Exception):
    """The model returned a label the CSV doesn't have (the two copies drifted). Maps to 502."""


def normalize_label(label: str) -> str:
    # Lowercase and collapse whitespace so "ID card", "id card" and "id  card " are
    # one cache row instead of three.
    return " ".join(label.lower().split())


def load_vocabulary(path: Path) -> Vocabulary:
    vocab: Vocabulary = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = normalize_label(row["label_en"])
            # Two rows that normalize the same would make the cache key ambiguous.
            if label in vocab:
                raise ValueError(f"{path}: duplicate label after normalizing: {label!r}")
            vocab[label] = VocabEntry(
                label_en=label,
                korean=row["korean"].strip(),
                romanization=row["romanization"].strip(),
                topik_level=int(row["topik_level"]),
            )
    return vocab


def _is_current(word: Word, entry: VocabEntry) -> bool:
    # A cached row only counts if its vocabulary still matches the CSV: after a CSV
    # fix (say 학생증 -> 신분증), the old example sentence uses the wrong word.
    return word.example is not None and (word.korean, word.romanization, word.topik_level) == (
        entry.korean,
        entry.romanization,
        entry.topik_level,
    )


def get_or_create_word(db: Session, vocab: Vocabulary, label_en: str, llm: LLMClient) -> Word:
    """Return the cached word for a label, generating its example sentence on a miss.

    Must run in a worker thread (a sync FastAPI route): the async LLM call is sent
    back to the event loop with anyio.from_thread.run.
    Raises UnknownLabelError, CardGenerationError, or LLMError.
    """
    entry = vocab.get(normalize_label(label_en))
    if entry is None:
        raise UnknownLabelError(label_en)

    word = db.scalar(select(Word).where(Word.label_en == entry.label_en))
    if word is not None and _is_current(word, entry):
        return word

    # End the read transaction before the slow LLM call, so this request doesn't
    # hold a pooled connection (idle in transaction) for seconds while it waits.
    # commit, not rollback: nothing is pending, and rollback would also expire the
    # objects already loaded (like the current user), costing reload queries.
    db.commit()
    card = anyio.from_thread.run(generate_card, entry, llm)

    # Vocabulary fields come from the CSV entry, never from the LLM (rule 1).
    values = {
        "korean": entry.korean,
        "romanization": entry.romanization,
        "topik_level": entry.topik_level,
        "example": card.example,
        "source": WordSource.HF,
    }
    # An upsert instead of a plain INSERT: two users photographing a new object at
    # once both generate, and the second write must update, not crash on the
    # unique label. Either sentence is valid, so last write wins.
    stmt = (
        insert(Word)
        .values(label_en=entry.label_en, **values)
        .on_conflict_do_update(index_elements=[Word.label_en], set_=values)
        .returning(Word)
    )
    # populate_existing refreshes a stale Word already loaded in this session.
    word = db.scalars(stmt, execution_options={"populate_existing": True}).one()
    # Committed now, apart from the photo and card: the LLM call is already paid
    # for, and the cached word stays useful even if a later step fails.
    db.commit()
    return word
