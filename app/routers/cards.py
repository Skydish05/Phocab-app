from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile, status
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.deps import (
    CurrentUser,
    DbSession,
    get_llm,
    get_recognizer,
    get_storage,
    get_vocabulary,
)
from app.models import Photo, UserCard
from app.schemas import CardOut
from app.services.cache import UnknownLabelError, Vocabulary, get_or_create_word
from app.services.card_generator import CardGenerationError, LLMClient, LLMError
from app.services.images import MAX_UPLOAD_BYTES, ImageError, prepare_upload
from app.services.recognizer import RecognitionError, Recognizer
from app.services.storage import PhotoStorage, StorageError

router = APIRouter(tags=["cards"])


def _add_to_deck(db: Session, user_id: int, word_id: int, photo_id: int) -> tuple[int, bool]:
    """Return (card id, created). A second photo of the same object reuses the card."""
    # ON CONFLICT DO NOTHING instead of SELECT-then-INSERT, for the same race
    # reason as registration: a double-tapped upload sends two requests at once.
    card_id = db.scalar(
        insert(UserCard)
        .values(user_id=user_id, word_id=word_id, photo_id=photo_id)
        .on_conflict_do_nothing(index_elements=[UserCard.user_id, UserCard.word_id])
        .returning(UserCard.id)
    )
    if card_id is not None:
        return card_id, True
    existing = select(UserCard.id).where(UserCard.user_id == user_id, UserCard.word_id == word_id)
    return db.scalars(existing).one(), False


# A plain `def`: SQLAlchemy and boto3 are sync, so FastAPI runs this in a worker
# thread and blocking calls don't stall the event loop. The trade-off is that a
# cache miss holds that thread while the LLM answers.
@router.post(
    "/cards/from-photo",
    status_code=status.HTTP_201_CREATED,
    responses={
        200: {"description": "This object is already in the deck; the existing card"},
        400: {"description": "Not a JPEG, PNG or WebP image"},
        413: {"description": "File larger than 5 MB"},
        502: {"description": "Model service, LLM or storage failure"},
    },
)
def create_card_from_photo(
    file: UploadFile,
    user: CurrentUser,
    db: DbSession,
    response: Response,
    recognizer: Annotated[Recognizer, Depends(get_recognizer)],
    storage: Annotated[PhotoStorage, Depends(get_storage)],
    vocab: Annotated[Vocabulary, Depends(get_vocabulary)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> CardOut:
    # Read one byte past the limit: enough to know the file is too big without
    # holding more of it. (Nginx's client_max_body_size will be the hard cap;
    # Starlette has already spooled the upload by the time this runs.)
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Photo must be 5 MB or smaller")
    try:
        jpeg = prepare_upload(data)
    except ImageError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    # Recognize and generate before uploading (PLAN.md uploads first): if the model
    # or the LLM fails, no orphaned photo is left in S3.
    try:
        recognition = recognizer.recognize(jpeg)
    except RecognitionError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Object recognition failed") from exc
    try:
        word = get_or_create_word(db, vocab, recognition.label_en, llm)
    except UnknownLabelError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "Recognized object is not in the vocabulary"
        ) from exc
    except (CardGenerationError, LLMError) as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Card generation failed") from exc

    try:
        s3_key = storage.put_photo(user.id, jpeg)
    except StorageError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Photo storage failed") from exc
    try:
        photo = Photo(
            user_id=user.id,
            s3_key=s3_key,
            word_id=word.id,
            confidence=recognition.confidence,
            model_used=recognition.model_id,
            latency_ms=recognition.latency_ms,
        )
        db.add(photo)
        db.flush()  # assigns photo.id for the card row
        card_id, created = _add_to_deck(db, user.id, word.id, photo.id)
        db.commit()
    except Exception:
        # The DB rows didn't land, so the uploaded photo would be unreferenced.
        db.rollback()
        storage.delete(s3_key)
        raise

    if not created:
        response.status_code = status.HTTP_200_OK
    return CardOut(
        card_id=card_id,
        label_en=word.label_en,
        korean=word.korean,
        romanization=word.romanization,
        topik_level=word.topik_level,
        example=word.example,
        confidence=recognition.confidence,
    )
