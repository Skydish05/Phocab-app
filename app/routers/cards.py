from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, contains_eager

from app.deps import (
    CurrentUser,
    DbSession,
    get_llm,
    get_recognizer,
    get_storage,
    get_vocabulary,
)
from app.models import Photo, UserCard
from app.schemas import CardList, CardOut, DeckCard
from app.services.cache import UnknownLabelError, Vocabulary, get_or_create_word
from app.services.card_generator import CardGenerationError, LLMClient, LLMError
from app.services.images import MAX_UPLOAD_BYTES, ImageError, prepare_upload
from app.services.recognizer import RecognitionError, Recognizer
from app.services.storage import PhotoStorage, StorageError
from app.templating import templates, wants_html

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
    # Explicit because the return annotation includes Response (the HTMX
    # fragment), which FastAPI can't turn into a schema; /docs still shows CardOut.
    response_model=CardOut,
    responses={
        200: {"description": "This object is already in the deck; the existing card"},
        # Documented so /docs readers know why the same route can return HTML.
        201: {"description": "New card (an HTML fragment when sent with HX-Request: true)"},
        400: {"description": "Not a JPEG, PNG or WebP image"},
        413: {"description": "File larger than 5 MB"},
        502: {"description": "Model service, LLM or storage failure"},
    },
)
def create_card_from_photo(
    file: UploadFile,
    request: Request,
    user: CurrentUser,
    db: DbSession,
    response: Response,
    recognizer: Annotated[Recognizer, Depends(get_recognizer)],
    storage: Annotated[PhotoStorage, Depends(get_storage)],
    vocab: Annotated[Vocabulary, Depends(get_vocabulary)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> CardOut | Response:
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

    card = CardOut(
        card_id=card_id,
        label_en=word.label_en,
        korean=word.korean,
        romanization=word.romanization,
        topik_level=word.topik_level,
        example=word.example,
        confidence=recognition.confidence,
    )
    card_status = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    if wants_html(request):
        return templates.TemplateResponse(
            request, "_card.html", {"card": card, "created": created}, status_code=card_status
        )
    response.status_code = card_status
    return card


@router.get("/cards", response_model=CardList)
def list_cards(
    request: Request,
    user: CurrentUser,
    db: DbSession,
    # Capped so one request can't pull a whole deck of thousands of cards.
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    box: Annotated[int | None, Query(ge=1, le=5)] = None,
) -> CardList | Response:
    # Offset paging rather than cursors: a personal deck is small (hundreds of
    # cards), so OFFSET stays cheap and the client can jump to any page.
    filters = [UserCard.user_id == user.id]
    if box is not None:
        filters.append(UserCard.box == box)
    total = db.scalar(select(func.count()).select_from(UserCard).where(*filters))
    rows = db.scalars(
        select(UserCard)
        .where(*filters)
        .join(UserCard.word)
        # Fill card.word from this JOIN; otherwise each card lazy-loads its word
        # with its own query (N+1).
        .options(contains_eager(UserCard.word))
        # Newest first, with id as a tie-breaker so paging is stable when two
        # cards share a timestamp (one transaction, or a bulk import).
        .order_by(UserCard.created_at.desc(), UserCard.id.desc())
        .limit(limit)
        .offset(offset)
    )
    deck = CardList(
        items=[
            DeckCard(
                card_id=card.id,
                label_en=card.word.label_en,
                korean=card.word.korean,
                romanization=card.word.romanization,
                topik_level=card.word.topik_level,
                example=card.word.example,
                box=card.box,
                next_review_at=card.next_review_at,
                created_at=card.created_at,
            )
            for card in rows
        ],
        total=total,
    )
    if wants_html(request):
        next_offset = offset + limit if offset + limit < total else None
        context = {
            "deck": deck,
            "offset": offset,
            "limit": limit,
            "box": box,
            "next_offset": next_offset,
        }
        return templates.TemplateResponse(request, "_deck.html", context)
    return deck
