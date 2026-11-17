from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, contains_eager

from app.config import Settings, get_settings
from app.deps import (
    CurrentUser,
    DbSession,
    Now,
    get_llm,
    get_recognizer,
    get_storage,
    get_vocabulary,
)
from app.models import Photo, UserCard, Word
from app.schemas import Candidate, CardList, CardOut, DeckCard, LabelIn, UnrecognizedOut
from app.services.cache import (
    UnknownLabelError,
    Vocabulary,
    get_or_create_word,
    normalize_label,
)
from app.services.card_generator import CardGenerationError, LLMClient, LLMError
from app.services.images import MAX_UPLOAD_BYTES, ImageError, prepare_upload
from app.services.recognizer import Recognition, RecognitionError, Recognizer
from app.services.storage import PhotoStorage, StorageError
from app.templating import templates, wants_html

router = APIRouter(tags=["cards"])


def to_deck_card(card: UserCard) -> DeckCard:
    """Shared by the deck and the review routes, so both show a card the same way."""
    return DeckCard(
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


def _save_photo(
    db: Session,
    storage: PhotoStorage,
    user_id: int,
    jpeg: bytes,
    recognition: Recognition,
    word_id: int | None,
) -> tuple[Photo, int | None, bool]:
    """Upload, record the photo, and add its word to the deck when there is one.

    Returns (photo, card id or None, created). word_id is None for a photo the
    model couldn't recognize: it's still stored, as an evaluation example and so
    the user can pick its word without uploading again.
    """
    try:
        s3_key = storage.put_photo(user_id, jpeg)
    except StorageError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Photo storage failed") from exc
    try:
        photo = Photo(
            user_id=user_id,
            s3_key=s3_key,
            word_id=word_id,
            confidence=recognition.confidence,
            model_used=recognition.model_id,
            latency_ms=recognition.latency_ms,
        )
        db.add(photo)
        db.flush()  # assigns photo.id for the card row
        card_id, created = (
            _add_to_deck(db, user_id, word_id, photo.id) if word_id is not None else (None, False)
        )
        db.commit()
    except Exception:
        # The DB rows didn't land, so the uploaded photo would be unreferenced.
        db.rollback()
        storage.delete(s3_key)
        raise
    return photo, card_id, created


def _card_response(
    request: Request,
    response: Response,
    card: CardOut,
    created: bool,
) -> CardOut | Response:
    card_status = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    if wants_html(request):
        return templates.TemplateResponse(
            request, "_card.html", {"card": card, "created": created}, status_code=card_status
        )
    response.status_code = card_status
    return card


def _word_or_http_error(
    db: Session, vocab: Vocabulary, label_en: str, llm: LLMClient, unknown: HTTPException
) -> Word:
    """get_or_create_word with its errors mapped to HTTP. `unknown` differs per caller:
    a model label missing from the CSV is our bug (502), a user's pick is theirs (422)."""
    try:
        return get_or_create_word(db, vocab, label_en, llm)
    except UnknownLabelError as exc:
        raise unknown from exc
    except (CardGenerationError, LLMError) as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Card generation failed") from exc


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
        422: {
            "model": UnrecognizedOut,
            "description": "Top score below RECOGNITION_MIN_SCORE: no card; pick a word "
            "with POST /photos/{photo_id}/label",
        },
        502: {"description": "Model service, LLM or storage failure"},
    },
)
def create_card_from_photo(
    file: UploadFile,
    request: Request,
    user: CurrentUser,
    db: DbSession,
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
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

    try:
        recognition = recognizer.recognize(jpeg)
    except RecognitionError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "Object recognition failed") from exc

    # The zero-shot model always names *some* label from the list, even for an
    # object that isn't on it (a tire comes back as "car"). A very low top score
    # is the signal for that, so instead of a card that looks just as sure as a
    # correct one, ask the user. No LLM call is spent on a guess nobody trusts.
    if recognition.confidence < settings.recognition_min_score:
        photo, _, _ = _save_photo(db, storage, user.id, jpeg, recognition, word_id=None)
        unrecognized = UnrecognizedOut(
            detail="Couldn't recognize this object",
            photo_id=photo.id,
            candidates=[
                Candidate(
                    label_en=entry.label_en,
                    korean=entry.korean,
                    romanization=entry.romanization,
                    score=score,
                )
                for label, score in recognition.candidates
                # Skipped rather than failing: a guess the CSV lacks (the two
                # copies drifted) just isn't offered.
                if (entry := vocab.get(normalize_label(label))) is not None
            ],
        )
        # 422, not 200: nothing was created, and a client that only handles
        # success must not mistake this for a card.
        if wants_html(request):
            return templates.TemplateResponse(
                request, "_unrecognized.html", {"result": unrecognized}, status_code=422
            )
        return JSONResponse(unrecognized.model_dump(), status_code=422)

    # Generate before uploading (PLAN.md uploads first): if the LLM fails, no
    # orphaned photo is left in S3.
    word = _word_or_http_error(
        db,
        vocab,
        recognition.label_en,
        llm,
        unknown=HTTPException(
            status.HTTP_502_BAD_GATEWAY, "Recognized object is not in the vocabulary"
        ),
    )
    photo, card_id, created = _save_photo(db, storage, user.id, jpeg, recognition, word.id)
    card = CardOut(
        card_id=card_id,
        label_en=word.label_en,
        korean=word.korean,
        romanization=word.romanization,
        topik_level=word.topik_level,
        example=word.example,
        confidence=recognition.confidence,
        photo_id=photo.id,
    )
    return _card_response(request, response, card, created)


@router.post(
    "/photos/{photo_id}/label",
    status_code=status.HTTP_201_CREATED,
    response_model=CardOut,
    responses={
        200: {"description": "This word is already in the deck; the existing card"},
        404: {"description": "No such photo for this user"},
        409: {"description": "This photo already has a word"},
        422: {"description": "The word isn't in the vocabulary"},
        502: {"description": "LLM failure"},
    },
)
def label_photo(
    photo_id: int,
    body: LabelIn,
    request: Request,
    response: Response,
    user: CurrentUser,
    db: DbSession,
    vocab: Annotated[Vocabulary, Depends(get_vocabulary)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> CardOut | Response:
    """Make the card for a photo the model couldn't recognize, from the user's pick."""
    mine = select(Photo).where(Photo.id == photo_id, Photo.user_id == user.id)
    already_labeled = HTTPException(status.HTTP_409_CONFLICT, "This photo already has a word")
    photo = db.scalars(mine).first()
    # 404 for another user's photo too, so photo ids can't be probed.
    if photo is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Photo not found")
    if photo.word_id is not None:
        raise already_labeled

    # Any vocabulary word, not only the 3 guesses: the guesses are what the page
    # offers, but the rule that matters is rule 1 (words come from the CSV).
    word = _word_or_http_error(
        db,
        vocab,
        body.label_en,
        llm,
        unknown=HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Unknown word"),
    )

    # Re-read under a row lock: a cache miss above committed (to release the
    # connection during the LLM call), so a second request could have labeled the
    # photo meanwhile. The lock makes the check-and-set atomic.
    photo = db.scalars(mine.with_for_update().execution_options(populate_existing=True)).one()
    if photo.word_id is not None:
        db.rollback()
        raise already_labeled
    photo.word_id = word.id
    # Marks the label as a person's, not the model's, so evaluation can tell them
    # apart (a user pick is closer to ground truth than to a model prediction).
    # A string prefix rather than a new column until the eval item needs more.
    photo.model_used = f"user-picked:{photo.model_used}"[:200]
    card_id, created = _add_to_deck(db, user.id, word.id, photo.id)
    db.commit()

    card = CardOut(
        card_id=card_id,
        label_en=word.label_en,
        korean=word.korean,
        romanization=word.romanization,
        topik_level=word.topik_level,
        example=word.example,
        confidence=None,
        photo_id=photo.id,
    )
    return _card_response(request, response, card, created)


@router.get("/cards", response_model=CardList)
def list_cards(
    request: Request,
    user: CurrentUser,
    db: DbSession,
    now: Now,
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
        items=[to_deck_card(card) for card in rows],
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
            # The request clock, so "due in 2 days" agrees with what review uses.
            "now": now,
        }
        return templates.TemplateResponse(request, "_deck.html", context)
    return deck


@router.delete(
    "/cards/{card_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"description": "No such card in this user's deck"}},
)
def remove_card(card_id: int, request: Request, user: CurrentUser, db: DbSession) -> Response:
    # Deletes the deck entry only. The photo and any feedback stay: they're the
    # evaluation record of what the model got wrong.
    deleted = db.scalar(
        delete(UserCard)
        .where(UserCard.id == card_id, UserCard.user_id == user.id)
        .returning(UserCard.id)
    )
    # 404 (not 403) for another user's card, as in review: ids can't be probed.
    if deleted is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Card not found")
    db.commit()
    if wants_html(request):
        # HTMX swaps nothing on a 204, so the page gets a short note to replace the card with.
        return HTMLResponse('<p class="text-sm opacity-70 px-1">Removed from your deck.</p>')
    return Response(status_code=status.HTTP_204_NO_CONTENT)
