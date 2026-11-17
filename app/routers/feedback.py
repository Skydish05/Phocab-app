"""Reports from the "This is wrong" button: the corrections dataset for evaluation."""

from fastapi import APIRouter, HTTPException, Request, Response, status
from sqlalchemy import select

from app.deps import CurrentUser, DbSession
from app.models import Feedback, Photo, UserCard
from app.schemas import FeedbackIn, FeedbackOut
from app.templating import templates, wants_html

router = APIRouter(tags=["feedback"])


@router.post(
    "/feedback",
    status_code=status.HTTP_201_CREATED,
    response_model=FeedbackOut,
    responses={404: {"description": "No such photo for this user"}},
)
def report(
    body: FeedbackIn, request: Request, user: CurrentUser, db: DbSession
) -> FeedbackOut | Response:
    photo = db.get(Photo, body.photo_id)
    # 404 for another user's photo too, so photo ids can't be probed.
    if photo is None or photo.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Photo not found")
    # word_id is copied from the photo, not sent by the client: the report is
    # about what the model said for this photo, which the server already knows.
    feedback = Feedback(
        user_id=user.id,
        photo_id=photo.id,
        word_id=photo.word_id,
        reason=body.reason,
        note=body.note,
    )
    db.add(feedback)
    db.commit()

    if wants_html(request):
        # The card for this photo's word, if it's still in the deck, so the page
        # can offer to remove a card the user just said is wrong. Looked up by
        # word rather than photo: an "already in your deck" card points at the
        # first photo of that object, not this one.
        card_id = None
        if photo.word_id is not None:
            card_id = db.scalar(
                select(UserCard.id).where(
                    UserCard.user_id == user.id, UserCard.word_id == photo.word_id
                )
            )
        return templates.TemplateResponse(
            request, "_feedback_done.html", {"card_id": card_id}, status_code=201
        )
    return FeedbackOut.model_validate(feedback)
