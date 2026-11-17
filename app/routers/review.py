"""Leitner review: fetch the next due card, then grade it."""

from fastapi import APIRouter, HTTPException, Request, Response, status
from sqlalchemy import func, select

from app.deps import CurrentUser, DbSession, Now
from app.models import UserCard
from app.routers.cards import to_deck_card
from app.schemas import ReviewAnswer, ReviewNext, ReviewResult
from app.services.leitner import schedule
from app.templating import templates, wants_html

router = APIRouter(tags=["review"])


@router.get("/review/next", response_model=ReviewNext)
def next_card(
    request: Request, user: CurrentUser, db: DbSession, now: Now
) -> ReviewNext | Response:
    mine = UserCard.user_id == user.id
    due = mine & (UserCard.next_review_at <= now)
    # Most overdue first, so a backlog is worked through in the order it built
    # up; this filter + sort is what the (user_id, next_review_at) index serves.
    card = db.scalars(
        select(UserCard).where(due).order_by(UserCard.next_review_at, UserCard.id).limit(1)
    ).first()
    due_count = db.scalar(select(func.count()).select_from(UserCard).where(due))
    # The soonest upcoming card, for "all done, next card due tomorrow".
    next_due_at = db.scalar(select(func.min(UserCard.next_review_at)).where(mine))
    result = ReviewNext(
        card=to_deck_card(card) if card else None, due_count=due_count, next_due_at=next_due_at
    )
    if wants_html(request):
        return templates.TemplateResponse(request, "_review.html", {"review": result, "now": now})
    return result


@router.post(
    "/review/{card_id}",
    responses={
        404: {"description": "No such card in this user's deck"},
        409: {"description": "The card is not due yet (e.g. a double-tapped answer)"},
    },
)
def answer(
    card_id: int,
    body: ReviewAnswer,
    request: Request,
    response: Response,
    user: CurrentUser,
    db: DbSession,
    now: Now,
) -> ReviewResult:
    # FOR UPDATE: two answers for one card (a double tap) are serialized, so the
    # second one sees the new due date and gets a 409 instead of promoting twice.
    mine = select(UserCard).where(UserCard.id == card_id, UserCard.user_id == user.id)
    card = db.scalars(mine.with_for_update()).first()
    # 404 (not 403) for another user's card, so ids can't be probed for existence.
    if card is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Card not found")
    # Only due cards can be graded; otherwise "review" could be spammed to push a
    # card to box 5 without the spacing that makes Leitner work.
    if card.next_review_at > now:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Card is not due for review yet")

    card.box, card.next_review_at = schedule(card.box, body.correct, now)
    if body.correct:
        card.times_correct += 1
    else:
        card.times_wrong += 1
    db.commit()

    if wants_html(request):
        # The page reloads its review panel on this event, which fetches the next
        # due card; the response body itself stays the JSON result.
        response.headers["HX-Trigger"] = "card-reviewed"
    return ReviewResult(
        card_id=card.id,
        box=card.box,
        next_review_at=card.next_review_at,
        times_correct=card.times_correct,
        times_wrong=card.times_wrong,
    )
