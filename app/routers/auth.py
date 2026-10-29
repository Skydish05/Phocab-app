from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.deps import CurrentUser, DbSession
from app.models import User
from app.schemas import LoginRequest, MeOut, RegisterRequest, TokenResponse, UserOut
from app.security import DUMMY_HASH, create_access_token, hash_password, verify_password

# /me lives here too: it's the one route that only needs the auth dependency.
router = APIRouter(tags=["auth"])


@router.post("/auth/register", status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: DbSession) -> UserOut:
    user = User(email=body.email, password_hash=hash_password(body.password))
    db.add(user)
    # Let the unique constraint decide instead of SELECT-then-INSERT: two
    # simultaneous sign-ups could both pass the SELECT, but only one INSERT wins.
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        if "uq_users_email" not in str(exc.orig):
            raise
        raise HTTPException(status.HTTP_409_CONFLICT, "Email is already registered") from None
    return UserOut.model_validate(user)


@router.post("/auth/login")
def login(body: LoginRequest, db: DbSession) -> TokenResponse:
    user = db.scalar(select(User).where(User.email == body.email))
    # Hash even when the email is unknown so both failures take the same time.
    password_ok = verify_password(user.password_hash if user else DUMMY_HASH, body.password)
    if user is None or not password_ok:
        # Same response for unknown email and wrong password: no account enumeration.
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return TokenResponse(access_token=create_access_token(user.id))


@router.get("/me")
def me(user: CurrentUser) -> MeOut:
    return MeOut(email=user.email)
