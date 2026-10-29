"""Password hashing and JWT access tokens."""

from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

from app.config import get_settings

# Argon2id with argon2-cffi's defaults (RFC 9106's low-memory profile): memory-hard,
# so GPU brute force of a leaked hash costs far more than with bcrypt or PBKDF2.
_hasher = PasswordHasher()

# Pinned, never read from the token header: trusting the header's "alg" is what
# enables the alg=none and algorithm-confusion attacks.
ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    # The salt is random per call and embedded in the returned string.
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerificationError, InvalidHashError):
        return False


# Login verifies against this when the email is unknown, so an unknown email takes
# as long as a wrong password and response timing doesn't reveal who has an account.
DUMMY_HASH = hash_password("dummy-password-for-timing")


def create_access_token(user_id: int, expires_delta: timedelta | None = None) -> str:
    settings = get_settings()
    now = datetime.now(UTC)
    expires = now + (expires_delta or timedelta(minutes=settings.jwt_expire_minutes))
    # sub is the user id, not the email, so changing an email can't break or
    # redirect existing tokens. JWT requires sub to be a string.
    claims = {"sub": str(user_id), "iat": now, "exp": expires}
    return jwt.encode(claims, settings.jwt_secret.get_secret_value(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> int | None:
    """Return the user id, or None for any invalid, expired, or tampered token."""
    try:
        claims = jwt.decode(
            token,
            get_settings().jwt_secret.get_secret_value(),
            algorithms=[ALGORITHM],
            # A token without exp would never expire, so reject it outright.
            options={"require": ["exp", "sub"]},
        )
        return int(claims["sub"])
    except (jwt.InvalidTokenError, ValueError):
        return None
