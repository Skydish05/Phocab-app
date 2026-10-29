from datetime import timedelta

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.models import User
from app.security import ALGORITHM, create_access_token, verify_password

EMAIL = "learner@example.com"
PASSWORD = "correct horse battery"


def register(client: TestClient, email: str = EMAIL, password: str = PASSWORD):
    return client.post("/auth/register", json={"email": email, "password": password})


def login(client: TestClient, email: str = EMAIL, password: str = PASSWORD):
    return client.post("/auth/login", json={"email": email, "password": password})


def get_me(client: TestClient, token: str):
    return client.get("/me", headers={"Authorization": f"Bearer {token}"})


def test_register_login_and_me(client: TestClient):
    registered = register(client)
    assert registered.status_code == 201
    assert registered.json()["email"] == EMAIL

    logged_in = login(client)
    assert logged_in.status_code == 200
    assert logged_in.json()["token_type"] == "bearer"

    me = get_me(client, logged_in.json()["access_token"])
    assert me.status_code == 200
    assert me.json() == {"email": EMAIL}


@pytest.mark.parametrize("second_email", [EMAIL, EMAIL.upper()])
def test_duplicate_email_returns_409(client: TestClient, second_email: str):
    register(client)

    response = register(client, email=second_email)

    assert response.status_code == 409


def test_wrong_password_and_unknown_email_get_the_same_401(client: TestClient):
    register(client)

    wrong_password = login(client, password="not the password")
    unknown_email = login(client, email="nobody@example.com")

    assert wrong_password.status_code == unknown_email.status_code == 401
    # Identical bodies, so the response doesn't reveal which emails have accounts.
    assert wrong_password.json() == unknown_email.json()


def test_plain_password_is_never_stored(client: TestClient, db_session: Session):
    register(client)

    user = db_session.scalar(select(User).where(User.email == EMAIL))
    assert PASSWORD not in user.password_hash
    assert user.password_hash.startswith("$argon2id$")
    assert verify_password(user.password_hash, PASSWORD)
    # And not in any other column either.
    row = db_session.execute(text("SELECT * FROM users WHERE id = :id"), {"id": user.id}).one()
    assert all(PASSWORD not in str(value) for value in row)


def test_expired_token_is_rejected(client: TestClient):
    user_id = register(client).json()["id"]
    token = create_access_token(user_id, expires_delta=timedelta(seconds=-1))

    assert get_me(client, token).status_code == 401


def test_tampered_payload_is_rejected(client: TestClient):
    register(client)
    other_id = register(client, email="other@example.com").json()["id"]
    header, _, signature = login(client).json()["access_token"].split(".")
    # Swap in another user's id but keep the original signature.
    forged_payload = jwt.utils.base64url_encode(f'{{"sub":"{other_id}","exp":9999999999}}'.encode())

    response = get_me(client, f"{header}.{forged_payload.decode()}.{signature}")

    assert response.status_code == 401


def test_token_signed_with_another_secret_is_rejected(client: TestClient):
    user_id = register(client).json()["id"]
    token = jwt.encode(
        {"sub": str(user_id), "exp": 9999999999},
        "an-attacker-secret-that-is-32-chars-long",
        algorithm=ALGORITHM,
    )

    assert get_me(client, token).status_code == 401


def test_unsigned_alg_none_token_is_rejected(client: TestClient):
    user_id = register(client).json()["id"]
    token = jwt.encode({"sub": str(user_id), "exp": 9999999999}, key=None, algorithm="none")

    assert get_me(client, token).status_code == 401


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Basic dXNlcjpwYXNz"}])
def test_missing_token_is_rejected(client: TestClient, headers: dict[str, str]):
    response = client.get("/me", headers=headers)

    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_token_of_deleted_user_is_rejected(client: TestClient, db_session: Session):
    register(client)
    token = login(client).json()["access_token"]
    db_session.execute(text("DELETE FROM users"))

    assert get_me(client, token).status_code == 401
