import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, make_url, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

# A separate database from the dev one, so running tests never wipes your data.
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://phocab:phocab@localhost:5432/phocab_test"
)
REPO_ROOT = Path(__file__).resolve().parent.parent


def alembic_config(database_url: str) -> Config:
    config = Config(REPO_ROOT / "alembic.ini")
    config.attributes["database_url"] = database_url
    config.attributes["configure_logger"] = False
    return config


def ensure_database_exists(url: str) -> None:
    target = make_url(url)
    # CREATE DATABASE can't run inside a transaction, hence AUTOCOMMIT, and it has
    # to be issued from another database, hence the "postgres" maintenance DB.
    admin = create_engine(target.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            exists = conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": target.database}
            )
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{target.database}"'))
    except OperationalError as exc:
        # Fail, not skip: DB tests silently skipping would hide real breakage.
        pytest.fail(
            f"Postgres is not reachable at {target.render_as_string(hide_password=True)}.\n"
            "Start it with: docker compose -f infra/docker-compose.yml up -d db\n"
            f"{exc}",
            pytrace=False,
        )
    finally:
        admin.dispose()


@pytest.fixture(scope="session")
def db_engine() -> Iterator[Engine]:
    """A real Postgres with the schema built by our Alembic migrations.

    Built with `alembic upgrade head`, not metadata.create_all(), so the tests run
    against exactly what production gets. The schema is dropped first so an edited
    migration can't leave a stale schema behind at the same revision number.
    """
    ensure_database_exists(TEST_DATABASE_URL)
    engine = create_engine(TEST_DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    command.upgrade(alembic_config(TEST_DATABASE_URL), "head")
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine: Engine) -> Iterator[Session]:
    """A session whose work is rolled back after each test.

    The outer transaction is never committed; create_savepoint turns the session's
    own commits into SAVEPOINTs inside it. Each test starts from an empty schema
    without the cost of truncating tables.
    """
    with db_engine.connect() as conn:
        outer = conn.begin()
        session = Session(bind=conn, join_transaction_mode="create_savepoint")
        try:
            yield session
        finally:
            session.close()
            outer.rollback()
