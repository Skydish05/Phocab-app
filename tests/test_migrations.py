from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Engine, inspect

from app.models import Base
from tests.conftest import TEST_DATABASE_URL, alembic_config

TABLES = {"users", "words", "photos", "user_cards", "feedback"}


def test_upgrade_head_creates_all_tables(db_engine: Engine):
    assert TABLES <= set(inspect(db_engine).get_table_names())


def test_models_match_migrations(db_engine: Engine):
    # Catches "changed a model, forgot to generate a migration". Autogenerate
    # doesn't compare CHECK constraints, so test_models covers those directly.
    with db_engine.connect() as conn:
        context = MigrationContext.configure(conn, opts={"compare_type": True})
        diff = compare_metadata(context, Base.metadata)

    assert diff == []


def test_downgrade_to_base_then_upgrade_again(db_engine: Engine):
    config = alembic_config(TEST_DATABASE_URL)

    command.downgrade(config, "base")
    assert not TABLES & set(inspect(db_engine).get_table_names())

    # Ends at head, so tests that run after this one still have the schema.
    command.upgrade(config, "head")
    assert TABLES <= set(inspect(db_engine).get_table_names())
