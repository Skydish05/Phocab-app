from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


# Lazy and cached: importing the app (e.g. in tests that don't touch the DB)
# never needs a database, and every request shares one connection pool.
@lru_cache
def get_engine() -> Engine:
    # pre_ping replaces connections Postgres dropped (restart, idle timeout)
    # instead of failing the first request that picks one up.
    return create_engine(get_settings().database_url, pool_pre_ping=True)


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    # expire_on_commit=False lets a route return ORM objects after committing
    # without a surprise reload query per attribute.
    return sessionmaker(get_engine(), expire_on_commit=False)
