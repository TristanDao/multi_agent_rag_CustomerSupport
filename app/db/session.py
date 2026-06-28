"""Database session and engine setup."""
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from app.config import settings


def _build_engine():
    url = settings.DATABASE_URL
    kwargs = {"future": True}
    if not url.startswith("sqlite"):
        kwargs.update({"pool_pre_ping": True, "pool_size": 10, "max_overflow": 20})
    return create_engine(url, **kwargs)


engine = _build_engine()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

Base = declarative_base()


def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
