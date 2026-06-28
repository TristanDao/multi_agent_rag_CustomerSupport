"""Create all tables (idempotent)."""
import logging

from app.db.session import Base, engine
import app.db.models  # noqa: F401  (register models on Base)

logger = logging.getLogger(__name__)


def init_db() -> None:
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("database_initialized")
    except Exception as e:
        logger.warning("database_init_failed: %s", str(e))
        raise
