"""Health and readiness endpoints."""
import logging

from fastapi import APIRouter
from sqlalchemy import text

from app.config import settings
from app.db.schemas import HealthResponse
from app.db.session import engine
from app.rag.vector_store import get_vector_store

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    db_status = "down"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "up"
    except Exception as e:
        logger.warning("db_health_failed err=%s", str(e))

    vdb_status = "down"
    try:
        vdb = get_vector_store()
        vdb_status = "up" if vdb is not None else "down"
    except Exception as e:
        logger.warning("vdb_health_failed err=%s", str(e))

    llm_status = "configured" if settings.has_llm_key else "not_configured"

    overall = "ok" if db_status == "up" and vdb_status == "up" else "degraded"
    return HealthResponse(
        status=overall,
        database=db_status,
        vector_db=vdb_status,
        llm=llm_status,
    )
