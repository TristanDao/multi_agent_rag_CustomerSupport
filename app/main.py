"""FastAPI application factory."""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes_admin import router as admin_router
from app.api.routes_chat import router as chat_router
from app.api.routes_health import router as health_router
from app.config import settings
from app.core.logging import setup_logging
from app.core.security import new_request_id
from app.db.seed import init_db
from app.rag.vector_store import get_vector_store

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup app=%s env=%s", settings.APP_NAME, settings.APP_ENV)
    try:
        init_db()
    except Exception as e:
        logger.warning("startup_db_init_failed err=%s", str(e))
    try:
        store = get_vector_store()
        logger.info("vector_store_ready count=%s", store.count())
    except Exception as e:
        logger.warning("vector_store_init_failed err=%s", str(e))
    yield
    logger.info("shutdown complete")


app = FastAPI(
    title="Multi-Agent RAG Retail Assistant",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = new_request_id()
    request.state.request_id = request_id
    started = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        logger.exception(
            "request_failed",
            extra={"request_id": request_id, "status": "error", "error": str(e)},
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "internal_server_error", "request_id": request_id},
        )
    latency_ms = int((time.time() - started) * 1000)
    logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "status": "success",
            "latency_ms": latency_ms,
            "path": request.url.path,
        },
    )
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(health_router, tags=["health"])
app.include_router(chat_router, tags=["chat"])
app.include_router(admin_router, tags=["admin"])
