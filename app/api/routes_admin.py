"""Admin endpoints: doc ingestion, seeding, sample logs."""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.db.seed import init_db
from app.rag.chunking import load_markdown_dir
from app.rag.embeddings import embed_texts
from app.rag.vector_store import VectorRecord, get_vector_store

router = APIRouter(prefix="/admin")
logger = logging.getLogger(__name__)


@router.post("/ingest-docs")
def ingest_docs() -> Dict[str, Any]:
    """Ingest all markdown files in data/docs into the vector store."""
    docs_dir = Path(os.environ.get("DOCS_DIR", "data/docs"))
    chunks = load_markdown_dir(docs_dir, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(status_code=400, detail=f"No markdown files found in {docs_dir}")
    texts = [c.content for c in chunks]
    try:
        vectors = embed_texts(texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")
    records = [
        VectorRecord(
            id=c.doc_id,
            content=c.content,
            source=c.source,
            section=c.section,
            chunk_index=c.chunk_index,
            embedding=vec,
        )
        for c, vec in zip(chunks, vectors)
    ]
    store = get_vector_store()
    store.upsert(records)
    return {
        "status": "ok",
        "ingested_chunks": len(records),
        "store_size": store.count(),
    }


@router.post("/seed-data")
def seed_data() -> Dict[str, Any]:
    """Create database tables (does NOT populate data; use scripts/seed_database.py)."""
    init_db()
    return {"status": "ok", "message": "Tables ensured. Run scripts/seed_database.py to insert synthetic data."}


@router.get("/logs/sample")
def logs_sample() -> Dict[str, Any]:
    """Return sample (last 5) structured log events. Lightweight implementation."""
    from app.core.logging import get_logger

    log = get_logger("admin.sample")
    for i in range(3):
        log.info(
            "sample_event",
            extra={"request_id": f"req_sample_{i}", "status": "ok", "intent": "product_search"},
        )
    return {"status": "ok", "written": 3}
