"""Retriever interface over the configured vector store."""
import logging
from typing import Any, Dict, List

from app.config import settings
from app.rag.embeddings import embed_query
from app.rag.vector_store import VectorRecord, get_vector_store

logger = logging.getLogger(__name__)


def retrieve(query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
    store = get_vector_store()
    top_k = top_k or settings.TOP_K_RETRIEVAL
    if not query.strip():
        return []
    if store.count() == 0:
        logger.warning("retrieval_empty_store")
        return []
    try:
        qvec = embed_query(query)
    except Exception as e:
        logger.warning("embed_query_failed err=%s", str(e))
        return []
    try:
        hits = store.search(qvec, top_k=top_k)
    except Exception as e:
        logger.warning("vector_search_failed err=%s", str(e))
        return []
    return [_format_hit(rec, score) for rec, score in hits]


def _format_hit(rec: VectorRecord, score: float) -> Dict[str, Any]:
    return {
        "doc_id": rec.id,
        "content": rec.content,
        "source": rec.source,
        "section": rec.section,
        "chunk_index": rec.chunk_index,
        "score": round(float(score), 4),
    }
