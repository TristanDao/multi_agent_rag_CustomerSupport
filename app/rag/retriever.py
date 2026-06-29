"""Hybrid retriever.

Pipeline (per query):

    1. (optional) metadata filter  — `intent_tags in [...]`, `source in [...]`
    2. BM25 top-N  (sparse leg)    — from the in-memory `BM25Index`
    3. Vector top-N  (dense leg)    — from Qdrant (or in-memory fallback)
    4. Merge by `doc_id`            — weighted score: `sparse * w_s + dense * w_d`
    5. Rerank top-K                 — Alibaba `gte-rerank` (or no-op fallback)
    6. Return final top-K chunks with both `vector_score` and `rerank_score`.

The legacy `retrieve(query, top_k)` signature is kept for back-compat; new
callers should use `hybrid_retrieve(query, ...)` for full control over filters
and rerank.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from app.config import settings
from app.rag.bm25_index import BM25Hit, get_bm25_index
from app.rag.embeddings import embed_query
from app.rag.rerank import rerank
from app.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


# --- Public API ---------------------------------------------------------


def retrieve(query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
    """Back-compat wrapper: calls `hybrid_retrieve` with no filters."""
    results = hybrid_retrieve(query=query, top_k=top_k or settings.RERANK_FINAL_K)
    return [_format_hit(c) for c in results]


def hybrid_retrieve(
    query: str,
    *,
    intent_filter: Optional[Iterable[str]] = None,
    source_filter: Optional[Iterable[str]] = None,
    top_n: Optional[int] = None,
    top_k: Optional[int] = None,
    use_rerank: bool = True,
) -> List[Dict[str, Any]]:
    """Run the full hybrid retrieval pipeline.

    Args:
        query: the user's natural-language question.
        intent_filter: optional iterable of intent tags (e.g. `["shipping_policy"]`).
            Hits are restricted to chunks whose `intent_tags` intersects this set.
        source_filter: optional iterable of source file names
            (e.g. `["wholesale_policy.md"]`).
        top_n: number of candidates per leg before merge (default
            `settings.HYBRID_TOP_N`).
        top_k: number of final hits returned after rerank
            (default `settings.RERANK_FINAL_K`).
        use_rerank: if False, skip the rerank step.

    Returns:
        List of chunk dicts (already reranked, top_k of them). Each dict has
        `vector_score`, `sparse_score`, `hybrid_score`, `rerank_score`.
    """
    top_n = int(top_n or settings.HYBRID_TOP_N)
    top_k = int(top_k or settings.RERANK_FINAL_K)
    if not query or not query.strip():
        return []

    # --- 1 + 2: BM25 sparse leg ---
    bm25 = get_bm25_index()
    bm25_hits = bm25.search(
        query=query,
        top_k=top_n,
        intent_filter=intent_filter,
        source_filter=source_filter,
    ) if bm25 and bm25.docs else []
    bm25_scores = _normalise_bm25_scores([h.score for h in bm25_hits])
    bm25_by_id: Dict[str, Dict[str, Any]] = {}
    for h, ns in zip(bm25_hits, bm25_scores):
        bm25_by_id[h.doc_id] = {
            "doc_id": h.doc_id,
            "content": h.content,
            "source": h.source,
            "section": h.section,
            "chunk_index": h.chunk_index,
            "intent_tags": h.intent_tags,
            "sparse_score": round(float(ns), 4),
        }

    # --- 1 + 3: Vector dense leg ---
    store = get_vector_store()
    if store.count() == 0:
        logger.warning("retrieval_empty_store")
        dense_hits: list = []
    else:
        try:
            qvec = embed_query(query)
            dense_hits = store.search(
                qvec,
                top_k=top_n,
                intent_filter=intent_filter,
                source_filter=source_filter,
            )
        except Exception as e:
            logger.warning("embed_or_search_failed err=%s", str(e))
            dense_hits = []
    dense_scores = _normalise_dense_scores([s for _, s in dense_hits])
    dense_by_id: Dict[str, Dict[str, Any]] = {}
    for (rec, _), ns in zip(dense_hits, dense_scores):
        dense_by_id[rec.id] = {
            "doc_id": rec.id,
            "content": rec.content,
            "source": rec.source,
            "section": rec.section,
            "chunk_index": rec.chunk_index,
            "intent_tags": list(rec.intent_tags or []),
            "vector_score": round(float(ns), 4),
        }

    # --- 4: Merge by doc_id ---
    merged = _merge_hits(bm25_by_id, dense_by_id)
    if not merged:
        return []

    # --- 5: Rerank ---
    if use_rerank and merged:
        merged = rerank(query, merged, top_k=top_k)
    else:
        # No rerank: still tag rerank_score with the hybrid score
        for c in merged:
            c["rerank_score"] = float(c.get("hybrid_score") or 0.0)
        merged = merged[:top_k]

    return merged


# --- Helpers ------------------------------------------------------------


def _normalise_dense_scores(scores: List[float]) -> List[float]:
    """Dense cosine is already in [-1, 1]; we shift to [0, 1] for blending."""
    if not scores:
        return []
    return [max(0.0, min(1.0, (s + 1.0) / 2.0)) for s in scores]


def _normalise_bm25_scores(scores: List[float]) -> List[float]:
    """Min-max normalise BM25 scores to [0, 1]."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _merge_hits(
    bm25_by_id: Dict[str, Dict[str, Any]],
    dense_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge two leg-by-doc_id dicts into a single list with hybrid score."""
    w_s = float(settings.HYBRID_SPARSE_WEIGHT)
    w_d = float(settings.HYBRID_DENSE_WEIGHT)
    all_ids = set(bm25_by_id) | set(dense_by_id)
    merged: List[Dict[str, Any]] = []
    for doc_id in all_ids:
        s = bm25_by_id.get(doc_id)
        d = dense_by_id.get(doc_id)
        base = dict(s) if s else dict(d)
        sparse = float(s["sparse_score"]) if s else 0.0
        dense = float(d["vector_score"]) if d else 0.0
        base["sparse_score"] = round(sparse, 4)
        base["vector_score"] = round(dense, 4)
        base["hybrid_score"] = round(w_s * sparse + w_d * dense, 4)
        merged.append(base)
    merged.sort(key=lambda x: x.get("hybrid_score") or 0.0, reverse=True)
    return merged


def _format_hit(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Public-format hit dict used by the legacy `retrieve` API."""
    return {
        "doc_id": chunk.get("doc_id", ""),
        "content": chunk.get("content", ""),
        "source": chunk.get("source", ""),
        "section": chunk.get("section", ""),
        "chunk_index": chunk.get("chunk_index", 0),
        "intent_tags": chunk.get("intent_tags", []),
        "vector_score": chunk.get("vector_score", 0.0),
        "sparse_score": chunk.get("sparse_score", 0.0),
        "hybrid_score": chunk.get("hybrid_score", 0.0),
        "rerank_score": chunk.get("rerank_score", 0.0),
        "score": chunk.get("rerank_score") or chunk.get("hybrid_score") or 0.0,
    }
