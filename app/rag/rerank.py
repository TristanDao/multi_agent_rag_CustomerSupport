"""Rerank module.

Primary: Alibaba `gte-rerank` (DashScope has a separate, non-OpenAI-compatible
REST endpoint for rerank — see `ALIBABA_RERANK_URL` in config).

Fallback (`RERANK_PROVIDER=none` or no API key): return the input candidates
sorted by their existing `hybrid_score`, no rerank. The retriever treats this
case identically to "rerank is a no-op".

Input/output contract for the retriever:
    rerank(query, candidates, top_k) -> list[candidate]
    where each candidate is a dict with at least {"doc_id", "content", ...,
    "vector_score", "sparse_score", "hybrid_score"} and is returned with an
    added "rerank_score" field (or 0.0 if no rerank happened).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Re-order `candidates` by relevance to `query`. Returns at most `top_k`.

    Adds (or overwrites) a `"rerank_score"` field on each returned dict.
    """
    if not candidates:
        return []
    provider = (settings.RERANK_PROVIDER or "alibaba").lower()
    if provider == "alibaba" and settings.has_alibaba_key:
        try:
            return _alibaba_rerank(query, candidates, top_k)
        except Exception as e:
            logger.warning("alibaba_rerank_failed err=%s; using hybrid fallback", str(e)[:200])
    return _no_rerank(candidates, top_k)


def _no_rerank(candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """No-op rerank: keep the order, tag `rerank_score=hybrid_score`."""
    out: List[Dict[str, Any]] = []
    for c in candidates:
        item = dict(c)
        item["rerank_score"] = float(item.get("hybrid_score") or item.get("vector_score") or 0.0)
        out.append(item)
    return out[:top_k]


def _alibaba_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Call Alibaba DashScope `gte-rerank` and return top_k."""
    if not candidates:
        return []
    # DashScope gte-rerank API: POST {RERANK_URL}
    #   body: {model, input: {query, documents}, parameters: {top_n, ...}}
    #   response: {output: {results: [{index, relevance_score}, ...]}}
    documents = [str(c.get("content", "")) for c in candidates]
    payload: Dict[str, Any] = {
        "model": settings.RERANK_MODEL,
        "input": {"query": query, "documents": documents},
        "parameters": {"top_n": min(int(top_k), len(candidates))},
    }
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key()}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(settings.ALIBABA_RERANK_URL, json=payload, headers=headers)
        if resp.status_code >= 400:
            logger.warning(
                "alibaba_rerank_http_error status=%s body=%s",
                resp.status_code,
                resp.text[:300],
            )
            return _no_rerank(candidates, top_k)
        data = resp.json()

    results = ((data.get("output") or {}).get("results")) or []
    reranked: List[Dict[str, Any]] = []
    for item in results:
        try:
            idx = int(item["index"])
            rel = float(item.get("relevance_score", 0.0))
        except (KeyError, ValueError, TypeError):
            continue
        if 0 <= idx < len(candidates):
            merged = dict(candidates[idx])
            merged["rerank_score"] = rel
            reranked.append(merged)
    # If the API returned fewer than expected, fill the rest from the input
    # order (preserves the dense+BM25 merge order).
    if len(reranked) < top_k:
        seen = {c.get("doc_id") for c in reranked}
        for c in candidates:
            if c.get("doc_id") in seen:
                continue
            merged = dict(c)
            merged["rerank_score"] = float(c.get("hybrid_score") or c.get("vector_score") or 0.0)
            reranked.append(merged)
            if len(reranked) >= top_k:
                break
    return reranked[:top_k]
