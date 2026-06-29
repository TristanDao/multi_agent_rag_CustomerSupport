"""Embedding model wrapper.

Alibaba is the sole embedding provider. Primary path is **Alibaba
`text-embedding-v3`** (1024-dim) via the OpenAI-compatible `/v1/embeddings`
endpoint — the OpenAI Python SDK is reused as the transport.

If the Alibaba call fails (no key, network error, rate limit), the retriever
falls back to a deterministic hash-based vector. This is **offline dev only**:
it produces meaningless vectors and the retrieval quality will be poor. There
is no second model provider.
"""
import hashlib
import logging
import math
import threading
from typing import List, Optional

from app.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def _alibaba_embed(texts: List[str]) -> Optional[List[List[float]]]:
    """Call Alibaba text-embedding-v3 via the OpenAI SDK. Returns None on failure."""
    if not settings.has_alibaba_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=settings.llm_api_key(), base_url=settings.llm_base_url())
        resp = client.embeddings.create(model=settings.EMBEDDING_MODEL, input=texts)
        return [list(d.embedding) for d in resp.data]
    except Exception as e:
        logger.warning("alibaba_embedding_failed err=%s; will fall back to hash", str(e)[:200])
        return None


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return one embedding vector per input text.

    Selection order:
      1. Alibaba `text-embedding-v3` (via OpenAI-compatible API)
      2. Last-resort hash-based vector (offline dev only; quality is poor)
    """
    if not texts:
        return []
    with _lock:
        vectors = _alibaba_embed(texts)
    if vectors is None:
        logger.warning(
            "embedding_falling_back_to_hash dim=%s (offline dev only — quality will be poor)",
            settings.EMBEDDING_DIM,
        )
        return [_hash_embed(t, dim=settings.EMBEDDING_DIM) for t in texts]
    return vectors


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]


def embedding_dim() -> int:
    """Return the active embedding dimension (Alibaba primary)."""
    return int(settings.EMBEDDING_DIM)


def _hash_embed(text: str, dim: int = 384) -> List[float]:
    """Deterministic fallback embedding so the system still functions offline.

    WARNING: this is for offline dev only. It produces vectors that are not
    semantically meaningful — retrieval quality will be poor. Always restore
    Alibaba connectivity for production.
    """
    vec = [0.0] * dim
    tokens = text.lower().split()
    if not tokens:
        return vec
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]
