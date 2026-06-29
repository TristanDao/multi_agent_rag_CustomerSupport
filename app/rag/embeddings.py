"""Embedding model wrapper.

Primary: Alibaba `text-embedding-v3` via the OpenAI-compatible `/v1/embeddings`
endpoint (the OpenAI Python SDK is reused as the transport).
Fallback: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
Last-resort: deterministic hash-based vector so the system still functions
offline (only useful for dev / tests).
"""
import hashlib
import logging
import math
import os
import threading
from typing import List, Optional

from app.config import settings

logger = logging.getLogger(__name__)

_st_model = None
_st_model_name: Optional[str] = None
_st_lock = threading.Lock()


def _load_sentence_transformer(name: str):
    global _st_model, _st_model_name
    with _st_lock:
        if _st_model is not None and _st_model_name == name:
            return _st_model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            cache_dir = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
            _st_model = SentenceTransformer(name, cache_folder=cache_dir)
            _st_model_name = name
            logger.info("embedding_model_loaded provider=sentence_transformers name=%s", name)
            return _st_model
        except Exception as e:
            logger.warning("embedding_model_load_failed name=%s err=%s", name, str(e))
            _st_model = None
            return None


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
        logger.warning("alibaba_embedding_failed err=%s; will use fallback", str(e)[:200])
        return None


def _st_embed(texts: List[str]) -> Optional[List[List[float]]]:
    model = _load_sentence_transformer(settings.EMBEDDING_FALLBACK_MODEL)
    if model is None:
        return None
    try:
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]
    except Exception as e:
        logger.warning("sentence_transformer_embed_failed err=%s", str(e))
        return None


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Return one embedding vector per input text.

    Selection order:
      1. `EMBEDDING_PROVIDER=alibaba` → Alibaba text-embedding-v3
      2. `EMBEDDING_PROVIDER=sentence_transformers` → sentence-transformers
      3. Auto-detect: try Alibaba if key configured, else sentence-transformers
      4. Last resort: hash-based vector
    """
    if not texts:
        return []
    provider = (settings.EMBEDDING_PROVIDER or "auto").lower()
    vectors: Optional[List[List[float]]] = None
    if provider in ("alibaba", "auto") and settings.has_alibaba_key:
        vectors = _alibaba_embed(texts)
    if vectors is None and provider in ("sentence_transformers", "auto"):
        vectors = _st_embed(texts)
    if vectors is None:
        logger.warning("embedding_falling_back_to_hash dim=%s", settings.EMBEDDING_DIM)
        return [_hash_embed(t, dim=settings.EMBEDDING_DIM) for t in texts]
    return vectors


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]


def embedding_dim() -> int:
    """Return the active embedding dimension."""
    provider = (settings.EMBEDDING_PROVIDER or "auto").lower()
    if provider == "alibaba" and settings.has_alibaba_key:
        return int(settings.EMBEDDING_DIM)
    if provider == "sentence_transformers":
        model = _load_sentence_transformer(settings.EMBEDDING_FALLBACK_MODEL)
        if model is not None:
            try:
                return int(model.get_sentence_embedding_dimension())
            except Exception:
                pass
        return int(settings.EMBEDDING_FALLBACK_DIM)
    # auto-detect
    if settings.has_alibaba_key:
        return int(settings.EMBEDDING_DIM)
    model = _load_sentence_transformer(settings.EMBEDDING_FALLBACK_MODEL)
    if model is not None:
        try:
            return int(model.get_sentence_embedding_dimension())
        except Exception:
            pass
    return int(settings.EMBEDDING_DIM)


def _hash_embed(text: str, dim: int = 384) -> List[float]:
    """Deterministic fallback embedding so the system still functions offline."""
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
