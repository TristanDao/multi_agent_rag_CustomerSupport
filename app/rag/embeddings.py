"""Embedding model wrapper. Sentence Transformers first, with safe fallback."""
import logging
import os
from typing import List, Optional

from app.config import settings

logger = logging.getLogger(__name__)

_model = None
_model_name_used: Optional[str] = None


def _get_model():
    global _model, _model_name_used
    if _model is not None:
        return _model
    name = settings.EMBEDDING_MODEL
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        cache_dir = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
        _model = SentenceTransformer(name, cache_folder=cache_dir)
        _model_name_used = name
        logger.info("embedding_model_loaded name=%s", name)
    except Exception as e:
        logger.warning("embedding_model_load_failed err=%s; falling back to hash embeddings", str(e))
        _model = None
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = _get_model()
    if model is not None:
        try:
            vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return [v.tolist() for v in vectors]
        except Exception as e:
            logger.warning("embedding_failed err=%s; using hash fallback", str(e))
    return [_hash_embed(t) for t in texts]


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]


def embedding_dim() -> int:
    model = _get_model()
    if model is not None:
        try:
            return int(model.get_sentence_embedding_dimension())
        except Exception:
            pass
    return settings.EMBEDDING_DIM


def _hash_embed(text: str, dim: int = 384) -> List[float]:
    """Deterministic fallback embedding so the system still functions offline."""
    import hashlib
    import math

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
