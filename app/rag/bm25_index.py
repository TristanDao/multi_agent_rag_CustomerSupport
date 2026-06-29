"""In-memory BM25 index over the document chunks.

This is the sparse leg of the hybrid retriever. The index is rebuilt at ingest
time (`scripts/ingest_docs.py`) by calling `build_index(chunks)`. The retriever
queries it in parallel with the dense Qdrant search.

A persistent-on-disk JSON file (`data/bm25_index.json`) is also written so the
index survives a process restart without re-running ingest.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Iterable, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


# Default location used by `scripts/ingest_docs.py` and the retriever.
BM25_INDEX_PATH = os.environ.get("BM25_INDEX_PATH", "data/bm25_index.json")


# --- Tokeniser ---------------------------------------------------------


_TOKEN_RE = re.compile(r"[\wÀ-￿]+", re.UNICODE)


def tokenise(text: str) -> List[str]:
    """Lowercase, unicode-aware word tokeniser (good enough for VN + EN)."""
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# --- Data --------------------------------------------------------------


@dataclass
class BM25Hit:
    doc_id: str
    content: str
    source: str
    section: str
    chunk_index: int
    intent_tags: List[str]
    score: float


# --- Index -------------------------------------------------------------


class BM25Index:
    """In-memory BM25Okapi index.

    Implements BM25 with the Okapi scoring variant. `k1` and `b` are the standard
    tunables. We keep the index small enough to fit in memory for the demo
    corpus (~7 markdown files, < 1000 chunks).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: List[dict] = []          # metadata per doc
        self.doc_tokens: List[List[str]] = []  # tokenised content per doc
        self.doc_lens: List[int] = []
        self.avgdl: float = 0.0
        self.df: dict = {}                  # term → doc frequency
        self._lock = threading.RLock()

    # --- build ----------------------------------------------------------

    def add(self, doc_id: str, content: str, source: str, section: str,
            chunk_index: int, intent_tags: Optional[List[str]] = None) -> None:
        tokens = tokenise(content)
        with self._lock:
            self.docs.append({
                "doc_id": doc_id,
                "content": content,
                "source": source,
                "section": section,
                "chunk_index": chunk_index,
                "intent_tags": list(intent_tags or []),
            })
            self.doc_tokens.append(tokens)
            self.doc_lens.append(len(tokens))
            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1
            self._recompute_avgdl()

    def _recompute_avgdl(self) -> None:
        if not self.doc_lens:
            self.avgdl = 0.0
        else:
            self.avgdl = sum(self.doc_lens) / len(self.doc_lens)

    # --- query ----------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 20,
        intent_filter: Optional[Iterable[str]] = None,
        source_filter: Optional[Iterable[str]] = None,
    ) -> List[BM25Hit]:
        q_tokens = tokenise(query)
        if not q_tokens or not self.docs:
            return []
        intent_set = set(intent_filter) if intent_filter else None
        source_set = set(source_filter) if source_filter else None
        N = len(self.docs)
        scores: List[tuple] = []
        with self._lock:
            for i, tokens in enumerate(self.doc_tokens):
                meta = self.docs[i]
                if intent_set is not None and not (intent_set & set(meta.get("intent_tags") or [])):
                    continue
                if source_set is not None and meta.get("source") not in source_set:
                    continue
                if not tokens:
                    continue
                dl = self.doc_lens[i]
                # term frequency map for this doc
                tf: dict = {}
                for t in tokens:
                    tf[t] = tf.get(t, 0) + 1
                score = 0.0
                for qt in q_tokens:
                    if qt not in tf:
                        continue
                    f = tf[qt]
                    n_q = self.df.get(qt, 0)
                    # IDF with +1 smoothing to avoid negatives
                    idf = math_log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)
                    denom = f + self.k1 * (1.0 - self.b + self.b * (dl / (self.avgdl or 1.0)))
                    score += idf * (f * (self.k1 + 1.0)) / (denom or 1.0)
                if score > 0:
                    scores.append((score, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        results: List[BM25Hit] = []
        for s, i in scores[:top_k]:
            meta = self.docs[i]
            results.append(
                BM25Hit(
                    doc_id=meta["doc_id"],
                    content=meta["content"],
                    source=meta["source"],
                    section=meta["section"],
                    chunk_index=meta["chunk_index"],
                    intent_tags=meta.get("intent_tags") or [],
                    score=float(s),
                )
            )
        return results

    # --- persistence ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "docs": self.docs,
            "doc_tokens": self.doc_tokens,
            "doc_lens": self.doc_lens,
            "avgdl": self.avgdl,
            "df": self.df,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BM25Index":
        idx = cls(k1=data.get("k1", 1.5), b=data.get("b", 0.75))
        idx.docs = list(data.get("docs") or [])
        idx.doc_tokens = [list(t) for t in (data.get("doc_tokens") or [])]
        idx.doc_lens = list(data.get("doc_lens") or [])
        idx.avgdl = float(data.get("avgdl") or 0.0)
        idx.df = dict(data.get("df") or {})
        return idx

    def save(self, path: str = BM25_INDEX_PATH) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)
        logger.info("bm25_index_saved path=%s docs=%s", path, len(self.docs))

    @classmethod
    def load(cls, path: str = BM25_INDEX_PATH) -> Optional["BM25Index"]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            idx = cls.from_dict(data)
            logger.info("bm25_index_loaded path=%s docs=%s", path, len(idx.docs))
            return idx
        except Exception as e:
            logger.warning("bm25_index_load_failed path=%s err=%s", path, str(e))
            return None


def math_log(x: float) -> float:
    import math
    return math.log(x)


# --- Module-level singleton -------------------------------------------

_lock = threading.Lock()
_index: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """Return the process-wide BM25 index (loads from disk on first call)."""
    global _index
    if _index is not None:
        return _index
    with _lock:
        if _index is None:
            loaded = BM25Index.load()
            _index = loaded if loaded is not None else BM25Index()
    return _index


def reset_bm25_index() -> None:
    """Drop the cached index. Used by tests / after re-ingest."""
    global _index
    _index = None
