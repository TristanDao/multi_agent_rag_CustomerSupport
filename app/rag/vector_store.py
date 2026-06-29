"""Vector store abstraction. Qdrant primary, in-memory fallback for dev/offline."""
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class VectorRecord:
    id: str
    content: str
    source: str
    section: str
    chunk_index: int
    embedding: List[float] = field(default_factory=list)
    intent_tags: List[str] = field(default_factory=list)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.records: List[VectorRecord] = []

    def upsert(self, records: List[VectorRecord]) -> None:
        existing = {r.id: i for i, r in enumerate(self.records)}
        for r in records:
            if r.id in existing:
                self.records[existing[r.id]] = r
            else:
                self.records.append(r)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        intent_filter: Optional[Iterable[str]] = None,
        source_filter: Optional[Iterable[str]] = None,
    ) -> List[Tuple[VectorRecord, float]]:
        if not self.records:
            return []
        intent_set = set(intent_filter) if intent_filter else None
        source_set = set(source_filter) if source_filter else None
        scored: List[Tuple[VectorRecord, float]] = []
        for r in self.records:
            if intent_set is not None and not (intent_set & set(r.intent_tags or [])):
                continue
            if source_set is not None and r.source not in source_set:
                continue
            if not r.embedding:
                continue
            score = _cosine(query_vector, r.embedding)
            scored.append((r, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self.records)

    def reset(self) -> None:
        self.records.clear()


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class QdrantVectorStore:
    def __init__(self, url: str, collection: str, dim: int) -> None:
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.http import models  # type: ignore

        self._models = models
        self.client = QdrantClient(url=url, prefer_grpc=False, timeout=10.0)
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            existing = {c.name for c in self.client.get_collections().collections}
            if self.collection not in existing:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=self._models.VectorParams(
                        size=self.dim, distance=self._models.Distance.COSINE
                    ),
                )
                logger.info("qdrant_collection_created name=%s", self.collection)
        except Exception as e:
            logger.warning("qdrant_ensure_failed err=%s", str(e))
            raise

    def upsert(self, records: List[VectorRecord]) -> None:
        if not records:
            return
        points = [
            self._models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, r.id)),
                vector=r.embedding,
                payload={
                    "doc_id": r.id,
                    "content": r.content,
                    "source": r.source,
                    "section": r.section,
                    "chunk_index": r.chunk_index,
                    "intent_tags": list(r.intent_tags or []),
                },
            )
            for r in records
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def _build_filter(
        self,
        intent_filter: Optional[Iterable[str]] = None,
        source_filter: Optional[Iterable[str]] = None,
    ):
        """Build a Qdrant Filter from the simple intent/source filters.

        `intent_filter` matches if ANY of the listed intents is in `intent_tags`
        (OR semantics). `source_filter` matches if `source` is one of the
        listed values (OR).
        """
        m = self._models
        must: List[Any] = []
        if intent_filter:
            intents = list(intent_filter)
            must.append(
                m.FieldCondition(
                    key="intent_tags",
                    match=m.MatchAny(any=intents),
                )
            )
        if source_filter:
            sources = list(source_filter)
            must.append(
                m.FieldCondition(
                    key="source",
                    match=m.MatchAny(any=sources),
                )
            )
        if not must:
            return None
        return m.Filter(must=must)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        intent_filter: Optional[Iterable[str]] = None,
        source_filter: Optional[Iterable[str]] = None,
    ) -> List[Tuple[VectorRecord, float]]:
        flt = self._build_filter(intent_filter=intent_filter, source_filter=source_filter)
        try:
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                query_filter=flt,
                limit=top_k,
                with_payload=True,
            )
        except Exception as e:
            logger.warning("qdrant_search_failed err=%s", str(e))
            return []
        results: List[Tuple[VectorRecord, float]] = []
        for h in hits:
            payload = h.payload or {}
            rec = VectorRecord(
                id=payload.get("doc_id", str(h.id)),
                content=payload.get("content", ""),
                source=payload.get("source", ""),
                section=payload.get("section", ""),
                chunk_index=int(payload.get("chunk_index", 0)),
                intent_tags=list(payload.get("intent_tags") or []),
            )
            results.append((rec, float(h.score or 0.0)))
        return results

    def count(self) -> int:
        try:
            info = self.client.get_collection(self.collection)
            return int(getattr(info, "points_count", 0) or 0)
        except Exception:
            return 0

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection)
        except Exception:
            pass
        self._ensure_collection()


_store: Optional[Any] = None


def get_vector_store() -> Any:
    global _store
    if _store is not None:
        return _store
    dim = settings.EMBEDDING_DIM
    try:
        from app.rag.embeddings import embedding_dim

        dim = embedding_dim() or dim
    except Exception:
        pass
    try:
        _store = QdrantVectorStore(settings.QDRANT_URL, settings.QDRANT_COLLECTION, dim)
        logger.info("vector_store=qdrant url=%s", settings.QDRANT_URL)
    except Exception as e:
        logger.warning("qdrant_unavailable err=%s; using in-memory store", str(e))
        _store = InMemoryVectorStore()
    return _store


def reset_vector_store() -> None:
    """Drop the cached store. Used in tests."""
    global _store
    _store = None
