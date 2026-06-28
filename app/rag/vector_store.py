"""Vector store abstraction. Qdrant primary, in-memory fallback for dev/offline."""
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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

    def search(self, query_vector: List[float], top_k: int = 4) -> List[Tuple[VectorRecord, float]]:
        if not self.records:
            return []
        scored: List[Tuple[VectorRecord, float]] = []
        for r in self.records:
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
                },
            )
            for r in records
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vector: List[float], top_k: int = 4) -> List[Tuple[VectorRecord, float]]:
        try:
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
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
