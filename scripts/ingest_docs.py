"""Ingest policy/FAQ markdown docs into:

* the vector store (Qdrant dense + in-memory fallback), and
* the on-disk BM25 index (`data/bm25_index.json`).

Each chunk is also tagged with `intent_tags` derived from the source file name
so the retriever can apply metadata filters like `intent=shipping_policy`.
"""
import logging
import os
from pathlib import Path

from app.config import settings
from app.rag.bm25_index import BM25_INDEX_PATH, BM25Index
from app.rag.chunking import DocumentChunk, load_markdown_dir
from app.rag.embeddings import embed_texts
from app.rag.vector_store import VectorRecord, get_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Source filename → list of intent tags. The retriever uses this for
# metadata filtering (e.g. `intent_filter=["shipping_policy"]`).
INTENT_TAGS_BY_SOURCE: dict[str, list[str]] = {
    "wholesale_policy.md": ["wholesale_pricing", "general_faq"],
    "return_refund_policy.md": ["return_refund", "general_faq"],
    "shipping_policy.md": ["shipping_policy", "general_faq"],
    "warranty_policy.md": ["warranty_policy", "general_faq"],
    "payment_terms.md": ["payment_terms", "general_faq"],
    "product_faq.md": ["general_faq", "product_search", "product_comparison"],
    "escalation_policy.md": ["human_escalation", "return_refund", "general_faq"],
}


def intent_tags_for(source: str) -> list[str]:
    return list(INTENT_TAGS_BY_SOURCE.get(source, ["general_faq"]))


def main() -> None:
    docs_dir = Path(os.environ.get("DOCS_DIR", "data/docs"))
    if not docs_dir.exists():
        raise SystemExit(f"Docs dir not found: {docs_dir}")

    chunks: list[DocumentChunk] = load_markdown_dir(
        docs_dir,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    if not chunks:
        raise SystemExit("No chunks produced. Add markdown files to data/docs.")
    logger.info("chunk_count=%s", len(chunks))

    # 1) Dense embeddings + vector store
    texts = [c.content for c in chunks]
    vectors = embed_texts(texts)
    records = [
        VectorRecord(
            id=c.doc_id,
            content=c.content,
            source=c.source,
            section=c.section,
            chunk_index=c.chunk_index,
            embedding=v,
            intent_tags=intent_tags_for(c.source),
        )
        for c, v in zip(chunks, vectors)
    ]
    store = get_vector_store()
    store.upsert(records)
    logger.info(
        "vector_upsert_done total_chunks=%s store_size=%s", len(records), store.count()
    )

    # 2) BM25 index (in-memory + on-disk JSON)
    bm25 = BM25Index()
    for c in chunks:
        bm25.add(
            doc_id=c.doc_id,
            content=c.content,
            source=c.source,
            section=c.section,
            chunk_index=c.chunk_index,
            intent_tags=intent_tags_for(c.source),
        )
    bm25.save(BM25_INDEX_PATH)
    logger.info("bm25_index_saved path=%s docs=%s", BM25_INDEX_PATH, len(bm25.docs))


if __name__ == "__main__":
    main()
