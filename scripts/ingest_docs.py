"""Ingest policy/FAQ markdown docs into the vector store."""
import logging
import os
from pathlib import Path

from app.config import settings
from app.rag.chunking import load_markdown_dir
from app.rag.embeddings import embed_texts
from app.rag.vector_store import VectorRecord, get_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    docs_dir = Path(os.environ.get("DOCS_DIR", "data/docs"))
    if not docs_dir.exists():
        raise SystemExit(f"Docs dir not found: {docs_dir}")
    chunks = load_markdown_dir(docs_dir, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    if not chunks:
        raise SystemExit("No chunks produced. Add markdown files to data/docs.")
    logger.info("chunk_count=%s", len(chunks))
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
        )
        for c, v in zip(chunks, vectors)
    ]
    store = get_vector_store()
    store.upsert(records)
    logger.info("ingest_done total_chunks=%s store_size=%s", len(records), store.count())


if __name__ == "__main__":
    main()
