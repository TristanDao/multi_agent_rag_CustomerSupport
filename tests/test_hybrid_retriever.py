"""Tests for the hybrid retriever (BM25 + dense + filter + rerank) and the
related RAG modules.

These tests do NOT require an Alibaba API key or a running Qdrant — they run
entirely in-process with hash embeddings and the no-op rerank fallback.
"""
import os

import pytest


@pytest.fixture(autouse=True)
def _force_offline_rag(monkeypatch):
    """Force the offline fallbacks so the tests don't hit any external service.

    With no `ALIBABA_API_KEY` in the test env, the embedding path falls back to
    a hash-based vector; with `RERANK_PROVIDER=none` the rerank is a no-op.
    """
    monkeypatch.delenv("ALIBABA_API_KEY", raising=False)
    monkeypatch.setenv("EMBEDDING_PROVIDER", "alibaba")  # only "alibaba" is supported
    monkeypatch.setenv("RERANK_PROVIDER", "none")
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    from app.config import get_settings
    from app.core.llm import reset_client
    from app.core.observability import reset_langfuse
    from app.rag import bm25_index as bm25_mod
    from app.rag import vector_store as vs_mod

    get_settings.cache_clear()
    reset_client()
    reset_langfuse()
    bm25_mod.reset_bm25_index()
    vs_mod.reset_vector_store()
    yield
    bm25_mod.reset_bm25_index()
    vs_mod.reset_vector_store()
    get_settings.cache_clear()


def _build_small_corpus():
    """Two short documents with clear intent tags."""
    from app.rag.bm25_index import BM25Index
    from app.rag.vector_store import InMemoryVectorStore, VectorRecord
    from app.rag.embeddings import embed_texts

    docs = [
        {
            "doc_id": "shipping_policy.md#city#0",
            "content": "Phí vận chuyển nội thành HCM là 15.000đ. Đơn từ 500.000đ miễn phí.",
            "source": "shipping_policy.md",
            "section": "Phí vận chuyển nội thành",
            "intent_tags": ["shipping_policy", "general_faq"],
        },
        {
            "doc_id": "return_refund_policy.md#window#0",
            "content": "Đơn hàng được đổi trả trong vòng 7 ngày kể từ khi nhận hàng.",
            "source": "return_refund_policy.md",
            "section": "Thời hạn đổi trả",
            "intent_tags": ["return_refund", "general_faq"],
        },
        {
            "doc_id": "wholesale_policy.md#tier#0",
            "content": "Bảng giá sỉ theo số lượng: mua 50 thùng giấy A4 được giảm 8%.",
            "source": "wholesale_policy.md",
            "section": "Bậc giá theo số lượng",
            "intent_tags": ["wholesale_pricing", "general_faq"],
        },
    ]
    # 1) BM25
    bm25 = BM25Index()
    for d in docs:
        bm25.add(
            doc_id=d["doc_id"],
            content=d["content"],
            source=d["source"],
            section=d["section"],
            chunk_index=0,
            intent_tags=d["intent_tags"],
        )
    # 2) In-memory vector store (use the real embed_texts to stay consistent
    #    with the prod retriever interface).
    vectors = embed_texts([d["content"] for d in docs])
    store = InMemoryVectorStore()
    for d, v in zip(docs, vectors):
        store.upsert(
            [
                VectorRecord(
                    id=d["doc_id"],
                    content=d["content"],
                    source=d["source"],
                    section=d["section"],
                    chunk_index=0,
                    embedding=v,
                    intent_tags=d["intent_tags"],
                )
            ]
        )
    # 3) Inject the BM25 index into the module-level singleton
    from app.rag import bm25_index as bm25_mod

    bm25_mod._index = bm25
    from app.rag import vector_store as vs_mod

    vs_mod._store = store
    return docs


# --- BM25 unit tests ---------------------------------------------------


def test_bm25_tokenise():
    from app.rag.bm25_index import tokenise

    out = tokenise("Phí ship nội thành 15.000đ")
    # Vietnamese word characters stick together; numbers separated by `.` split.
    assert "phí" in out
    assert "ship" in out
    assert "nội" in out
    assert "thành" in out
    assert "15" in out
    assert tokenise("") == []


def test_bm25_search_returns_relevant_doc():
    from app.rag.bm25_index import BM25Index

    idx = BM25Index()
    idx.add(
        doc_id="a", content="phí vận chuyển nội thành HCM 15.000đ",
        source="shipping_policy.md", section="x", chunk_index=0,
        intent_tags=["shipping_policy"],
    )
    idx.add(
        doc_id="b", content="đổi trả trong 7 ngày",
        source="return_refund_policy.md", section="x", chunk_index=0,
        intent_tags=["return_refund"],
    )
    hits = idx.search("phí ship HCM", top_k=2)
    assert hits and hits[0].doc_id == "a"
    assert hits[0].score > 0


def test_bm25_search_with_intent_filter():
    from app.rag.bm25_index import BM25Index

    idx = BM25Index()
    idx.add("a", "phí vận chuyển", "s.md", "x", 0, intent_tags=["shipping_policy"])
    idx.add("b", "đổi trả 7 ngày", "r.md", "x", 0, intent_tags=["return_refund"])
    hits = idx.search("chính sách", top_k=5, intent_filter=["return_refund"])
    assert all(h.doc_id == "b" for h in hits)


def test_bm25_persistence_roundtrip(tmp_path):
    from app.rag.bm25_index import BM25Index

    p = tmp_path / "idx.json"
    idx = BM25Index()
    idx.add("a", "alpha beta gamma", "s.md", "x", 0, intent_tags=["t1"])
    idx.add("b", "delta echo", "s.md", "x", 0, intent_tags=["t2"])
    idx.save(str(p))

    loaded = BM25Index.load(str(p))
    assert loaded is not None
    assert len(loaded.docs) == 2
    hits = loaded.search("alpha", top_k=1)
    assert hits and hits[0].doc_id == "a"


# --- Rerank no-op fallback --------------------------------------------


def test_rerank_noop_fallback_preserves_order():
    from app.rag.rerank import rerank

    candidates = [
        {"doc_id": "a", "content": "x", "vector_score": 0.9, "sparse_score": 0.8, "hybrid_score": 0.85},
        {"doc_id": "b", "content": "y", "vector_score": 0.5, "sparse_score": 0.4, "hybrid_score": 0.45},
    ]
    out = rerank("test", candidates, top_k=2)
    assert [c["doc_id"] for c in out] == ["a", "b"]
    assert all("rerank_score" in c for c in out)


# --- Hybrid retriever end-to-end ---------------------------------------


def test_hybrid_retrieve_returns_relevant_doc_for_shipping_query():
    _build_small_corpus()
    from app.rag.retriever import hybrid_retrieve

    out = hybrid_retrieve("phí ship nội thành HCM", top_k=3, use_rerank=False)
    assert out, "retriever should return at least one hit"
    assert out[0]["source"] == "shipping_policy.md"
    # All required score fields are present
    for c in out:
        assert {"doc_id", "content", "source", "section", "intent_tags",
                "vector_score", "sparse_score", "hybrid_score",
                "rerank_score"} <= set(c.keys())


def test_hybrid_retrieve_intent_filter_narrows_results():
    _build_small_corpus()
    from app.rag.retriever import hybrid_retrieve

    out = hybrid_retrieve(
        "chính sách", top_k=5, use_rerank=False,
        intent_filter=["return_refund"],
    )
    assert out
    assert all("return_refund" in (c.get("intent_tags") or []) for c in out)


def test_hybrid_retrieve_source_filter():
    _build_small_corpus()
    from app.rag.retriever import hybrid_retrieve

    out = hybrid_retrieve(
        "chính sách", top_k=5, use_rerank=False,
        source_filter=["wholesale_policy.md"],
    )
    assert out
    assert all(c["source"] == "wholesale_policy.md" for c in out)


def test_hybrid_retrieve_legacy_compat_wrapper():
    _build_small_corpus()
    from app.rag.retriever import retrieve

    out = retrieve("phí ship")
    assert isinstance(out, list)
    if out:
        # legacy `retrieve` should still expose a single `score` field
        assert "score" in out[0]
