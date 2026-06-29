"""Admin endpoints: doc ingestion, seeding, sample logs, agent registry, graph diagram, thread history."""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.agents.graph import get_orchestrator_graph
from app.agents.registry import (
    AGENT_REGISTRY,
    TERMINAL_NODES,
    get_agent,
    registry_for_api,
)
from app.config import settings
from app.db.schemas import (
    AgentListResponse,
    AgentSpecOut,
    CheckpointInfo,
    GraphResponse,
    ThreadHistoryResponse,
)
from app.db.seed import init_db
from app.rag.chunking import load_markdown_dir
from app.rag.embeddings import embed_texts
from app.rag.vector_store import VectorRecord, get_vector_store

router = APIRouter(prefix="/admin")
logger = logging.getLogger(__name__)


# --- Ingestion / seeding (unchanged behaviour) --------------------


@router.post("/ingest-docs")
def ingest_docs() -> Dict[str, Any]:
    """Ingest all markdown files in data/docs into the vector store."""
    docs_dir = Path(os.environ.get("DOCS_DIR", "data/docs"))
    chunks = load_markdown_dir(docs_dir, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(status_code=400, detail=f"No markdown files found in {docs_dir}")
    texts = [c.content for c in chunks]
    try:
        vectors = embed_texts(texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")
    records = [
        VectorRecord(
            id=c.doc_id,
            content=c.content,
            source=c.source,
            section=c.section,
            chunk_index=c.chunk_index,
            embedding=vec,
        )
        for c, vec in zip(chunks, vectors)
    ]
    store = get_vector_store()
    store.upsert(records)
    return {
        "status": "ok",
        "ingested_chunks": len(records),
        "store_size": store.count(),
    }


@router.post("/seed-data")
def seed_data() -> Dict[str, Any]:
    """Create database tables (does NOT populate data; use scripts/seed_database.py)."""
    init_db()
    return {"status": "ok", "message": "Tables ensured. Run scripts/seed_database.py to insert synthetic data."}


@router.get("/logs/sample")
def logs_sample() -> Dict[str, Any]:
    """Return sample (last 5) structured log events. Lightweight implementation."""
    from app.core.logging import get_logger

    log = get_logger("admin.sample")
    for i in range(3):
        log.info(
            "sample_event",
            extra={"request_id": f"req_sample_{i}", "status": "ok", "intent": "product_search"},
        )
    return {"status": "ok", "written": 3}


# --- Agent Registry introspection ---------------------------------


@router.get("/agents", response_model=AgentListResponse)
def list_agents() -> AgentListResponse:
    """List every registered agent and its metadata."""
    payload = registry_for_api()
    return AgentListResponse(
        agents=[AgentSpecOut(**a) for a in payload["agents"]],
        total=payload["total"],
        terminal_nodes=payload["terminal_nodes"],
    )


@router.get("/agents/{key}", response_model=AgentSpecOut)
def get_agent_spec(key: str) -> AgentSpecOut:
    """Return the AgentSpec for a single agent, or 404."""
    spec = get_agent(key)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Agent {key!r} not found in registry")
    return AgentSpecOut(**spec.to_dict())


# --- Graph diagram -----------------------------------------------


def _build_mermaid_diagram(graph) -> str:
    """Generate a Mermaid diagram string for the compiled graph.

    `get_graph().draw_mermaid()` is the standard LangGraph helper. We fall back to
    a hand-rolled diagram if the helper is unavailable in the installed version.
    """
    try:
        return graph.get_graph().draw_mermaid()
    except Exception:
        # Hand-rolled fallback: nodes and edges in registry order.
        lines = ["graph TD", "  START([START]) --> route_intent"]
        for key in AGENT_REGISTRY:
            lines.append(f"  route_intent -->|{key}| {key}")
            lines.append(f"  {key} --> response_agent")
        for term in TERMINAL_NODES:
            if term == "response_agent":
                lines.append("  response_agent --> END([END])")
            else:
                lines.append(f"  route_intent -->|{term}| {term}")
                lines.append(f"  {term} --> END([END])")
        return "\n".join(lines)


@router.get("/graph", response_model=GraphResponse)
def get_graph() -> GraphResponse:
    """Return a Mermaid diagram of the compiled LangGraph orchestrator."""
    graph = get_orchestrator_graph()
    diagram = _build_mermaid_diagram(graph)
    return GraphResponse(format="mermaid", diagram=diagram)


# --- Thread / checkpoint history ---------------------------------


@router.get("/threads/{thread_id}/history", response_model=ThreadHistoryResponse)
def get_thread_history(thread_id: str) -> ThreadHistoryResponse:
    """Return the list of checkpoints stored for a thread.

    The implementation depends on the configured checkpointer backend. For the
    in-memory backend this list is empty after a process restart. For the
    Postgres backend, we query the `checkpoints` table that LangGraph manages.
    """
    checkpoints: List[CheckpointInfo] = []
    backend = (settings.CHECKPOINT_BACKEND or "memory").lower()
    if backend == "postgres":
        try:
            from sqlalchemy import text

            from app.db.session import engine

            with engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT checkpoint_id, metadata, created_at "
                        "FROM checkpoints WHERE thread_id = :tid ORDER BY created_at ASC"
                    ),
                    {"tid": thread_id},
                ).fetchall()
            for row in rows:
                meta: Dict[str, Any] = {}
                if row[1]:
                    try:
                        meta = json.loads(row[1]) if isinstance(row[1], str) else dict(row[1])
                    except Exception:
                        meta = {}
                checkpoints.append(
                    CheckpointInfo(
                        checkpoint_id=row[0],
                        node=meta.get("node") or meta.get("source") or None,
                        created_at=str(row[2]) if row[2] is not None else None,
                        metadata=meta,
                    )
                )
        except Exception as e:
            logger.warning("thread_history_query_failed err=%s", str(e))
    return ThreadHistoryResponse(thread_id=thread_id, checkpoints=checkpoints, total=len(checkpoints))
