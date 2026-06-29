"""Checkpointer factory for the LangGraph orchestrator.

Two backends are supported:

* `"postgres"` — production. Uses `langgraph-checkpoint-postgres` against the same
  PostgreSQL the app already talks to. Persists state in a `checkpoints` table.
* `"memory"` — local dev and tests. Uses an in-memory `MemorySaver` so the project
  can run without a Postgres connection (e.g. unit tests, the diagram generator).

Selection is driven by `settings.CHECKPOINT_BACKEND`.
"""
from __future__ import annotations

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


def build_checkpointer() -> Any:
    """Return a LangGraph-compatible checkpointer based on settings.

    The return type is intentionally `Any` because the concrete class comes from
    either `langgraph.checkpoint.memory` or `langgraph.checkpoint.postgres`.
    """
    backend = (settings.CHECKPOINT_BACKEND or "memory").lower()
    if backend == "postgres":
        return _build_postgres_checkpointer()
    return _build_memory_checkpointer()


def _build_memory_checkpointer():
    from langgraph.checkpoint.memory import MemorySaver

    logger.info("checkpointer_backend=memory (in-process, not persisted)")
    return MemorySaver()


def _build_postgres_checkpointer():
    try:
        from langgraph.checkpoint.postgres import PostgresCheckpoint
    except ImportError as e:
        logger.warning(
            "langgraph-checkpoint-postgres not installed, falling back to memory backend: %s", e
        )
        return _build_memory_checkpointer()

    url = settings.DATABASE_URL
    # SQLAlchemy URL → libpq URL expected by PostgresCheckpoint
    if url.startswith("postgresql+psycopg2://"):
        conn_string = "postgresql://" + url[len("postgresql+psycopg2://"):]
    elif url.startswith("postgresql+"):
        conn_string = "postgresql://" + url.split("://", 1)[1]
    else:
        conn_string = url

    try:
        cp = PostgresCheckpoint.from_conn_string(conn_string)
        logger.info("checkpointer_backend=postgres url=%s", conn_string)
        return cp
    except Exception as e:
        logger.warning(
            "postgres checkpointer init failed (%s), falling back to memory backend", e
        )
        return _build_memory_checkpointer()
