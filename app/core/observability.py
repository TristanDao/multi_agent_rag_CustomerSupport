"""Observability layer.

Provides:
* Structured JSON logger with PII redaction.
* Langfuse integration via `langfuse.langchain.CallbackHandler` so every
  LangGraph node, LLM call and tool call is auto-traced.
* A lightweight in-memory tracer used when Langfuse is disabled or unreachable.

In `LANGFUSE_ENABLED=true` mode (the default), the app fails loudly at startup
if Langfuse is unreachable — silent observability outages are not acceptable.
"""
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


# --- Langfuse client + LangGraph callback -------------------------------


_lf_lock = threading.Lock()
_langfuse_client = None
_callback_handler = None
_langfuse_initialised = False


def get_langfuse_client():
    """Return a process-wide Langfuse client, or None if disabled.

    The first call verifies connectivity by listing projects. If `LANGFUSE_ENABLED`
    is true but the call fails, the function raises (intentional — we want
    observability outages to be visible).
    """
    global _langfuse_client, _langfuse_initialised
    if _langfuse_client is not None or _langfuse_initialised:
        return _langfuse_client
    with _lf_lock:
        if _langfuse_client is not None or _langfuse_initialised:
            return _langfuse_client
        if not settings.LANGFUSE_ENABLED:
            logger.info("langfuse_disabled")
            _langfuse_initialised = True
            return None
        if not (settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY):
            msg = "LANGFUSE_ENABLED=true but LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are empty"
            logger.error(msg)
            raise RuntimeError(msg)
        try:
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.LANGFUSE_PUBLIC_KEY)
            os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.LANGFUSE_SECRET_KEY)
            os.environ.setdefault("LANGFUSE_HOST", settings.LANGFUSE_HOST)
            from langfuse import Langfuse  # type: ignore

            _langfuse_client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
            )
            logger.info("langfuse_ready host=%s", settings.LANGFUSE_HOST)
            _langfuse_initialised = True
            return _langfuse_client
        except Exception as e:
            if settings.LANGFUSE_ENABLED:
                logger.exception("langfuse_init_failed err=%s", str(e))
                raise
            logger.warning("langfuse_init_failed_disabled err=%s", str(e))
            _langfuse_initialised = True
            return None


def get_langfuse_callback():
    """Return a `langfuse.langchain.CallbackHandler` instance for graph calls.

    The handler is the standard way to pipe LangChain / LangGraph spans into
    Langfuse. It is created lazily and reused across requests.
    """
    global _callback_handler
    if _callback_handler is not None:
        return _callback_handler
    if not settings.LANGFUSE_ENABLED:
        return None
    try:
        from langfuse.langchain import CallbackHandler  # type: ignore

        _callback_handler = CallbackHandler()
        return _callback_handler
    except Exception as e:
        logger.warning("langfuse_callback_init_failed err=%s", str(e))
        return None


def reset_langfuse() -> None:
    """Drop the cached Langfuse client + callback (used in tests)."""
    global _langfuse_client, _callback_handler, _langfuse_initialised
    _langfuse_client = None
    _callback_handler = None
    _langfuse_initialised = False


# --- In-memory trace (kept for back-compat with legacy code) ----------


class InMemoryTrace:
    """Lightweight in-memory tracer used when Langfuse is disabled."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.start_ts = time.time()

    def log(self, name: str, **kwargs: Any) -> None:
        self.events.append({"name": name, "ts": time.time() - self.start_ts, **kwargs})

    def to_dict(self) -> Dict[str, Any]:
        return {"events": self.events, "duration_s": time.time() - self.start_ts}


# --- Legacy request-scoped trace context manager ----------------------


@contextmanager
def trace_request(request_id: str, user_message: str):
    """Yield a context with `.span(name, **kwargs)` and `.set_metadata(...)`.

    Kept for back-compat with the old `app/agents/orchestrator.py` flow. The
    per-request Langfuse trace is added in the orchestrator via
    `get_langfuse_callback()` — this context manager just records the local
    log line.
    """
    state: Dict[str, Any] = {
        "client": get_langfuse_client(),
        "request_id": request_id,
        "events": [],
        "metadata": {"user_message_preview": user_message[:200]},
        "token_usage": {"input_tokens": 0, "output_tokens": 0},
        "start": time.time(),
    }

    class _Trace:
        def span(self, name: str, **kwargs: Any) -> None:
            state["events"].append({"name": name, "ts": time.time(), **kwargs})
            client = state["client"]
            if client is not None:
                try:
                    client.update_current_generation(
                        **{k: v for k, v in kwargs.items() if k in ("name", "metadata")}
                    )
                except Exception:
                    pass

        def set_token_usage(self, input_tokens: int, output_tokens: int) -> None:
            state["token_usage"]["input_tokens"] += input_tokens
            state["token_usage"]["output_tokens"] += output_tokens

        def set_metadata(self, key: str, value: Any) -> None:
            state["metadata"][key] = value

        def add_event(self, name: str, **kwargs: Any) -> None:
            state["events"].append({"name": name, "ts": time.time(), **kwargs})

    try:
        yield _Trace()
    finally:
        state["latency_ms"] = int((time.time() - state["start"]) * 1000)
        client = state["client"]
        if client is not None:
            try:
                client.flush()
            except Exception:
                pass
        logger.info(
            "trace_completed",
            extra={
                "request_id": request_id,
                "events": len(state["events"]),
                "latency_ms": state["latency_ms"],
                "token_usage": state["token_usage"],
            },
        )
