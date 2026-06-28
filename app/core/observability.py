"""Observability layer: Langfuse integration with safe in-memory fallback."""
import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class InMemoryTrace:
    """Lightweight in-memory tracer used when Langfuse is disabled or unavailable."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.start_ts = time.time()

    def log(self, name: str, **kwargs: Any) -> None:
        self.events.append({"name": name, "ts": time.time() - self.start_ts, **kwargs})

    def to_dict(self) -> Dict[str, Any]:
        return {"events": self.events, "duration_s": time.time() - self.start_ts}


def get_langfuse_client():
    if not settings.LANGFUSE_ENABLED:
        return None
    if not (settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY):
        return None
    try:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.LANGFUSE_PUBLIC_KEY)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.LANGFUSE_SECRET_KEY)
        os.environ.setdefault("LANGFUSE_HOST", settings.LANGFUSE_HOST)
        from langfuse import Langfuse  # type: ignore

        return Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
    except Exception as e:
        logger.warning("Langfuse not initialized: %s", str(e))
        return None


@contextmanager
def trace_request(request_id: str, user_message: str):
    """Yield a context with `.span(name, **kwargs)` for logging trace events."""
    client = get_langfuse_client()
    state: Dict[str, Any] = {
        "client": client,
        "request_id": request_id,
        "events": [],
        "metadata": {"user_message_preview": user_message[:200]},
        "token_usage": {"input_tokens": 0, "output_tokens": 0},
        "start": time.time(),
    }

    class _Trace:
        def span(self, name: str, **kwargs: Any) -> None:
            state["events"].append({"name": name, "ts": time.time(), **kwargs})
            if client is not None:
                try:
                    client.update_current_generation(**{k: v for k, v in kwargs.items() if k in ("name", "metadata")})
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
