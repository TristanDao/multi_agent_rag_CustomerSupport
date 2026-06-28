"""Structured JSON logging with PII redaction."""
import json
import logging
import sys
import time
from typing import Any, Dict

from app.config import settings
from app.core.pii_redaction import redact_pii


class PiiRedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_pii(record.msg)
        if record.args:
            try:
                if isinstance(record.args, dict):
                    record.args = {
                        k: (redact_pii(str(v)) if isinstance(v, str) else v)
                        for k, v in record.args.items()
                    }
            except Exception:
                pass
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int((record.created % 1) * 1000):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (
            "request_id",
            "intent",
            "agents_called",
            "tools_called",
            "retrieved_doc_ids",
            "latency_ms",
            "model_name",
            "token_usage",
            "guardrail_result",
            "pii_redacted",
            "status",
            "customer_id",
            "error",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    if settings.ENABLE_PII_REDACTION:
        handler.addFilter(PiiRedactingFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
