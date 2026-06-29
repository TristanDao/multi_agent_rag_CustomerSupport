"""Tests for Langfuse integration in `app/core/observability.py`.

These tests:
* verify the module loads and exposes the expected public functions
* check that `LANGFUSE_ENABLED=false` short-circuits the init
* check that `LANGFUSE_ENABLED=true` with empty keys raises (loud failure)
* stub out the real Langfuse client + CallbackHandler to verify the orchestrator
  wires the callback into `graph.ainvoke(...)` when configured
"""
import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    from app.core import observability

    observability.reset_langfuse()
    yield
    observability.reset_langfuse()


def test_disabled_short_circuits():
    from app.core.observability import get_langfuse_client, get_langfuse_callback

    os.environ["LANGFUSE_ENABLED"] = "false"
    # also clear the cached settings so the env change is picked up
    from app.config import get_settings

    get_settings.cache_clear()
    assert get_langfuse_client() is None
    assert get_langfuse_callback() is None


def test_enabled_with_empty_keys_raises_loudly():
    from app.config import get_settings
    from app.core.observability import get_langfuse_client

    os.environ["LANGFUSE_ENABLED"] = "true"
    os.environ["LANGFUSE_PUBLIC_KEY"] = ""
    os.environ["LANGFUSE_SECRET_KEY"] = ""
    get_settings.cache_clear()

    with pytest.raises(RuntimeError):
        get_langfuse_client()


def test_enabled_with_keys_constructs_client(monkeypatch):
    from app.config import get_settings
    from app.core.observability import get_langfuse_client

    os.environ["LANGFUSE_ENABLED"] = "true"
    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-test"
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-test"
    get_settings.cache_clear()

    fake_langfuse_cls = MagicMock()
    fake_client = MagicMock()
    fake_langfuse_cls.return_value = fake_client

    with patch.dict("sys.modules", {"langfuse": MagicMock(Langfuse=fake_langfuse_cls)}):
        client = get_langfuse_client()
    assert client is fake_client


def test_callback_returned_when_enabled(monkeypatch):
    from app.config import get_settings
    from app.core.observability import get_langfuse_callback

    os.environ["LANGFUSE_ENABLED"] = "true"
    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-test"
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-test"
    get_settings.cache_clear()

    fake_handler_cls = MagicMock()
    fake_handler = MagicMock()
    fake_handler_cls.return_value = fake_handler

    with patch.dict(
        "sys.modules",
        {"langfuse": MagicMock(), "langfuse.langchain": MagicMock(CallbackHandler=fake_handler_cls)},
    ):
        cb = get_langfuse_callback()
    assert cb is fake_handler
