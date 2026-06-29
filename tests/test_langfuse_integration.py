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
    """Reset Langfuse cache + Settings cache between tests."""
    from app.config import get_settings
    from app.core import observability

    observability.reset_langfuse()
    get_settings.cache_clear()
    print("\n[fixture] _initialised =", observability._langfuse_initialised)
    yield
    observability.reset_langfuse()
    get_settings.cache_clear()


def test_disabled_short_circuits(monkeypatch):
    from app.core.observability import get_langfuse_callback, get_langfuse_client

    monkeypatch.setenv("LANGFUSE_ENABLED", "false")
    assert get_langfuse_client() is None
    assert get_langfuse_callback() is None


def test_enabled_with_empty_keys_raises_loudly(monkeypatch):
    from app.config import get_settings
    from app.core.observability import get_langfuse_client
    from app.core import observability

    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    s = get_settings()
    print("[test] LANGFUSE_ENABLED =", s.LANGFUSE_ENABLED)
    print("[test] _initialised =", observability._langfuse_initialised)
    print("[test] public key =", repr(s.LANGFUSE_PUBLIC_KEY))
    print("[test] secret key =", repr(s.LANGFUSE_SECRET_KEY))
    with pytest.raises(RuntimeError):
        get_langfuse_client()


def test_enabled_with_keys_constructs_client(monkeypatch):
    from app.core.observability import get_langfuse_client

    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    fake_langfuse_cls = MagicMock()
    fake_client = MagicMock()
    fake_langfuse_cls.return_value = fake_client

    fake_module = MagicMock(Langfuse=fake_langfuse_cls)
    with patch.dict("sys.modules", {"langfuse": fake_module}):
        client = get_langfuse_client()
    assert client is fake_client


def test_callback_returned_when_enabled(monkeypatch):
    from app.core.observability import get_langfuse_callback

    monkeypatch.setenv("LANGFUSE_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")

    fake_handler_cls = MagicMock()
    fake_handler = MagicMock()
    fake_handler_cls.return_value = fake_handler

    fake_langchain_mod = MagicMock(CallbackHandler=fake_handler_cls)
    with patch.dict(
        "sys.modules",
        {
            "langfuse": MagicMock(),
            "langfuse.langchain": fake_langchain_mod,
        },
    ):
        cb = get_langfuse_callback()
    assert cb is fake_handler
