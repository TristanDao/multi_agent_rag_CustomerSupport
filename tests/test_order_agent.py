"""Test order agent (skips when DB unavailable)."""
import os

import pytest


@pytest.fixture(scope="module", autouse=True)
def ensure_db():
    url = os.environ.get("DATABASE_URL", "")
    if not url.startswith("postgres"):
        pytest.skip("DB tests require PostgreSQL; set DATABASE_URL=postgresql+psycopg2://...")
    try:
        from app.db.session import engine
        from sqlalchemy import text

        with engine.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception:
        pytest.skip("Postgres not available; skipping order tests")


def test_order_agent_extracts_order_id():
    from app.agents.order_agent import order_agent

    res = order_agent("order_tracking", {}, "Đơn DH00001 đang ở đâu?")
    assert res["order_id"] == "DH00001"
    assert "data" in res
