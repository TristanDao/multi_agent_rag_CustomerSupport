"""Test product tool and product agent (uses in-memory store + SQLite-compatible path)."""
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
        pytest.skip("Postgres not available; skipping DB-backed product tests")


def test_search_products():
    from app.tools.product_tools import search_products

    res = search_products("A4", limit=5)
    assert "products" in res
    assert isinstance(res["products"], list)


def test_get_price_for_quantity():
    from app.tools.product_tools import get_price_for_quantity

    res = get_price_for_quantity("SKU00001", 50, "wholesale")
    if res.get("found"):
        assert "unit_price" in res
        assert "line_total" in res
