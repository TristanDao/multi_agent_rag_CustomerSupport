"""Test policy RAG agent (uses in-memory store when Qdrant unavailable)."""
import os

import pytest


def test_policy_rag_agent_returns_chunks():
    from app.agents.policy_rag_agent import policy_rag_agent

    res = policy_rag_agent("return_refund", "Tôi muốn đổi trả trong 7 ngày")
    assert "data" in res
    assert res["agent"] == "policy_rag_agent"
