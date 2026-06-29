"""Tests for the Agent Registry, LangGraph orchestrator, and admin endpoints."""
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    os.environ.setdefault("APP_ENV", "test")
    from app.main import app

    return TestClient(app)


# --- Agent Registry unit tests ------------------------------------


def test_registry_has_required_agents():
    from app.agents.registry import AGENT_REGISTRY

    required = {
        "product_agent",
        "order_agent",
        "policy_rag_agent",
        "sales_recommendation_agent",
        "refund_decision_agent",
    }
    assert required.issubset(AGENT_REGISTRY.keys()), (
        f"Missing agents: {required - AGENT_REGISTRY.keys()}"
    )


def test_registry_intents_are_unique():
    """The validate_registry() helper must not raise."""
    from app.agents.graph import build_orchestrator_graph
    from app.agents.registry import validate_registry

    # Building the graph runs validate_registry internally.
    build_orchestrator_graph()
    validate_registry()  # explicit second call, must be a no-op


def test_registry_for_api_shape():
    from app.agents.registry import registry_for_api

    payload = registry_for_api()
    assert "agents" in payload
    assert "total" in payload
    assert "terminal_nodes" in payload
    assert payload["total"] == len(payload["agents"])
    for a in payload["agents"]:
        assert {"key", "name", "description", "capabilities", "intents", "tools"} <= a.keys()


# --- Graph build / runtime tests ----------------------------------


def test_graph_builds_and_has_nodes():
    from app.agents.graph import build_orchestrator_graph

    g = build_orchestrator_graph()
    node_names = set(g.get_graph().nodes.keys())
    for required in (
        "route_intent",
        "product_agent",
        "order_agent",
        "policy_rag_agent",
        "sales_recommendation_agent",
        "refund_decision_agent",
        "response_agent",
        "human_escalation",
    ):
        assert required in node_names, f"missing node: {required}"


def test_graph_runs_end_to_end_with_state_checkpointer():
    """A minimal end-to-end run with an in-memory checkpointer."""
    import asyncio
    from app.agents.graph import build_orchestrator_graph
    from app.agents.state import empty_state

    g = build_orchestrator_graph()
    state = empty_state(
        user_message="Tìm giấy A4 giá dưới 400k",
        request_id="req_test_1",
        thread_id="thread_test_1",
    )
    config = {"configurable": {"thread_id": "thread_test_1"}}
    result = asyncio.run(g.ainvoke(state, config=config))
    assert result.get("intent") in ("product_search", "wholesale_pricing", "unknown")
    assert "nodes_visited" in result
    assert "route_intent" in result["nodes_visited"]
    assert "response_agent" in result["nodes_visited"]
    assert result.get("final_answer"), "response_agent must produce an answer"


# --- Admin endpoints -----------------------------------------------


def test_admin_list_agents(client):
    r = client.get("/admin/agents")
    assert r.status_code == 200
    body = r.json()
    assert "agents" in body
    assert body["total"] >= 5
    assert "terminal_nodes" in body
    keys = {a["key"] for a in body["agents"]}
    assert "product_agent" in keys
    assert "refund_decision_agent" in keys


def test_admin_get_agent_detail(client):
    r = client.get("/admin/agents/product_agent")
    assert r.status_code == 200
    body = r.json()
    assert body["key"] == "product_agent"
    assert "intents" in body
    assert "wholesale_pricing" in body["intents"]


def test_admin_get_agent_not_found(client):
    r = client.get("/admin/agents/does_not_exist")
    assert r.status_code == 404


def test_admin_graph_returns_mermaid(client):
    r = client.get("/admin/graph")
    assert r.status_code == 200
    body = r.json()
    assert body["format"] == "mermaid"
    assert "graph" in body["diagram"].lower()
    assert "route_intent" in body["diagram"]


def test_admin_thread_history_safe_for_unknown_thread(client):
    r = client.get("/admin/threads/unknown-thread-12345/history")
    assert r.status_code == 200
    body = r.json()
    assert body["thread_id"] == "unknown-thread-12345"
    assert body["total"] == 0
    assert body["checkpoints"] == []


# --- /chat with thread_id ------------------------------------------


def test_chat_returns_thread_id_and_checkpoint_id(client):
    r = client.post(
        "/chat",
        json={"message": "Tìm giấy A4", "thread_id": "thread-api-test-1"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["thread_id"] == "thread-api-test-1"
    # checkpoint_id may be None when using the in-memory backend and the
    # installed langgraph version doesn't expose one, but the field must be
    # present in the response.
    assert "checkpoint_id" in body
    assert "thread_id" in body
    # The new graph-driven pipeline should always visit these nodes.
    assert "route_intent" in body["agents_called"]
    assert "response_agent" in body["agents_called"]


def test_chat_auto_generates_thread_id_when_missing(client):
    r = client.post("/chat", json={"message": "Xin chào"})
    assert r.status_code == 200
    body = r.json()
    assert body["thread_id"], "server must auto-generate a thread_id when none is supplied"
