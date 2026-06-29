"""LangGraph orchestrator (StateGraph).

Builds the StateGraph that wires every registered agent as a node, with `route_intent`
as the entrypoint and `response_agent` / `human_escalation` as terminal nodes. The
graph is compiled with the configured checkpointer and cached as a module-level
singleton so FastAPI reuses one instance per process.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph

from app.agents.checkpointer import build_checkpointer
from app.agents.nodes.human_escalation import human_escalation_node
from app.agents.nodes.policy_rag_agent_node import policy_rag_agent_node
from app.agents.nodes.product_agent_node import product_agent_node
from app.agents.nodes.order_agent_node import order_agent_node
from app.agents.nodes.refund_decision_agent_node import refund_decision_agent_node
from app.agents.nodes.response_agent import response_agent_node
from app.agents.nodes.route_intent import route_intent_node
from app.agents.nodes.sales_recommendation_agent_node import sales_recommendation_agent_node
from app.agents.registry import AGENT_REGISTRY, TERMINAL_NODES, set_node_fn, validate_registry
from app.agents.state import SupportState

logger = logging.getLogger(__name__)

_graph_lock = threading.Lock()
_compiled_graph: Optional[Any] = None


# --- Routing --------------------------------------------------------

# Map an agent key to the terminal node that should run after it.
# Default: most agents → response_agent. Refund decision may branch to
# human_escalation when the policy says it requires human review.
DEFAULT_NEXT_AFTER_AGENT = "response_agent"


def route_after_agent(state: SupportState) -> str:
    """Conditional edge: pick the next node after an agent runs.

    For now: every agent node goes to response_agent. The refund_decision_agent
    can set `requires_human=True` itself (in `agent_results["refund_decision_agent"]`)
    and the response_agent will surface that to the user.
    """
    return "response_agent"


def route_after_route_intent(state: SupportState) -> str:
    """Conditional edge: pick the next node after route_intent."""
    next_node = state.get("next_node")
    if not next_node:
        return "response_agent"
    if next_node in AGENT_REGISTRY or next_node in TERMINAL_NODES:
        return next_node
    return "response_agent"


# --- Builder -------------------------------------------------------


def _bind_node_fns() -> None:
    """Bind every registered agent's node_fn into the registry."""
    set_node_fn("product_agent", product_agent_node)
    set_node_fn("order_agent", order_agent_node)
    set_node_fn("policy_rag_agent", policy_rag_agent_node)
    set_node_fn("sales_recommendation_agent", sales_recommendation_agent_node)
    set_node_fn("refund_decision_agent", refund_decision_agent_node)


def build_orchestrator_graph(checkpointer: Any = None) -> Any:
    """Build and compile the LangGraph orchestrator.

    `checkpointer` is optional: if None, the configured backend is used.
    The graph is fully synchronous in its node functions except for the LLM calls
    (the legacy agent functions handle their own async internally).
    """
    _bind_node_fns()
    validate_registry()

    builder = StateGraph(SupportState)

    # Nodes
    builder.add_node("route_intent", route_intent_node)
    for key, spec in AGENT_REGISTRY.items():
        builder.add_node(key, spec.node_fn)
    builder.add_node("response_agent", response_agent_node)
    builder.add_node("human_escalation", human_escalation_node)

    # Edges
    builder.add_edge(START, "route_intent")

    # route_intent → chosen node (or terminal)
    builder.add_conditional_edges(
        "route_intent",
        route_after_route_intent,
        {key: key for key in list(AGENT_REGISTRY.keys()) + list(TERMINAL_NODES)},
    )

    # each agent node → response_agent
    for key in AGENT_REGISTRY:
        builder.add_edge(key, "response_agent")

    # terminal nodes → END
    builder.add_edge("response_agent", END)
    builder.add_edge("human_escalation", END)

    cp = checkpointer if checkpointer is not None else build_checkpointer()
    compiled = builder.compile(checkpointer=cp)
    return compiled


def get_orchestrator_graph() -> Any:
    """Return the compiled graph singleton (built lazily, thread-safe)."""
    global _compiled_graph
    if _compiled_graph is None:
        with _graph_lock:
            if _compiled_graph is None:
                _compiled_graph = build_orchestrator_graph()
    return _compiled_graph


def reset_orchestrator_graph() -> None:
    """Drop the cached compiled graph. Useful for tests."""
    global _compiled_graph
    _compiled_graph = None
