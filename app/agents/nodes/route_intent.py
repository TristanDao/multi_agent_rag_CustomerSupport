"""route_intent node: classify user intent + extract entities.

The actual classification logic lives in `app.agents.intent_classifier` (unchanged
from the previous build). This node just bridges it into the LangGraph state shape.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from app.agents.intent_classifier import classify_intent
from app.agents.state import SupportState
from app.agents.registry import AGENT_REGISTRY, TERMINAL_NODES

logger = logging.getLogger(__name__)


def _default_agents_for_intent(intent: str) -> list[str]:
    table = {
        "product_search": ["product_agent"],
        "product_comparison": ["product_agent"],
        "inventory_check": ["product_agent"],
        "wholesale_pricing": ["product_agent"],
        "order_tracking": ["order_agent"],
        "return_refund": ["refund_decision_agent"],
        "shipping_policy": ["policy_rag_agent"],
        "payment_terms": ["policy_rag_agent"],
        "warranty_policy": ["policy_rag_agent"],
        "sales_recommendation": ["sales_recommendation_agent"],
        "human_escalation": ["response_agent"],
        "general_faq": ["policy_rag_agent"],
        "unknown": ["response_agent"],
    }
    return table.get(intent, ["response_agent"])


async def route_intent_node(state: SupportState) -> Dict[str, Any]:
    """Read user message, classify intent, choose next node.

    Returns a partial state update with: `intent`, `entities`, `next_node`,
    `nodes_visited`.
    """
    user_message = state.get("user_message", "")
    customer_id = state.get("customer_id")

    try:
        decision = await classify_intent(user_message, customer_id=customer_id)
    except Exception as e:
        logger.warning("intent_classifier_failed err=%s", str(e))
        decision = {
            "intent": "unknown",
            "required_agents": ["response_agent"],
            "entities": {},
            "confidence": 0.0,
        }

    intent = decision.get("intent") or "unknown"
    entities = dict(decision.get("entities") or {})
    entities.setdefault("customer_id", customer_id)

    # Pick the next node: first required_agent that is a known node, else response_agent.
    required = decision.get("required_agents") or _default_agents_for_intent(intent)
    next_node = None
    for cand in required:
        if cand in AGENT_REGISTRY or cand in TERMINAL_NODES:
            next_node = cand
            break
    if next_node is None:
        next_node = "response_agent"

    return {
        "intent": intent,
        "entities": entities,
        "next_node": next_node,
        "nodes_visited": ["route_intent"],
    }
