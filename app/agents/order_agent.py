"""Order agent: status, details, customer history."""
import logging
import re
from typing import Any, Dict

from app.tools.sql_tools import call_tool

logger = logging.getLogger(__name__)


def _extract_order_id(message: str, entities: Dict[str, Any]) -> str | None:
    if entities.get("order_id"):
        return entities["order_id"]
    m = re.search(r"\b(DH[\-_]?\w+|\w*ORD\w*[\-_]?\d+|\w+\d{3,})\b", message, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def _extract_customer_id(entities: Dict[str, Any]) -> str | None:
    return entities.get("customer_id")


def order_agent(intent: str, entities: Dict[str, Any], message: str) -> Dict[str, Any]:
    order_id = _extract_order_id(message, entities)
    customer_id = _extract_customer_id(entities)
    tools_called = []
    blocks: list = []

    if intent == "order_tracking" and order_id:
        tools_called.append("get_order_status")
        blocks.append(call_tool("get_order_status", order_id=order_id))
        if entities.get("include_details"):
            tools_called.append("get_order_details")
            blocks.append(call_tool("get_order_details", order_id=order_id))

    if customer_id and (intent in ("order_tracking", "general_faq", "sales_recommendation")):
        tools_called.append("get_customer_order_history")
        blocks.append(call_tool("get_customer_order_history", customer_id=customer_id, limit=5))

    return {
        "agent": "order_agent",
        "intent": intent,
        "tools_called": tools_called,
        "data": blocks,
        "order_id": order_id,
        "customer_id": customer_id,
    }
