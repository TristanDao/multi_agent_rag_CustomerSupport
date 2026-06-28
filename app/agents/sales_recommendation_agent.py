"""Sales recommendation agent: suggest cross-sell / bundle / re-order items."""
import logging
from typing import Any, Dict

from app.tools.sql_tools import call_tool

logger = logging.getLogger(__name__)


def _extract_category(entities: Dict[str, Any], message: str) -> str | None:
    if entities.get("category"):
        return entities["category"]
    text = message.lower()
    for kw in ["giấy a4", "giấy in", "bút", "sổ", "mực in", "băng keo", "kẹp giấy", "bìa"]:
        if kw in text:
            return kw
    return entities.get("product_name") or entities.get("product")


def sales_recommendation_agent(intent: str, entities: Dict[str, Any], message: str) -> Dict[str, Any]:
    customer_id = entities.get("customer_id")
    category = _extract_category(entities, message)
    blocks: list = []
    tools_called: list = []

    if customer_id:
        tools_called.append("get_customer_order_history")
        history = call_tool("get_customer_order_history", customer_id=customer_id, limit=10)
        blocks.append({"history": history})

    if category:
        tools_called.append("search_products")
        search = call_tool("search_products", query=category, in_stock_only=True, limit=5)
        blocks.append({"category": category, "search": search})

        if search.get("products"):
            first = search["products"][0]
            tools_called.append("get_related_products")
            related = call_tool("get_related_products", sku_or_category=first["sku"], limit=4)
            blocks.append({"related": related})

    return {
        "agent": "sales_recommendation_agent",
        "intent": intent,
        "tools_called": tools_called,
        "data": blocks,
    }
