"""Product agent: search, inventory, pricing, related products."""
import logging
from typing import Any, Dict, List, Optional

from app.tools.product_tools import (
    check_inventory,
    get_price_for_quantity,
    get_product_by_sku,
    get_related_products,
    search_products,
)
from app.tools.sql_tools import call_tool

logger = logging.getLogger(__name__)


def _resolve_query(entities: Dict[str, Any], message: str) -> Dict[str, Any]:
    product_name = entities.get("product_name") or entities.get("product") or entities.get("query")
    sku = entities.get("sku")
    category = entities.get("category")
    quantity = entities.get("quantity")
    customer_type = entities.get("customer_type") or "retail"
    return {
        "product_name": product_name,
        "sku": sku,
        "category": category,
        "quantity": quantity,
        "customer_type": customer_type,
        "raw": message,
    }


def _fetch_wholesale_policy(message: str) -> List[Dict[str, Any]]:
    """Retrieve wholesale-policy chunks for the answer.

    Used only when `intent == 'wholesale_pricing'` so the response can cite the
    minimum order value, distributor rules, etc.
    """
    try:
        from app.rag.retriever import retrieve

        chunks: List[Dict[str, Any]] = []
        seen: set = set()
        for q in (message, "wholesale pricing policy"):
            for hit in retrieve(q):
                if hit.get("doc_id") in seen:
                    continue
                seen.add(hit.get("doc_id"))
                chunks.append(hit)
                if len(chunks) >= 4:
                    break
            if len(chunks) >= 4:
                break
        return chunks
    except Exception as e:
        logger.warning("wholesale_policy_retrieve_failed err=%s", str(e))
        return []


def product_agent(
    intent: str,
    entities: Dict[str, Any],
    message: str,
) -> Dict[str, Any]:
    """Decide which product tool(s) to call based on intent and return combined results."""
    ctx = _resolve_query(entities, message)
    tools_called: List[str] = []
    blocks: List[Dict[str, Any]] = []

    if intent in ("product_search", "product_comparison", "general_faq"):
        if ctx["sku"]:
            tools_called.append("get_product_by_sku")
            blocks.append(call_tool("get_product_by_sku", sku=ctx["sku"]))
        else:
            tools_called.append("search_products")
            blocks.append(
                call_tool(
                    "search_products",
                    query=ctx["product_name"] or ctx["raw"],
                    category=ctx["category"],
                    max_price=entities.get("max_price"),
                    in_stock_only=bool(entities.get("in_stock_only", True)),
                    limit=5,
                )
            )

    if intent == "inventory_check":
        tools_called.append("check_inventory")
        if ctx["sku"]:
            blocks.append(
                call_tool(
                    "check_inventory",
                    sku=ctx["sku"],
                    quantity=ctx["quantity"],
                    warehouse_location=entities.get("warehouse_location"),
                )
            )
        else:
            search = call_tool(
                "search_products",
                query=ctx["product_name"] or ctx["raw"],
                in_stock_only=False,
                limit=3,
            )
            for p in search.get("products", []):
                inv = call_tool("check_inventory", sku=p["sku"], quantity=ctx["quantity"])
                inv["product_name"] = p["product_name"]
                inv["base_price"] = p["base_price"]
                blocks.append(inv)
            tools_called.append("search_products")

    if intent == "wholesale_pricing":
        if ctx["sku"]:
            tools_called.append("get_price_for_quantity")
            blocks.append(
                call_tool(
                    "get_price_for_quantity",
                    sku=ctx["sku"],
                    quantity=ctx["quantity"] or 1,
                    customer_type=ctx["customer_type"],
                )
            )
        else:
            tools_called.append("search_products")
            search = call_tool(
                "search_products",
                query=ctx["product_name"] or ctx["raw"],
                limit=3,
            )
            for p in search.get("products", []):
                pricing = call_tool(
                    "get_price_for_quantity",
                    sku=p["sku"],
                    quantity=ctx["quantity"] or 1,
                    customer_type=ctx["customer_type"],
                )
                pricing["product_name"] = p["product_name"]
                blocks.append(pricing)
            tools_called.append("get_price_for_quantity")
        # Always pull wholesale policy citations for this intent so the response
        # can include minimum order value, distributor rules, etc.
        policy_chunks = _fetch_wholesale_policy(message)
        if policy_chunks:
            tools_called.append("retrieve_policy_chunks")
            blocks.append({"policy": policy_chunks})

    if intent == "sales_recommendation" and (ctx["sku"] or ctx["category"]):
        tools_called.append("get_related_products")
        key = ctx["sku"] or ctx["category"] or ctx["product_name"]
        blocks.append(call_tool("get_related_products", sku_or_category=key, limit=5))

    return {
        "agent": "product_agent",
        "intent": intent,
        "tools_called": tools_called,
        "data": blocks,
    }
