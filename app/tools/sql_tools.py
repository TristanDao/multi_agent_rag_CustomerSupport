"""Tool registry that exposes safe functions to agents."""
from typing import Any, Callable, Dict

from app.tools.product_tools import (
    check_inventory,
    get_active_promotions,
    get_price_for_quantity,
    get_product_by_sku,
    get_related_products,
    search_products,
)
from app.tools.order_tools import (
    get_customer_order_history,
    get_customer_profile,
    get_order_details,
    get_order_status,
)
from app.tools.pricing_tools import calculate_bulk_price, get_promotions_for_order

TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "search_products": search_products,
    "get_product_by_sku": get_product_by_sku,
    "check_inventory": check_inventory,
    "get_price_for_quantity": get_price_for_quantity,
    "get_related_products": get_related_products,
    "get_active_promotions": get_active_promotions,
    "get_customer_profile": get_customer_profile,
    "get_order_status": get_order_status,
    "get_order_details": get_order_details,
    "get_customer_order_history": get_customer_order_history,
    "calculate_bulk_price": calculate_bulk_price,
    "get_promotions_for_order": get_promotions_for_order,
}


def call_tool(name: str, **kwargs: Any) -> Any:
    if name not in TOOL_REGISTRY:
        return {"error": f"unknown_tool: {name}"}
    try:
        return TOOL_REGISTRY[name](**kwargs)
    except Exception as e:  # tools must never crash the agent
        return {"error": "tool_exception", "tool": name, "detail": str(e)[:200]}


def list_tools() -> Dict[str, str]:
    return {k: v.__doc__ or "" for k, v in TOOL_REGISTRY.items()}
