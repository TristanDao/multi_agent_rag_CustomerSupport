"""Pricing convenience tools built on top of product_tools."""
from typing import Any, Dict, List

from app.tools.product_tools import get_price_for_quantity, get_active_promotions


def calculate_bulk_price(
    items: List[Dict[str, Any]],
    customer_type: str = "wholesale",
) -> Dict[str, Any]:
    """Calculate total for a cart of items with mixed SKUs.

    items: [{"sku": "SKU001", "quantity": 50}, ...]
    """
    line_items = []
    grand_total = 0.0
    for item in items:
        sku = item.get("sku")
        qty = int(item.get("quantity", 0))
        if not sku or qty <= 0:
            continue
        price_info = get_price_for_quantity(sku, qty, customer_type=customer_type)
        if not price_info.get("found"):
            line_items.append({"sku": sku, "error": "product not found"})
            continue
        line_total = float(price_info["line_total"])
        grand_total += line_total
        line_items.append(
            {
                "sku": sku,
                "product_name": price_info.get("product_name"),
                "quantity": qty,
                "unit_price": price_info.get("unit_price"),
                "discount_percent": price_info.get("discount_percent"),
                "line_total": line_total,
            }
        )
    return {
        "customer_type": customer_type,
        "line_items": line_items,
        "subtotal": round(grand_total, 2),
        "source": "products + price_tiers",
    }


def get_promotions_for_order(order_total: float) -> Dict[str, Any]:
    """Return applicable promotions (does not apply them automatically)."""
    promos = get_active_promotions()
    eligible = []
    for p in promos.get("promotions", []):
        if order_total >= (p.get("min_order_amount") or 0):
            eligible.append(p)
    return {"order_total": order_total, "eligible": eligible, "count": len(eligible)}
