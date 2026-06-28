"""Product/inventory/pricing tools (read-only)."""
import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy import or_, func

from app.db.models import Inventory, PriceTier, Product, Promotion
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)


def _to_float(v: Any) -> Any:
    if isinstance(v, Decimal):
        return float(v)
    return v


def search_products(
    query: str,
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    in_stock_only: bool = True,
    limit: int = 10,
) -> Dict[str, Any]:
    """Search products by name/brand/category with optional filters."""
    with SessionLocal() as db:
        q = db.query(Product).filter(Product.status == "active")
        if query:
            like = f"%{query}%"
            q = q.filter(
                or_(
                    Product.product_name.ilike(like),
                    Product.brand.ilike(like),
                    Product.sku.ilike(like),
                    Product.description.ilike(like),
                )
            )
        if category:
            q = q.filter(Product.category.ilike(f"%{category}%"))
        if max_price is not None:
            q = q.filter(Product.base_price <= max_price)

        products = q.order_by(Product.product_name).limit(limit * 2).all()
        results: List[Dict[str, Any]] = []
        for p in products:
            inv_q = db.query(func.coalesce(func.sum(Inventory.stock_quantity - Inventory.reserved_quantity), 0)).filter(
                Inventory.sku == p.sku
            )
            available = int(inv_q.scalar() or 0)
            if in_stock_only and available <= 0:
                continue
            results.append(
                {
                    "sku": p.sku,
                    "product_name": p.product_name,
                    "brand": p.brand,
                    "category": p.category,
                    "unit": p.unit,
                    "base_price": _to_float(p.base_price),
                    "stock_available": available,
                    "warranty_months": p.warranty_months,
                    "returnable": p.returnable,
                    "description": (p.description or "")[:200],
                }
            )
            if len(results) >= limit:
                break
        return {"count": len(results), "products": results, "source": "products + inventory"}


def get_product_by_sku(sku: str) -> Dict[str, Any]:
    with SessionLocal() as db:
        p = db.query(Product).filter(Product.sku == sku).first()
        if not p:
            return {"found": False, "sku": sku}
        inv = db.query(Inventory).filter(Inventory.sku == sku).all()
        inv_payload = [
            {
                "warehouse_id": i.warehouse_id,
                "warehouse_location": i.warehouse_location,
                "stock_quantity": int(i.stock_quantity),
                "reserved_quantity": int(i.reserved_quantity),
                "available": int(i.stock_quantity - i.reserved_quantity),
            }
            for i in inv
        ]
        return {
            "found": True,
            "sku": p.sku,
            "product_name": p.product_name,
            "brand": p.brand,
            "category": p.category,
            "unit": p.unit,
            "base_price": _to_float(p.base_price),
            "description": p.description,
            "status": p.status,
            "returnable": p.returnable,
            "warranty_months": p.warranty_months,
            "inventory": inv_payload,
            "source": "products + inventory",
        }


def check_inventory(
    sku: str,
    quantity: Optional[int] = None,
    warehouse_location: Optional[str] = None,
) -> Dict[str, Any]:
    with SessionLocal() as db:
        q = db.query(Inventory).filter(Inventory.sku == sku)
        if warehouse_location:
            q = q.filter(Inventory.warehouse_location.ilike(f"%{warehouse_location}%"))
        rows = q.all()
        if not rows:
            return {"sku": sku, "found": False}
        total_available = sum(int(r.stock_quantity - r.reserved_quantity) for r in rows)
        out = {
            "sku": sku,
            "found": True,
            "total_available": total_available,
            "warehouses": [
                {
                    "warehouse_id": r.warehouse_id,
                    "warehouse_location": r.warehouse_location,
                    "available": int(r.stock_quantity - r.reserved_quantity),
                    "stock_quantity": int(r.stock_quantity),
                    "reserved_quantity": int(r.reserved_quantity),
                }
                for r in rows
            ],
            "source": "inventory",
        }
        if quantity is not None:
            out["requested_quantity"] = quantity
            out["sufficient"] = total_available >= quantity
        return out


def get_price_for_quantity(
    sku: str,
    quantity: int,
    customer_type: str = "retail",
) -> Dict[str, Any]:
    """Compute the effective unit price for a given quantity and customer type."""
    with SessionLocal() as db:
        p = db.query(Product).filter(Product.sku == sku).first()
        if not p:
            return {"found": False, "sku": sku}
        tiers = (
            db.query(PriceTier)
            .filter(PriceTier.sku == sku, PriceTier.customer_type == customer_type)
            .order_by(PriceTier.min_quantity.asc())
            .all()
        )
        best = None
        for t in tiers:
            if quantity >= (t.min_quantity or 1):
                best = t
        if best is None:
            best_unit_price = _to_float(p.base_price)
            discount_pct = 0.0
            tier_used = None
        else:
            best_unit_price = _to_float(best.unit_price)
            discount_pct = _to_float(best.discount_percent or 0)
            tier_used = {
                "min_quantity": int(best.min_quantity),
                "unit_price": best_unit_price,
                "discount_percent": discount_pct,
            }
        return {
            "found": True,
            "sku": sku,
            "product_name": p.product_name,
            "customer_type": customer_type,
            "quantity": quantity,
            "unit_price": best_unit_price,
            "discount_percent": discount_pct,
            "line_total": round(best_unit_price * quantity, 2),
            "tier_used": tier_used,
            "source": "products + price_tiers",
        }


def get_related_products(sku_or_category: str, limit: int = 5) -> Dict[str, Any]:
    """Get related products in the same category as the given SKU or category name."""
    with SessionLocal() as db:
        category = None
        if sku_or_category.startswith("SKU") or sku_or_category.startswith("sku"):
            p = db.query(Product).filter(Product.sku == sku_or_category).first()
            if p:
                category = p.category
        if not category:
            category = sku_or_category
        rows = (
            db.query(Product)
            .filter(Product.status == "active", Product.category.ilike(f"%{category}%"))
            .limit(limit + 1)
            .all()
        )
        related = []
        for r in rows:
            if r.sku != sku_or_category:
                related.append(
                    {
                        "sku": r.sku,
                        "product_name": r.product_name,
                        "category": r.category,
                        "base_price": _to_float(r.base_price),
                    }
                )
            if len(related) >= limit:
                break
        return {"category": category, "count": len(related), "products": related, "source": "products"}


def get_active_promotions() -> Dict[str, Any]:
    with SessionLocal() as db:
        promos = db.query(Promotion).filter(Promotion.active == True).all()  # noqa: E712
        return {
            "count": len(promos),
            "promotions": [
                {
                    "code": p.code,
                    "name": p.name,
                    "description": p.description,
                    "discount_percent": _to_float(p.discount_percent),
                    "min_order_amount": _to_float(p.min_order_amount),
                }
                for p in promos
            ],
            "source": "promotions",
        }
