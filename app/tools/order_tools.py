"""Order and customer tools (read-only)."""
import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy import desc

from app.db.models import Customer, Order, OrderItem
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)


def _to_float(v: Any) -> Any:
    if isinstance(v, Decimal):
        return float(v)
    return v


def _to_iso(v: Any) -> Any:
    return v.isoformat() if hasattr(v, "isoformat") else v


def get_customer_profile(customer_id: str) -> Dict[str, Any]:
    with SessionLocal() as db:
        c = db.query(Customer).filter(Customer.customer_id == customer_id).first()
        if not c:
            return {"found": False, "customer_id": customer_id}
        return {
            "found": True,
            "customer_id": c.customer_id,
            "customer_name": c.customer_name,
            "customer_type": c.customer_type,
            "city": c.city,
            "district": c.district,
            "payment_terms": c.payment_terms,
            "credit_limit": _to_float(c.credit_limit),
            "source": "customers",
        }


def get_order_status(order_id: str) -> Dict[str, Any]:
    with SessionLocal() as db:
        o = db.query(Order).filter(Order.order_id == order_id).first()
        if not o:
            return {"found": False, "order_id": order_id}
        return {
            "found": True,
            "order_id": o.order_id,
            "customer_id": o.customer_id,
            "order_date": _to_iso(o.order_date),
            "status": o.status,
            "payment_status": o.payment_status,
            "shipping_status": o.shipping_status,
            "total_amount": _to_float(o.total_amount),
            "delivered_at": _to_iso(o.delivered_at),
            "source": "orders",
        }


def get_order_details(order_id: str) -> Dict[str, Any]:
    with SessionLocal() as db:
        o = db.query(Order).filter(Order.order_id == order_id).first()
        if not o:
            return {"found": False, "order_id": order_id}
        items = db.query(OrderItem).filter(OrderItem.order_id == order_id).all()
        return {
            "found": True,
            "order_id": o.order_id,
            "customer_id": o.customer_id,
            "order_date": _to_iso(o.order_date),
            "status": o.status,
            "payment_status": o.payment_status,
            "shipping_status": o.shipping_status,
            "total_amount": _to_float(o.total_amount),
            "shipping_address": o.shipping_address,
            "delivered_at": _to_iso(o.delivered_at),
            "items": [
                {
                    "sku": it.sku,
                    "quantity": int(it.quantity),
                    "unit_price": _to_float(it.unit_price),
                    "discount_percent": _to_float(it.discount_percent),
                    "line_total": _to_float(it.line_total),
                }
                for it in items
            ],
            "source": "orders + order_items",
        }


def get_customer_order_history(customer_id: str, limit: int = 10) -> Dict[str, Any]:
    with SessionLocal() as db:
        rows = (
            db.query(Order)
            .filter(Order.customer_id == customer_id)
            .order_by(desc(Order.order_date))
            .limit(limit)
            .all()
        )
        orders: List[Dict[str, Any]] = []
        for o in rows:
            orders.append(
                {
                    "order_id": o.order_id,
                    "order_date": _to_iso(o.order_date),
                    "status": o.status,
                    "total_amount": _to_float(o.total_amount),
                }
            )
        return {
            "customer_id": customer_id,
            "count": len(orders),
            "orders": orders,
            "source": "orders",
        }
