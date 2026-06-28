"""Load synthetic CSVs into the database via SQLAlchemy."""
import csv
import logging
import os
from datetime import datetime

from app.config import settings
from app.db.models import (
    Customer,
    Inventory,
    Order,
    OrderItem,
    PriceTier,
    Product,
    Promotion,
    SupportTicket,
)
from app.db.session import SessionLocal, init_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "structured")


def _parse_dt(v):
    if not v or v == "" or v == "None":
        return None
    try:
        return datetime.fromisoformat(v)
    except Exception:
        try:
            return datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None


def _to_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes", "y")


def load_products(db, path):
    count = 0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            db.add(
                Product(
                    sku=row["sku"],
                    product_name=row["product_name"],
                    brand=row.get("brand") or None,
                    category=row.get("category") or None,
                    unit=row.get("unit") or None,
                    base_price=row["base_price"],
                    description=row.get("description") or None,
                    status=row.get("status") or "active",
                    returnable=_to_bool(row.get("returnable", "true")),
                    warranty_months=int(row.get("warranty_months") or 0),
                )
            )
            count += 1
    db.commit()
    logger.info("products_loaded count=%s", count)


def load_inventory(db, path):
    count = 0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            db.add(
                Inventory(
                    sku=row["sku"],
                    warehouse_id=row["warehouse_id"],
                    warehouse_location=row.get("warehouse_location") or None,
                    stock_quantity=int(row.get("stock_quantity") or 0),
                    reserved_quantity=int(row.get("reserved_quantity") or 0),
                    reorder_level=int(row.get("reorder_level") or 0),
                )
            )
            count += 1
    db.commit()
    logger.info("inventory_loaded count=%s", count)


def load_price_tiers(db, path):
    count = 0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            db.add(
                PriceTier(
                    sku=row["sku"],
                    customer_type=row["customer_type"],
                    min_quantity=int(row.get("min_quantity") or 1),
                    unit_price=row["unit_price"],
                    discount_percent=float(row.get("discount_percent") or 0),
                )
            )
            count += 1
    db.commit()
    logger.info("price_tiers_loaded count=%s", count)


def load_customers(db, path):
    count = 0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            db.add(
                Customer(
                    customer_id=row["customer_id"],
                    customer_name=row["customer_name"],
                    customer_type=row["customer_type"],
                    city=row.get("city") or None,
                    district=row.get("district") or None,
                    payment_terms=row.get("payment_terms") or "prepaid",
                    credit_limit=float(row.get("credit_limit") or 0),
                )
            )
            count += 1
    db.commit()
    logger.info("customers_loaded count=%s", count)


def load_orders(db, orders_path, items_path):
    order_count = 0
    items_count = 0
    with open(orders_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            db.add(
                Order(
                    order_id=row["order_id"],
                    customer_id=row["customer_id"],
                    order_date=_parse_dt(row.get("order_date")),
                    status=row.get("status") or "pending",
                    payment_status=row.get("payment_status") or "unpaid",
                    shipping_status=row.get("shipping_status") or "not_shipped",
                    total_amount=float(row.get("total_amount") or 0),
                    shipping_address=row.get("shipping_address") or None,
                    delivered_at=_parse_dt(row.get("delivered_at")),
                )
            )
            order_count += 1
    db.commit()
    with open(items_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            db.add(
                OrderItem(
                    order_id=row["order_id"],
                    sku=row["sku"],
                    quantity=int(row.get("quantity") or 0),
                    unit_price=float(row.get("unit_price") or 0),
                    discount_percent=float(row.get("discount_percent") or 0),
                    line_total=float(row.get("line_total") or 0),
                )
            )
            items_count += 1
    db.commit()
    logger.info("orders_loaded=%s items_loaded=%s", order_count, items_count)


def load_promotions(db, path):
    count = 0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            db.add(
                Promotion(
                    code=row["code"],
                    name=row["name"],
                    description=row.get("description"),
                    discount_percent=float(row.get("discount_percent") or 0),
                    min_order_amount=float(row.get("min_order_amount") or 0),
                    applies_to=row.get("applies_to") or "all",
                    active=True,
                )
            )
            count += 1
    db.commit()
    logger.info("promotions_loaded count=%s", count)


def load_tickets(db, path):
    count = 0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            db.add(
                SupportTicket(
                    ticket_id=row["ticket_id"],
                    customer_id=row.get("customer_id"),
                    subject=row.get("subject"),
                    status=row.get("status") or "open",
                    priority=row.get("priority") or "normal",
                )
            )
            count += 1
    db.commit()
    logger.info("tickets_loaded count=%s", count)


def main():
    init_db()
    db = SessionLocal()
    try:
        load_products(db, os.path.join(DATA_DIR, "products.csv"))
        load_inventory(db, os.path.join(DATA_DIR, "inventory.csv"))
        load_price_tiers(db, os.path.join(DATA_DIR, "price_tiers.csv"))
        load_customers(db, os.path.join(DATA_DIR, "customers.csv"))
        load_orders(db, os.path.join(DATA_DIR, "orders.csv"), os.path.join(DATA_DIR, "order_items.csv"))
        load_promotions(db, os.path.join(DATA_DIR, "promotions.csv"))
        load_tickets(db, os.path.join(DATA_DIR, "support_tickets.csv"))
        print("Seed complete.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
