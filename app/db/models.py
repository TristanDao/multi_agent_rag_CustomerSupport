"""SQLAlchemy ORM models for the retail/wholesale data model."""
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    Boolean,
)
from sqlalchemy.orm import relationship

from app.db.session import Base


class Product(Base):
    __tablename__ = "products"

    sku = Column(String(64), primary_key=True)
    product_name = Column(String(255), nullable=False, index=True)
    brand = Column(String(128), index=True)
    category = Column(String(128), index=True)
    unit = Column(String(32))
    base_price = Column(Numeric(14, 2), nullable=False)
    description = Column(Text)
    status = Column(String(32), default="active", index=True)
    returnable = Column(Boolean, default=True)
    warranty_months = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    inventory_items = relationship("Inventory", back_populates="product", cascade="all, delete-orphan")
    price_tiers = relationship("PriceTier", back_populates="product", cascade="all, delete-orphan")
    order_items = relationship("OrderItem", back_populates="product")


class Inventory(Base):
    __tablename__ = "inventory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sku = Column(String(64), ForeignKey("products.sku"), nullable=False, index=True)
    warehouse_id = Column(String(32), nullable=False)
    warehouse_location = Column(String(128), index=True)
    stock_quantity = Column(Integer, default=0)
    reserved_quantity = Column(Integer, default=0)
    reorder_level = Column(Integer, default=0)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    product = relationship("Product", back_populates="inventory_items")


class PriceTier(Base):
    __tablename__ = "price_tiers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sku = Column(String(64), ForeignKey("products.sku"), nullable=False, index=True)
    customer_type = Column(String(32), nullable=False, index=True)
    min_quantity = Column(Integer, default=1)
    unit_price = Column(Numeric(14, 2), nullable=False)
    discount_percent = Column(Numeric(5, 2), default=0)
    effective_from = Column(DateTime, default=datetime.utcnow)
    effective_to = Column(DateTime, nullable=True)

    product = relationship("Product", back_populates="price_tiers")


class Customer(Base):
    __tablename__ = "customers"

    customer_id = Column(String(32), primary_key=True)
    customer_name = Column(String(255), nullable=False, index=True)
    customer_type = Column(String(32), nullable=False, index=True)
    city = Column(String(128), index=True)
    district = Column(String(128))
    payment_terms = Column(String(32), default="prepaid")
    credit_limit = Column(Numeric(14, 2), default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    orders = relationship("Order", back_populates="customer")


class Order(Base):
    __tablename__ = "orders"

    order_id = Column(String(32), primary_key=True)
    customer_id = Column(String(32), ForeignKey("customers.customer_id"), nullable=False, index=True)
    order_date = Column(DateTime, default=datetime.utcnow, index=True)
    status = Column(String(32), default="pending", index=True)
    payment_status = Column(String(32), default="unpaid", index=True)
    shipping_status = Column(String(32), default="not_shipped", index=True)
    total_amount = Column(Numeric(14, 2), default=0)
    shipping_address = Column(Text)
    delivered_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    customer = relationship("Customer", back_populates="orders")
    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")


class OrderItem(Base):
    __tablename__ = "order_items"

    order_item_id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(32), ForeignKey("orders.order_id"), nullable=False, index=True)
    sku = Column(String(64), ForeignKey("products.sku"), nullable=False, index=True)
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Numeric(14, 2), nullable=False)
    discount_percent = Column(Numeric(5, 2), default=0)
    line_total = Column(Numeric(14, 2), nullable=False)

    order = relationship("Order", back_populates="items")
    product = relationship("Product", back_populates="order_items")


class Promotion(Base):
    __tablename__ = "promotions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    discount_percent = Column(Numeric(5, 2), default=0)
    min_order_amount = Column(Numeric(14, 2), default=0)
    applies_to = Column(String(64), default="all")
    active = Column(Boolean, default=True)
    start_date = Column(DateTime, default=datetime.utcnow)
    end_date = Column(DateTime, nullable=True)


class SupportTicket(Base):
    __tablename__ = "support_tickets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id = Column(String(32), unique=True, nullable=False, index=True)
    customer_id = Column(String(32), index=True)
    subject = Column(String(255))
    status = Column(String(32), default="open", index=True)
    priority = Column(String(16), default="normal")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
