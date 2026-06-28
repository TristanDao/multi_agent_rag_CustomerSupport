"""Generate realistic synthetic data for retail/wholesale office supplies.

Outputs:
- data/structured/products.csv
- data/structured/inventory.csv
- data/structured/price_tiers.csv
- data/structured/customers.csv
- data/structured/orders.csv
- data/structured/order_items.csv
- data/structured/promotions.csv
- data/structured/support_tickets.csv
"""
import csv
import os
import random
from datetime import datetime, timedelta
from decimal import Decimal

from faker import Faker

random.seed(42)
Faker.seed(42)
fake = Faker("vi_VN")

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "structured")
os.makedirs(OUT_DIR, exist_ok=True)


BRANDS = [
    "Thiên Long", "Hồng Hà", "Bến Nghé", "Casio", "Stabilo", "Muji",
    "Uniball", "Pilot", "Faber-Castell", "Deli", "Office One", "IK Plus", "Double A",
    "Paper One", "Plus", "HP", "Canon", "Epson", "Brother", "Panasonic",
]

CATEGORIES = [
    ("Bút viết", ["bút bi", "bút gel", "bút chì", "bút dạ quang", "bút lông bảng"]),
    ("Giấy in", ["giấy A4", "giấy A3", "giấy photo", "giấy in nhiệt"]),
    ("Sổ tập", ["sổ tay", "sổ lò xo", "sổ kẻ ngang", "sổ kẻ ca rô", "sổ da"]),
    ("Bìa hồ sơ", ["bìa còng", "bìa lá", "bìa nhựa", "bìa da"]),
    ("Mực in", ["mực HP", "mực Canon", "mực Epson", "mực Brother"]),
    ("Băng keo", ["băng keo trong", "băng keo đục", "băng keo giấy", "băng keo 2 mặt"]),
    ("Kẹp giấy & ghim", ["kẹp giấy", "ghim dập", "ghim bấm", "kim kẹp"]),
    ("Máy tính cầm tay", ["máy tính khoa học", "máy tính đơn giản", "máy tính tài chính"]),
    ("Bảng văn phòng", ["bảng trắng", "bảng từ", "bảng cork", "bảng flipchart"]),
    ("Khay & hộp đựng", ["khay nhựa", "khay kim loại", "hộp bút", "hộp tài liệu"]),
    ("Dụng cụ học sinh", ["thước kẻ", "com-pa", "tẩy", "gọt bút chì"]),
    ("Thiết bị văn phòng", ["máy in", "máy hủy tài liệu", "máy đóng gáy", "máy ép plastic"]),
]

UNITS = {
    "Bút viết": "cây",
    "Giấy in": "ram",
    "Sổ tập": "cuốn",
    "Bìa hồ sơ": "cái",
    "Mực in": "hộp",
    "Băng keo": "cuộn",
    "Kẹp giấy & ghim": "hộp",
    "Máy tính cầm tay": "cái",
    "Bảng văn phòng": "cái",
    "Khay & hộp đựng": "cái",
    "Dụng cụ học sinh": "cái",
    "Thiết bị văn phòng": "cái",
}

CITIES = ["Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Cần Thơ", "Hải Phòng", "Nha Trang", "Bình Dương", "Đồng Nai", "Long An"]
DISTRICTS = ["Quận 1", "Quận 3", "Quận Bình Thạnh", "Quận Gò Vấp", "Quận Tân Bình", "Huyện Bình Chánh", "Quận Cầu Giấy", "Quận Hai Bà Trưng", "Quận Đống Đa", "Quận Hải Châu"]
WAREHOUSES = [
    ("WH_HCM_1", "Kho TP.HCM - Bình Tân"),
    ("WH_HN_1", "Kho Hà Nội - Long Biên"),
    ("WH_DN_1", "Kho Đà Nẵng - Hải Châu"),
    ("WH_CT_1", "Kho Cần Thơ - Ninh Kiều"),
]


def gen_product_name(category: str, sub: str, brand: str) -> str:
    templates = {
        "Bút viết": [
            "{brand} {sub} {model}",
            "{brand} {sub} cao cấp {model}",
        ],
        "Giấy in": [
            "Giấy A4 {brand} {gsm}gsm",
            "Giấy A4 {brand} {gsm}gsm - Thùng 5 ram",
        ],
        "Sổ tập": [
            "Sổ tay {brand} {pages} trang",
            "Sổ lò xo {brand} {pages} trang",
        ],
        "Bìa hồ sơ": [
            "Bìa còng {brand} {size}",
            "Bìa lá {brand} {size}",
        ],
        "Mực in": [
            "Mực in {brand} {model}",
            "Hộp mực {brand} {model}",
        ],
        "Băng keo": [
            "Băng keo {sub} {brand} {width}mm",
        ],
        "Kẹp giấy & ghim": [
            "{sub} {brand} {size}",
        ],
        "Máy tính cầm tay": [
            "Máy tính {brand} {model}",
        ],
        "Bảng văn phòng": [
            "{sub} {brand} {size}cm",
        ],
        "Khay & hộp đựng": [
            "{sub} {brand} {layers} tầng",
        ],
        "Dụng cụ học sinh": [
            "{sub} {brand}",
        ],
        "Thiết bị văn phòng": [
            "{brand} {model} ({sub})",
        ],
    }
    tpl = random.choice(templates.get(category, ["{brand} {sub} {model}"]))
    return tpl.format(
        brand=brand,
        sub=sub,
        model=f"{random.choice(['TL','PR','HD','XP','MC'])}-{random.randint(100,999)}",
        gsm=random.choice([70, 80, 100]),
        pages=random.choice([80, 100, 120, 200]),
        size=random.choice(["A4", "A5", "F4"]),
        width=random.choice([12, 24, 48]),
        layers=random.choice([2, 3, 4, 5]),
    )


def gen_products(n: int = 500):
    products = []
    for i in range(n):
        category, subs = random.choice(CATEGORIES)
        sub = random.choice(subs)
        brand = random.choice(BRANDS)
        name = gen_product_name(category, sub, brand)
        sku = f"SKU{i+1:05d}"
        unit = UNITS.get(category, "cái")
        base_price = random.randint(5_000, 1_200_000)
        warranty = random.choice([0, 0, 0, 6, 12, 24])
        returnable = random.random() < 0.85
        status = "active" if random.random() > 0.05 else "discontinued"
        description = f"{name} thuộc danh mục {category}, phù hợp cho văn phòng, trường học và cửa hàng."
        products.append(
            dict(
                sku=sku,
                product_name=name,
                brand=brand,
                category=category,
                unit=unit,
                base_price=base_price,
                description=description,
                status=status,
                returnable=returnable,
                warranty_months=warranty,
            )
        )
    return products


def gen_inventory(products):
    inv = []
    for p in products:
        if p["status"] != "active":
            continue
        n_wh = random.choice([1, 1, 2, 3])
        for w in random.sample(WAREHOUSES, n_wh):
            qty = random.randint(0, 5_000)
            reserved = random.randint(0, min(qty, 200))
            inv.append(
                dict(
                    sku=p["sku"],
                    warehouse_id=w[0],
                    warehouse_location=w[1],
                    stock_quantity=qty,
                    reserved_quantity=reserved,
                    reorder_level=50,
                )
            )
    return inv


def gen_price_tiers(products):
    tiers = []
    for p in products:
        if p["status"] != "active":
            continue
        for ctype in ["retail", "wholesale", "corporate", "distributor"]:
            base = p["base_price"]
            if ctype == "retail":
                tiers.append(
                    dict(
                        sku=p["sku"],
                        customer_type=ctype,
                        min_quantity=1,
                        unit_price=base,
                        discount_percent=0,
                    )
                )
            elif ctype == "wholesale":
                for q, d in [(10, 3), (50, 8), (200, 15)]:
                    price = round(base * (1 - d / 100.0) / 1000) * 1000
                    tiers.append(
                        dict(
                            sku=p["sku"],
                            customer_type=ctype,
                            min_quantity=q,
                            unit_price=price,
                            discount_percent=d,
                        )
                    )
            elif ctype == "corporate":
                for q, d in [(5, 5), (30, 10), (100, 18)]:
                    price = round(base * (1 - d / 100.0) / 1000) * 1000
                    tiers.append(
                        dict(
                            sku=p["sku"],
                            customer_type=ctype,
                            min_quantity=q,
                            unit_price=price,
                            discount_percent=d,
                        )
                    )
            else:  # distributor
                for q, d in [(50, 12), (200, 20), (500, 25)]:
                    price = round(base * (1 - d / 100.0) / 1000) * 1000
                    tiers.append(
                        dict(
                            sku=p["sku"],
                            customer_type=ctype,
                            min_quantity=q,
                            unit_price=price,
                            discount_percent=d,
                        )
                    )
    return tiers


def gen_customers(n: int = 120):
    customers = []
    for i in range(n):
        ctype = random.choices(
            ["retail", "wholesale", "corporate", "distributor"], weights=[60, 25, 10, 5]
        )[0]
        cid = f"C{i+1:05d}"
        name = (
            fake.company() if ctype != "retail" else fake.name()
        )
        city = random.choice(CITIES)
        district = random.choice(DISTRICTS)
        if ctype == "retail":
            terms = "prepaid"
            credit = 0
        elif ctype == "wholesale":
            terms = random.choice(["NET 7", "NET 15", "prepaid"])
            credit = random.choice([0, 0, 10_000_000, 30_000_000])
        elif ctype == "corporate":
            terms = random.choice(["NET 15", "NET 30"])
            credit = random.choice([50_000_000, 100_000_000, 200_000_000])
        else:
            terms = "NET 30"
            credit = random.choice([100_000_000, 200_000_000, 500_000_000])
        customers.append(
            dict(
                customer_id=cid,
                customer_name=name,
                customer_type=ctype,
                city=city,
                district=district,
                payment_terms=terms,
                credit_limit=credit,
            )
        )
    return customers


def gen_orders(customers, products, n: int = 1500):
    orders = []
    items_pool = []
    statuses = ["pending", "confirmed", "packed", "shipped", "delivered", "delivered", "delivered", "cancelled", "returned"]
    pay_statuses = ["unpaid", "partial", "paid", "paid", "paid", "refunded"]
    ship_statuses = ["not_shipped", "preparing", "in_transit", "delivered", "delivered", "delivered", "failed"]
    start = datetime.utcnow() - timedelta(days=365)
    for i in range(n):
        c = random.choice(customers)
        order_date = start + timedelta(days=random.randint(0, 365))
        status = random.choice(statuses)
        pay = "paid" if status in ("delivered", "shipped", "packed") else random.choice(pay_statuses)
        ship = (
            "delivered" if status == "delivered"
            else "in_transit" if status == "shipped"
            else "preparing" if status == "packed"
            else "not_shipped" if status in ("pending", "confirmed")
            else "failed" if status == "cancelled"
            else "delivered"
        )
        delivered_at = None
        if status == "delivered":
            delivered_at = order_date + timedelta(days=random.randint(1, 7))
        oid = f"DH{i+1:05d}"
        n_items = random.randint(1, 5)
        line_total = 0
        for j in range(n_items):
            p = random.choice(products)
            if p["status"] != "active":
                continue
            qty = random.randint(1, 30)
            unit = p["base_price"]
            disc = random.choice([0, 0, 0, 3, 5, 8])
            lt = round(float(unit) * qty * (1 - disc / 100.0) / 1000) * 1000
            line_total += lt
            items_pool.append(
                dict(
                    order_id=oid,
                    sku=p["sku"],
                    quantity=qty,
                    unit_price=float(unit),
                    discount_percent=disc,
                    line_total=lt,
                )
            )
        orders.append(
            dict(
                order_id=oid,
                customer_id=c["customer_id"],
                order_date=order_date,
                status=status,
                payment_status=pay,
                shipping_status=ship,
                total_amount=line_total,
                shipping_address=f"{random.randint(1, 500)} {random.choice(['Nguyễn Trãi','Lê Lợi','Trần Hưng Đạo','Cách Mạng Tháng 8'])}, {random.choice(DISTRICTS)}, {c['city']}",
                delivered_at=delivered_at,
            )
        )
    return orders, items_pool


def gen_promotions():
    return [
        dict(
            code="WELCOME10",
            name="Giảm 10% đơn hàng đầu tiên",
            description="Áp dụng cho khách hàng mới, đơn tối thiểu 200.000đ",
            discount_percent=10,
            min_order_amount=200_000,
            applies_to="all",
        ),
        dict(
            code="BULK15",
            name="Giảm 15% cho đơn sỉ trên 10 triệu",
            description="Áp dụng cho đơn wholesale từ 10.000.000đ",
            discount_percent=15,
            min_order_amount=10_000_000,
            applies_to="wholesale",
        ),
        dict(
            code="SCHOOL2026",
            name="Ưu đãi mùa tựu trường",
            description="Áp dụng cho danh mục Bút viết, Sổ tập, Dụng cụ học sinh",
            discount_percent=8,
            min_order_amount=300_000,
            applies_to="category:student",
        ),
    ]


def gen_support_tickets(customers, n: int = 30):
    tickets = []
    for i in range(n):
        c = random.choice(customers)
        tickets.append(
            dict(
                ticket_id=f"TK{i+1:05d}",
                customer_id=c["customer_id"],
                subject=random.choice(
                    [
                        "Yêu cầu hoàn tiền",
                        "Hỏi về tình trạng đơn hàng",
                        "Khiếu nại chất lượng sản phẩm",
                        "Đề xuất hợp tác wholesale",
                        "Hỏi chính sách đổi trả",
                    ]
                ),
                status=random.choice(["open", "in_progress", "resolved", "closed"]),
                priority=random.choice(["low", "normal", "high"]),
            )
        )
    return tickets


def write_csv(path: str, rows: list, fieldnames: list):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    print("Generating products...")
    products = gen_products(500)
    write_csv(
        os.path.join(OUT_DIR, "products.csv"),
        products,
        [
            "sku",
            "product_name",
            "brand",
            "category",
            "unit",
            "base_price",
            "description",
            "status",
            "returnable",
            "warranty_months",
        ],
    )
    print("Generating inventory...")
    inv = gen_inventory(products)
    write_csv(
        os.path.join(OUT_DIR, "inventory.csv"),
        inv,
        ["sku", "warehouse_id", "warehouse_location", "stock_quantity", "reserved_quantity", "reorder_level"],
    )
    print("Generating price_tiers...")
    tiers = gen_price_tiers(products)
    write_csv(
        os.path.join(OUT_DIR, "price_tiers.csv"),
        tiers,
        ["sku", "customer_type", "min_quantity", "unit_price", "discount_percent"],
    )
    print("Generating customers...")
    customers = gen_customers(120)
    write_csv(
        os.path.join(OUT_DIR, "customers.csv"),
        customers,
        ["customer_id", "customer_name", "customer_type", "city", "district", "payment_terms", "credit_limit"],
    )
    print("Generating orders + order_items...")
    orders, items = gen_orders(customers, products, 1500)
    write_csv(
        os.path.join(OUT_DIR, "orders.csv"),
        orders,
        ["order_id", "customer_id", "order_date", "status", "payment_status", "shipping_status", "total_amount", "shipping_address", "delivered_at"],
    )
    write_csv(
        os.path.join(OUT_DIR, "order_items.csv"),
        items,
        ["order_id", "sku", "quantity", "unit_price", "discount_percent", "line_total"],
    )
    print("Generating promotions...")
    promos = gen_promotions()
    write_csv(
        os.path.join(OUT_DIR, "promotions.csv"),
        promos,
        ["code", "name", "description", "discount_percent", "min_order_amount", "applies_to"],
    )
    print("Generating support_tickets...")
    tickets = gen_support_tickets(customers, 30)
    write_csv(
        os.path.join(OUT_DIR, "support_tickets.csv"),
        tickets,
        ["ticket_id", "customer_id", "subject", "status", "priority"],
    )
    print("Done.")


if __name__ == "__main__":
    main()
