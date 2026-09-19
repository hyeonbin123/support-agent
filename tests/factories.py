"""Row-building helpers for the tool tests. They extend conftest's tiny database; nothing depends on seed.py.

The shop after `extend_tiny_rows` (NOW is 2026-09-14 10:00 KST):

  C-1 김하준  addresses AD-C-1-1 (집, default), AD-C-1-2 (회사)
    O-1 paid       V-1-01 x1            promised 09-13, not delivered (1 day late)
    O-2 shipped    V-1-01 x1            promised 09-13, not delivered (1 day late)
    O-3 delivered  V-1-01 x1, V-2-01 x2 delivered 09-08 15:00, promised 09-09 (on time, window open)
    O-4 delivered  V-1-01 x1            delivered 09-02 14:20, promised 08-30 (3 days late, window closed)
    O-5 preparing  V-1-01 x1            promised 09-16 (not late)
    O-6 cancelled  V-1-01 x1
  C-2 이서연  address AD-C-2-1
    O-9 paid       V-1-01 x1
  P-1 무선 이어폰: V-1-01 블랙 (5), V-1-02 화이트 (3), V-1-03 레드 (0)   P-2 텀블러: V-2-01 (10), V-2-02 (10)
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from sqlalchemy.orm import Session

from support_agent import db
from support_agent.clock import KST

SHIPMENT_STATUS = {
    db.OrderStatus.PAID: db.ShipmentStatus.READY,
    db.OrderStatus.PREPARING: db.ShipmentStatus.READY,
    db.OrderStatus.SHIPPED: db.ShipmentStatus.IN_TRANSIT,
    db.OrderStatus.DELIVERED: db.ShipmentStatus.DELIVERED,
    db.OrderStatus.CANCELLED: db.ShipmentStatus.READY,
}


def kst(month: int, day: int, hour: int = 9, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=KST)


def add_customer(session: Session, customer_id: str, name: str, phone: str, email: str) -> None:
    session.add(
        db.Customer(
            id=customer_id,
            name=name,
            phone=phone,
            email=email,
            grade=db.CustomerGrade.NORMAL,
            joined_at=kst(6, 1),
        )
    )
    session.flush()


def add_address(
    session: Session, address_id: str, customer_id: str, label: str, address: str, *, is_default: bool = False
) -> None:
    customer = session.get(db.Customer, customer_id)
    session.add(
        db.CustomerAddress(
            id=address_id,
            customer_id=customer_id,
            label=label,
            recipient=customer.name,
            postal_code="04524",
            address=address,
            is_default=is_default,
        )
    )
    session.flush()


def add_product(
    session: Session, product_id: str, name: str, variants: list[tuple[str, str, int, int]]
) -> None:
    """variants: (variant_id, option_label, price_won, stock)."""
    if session.get(db.Product, product_id) is None:
        session.add(db.Product(id=product_id, name=name, category="생활"))
        session.flush()
    for variant_id, option_label, price_won, stock in variants:
        session.add(
            db.ProductVariant(
                id=variant_id,
                product_id=product_id,
                option_label=option_label,
                price_won=price_won,
                stock=stock,
            )
        )
    session.flush()


def add_order(
    session: Session,
    order_id: str,
    customer_id: str,
    address_id: str,
    status: db.OrderStatus,
    lines: list[tuple[str, int]],
    *,
    ordered_at: datetime,
    promised_by: date,
    delivered_at: datetime | None = None,
    shipping_fee_won: int = 0,
) -> None:
    """lines: (variant_id, quantity). A cancelled order gets the rows cancel_order would have written."""
    address = session.get(db.CustomerAddress, address_id)
    cancelled = status == db.OrderStatus.CANCELLED
    variants = [(session.get(db.ProductVariant, variant_id), quantity) for variant_id, quantity in lines]
    items_won = sum(v.price_won * quantity for v, quantity in variants)
    total = items_won + shipping_fee_won
    session.add(
        db.Order(
            id=order_id,
            customer_id=customer_id,
            status=status,
            ordered_at=ordered_at,
            items_won=items_won,
            shipping_fee_won=shipping_fee_won,
            discount_won=0,
            total_won=total,
            ship_address_id=address.id,
            ship_recipient=address.recipient,
            ship_postal_code=address.postal_code,
            ship_address=address.address,
            cancelled_at=ordered_at + timedelta(hours=1) if cancelled else None,
            cancel_reason=db.CancelReason.CHANGED_MIND if cancelled else None,
        )
    )
    session.flush()
    for line_no, (variant, quantity) in enumerate(variants, start=1):
        session.add(
            db.OrderItem(
                order_id=order_id,
                line_no=line_no,
                variant_id=variant.id,
                product_name=session.get(db.Product, variant.product_id).name,
                option_label=variant.option_label,
                quantity=quantity,
                unit_price_won=variant.price_won,
                status=db.ItemStatus.CANCELLED if cancelled else db.ItemStatus.ORDERED,
            )
        )
    session.add(
        db.Payment(
            order_id=order_id,
            method=db.PaymentMethod.CARD,
            amount_won=total,
            status=db.PaymentStatus.REFUND_PENDING if cancelled else db.PaymentStatus.PAID,
            refund_won=total if cancelled else 0,
            paid_at=ordered_at,
        )
    )
    shipped = status in (db.OrderStatus.SHIPPED, db.OrderStatus.DELIVERED)
    session.add(
        db.Shipment(
            order_id=order_id,
            carrier="한빛택배",
            tracking_no=f"5550-{order_id}",
            status=SHIPMENT_STATUS[status],
            shipped_at=ordered_at + timedelta(days=1) if shipped else None,
            delivered_at=delivered_at,
            promised_by=promised_by,
        )
    )
    session.flush()


def add_compensation_coupon(
    session: Session,
    coupon_id: str,
    customer_id: str,
    order_id: str,
    issued_at: datetime,
    amount_won: int = 2000,
) -> None:
    session.add(
        db.Coupon(
            id=coupon_id,
            customer_id=customer_id,
            kind=db.CouponKind.COMPENSATION,
            amount_won=amount_won,
            reason=db.CompensationReason.DELIVERY_DELAY,
            order_id=order_id,
            issued_at=issued_at,
            expires_at=issued_at + timedelta(days=30),
            used_at=None,
        )
    )
    session.flush()


def extend_tiny_rows(session: Session) -> None:
    """Add the rows of the module docstring on top of conftest.add_tiny_rows."""
    add_address(session, "AD-C-1-2", "C-1", "회사", "서울특별시 중구 가상로 22")
    add_customer(session, "C-2", "이서연", "01000000002", "seoyeon@example.com")
    add_address(session, "AD-C-2-1", "C-2", "집", "부산광역시 가상구 모의로 3", is_default=True)
    add_product(session, "P-1", "무선 이어폰", [("V-1-02", "화이트", 38900, 3), ("V-1-03", "레드", 38900, 0)])
    add_product(session, "P-2", "텀블러", [("V-2-01", "500ml", 14700, 10), ("V-2-02", "700ml", 16700, 10)])
    delivered = db.OrderStatus.DELIVERED
    add_order(
        session,
        "O-3",
        "C-1",
        "AD-C-1-1",
        delivered,
        [("V-1-01", 1), ("V-2-01", 2)],
        ordered_at=kst(9, 5),
        promised_by=date(2026, 9, 9),
        delivered_at=kst(9, 8, 15, 0),
        shipping_fee_won=3000,
    )
    add_order(
        session,
        "O-4",
        "C-1",
        "AD-C-1-1",
        delivered,
        [("V-1-01", 1)],
        ordered_at=kst(8, 28),
        promised_by=date(2026, 8, 30),
        delivered_at=kst(9, 2, 14, 20),
    )
    add_order(
        session,
        "O-5",
        "C-1",
        "AD-C-1-1",
        db.OrderStatus.PREPARING,
        [("V-1-01", 1)],
        ordered_at=kst(9, 12),
        promised_by=date(2026, 9, 16),
    )
    add_order(
        session,
        "O-6",
        "C-1",
        "AD-C-1-1",
        db.OrderStatus.CANCELLED,
        [("V-1-01", 1)],
        ordered_at=kst(9, 9),
        promised_by=date(2026, 9, 12),
    )
    add_order(
        session,
        "O-9",
        "C-2",
        "AD-C-2-1",
        db.OrderStatus.PAID,
        [("V-1-01", 1)],
        ordered_at=kst(9, 11),
        promised_by=date(2026, 9, 15),
    )
