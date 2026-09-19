"""Shared fixtures: a tiny hand-made database that does not depend on the seed generator."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from sqlalchemy.orm import Session

from support_agent import db
from support_agent.clock import KST

NOW = datetime(2026, 9, 14, 10, 0, tzinfo=KST)


def add_tiny_rows(session: Session) -> None:
    """One customer, one address, one product, two orders: O-1 (paid) and O-2 (shipped)."""
    at = datetime(2026, 9, 10, 9, 0, tzinfo=KST)
    session.add(
        db.Customer(
            id="C-1",
            name="김하준",
            phone="01000000001",
            email="hajun@example.com",
            grade=db.CustomerGrade.NORMAL,
            joined_at=at,
        )
    )
    session.flush()
    session.add(
        db.CustomerAddress(
            id="AD-C-1-1",
            customer_id="C-1",
            label="집",
            recipient="김하준",
            postal_code="04524",
            address="서울특별시 중구 가상로 1",
            is_default=True,
        )
    )
    session.add(db.Product(id="P-1", name="무선 이어폰", category="전자기기"))
    session.flush()
    session.add(
        db.ProductVariant(id="V-1-01", product_id="P-1", option_label="블랙", price_won=38900, stock=5)
    )
    session.flush()
    for order_id, status in (("O-1", db.OrderStatus.PAID), ("O-2", db.OrderStatus.SHIPPED)):
        session.add(
            db.Order(
                id=order_id,
                customer_id="C-1",
                status=status,
                ordered_at=at,
                items_won=38900,
                shipping_fee_won=0,
                discount_won=0,
                total_won=38900,
                ship_address_id="AD-C-1-1",
                ship_recipient="김하준",
                ship_postal_code="04524",
                ship_address="서울특별시 중구 가상로 1",
            )
        )
        session.flush()
        session.add(
            db.OrderItem(
                order_id=order_id,
                line_no=1,
                variant_id="V-1-01",
                product_name="무선 이어폰",
                option_label="블랙",
                quantity=1,
                unit_price_won=38900,
                status=db.ItemStatus.ORDERED,
            )
        )
        session.add(
            db.Payment(
                order_id=order_id,
                method=db.PaymentMethod.CARD,
                amount_won=38900,
                status=db.PaymentStatus.PAID,
                refund_won=0,
                paid_at=at,
            )
        )
        session.add(
            db.Shipment(
                order_id=order_id,
                carrier="한빛택배",
                tracking_no=f"5550-{order_id}",
                status=db.ShipmentStatus.READY,
                shipped_at=None,
                delivered_at=None,
                promised_by=date(2026, 9, 13),
            )
        )


@pytest.fixture
def tiny_engine():
    """A seed-like engine; copy it with db.memory_engine(tiny_engine) for a private episode database."""
    engine = db.memory_engine()
    db.Base.metadata.create_all(engine)
    with Session(engine) as session:
        add_tiny_rows(session)
        session.commit()
    yield engine
    engine.dispose()
