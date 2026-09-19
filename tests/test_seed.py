"""The seed database: determinism, the fixture rows tasks rely on, and consistency of every row."""

from __future__ import annotations

import ast
import re
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from support_agent import db, seed
from support_agent.clock import KST


@pytest.fixture(scope="module")
def engine():
    return seed.build_seed_engine()


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s


def kst(month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=KST)


# --- determinism ---


def test_two_fresh_builds_have_the_same_hash():
    first = seed.build_seed_engine.__wrapped__()
    second = seed.build_seed_engine.__wrapped__()
    assert first is not second
    assert db.state_hash(db.dump_db(first)) == db.state_hash(db.dump_db(second))


def test_cached_build_is_shared_and_equal_to_a_fresh_one(engine):
    assert seed.build_seed_engine() is engine
    fresh = seed.build_seed_engine.__wrapped__()
    assert db.diff_dumps(db.dump_db(engine), db.dump_db(fresh)) == {}


def test_copy_of_seed_is_equal(engine):
    copy = db.memory_engine(engine)
    assert db.state_hash(db.dump_db(copy)) == db.state_hash(db.dump_db(engine))


def test_seed_source_has_no_clock_uuid_or_global_random():
    tree = ast.parse(Path(seed.__file__).read_text(encoding="utf-8"))

    def dotted(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{dotted(node.value)}.{node.attr}"
        return "?"

    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [
                a.name for a in node.names if a.name.split(".")[0] in {"uuid", "faker", "secrets", "time"}
            ]
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module in {"uuid", "faker", "secrets", "time"}:
                bad.append(f"from {node.module}")
            if module == "random":
                bad += [f"random.{a.name}" for a in node.names if a.name != "Random"]
        elif isinstance(node, ast.Call):
            name = dotted(node.func)
            if name.split(".")[-1] in {"now", "utcnow", "today"}:
                bad.append(name)
            if name.startswith("uuid."):
                bad.append(name)
            if name.startswith("random.") and name != "random.Random":
                bad.append(name)
    assert bad == []


# --- fixtures ---


def test_constants():
    assert seed.SEED == 20260919
    assert seed.SEED_END == kst(9, 10, 23, 59)
    assert seed.SEED_END.utcoffset() is not None
    assert seed.FIXTURE_CUSTOMER_IDS == ("C-9001", "C-9002", "C-9003", "C-9004", "C-9005")
    assert seed.FIXTURE_ORDER_IDS == ("O-90001", "O-90002", "O-90003", "O-90004", "O-90005", "O-90006")


@pytest.mark.parametrize(
    ("customer_id", "name", "phone", "email", "address_labels"),
    [
        ("C-9001", "김하준", "01000009001", "hajun.kim@example.com", {"AD-C-9001-1": "집"}),
        ("C-9002", "이서연", "01000009002", "seoyeon.lee@example.com", {"AD-C-9002-1": "집"}),
        ("C-9003", "박도윤", "01000009003", "doyun.park@example.com", {"AD-C-9003-1": "집"}),
        ("C-9004", "최지우", "01000009004", "jiwoo.choi@example.com", {"AD-C-9004-1": "집"}),
        (
            "C-9005",
            "정예준",
            "01000009005",
            "yejun.jung@example.com",
            {"AD-C-9005-1": "집", "AD-C-9005-2": "회사"},
        ),
    ],
)
def test_fixture_customers(session, customer_id, name, phone, email, address_labels):
    customer = session.get(db.Customer, customer_id)
    assert (customer.name, customer.phone, customer.email) == (name, phone, email)
    assert {a.id: a.label for a in customer.addresses} == address_labels
    assert [a.id for a in customer.addresses if a.is_default] == [f"AD-{customer_id}-1"]
    assert all(a.recipient == name for a in customer.addresses)


def test_fixture_customers_have_no_coupons_tickets_or_requests(session):
    ids = seed.FIXTURE_CUSTOMER_IDS
    assert session.scalars(select(db.Coupon).where(db.Coupon.customer_id.in_(ids))).all() == []
    assert session.scalars(select(db.Ticket).where(db.Ticket.customer_id.in_(ids))).all() == []
    requests = select(db.ServiceRequest).where(db.ServiceRequest.order_id.in_(seed.FIXTURE_ORDER_IDS))
    assert session.scalars(requests).all() == []


def test_fixture_orders_belong_to_the_right_customers(session):
    owners = {order_id: session.get(db.Order, order_id).customer_id for order_id in seed.FIXTURE_ORDER_IDS}
    assert owners == {
        "O-90001": "C-9001",
        "O-90002": "C-9002",
        "O-90003": "C-9003",
        "O-90004": "C-9004",
        "O-90005": "C-9005",
        "O-90006": "C-9005",
    }
    for order_id in seed.FIXTURE_ORDER_IDS:
        order = session.get(db.Order, order_id)
        assert order.discount_won == 0
        assert order.ship_address_id == f"AD-{order.customer_id}-1"
    fixture_orders = select(db.Order.id).where(db.Order.customer_id.in_(seed.FIXTURE_CUSTOMER_IDS))
    assert sorted(session.scalars(fixture_orders)) == list(seed.FIXTURE_ORDER_IDS)


def test_o90001_in_transit(session):
    order = session.get(db.Order, "O-90001")
    assert order.status == db.OrderStatus.SHIPPED
    assert (order.total_won, order.shipping_fee_won) == (38900, 0)
    assert [(i.product_name, i.quantity, i.unit_price_won) for i in order.items] == [
        ("무선 이어폰", 1, 38900)
    ]
    shipment = order.shipment
    assert shipment.status == db.ShipmentStatus.IN_TRANSIT
    assert (shipment.carrier, shipment.tracking_no) == ("한빛택배", "5550-1207-9001")
    assert shipment.shipped_at.astimezone(KST).date() == date(2026, 9, 12)
    assert shipment.delivered_at is None
    assert shipment.promised_by == date(2026, 9, 15)
    assert order.payment.status == db.PaymentStatus.PAID


def test_o90002_cancelled(session):
    order = session.get(db.Order, "O-90002")
    assert order.status == db.OrderStatus.CANCELLED
    assert order.cancelled_at.astimezone(KST).date() == date(2026, 9, 11)
    assert order.cancel_reason == db.CancelReason.CHANGED_MIND
    assert order.ordered_at < order.cancelled_at
    payment = order.payment
    assert (payment.amount_won, payment.status, payment.refund_won) == (
        47300,
        db.PaymentStatus.REFUND_PENDING,
        47300,
    )
    assert order.items and all(i.status == db.ItemStatus.CANCELLED for i in order.items)
    assert order.shipment.status == db.ShipmentStatus.READY
    assert order.shipment.shipped_at is None


def test_o90003_paid(session):
    order = session.get(db.Order, "O-90003")
    assert order.status == db.OrderStatus.PAID
    assert order.ordered_at.astimezone(KST).date() == date(2026, 9, 13)
    assert [(i.product_name, i.quantity, i.unit_price_won) for i in order.items] == [("텀블러", 2, 14700)]
    assert (order.items_won, order.shipping_fee_won, order.total_won) == (29400, 3000, 32400)
    assert order.payment.method == db.PaymentMethod.CARD
    assert order.payment.status == db.PaymentStatus.PAID
    assert order.shipment.status == db.ShipmentStatus.READY


def test_o90004_delivered(session):
    order = session.get(db.Order, "O-90004")
    assert order.status == db.OrderStatus.DELIVERED
    assert [(i.product_name, i.quantity, i.unit_price_won) for i in order.items] == [
        ("블루투스 스피커", 1, 56800)
    ]
    assert (order.total_won, order.discount_won) == (56800, 0)
    shipment = order.shipment
    assert shipment.status == db.ShipmentStatus.DELIVERED
    assert shipment.delivered_at == kst(9, 2, 14, 20)
    assert shipment.shipped_at < shipment.delivered_at
    assert shipment.promised_by >= date(2026, 9, 2)  # on time: no delay compensation


def test_o90005_preparing(session):
    order = session.get(db.Order, "O-90005")
    assert order.status == db.OrderStatus.PREPARING
    assert order.ordered_at.astimezone(KST).date() == date(2026, 9, 12)
    assert [(i.product_name, i.quantity, i.unit_price_won) for i in order.items] == [("러닝화", 1, 89100)]
    assert order.total_won == 89100
    assert order.ship_address_id == "AD-C-9005-1"
    assert order.shipment.status == db.ShipmentStatus.READY


def test_o90006_paid(session):
    order = session.get(db.Order, "O-90006")
    assert order.status == db.OrderStatus.PAID
    assert order.ordered_at.astimezone(KST).date() == date(2026, 9, 13)
    assert [(i.product_name, i.quantity, i.unit_price_won) for i in order.items] == [("양말 세트", 3, 7300)]
    assert (order.items_won, order.shipping_fee_won, order.total_won) == (21900, 3000, 24900)


def test_fixture_products_are_dedicated_and_exchangeable(session):
    fixture_variants = set()
    for order_id in seed.FIXTURE_ORDER_IDS:
        for item in session.get(db.Order, order_id).items:
            variant = session.get(db.ProductVariant, item.variant_id)
            assert variant.product_id.startswith("P-9")
            fixture_variants.add(variant.id)
            in_stock = [v for v in session.get(db.Product, variant.product_id).variants if v.stock > 0]
            assert len(in_stock) >= 2
    bulk_items = select(db.OrderItem).where(db.OrderItem.order_id.not_in(seed.FIXTURE_ORDER_IDS))
    assert not {i.variant_id for i in session.scalars(bulk_items)} & fixture_variants


# --- every row ---


def test_row_counts(session):
    def count(model) -> int:
        return len(session.scalars(select(model)).all())

    assert count(db.Customer) == 40 + 5
    assert 60 <= count(db.CustomerAddress) - 6 <= 80
    assert count(db.Product) == 25 + 6
    assert 50 <= count(db.ProductVariant) - 13 <= 70
    assert count(db.Order) == 120 + 6
    assert 3 <= count(db.ServiceRequest) <= 12
    assert 25 <= count(db.Coupon) <= 35
    assert count(db.Ticket) == 15
    assert count(db.Handoff) == 0
    assert {o.status for o in session.scalars(select(db.Order))} == set(db.OrderStatus)
    assert {r.kind for r in session.scalars(select(db.ServiceRequest))} == set(db.RequestKind)
    assert {c.kind for c in session.scalars(select(db.Coupon))} == set(db.CouponKind)


def test_id_formats(session):
    patterns = {
        db.Customer: r"C-\d{4}",
        db.CustomerAddress: r"AD-C-\d{4}-\d",
        db.Product: r"P-\d{4}",
        db.ProductVariant: r"V-\d{4}-\d{2}",
        db.Order: r"O-\d{5}",
        db.ServiceRequest: r"(RT|EX)-O-\d{5}-\d",
        db.Coupon: r"CP-(C-\d{4}-P\d+|O-\d{5}-\d+)",
        db.Ticket: r"TK-C-\d{4}-\d+",
    }
    for model, pattern in patterns.items():
        for row in session.scalars(select(model)):
            assert re.fullmatch(pattern, row.id), row.id
    for address in session.scalars(select(db.CustomerAddress)):
        assert address.id.startswith(f"AD-{address.customer_id}-")
    for variant in session.scalars(select(db.ProductVariant)):
        assert variant.id.startswith(f"V-{variant.product_id[2:]}-")


def test_customers_are_fictional_and_unique(session):
    customers = session.scalars(select(db.Customer)).all()
    for customer in customers:
        assert re.fullmatch(r"0100000\d{4}", customer.phone)
        assert customer.email == customer.email.lower()
        assert customer.email.endswith("@example.com")
    assert len({c.phone for c in customers}) == len(customers)
    assert len({c.email for c in customers}) == len(customers)
    assert len({(c.name, c.phone) for c in customers}) == len(customers)
    fixture_names = {c.name for c in customers if c.id in seed.FIXTURE_CUSTOMER_IDS}
    assert not fixture_names & {c.name for c in customers if c.id not in seed.FIXTURE_CUSTOMER_IDS}
    for customer in customers:
        assert [a.is_default for a in customer.addresses].count(True) == 1


def test_order_totals_payments_and_snapshots(session):
    for order in session.scalars(select(db.Order)):
        assert order.items, order.id
        assert [i.line_no for i in order.items] == list(range(1, len(order.items) + 1))
        assert order.items_won == sum(i.unit_price_won * i.quantity for i in order.items)
        assert order.total_won == order.items_won + order.shipping_fee_won - order.discount_won
        assert order.shipping_fee_won == seed.shipping_fee(order.items_won)
        assert 0 <= order.discount_won < order.items_won
        assert order.payment.amount_won == order.total_won
        assert order.ordered_at <= order.payment.paid_at

        address = session.get(db.CustomerAddress, order.ship_address_id)
        assert address.customer_id == order.customer_id
        assert (order.ship_recipient, order.ship_postal_code, order.ship_address) == (
            address.recipient,
            address.postal_code,
            address.address,
        )
        for item in order.items:
            variant = session.get(db.ProductVariant, item.variant_id)
            product = session.get(db.Product, variant.product_id)
            assert (item.product_name, item.option_label) == (product.name, variant.option_label)
            assert item.unit_price_won == variant.price_won
            assert item.quantity >= 1


def test_status_matches_payment_and_shipment(session):
    ready, moving, done = db.ShipmentStatus.READY, db.ShipmentStatus.IN_TRANSIT, db.ShipmentStatus.DELIVERED
    expected = {
        db.OrderStatus.PAID: ready,
        db.OrderStatus.PREPARING: ready,
        db.OrderStatus.CANCELLED: ready,
        db.OrderStatus.SHIPPED: moving,
        db.OrderStatus.DELIVERED: done,
    }
    for order in session.scalars(select(db.Order)):
        shipment, payment = order.shipment, order.payment
        assert shipment.status == expected[order.status], order.id
        assert (shipment.shipped_at is None) == (shipment.status == ready)
        assert (shipment.delivered_at is None) == (shipment.status != done)
        if shipment.shipped_at is not None:
            assert order.ordered_at < shipment.shipped_at
        if shipment.delivered_at is not None:
            assert shipment.shipped_at < shipment.delivered_at
        assert shipment.promised_by > order.ordered_at.astimezone(KST).date()
        assert shipment.carrier.endswith("택배")

        if order.status == db.OrderStatus.CANCELLED:
            assert order.cancelled_at is not None and order.cancel_reason is not None
            assert order.ordered_at < order.cancelled_at
            assert payment.status in (db.PaymentStatus.REFUND_PENDING, db.PaymentStatus.REFUNDED)
            assert payment.refund_won == payment.amount_won
            assert all(i.status == db.ItemStatus.CANCELLED for i in order.items)
        else:
            assert order.cancelled_at is None and order.cancel_reason is None
            assert (payment.status, payment.refund_won) == (db.PaymentStatus.PAID, 0)
            assert all(i.status != db.ItemStatus.CANCELLED for i in order.items)
    tracking = session.scalars(select(db.Shipment.tracking_no)).all()
    assert len(set(tracking)) == len(tracking)


def test_requests_match_item_statuses(session):
    requested: dict[tuple[str, int], db.ItemStatus] = {}
    for request in session.scalars(select(db.ServiceRequest)):
        order = session.get(db.Order, request.order_id)
        assert order.status == db.OrderStatus.DELIVERED
        assert order.shipment.delivered_at < request.created_at
        lines = {i.line_no: i for i in order.items}
        line_nos = [ri.line_no for ri in request.items]
        assert line_nos and set(line_nos) <= set(lines)
        fee = 3000 if request.reason == db.RequestReason.CHANGED_MIND else 0
        assert request.return_fee_won == fee
        if request.kind == db.RequestKind.RETURN:
            assert request.id == f"RT-{order.id}-{min(line_nos)}"
            goods = sum(lines[n].unit_price_won * lines[n].quantity for n in line_nos)
            assert request.refund_won == goods - fee
            status = db.ItemStatus.RETURN_REQUESTED
        else:
            assert request.id == f"EX-{order.id}-{line_nos[0]}"
            assert len(line_nos) == 1 and request.refund_won == 0
            status = db.ItemStatus.EXCHANGE_REQUESTED
        for request_item in request.items:
            line = lines[request_item.line_no]
            assert request_item.quantity == line.quantity
            assert (order.id, line.line_no) not in requested
            requested[(order.id, line.line_no)] = status
            if request.kind == db.RequestKind.EXCHANGE:
                old = session.get(db.ProductVariant, line.variant_id)
                new = session.get(db.ProductVariant, request_item.exchange_variant_id)
                assert new.product_id == old.product_id and new.id != old.id
            else:
                assert request_item.exchange_variant_id is None

    special = (db.ItemStatus.RETURN_REQUESTED, db.ItemStatus.EXCHANGE_REQUESTED)
    for item in session.scalars(select(db.OrderItem)):
        key = (item.order_id, item.line_no)
        if key in requested:
            assert item.status == requested[key]
        else:
            assert item.status not in special


def test_coupons_and_tickets(session):
    for coupon in session.scalars(select(db.Coupon)):
        assert coupon.amount_won > 0
        assert coupon.issued_at < coupon.expires_at
        if coupon.used_at is not None:
            assert coupon.issued_at <= coupon.used_at <= coupon.expires_at
        if coupon.kind == db.CouponKind.COMPENSATION:
            order = session.get(db.Order, coupon.order_id)
            assert coupon.id == f"CP-{order.id}-1"
            assert order.customer_id == coupon.customer_id
            assert coupon.reason is not None
            assert coupon.amount_won in (2000, 3000, 5000)
            assert (coupon.expires_at - coupon.issued_at).days == 30
            if coupon.reason == db.CompensationReason.DEFECTIVE_ITEM:
                assert any(r.reason == db.RequestReason.DEFECTIVE for r in order.requests)
            else:
                late = (order.shipment.delivered_at.astimezone(KST).date() - order.shipment.promised_by).days
                assert late >= 1
                assert coupon.amount_won == (2000 if late <= 2 else 5000)
        else:
            assert coupon.id.startswith(f"CP-{coupon.customer_id}-P")
            assert coupon.reason is None and coupon.order_id is None

    discounts = sorted(
        (o.customer_id, o.discount_won, o.ordered_at)
        for o in session.scalars(select(db.Order))
        if o.discount_won
    )
    used = sorted(
        (c.customer_id, c.amount_won, c.used_at) for c in session.scalars(select(db.Coupon)) if c.used_at
    )
    assert discounts == used

    for ticket in session.scalars(select(db.Ticket)):
        assert ticket.body
        if ticket.order_id is not None:
            assert session.get(db.Order, ticket.order_id).customer_id == ticket.customer_id


def test_bulk_datetimes_are_inside_the_window(engine):
    """Every datetime of a generated row (coupon expiry included) is in [2026-06-01, SEED_END]."""
    fixture_ids = set(seed.FIXTURE_CUSTOMER_IDS) | set(seed.FIXTURE_ORDER_IDS)
    start, end = seed.BULK_START.astimezone(UTC), seed.SEED_END.astimezone(UTC)
    checked = 0
    for table, rows in db.dump_db(engine).items():
        columns = db.Base.metadata.tables[table].columns
        time_columns = [c.name for c in columns if isinstance(c.type, db.UTCDateTime)]
        for row in rows:
            if {row.get("id"), row.get("order_id"), row.get("customer_id")} & fixture_ids:
                continue
            for name in time_columns:
                if row[name] is not None:
                    value = datetime.fromisoformat(row[name])
                    assert start <= value <= end, (table, row)
                    checked += 1
    assert checked > 500


def test_fixture_datetimes_are_before_the_smoke_now(session):
    now = kst(9, 14, 10, 0)
    for order_id in seed.FIXTURE_ORDER_IDS:
        order = session.get(db.Order, order_id)
        times = [order.ordered_at, order.cancelled_at, order.payment.paid_at]
        times += [order.shipment.shipped_at, order.shipment.delivered_at]
        assert all(t < now for t in times if t is not None)
    for customer_id in seed.FIXTURE_CUSTOMER_IDS:
        assert session.get(db.Customer, customer_id).joined_at < now


def test_dump_has_no_floats_and_handoffs_are_empty(engine):
    dump = db.dump_db(engine, ignore=frozenset())  # _plain raises TypeError on a float
    assert dump["handoffs"] == []
    for rows in dump.values():
        for row in rows:
            assert all(value is None or isinstance(value, bool | int | str) for value in row.values())


def test_no_real_company_names(engine):
    dump = db.dump_db(engine, ignore=frozenset())
    carriers = {row["carrier"] for row in dump["shipments"]}
    assert carriers <= set(seed.CARRIERS)
    assert {"한빛택배", "새벽택배"} <= set(seed.CARRIERS)
