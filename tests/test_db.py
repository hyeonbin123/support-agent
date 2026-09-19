"""db.py helpers: the UTC datetime type, constraints, engine copies, dumps, diffs, hashes."""

from __future__ import annotations

import copy
from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlalchemy import insert, text, update
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session

from support_agent import db
from support_agent.clock import KST


def make_customer(customer_id: str, joined_at: datetime) -> db.Customer:
    return db.Customer(
        id=customer_id,
        name="홍길동",
        phone="01000000002",
        email="gildong@example.com",
        grade=db.CustomerGrade.NORMAL,
        joined_at=joined_at,
    )


def test_utc_datetime_rejects_naive(tiny_engine):
    with Session(tiny_engine) as session:
        session.add(make_customer("C-2", datetime(2026, 9, 1, 12, 0)))
        with pytest.raises(StatementError, match="naive datetime"):
            session.flush()


def test_utc_datetime_round_trips_as_utc(tiny_engine):
    joined_at = datetime(2026, 9, 1, 8, 30, tzinfo=KST)
    other_zone = datetime(2026, 9, 1, 8, 30, tzinfo=timezone(timedelta(hours=-5)))
    with Session(tiny_engine) as session:
        session.add_all([make_customer("C-2", joined_at), make_customer("C-3", other_zone)])
        session.commit()
    with Session(tiny_engine) as session:
        loaded = session.get(db.Customer, "C-2").joined_at
        assert loaded == joined_at
        assert loaded.utcoffset() == timedelta(0)
        assert (loaded.day, loaded.hour, loaded.minute) == (31, 23, 30)
        assert session.get(db.Customer, "C-3").joined_at == datetime(2026, 9, 1, 13, 30, tzinfo=UTC)
        assert session.get(db.Order, "O-1").cancelled_at is None
    with tiny_engine.connect() as conn:
        stored = conn.execute(text("SELECT joined_at FROM customers WHERE id = 'C-2'")).scalar_one()
    assert stored.startswith("2026-08-31 23:30:00")


def test_enum_rejects_invalid_string(tiny_engine):
    with Session(tiny_engine) as session:
        with pytest.raises(StatementError):  # validate_strings: caught before it reaches the database
            session.execute(update(db.Order).where(db.Order.id == "O-1").values(status="lost"))
    with tiny_engine.connect() as conn:
        with pytest.raises(IntegrityError):  # the CHECK constraint
            conn.execute(text("UPDATE orders SET status = 'lost' WHERE id = 'O-1'"))
        conn.rollback()
        conn.execute(text("UPDATE orders SET status = 'preparing' WHERE id = 'O-1'"))
        conn.rollback()


def test_enum_stores_the_value(tiny_engine):
    with tiny_engine.connect() as conn:
        stored = conn.execute(text("SELECT status FROM orders WHERE id = 'O-2'")).scalar_one()
    assert stored == "shipped"


def test_foreign_keys_are_enforced(tiny_engine):
    with tiny_engine.connect() as conn:
        assert conn.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
    address = {
        "id": "AD-C-404-1",
        "customer_id": "C-404",
        "label": "집",
        "recipient": "홍길동",
        "postal_code": "00000",
        "address": "가온시 한빛구 가상로 1",
        "is_default": True,
    }
    for engine in (tiny_engine, db.memory_engine(tiny_engine)):
        with Session(engine) as session:
            with pytest.raises(IntegrityError):
                session.execute(insert(db.CustomerAddress).values(**address))
            session.rollback()
            with pytest.raises(IntegrityError):
                session.execute(text("DELETE FROM customers WHERE id = 'C-1'"))


def test_memory_engine_copy_is_independent(tiny_engine):
    before = db.dump_db(tiny_engine)
    first = db.memory_engine(tiny_engine)
    second = db.memory_engine(tiny_engine)
    assert db.dump_db(first) == before

    with Session(first) as session:
        session.get(db.Order, "O-1").status = db.OrderStatus.CANCELLED
        session.commit()

    assert db.dump_db(tiny_engine) == before
    assert db.dump_db(second) == before
    changed = db.diff_dumps(before, db.dump_db(first))
    assert list(changed) == ["orders"]


def test_memory_engine_without_source_is_empty():
    engine = db.memory_engine()
    db.Base.metadata.create_all(engine)
    dump = db.dump_db(engine)
    assert set(dump) == set(db.Base.metadata.tables)
    assert len(dump) == 13
    assert all(rows == [] for rows in dump.values())


def test_dump_sorts_by_primary_key_and_uses_plain_values(tiny_engine):
    at = datetime(2026, 9, 10, 9, 0, tzinfo=KST)
    with Session(tiny_engine) as session:
        for customer_id in ("C-9", "C-0", "C-5"):
            session.add(make_customer(customer_id, at))
        session.add(
            db.OrderItem(
                order_id="O-1",
                line_no=2,
                variant_id="V-1-01",
                product_name="무선 이어폰",
                option_label="블랙",
                quantity=1,
                unit_price_won=38900,
                status=db.ItemStatus.ORDERED,
            )
        )
        session.commit()
    dump = db.dump_db(tiny_engine)
    assert [row["id"] for row in dump["customers"]] == ["C-0", "C-1", "C-5", "C-9"]
    assert [(r["order_id"], r["line_no"]) for r in dump["order_items"]] == [
        ("O-1", 1),
        ("O-1", 2),
        ("O-2", 1),
    ]

    customer = dump["customers"][0]
    assert customer["joined_at"] == "2026-09-10T00:00:00+00:00"
    assert customer["grade"] == "normal"
    assert dump["customer_addresses"][0]["is_default"] is True
    assert dump["shipments"][0]["promised_by"] == "2026-09-13"
    assert dump["orders"][0]["cancelled_at"] is None


def test_dump_ignores_free_text_columns(tiny_engine):
    at = datetime(2026, 9, 12, 9, 0, tzinfo=KST)

    def with_texts(body: str, summary: str, reason: db.HandoffReason) -> db.Dump:
        engine = db.memory_engine(tiny_engine)
        with Session(engine) as session:
            session.add(
                db.Ticket(
                    id="TK-C-1-1",
                    customer_id="C-1",
                    order_id=None,
                    category=db.TicketCategory.OTHER,
                    body=body,
                    created_at=at,
                )
            )
            session.add(
                db.Handoff(id="HO-1", customer_id="C-1", reason=reason, summary=summary, created_at=at)
            )
            session.commit()
        return db.dump_db(engine)

    one = with_texts("영수증 문의", "고객 요청", db.HandoffReason.CUSTOMER_REQUEST)
    two = with_texts("다른 본문", "다른 요약", db.HandoffReason.OUT_OF_SCOPE)
    assert set(one["tickets"][0]) == {"id", "customer_id", "order_id", "category", "created_at"}
    assert set(one["handoffs"][0]) == {"id", "customer_id", "created_at"}
    assert one == two
    assert db.diff_dumps(one, two) == {}
    assert db.state_hash(one) == db.state_hash(two)

    assert db.IGNORED_COLUMNS == {("tickets", "body"), ("handoffs", "summary"), ("handoffs", "reason")}


def test_dump_rejects_floats(tiny_engine):
    engine = db.memory_engine(tiny_engine)
    with engine.connect() as conn:
        conn.execute(text("UPDATE orders SET total_won = 38900.5 WHERE id = 'O-1'"))
        conn.commit()
    with pytest.raises(TypeError, match="no floats"):
        db.dump_db(engine)


def test_diff_dumps(tiny_engine):
    expected = db.dump_db(tiny_engine)
    assert db.diff_dumps(expected, copy.deepcopy(expected)) == {}

    engine = db.memory_engine(tiny_engine)
    with Session(engine) as session:
        session.get(db.Order, "O-1").status = db.OrderStatus.CANCELLED
        session.add(
            db.Coupon(
                id="CP-O-2-1",
                customer_id="C-1",
                kind=db.CouponKind.COMPENSATION,
                amount_won=2000,
                reason=db.CompensationReason.DELIVERY_DELAY,
                order_id="O-2",
                issued_at=datetime(2026, 9, 14, 10, 0, tzinfo=KST),
                expires_at=datetime(2026, 10, 14, 10, 0, tzinfo=KST),
                used_at=None,
            )
        )
        session.commit()
    actual = db.dump_db(engine)

    diff = db.diff_dumps(expected, actual)
    assert set(diff) == {"orders", "coupons"}
    assert [r["status"] for r in diff["orders"]["only_expected"]] == ["paid"]
    assert [r["status"] for r in diff["orders"]["only_actual"]] == ["cancelled"]
    assert diff["orders"]["only_expected"][0]["id"] == diff["orders"]["only_actual"][0]["id"] == "O-1"
    assert diff["coupons"]["only_expected"] == []
    assert [r["id"] for r in diff["coupons"]["only_actual"]] == ["CP-O-2-1"]

    reverse = db.diff_dumps(actual, expected)
    assert [r["id"] for r in reverse["coupons"]["only_expected"]] == ["CP-O-2-1"]
    assert reverse["coupons"]["only_actual"] == []


def test_state_hash_is_stable(tiny_engine):
    dump = db.dump_db(tiny_engine)
    digest = db.state_hash(dump)
    assert len(digest) == 64 and int(digest, 16) >= 0
    assert digest == db.state_hash(db.dump_db(tiny_engine))
    assert digest == db.state_hash(db.dump_db(db.memory_engine(tiny_engine)))
    assert digest == db.state_hash(
        dict(reversed(list(copy.deepcopy(dump).items())))
    )  # key order is irrelevant

    changed = copy.deepcopy(dump)
    changed["orders"][0]["total_won"] += 1
    assert db.state_hash(changed) != digest
