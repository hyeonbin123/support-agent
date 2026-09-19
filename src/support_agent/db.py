"""Domain tables and the database helpers used by evaluation.

The same models run on in-memory SQLite (one private copy per episode) and later on PostgreSQL. To keep the
two alike: string business keys (no autoincrement), money as integer won, UTC datetimes through UTCDateTime,
enums as VARCHAR + CHECK, no JSON columns, no server defaults.
"""

from __future__ import annotations

import enum
import hashlib
import json
import sqlite3
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Engine,
    Enum,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Text,
    TypeDecorator,
    create_engine,
    event,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.pool import StaticPool


class UTCDateTime(TypeDecorator):
    """Stores UTC, returns aware UTC. Naive datetimes are rejected so SQLite and PostgreSQL behave alike."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime is not allowed")
        return value.astimezone(UTC)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class CustomerGrade(enum.StrEnum):
    NORMAL = "normal"
    VIP = "vip"


class OrderStatus(enum.StrEnum):
    PAID = "paid"
    PREPARING = "preparing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class ItemStatus(enum.StrEnum):
    ORDERED = "ordered"
    CANCELLED = "cancelled"
    RETURN_REQUESTED = "return_requested"
    EXCHANGE_REQUESTED = "exchange_requested"


class PaymentMethod(enum.StrEnum):
    CARD = "card"
    BANK_TRANSFER = "bank_transfer"
    SIMPLE_PAY = "simple_pay"


class PaymentStatus(enum.StrEnum):
    PAID = "paid"
    REFUND_PENDING = "refund_pending"
    REFUNDED = "refunded"


class ShipmentStatus(enum.StrEnum):
    READY = "ready"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"


class CancelReason(enum.StrEnum):
    CHANGED_MIND = "changed_mind"
    ORDERED_BY_MISTAKE = "ordered_by_mistake"
    DELIVERY_TOO_SLOW = "delivery_too_slow"


class RequestKind(enum.StrEnum):
    RETURN = "return"
    EXCHANGE = "exchange"


class RequestReason(enum.StrEnum):
    CHANGED_MIND = "changed_mind"
    DEFECTIVE = "defective"
    WRONG_ITEM = "wrong_item"


class CouponKind(enum.StrEnum):
    PROMO = "promo"
    COMPENSATION = "compensation"


class CompensationReason(enum.StrEnum):
    DELIVERY_DELAY = "delivery_delay"
    DEFECTIVE_ITEM = "defective_item"


class TicketCategory(enum.StrEnum):
    DELIVERY = "delivery"
    REFUND = "refund"
    PRODUCT = "product"
    ACCOUNT = "account"
    OTHER = "other"


class HandoffReason(enum.StrEnum):
    CUSTOMER_REQUEST = "customer_request"
    OUT_OF_SCOPE = "out_of_scope"
    CANNOT_VERIFY = "cannot_verify"


def enum_type(enum_cls: type[enum.Enum], name: str) -> Enum:
    """VARCHAR + CHECK on both databases, storing the value (not the member name)."""
    return Enum(
        enum_cls,
        native_enum=False,
        create_constraint=True,
        validate_strings=True,
        length=30,
        values_callable=lambda cls: [member.value for member in cls],
        name=name,
    )


# Constraint names are fixed so that Alembic migrations can be added later without touching the models.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Domain tables only. Service tables (chat sessions, audit log) will use their own MetaData."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Customer(Base):
    __tablename__ = "customers"
    id: Mapped[str] = mapped_column(String(20), primary_key=True)  # C-0001
    name: Mapped[str] = mapped_column(String(40))
    phone: Mapped[str] = mapped_column(String(20))  # digits only: 01000001234
    email: Mapped[str] = mapped_column(String(80))  # lower case
    grade: Mapped[CustomerGrade] = mapped_column(enum_type(CustomerGrade, "customer_grade"))
    joined_at: Mapped[datetime] = mapped_column(UTCDateTime)

    addresses: Mapped[list[CustomerAddress]] = relationship(order_by="CustomerAddress.id")


class CustomerAddress(Base):
    __tablename__ = "customer_addresses"
    id: Mapped[str] = mapped_column(String(30), primary_key=True)  # AD-C-0001-1
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"))
    label: Mapped[str] = mapped_column(String(20))  # 집, 회사, ...
    recipient: Mapped[str] = mapped_column(String(40))
    postal_code: Mapped[str] = mapped_column(String(10))
    address: Mapped[str] = mapped_column(String(200))
    is_default: Mapped[bool] = mapped_column(Boolean)


class Product(Base):
    __tablename__ = "products"
    id: Mapped[str] = mapped_column(String(20), primary_key=True)  # P-0001
    name: Mapped[str] = mapped_column(String(80))
    category: Mapped[str] = mapped_column(String(40))

    variants: Mapped[list[ProductVariant]] = relationship(order_by="ProductVariant.id")


class ProductVariant(Base):
    __tablename__ = "product_variants"
    id: Mapped[str] = mapped_column(String(20), primary_key=True)  # V-0001-01
    product_id: Mapped[str] = mapped_column(ForeignKey("products.id"))
    option_label: Mapped[str] = mapped_column(String(60))  # 블랙 / 270mm
    price_won: Mapped[int] = mapped_column(Integer)
    stock: Mapped[int] = mapped_column(Integer)


class Order(Base):
    __tablename__ = "orders"
    id: Mapped[str] = mapped_column(String(20), primary_key=True)  # O-10001
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"))
    status: Mapped[OrderStatus] = mapped_column(enum_type(OrderStatus, "order_status"))
    ordered_at: Mapped[datetime] = mapped_column(UTCDateTime)
    items_won: Mapped[int] = mapped_column(Integer)
    shipping_fee_won: Mapped[int] = mapped_column(Integer)
    discount_won: Mapped[int] = mapped_column(Integer)
    total_won: Mapped[int] = mapped_column(Integer)  # items + shipping fee - discount
    ship_address_id: Mapped[str] = mapped_column(ForeignKey("customer_addresses.id"))
    ship_recipient: Mapped[str] = mapped_column(String(40))  # snapshot of the address at order time
    ship_postal_code: Mapped[str] = mapped_column(String(10))
    ship_address: Mapped[str] = mapped_column(String(200))
    cancelled_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    cancel_reason: Mapped[CancelReason | None] = mapped_column(enum_type(CancelReason, "cancel_reason"))

    items: Mapped[list[OrderItem]] = relationship(order_by="OrderItem.line_no")
    payment: Mapped[Payment] = relationship(uselist=False)
    shipment: Mapped[Shipment] = relationship(uselist=False)
    requests: Mapped[list[ServiceRequest]] = relationship(order_by="ServiceRequest.id")


class OrderItem(Base):
    __tablename__ = "order_items"
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), primary_key=True)
    line_no: Mapped[int] = mapped_column(Integer, primary_key=True)
    variant_id: Mapped[str] = mapped_column(ForeignKey("product_variants.id"))
    product_name: Mapped[str] = mapped_column(String(80))  # snapshot
    option_label: Mapped[str] = mapped_column(String(60))  # snapshot
    quantity: Mapped[int] = mapped_column(Integer)
    unit_price_won: Mapped[int] = mapped_column(Integer)
    status: Mapped[ItemStatus] = mapped_column(enum_type(ItemStatus, "item_status"))


class Payment(Base):
    __tablename__ = "payments"
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), primary_key=True)
    method: Mapped[PaymentMethod] = mapped_column(enum_type(PaymentMethod, "payment_method"))
    amount_won: Mapped[int] = mapped_column(Integer)
    status: Mapped[PaymentStatus] = mapped_column(enum_type(PaymentStatus, "payment_status"))
    refund_won: Mapped[int] = mapped_column(Integer)  # 0 until a refund is decided
    paid_at: Mapped[datetime] = mapped_column(UTCDateTime)


class Shipment(Base):
    __tablename__ = "shipments"
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), primary_key=True)
    carrier: Mapped[str] = mapped_column(String(40))
    tracking_no: Mapped[str] = mapped_column(String(30))
    status: Mapped[ShipmentStatus] = mapped_column(enum_type(ShipmentStatus, "shipment_status"))
    shipped_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    delivered_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    promised_by: Mapped[date] = mapped_column(Date)  # promised arrival date (KST calendar date)


class ServiceRequest(Base):
    """A return or an exchange. The id is built from its content, so it does not depend on call order:
    RT-{order_id}-{lowest line_no} for a return, EX-{order_id}-{line_no} for an exchange."""

    __tablename__ = "service_requests"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"))
    kind: Mapped[RequestKind] = mapped_column(enum_type(RequestKind, "request_kind"))
    reason: Mapped[RequestReason] = mapped_column(enum_type(RequestReason, "request_reason"))
    refund_won: Mapped[int] = mapped_column(Integer)  # 0 for an exchange
    return_fee_won: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)

    items: Mapped[list[ServiceRequestItem]] = relationship(order_by="ServiceRequestItem.line_no")


class ServiceRequestItem(Base):
    __tablename__ = "service_request_items"
    request_id: Mapped[str] = mapped_column(ForeignKey("service_requests.id"), primary_key=True)
    line_no: Mapped[int] = mapped_column(Integer, primary_key=True)
    quantity: Mapped[int] = mapped_column(Integer)
    exchange_variant_id: Mapped[str | None] = mapped_column(ForeignKey("product_variants.id"))


class Coupon(Base):
    __tablename__ = "coupons"
    id: Mapped[str] = mapped_column(
        String(40), primary_key=True
    )  # CP-{order_id}-{n} or CP-{customer_id}-P{n}
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"))
    kind: Mapped[CouponKind] = mapped_column(enum_type(CouponKind, "coupon_kind"))
    amount_won: Mapped[int] = mapped_column(Integer)
    reason: Mapped[CompensationReason | None] = mapped_column(
        enum_type(CompensationReason, "compensation_reason")
    )
    order_id: Mapped[str | None] = mapped_column(ForeignKey("orders.id"))
    issued_at: Mapped[datetime] = mapped_column(UTCDateTime)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime)
    used_at: Mapped[datetime | None] = mapped_column(UTCDateTime)


class Ticket(Base):
    __tablename__ = "tickets"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)  # TK-{customer_id}-{n}
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"))
    order_id: Mapped[str | None] = mapped_column(ForeignKey("orders.id"))
    category: Mapped[TicketCategory] = mapped_column(enum_type(TicketCategory, "ticket_category"))
    body: Mapped[str] = mapped_column(Text)  # free text written by the model: not compared
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)


class Handoff(Base):
    """A hand-off to a human agent. It is a row so that a needless hand-off shows up as a DB difference."""

    __tablename__ = "handoffs"
    id: Mapped[str] = mapped_column(String(20), primary_key=True)  # HO-{n}
    customer_id: Mapped[str | None] = mapped_column(ForeignKey("customers.id"))  # None before verification
    reason: Mapped[HandoffReason] = mapped_column(enum_type(HandoffReason, "handoff_reason"))
    summary: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)


# Columns left out of the DB comparison: free text, and the hand-off reason (two reasons can both be fair).
IGNORED_COLUMNS: frozenset[tuple[str, str]] = frozenset(
    {("tickets", "body"), ("handoffs", "summary"), ("handoffs", "reason")}
)


def _fk_on(dbapi_connection, _record) -> None:  # Python 3.11: sqlite3 connections have no .autocommit
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def memory_engine(source: Engine | None = None) -> Engine:
    """A private in-memory SQLite engine; with `source`, a byte copy of that database (under 1 ms)."""
    raw = sqlite3.connect(":memory:", check_same_thread=False)
    if source is not None:
        with source.connect() as conn:
            conn.connection.dbapi_connection.backup(raw)
    engine = create_engine("sqlite://", creator=lambda: raw, poolclass=StaticPool)
    event.listen(engine, "connect", _fk_on)  # this engine only, never a PostgreSQL one
    return engine


def _plain(value: Any) -> Any:
    if isinstance(value, enum.Enum):  # before str: every enum here is a StrEnum
        return value.value
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat(timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unexpected column value {value!r}: money is integer won, there are no floats")


Dump = dict[str, list[dict[str, Any]]]


def dump_db(engine: Engine, ignore: frozenset[tuple[str, str]] = IGNORED_COLUMNS) -> Dump:
    """Every domain table as plain rows, sorted by primary key in Python (database collations differ)."""
    out: Dump = {}
    with engine.connect() as conn:
        for table in Base.metadata.sorted_tables:
            cols = [c for c in table.columns if (table.name, c.name) not in ignore]
            pk = [c.name for c in table.primary_key.columns]
            rows = [
                {c.name: _plain(row[c.name]) for c in cols} for row in conn.execute(select(*cols)).mappings()
            ]
            out[table.name] = sorted(rows, key=lambda r: tuple(r[name] for name in pk))
    return out


def state_hash(dump: Dump) -> str:
    return hashlib.sha256(json.dumps(dump, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def diff_dumps(expected: Dump, actual: Dump) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Rows that are only in one of the two dumps, per table. Empty when the dumps are equal."""
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name in expected:
        a = {json.dumps(r, sort_keys=True, ensure_ascii=False) for r in expected[name]}
        b = {json.dumps(r, sort_keys=True, ensure_ascii=False) for r in actual.get(name, [])}
        if a != b:
            result[name] = {
                "only_expected": [json.loads(x) for x in sorted(a - b)],
                "only_actual": [json.loads(x) for x in sorted(b - a)],
            }
    return result
