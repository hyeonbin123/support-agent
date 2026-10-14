"""도메인 테이블과 평가가 쓰는 데이터베이스 도우미.

같은 모델이 메모리 SQLite(에피소드마다 개인 복사본 하나)와, 나중에 붙인 서비스의 PostgreSQL에서 돈다. 두
DB를 같게 유지하려고 다음을 지킨다: 문자열 업무 키(자동 증가 없음), 원 단위 정수 금액, UTCDateTime을 거친
UTC 시각, VARCHAR + CHECK로 저장하는 enum, JSON 컬럼 없음, 서버 기본값 없음.
"""

# [파일 역할] 가상 쇼핑몰의 도메인 테이블 13개(고객·주소·상품·옵션·주문·주문 줄·결제·배송·반품/교환 요청과 그
# 줄·쿠폰·티켓·이관)와, 평가가 쓰는 DB 도우미(메모리 복사, 덤프, 해시, 차이)를 정의한다.
# [설계 이유] 같은 모델이 평가에서는 에피소드마다 메모리 SQLite 복사본 위에서, 서비스에서는 PostgreSQL 위에서
# 돈다(README 구조 절). 두 DB가 같게 동작하도록 docstring의 규칙(문자열 업무 키, 원 단위 정수 금액, UTC 시각,
# VARCHAR+CHECK enum, JSON 컬럼·서버 기본값 없음)을 지킨다. 자동 증가 id를 쓰지 않는 것은 새 행의 id를
# 내용에서 만들어, 호출 순서가 달라도 최종 DB가 같게 나오게 하려는 것이다(docs/design.md 결정성).
# [판정과의 관계] 판정의 핵심은 '에피소드가 끝난 DB 덤프 == 정답 동작만 실행한 덤프'다. dump_db가 그 비교
# 대상을 만들고, IGNORED_COLUMNS가 비교에서 뺄 자유 입력 컬럼을 정한다.
# [연결] seed.py가 이 테이블에 데이터를 채우고, tools.py가 읽고 쓰며, judge.py·episode.py·run.py가
# 덤프·해시·복사를 쓴다. 서비스의 Alembic 마이그레이션과 service/store.py도 이 정의와 NAMING_CONVENTION을
# 쓴다. 다른 모듈이 모두 기대는 계약 파일이다.

from __future__ import annotations

# hashlib·json: 덤프를 정렬된 JSON으로 만들어 sha256 해시를 낸다(state_hash).
# sqlite3: 표준 sqlite3의 backup()으로 메모리 DB를 통째로 복사한다(memory_engine).
import enum
import hashlib
import json
import sqlite3
from datetime import UTC, date, datetime
from typing import Any

# SQLAlchemy 2의 선언형 ORM(Mapped, mapped_column)으로 테이블을 정의한다. 같은 모델을 SQLite와 PostgreSQL에서
# 함께 쓰고, 서비스는 Alembic 마이그레이션까지 이어 쓰기 위해서다(README 스택).
# StaticPool: 메모리 SQLite는 연결마다 새 빈 DB가 되므로, 연결 하나를 엔진 전체가 나눠 쓰게 한다.
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


# 시각 컬럼 타입. 저장할 때는 UTC로 바꾸고, 읽을 때는 항상 UTC 시간대가 붙은 값으로 돌려준다.
# 왜 필요한가: SQLite는 시간대 정보를 보존하지 않아 시간대 없는 값을 돌려주고 PostgreSQL은 시간대가 붙은 값을
# 돌려준다. 이 차이를 여기서 맞춰 도구 코드가 어느 DB에서든 같은 비교·계산을 하게 한다. 시간대 없는 입력은
# UTC인지 KST인지 알 수 없으니 저장 전에 막는다.
class UTCDateTime(TypeDecorator):
    """UTC로 저장하고 시간대가 붙은 UTC로 돌려준다. SQLite와 PostgreSQL이 같게 동작하도록 시간대 없는
    datetime은 거절한다."""

    impl = DateTime(timezone=True)
    # 인자 없는 타입이라 SQLAlchemy의 구문 캐시에 넣어도 안전하다는 표시다(없으면 경고가 난다).
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
        # SQLite에서 온 시간대 없는 값은 UTC로 저장한 것이므로 UTC를 붙이고, PostgreSQL 값은 UTC로 맞춘다.
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


# 아래 enum들은 DB와 도구 인자에 쓰는 ASCII 코드 집합이다. StrEnum이라 문자열과 바로 비교되고 JSON으로도 값
# 그대로 나간다. 한국어 표시 이름은 labels.py에 따로 있다.
class CustomerGrade(enum.StrEnum):
    NORMAL = "normal"
    VIP = "vip"


# 주문 상태. 취소·배송지 변경은 paid·preparing에서만, 반품·교환은 delivered에서만 된다(docs/design.md 규정
# 검사 표).
class OrderStatus(enum.StrEnum):
    PAID = "paid"
    PREPARING = "preparing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


# 주문 줄 상태. 반품·교환을 접수하면 그 줄이 return_requested·exchange_requested가 되어, 같은 줄의 중복
# 접수(already_requested)를 막는 근거가 된다.
class ItemStatus(enum.StrEnum):
    ORDERED = "ordered"
    CANCELLED = "cancelled"
    RETURN_REQUESTED = "return_requested"
    EXCHANGE_REQUESTED = "exchange_requested"


class PaymentMethod(enum.StrEnum):
    CARD = "card"
    BANK_TRANSFER = "bank_transfer"
    SIMPLE_PAY = "simple_pay"


# 결제 상태. 취소하면 refund_pending이 된다. 반품은 결제를 건드리지 않는다(회수 뒤 환불, docs/design.md 금액
# 규칙).
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


# 보상 쿠폰 사유. 금액과 자격은 rules.py와 tools.py가 사유별로 정한다.
class CompensationReason(enum.StrEnum):
    DELIVERY_DELAY = "delivery_delay"
    DEFECTIVE_ITEM = "defective_item"


class TicketCategory(enum.StrEnum):
    DELIVERY = "delivery"
    REFUND = "refund"
    PRODUCT = "product"
    ACCOUNT = "account"
    OTHER = "other"


# 상담원 이관 사유. 두 사유가 모두 타당할 수 있어 판정의 DB 비교에서는 뺀다(IGNORED_COLUMNS).
class HandoffReason(enum.StrEnum):
    CUSTOMER_REQUEST = "customer_request"
    OUT_OF_SCOPE = "out_of_scope"
    CANNOT_VERIFY = "cannot_verify"


# enum 컬럼 타입을 만든다. PostgreSQL의 고유 ENUM 타입 대신 VARCHAR + CHECK 제약을 쓰므로(native_enum=False,
# create_constraint=True) 두 DB의 스키마가 같아진다.
# values_callable은 멤버 이름(CHANGED_MIND)이 아니라 값(changed_mind)을 저장하게 하고, validate_strings는
# 목록에 없는 문자열을 넣으려 하면 바로 오류를 낸다. name은 CHECK 제약 이름의 재료다.
def enum_type(enum_cls: type[enum.Enum], name: str) -> Enum:
    """두 DB 모두에서 VARCHAR + CHECK이고, 멤버 이름이 아니라 값을 저장한다."""
    return Enum(
        enum_cls,
        native_enum=False,
        create_constraint=True,
        validate_strings=True,
        length=30,
        values_callable=lambda cls: [member.value for member in cls],
        name=name,
    )


# 제약 이름 규칙. 이름을 고정해 두면 Alembic 마이그레이션이 모델과 같은 이름을 쓴다(서비스 마이그레이션 0001의
# op.f("pk_...") 등이 이 규칙으로 정해진 이름이다). 제약 이름을 고정해 두었기 때문에 모델을 건드리지 않고도
# 나중에 Alembic 마이그레이션을 더할 수 있다.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


# 도메인 테이블만의 선언 기반 클래스. 서비스 테이블(세션, 감사 로그 등)은 service/store.py에서 다른 MetaData를
# 쓰므로, dump_db가 Base.metadata만 돌면 판정 대상 상태에 서비스 기록이 섞이지 않는다.
class Base(DeclarativeBase):
    """도메인 테이블 전용. 서비스 테이블(채팅 세션, 감사 로그)은 자기 MetaData를 쓴다."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


# 고객. phone은 숫자만, email은 소문자로 저장해 find_customer가 정규화한 입력과 그대로 비교할 수 있게 한다.
# 등급(grade)은 출력용 정보다.
class Customer(Base):
    __tablename__ = "customers"
    id: Mapped[str] = mapped_column(String(20), primary_key=True)  # C-0001
    name: Mapped[str] = mapped_column(String(40))
    phone: Mapped[str] = mapped_column(String(20))  # 숫자만: 01000001234
    email: Mapped[str] = mapped_column(String(80))  # 소문자
    grade: Mapped[CustomerGrade] = mapped_column(enum_type(CustomerGrade, "customer_grade"))
    joined_at: Mapped[datetime] = mapped_column(UTCDateTime)

    addresses: Mapped[list[CustomerAddress]] = relationship(order_by="CustomerAddress.id")


# 고객의 배송지 목록. 배송지 변경(change_shipping_address)은 고객 소유의 주소 id로만 할 수
# 있다(address_not_owned 검사).
class CustomerAddress(Base):
    __tablename__ = "customer_addresses"
    id: Mapped[str] = mapped_column(String(30), primary_key=True)  # AD-C-0001-1
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"))
    label: Mapped[str] = mapped_column(String(20))  # 집, 회사, ...
    recipient: Mapped[str] = mapped_column(String(40))
    postal_code: Mapped[str] = mapped_column(String(10))
    address: Mapped[str] = mapped_column(String(200))
    is_default: Mapped[bool] = mapped_column(Boolean)


# 상품과 옵션(ProductVariant). 교환은 같은 상품의 다른 옵션으로만 되고, 새 옵션의 재고(stock)를 수량만큼
# 줄인다(docs/design.md 금액 규칙).
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


# 주문. 금액은 모두 원 단위 정수다(total = 상품 합계 + 배송비 − 할인). 배송지는 주문 시점의 주소를 복사해
# 두므로(ship_* 컬럼), 고객 주소록이 바뀌어도 이 주문의 배송지는 그대로이고 배송지 변경 도구가 이 컬럼들을
# 바꾼다.
# relationship들은 도구가 주문 하나로 줄·결제·배송·요청을 한 번에 따라가게 한다. order_by를 두어 출력 순서가
# 항상 같다.
class Order(Base):
    __tablename__ = "orders"
    id: Mapped[str] = mapped_column(String(20), primary_key=True)  # O-10001
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"))
    status: Mapped[OrderStatus] = mapped_column(enum_type(OrderStatus, "order_status"))
    ordered_at: Mapped[datetime] = mapped_column(UTCDateTime)
    items_won: Mapped[int] = mapped_column(Integer)
    shipping_fee_won: Mapped[int] = mapped_column(Integer)
    discount_won: Mapped[int] = mapped_column(Integer)
    total_won: Mapped[int] = mapped_column(Integer)  # 상품 합계 + 배송비 - 할인
    ship_address_id: Mapped[str] = mapped_column(ForeignKey("customer_addresses.id"))
    ship_recipient: Mapped[str] = mapped_column(String(40))  # 주문 시점 주소를 복사해 둔 값
    ship_postal_code: Mapped[str] = mapped_column(String(10))
    ship_address: Mapped[str] = mapped_column(String(200))
    cancelled_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    cancel_reason: Mapped[CancelReason | None] = mapped_column(enum_type(CancelReason, "cancel_reason"))

    items: Mapped[list[OrderItem]] = relationship(order_by="OrderItem.line_no")
    payment: Mapped[Payment] = relationship(uselist=False)
    shipment: Mapped[Shipment] = relationship(uselist=False)
    requests: Mapped[list[ServiceRequest]] = relationship(order_by="ServiceRequest.id")


# 주문의 한 줄. (order_id, line_no)가 기본 키다. 상품명·옵션명은 주문 시점 값을 복사해 둔 것(snapshot)이다.
# 반품 환불액은 이 줄의 단가 × 수량으로 계산한다.
class OrderItem(Base):
    __tablename__ = "order_items"
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), primary_key=True)
    line_no: Mapped[int] = mapped_column(Integer, primary_key=True)
    variant_id: Mapped[str] = mapped_column(ForeignKey("product_variants.id"))
    product_name: Mapped[str] = mapped_column(String(80))  # 주문 시점 값 복사
    option_label: Mapped[str] = mapped_column(String(60))  # 주문 시점 값 복사
    quantity: Mapped[int] = mapped_column(Integer)
    unit_price_won: Mapped[int] = mapped_column(Integer)
    status: Mapped[ItemStatus] = mapped_column(enum_type(ItemStatus, "item_status"))


# 주문당 결제 하나. refund_won은 환불이 정해지기 전에는 0이고, 취소하면 결제 금액 전액이 된다.
class Payment(Base):
    __tablename__ = "payments"
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), primary_key=True)
    method: Mapped[PaymentMethod] = mapped_column(enum_type(PaymentMethod, "payment_method"))
    amount_won: Mapped[int] = mapped_column(Integer)
    status: Mapped[PaymentStatus] = mapped_column(enum_type(PaymentStatus, "payment_status"))
    refund_won: Mapped[int] = mapped_column(Integer)  # 환불이 정해지기 전에는 0
    paid_at: Mapped[datetime] = mapped_column(UTCDateTime)


# 주문당 배송 하나. promised_by(도착 예정일)는 시각이 아니라 KST 날짜라 Date 타입이다. 지연 보상 쿠폰의 늦은
# 일수를 이 날짜와 수령일로 센다.
class Shipment(Base):
    __tablename__ = "shipments"
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"), primary_key=True)
    carrier: Mapped[str] = mapped_column(String(40))
    tracking_no: Mapped[str] = mapped_column(String(30))
    status: Mapped[ShipmentStatus] = mapped_column(enum_type(ShipmentStatus, "shipment_status"))
    shipped_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    delivered_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    promised_by: Mapped[date] = mapped_column(Date)  # 도착 예정일(KST 달력 날짜)


class ServiceRequest(Base):
    """반품 또는 교환 요청. id를 내용에서 만들어 호출 순서에 기대지 않는다: 반품은
    RT-{order_id}-{가장 작은 line_no}, 교환은 EX-{order_id}-{line_no}."""

    __tablename__ = "service_requests"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    order_id: Mapped[str] = mapped_column(ForeignKey("orders.id"))
    kind: Mapped[RequestKind] = mapped_column(enum_type(RequestKind, "request_kind"))
    reason: Mapped[RequestReason] = mapped_column(enum_type(RequestReason, "request_reason"))
    refund_won: Mapped[int] = mapped_column(Integer)  # 교환이면 0
    return_fee_won: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)

    items: Mapped[list[ServiceRequestItem]] = relationship(order_by="ServiceRequestItem.line_no")


# 반품·교환 요청에 포함된 주문 줄. 교환이면 바꿀 옵션(exchange_variant_id)이 들어가고, 반품이면 None이다.
class ServiceRequestItem(Base):
    __tablename__ = "service_request_items"
    request_id: Mapped[str] = mapped_column(ForeignKey("service_requests.id"), primary_key=True)
    line_no: Mapped[int] = mapped_column(Integer, primary_key=True)
    quantity: Mapped[int] = mapped_column(Integer)
    exchange_variant_id: Mapped[str | None] = mapped_column(ForeignKey("product_variants.id"))


# 쿠폰. 보상 쿠폰은 CP-{주문}-{n}, seed의 프로모션 쿠폰은 CP-{고객}-P{n}으로 id를 만든다. 보상 쿠폰의 고객당
# 30일 한도는 issued_at을, 유효 기간은 expires_at을 본다. expires_at은 미래 시각이라 과제 검증의 'now는 seed의
# 모든 시각보다 뒤' 검사에서 예외다(judge.FUTURE_COLUMNS).
class Coupon(Base):
    __tablename__ = "coupons"
    id: Mapped[str] = mapped_column(
        String(40), primary_key=True
    )  # CP-{order_id}-{n} 또는 CP-{customer_id}-P{n}
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


# 상담 티켓. body는 모델이 쓰는 자유 글이라 정답과 같을 수 없으므로 DB 비교에서 뺀다.
class Ticket(Base):
    __tablename__ = "tickets"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)  # TK-{customer_id}-{n}
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.id"))
    order_id: Mapped[str | None] = mapped_column(ForeignKey("orders.id"))
    category: Mapped[TicketCategory] = mapped_column(enum_type(TicketCategory, "ticket_category"))
    body: Mapped[str] = mapped_column(Text)  # 모델이 쓴 자유 글이라 비교하지 않는다
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)


class Handoff(Base):
    """사람 상담원에게 넘기기(이관). 행으로 남겨, 필요 없는 이관이 DB 차이로 드러나게 한다."""

    __tablename__ = "handoffs"
    id: Mapped[str] = mapped_column(String(20), primary_key=True)  # HO-{n}
    customer_id: Mapped[str | None] = mapped_column(ForeignKey("customers.id"))  # 본인 확인 전이면 None
    reason: Mapped[HandoffReason] = mapped_column(enum_type(HandoffReason, "handoff_reason"))
    summary: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)


# DB 비교에서 빼는 컬럼(docs/experiments.md 공통 규칙 판정): 티켓 본문, 이관 요약, 이관 사유, 이관 행의 고객
# 번호. 이관 행은 '이관이 있었다'만 비교된다. dump_db의 기본 ignore 값이라, 덤프·해시·차이 모두 이 컬럼 없이
# 계산된다. 자유 글을 빼는 것은 정답과 같을 수 없기 때문이고, 이관 행에서 '이관이 있었다' 말고 나머지를 빼는
# 것은 두 사유가 모두 타당할 수 있고 규정이 본인 확인 전후 어느 쪽의 이관도 허용하기 때문이다.
IGNORED_COLUMNS: frozenset[tuple[str, str]] = frozenset(
    {("tickets", "body"), ("handoffs", "summary"), ("handoffs", "reason"), ("handoffs", "customer_id")}
)


# SQLite는 기본적으로 외래 키를 검사하지 않는다. 연결마다 foreign_keys=ON 설정 명령을 보내 PostgreSQL처럼 없는
# 고객·주문을 가리키는 행이 들어가면 오류가 나게 한다.
def _fk_on(dbapi_connection, _record) -> None:  # Python 3.11의 sqlite3 연결에는 .autocommit이 없다
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# 에피소드 전용 메모리 SQLite 엔진을 만든다. source를 주면 그 DB를 sqlite3 backup()으로 바이트 단위 복사한다.
# 왜 이 방식인가: 에피소드마다 seed를 처음부터 다시 생성하지 않고, 한 번 만든 seed DB를 복사해 격리된 사본을
# 쓴다(빠르고, 에피소드끼리 서로의 쓰기를 보지 않음). 정답 재실행(judge.py), 과제 검증, 에피소드(episode.py),
# seed 생성(seed.py)이 모두 이것을 쓴다.
def memory_engine(source: Engine | None = None) -> Engine:
    """개인용 메모리 SQLite 엔진. `source`를 주면 그 데이터베이스의 바이트 복사본이다(1 ms 미만)."""
    # 연결 하나를 만들어 엔진이 계속 그 연결만 쓰게 한다(creator + StaticPool). 메모리 DB는 연결이 끊기면
    # 사라지기 때문이다. check_same_thread=False는 만든 스레드가 아닌 곳에서 연결을 써도 sqlite3가 막지 않게
    # 한다.
    raw = sqlite3.connect(":memory:", check_same_thread=False)
    if source is not None:
        with source.connect() as conn:
            conn.connection.dbapi_connection.backup(raw)
    engine = create_engine("sqlite://", creator=lambda: raw, poolclass=StaticPool)
    # 외래 키 설정 리스너는 이 SQLite 엔진에만 건다. 전역 리스너로 걸면 PostgreSQL 엔진에도 SQLite 전용 명령을
    # 보내게 된다.
    event.listen(engine, "connect", _fk_on)  # 이 엔진에만 건다. PostgreSQL 엔진에는 절대 걸지 않는다
    return engine


# 컬럼 값을 JSON으로 비교·저장할 수 있는 단순한 값으로 바꾼다. enum은 값 문자열로, 시각은 초 단위 UTC ISO
# 문자열로, 날짜는 ISO 날짜로 바꾼다.
# enum 검사를 str 검사보다 먼저 하는 이유: 모든 enum이 StrEnum이라 str로도 통과하기 때문이다. 실수 값이 나오면
# TypeError를 내, 금액이 정수 원이라는 규칙이 깨진 것을 바로 드러낸다.
def _plain(value: Any) -> Any:
    if isinstance(value, enum.Enum):  # str보다 먼저: 여기 enum은 모두 StrEnum이다
        return value.value
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat(timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unexpected column value {value!r}: money is integer won, there are no floats")


# 덤프 형태: 테이블 이름 → 행(dict) 목록.
Dump = dict[str, list[dict[str, Any]]]


# 도메인 테이블 전체를 단순한 행 목록으로 꺼낸다. 판정의 db_match는 이 결과끼리의 == 비교다.
# 정렬을 SQL ORDER BY가 아니라 파이썬에서 기본 키로 하는 이유: DB마다 문자열 정렬 규칙(collation)이 달라,
# SQLite와 PostgreSQL에서 같은 순서를 보장하려는 것이다. sorted_tables는 부모 테이블부터 나오므로 테이블
# 순서도 항상 같다.
def dump_db(engine: Engine, ignore: frozenset[tuple[str, str]] = IGNORED_COLUMNS) -> Dump:
    """모든 도메인 테이블을 단순한 행으로 꺼내, 파이썬에서 기본 키로 정렬한다(DB마다 정렬 규칙이 다르다)."""
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


# 덤프의 sha256. 키를 정렬한 JSON으로 만들어 같은 상태면 언제나 같은 해시가 나온다. 기록의
# seed_hash·final_hash·gold_hash와 manifest의 seed_hash가 이 값이라, 두 실행이 같은 seed DB에서 시작했는지를
# 해시로 확인한다.
def state_hash(dump: Dump) -> str:
    return hashlib.sha256(json.dumps(dump, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


# 두 덤프의 차이를 테이블별로 '한쪽에만 있는 행'으로 돌려준다. 행을 JSON 문자열로 바꿔 집합 차이를 구하므로 행
# 순서와 무관하다. 바뀐 행은 only_expected(이전 값)와 only_actual(새 값)에 한 번씩 나타난다.
# 쓰임: 에피소드 기록의 db_changes(seed → 최종), 판정의 db_diff(정답 → 최종). 실패 분석에서 무엇이 달랐는지
# 바로 보이게 한다.
def diff_dumps(expected: Dump, actual: Dump) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """두 덤프 가운데 한쪽에만 있는 행들을 테이블별로 돌려준다. 두 덤프가 같으면 비어 있다."""
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    # expected의 테이블만 돈다. 두 덤프 모두 같은 Base.metadata에서 나오므로 테이블 목록은 같다.
    for name in expected:
        a = {json.dumps(r, sort_keys=True, ensure_ascii=False) for r in expected[name]}
        b = {json.dumps(r, sort_keys=True, ensure_ascii=False) for r in actual.get(name, [])}
        if a != b:
            result[name] = {
                "only_expected": [json.loads(x) for x in sorted(a - b)],
                "only_actual": [json.loads(x) for x in sorted(b - a)],
            }
    return result
