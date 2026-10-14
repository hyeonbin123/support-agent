"""seed 데이터베이스: 결정성, 과제가 기대는 고정 행, 모든 행의 일관성."""

# [테스트 묶음] seed DB(seed.py가 만드는 가상 쇼핑몰 데이터)를 세 방향에서 확인한다: (1) 결정성(같은 seed면
# 같은 DB, 현재 시각·uuid·전역 난수를 쓰지 않음), (2) 과제가 기대는 고정 행(고객 C-9001~9005, 주문
# O-90001~90006)이 docs/design.md 표와 같은지, (3) 모든 행이 서로 모순이 없는지(금액 합, 상태와 결제·배송,
# 접수와 줄 상태, 쿠폰과 할인, 시각의 범위).
# [왜 중요한가] 판정은 '정답만 실행한 seed 복사본'과 비교하고, 실행마다 seed DB의 해시를 manifest에 남겨
# 짝지을 수 있는지 본다. seed가 실행마다 달라지거나 행끼리 모순되면 같은 과제가 다른 결과를 내고, 정답 재실행
# 자체가 틀어진다.
# [방법] 모듈에서 seed 엔진을 한 번 만들고(만드는 데 시간이 들고 결과는 같으므로) 테스트마다 세션을 연다.

from __future__ import annotations

# ast: seed.py 소스를 구문 트리로 읽어 금지된 호출(현재 시각, uuid, 전역 random)이 없는지 검사한다.
import ast
import re
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from support_agent import db, seed
from support_agent.clock import KST


# seed 엔진은 프로세스마다 한 번 만들어 공유된다(build_seed_engine은 캐시된 함수). 이 DB에는 절대 쓰지 않는다.
@pytest.fixture(scope="module")
def engine():
    return seed.build_seed_engine()


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s


# 2026년 KST 시각을 만드는 도우미 (기본 자정).
def kst(month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=KST)


# --- determinism ---


# 캐시를 우회해(__wrapped__) 새로 두 번 만들어도 상태 해시가 같아야 한다. 같은 명령이 같은 seed DB에서
# 시작한다는 재현성의 바탕이다.
def test_two_fresh_builds_have_the_same_hash():
    first = seed.build_seed_engine.__wrapped__()
    second = seed.build_seed_engine.__wrapped__()
    assert first is not second
    assert db.state_hash(db.dump_db(first)) == db.state_hash(db.dump_db(second))


# 캐시된 엔진은 같은 객체로 공유되고, 새로 만든 것과 차이가 없어야 한다.
def test_cached_build_is_shared_and_equal_to_a_fresh_one(engine):
    assert seed.build_seed_engine() is engine
    fresh = seed.build_seed_engine.__wrapped__()
    assert db.diff_dumps(db.dump_db(engine), db.dump_db(fresh)) == {}


# memory_engine으로 만든 복사본(에피소드마다 쓰는 DB)의 해시가 원본과 같아야 한다.
def test_copy_of_seed_is_equal(engine):
    copy = db.memory_engine(engine)
    assert db.state_hash(db.dump_db(copy)) == db.state_hash(db.dump_db(engine))


# seed.py가 현재 시각(now, utcnow, today), uuid, faker, secrets, time, 모듈 수준 random 함수를 쓰지 않는지
# 소스의 구문 트리로 검사한다. random.Random(SEED)처럼 seed를 준 생성기만 허용한다. 도구·seed·판정 코드는 현재
# 시각, uuid, 난수를 쓰지 않는다는 설계(README "구조", docs/design.md "결정성")를 코드로 지킨다.
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


# seed 값, 생성 데이터의 마지막 시각(9월 10일 23:59 KST, 시간대가 붙은 값), 고정 고객·주문 id 목록을 고정한다.
# 과제의 now(9월 14일)는 이 시각보다 뒤다.
def test_constants():
    assert seed.SEED == 20260919
    assert seed.SEED_END == kst(9, 10, 23, 59)
    assert seed.SEED_END.utcoffset() is not None
    assert seed.FIXTURE_CUSTOMER_IDS == ("C-9001", "C-9002", "C-9003", "C-9004", "C-9005")
    assert seed.FIXTURE_ORDER_IDS == ("O-90001", "O-90002", "O-90003", "O-90004", "O-90005", "O-90006")


# 고정 고객 다섯 명의 이름·전화·이메일·주소가 docs/design.md 표와 같고, 기본 주소는 하나(-1)이며 받는 사람은
# 본인 이름이다. 과제의 시나리오와 정답이 이 값을 그대로 쓴다.
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


# 고정 고객에게는 쿠폰·티켓·반품 접수가 없어야 한다. 있으면 쿠폰 한도나 '이미 접수된 줄' 같은 규정이 과제의
# 정답을 바꾼다.
def test_fixture_customers_have_no_coupons_tickets_or_requests(session):
    ids = seed.FIXTURE_CUSTOMER_IDS
    assert session.scalars(select(db.Coupon).where(db.Coupon.customer_id.in_(ids))).all() == []
    assert session.scalars(select(db.Ticket).where(db.Ticket.customer_id.in_(ids))).all() == []
    requests = select(db.ServiceRequest).where(db.ServiceRequest.order_id.in_(seed.FIXTURE_ORDER_IDS))
    assert session.scalars(requests).all() == []


# 고정 주문의 주인, 할인 0원(반품 환불 규칙이 할인을 다루지 않으므로), 기본 주소로 배송, 그리고 고정 고객에게
# 다른 주문이 섞여 있지 않은지 확인한다.
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


# O-90001: 배송 중인 무선 이어폰 38,900원. 택배사·송장 번호·출고일·도착 예정일(9월 15일)이 조회·배송 과제의
# 전달 값이다.
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


# O-90002: 9월 11일 단순 변심 취소, 결제 47,300원이 환불 대기, 모든 줄 취소, 출고 전. '이미 취소된 주문'
# 과제의 바탕이다.
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


# O-90003: 결제 완료(취소 가능), 텀블러 2개 × 14,700원 + 배송비 3,000원 = 32,400원, 카드.
def test_o90003_paid(session):
    order = session.get(db.Order, "O-90003")
    assert order.status == db.OrderStatus.PAID
    assert order.ordered_at.astimezone(KST).date() == date(2026, 9, 13)
    assert [(i.product_name, i.quantity, i.unit_price_won) for i in order.items] == [("텀블러", 2, 14700)]
    assert (order.items_won, order.shipping_fee_won, order.total_won) == (29400, 3000, 32400)
    assert order.payment.method == db.PaymentMethod.CARD
    assert order.payment.status == db.PaymentStatus.PAID
    assert order.shipment.status == db.ShipmentStatus.READY


# O-90004: 9월 2일 14:20 수령한 블루투스 스피커 56,800원. 반품 기한(9월 9일)이 지난 거절 과제의 바탕이고, 제때
# 도착해 배송 지연 보상 자격이 없다.
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
    assert shipment.promised_by >= date(2026, 9, 2)  # 제때 도착: 지연 보상 없음


# O-90005: 상품 준비 중인 러닝화 89,100원, 배송지 AD-C-9005-1. 취소·배송지 변경이 가능한 상태다.
def test_o90005_preparing(session):
    order = session.get(db.Order, "O-90005")
    assert order.status == db.OrderStatus.PREPARING
    assert order.ordered_at.astimezone(KST).date() == date(2026, 9, 12)
    assert [(i.product_name, i.quantity, i.unit_price_won) for i in order.items] == [("러닝화", 1, 89100)]
    assert order.total_won == 89100
    assert order.ship_address_id == "AD-C-9005-1"
    assert order.shipment.status == db.ShipmentStatus.READY


# O-90006: 결제 완료, 양말 세트 3개 × 7,300원 + 배송비 3,000원 = 24,900원.
def test_o90006_paid(session):
    order = session.get(db.Order, "O-90006")
    assert order.status == db.OrderStatus.PAID
    assert order.ordered_at.astimezone(KST).date() == date(2026, 9, 13)
    assert [(i.product_name, i.quantity, i.unit_price_won) for i in order.items] == [("양말 세트", 3, 7300)]
    assert (order.items_won, order.shipping_fee_won, order.total_won) == (21900, 3000, 24900)


# 고정 주문의 상품은 고정 상품(P-9…)만 쓰고, 상품마다 재고 있는 옵션이 둘 이상이라 교환 과제를 만들 수 있다.
# 배경 주문은 이 옵션을 쓰지 않아, 배경 데이터가 고정 상품의 재고를 흔들지 않는다.
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


# --- 모든 행 ---


# 개발용·시험용 과제를 위해 손으로 넣은 행인지 id의 9xxx 번호로 가린다. 아래 수 세기와 시각 범위 검사에서 이
# 행들은 빼고 생성 행만 본다.
def _is_task_row(value) -> bool:
    """스모크·개발용·시험용 과제를 위해 손으로 쓴 행: id에 9xxx 번호가 붙는다."""
    import re

    return isinstance(value, str) and re.search(r"(?:^|-)9\d{3,4}(?:-|$)", value) is not None


# 생성 행의 수와 범위(고객 40, 상품 25, 주문 120, 티켓 15, 이관 0 등)와, 주문 상태·접수 종류·쿠폰 종류가 모두
# 한 번 이상 나오는지 확인한다. 배경 데이터가 모든 상태를 덮어야 도구가 여러 경우를 만난다.
def test_row_counts(session):
    def count(model) -> int:
        # 생성 행만 센다. 손으로 쓴 과제 행(90xx, 91xx, 92xx)은 과제 테스트가 검사한다.
        rows = session.scalars(select(model)).all()
        return sum(not _is_task_row(getattr(r, "id", None) or getattr(r, "order_id", None)) for r in rows)

    assert count(db.Customer) == 40
    assert 60 <= count(db.CustomerAddress) <= 80
    assert count(db.Product) == 25
    assert 50 <= count(db.ProductVariant) <= 70
    assert count(db.Order) == 120
    assert 3 <= count(db.ServiceRequest) <= 12
    assert 25 <= count(db.Coupon) <= 35
    assert count(db.Ticket) == 15
    assert count(db.Handoff) == 0
    assert {o.status for o in session.scalars(select(db.Order))} == set(db.OrderStatus)
    assert {r.kind for r in session.scalars(select(db.ServiceRequest))} == set(db.RequestKind)
    assert {c.kind for c in session.scalars(select(db.Coupon))} == set(db.CouponKind)


# 모든 행 id의 형식을 고정한다(C-, AD-C-, P-, V-, O-, RT-/EX-, CP-, TK-). 반품·교환 id가 내용에서 만들어지는
# 규칙과, 주소·옵션 id가 부모 id로 시작하는지도 본다 (docs/design.md "결정성").
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


# 고객은 가상이다: 전화번호는 0100000XXXX 형식, 이메일은 소문자 @example.com. 전화·이메일·(이름, 전화)가
# 겹치지 않아 본인 확인이 한 사람만 가리키고, 고정 고객의 이름이 배경 고객과 겹치지 않는다. 고객마다 기본
# 주소는 하나다.
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


# 모든 주문의 금액이 맞는지 본다: 줄 번호 1부터 연속, 상품 금액 = Σ(단가 × 수량), 총액 = 상품 금액 + 배송비 −
# 할인, 배송비는 seed 규칙대로, 결제액 = 총액. 주문의 배송지 복사본과 줄의 상품명·옵션·단가가 원래 행과 같아야
# 한다.
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


# 주문 상태와 배송·결제 상태가 맞물리는지 본다: 출고 전 상태는 배송 READY, 출고는 배송 중, 수령은 배송
# 완료이고, 출고·수령 시각의 유무와 순서가 상태와 맞는다. 취소 주문은 취소 시각·사유, 환불 대기 또는 환불
# 완료, 환불액 = 결제액, 모든 줄 취소다. 송장 번호는 겹치지 않는다.
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


# 반품·교환 접수와 주문 줄 상태가 맞는지 본다: 수령한 주문에만, 수령 뒤에 접수되고, 반품 배송비는 단순 변심
# 3,000원·그 밖 0원, 반품 환불액 = 상품 금액 − 배송비, 교환은 한 줄·같은 상품의 다른 옵션. 접수된 줄만 접수
# 상태이고 한 줄이 두 번 접수되지 않는다.
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


# 쿠폰과 티켓의 일관성: 보상 쿠폰은 주문당 1장(CP-주문-1), 금액은 2,000·3,000·5,000원, 유효 30일, 불량 사유는
# 불량 접수가 있을 때만, 지연 사유는 실제로 늦었고 금액이 늦은 일수 구간과 맞을 때만이다. 쓴 쿠폰은 같은
# 고객·금액·시각의 주문 할인과 일대일로 맞아야 한다. 티켓의 주문은 그 고객의 것이다.
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


# 생성 행의 모든 시각(쿠폰 만료 포함)이 2026-06-01 ~ SEED_END 안에 있다. 그래야 과제의 now(9월 14일)가 모든
# 생성 데이터보다 뒤라 미래의 데이터를 보고 판정하지 않는다. 검사한 시각이 500개를 넘는지도 본다(검사가 실제로
# 돌았는지).
def test_bulk_datetimes_are_inside_the_window(engine):
    """생성 행의 모든 시각(쿠폰 만료 포함)이 [2026-06-01, SEED_END] 안에 있다."""
    start, end = seed.BULK_START.astimezone(UTC), seed.SEED_END.astimezone(UTC)
    checked = 0
    for table, rows in db.dump_db(engine).items():
        columns = db.Base.metadata.tables[table].columns
        time_columns = [c.name for c in columns if isinstance(c.type, db.UTCDateTime)]
        for row in rows:
            if any(_is_task_row(row.get(key)) for key in ("id", "order_id", "customer_id", "request_id")):
                continue
            for name in time_columns:
                if row[name] is not None:
                    value = datetime.fromisoformat(row[name])
                    assert start <= value <= end, (table, row)
                    checked += 1
    assert checked > 500


# 고정 주문·고객의 모든 시각이 스모크 과제의 now(9월 14일 10:00)보다 앞이다.
def test_fixture_datetimes_are_before_the_smoke_now(session):
    now = kst(9, 14, 10, 0)
    for order_id in seed.FIXTURE_ORDER_IDS:
        order = session.get(db.Order, order_id)
        times = [order.ordered_at, order.cancelled_at, order.payment.paid_at]
        times += [order.shipment.shipped_at, order.shipment.delivered_at]
        assert all(t < now for t in times if t is not None)
    for customer_id in seed.FIXTURE_CUSTOMER_IDS:
        assert session.get(db.Customer, customer_id).joined_at < now


# 제외 칸 없이 전체를 덤프해도 값은 None·bool·int·str뿐이다(금액은 정수 원). 이관 행은 seed에 없다.
def test_dump_has_no_floats_and_handoffs_are_empty(engine):
    dump = db.dump_db(engine, ignore=frozenset())  # _plain은 float를 만나면 TypeError를 낸다
    assert dump["handoffs"] == []
    for rows in dump.values():
        for row in rows:
            assert all(value is None or isinstance(value, bool | int | str) for value in row.values())


# 택배사는 모두 지어낸 이름 목록(CARRIERS) 안에 있다. 데이터는 모두 가상이고 실제 회사나 사람의 정보가 없다는
# 것(README "데이터")을 확인한다.
def test_no_real_company_names(engine):
    dump = db.dump_db(engine, ignore=frozenset())
    carriers = {row["carrier"] for row in dump["shipments"]}
    assert carriers <= set(seed.CARRIERS)
    assert {"한빛택배", "새벽택배"} <= set(seed.CARRIERS)
