"""도구 테스트용 행 만들기 도우미. conftest의 작은 데이터베이스를 넓히고, seed.py에는 기대지 않는다.

`extend_tiny_rows` 뒤의 가게 (NOW는 2026-09-14 10:00 KST):

  C-1 김하준  주소 AD-C-1-1 (집, 기본), AD-C-1-2 (회사)
    O-1 결제 완료  V-1-01 x1            도착 예정 09-13, 미배송 (1일 지연)
    O-2 출고       V-1-01 x1            도착 예정 09-13, 미배송 (1일 지연)
    O-3 배송 완료  V-1-01 x1, V-2-01 x2 09-08 15:00 수령, 도착 예정 09-09 (제때, 반품 기한 안)
    O-4 배송 완료  V-1-01 x1            09-02 14:20 수령, 도착 예정 08-30 (3일 지연, 반품 기한 지남)
    O-5 상품 준비  V-1-01 x1            도착 예정 09-16 (지연 아님)
    O-6 취소       V-1-01 x1
  C-2 이서연  주소 AD-C-2-1
    O-9 결제 완료  V-1-01 x1
  P-1 무선 이어폰: V-1-01 블랙 (5), V-1-02 화이트 (3), V-1-03 레드 (0)   P-2 텀블러: V-2-01 (10), V-2-02 (10)
"""

# [파일 역할] 도구 테스트용 행 만들기 도우미. conftest.py의 작은 DB 위에 고객·주소·상품·주문·보상 쿠폰을 더해,
# 위 docstring의 가게(주문 O-1~O-6, O-9)를 만든다.
# [왜 필요한가] 반품 기한, 배송 지연 보상, 취소 가능 상태 같은 규정은 주문 상태와 날짜 조합마다 결과가 다르다.
# 그 조합을 seed.py에 기대지 않고 몇 줄로 만들 수 있게 해, 테스트가 seed 생성기의 변경에 흔들리지 않게 한다.
# [연결] tests/test_tools.py(가게 픽스처와 보상 쿠폰), tests/test_stage9.py·test_stage10.py가
# extend_tiny_rows를 쓴다.

from __future__ import annotations

# SQLAlchemy Session에 행을 넣고, 프로젝트의 db(테이블 정의와 enum)와 KST 시간대를 쓴다.
from datetime import date, datetime, timedelta

from sqlalchemy.orm import Session

from support_agent import db
from support_agent.clock import KST

# 주문 상태에 맞는 배송 상태. 결제·준비·취소는 출고 전(READY), 출고는 배송 중, 수령은 배송 완료로 맞춰 주문과
# 배송 행이 서로 모순되지 않게 한다.
SHIPMENT_STATUS = {
    db.OrderStatus.PAID: db.ShipmentStatus.READY,
    db.OrderStatus.PREPARING: db.ShipmentStatus.READY,
    db.OrderStatus.SHIPPED: db.ShipmentStatus.IN_TRANSIT,
    db.OrderStatus.DELIVERED: db.ShipmentStatus.DELIVERED,
    db.OrderStatus.CANCELLED: db.ShipmentStatus.READY,
}


# 2026년의 KST 시각을 짧게 만드는 도우미 (기본 오전 9시).
def kst(month: int, day: int, hour: int = 9, minute: int = 0) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=KST)


# 일반 등급 고객 한 명(가입일 6월 1일)을 넣는다. 바로 flush해서 이어지는 주소·주문 행이 외래 키로 가리킬 수
# 있게 한다.
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


# 고객의 배송지를 넣는다. 받는 사람은 그 고객 이름, 우편번호는 고정값이다.
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


# 상품과 옵션(재고 포함)을 넣는다. 상품이 이미 있으면(conftest의 P-1) 옵션만 더한다. 재고 0인 옵션(V-1-03)은
# 교환의 out_of_stock 검사용이다.
def add_product(
    session: Session, product_id: str, name: str, variants: list[tuple[str, str, int, int]]
) -> None:
    """variants: (variant_id, option_label, price_won, stock) 목록."""
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


# 주문 하나와 그에 딸린 줄·결제·배송 행을 한 번에 만든다. 금액은 옵션 가격 × 수량의 합 + 배송비로 계산해
# 행끼리 맞게 한다.
# 취소 상태로 만들 때 남기는 행: 취소 시각·사유(단순 변심), 모든 줄 cancelled, 결제는 환불
# 대기(refund_pending)에 결제액 전액 환불. 도구가 취소한 주문과 같은 모양이어야 '이미 취소된 주문' 검사를
# 제대로 시험할 수 있다.
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
    """lines: (variant_id, quantity) 목록. 취소 상태 주문에는 cancel_order가 썼을 행을 그대로 남긴다."""
    address = session.get(db.CustomerAddress, address_id)
    cancelled = status == db.OrderStatus.CANCELLED
    # 줄마다 옵션 행을 읽어 가격을 가져오고, 상품 금액 합과 총액을 계산한다 (할인은 0원).
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
    # 출고·수령 상태의 주문만 출고 시각(주문 다음 날)을 갖는다. 수령 시각은 인자로 받은 그대로이고, 도착
    # 예정일은 배송 지연 보상의 계산 기준이다.
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


# 이미 발급된 배송 지연 보상 쿠폰 행을 직접 넣는다. 유효 기간은 규정대로 발급 시각 + 30일이다.
# 호출 위치: tests/test_tools.py가 '주문당 1장', '고객당 최근 30일 2장' 한도 규정을 시험할 때 미리 쿠폰이 있는
# 상태를 만든다.
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


# 주문별로 시험하는 상황이 다르다.
# - O-3: 9월 8일 수령, 반품 기한 안(9월 15일까지), 두 상품 세 개라 일부 반품·교환을 시험할 수 있다.
# - O-4: 9월 2일 수령, 반품 기한(9월 9일)이 지났고 도착 예정보다 3일 늦었다.
# - O-5: 상품 준비 중(취소·배송지 변경 가능), 도착 예정이 지나지 않았다.
# - O-6: 이미 취소된 주문.
# - O-9: 다른 고객(C-2)의 주문이라 '본인 주문이 아님' 검사에 쓴다.
def extend_tiny_rows(session: Session) -> None:
    """conftest.add_tiny_rows 위에 모듈 docstring의 가게 행을 더한다."""
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
