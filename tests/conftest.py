"""Shared fixtures: a tiny hand-made database that does not depend on the seed generator."""

# [파일 역할] 여러 테스트가 함께 쓰는 pytest fixture와 고정 시각을 둔다. 손으로 만든 아주 작은 쇼핑몰 DB(고객
# 1명, 주소 1개, 상품 1개, 주문 2개)를 tiny_engine으로 제공한다.
# [왜 필요한가] seed 생성기(seed.py)와 무관한 DB라, 생성 데이터가 바뀌어도 도구·에이전트 루프·판정 테스트가
# 흔들리지 않는다. 어떤 행이 있는지 한눈에 보이므로 테스트의 기대값을 이 파일만 보고 확인할 수 있다.
# [연결] pytest가 tests/ 아래 테스트에 자동으로 fixture를 준다. test_agent.py, test_claims.py,
# test_review_*.py 등은 NOW를 직접 import하고, factories.py는 add_tiny_rows 위에 행을 더한다.

from __future__ import annotations

# SQLAlchemy Session으로 행을 넣고, 프로젝트의 db 모듈(테이블 정의, memory_engine)과 KST(고정 +9시간 시간대)를
# 쓴다.
from datetime import date, datetime

import pytest
from sqlalchemy.orm import Session

from support_agent import db
from support_agent.clock import KST

# 테스트의 '지금'. 도구와 규정은 실제 시계가 아니라 이 고정 시각(ctx.now)만 보므로, 반품 기한 같은 규정이 언제
# 돌려도 같게 판정된다. 스모크 과제의 now, 서비스의 고정 시계와 같은 2026-09-14 10:00 KST다.
NOW = datetime(2026, 9, 14, 10, 0, tzinfo=KST)


# 작은 DB의 행을 넣는다. 고객 C-1(김하준)의 주문 O-1은 결제 완료(paid)라 취소할 수 있고, O-2는 출고(shipped)라
# 규정상 취소할 수 없다. 두 주문 모두 무선 이어폰 1개 38,900원, 카드 결제다.
# 호출 위치: 아래 tiny_engine fixture, tests/factories.py extend_tiny_rows의 바탕.
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
    # 부모 행을 먼저 DB에 보내야(flush) 외래 키로 그 행을 가리키는 자식 행(주소, 주문)을 넣을 수 있다. 메모리
    # SQLite 엔진은 외래 키 검사를 켜 두었다.
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
    # 상태만 다른 주문 두 개를 같은 모양으로 만든다: 주문 → 주문 줄 1개 → 결제 → 배송(READY, 도착 예정 9월
    # 13일). 송장 번호는 주문 번호를 넣어 서로 다르게 한다.
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


# 작은 DB를 담은 메모리 SQLite 엔진. 테스트는 db.memory_engine(tiny_engine)으로 바이트 복사본을 만들어
# 에피소드 하나의 전용 DB로 쓴다(평가가 에피소드마다 seed DB를 복사하는 방식과 같다). 테스트가 끝나면 엔진을
# 닫는다.
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
