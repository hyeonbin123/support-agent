"""db.py 도우미: UTC 시각 타입, 제약, 엔진 복사, 덤프, 차이, 해시."""

# [테스트 묶음] db.py의 도우미를 확인한다: UTC 시각 타입, enum과 외래 키 제약, 메모리 엔진 복사, 판정용
# 덤프(dump_db)와 차이(diff_dumps), 상태 해시(state_hash).
# [왜 중요한가] 판정은 '끝난 DB 덤프 == 정답만 실행한 덤프'다. 덤프가 행 순서나 시간대 표기, 자유 입력 글에
# 따라 달라지면 같은 상태가 다르게 판정되고, 해시로 '같은 seed DB에서 시작했는가'를 확인할 수도 없게 된다.
# db.py는 다른 모듈이 모두 기대는 계약이라 동작을 하나하나 고정한다.

from __future__ import annotations

# copy: 덤프를 깊은 복사해 고친 뒤 비교. SQLAlchemy의 insert·update·text는 ORM을 거치지 않고 DB 제약을 직접
# 시험할 때 쓴다.
import copy
from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlalchemy import insert, text, update
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session

from support_agent import db
from support_agent.clock import KST


# 가입 시각만 바꿔 가며 고객 행을 만드는 도우미.
def make_customer(customer_id: str, joined_at: datetime) -> db.Customer:
    return db.Customer(
        id=customer_id,
        name="홍길동",
        phone="01000000002",
        email="gildong@example.com",
        grade=db.CustomerGrade.NORMAL,
        joined_at=joined_at,
    )


# 시간대가 없는 datetime은 저장 전에 거절해야 한다. UTC인지 KST인지 알 수 없는 값이 섞이면 반품 기한 같은 날짜
# 계산이 하루 어긋난다.
def test_utc_datetime_rejects_naive(tiny_engine):
    with Session(tiny_engine) as session:
        session.add(make_customer("C-2", datetime(2026, 9, 1, 12, 0)))
        with pytest.raises(StatementError, match="naive datetime"):
            session.flush()


# 어느 시간대로 넣든 UTC로 저장되고, 읽으면 같은 순간을 가리키는 UTC 시각으로 돌아와야 한다. KST 9월 1일
# 08:30은 UTC 8월 31일 23:30이다. SQLite에 실제로 저장된 글도 UTC인지 SQL로 직접 읽어 확인한다.
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


# enum 컬럼에 없는 값('lost')은 두 단계에서 막힌다: ORM을 거치면 SQLAlchemy가 DB에 닿기 전에(StatementError),
# SQL로 직접 넣으면 CHECK 제약이(IntegrityError). 목록에 있는 값은 들어간다. PostgreSQL의 고유 ENUM 대신
# VARCHAR + CHECK로 두 DB의 스키마를 같게 한 설계를 확인한다.
def test_enum_rejects_invalid_string(tiny_engine):
    with Session(tiny_engine) as session:
        with pytest.raises(StatementError):  # validate_strings: DB에 닿기 전에 잡힌다
            session.execute(update(db.Order).where(db.Order.id == "O-1").values(status="lost"))
    with tiny_engine.connect() as conn:
        with pytest.raises(IntegrityError):  # CHECK 제약
            conn.execute(text("UPDATE orders SET status = 'lost' WHERE id = 'O-1'"))
        conn.rollback()
        conn.execute(text("UPDATE orders SET status = 'preparing' WHERE id = 'O-1'"))
        conn.rollback()


# enum은 멤버 이름(SHIPPED)이 아니라 값(shipped)으로 저장된다. 덤프와 도구 출력이 이 값을 쓴다.
def test_enum_stores_the_value(tiny_engine):
    with tiny_engine.connect() as conn:
        stored = conn.execute(text("SELECT status FROM orders WHERE id = 'O-2'")).scalar_one()
    assert stored == "shipped"


# SQLite는 기본으로 외래 키를 검사하지 않으므로, 원본 엔진과 그 복사본 모두에서 검사가 켜져 있는지 본다: 없는
# 고객을 가리키는 주소는 넣을 수 없고, 주문이 있는 고객은 지울 수 없다.
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


# memory_engine(원본)으로 만든 복사본은 서로 독립이어야 한다. 한 복사본에서 주문을 취소해도 원본과 다른
# 복사본은 그대로다. 평가가 에피소드마다 seed DB 복사본을 쓰므로, 한 에피소드의 쓰기가 다른 에피소드에 새면
# 판정이 깨진다.
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


# 원본 없이 만든 엔진에 테이블을 만들면 13개 테이블이 모두 비어 있다. 쇼핑몰 테이블 수(README의 13개)를
# 고정한다.
def test_memory_engine_without_source_is_empty():
    engine = db.memory_engine()
    db.Base.metadata.create_all(engine)
    dump = db.dump_db(engine)
    assert set(dump) == set(db.Base.metadata.tables)
    assert len(dump) == 13
    assert all(rows == [] for rows in dump.values())


# 덤프는 넣은 순서와 상관없이 기본 키로 정렬되고(복합 키는 (주문, 줄 번호) 순), 값은 JSON에 바로 쓸 수 있는
# 형태다: 시각은 UTC ISO 문자열, enum은 값, 불리언은 True/False, 날짜는 YYYY-MM-DD. 이래야 두 덤프를 그대로
# 비교하고 기록에 남길 수 있다.
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


# 자유 입력 칸(티켓 본문, 이관 요약)과 이관 행의 사유·고객 번호는 덤프에서 빠져, 글이 달라도 덤프·차이·해시가
# 같아야 한다. 모델마다 다르게 쓰는 글로 판정이 갈리지 않게 하려는 것이고, 이관 사유는 두 가지 모두 타당할 수
# 있어 '이관이 있었다'만 비교한다 (docs/design.md "결정성").
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
    assert set(one["handoffs"][0]) == {"id", "created_at"}
    assert one == two
    assert db.diff_dumps(one, two) == {}
    assert db.state_hash(one) == db.state_hash(two)

    assert db.IGNORED_COLUMNS == {
        ("tickets", "body"),
        ("handoffs", "summary"),
        ("handoffs", "reason"),
        ("handoffs", "customer_id"),
    }


# 금액은 정수 원 단위라 덤프에 실수가 나오면 TypeError로 멈춘다. 실수가 들어왔다는 것은 어딘가의 버그이고,
# 실수 비교는 판정을 불안정하게 만든다.
def test_dump_rejects_floats(tiny_engine):
    engine = db.memory_engine(tiny_engine)
    with engine.connect() as conn:
        conn.execute(text("UPDATE orders SET total_won = 38900.5 WHERE id = 'O-1'"))
        conn.commit()
    with pytest.raises(TypeError, match="no floats"):
        db.dump_db(engine)


# diff_dumps는 같으면 빈 dict, 다르면 테이블별로 '한쪽에만 있는 행'을 준다. 바뀐 행은 이전 값(only_expected)과
# 새 값(only_actual)에 한 번씩, 새로 생긴 행은 한쪽에만 나온다. 방향을 바꾸면 두 쪽이 뒤바뀐다. 실패 분석에서
# 무엇이 달랐는지 보는 데 쓴다.
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


# 상태 해시는 sha256 16진수 64자이고, 같은 상태면 다시 덤프해도, 복사본이어도, dict 키 순서가 달라도 같아야
# 한다. 값 하나만 바뀌어도 달라져야 한다. manifest의 seed_hash로 두 실행이 같은 seed DB에서 시작했는지
# 확인하는 근거다.
def test_state_hash_is_stable(tiny_engine):
    dump = db.dump_db(tiny_engine)
    digest = db.state_hash(dump)
    assert len(digest) == 64 and int(digest, 16) >= 0
    assert digest == db.state_hash(db.dump_db(tiny_engine))
    assert digest == db.state_hash(db.dump_db(db.memory_engine(tiny_engine)))
    assert digest == db.state_hash(
        dict(reversed(list(copy.deepcopy(dump).items())))
    )  # 키 순서는 상관없다(같은 내용을 키 순서만 뒤집어 넣어도 해시가 같다)

    changed = copy.deepcopy(dump)
    changed["orders"][0]["total_won"] += 1
    assert db.state_hash(changed) != digest
