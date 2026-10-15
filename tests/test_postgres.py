"""PostgreSQL 위의 서비스. SUPPORT_AGENT_TEST_PG_URL이 서버를 가리킬 때만 돌고, 아니면 건너뛴다. 예:

    docker compose up -d db
    SUPPORT_AGENT_TEST_PG_URL=postgresql+psycopg://support:support-local@127.0.0.1:55462/support uv run pytest

테스트는 그 서버에 자기 데이터베이스(`support_agent_test`)를 쓰고, 테스트마다 먼저 비운다.
다른 테스트가 쓰는 SQLite는 쓰기를 한 줄로 세우고 시간대를 무시한다. 이 테스트들은 PostgreSQL이 다르게
동작하는 곳을 본다: 타입, 시간대, 동시에 결정하는 사람들.
"""

# [테스트 묶음] 서비스가 PostgreSQL에서만 다르게 움직일 수 있는 곳을 확인한다: 마이그레이션 결과가 선언한
# 테이블과 같은지, 복사한 쇼핑몰 데이터가 생성 데이터와 시각까지 같은지, 감사 행을 쓰지 못하면 주문도 바뀌지
# 않는지, 같은 주문의 반품 합계, 두 담당자가 같은 승인을 동시에 결정하면 한쪽만 실행하는지 (docs/service.md
# "테스트").
# [왜 따로 두나] 나머지 테스트가 쓰는 SQLite는 쓰기를 한 번에 하나씩만 하고 시간대를 무시해,
# 동시성·시간대·타입 문제를 드러내지 못한다. SUPPORT_AGENT_TEST_PG_URL이 있을 때만 돌고(없으면 건너뜀), CI는
# PostgreSQL 서비스 컨테이너로 이 테스트까지 돌린다.
# [방법] 그 서버에 support_agent_test 데이터베이스를 따로 만들고 테스트마다 비운 뒤 서비스와 같은
# 준비(prepare_database)를 한다. 채팅 시나리오는 tests/test_service.py의 도우미를 그대로 가져와 같은 대화를
# PostgreSQL에서 다시 돌린다.

from __future__ import annotations

# threading: 두 담당자의 동시 결정을 두 스레드로 흉내 낸다. alembic의 compare_metadata: 마이그레이션한 DB와
# 코드의 테이블 선언의 차이를 자동으로 비교한다. test_service의 도우미: 테스트 클라이언트, 대본, 큰·작은 취소
# 시나리오.
import json
import os
import threading
import time

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from factories import add_address, add_compensation_coupon, add_customer, add_order, add_product, kst
from sqlalchemy import create_engine, make_url, select, text
from sqlalchemy.orm import Session
from test_service import (
    ADMIN,
    BIG,
    SMALL,
    ask_for_big_cancel,
    make_client,
    order_status,
    say,
    verify_and_cancel,
)
from test_service import (
    test_lines_of_one_order_returned_one_at_a_time_add_up_against_the_threshold as returns_one_at_a_time,
)

from support_agent import db
from support_agent.chat import ScriptedProvider
from support_agent.seed import build_seed_engine
from support_agent.service import core, store
from support_agent.service.bootstrap import prepare_database
from support_agent.service.core import ApprovalStateError, ChatService
from support_agent.service.settings import Settings
from support_agent.toolkit import ConversationState, ToolBugError, ToolContext, ToolResult, execute
from support_agent.tools import build_registry

# 서버 주소가 없으면 이 파일 전체를 건너뛴다. 테스트용 데이터베이스 이름은 서비스의 데이터베이스와 따로 둔다.
SERVER_URL = os.environ.get("SUPPORT_AGENT_TEST_PG_URL", "")
TEST_DATABASE = "support_agent_test"
pytestmark = pytest.mark.skipif(not SERVER_URL, reason="SUPPORT_AGENT_TEST_PG_URL is not set")


# 테스트마다 깨끗한 PostgreSQL DB를 준비한다: 테스트용 데이터베이스가 없으면 만들고(CREATE DATABASE는 트랜잭션
# 밖에서만 되므로 AUTOCOMMIT), public 스키마를 지웠다 다시 만들어 앞 테스트의 흔적을 없앤 뒤, 마이그레이션과
# seed 복사를 한다.
@pytest.fixture
def engine():
    server = create_engine(make_url(SERVER_URL), isolation_level="AUTOCOMMIT")
    with server.connect() as connection:
        exists = connection.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": TEST_DATABASE}
        ).scalar()
        if not exists:
            connection.execute(text(f'CREATE DATABASE "{TEST_DATABASE}"'))
    server.dispose()
    url = make_url(SERVER_URL).set(database=TEST_DATABASE)
    wipe = create_engine(url, isolation_level="AUTOCOMMIT")
    with wipe.connect() as connection:
        connection.execute(text("DROP SCHEMA public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
    wipe.dispose()
    engine = prepare_database(url.render_as_string(hide_password=False))
    yield engine
    engine.dispose()


# Alembic 마이그레이션 결과가 쇼핑몰 테이블과 서비스 테이블의 선언과 똑같은지 확인한다 (차이 목록이 비어야
# 한다). 마이그레이션을 빠뜨리고 모델만 고친 경우를 잡는다.
def test_migrations_build_the_declared_tables_on_postgresql(engine):
    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        assert compare_metadata(context, [db.Base.metadata, store.ServiceBase.metadata]) == []


# PostgreSQL에 복사한 쇼핑몰 데이터의 덤프가 메모리 SQLite의 seed 덤프와 시각까지 같아야 한다. 서비스가 평가와
# 같은 데이터 위에서 돈다는 보장이고, 시간대 처리 차이를 잡는다.
def test_the_shop_copy_equals_the_seed_including_every_timestamp(engine):
    assert db.dump_db(engine) == db.dump_db(build_seed_engine())


# 쇼핑몰 데이터의 변경과 감사 행은 한 트랜잭션이다 (docs/service.md "정한 것과 이유"). 정상이면 취소와 감사
# 기록이 함께 남는다. 감사 행을 만들 때 오류가 나도록 monkeypatch로 바꾸면 취소도 롤백되어 주문이 그대로여야
# 한다. 감사 기록 없이 주문만 바뀐 상태가 생기지 않는다는 것을 확인한다.
def test_a_turn_writes_the_change_and_its_audit_row_together(engine, monkeypatch):
    with make_client(engine, [*verify_and_cancel(SMALL), "취소했습니다."]) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        say(client, session_id, "정예준, 01000009005입니다. O-90005 취소해 주세요.")
        audit = client.get(f"/api/admin/sessions/{session_id}", headers=ADMIN).json()["audit"]
        calls = [e["payload"] for e in audit if e["kind"] == "tool_call"]
        assert [(c["name"], c["ok"]) for c in calls] == [("find_customer", True), ("cancel_order", True)]
    assert order_status(engine, "O-90005") == "cancelled"

    real = store.AuditEvent

    def failing(**fields):
        if fields.get("kind") == "tool_call" and fields["payload"]["name"] == "cancel_order":
            raise RuntimeError("audit storage failed")
        return real(**fields)

    monkeypatch.setattr(core, "AuditEvent", failing)
    with make_client(
        engine, [*verify_and_cancel(("박도윤", "01000009003", "O-90003", 0)), "취소했습니다."]
    ) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        say(client, session_id, "박도윤, 01000009003입니다. O-90003 취소해 주세요.")
    assert order_status(engine, "O-90003") == "paid"  # 감사 행도 취소도 없다


# 한 주문의 반품을 줄마다 나눠 접수해도 합계가 승인 기준에 닿으면 사람 승인으로 가는지를 PostgreSQL에서 다시
# 확인한다 (test_service.py의 같은 시나리오). 합계 계산이 도구의 트랜잭션 안에서 돈다.
def test_returns_of_one_order_add_up_against_the_threshold_on_postgresql(engine):
    returns_one_at_a_time(engine)


# 큰 취소가 승인 대기에 걸리면 DB는 바뀌지 않는다(롤백). 그 승인을 두 서비스 객체(두 프로세스 흉내)가
# Barrier로 같은 순간에 결정하면 한쪽만 approved로 실행하고 다른 쪽은 ApprovalStateError로 거절돼야 한다.
# 조건부 UPDATE 한 번으로 승인을 선점하는 설계가 실제 DB의 동시성에서도 지켜지는지 보는 테스트다. 결정 시각은
# 시간대가 붙은 값이고, 승인으로 실행된 도구 호출은 하나만 기록된다.
def test_of_two_deciders_one_carries_the_write_out_on_postgresql(engine):
    with make_client(engine, [*verify_and_cancel(BIG), "담당자 확인 후 처리됩니다."]) as client:
        before = db.dump_db(engine)
        ask_for_big_cancel(client)
        assert db.dump_db(engine) == before  # 승인을 기다리므로 롤백됐다
    with Session(engine) as session:
        approval_id = session.scalars(select(store.Approval.id)).one()
    services = [ChatService(Settings(), engine, ScriptedProvider([])) for _ in range(2)]
    barrier, outcomes = threading.Barrier(2), []

    def decide(service: ChatService) -> None:
        barrier.wait()
        try:
            outcomes.append(service.decide(approval_id, approve=True, by="kim")["status"])
        except ApprovalStateError:
            outcomes.append("refused")

    threads = [threading.Thread(target=decide, args=(service,)) for service in services]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert sorted(outcomes) == ["approved", "refused"]
    with Session(engine) as session:
        approval = session.get(store.Approval, approval_id)
        assert (approval.status, json.loads(approval.result)["refund_won"]) == ("approved", BIG[3])
        assert approval.decided_at.tzinfo is not None
        calls = session.scalars(select(store.AuditEvent).where(store.AuditEvent.kind == "tool_call")).all()
        assert [e.payload["ok"] for e in calls if e.payload.get("approval")] == [True]
    assert order_status(engine, "O-10097") == "cancelled"


# ---------------------------------------------------------------- 두 세션이 같은 순간에 쓰기 (2026-10-15)
# 쓰기 도구의 행 잠금(주문 → 고객 순)과 재고 조건부 갱신을 실제 PostgreSQL의 두 트랜잭션으로 확인한다.
# 방법: 첫 호출을 on_success 훅 안에서(변경을 flush해 행 잠금을 쥔 채) 멈춰 두고, 둘째 호출을 시작해 1초 둔 뒤
# 첫 호출을 커밋시킨다. 잠금이 없던 코드에서는 둘째 호출이 첫째의 커밋 전 상태를 읽고 통과해 네 테스트가 모두
# 실패했다(고치기 전 코드로 돌려 확인). 고객·주문 행은 seed와 겹치지 않는 번호로 이 테스트가 직접 넣는다.

NOW = kst(9, 14, 10)  # 9월 12일에 받은 주문의 반품 기한(9월 19일) 안
REGISTRY = build_registry()


# 이 묶음이 쓰는 가게: 고객 셋(C-PG1·2·3), 상품 P-PG의 옵션 둘(재고 5씩), 9월 12일에 받은 주문들.
# O-PG1은 세 줄(반품 경합), O-PG2·O-PG3은 다른 고객의 같은 옵션 한 줄(교환 재고 경합), O-PG4·O-PG5는 C-PG3의
# 도착 예정보다 이틀 늦은 주문(쿠폰 한도 경합. C-PG3은 9월 5일에 이미 한 장을 받음, O-PG6).
def add_race_rows(engine) -> None:
    with Session(engine) as session:
        for number, name in ((1, "도시아"), (2, "모하진"), (3, "편세율")):
            add_customer(session, f"C-PG{number}", name, f"0109999000{number}", f"pg{number}@example.com")
            add_address(
                session, f"AD-C-PG{number}-1", f"C-PG{number}", "집", "가상시 시험구 1", is_default=True
            )
        add_product(
            session, "P-PG", "시험 머그", [("V-PG-01", "흰색", 9000, 5), ("V-PG-02", "검정", 9000, 5)]
        )
        delivered = {
            "ordered_at": kst(9, 9),
            "promised_by": kst(9, 10).date(),
            "delivered_at": kst(9, 12, 15),
        }
        rows = [
            ("O-PG1", "C-PG1", [("V-PG-01", 1)] * 3),
            ("O-PG2", "C-PG1", [("V-PG-01", 1)]),
            ("O-PG3", "C-PG2", [("V-PG-01", 1)]),
            ("O-PG4", "C-PG3", [("V-PG-01", 1)]),
            ("O-PG5", "C-PG3", [("V-PG-01", 1)]),
            ("O-PG6", "C-PG3", [("V-PG-01", 1)]),
        ]
        for order_id, customer_id, lines in rows:
            add_order(
                session,
                order_id,
                customer_id,
                f"AD-{customer_id}-1",
                db.OrderStatus.DELIVERED,
                lines,
                **delivered,
            )
        add_compensation_coupon(session, "CP-O-PG6-1", "C-PG3", "O-PG6", issued_at=kst(9, 5))
        session.commit()


def as_customer(customer_id: str) -> ToolContext:
    return ToolContext(
        now=NOW, enforce_policy=True, state=ConversationState(verified_customer_id=customer_id)
    )


def race(engine, first: tuple, second: tuple) -> tuple[ToolResult | Exception, ToolResult | Exception]:
    """first=(고객, 도구, 인자)를 커밋 직전에 멈춰 둔 사이에 second를 시작하고, 1초 뒤 first를 커밋시킨다."""
    inside, release = threading.Event(), threading.Event()
    results: dict[str, ToolResult | Exception] = {}

    def hold(session, spec, clean, answer) -> None:
        session.flush()  # 변경을 DB에 보내 행 잠금을 쥔다
        inside.set()
        release.wait(30)

    def call(slot: str, customer: str, tool: str, arguments: dict, **hooks) -> None:
        try:
            results[slot] = execute(REGISTRY, engine, as_customer(customer), tool, arguments, **hooks)
        except ToolBugError as error:  # 기본 키가 겹친 쓰기 같은 DB 오류
            results[slot] = error

    one = threading.Thread(target=call, args=("first", *first), kwargs={"on_success": hold})
    one.start()
    assert inside.wait(30)
    two = threading.Thread(target=call, args=("second", *second))
    two.start()
    time.sleep(1.0)  # 둘째가 잠금에서 기다리거나(고친 코드) 커밋 전 상태를 읽고 지나가게(옛 코드) 둔다
    release.set()
    one.join(60)
    two.join(60)
    return results["first"], results["second"]


# 같은 주문의 줄 [1, 3]과 [3]을 같은 순간에 반품: 접수 id(첫 줄 번호)가 달라 둘 다 통과하던 경합. 주문 행
# 잠금으로 늦은 쪽은 3번 줄이 이미 접수된 것을 보고 거절된다.
def test_overlapping_returns_of_one_order_cannot_both_go_through(engine):
    add_race_rows(engine)
    first, second = race(
        engine,
        ("C-PG1", "request_return", {"order_id": "O-PG1", "line_nos": [1, 3], "reason": "changed_mind"}),
        ("C-PG1", "request_return", {"order_id": "O-PG1", "line_nos": [3], "reason": "changed_mind"}),
    )
    assert first.ok
    assert (second.ok, second.error_code) == (False, "already_requested")
    with Session(engine) as session:
        lines = session.scalars(
            select(db.ServiceRequestItem.line_no)
            .join(db.ServiceRequest)
            .where(db.ServiceRequest.order_id == "O-PG1")
        ).all()
    assert sorted(lines) == [1, 3]


# 같은 옵션으로 두 주문을 같은 순간에 교환: 읽고-빼기는 둘 다 재고 5를 읽어 4로 썼다(차감 하나가 사라짐).
# 조건부 UPDATE는 늦은 쪽이 새 재고로 다시 계산해 3이 된다.
def test_two_exchanges_to_one_option_keep_both_stock_decrements(engine):
    add_race_rows(engine)
    arguments = {"line_no": 1, "new_variant_id": "V-PG-02", "reason": "changed_mind"}
    first, second = race(
        engine,
        ("C-PG1", "request_exchange", {"order_id": "O-PG2", **arguments}),
        ("C-PG2", "request_exchange", {"order_id": "O-PG3", **arguments}),
    )
    assert first.ok and second.ok
    with Session(engine) as session:
        assert session.get(db.ProductVariant, "V-PG-02").stock == 3


# 재고가 하나뿐인 옵션으로 같은 순간에 두 번 교환하면 늦은 쪽은 out_of_stock으로 거절되고 재고는 0 아래로 가지
# 않는다(조건부 UPDATE가 바꾼 행이 0).
def test_the_last_unit_of_stock_goes_to_one_exchange_only(engine):
    add_race_rows(engine)
    with Session(engine) as session:
        session.get(db.ProductVariant, "V-PG-02").stock = 1
        session.commit()
    arguments = {"line_no": 1, "new_variant_id": "V-PG-02", "reason": "changed_mind"}
    first, second = race(
        engine,
        ("C-PG1", "request_exchange", {"order_id": "O-PG2", **arguments}),
        ("C-PG2", "request_exchange", {"order_id": "O-PG3", **arguments}),
    )
    assert first.ok
    assert (second.ok, second.error_code) == (False, "out_of_stock")
    with Session(engine) as session:
        assert session.get(db.ProductVariant, "V-PG-02").stock == 0
        assert session.get(db.ServiceRequest, "EX-O-PG3-1") is None


# 한 고객의 다른 주문 두 개에 같은 순간 보상 쿠폰: 둘 다 최근 30일 1장만 보고 3장째를 냈다. 고객 행 잠금으로
# 늦은 쪽은 2장을 보고 coupon_limit_exceeded로 거절된다.
def test_two_coupons_for_one_customer_at_once_respect_the_thirty_day_limit(engine):
    add_race_rows(engine)
    first, second = race(
        engine,
        ("C-PG3", "issue_compensation_coupon", {"order_id": "O-PG4", "reason": "delivery_delay"}),
        ("C-PG3", "issue_compensation_coupon", {"order_id": "O-PG5", "reason": "delivery_delay"}),
    )
    assert first.ok
    assert (second.ok, second.error_code) == (False, "coupon_limit_exceeded")
    with Session(engine) as session:
        coupons = session.scalars(select(db.Coupon.id).where(db.Coupon.customer_id == "C-PG3")).all()
    assert sorted(coupons) == ["CP-O-PG4-1", "CP-O-PG6-1"]


# 한 고객이 두 세션에서 같은 순간 티켓을 남기면 같은 번호(TK-고객-1)를 골라 늦은 쪽이 기본 키 충돌로
# 실패했다. 고객 행 잠금으로 늦은 쪽은 2번을 쓴다.
def test_two_tickets_for_one_customer_at_once_get_different_numbers(engine):
    add_race_rows(engine)
    ticket = {"category": "other", "body": "문의"}
    first, second = race(engine, ("C-PG1", "create_ticket", ticket), ("C-PG1", "create_ticket", ticket))
    assert first.ok and second.ok
    assert {json.loads(r.content)["ticket_id"] for r in (first, second)} == {"TK-C-PG1-1", "TK-C-PG1-2"}
