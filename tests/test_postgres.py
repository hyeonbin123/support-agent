"""The service on PostgreSQL. Skipped unless SUPPORT_AGENT_TEST_PG_URL points at a server, for example

    docker compose up -d db
    SUPPORT_AGENT_TEST_PG_URL=postgresql+psycopg://support:support-local@127.0.0.1:55462/support uv run pytest

The tests use their own database (`support_agent_test`) on that server and empty it before every test.
SQLite, which the other tests use, serialises writers and ignores time zones; these tests are about the
places where PostgreSQL behaves differently: types, time zones, concurrent deciders.
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

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
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
    assert order_status(engine, "O-90003") == "paid"  # no audit row, no cancellation


# 한 주문의 반품을 줄마다 나눠 접수해도 합계가 승인 기준에 닿으면 사람 승인으로 가는지를 PostgreSQL에서 다시
# 확인한다 (test_service.py의 같은 시나리오). 합계 계산이 도구의 트랜잭션 안에서 돈다.
def test_returns_of_one_order_add_up_against_the_threshold_on_postgresql(engine):
    returns_one_at_a_time(engine)  # the sum over the order runs in the tool's transaction


# 큰 취소가 승인 대기에 걸리면 DB는 바뀌지 않는다(롤백). 그 승인을 두 서비스 객체(두 프로세스 흉내)가
# Barrier로 같은 순간에 결정하면 한쪽만 approved로 실행하고 다른 쪽은 ApprovalStateError로 거절돼야 한다.
# 조건부 UPDATE 한 번으로 승인을 선점하는 설계가 실제 DB의 동시성에서도 지켜지는지 보는 테스트다. 결정 시각은
# 시간대가 붙은 값이고, 승인으로 실행된 도구 호출은 하나만 기록된다.
def test_of_two_deciders_one_carries_the_write_out_on_postgresql(engine):
    with make_client(engine, [*verify_and_cancel(BIG), "담당자 확인 후 처리됩니다."]) as client:
        before = db.dump_db(engine)
        ask_for_big_cancel(client)
        assert db.dump_db(engine) == before  # held for approval: rolled back
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
