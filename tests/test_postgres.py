"""The service on PostgreSQL. Skipped unless SUPPORT_AGENT_TEST_PG_URL points at a server, for example

    docker compose up -d db
    SUPPORT_AGENT_TEST_PG_URL=postgresql+psycopg://support:support-local@127.0.0.1:55462/support uv run pytest

The tests use their own database (`support_agent_test`) on that server and empty it before every test.
SQLite, which the other tests use, serialises writers and ignores time zones; these tests are about the
places where PostgreSQL behaves differently: types, time zones, concurrent deciders.
"""

from __future__ import annotations

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

from support_agent import db
from support_agent.chat import ScriptedProvider
from support_agent.seed import build_seed_engine
from support_agent.service import core, store
from support_agent.service.bootstrap import prepare_database
from support_agent.service.core import ApprovalStateError, ChatService
from support_agent.service.settings import Settings

SERVER_URL = os.environ.get("SUPPORT_AGENT_TEST_PG_URL", "")
TEST_DATABASE = "support_agent_test"
pytestmark = pytest.mark.skipif(not SERVER_URL, reason="SUPPORT_AGENT_TEST_PG_URL is not set")


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


def test_migrations_build_the_declared_tables_on_postgresql(engine):
    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        assert compare_metadata(context, [db.Base.metadata, store.ServiceBase.metadata]) == []


def test_the_shop_copy_equals_the_seed_including_every_timestamp(engine):
    assert db.dump_db(engine) == db.dump_db(build_seed_engine())


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
