"""The chat service over a migrated SQLite file, with a scripted model."""

from __future__ import annotations

import json

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from fastapi.testclient import TestClient
from sqlalchemy import func, inspect, select
from sqlalchemy.orm import Session

from support_agent import db
from support_agent.chat import ProviderError, ScriptedProvider, ToolCall
from support_agent.seed import build_seed_engine
from support_agent.service import store
from support_agent.service.app import create_app
from support_agent.service.bootstrap import copy_seed, prepare_database
from support_agent.service.core import CLOSED_REPLY, FALLBACK_REPLY, BusyError, ChatService
from support_agent.service.settings import DEMO_NOW, Settings

ADMIN = {"X-Admin-Token": "test-token"}
BIG = ("강정우", "01000006189", "O-10097", 108_200)  # a cancellable order above the approval threshold
SMALL = ("정예준", "01000009005", "O-90005", 89_100)


def verify_and_cancel(who: tuple) -> list:
    name, phone, order_id, _ = who
    return [
        ToolCall("find_customer", {"name": name, "contact": phone}),
        ToolCall("cancel_order", {"order_id": order_id, "reason": "changed_mind"}),
    ]


@pytest.fixture(scope="module")
def seeded_template(tmp_path_factory):
    """Migrating and seeding takes a second; do it once and copy the file for every test."""
    path = tmp_path_factory.mktemp("service") / "template.db"
    prepare_database(f"sqlite:///{path.as_posix()}").dispose()
    return path


@pytest.fixture
def engine(seeded_template, tmp_path):
    path = tmp_path / "service.db"
    path.write_bytes(seeded_template.read_bytes())
    engine = prepare_database(f"sqlite:///{path.as_posix()}")
    yield engine
    engine.dispose()


def make_client(engine, script, **overrides) -> TestClient:
    settings = Settings(admin_token="test-token", **overrides)
    return TestClient(create_app(settings, provider=ScriptedProvider(script), engine=engine))


def events_of(response) -> list[tuple[str, dict]]:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    out = []
    for frame in response.text.strip().split("\n\n"):
        event, data = frame.split("\n", 1)
        out.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def say(client: TestClient, session_id: str, text: str) -> list[tuple[str, dict]]:
    return events_of(client.post(f"/api/sessions/{session_id}/messages", json={"text": text}))


def order_status(engine, order_id: str) -> str:
    with Session(engine) as session:
        return session.get(db.Order, order_id).status.value


# -------------------------------------------------------------------- database


def test_migrations_build_exactly_the_declared_tables(engine):
    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        diff = compare_metadata(context, [db.Base.metadata, store.ServiceBase.metadata])
    assert diff == []
    assert {"orders", "chat_sessions", "audit_events", "approvals"} <= set(inspect(engine).get_table_names())


def test_the_shop_is_a_copy_of_the_seed_and_is_not_copied_twice(engine):
    assert db.dump_db(engine) == db.dump_db(build_seed_engine())
    assert copy_seed(engine) is False


def test_settings_come_from_the_environment():
    settings = Settings.from_env(
        {"SUPPORT_AGENT_MODEL": "m", "SUPPORT_AGENT_APPROVAL_REFUND_WON": "5000", "SUPPORT_AGENT_NOW": "real"}
    )
    assert (settings.model, settings.approval_refund_won, settings.now) == ("m", 5000, "real")
    assert settings.clock().tzinfo is not None
    assert Settings.from_env({}).clock().isoformat() == DEMO_NOW
    assert Settings.from_env({}).policy == "P1"


# -------------------------------------------------------------------- chat


def test_a_turn_streams_progress_and_the_reply_and_is_audited(engine):
    script = [*verify_and_cancel(SMALL), "주문 O-90005를 취소했습니다. 89,100원이 환불됩니다."]
    with make_client(engine, script) as client:
        created = client.post("/api/sessions")
        assert created.status_code == 201
        session_id = created.json()["session_id"]
        assert created.json()["messages"][0]["role"] == "agent"

        events = say(client, session_id, "정예준이고 01000009005입니다. O-90005 취소해 주세요.")
        names = [name for name, _ in events]
        assert names[0] == "status" and names[-2:] == ["reply", "end"]
        assert [d["label"] for n, d in events if n == "tool"] == ["본인 확인", "주문 취소"]
        assert all(d["ok"] for n, d in events if n == "tool_result")
        assert events[-1][1] == {"status": "open"}
        assert order_status(engine, "O-90005") == "cancelled"

        transcript = client.get(f"/api/sessions/{session_id}").json()
        assert [m["role"] for m in transcript["messages"]] == ["agent", "customer", "agent"]
        assert transcript["pending_approvals"] == []

        detail = client.get(f"/api/admin/sessions/{session_id}", headers=ADMIN).json()
        kinds = [e["kind"] for e in detail["audit"]]
        assert kinds[:2] == ["session_started", "customer_message"] and kinds[-1] == "agent_reply"
        calls = [e["payload"] for e in detail["audit"] if e["kind"] == "tool_call"]
        assert [(c["name"], c["write"], c["ok"]) for c in calls] == [
            ("find_customer", False, True),
            ("cancel_order", True, True),
        ]
        assert calls[1]["customer_id"] == "C-9005"
        assert kinds.count("llm_call") == 3
        listed = client.get("/api/admin/sessions", headers=ADMIN).json()
        assert listed[0]["session_id"] == session_id and listed[0]["customer_id"] == "C-9005"


def test_the_conversation_survives_between_requests(engine):
    script = [
        ToolCall("find_customer", {"name": "정예준", "contact": "01000009005"}),
        "확인되었습니다.",
        "네.",
    ]
    with make_client(engine, script) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        say(client, session_id, "정예준, 01000009005입니다.")
        say(client, session_id, "감사합니다.")
    with Session(engine) as session:
        row = session.get(store.ChatSession, session_id)
        assert row.turns == 2 and row.verified_customer_id == "C-9005"
        roles = [m["role"] for m in row.state["messages"]]
    assert roles == ["system", "assistant", "user", "assistant", "tool", "assistant", "user", "assistant"]


def test_the_service_refuses_what_the_policy_forbids(engine):
    # O-90005 belongs to C-9005; another verified customer must not cancel it, whatever the model tries.
    script = [
        ToolCall("find_customer", {"name": "강정우", "contact": "01000006189"}),
        ToolCall("cancel_order", {"order_id": "O-90005", "reason": "changed_mind"}),
        "해당 주문은 확인되지 않습니다.",
    ]
    with make_client(engine, script) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        events = say(client, session_id, "O-90005 취소해 주세요.")
    assert [d["ok"] for n, d in events if n == "tool_result"] == [True, False]
    assert order_status(engine, "O-90005") == "preparing"


def test_hand_off_ends_the_session(engine):
    script = [
        ToolCall("find_customer", {"name": "정예준", "contact": "01000009005"}),
        ToolCall("transfer_to_human", {"reason": "customer_request", "summary": "상담원 연결 요청"}),
    ]
    with make_client(engine, script) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        events = say(client, session_id, "상담원 연결해 주세요.")
        assert events[-1][1] == {"status": "handoff"}
        assert (
            client.post(f"/api/sessions/{session_id}/messages", json={"text": "여보세요"}).status_code == 409
        )
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(db.Handoff)) == 1


def test_a_failing_model_gives_a_fallback_reply_and_keeps_the_session_usable(engine):
    class Broken(ScriptedProvider):
        def chat(self, *args, **kwargs):
            if not self.requests:
                self.requests.append({})
                raise ProviderError("connection refused")
            return super().chat(*args, **kwargs)

    settings = Settings(admin_token="test-token")
    with TestClient(create_app(settings, provider=Broken(["다시 말씀해 주세요."]), engine=engine)) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        events = say(client, session_id, "안녕하세요")
        assert [n for n, _ in events][-3:] == ["error", "reply", "end"]
        assert dict(events)["reply"]["text"] == FALLBACK_REPLY
        audit = client.get(f"/api/admin/sessions/{session_id}", headers=ADMIN).json()["audit"]
        assert "ProviderError" in next(e["payload"]["error"] for e in audit if e["kind"] == "error")
        assert dict(say(client, session_id, "주문 조회요"))["reply"]["text"] == "다시 말씀해 주세요."


def test_requests_are_validated(engine):
    with make_client(engine, [], max_message_chars=10) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        assert client.post(f"/api/sessions/{session_id}/messages", json={"text": "   "}).status_code == 422
        assert (
            client.post(f"/api/sessions/{session_id}/messages", json={"text": "가" * 11}).status_code == 422
        )
        assert client.post("/api/sessions/nope/messages", json={"text": "안녕"}).status_code == 404
        assert client.get("/api/sessions/nope").status_code == 404
        page = client.get("/healthz")
        assert page.headers["X-Content-Type-Options"] == "nosniff"
        assert "default-src 'self'" in page.headers["Content-Security-Policy"]
        assert client.get(f"/api/sessions/{session_id}").headers["Cache-Control"] == "no-store"


def test_a_long_session_is_closed(engine):
    with make_client(engine, ["네.", "네."], max_turns_per_session=2) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        assert say(client, session_id, "하나")[-1][1] == {"status": "open"}
        last = say(client, session_id, "둘")
        assert last[-1][1] == {"status": "closed"}
        assert [d["text"] for n, d in last if n == "reply"] == ["네.", CLOSED_REPLY]
        shown = client.get(f"/api/sessions/{session_id}").json()["messages"]
        assert [m["text"] for m in shown[-2:]] == ["네.", CLOSED_REPLY]
        assert client.post(f"/api/sessions/{session_id}/messages", json={"text": "셋"}).status_code == 409


def test_one_session_answers_one_message_at_a_time(engine):
    service = ChatService(Settings(), engine, ScriptedProvider([]))
    session_id = service.new_session()["session_id"]
    service._lock(session_id).acquire()
    with pytest.raises(BusyError):
        service.handle(session_id, "안녕하세요", lambda *_: None)


# -------------------------------------------------------------------- approvals


def test_the_admin_api_needs_its_token(engine):
    with make_client(engine, []) as client:
        assert client.get("/api/admin/approvals").status_code == 401
        assert client.get("/api/admin/approvals", headers={"X-Admin-Token": "wrong"}).status_code == 401
        assert client.get("/api/admin/approvals", headers=ADMIN).json() == []
    with TestClient(create_app(Settings(), provider=ScriptedProvider([]), engine=engine)) as client:
        assert client.get("/api/admin/approvals", headers=ADMIN).status_code == 503


def ask_for_big_cancel(client: TestClient) -> tuple[str, list]:
    session_id = client.post("/api/sessions").json()["session_id"]
    return session_id, say(client, session_id, "강정우, 01000006189입니다. O-10097 취소해 주세요.")


def test_a_large_refund_waits_for_a_person_and_nothing_is_written(engine):
    script = [*verify_and_cancel(BIG), "담당자 확인 후 처리됩니다."]
    with make_client(engine, script) as client:
        before = db.dump_db(engine)
        session_id, events = ask_for_big_cancel(client)
        approval = next(d for n, d in events if n == "approval")
        assert approval == {"code": "AP-1", "tool": "cancel_order", "refund_won": BIG[3]}
        gate = [d for n, d in events if n == "tool_result"][-1]
        assert gate == {"name": "cancel_order", "ok": False, "error_code": "approval_required"}
        assert db.dump_db(engine) == before  # the preview was rolled back

        pending = client.get("/api/admin/approvals?status=pending", headers=ADMIN).json()
        assert [(a["code"], a["args"]["order_id"], a["customer_id"]) for a in pending] == [
            ("AP-1", "O-10097", "C-0001")
        ]
        assert client.get(f"/api/sessions/{session_id}").json()["pending_approvals"] == [approval]


def test_asking_again_does_not_queue_the_same_write_twice(engine):
    cancel = verify_and_cancel(BIG)[1]
    with make_client(engine, [*verify_and_cancel(BIG), cancel, "담당자 확인 후 처리됩니다."]) as client:
        ask_for_big_cancel(client)
        assert len(client.get("/api/admin/approvals", headers=ADMIN).json()) == 1


def test_approval_carries_the_write_out_and_tells_the_customer(engine):
    with make_client(engine, [*verify_and_cancel(BIG), "담당자 확인 후 처리됩니다."]) as client:
        session_id, _ = ask_for_big_cancel(client)
        decided = client.post(
            "/api/admin/approvals/1/decision", headers=ADMIN, json={"approve": True, "by": "kim"}
        ).json()
        assert (decided["status"], decided["decided_by"]) == ("approved", "kim")
        assert json.loads(decided["result"])["refund_won"] == BIG[3]
        assert order_status(engine, "O-10097") == "cancelled"

        transcript = client.get(f"/api/sessions/{session_id}").json()
        assert "108,200원" in transcript["messages"][-1]["text"] and transcript["pending_approvals"] == []
        again = client.post("/api/admin/approvals/1/decision", headers=ADMIN, json={"approve": True})
        assert again.status_code == 409
        kinds = [
            e["kind"] for e in client.get(f"/api/admin/sessions/{session_id}", headers=ADMIN).json()["audit"]
        ]
        assert kinds[-1] == "approval_decided" and "approval_requested" in kinds


def test_rejection_writes_nothing(engine):
    with make_client(engine, [*verify_and_cancel(BIG), "담당자 확인 후 처리됩니다."]) as client:
        session_id, _ = ask_for_big_cancel(client)
        body = {"approve": False, "note": "중복 요청"}
        assert (
            client.post("/api/admin/approvals/1/decision", headers=ADMIN, json=body).json()["status"]
            == "rejected"
        )
        assert order_status(engine, "O-10097") == "preparing"
        assert "중복 요청" in client.get(f"/api/sessions/{session_id}").json()["messages"][-1]["text"]
        assert client.post("/api/admin/approvals/9/decision", headers=ADMIN, json=body).status_code == 409


def test_an_approval_that_comes_too_late_fails_cleanly(engine):
    with make_client(engine, [*verify_and_cancel(BIG), "담당자 확인 후 처리됩니다."]) as client:
        session_id, _ = ask_for_big_cancel(client)
        with Session(engine) as session:  # the parcel left in the meantime
            session.get(db.Order, "O-10097").status = db.OrderStatus.SHIPPED
            session.commit()
        decided = client.post("/api/admin/approvals/1/decision", headers=ADMIN, json={"approve": True}).json()
        assert decided["status"] == "failed" and decided["result"].startswith("Error")
        assert (
            "처리하지 못했습니다" in client.get(f"/api/sessions/{session_id}").json()["messages"][-1]["text"]
        )
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(store.Approval)) == 1


def test_a_refund_below_the_threshold_needs_nobody(engine):
    with make_client(
        engine, [*verify_and_cancel(BIG), "취소했습니다."], approval_refund_won=200_000
    ) as client:
        _, events = ask_for_big_cancel(client)
        assert not [d for n, d in events if n == "approval"]
        assert order_status(engine, "O-10097") == "cancelled"


def test_static_pages_carry_no_inline_code():
    from support_agent.service.app import STATIC

    for name in ("index.html", "admin.html"):
        html = (STATIC / name).read_text(encoding="utf-8")
        assert " style=" not in html and "onclick" not in html.lower()
        assert all("src=" in tag for tag in html.split("<script")[1:])
