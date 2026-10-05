"""The chat service over a migrated SQLite file, with a scripted model."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import yaml
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import func, inspect, select
from sqlalchemy.orm import Session

from support_agent import db
from support_agent.chat import ChatResponse, ProviderError, ScriptedProvider, ToolCall
from support_agent.seed import build_seed_engine
from support_agent.service import app as app_module
from support_agent.service import store
from support_agent.service.app import create_app
from support_agent.service.bootstrap import copy_seed, prepare_database
from support_agent.service.core import (
    CLOSED_REPLY,
    FALLBACK_REPLY,
    BusyError,
    ChatService,
    without_agent_only,
)
from support_agent.service.offline import OfflineProvider, create_offline_app
from support_agent.service.settings import DEFAULT_MODEL, DEMO_NOW, Settings
from support_agent.service.voice_frontend import VoiceFrontEnd, repair_heard
from support_agent.voice.speech import Audio

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


def test_the_service_runs_qwen35_4b_without_thinking_by_default():
    settings = Settings.from_env({})
    assert (settings.model, settings.think, settings.agent_think()) == ("qwen3.5:4b", "off", False)
    assert (Settings().model, Settings().think) == (DEFAULT_MODEL, "off")


def test_the_old_model_stays_selectable_and_is_sent_no_think_field():
    old = Settings.from_env({"SUPPORT_AGENT_MODEL": "qwen2.5:7b-instruct"})
    assert (old.model, old.think, old.agent_think()) == ("qwen2.5:7b-instruct", "omit", None)
    assert Settings(model="qwen2.5:7b-instruct").think == "omit"
    # An empty value (compose passes THINK through as "") means "not set", not a value.
    empty = {"SUPPORT_AGENT_MODEL": "qwen2.5:7b-instruct", "SUPPORT_AGENT_THINK": ""}
    assert Settings.from_env(empty).think == "omit"
    # An explicit value wins over the model's default.
    explicit = {"SUPPORT_AGENT_MODEL": "other:4b", "SUPPORT_AGENT_THINK": "off"}
    assert Settings.from_env(explicit).agent_think() is False
    assert Settings.from_env({"SUPPORT_AGENT_THINK": "on"}).agent_think() is True
    assert Settings.from_env({"SUPPORT_AGENT_THINK": "omit"}).agent_think() is None
    with pytest.raises(ValidationError):
        Settings.from_env({"SUPPORT_AGENT_THINK": "false"})


class RecordingOllama:
    """An Ollama server behind httpx.MockTransport that records every /api/chat body."""

    def __init__(self):
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/api/chat":
            return httpx.Response(404, text="not found")
        body = json.loads(request.content)
        self.bodies.append(body)
        message = {"role": "assistant", "content": "무엇을 도와드릴까요?"}
        return httpx.Response(200, json={"model": body["model"], "message": message, "done": True})


@pytest.mark.parametrize(
    ("overrides", "model", "think"),
    [({}, "qwen3.5:4b", False), ({"model": "qwen2.5:7b-instruct"}, "qwen2.5:7b-instruct", None)],
)
def test_the_think_setting_reaches_the_ollama_request(engine, monkeypatch, overrides, model, think):
    """The provider the service builds itself (no test double for it): every agent request carries think."""
    server = RecordingOllama()
    real = app_module.OllamaProvider
    monkeypatch.setattr(
        app_module,
        "OllamaProvider",
        lambda *args, **kwargs: real(*args, transport=httpx.MockTransport(server), **kwargs),
    )
    settings = Settings(admin_token="test-token", **overrides)
    with TestClient(create_app(settings, engine=engine)) as client:
        assert client.get("/healthz").json()["think"] == ("off" if think is False else "omit")
        session_id = client.post("/api/sessions").json()["session_id"]
        assert dict(say(client, session_id, "안녕하세요"))["reply"]["text"] == "무엇을 도와드릴까요?"
        detail = client.get(f"/api/admin/sessions/{session_id}", headers=ADMIN).json()
    assert server.bodies, "the service never called the model"
    for body in server.bodies:
        assert body["model"] == model
        assert body.get("think", "absent") == ("absent" if think is None else think)
        assert "think" not in body["options"]
    started = next(e["payload"] for e in detail["audit"] if e["kind"] == "session_started")
    assert (started["model"], started["think"]) == (model, settings.think)


def test_compose_and_the_env_example_name_the_service_default_model():
    root = Path(__file__).resolve().parents[1]
    compose = yaml.safe_load((root / "compose.yaml").read_text(encoding="utf-8"))
    app_env = compose["services"]["app"]["environment"]
    assert app_env["SUPPORT_AGENT_MODEL"] == f"${{MODEL:-{DEFAULT_MODEL}}}"
    assert app_env["SUPPORT_AGENT_THINK"] == "${THINK:-}"  # empty: the think value follows the model
    example = (root / ".env.example").read_text(encoding="utf-8").splitlines()
    assert f"MODEL={DEFAULT_MODEL}" in example and "THINK=" in example


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


def test_tool_errors_of_earlier_turns_do_not_end_a_later_turn(engine):
    """The limit of tool errors is per turn, like the limit of LLM calls: a session that met refusals over
    many turns (not found, not verified, held for approval) still gets its answers."""

    class LooksUpAMissingOrder(ScriptedProvider):
        def chat(self, messages, tools=(), **options):
            if messages[-1].role == "user":
                return ChatResponse(tool_calls=(ToolCall("get_order", {"order_id": "O-99999"}),))
            return ChatResponse(text="주문을 찾지 못했습니다.")

    service = ChatService(Settings(), engine, LooksUpAMissingOrder([]))
    session_id = service.new_session()["session_id"]
    events: list[tuple[str, dict]] = []
    for turn in range(11):
        events.clear()
        service.handle(session_id, "O-99999 주문이요.", lambda name, data: events.append((name, data)))
        assert "error" not in [name for name, _ in events], turn
        assert dict(events)["reply"] == {"text": "주문을 찾지 못했습니다."}
        assert events[-1][1] == {"status": "open"}
    with Session(engine) as session:
        assert session.get(store.ChatSession, session_id).state["tool_errors"] == 11  # the session total


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
        _, events = ask_for_big_cancel(client)
        assert len(client.get("/api/admin/approvals", headers=ADMIN).json()) == 1
        assert [d["text"].count("AP-1") for n, d in events if n == "reply"] == [1]  # one notice per approval


# The qwen3.5:4b reply in the real service turn of 2026-10-04 (docs/service.md): the tool's instructions
# for the agent ("…알리세요", "같은 요청을 다시 호출하지 마세요") passed on to the customer.
LEAKED_REPLY = (
    "주문 O-10097의 환불액 108,200원은 담당자 승인이 필요하여 승인 대기열에 올렸습니다. "
    "고객님께 담당자 확인 후 처리되며 결과는 이 대화창으로 안내된다고 알려드리겠습니다. "
    "같은 요청을 다시 호출하지 마세요."
)


def test_the_customer_never_reads_the_instructions_a_held_write_gives_the_agent(engine):
    with make_client(engine, [*verify_and_cancel(BIG), LEAKED_REPLY]) as client:
        session_id, events = ask_for_big_cancel(client)
        replies = [d["text"] for n, d in events if n == "reply"]
        shown = [m["text"] for m in client.get(f"/api/sessions/{session_id}").json()["messages"]]
        with Session(engine) as session:
            kept = session.get(store.ChatSession, session_id).state["messages"]
        audit = client.get(f"/api/admin/sessions/{session_id}", headers=ADMIN).json()["audit"]

    told_the_agent = next(m["content"] for m in kept if m.get("tool_name") == "cancel_order")
    assert "같은 요청을 다시 호출하지 마세요" in told_the_agent  # the model still reads the whole notice
    for text in [*replies, *shown]:
        assert "호출" not in text and "알리세요" not in text and "알려드리겠습니다" not in text
    assert LEAKED_REPLY not in shown
    # The customer reads the model's first sentence, then the service's notice: the order, the amount and
    # the approval number.
    assert len(replies) == 1 and replies[0] == shown[-1]
    assert replies[0].startswith(
        "주문 O-10097의 환불액 108,200원은 담당자 승인이 필요하여 승인 대기열에 올렸습니다.\n"
    )
    assert all(part in replies[0] for part in ("O-10097", "취소", "108,200원", "AP-1", "담당자"))
    # The conversation (what the model reads next) holds what the customer got and nothing undelivered;
    # the audit log has both.
    held_at = next(i for i, m in enumerate(kept) if m.get("tool_name") == "cancel_order")
    assert [(m["role"], m["content"]) for m in kept[held_at + 1 :]] == [("assistant", replies[0])]
    assert not [m for m in kept if m.get("delivered") is False]
    reply_event = next(e["payload"] for e in audit if e["kind"] == "agent_reply")
    assert (reply_event["text"], reply_event["not_delivered"]) == (replies[0], LEAKED_REPLY)


def test_a_later_reply_after_a_held_write_is_the_models_own(engine):
    later = "담당자 확인을 기다리고 있습니다. 결과는 이 대화창으로 알려 드리겠습니다."
    with make_client(engine, [*verify_and_cancel(BIG), LEAKED_REPLY, later]) as client:
        session_id, _ = ask_for_big_cancel(client)
        assert dict(say(client, session_id, "언제 처리되나요?"))["reply"]["text"] == later


BIG_NOTICE = (
    "주문 O-10097의 취소 요청(환불 예정 108,200원)은 담당자 승인이 필요해 아직 처리되지 않았습니다. "
    "승인 번호는 AP-1이며, 담당자가 확인한 뒤 결과를 이 대화창으로 안내해 드립니다."
)


def test_a_held_turn_keeps_what_the_model_looked_up_and_asks(engine):
    """The review of 972317b: find_customer, get_order and a held cancel in one turn dropped the whole
    model reply, so the lookup never reached the customer, while the next turn's model read that reply
    as said. Now the customer gets the reply without the instructions, then the notice, and the model's
    next turn reads exactly that."""
    name, phone, order_id, _ = BIG
    said = (
        "주문 O-10097은 배송준비중이며 아령 세트와 보조배터리 두 상품입니다. "
        "고객님께 담당자 확인 후 처리되며 결과는 이 대화창으로 안내된다고 알려드리겠습니다. "
        "같은 요청을 다시 호출하지 마세요. 다른 주문도 확인해 드릴까요?"
    )
    later = "네, 담당자가 확인하고 있습니다."
    provider = ScriptedProvider(
        [
            ToolCall("find_customer", {"name": name, "contact": phone}),
            ToolCall("get_order", {"order_id": order_id}),
            ToolCall("cancel_order", {"order_id": order_id, "reason": "changed_mind"}),
            said,
            later,
        ]
    )
    settings = Settings(admin_token="test-token")
    with TestClient(create_app(settings, provider=provider, engine=engine)) as client:
        session_id, events = ask_for_big_cancel(client)
        replies = [d["text"] for n, d in events if n == "reply"]
        assert dict(say(client, session_id, "언제 처리되나요?"))["reply"]["text"] == later
        shown = [m["text"] for m in client.get(f"/api/sessions/{session_id}").json()["messages"]]
        audit = client.get(f"/api/admin/sessions/{session_id}", headers=ADMIN).json()["audit"]
        with Session(engine) as session:
            kept = session.get(store.ChatSession, session_id).state["messages"]

    delivered = (
        "주문 O-10097은 배송준비중이며 아령 세트와 보조배터리 두 상품입니다. 다른 주문도 확인해 드릴까요?\n"
        + BIG_NOTICE
    )
    assert replies == [delivered]
    assert shown[-3:] == [delivered, "언제 처리되나요?", later]
    # The next turn's model reads, after the held call, what the customer got, then the customer.
    seen = provider.requests[-1]["messages"]
    held_at = next(i for i, m in enumerate(seen) if m.tool_name == "cancel_order")
    assert [(m.role, m.content) for m in seen[held_at + 1 :]] == [
        ("assistant", delivered),
        ("user", "언제 처리되나요?"),
    ]
    assert not [m for m in kept if m.get("delivered") is False]
    told = [e["payload"] for e in audit if e["kind"] == "agent_reply"]
    assert [(p["text"], p.get("not_delivered")) for p in told] == [(delivered, said), (later, None)]


LOOKED_UP = "주문 O-10097은 배송준비중입니다."
REAL_TURN = "고객님께 담당자 확인 후 처리되며 결과는 이 대화창으로 안내된다고 알려드리겠습니다."
DONT_CALL = "같은 요청을 다시 호출하지 마세요."


@pytest.mark.parametrize(
    ("said", "kept"),
    [
        # The real turn's two sentences (docs/service.md), and the tool's own words.
        (f"{LOOKED_UP} {REAL_TURN} {DONT_CALL}", LOOKED_UP),
        (
            f"{LOOKED_UP} 고객에게 담당자 확인 후 처리되며 결과는 이 대화창으로 "
            f"안내된다고 알리세요. {DONT_CALL}",
            LOOKED_UP,
        ),
        # No space after a full stop, so one sentence; spaces moved and another ending.
        (f"{LOOKED_UP} {REAL_TURN}{DONT_CALL}", LOOKED_UP),
        (f"{LOOKED_UP} 담당자확인 후처리되며 결과는 이대화창으로 안내된다고 알려드릴게요.", LOOKED_UP),
        # Cut over two lines; markdown and a list item without a full stop.
        (
            f"{LOOKED_UP}\n고객님께 담당자 확인 후\n"
            "처리되며 결과는 이 대화창으로 안내된다고 알려드리겠습니다.",
            LOOKED_UP,
        ),
        (
            f"{LOOKED_UP}\n\n**안내**: 담당자 확인 후 처리되며 결과는 이 대화창으로 안내됩니다.\n"
            "- 같은 요청은 다시 호출하지 말아 주세요",
            LOOKED_UP,
        ),
        # Clauses that are all instruction, a quoting ending (…다고,), a fragment a comma left behind.
        (
            f"{LOOKED_UP} 고객님, 담당자 확인 후 처리되며, 결과는 이 대화창으로 안내된다고 알려 드립니다.",
            LOOKED_UP,
        ),
        (f"{LOOKED_UP} 결과는 이 대화창으로 안내된다고, 고객님께 알려드리겠습니다.", LOOKED_UP),
        (f"{LOOKED_UP} 확인 후 처리되며, 결과는 이 대화창으로 안내됩니다.", LOOKED_UP),
        # A condition is never cut off: alone, its main clause would say more than the sentence did.
        (
            f"{LOOKED_UP} 담당자 확인 후 처리되며 결과는 이 대화창으로 안내된다면, 환불은 3일 안에 됩니다.",
            LOOKED_UP,
        ),
        # Joined to the lookup by a comma after …고: the lookup's clause stays, the instruction goes.
        (
            "주문 O-10097은 배송준비중이고, 고객님께 담당자 확인 후 처리되며 결과는 이 대화창으로 "
            "안내된다고 알려드리겠습니다.",
            "주문 O-10097은 배송준비중이고.",
        ),
        (
            "2번 상품은 담당자 확인 후 처리되며, 1번 상품 반품은 접수되었습니다.",
            "1번 상품 반품은 접수되었습니다.",
        ),
        # The report of the filed line comes first and the instruction starts right after its comma: the
        # copied run crosses the comma, but only the instruction's clause goes.
        (
            "1번 상품 반품은 접수되었으며, 결과는 이 대화창으로 안내됩니다.",
            "1번 상품 반품은 접수되었으며.",
        ),
        (
            "1번 상품 반품은 바로 처리되며, 결과는 이 대화창으로 안내됩니다.",
            "1번 상품 반품은 바로 처리되며.",
        ),
        # Nothing to take out: the reply stays as it was, line breaks and all.
        (f"{LOOKED_UP}\n\n다른 주문도 확인해 드릴까요?", f"{LOOKED_UP}\n\n다른 주문도 확인해 드릴까요?"),
    ],
)
def test_the_instructions_for_the_agent_never_reach_the_customer(said, kept):
    assert without_agent_only(said) == kept


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


SPLIT = ("조시우", "01000007605", "O-10086")  # delivered, in its window: line 1 90,400 + line 2 36,700 won


def test_lines_of_one_order_returned_one_at_a_time_add_up_against_the_threshold(engine):
    name, phone, order_id = SPLIT
    script = [
        ToolCall("find_customer", {"name": name, "contact": phone}),
        ToolCall("request_return", {"order_id": order_id, "line_nos": [1], "reason": "defective"}),
        ToolCall("request_return", {"order_id": order_id, "line_nos": [2], "reason": "defective"}),
        "접수했습니다.",
    ]
    with make_client(engine, script) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        events = say(client, session_id, f"{name}, {phone}입니다. {order_id} 불량이라 반품해 주세요.")
        returns = [d for n, d in events if n == "tool_result" and d["name"] == "request_return"]
        assert [(d["ok"], d["error_code"]) for d in returns] == [(True, None), (False, "approval_required")]
        with Session(engine) as session:
            requests = session.scalars(
                select(db.ServiceRequest).where(db.ServiceRequest.order_id == order_id)
            )
            assert [(r.id, r.refund_won) for r in requests] == [("RT-O-10086-1", 90_400)]
        pending = client.get("/api/admin/approvals?status=pending", headers=ADMIN).json()
        assert [(a["args"]["line_nos"], a["refund_won"]) for a in pending] == [([2], 36_700)]
        audit = client.get(f"/api/admin/sessions/{session_id}", headers=ADMIN).json()["audit"]
        asked = next(e["payload"] for e in audit if e["kind"] == "approval_requested")
        assert (asked["refund_won"], asked["order_refund_won"]) == (36_700, 127_100)
        notice = dict(events)["reply"]["text"]
        assert all(part in notice for part in ("O-10086", "반품", "36,700원", "127,100원", "AP-1"))
        assert notice.startswith("접수했습니다.")  # line 1 was filed in this turn: the model says so
        decided = client.post(
            f"/api/admin/approvals/{pending[0]['id']}/decision", headers=ADMIN, json={"approve": True}
        ).json()
        assert decided["status"] == "approved"  # a person decided: the gate does not hold it again
    with Session(engine) as session:
        assert session.get(db.ServiceRequest, "RT-O-10086-2").refund_won == 36_700


def test_a_turn_that_also_wrote_keeps_the_models_words_without_the_agent_only_ones(engine):
    """Line 1 is filed and line 2 is held in one turn: the model's account of line 1 reaches the customer,
    the sentences that pass on the notice's instructions to the agent do not, and the service's notice
    for line 2 follows."""
    name, phone, order_id = SPLIT
    said = (
        "1번 상품 반품이 접수되었습니다(환불 예정 90,400원). 2번 상품은 담당자 승인이 필요합니다.\n"
        "고객님께 담당자 확인 후 처리되며 결과는 이 대화창으로 안내된다고 알려드리겠습니다. "
        "같은 요청을 다시 호출하지 마세요."
    )
    script = [
        ToolCall("find_customer", {"name": name, "contact": phone}),
        ToolCall("request_return", {"order_id": order_id, "line_nos": [1], "reason": "defective"}),
        ToolCall("request_return", {"order_id": order_id, "line_nos": [2], "reason": "defective"}),
        said,
    ]
    with make_client(engine, script) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        reply = dict(say(client, session_id, f"{name}, {phone}입니다. {order_id} 불량이라 반품해 주세요."))
    text = reply["reply"]["text"]
    assert text.startswith(
        "1번 상품 반품이 접수되었습니다(환불 예정 90,400원). 2번 상품은 담당자 승인이 필요합니다.\n"
    )
    assert "호출" not in text and "알려드리겠습니다" not in text
    assert all(part in text for part in ("O-10086", "36,700원", "AP-1"))


def test_a_sentence_that_reports_a_filed_write_keeps_that_clause(engine):
    """The review of 972317b: a sentence reporting line 1 and repeating the instruction for line 2 was
    dropped whole. Its clauses are divided by a comma after …고, so only the instruction goes."""
    name, phone, order_id = SPLIT
    said = (
        "1번 상품 반품은 접수되었고(환불 예정 90,400원), "
        "2번 상품은 담당자 확인 후 처리되며 결과는 이 대화창으로 안내됩니다."
    )
    script = [
        ToolCall("find_customer", {"name": name, "contact": phone}),
        ToolCall("request_return", {"order_id": order_id, "line_nos": [1], "reason": "defective"}),
        ToolCall("request_return", {"order_id": order_id, "line_nos": [2], "reason": "defective"}),
        said,
    ]
    with make_client(engine, script) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        reply = dict(say(client, session_id, f"{name}, {phone}입니다. {order_id} 불량이라 반품해 주세요."))
    text = reply["reply"]["text"]
    assert text.startswith("1번 상품 반품은 접수되었고(환불 예정 90,400원).\n주문 O-10086의 반품 요청")
    assert "처리되며" not in text


def test_the_notice_the_agent_reads_is_unchanged():
    """The model-facing text of a held write is what it was before the customer got a notice of its own
    (core.py before 2026-10-05), so the agent's inputs and the claim guard's evidence stay the same."""
    from support_agent.service.core import held_for_agent

    assert held_for_agent(108_200, 108_200, "AP-1") == (
        "환불액 108,200원은 담당자 승인이 필요해 아직 처리되지 않았고, "
        "승인 대기열에 올렸습니다(승인 번호 AP-1). "
        "고객에게 담당자 확인 후 처리되며 결과는 이 대화창으로 안내된다고 알리세요. "
        "같은 요청을 다시 호출하지 마세요."
    )
    assert held_for_agent(36_700, 127_100, "AP-2").startswith(
        "환불액 36,700원(같은 주문의 반품 환불액 합계 127,100원)은 담당자 승인이 필요해"
    )


def test_static_pages_carry_no_inline_code():
    from support_agent.service.app import STATIC

    for name in ("index.html", "admin.html"):
        html = (STATIC / name).read_text(encoding="utf-8")
        assert " style=" not in html and "onclick" not in html.lower()
        assert all("src=" in tag for tag in html.split("<script")[1:])


# -------------------------------------------------------------------- voice


class FakeSpeaker:
    def describe(self):
        return {"tts": "fake"}

    def synthesize(self, spoken_text: str, seed: int = 0) -> Audio:
        return Audio(b"RIFF" + spoken_text.encode("utf-8"), seconds=1.0)


class FakeListener:
    """The 'recording' is UTF-8 text; b"static" stands for a recording with nothing intelligible."""

    def describe(self):
        return {"stt": "fake"}

    def transcribe(self, audio: bytes) -> str:
        if audio == b"broken":
            raise ValueError("cannot decode")
        return "" if audio == b"static" else audio.decode("utf-8")


def voice_client(engine, script, **overrides) -> TestClient:
    settings = Settings(admin_token="test-token", **overrides)
    front_end = VoiceFrontEnd(FakeSpeaker(), FakeListener())
    return TestClient(create_app(settings, provider=ScriptedProvider(script), engine=engine, voice=front_end))


def test_a_spoken_message_takes_the_same_path_as_a_typed_one(engine):
    script = [ToolCall("find_customer", {"name": "정예준", "contact": "01000009005"}), "확인되었습니다."]
    with voice_client(engine, script) as client:
        assert client.get("/healthz").json()["voice"] is True
        assert "microphone=(self)" in client.get("/healthz").headers["Permissions-Policy"]
        assert "media-src 'self' blob:" in client.get("/healthz").headers["Content-Security-Policy"]
        session_id = client.post("/api/sessions").json()["session_id"]
        recording = "정예준이고 010-0000-9005입니다.".encode()
        events = events_of(client.post(f"/api/sessions/{session_id}/voice", content=recording))
        names = [n for n, _ in events]
        assert names[:2] == ["status", "heard"] and names[-2:] == ["reply", "end"]
        assert events[0][1] == {"stage": "listening"}
        assert dict(events)["heard"] == {"text": "정예준이고 010-0000-9005입니다."}
        shown = client.get(f"/api/sessions/{session_id}").json()["messages"]
        assert [m["role"] for m in shown] == ["agent", "customer", "agent"]
        audit = client.get(f"/api/admin/sessions/{session_id}", headers=ADMIN).json()["audit"]
        message = next(e["payload"] for e in audit if e["kind"] == "customer_message")
        assert message["via"] == {"voice": True, "heard": message["text"], "audio_bytes": len(recording)}


def test_unintelligible_or_broken_audio_costs_no_turn(engine):
    with voice_client(engine, []) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        for body in (b"static", b"broken"):
            events = events_of(client.post(f"/api/sessions/{session_id}/voice", content=body))
            assert [n for n, _ in events if n in ("error", "reply", "end")] == ["error", "end"]
        assert len(client.get(f"/api/sessions/{session_id}").json()["messages"]) == 1
    with Session(engine) as session:
        assert session.get(store.ChatSession, session_id).turns == 0


def test_voice_requests_are_validated(engine):
    with voice_client(engine, [], max_audio_bytes=10, max_tts_chars=5) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        assert client.post(f"/api/sessions/{session_id}/voice", content=b"x" * 11).status_code == 413
        assert client.post(f"/api/sessions/{session_id}/voice", content=b"").status_code == 422
        assert client.post("/api/sessions/nope/voice", content=b"x").status_code == 404
        long_piece = {"text": "안녕하세요, 고객센터"}  # part of the greeting, but over the limit of 5
        assert client.post(f"/api/sessions/{session_id}/speech", json=long_piece).status_code == 422
        assert client.post("/api/sessions/nope/speech", json={"text": "네"}).status_code == 404


def test_replies_can_be_read_aloud(engine):
    reply = "환불액은 38,900원입니다." + chr(10) + "3~5일 안에 들어옵니다."
    with voice_client(engine, [reply]) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        say(client, session_id, "환불액이 얼마죠?")
        url = f"/api/sessions/{session_id}/speech"
        spoken = client.post(url, json={"text": "환불액은 38,900원입니다."})
        assert spoken.status_code == 200 and spoken.headers["content-type"] == "audio/wav"
        assert spoken.content == b"RIFF" + "환불액은 삼만 팔천구백 원입니다.".encode()  # spelled out first
        # Only what the agent said in this session, whole or in part; spacing does not matter.
        assert client.post(url, json={"text": "38,900원입니다. 3~5일 안에"}).status_code == 200
        assert client.post(url, json={"text": "아무 글이나 읽어 주세요."}).status_code == 403
        assert client.post(url, json={"text": "  "}).status_code == 403


def test_without_the_voice_front_end_the_endpoints_say_so(engine):
    with make_client(engine, []) as client:
        assert client.get("/healthz").json()["voice"] is False
        assert "microphone=()" in client.get("/healthz").headers["Permissions-Policy"]
        session_id = client.post("/api/sessions").json()["session_id"]
        assert client.post(f"/api/sessions/{session_id}/voice", content=b"x").status_code == 503
        assert client.post(f"/api/sessions/{session_id}/speech", json={"text": "네"}).status_code == 503


# -------------------------------------------------------------------- without a model


def test_the_api_description_is_served_only_on_request(engine):
    with make_client(engine, []) as client:
        assert client.get("/openapi.json").status_code == 404
    with make_client(engine, [], expose_openapi=True) as client:
        paths = client.get("/openapi.json").json()["paths"]
        assert "/api/sessions/{session_id}/messages" in paths and "/api/admin/approvals" in paths


def test_the_offline_stand_in_drives_the_real_web_layer(engine):
    settings = Settings(admin_token="test-token")
    with TestClient(create_app(settings, provider=OfflineProvider(), engine=engine)) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        assert "성함" in dict(say(client, session_id, "안녕하세요"))["reply"]["text"]
        events = say(client, session_id, "정예준이고 010-0000-9005입니다.")
        assert [d["name"] for n, d in events if n == "tool"] == ["find_customer"]
        events = say(client, session_id, "O-90005 주문 취소해 주세요.")
        assert [(d["name"], d["ok"]) for n, d in events if n == "tool_result"] == [("cancel_order", True)]
        assert order_status(engine, "O-90005") == "cancelled"
        events = say(client, session_id, "O-10097 주문은요?")  # somebody else's order: the tool refuses
        assert [(d["name"], d["ok"]) for n, d in events if n == "tool_result"] == [("get_order", False)]


def test_the_offline_app_answers_the_voice_endpoints_without_speech_models(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPPORT_AGENT_DATABASE_URL", f"sqlite:///{(tmp_path / 'offline.db').as_posix()}")
    monkeypatch.setenv("SUPPORT_AGENT_VOICE", "1")
    with TestClient(create_offline_app()) as client:
        assert client.get("/healthz").json()["voice"] is True
        session_id = client.post("/api/sessions").json()["session_id"]
        url = f"/api/sessions/{session_id}"
        events = events_of(client.post(f"{url}/voice", content="안녕하세요".encode()))
        assert dict(events)["heard"] == {"text": "안녕하세요"} and events[-1][0] == "end"
        broken = events_of(client.post(f"{url}/voice", content=bytes([255, 254, 0])))  # not UTF-8
        assert [n for n, _ in broken if n in ("error", "reply")] == ["error"]
        spoken = client.post(f"{url}/speech", json={"text": "안녕하세요, 고객센터입니다."})
        assert spoken.status_code == 200 and spoken.content[:4] == b"RIFF"


def test_the_service_repairs_what_was_heard_before_the_agent_reads_it(engine):
    assert repair_heard("강정월구 공일공 공공공공 육일팔구입니다") == "강정월구 010-0000-6189입니다"
    assert repair_heard("주문번호는 5-91001이고 메일은 yunseo.heo-example.com입니다.") == (
        "주문번호는 O-91001이고 메일은 yunseo.heo@example.com입니다."
    )
    script = [ToolCall("find_customer", {"name": "정예준", "contact": "010-0000-9005"}), "확인되었습니다."]
    settings = Settings(admin_token="test-token")
    front_end = VoiceFrontEnd(FakeSpeaker(), FakeListener(), normalizer=repair_heard)
    app = create_app(settings, provider=ScriptedProvider(script), engine=engine, voice=front_end)
    with TestClient(app) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        recording = "정예준이고 공일공 공공공공 구공공오입니다".encode()
        events = events_of(client.post(f"/api/sessions/{session_id}/voice", content=recording))
        assert dict(events)["heard"] == {"text": "정예준이고 010-0000-9005입니다"}
        audit = client.get(f"/api/admin/sessions/{session_id}", headers=ADMIN).json()["audit"]
        message = next(e["payload"] for e in audit if e["kind"] == "customer_message")
        assert (
            message["via"]["heard"] == "정예준이고 공일공 공공공공 구공공오입니다"
        )  # the raw transcript is kept


# -------------------------------------------------------------------- the claim guard

DONE = "주문 O-90005의 취소가 완료되었습니다. 89,100원이 환불됩니다."
VERIFY_SMALL, CANCEL_SMALL = verify_and_cancel(SMALL)


def test_a_claim_that_no_tool_result_shows_is_not_delivered_and_the_retry_does_the_work(engine):
    with make_client(engine, [VERIFY_SMALL, DONE, CANCEL_SMALL, DONE], claims="C1") as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        events = say(client, session_id, "정예준이고 01000009005입니다. O-90005 취소해 주세요.")
        assert [d["text"] for n, d in events if n == "reply"] == [DONE]  # said once, after the write
        assert order_status(engine, "O-90005") == "cancelled"
        audit = client.get(f"/api/admin/sessions/{session_id}", headers=ADMIN).json()["audit"]
        held = [e["payload"]["format_error"] for e in audit if e["kind"] == "llm_call"]
        assert held == [None, "unbacked_claim", None, None]


def test_when_the_model_insists_the_customer_hears_that_nothing_was_done(engine):
    with make_client(engine, [VERIFY_SMALL, DONE, DONE, DONE, "무엇을 도와드릴까요?"], claims="C1") as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        events = say(client, session_id, "정예준이고 01000009005입니다. O-90005 취소해 주세요.")
        assert [n for n, _ in events][-3:] == ["error", "reply", "end"]
        assert dict(events)["reply"]["text"] == FALLBACK_REPLY and events[-1][1] == {"status": "open"}
        assert order_status(engine, "O-90005") == "preparing"
        transcript = client.get(f"/api/sessions/{session_id}").json()
        assert DONE not in [m["text"] for m in transcript["messages"]]
        audit = client.get(f"/api/admin/sessions/{session_id}", headers=ADMIN).json()["audit"]
        assert next(e["payload"]["error"] for e in audit if e["kind"] == "error") == "unbacked_claim"
        assert dict(say(client, session_id, "여보세요?"))["reply"]["text"] == "무엇을 도와드릴까요?"


def test_without_the_guard_the_same_claim_reaches_the_customer(engine):
    with make_client(engine, [VERIFY_SMALL, DONE], claims="C0") as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        events = say(client, session_id, "정예준이고 01000009005입니다. O-90005 취소해 주세요.")
        assert dict(events)["reply"]["text"] == DONE and order_status(engine, "O-90005") == "preparing"


def test_a_write_that_waits_for_a_person_is_not_done_until_it_was_carried_out(engine):
    done = "주문 O-10097의 취소가 완료되었습니다."
    waiting = "아직 담당자 확인을 기다리고 있습니다."
    script = [*verify_and_cancel(BIG), "담당자 확인 후 처리됩니다.", done, waiting, done]
    with make_client(engine, script, claims="C1") as client:
        session_id, _ = ask_for_big_cancel(client)
        assert dict(say(client, session_id, "취소됐나요?"))["reply"]["text"] == waiting  # `done` was held

        client.post("/api/admin/approvals/1/decision", headers=ADMIN, json={"approve": True})
        with Session(engine) as session:
            kept = session.get(store.ChatSession, session_id).state["messages"][-3:]
        assert kept[0]["tool_calls"][0]["name"] == "cancel_order" and kept[1]["tool_name"] == "cancel_order"
        assert json.loads(kept[1]["content"])["status"] == "cancelled" and "108,200원" in kept[2]["content"]
        assert dict(say(client, session_id, "이제 취소됐나요?"))["reply"]["text"] == done  # now it is shown
        shown = [m["text"] for m in client.get(f"/api/sessions/{session_id}").json()["messages"]]
        assert not any(text.startswith("{") for text in shown)  # the tool result stays out of the chat
