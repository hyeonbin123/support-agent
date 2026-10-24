"""마이그레이션한 SQLite 파일 위의 채팅 서비스를 대본 모델로 시험한다."""

# [테스트 묶음] 3단계 채팅 서비스를 마이그레이션한 SQLite 파일과 대본대로 답하는 가짜 모델로 확인한다: DB
# 준비와 설정, SSE 채팅 한 턴과 감사 로그, 규정 차단, 이관, 모델 장애, 요청 검증과 한도, 세션 잠금(한 세션은
# 한 번에 메시지 하나, 다 쓴 잠금은 지움), 큰 환불의 사람 승인(대기, 실행, 거절, 늦은 승인, 반품 합계), 승인
# 대기 턴에 에이전트용 지시가 고객에게 새지 않게 하는 처리, 승인 대기 뒤 이관·한도·오류·컨텍스트 한도로 끝난
# 턴의 안내, 음성 끝점, 모델 없는 오프라인 앱, 완료 주장 가드 (docs/service.md "테스트").
# [왜 중요한가] 서비스에서 잘못된 처리는 돈이 나가는 일이다. 평가와 같은 에이전트 루프·도구가 서비스에서도
# 규정을 막고, 큰 환불은 사람을 거치며, 모든 일이 감사 로그에 남는다는 약속을 GPU 없이 확인한다.
# [연결] 여기의 도우미(ADMIN, BIG, SMALL, verify_and_cancel, make_client, say, ask_for_big_cancel,
# voice_client 등)와 fixture는 tests/test_review_t7.py와 tests/test_postgres.py도 가져다 쓴다.

from __future__ import annotations

# httpx: 서비스가 직접 만드는 Ollama 제공자 앞에 가짜 서버(MockTransport)를 끼운다. yaml: compose.yaml의 기본
# 모델 설정을 읽는다. alembic compare_metadata: 마이그레이션 결과와 테이블 선언 비교. TestClient: 서버 없이
# FastAPI 앱을 부른다. 서비스 모듈에서는 앱, DB 준비, 한 턴 처리(ChatService)와 고정 안내 문장, 오프라인 대역,
# 설정, 음성 앞단을 가져온다. threading·time: 세션 잠금 테스트에서 한 턴이나 승인 결정을 작업 스레드로 돌려
# 잠금을 쥔 채 멈춰 두고(Event), 다른 요청이 그 잠금을 기다리기 시작할 때까지 짧게 되풀이해 확인한다. asyncio:
# 음성 끝점이 세션 확인(DB 조회)을 이벤트 루프 스레드 밖에서 하는지 본다(그 스레드에서는 get_running_loop가
# 실패한다). sqlalchemy.event: 한 요청이 실행한 SQL을 before_cursor_execute로 받아 적는다.
import asyncio
import json
import threading
import time
from pathlib import Path

import httpx
import pytest
import yaml
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import event, func, inspect, select
from sqlalchemy.orm import Session

from support_agent import db
from support_agent.chat import ChatResponse, ContextOverflowError, ProviderError, ScriptedProvider, ToolCall
from support_agent.config import HANDOFF_MESSAGE
from support_agent.seed import build_seed_engine
from support_agent.service import app as app_module
from support_agent.service import store
from support_agent.service.app import create_app
from support_agent.service.bootstrap import copy_seed, prepare_database
from support_agent.service.core import (
    CLOSED_REPLY,
    FALLBACK_REPLY,
    HELD_FALLBACK_REPLY,
    BusyError,
    ChatService,
    SessionClosedError,
    SessionNotFoundError,
    without_agent_only,
)
from support_agent.service.offline import OfflineProvider, create_offline_app
from support_agent.service.settings import DEFAULT_MODEL, DEMO_NOW, Settings
from support_agent.service.voice_frontend import VoiceFrontEnd, repair_heard
from support_agent.voice.speech import Audio

# 관리자 API 헤더와 두 시나리오의 고객: BIG은 환불액 108,200원으로 승인 기준(10만 원) 이상인 취소, SMALL은
# 89,100원으로 기준 아래인 취소다. (이름, 전화, 주문, 환불액).
ADMIN = {"X-Admin-Token": "test-token"}
BIG = ("강정우", "01000006189", "O-10097", 108_200)  # 승인 기준 이상인, 취소할 수 있는 주문
SMALL = ("정예준", "01000009005", "O-90005", 89_100)


# 본인 확인 → 단순 변심 취소, 두 도구 호출로 된 대본 앞부분.
def verify_and_cancel(who: tuple) -> list:
    name, phone, order_id, _ = who
    return [
        ToolCall("find_customer", {"name": name, "contact": phone}),
        ToolCall("cancel_order", {"order_id": order_id, "reason": "changed_mind"}),
    ]


# 마이그레이션과 seed 복사를 한 SQLite 파일을 모듈에서 한 번 만든다(1초쯤 걸린다).
@pytest.fixture(scope="module")
def seeded_template(tmp_path_factory):
    """마이그레이션과 seed 넣기는 1초쯤 걸리므로 한 번만 하고, 테스트마다 파일을 복사해 쓴다."""
    path = tmp_path_factory.mktemp("service") / "template.db"
    prepare_database(f"sqlite:///{path.as_posix()}").dispose()
    return path


# 테스트마다 템플릿 파일을 복사한 새 DB. 승인 대기열과 세션이 테스트 사이에 섞이지 않는다.
@pytest.fixture
def engine(seeded_template, tmp_path):
    path = tmp_path / "service.db"
    path.write_bytes(seeded_template.read_bytes())
    engine = prepare_database(f"sqlite:///{path.as_posix()}")
    yield engine
    engine.dispose()


# 관리자 토큰과 대본 제공자를 넣은 앱의 테스트 클라이언트. 다른 설정은 키워드로 덮어쓴다.
def make_client(engine, script, **overrides) -> TestClient:
    settings = Settings(admin_token="test-token", **overrides)
    return TestClient(create_app(settings, provider=ScriptedProvider(script), engine=engine))


# SSE 응답 본문을 (이벤트 이름, 데이터) 목록으로 푼다. 응답은 200이고 text/event-stream이어야 한다.
def events_of(response) -> list[tuple[str, dict]]:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    out = []
    for frame in response.text.strip().split("\n\n"):
        event, data = frame.split("\n", 1)
        out.append((event.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


# 고객 메시지 하나를 보내고 그 턴의 이벤트 목록을 돌려준다.
def say(client: TestClient, session_id: str, text: str) -> list[tuple[str, dict]]:
    return events_of(client.post(f"/api/sessions/{session_id}/messages", json={"text": text}))


# DB에서 주문 상태 값을 직접 읽는다 (화면이 아니라 실제로 바뀌었는지 확인).
def order_status(engine, order_id: str) -> str:
    with Session(engine) as session:
        return session.get(db.Order, order_id).status.value


# -------------------------------------------------------------------- database


# Alembic 마이그레이션 결과가 쇼핑몰·서비스 테이블 선언과 똑같은지 확인한다. 모델만 고치고 마이그레이션을
# 빠뜨리면 여기서 실패한다.
def test_migrations_build_exactly_the_declared_tables(engine):
    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        diff = compare_metadata(context, [db.Base.metadata, store.ServiceBase.metadata])
    assert diff == []
    assert {"orders", "chat_sessions", "audit_events", "approvals"} <= set(inspect(engine).get_table_names())


# 서비스의 쇼핑몰 데이터는 평가의 seed DB와 같고, 이미 데이터가 있으면 다시 복사하지 않는다(재시작해도 주문이
# 초기화되지 않게).
def test_the_shop_is_a_copy_of_the_seed_and_is_not_copied_twice(engine):
    assert db.dump_db(engine) == db.dump_db(build_seed_engine())
    assert copy_seed(engine) is False


# 설정은 SUPPORT_AGENT_ 접두사 환경 변수에서 온다. 기본 시계는 시연용 고정 시각(DEMO_NOW)이고 NOW=real이면
# 실제 시계다. 생성 데이터가 2026-09-14 무렵에 맞춰져 있어 실제 시계를 쓰면 곧 모든 주문이 반품 기한을 넘기기
# 때문이다 (docs/service.md "고정 시계"). 기본 규정은 P1(도구가 막음)이다.
def test_settings_come_from_the_environment():
    settings = Settings.from_env(
        {"SUPPORT_AGENT_MODEL": "m", "SUPPORT_AGENT_APPROVAL_REFUND_WON": "5000", "SUPPORT_AGENT_NOW": "real"}
    )
    assert (settings.model, settings.approval_refund_won, settings.now) == ("m", 5000, "real")
    assert settings.clock().tzinfo is not None
    assert Settings.from_env({}).clock().isoformat() == DEMO_NOW
    assert Settings.from_env({}).policy == "P1"


# 서비스의 기본 모델은 qwen3.5:4b이고 think는 off다. 8단계 측정 결과를 보고 2026-10-04에 바꾼 기본값이다
# (README 8단계).
def test_the_service_runs_qwen35_4b_without_thinking_by_default():
    settings = Settings.from_env({})
    assert (settings.model, settings.think, settings.agent_think()) == ("qwen3.5:4b", "off", False)
    assert (Settings().model, Settings().think) == (DEFAULT_MODEL, "off")


# 이전 모델(qwen2.5 7B)을 고르면 think를 보내지 않는다(omit). 사고 모드가 없는 모델은 think 값을 받으면 거절할
# 수 있기 때문이다. Compose가 넘기는 빈 문자열은 '설정 안 함'이고, 명시한 값(on/off/omit)이 모델 기본값보다
# 우선하며, 그 밖의 값은 설정 오류다.
def test_the_old_model_stays_selectable_and_is_sent_no_think_field():
    old = Settings.from_env({"SUPPORT_AGENT_MODEL": "qwen2.5:7b-instruct"})
    assert (old.model, old.think, old.agent_think()) == ("qwen2.5:7b-instruct", "omit", None)
    assert Settings(model="qwen2.5:7b-instruct").think == "omit"
    # 빈 값(compose는 THINK를 ""로 넘긴다)은 값이 아니라 "설정 안 함"이다.
    empty = {"SUPPORT_AGENT_MODEL": "qwen2.5:7b-instruct", "SUPPORT_AGENT_THINK": ""}
    assert Settings.from_env(empty).think == "omit"
    # 명시한 값이 모델의 기본값보다 우선한다.
    explicit = {"SUPPORT_AGENT_MODEL": "other:4b", "SUPPORT_AGENT_THINK": "off"}
    assert Settings.from_env(explicit).agent_think() is False
    assert Settings.from_env({"SUPPORT_AGENT_THINK": "on"}).agent_think() is True
    assert Settings.from_env({"SUPPORT_AGENT_THINK": "omit"}).agent_think() is None
    with pytest.raises(ValidationError):
        Settings.from_env({"SUPPORT_AGENT_THINK": "false"})


# 정해진 인사로 답한다.
class RecordingOllama:
    """httpx.MockTransport 뒤의 가짜 Ollama 서버. /api/chat 요청 본문을 모두 기록한다."""

    def __init__(self):
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/api/chat":
            return httpx.Response(404, text="not found")
        body = json.loads(request.content)
        self.bodies.append(body)
        message = {"role": "assistant", "content": "무엇을 도와드릴까요?"}
        return httpx.Response(200, json={"model": body["model"], "message": message, "done": True})


# 대역 없이 서비스가 직접 만드는 OllamaProvider로, 모든 에이전트 요청에 모델 이름과 think 값이 설정대로
# 들어가는지(기본 모델은 think false, 이전 모델은 think 칸 없음) 확인한다. /healthz와 감사 로그의 세션 시작
# 기록에도 같은 값이 남는다.
@pytest.mark.parametrize(
    ("overrides", "model", "think"),
    [({}, "qwen3.5:4b", False), ({"model": "qwen2.5:7b-instruct"}, "qwen2.5:7b-instruct", None)],
)
def test_the_think_setting_reaches_the_ollama_request(engine, monkeypatch, overrides, model, think):
    """서비스가 직접 만드는 제공자(대역 없음)로: 에이전트의 모든 요청에 think가 실린다."""
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


# compose.yaml과 .env.example이 코드의 기본 모델과 같은 이름을 쓰는지 확인한다. 기본 모델을 바꿀 때 한쪽만
# 고치는 실수를 막는다.
def test_compose_and_the_env_example_name_the_service_default_model():
    root = Path(__file__).resolve().parents[1]
    compose = yaml.safe_load((root / "compose.yaml").read_text(encoding="utf-8"))
    app_env = compose["services"]["app"]["environment"]
    assert app_env["SUPPORT_AGENT_MODEL"] == f"${{MODEL:-{DEFAULT_MODEL}}}"
    assert app_env["SUPPORT_AGENT_THINK"] == "${THINK:-}"  # 비어 있으면 think 값은 모델을 따른다
    example = (root / ".env.example").read_text(encoding="utf-8").splitlines()
    assert f"MODEL={DEFAULT_MODEL}" in example and "THINK=" in example


# -------------------------------------------------------------------- chat


# 작은 취소 한 턴: 진행 이벤트(status, tool, tool_result)가 먼저 흐르고 reply와 end로 끝나며, 주문이 실제로
# 취소된다. 대화 조회에는 상담원·고객·상담원 순서로 남고, 감사 로그에는 세션 시작, 고객 메시지, 도구 호출(쓰기
# 여부, 확인된 고객), LLM 호출 3번, 답이 남는다. 관리자 세션 목록에도 확인된 고객이 보인다.
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


# 대화 상태는 요청 사이에 DB에 저장된다: 두 턴 뒤 턴 수, 확인된 고객, 모델이 보는 메시지 전체(시스템, 도구
# 호출과 결과 포함)가 세션 행에 남는다.
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


# 모델이 무엇을 시도하든 도구가 막는다: 다른 고객으로 확인된 세션이 남의 주문을 취소하려 하면 거절되고 주문은
# 그대로다.
def test_the_service_refuses_what_the_policy_forbids(engine):
    # O-90005는 C-9005의 주문이다. 모델이 무엇을 시도하든 다른 본인 확인 고객은 취소할 수 없어야 한다.
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


# 상담원 이관이 성공하면 세션이 handoff로 닫히고, 그 뒤의 메시지는 409다. 이관 행이 하나 남는다.
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


# 모델 서버 오류가 나면 error 이벤트와 고정 안내(FALLBACK_REPLY)로 그 턴을 끝내고 감사 로그에 오류를 남긴다.
# 세션은 닫히지 않아 다음 메시지는 정상 처리된다.
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


# 요청 검증과 보안 헤더: 빈 메시지와 길이 한도를 넘는 메시지는 422, 없는 세션은 404. 응답에는 nosniff,
# 'self'만 허용하는 CSP, 대화 조회에는 no-store 캐시 헤더가 붙는다.
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


# 요청 본문에 짝 없는 서로게이트(JSON 이스케이프 "\ud800")가 있으면 422다 (2026-10-24 고침). pydantic은 이런
# 글을 거절하는데, 전에는 FastAPI의 기본 검증 오류 응답이 오류마다 입력 원문(`input`)을 되돌려 보내다가 UTF-8
# 인코딩에서 UnicodeEncodeError가 나 500과 트레이스백이 남았다(글 메시지·답 읽어 주기·관리자 결정 본문 모두.
# 보안 스캔은 이런 입력을 보내지 않아 '5xx 0건'에 잡히지 않았다). 지금은 앱의 검증 오류 처리기가 입력 원문을
# 빼고 어느 칸이 왜 틀렸는지(loc·msg·type)만 돌려준다. 세션은 그대로 쓸 수 있다.
# raise_server_exceptions=False라서 처리되지 않은 예외가 다시 생기면 테스트가 터지는 대신 500으로 보인다.
def test_a_lone_surrogate_in_the_body_is_a_client_error(engine):
    settings = Settings(admin_token="test-token")
    front_end = VoiceFrontEnd(FakeSpeaker(), FakeListener())
    app = create_app(settings, provider=ScriptedProvider(["네."]), engine=engine, voice=front_end)
    as_json = {"Content-Type": "application/json"}
    with TestClient(app, raise_server_exceptions=False) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        body = b'{"text": "\\ud800 marker"}'
        answers = {
            "messages": client.post(f"/api/sessions/{session_id}/messages", content=body, headers=as_json),
            "speech": client.post(f"/api/sessions/{session_id}/speech", content=body, headers=as_json),
            "decision": client.post(
                "/api/admin/approvals/1/decision",
                content=b'{"approve": false, "note": "\\ud800 marker"}',
                headers={**ADMIN, **as_json},
            ),
        }
        for name, answer in answers.items():
            assert answer.status_code == 422, (name, answer.text)
            assert answer.headers["content-type"].startswith("application/json"), name
            assert "marker" not in answer.text, name  # 입력 원문은 되돌려 보내지 않는다
            detail = answer.json()["detail"]
            assert detail and all({"loc", "msg", "type"} <= set(e) and "input" not in e for e in detail), name
        assert answers["messages"].json()["detail"][0]["loc"] == ["body", "text"]
        assert answers["decision"].json()["detail"][0]["loc"] == ["body", "note"]
        assert say(client, session_id, "안녕하세요")[-1] == ("end", {"status": "open"})  # 세션은 멀쩡하다


# 보통의 검증 오류도 같은 처리기를 지난다: 어느 칸이 왜 틀렸는지(loc·msg·type)는 알려 주되 FastAPI 기본 응답에
# 있던 입력 원문(`input`)은 없다. 고객이 보낸 글을 오류 응답에 그대로 되돌려 보낼 이유가 없다
# (2026-10-24부터).
def test_validation_errors_name_the_field_but_do_not_echo_the_input(engine):
    with make_client(engine, []) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        url = f"/api/sessions/{session_id}/messages"
        missing = client.post(url, json={"txt": "안녕"})
        assert missing.status_code == 422
        assert [(e["loc"], e["type"]) for e in missing.json()["detail"]] == [(["body", "text"], "missing")]
        too_long = client.post(url, json={"text": "가나다라" * 1001})  # 요청 모델의 상한 4,000자를 넘는다
        assert too_long.status_code == 422
        (error,) = too_long.json()["detail"]
        assert (error["loc"], error["type"]) == (["body", "text"], "string_too_long")
        assert "input" not in error and "가나다라" not in too_long.text


# 세션당 턴 한도에 닿으면 그 턴의 답 뒤에 고정 안내(CLOSED_REPLY)를 붙이고 세션을 닫는다. 닫힌 세션의 메시지는
# 409다 (docs/service.md "한도": 세션당 40턴, 닫으면 새 상담을 권한다).
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


# 메시지를 받기 전의 세션 확인(app.py open_session)은 세션 상태 한 열만 읽는다(ChatService.session_status).
# 전에는 대화 전체(state JSON)와 승인 대기열까지 읽어 고객 화면용 대화를 만드는 transcript를 불러 그 가운데
# status 하나만 썼으므로, 턴마다 대화 길이에 비례한 읽기가 한 번 더 있었다. 닫힌 세션에 보낸 메시지는 409로
# 끝나 그 확인이 그 요청의 유일한 DB 접근이라, 그 요청에서 실행된 SQL을 그대로 받아 적어 본다.
def test_the_session_check_before_a_turn_reads_only_its_status(engine):
    statements: list[str] = []

    def record(conn, cursor, statement, parameters, context, executemany):
        statements.append(" ".join(statement.split()))

    with make_client(engine, []) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        with Session(engine) as session:
            session.get(store.ChatSession, session_id).status = "closed"
            session.commit()
        service = client.app.state.service
        assert service.session_status(session_id) == "closed"
        with pytest.raises(SessionNotFoundError):
            service.session_status("nope")
        event.listen(engine, "before_cursor_execute", record)
        try:
            assert client.post(f"/api/sessions/{session_id}/messages", json={"text": "셋"}).status_code == 409
        finally:
            event.remove(engine, "before_cursor_execute", record)
    assert len(statements) == 1, statements
    assert "chat_sessions.status" in statements[0]
    assert "chat_sessions.state" not in statements[0] and "approvals" not in statements[0]


# 도구 오류 한도는 턴마다 센다. 없는 주문을 11턴 동안 조회해 거절이 쌓여도(세션 전체 누계 11) 각 턴은 오류
# 없이 답을 받는다. 누계로 세면 오래된 세션이 이유 없이 '처리하지 못했습니다'만 받게 된다.
def test_tool_errors_of_earlier_turns_do_not_end_a_later_turn(engine):
    """도구 오류 한도는 LLM 호출 한도처럼 턴 단위다: 여러 턴에 걸쳐 거절(없음, 본인 확인 안 됨, 승인
    대기)을 만난 세션도 계속 답을 받는다."""

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
        assert session.get(store.ChatSession, session_id).state["tool_errors"] == 11  # 세션 전체 누계


# 한 세션은 한 번에 메시지 하나만 처리한다. 세션 잠금을 미리 잡아 두면 다음 처리는 BusyError다. 프로세스 안의
# 잠금이라 앱은 프로세스 하나로 띄운다 (docs/service.md). 잠금을 놓으면 그 세션의 잠금 항목도 남지 않는다.
def test_one_session_answers_one_message_at_a_time(engine):
    service = ChatService(Settings(), engine, ScriptedProvider([]))
    session_id = service.new_session()["session_id"]
    with service._hold(session_id):
        with pytest.raises(BusyError):
            service.handle(session_id, "안녕하세요", lambda *_: None)
        assert service._locks[session_id].users == 1  # 실패한 시도는 참조 수를 돌려줬다
    assert service._locks == {}


# 세션 잠금 테스트용 대역: `at`번째 호출에서 `go`가 켜질 때까지 멈춰 있다가(멈춘 동안 `inside`가 켜짐)
# 대본대로 답한다. 한 턴이 세션 잠금을 쥔 채 모델을 기다리는 상태를 만든다. 테스트가 잘못돼도 10초 뒤에는
# 풀린다.
class PausesOnCall(ScriptedProvider):
    def __init__(self, script, at: int):
        super().__init__(script)
        self.at, self.calls = at, 0
        self.inside, self.go = threading.Event(), threading.Event()

    def chat(self, *args, **kwargs):
        self.calls += 1
        if self.calls == self.at:
            self.inside.set()
            self.go.wait(10)
        return super().chat(*args, **kwargs)


# 고객 메시지 하나를 작업 스레드에서 처리하기 시작한다(웹의 turn_stream처럼). 이벤트는 버린다.
def turn_in_thread(service: ChatService, session_id: str, text: str) -> threading.Thread:
    thread = threading.Thread(target=service.handle, args=(session_id, text, lambda *_: None), daemon=True)
    thread.start()
    return thread


# 조건이 참이 될 때까지 짧게 되풀이해 본다. 10초 안에 안 되면 테스트 실패다.
def wait_until(condition, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "the condition never came true"
        time.sleep(0.01)


# 세션별 잠금은 그 세션의 요청이 끝나면 메모리에 남지 않는다. 세션 id는 다시 쓰이지 않으므로(난수), 남기면
# 프로세스가 사는 동안 세션 수만큼 는다. 정상 턴, 세션당 턴 한도로 닫히는 턴, 닫힌 세션·없는 세션에 온
# 메시지(예외로 끝남), 승인 결정 뒤를 모두 본다.
def test_session_locks_do_not_outlive_their_requests(engine):
    service = ChatService(Settings(max_turns_per_session=2), engine, ScriptedProvider(["네."] * 6))
    sessions = [service.new_session()["session_id"] for _ in range(3)]
    for session_id in sessions:
        for text in ("하나", "둘"):
            service.handle(session_id, text, lambda *_: None)
    assert [service.transcript(s)["status"] for s in sessions] == ["closed"] * 3
    assert service._locks == {}
    with pytest.raises(SessionClosedError):
        service.handle(sessions[0], "셋", lambda *_: None)
    with pytest.raises(SessionNotFoundError):
        service.handle("no-such-session", "안녕하세요", lambda *_: None)
    assert service._locks == {}

    held = ChatService(
        Settings(), engine, ScriptedProvider([*verify_and_cancel(BIG), "확인 후 안내드립니다."])
    )
    session_id = held.new_session()["session_id"]
    held.handle(session_id, "강정우, 01000006189입니다. O-10097 취소해 주세요.", lambda *_: None)
    assert held.decide(1, approve=False, by="kim")["status"] == "rejected"
    assert held._locks == {}


# 잠금을 지우게 된 뒤에도 한 세션은 한 번에 메시지 하나다: 앞 턴이 모델을 기다리는 동안 같은 세션의 둘째
# 메시지는 BusyError이고, 그동안 잠금은 남아 있다가 앞 턴이 끝나면 지워진다. 대화에는 첫 메시지와 그 답만
# 있다.
def test_two_messages_to_one_session_still_take_turns(engine):
    provider = PausesOnCall(["첫 메시지에 대한 답입니다."], at=1)
    service = ChatService(Settings(), engine, provider)
    session_id = service.new_session()["session_id"]
    first = turn_in_thread(service, session_id, "첫 메시지")
    assert provider.inside.wait(10)
    with pytest.raises(BusyError):
        service.handle(session_id, "둘째 메시지", lambda *_: None)
    assert list(service._locks) == [session_id]  # 앞 턴이 쥐고 있는 동안은 있다
    provider.go.set()
    first.join(10)
    assert not first.is_alive() and service._locks == {}
    shown = [m["text"] for m in service.transcript(session_id)["messages"]]
    assert shown[1:] == ["첫 메시지", "첫 메시지에 대한 답입니다."]


# 턴과 승인 결정이 같은 세션에서 겹칠 때: 결정은 턴의 잠금을 기다렸다가 이어받고, 턴이 끝날 때 기다리던 결정이
# 있으면 잠금을 지우지 않는다. 지웠다가 새로 만들면 결정이 도는 동안 같은 세션의 새 메시지가 그 새 잠금으로
# 통과해 둘이 함께 대화 상태를 고친다. 모두 끝나면 잠금이 남지 않고, 승인 결과는 턴이 저장한 대화 뒤에 붙는다.
def test_a_decision_waits_for_the_turn_and_the_lock_survives_the_handover(engine, monkeypatch):
    # 첫 턴은 큰 취소를 승인 대기(AP-1)에 올리고, 둘째 턴은 그 턴의 첫 모델 호출(대본 넷째)에서 멈춘다.
    script = [*verify_and_cancel(BIG), "확인 후 안내드립니다.", "기다려 주세요."]
    provider = PausesOnCall(script, at=4)
    service = ChatService(Settings(), engine, provider)
    session_id = service.new_session()["session_id"]
    service.handle(session_id, "강정우, 01000006189입니다. O-10097 취소해 주세요.", lambda *_: None)
    assert service._locks == {}  # 끝난 턴의 잠금은 남지 않는다
    turn = turn_in_thread(service, session_id, "언제 처리되나요?")
    assert provider.inside.wait(10)

    # 결정은 잠금을 잡은 뒤 멈춰 두어(deciding), 그동안 같은 세션의 새 메시지가 막히는지 본다.
    deciding, finish = threading.Event(), threading.Event()
    decide_locked = service._decide_locked

    def paused_decision(*args, **kwargs):
        deciding.set()
        finish.wait(10)
        return decide_locked(*args, **kwargs)

    monkeypatch.setattr(service, "_decide_locked", paused_decision)
    decided: dict = {}
    decision = threading.Thread(
        target=lambda: decided.update(service.decide(1, approve=True, by="kim")), daemon=True
    )
    decision.start()
    wait_until(lambda: getattr(service._locks.get(session_id), "users", 0) == 2)  # 결정이 기다린다
    provider.go.set()  # 턴이 끝나며 잠금을 넘긴다
    assert deciding.wait(10)
    turn.join(10)
    assert not turn.is_alive()
    assert service._locks[session_id].users == 1  # 결정이 쥐고 있으므로 지우지 않았다
    with pytest.raises(BusyError):  # 지웠다가 새로 만들었다면 이 메시지가 결정과 함께 돌았다
        service.handle(session_id, "여보세요?", lambda *_: None)
    finish.set()
    decision.join(10)
    assert not decision.is_alive() and decided["status"] == "approved"
    assert service._locks == {}
    shown = [m["text"] for m in service.transcript(session_id)["messages"]]
    assert shown[-3:-1] == ["언제 처리되나요?", "기다려 주세요."] and "처리되었습니다" in shown[-1]


# -------------------------------------------------------------------- approvals


# 관리자 API는 X-Admin-Token이 맞아야 하고(없거나 틀리면 401), 토큰을 설정하지 않은 앱은 관리자 API 전체를
# 503으로 끈다. 빈 토큰으로 통과되는 일이 없게 한다.
def test_the_admin_api_needs_its_token(engine):
    with make_client(engine, []) as client:
        assert client.get("/api/admin/approvals").status_code == 401
        assert client.get("/api/admin/approvals", headers={"X-Admin-Token": "wrong"}).status_code == 401
        assert client.get("/api/admin/approvals", headers=ADMIN).json() == []
    with TestClient(create_app(Settings(), provider=ScriptedProvider([]), engine=engine)) as client:
        assert client.get("/api/admin/approvals", headers=ADMIN).status_code == 503


# 새 세션을 열고 강정우 고객의 큰 취소(O-10097)를 한 메시지로 요청한다. (세션 id, 이벤트)를 돌려준다.
def ask_for_big_cancel(client: TestClient) -> tuple[str, list]:
    session_id = client.post("/api/sessions").json()["session_id"]
    return session_id, say(client, session_id, "강정우, 01000006189입니다. O-10097 취소해 주세요.")


# 환불액이 기준 이상이면 도구 트랜잭션을 롤백하고 승인 대기(AP-1)에 올린다. approval 이벤트와
# approval_required 도구 결과가 나오고 DB는 그대로다. 관리자 대기 목록과 고객의 대화 조회에 같은 승인이
# 보인다.
def test_a_large_refund_waits_for_a_person_and_nothing_is_written(engine):
    script = [*verify_and_cancel(BIG), "담당자 확인 후 처리됩니다."]
    with make_client(engine, script) as client:
        before = db.dump_db(engine)
        session_id, events = ask_for_big_cancel(client)
        approval = next(d for n, d in events if n == "approval")
        assert approval == {"code": "AP-1", "tool": "cancel_order", "refund_won": BIG[3]}
        gate = [d for n, d in events if n == "tool_result"][-1]
        assert gate == {"name": "cancel_order", "ok": False, "error_code": "approval_required"}
        assert db.dump_db(engine) == before  # 미리 실행한 것은 롤백됐다

        pending = client.get("/api/admin/approvals?status=pending", headers=ADMIN).json()
        assert [(a["code"], a["args"]["order_id"], a["customer_id"]) for a in pending] == [
            ("AP-1", "O-10097", "C-0001")
        ]
        assert client.get(f"/api/sessions/{session_id}").json()["pending_approvals"] == [approval]


# 모델이 같은 취소를 다시 호출해도 대기열에는 하나만 남고, 고객이 받는 답에도 승인 안내가 한 번만 들어간다.
def test_asking_again_does_not_queue_the_same_write_twice(engine):
    cancel = verify_and_cancel(BIG)[1]
    with make_client(engine, [*verify_and_cancel(BIG), cancel, "담당자 확인 후 처리됩니다."]) as client:
        _, events = ask_for_big_cancel(client)
        assert len(client.get("/api/admin/approvals", headers=ADMIN).json()) == 1
        assert [d["text"].count("AP-1") for n, d in events if n == "reply"] == [1]  # 승인 하나에 안내 하나


# 2026-10-04 실제 서비스 턴에서 qwen3.5:4b가 한 답을 그대로 옮긴 대본(docs/service.md). 도구가 에이전트에게만
# 준 지시("…알리세요", "같은 요청을 다시 호출하지 마세요")를 고객에게 옮겨 말했다.
LEAKED_REPLY = (
    "주문 O-10097의 환불액 108,200원은 담당자 승인이 필요하여 승인 대기열에 올렸습니다. "
    "고객님께 담당자 확인 후 처리되며 결과는 이 대화창으로 안내된다고 알려드리겠습니다. "
    "같은 요청을 다시 호출하지 마세요."
)


# 승인 대기에 걸린 턴: 모델은 지시가 든 도구 결과를 그대로 읽지만, 고객이 받는 답과 대화 조회에는 '호출',
# '알리세요' 같은 지시가 없다. 고객은 모델의 첫 문장 뒤에 서비스의 정해진 안내(주문, 취소, 금액, 승인 번호,
# 담당자)를 받는다. 다음 턴의 모델이 읽는 대화에도 고객이 받은 글만 남고, 모델의 원래 답은 감사 로그의
# not_delivered에만 남는다 (docs/service.md "사람 승인").
def test_the_customer_never_reads_the_instructions_a_held_write_gives_the_agent(engine):
    with make_client(engine, [*verify_and_cancel(BIG), LEAKED_REPLY]) as client:
        session_id, events = ask_for_big_cancel(client)
        replies = [d["text"] for n, d in events if n == "reply"]
        shown = [m["text"] for m in client.get(f"/api/sessions/{session_id}").json()["messages"]]
        with Session(engine) as session:
            kept = session.get(store.ChatSession, session_id).state["messages"]
        audit = client.get(f"/api/admin/sessions/{session_id}", headers=ADMIN).json()["audit"]

    told_the_agent = next(m["content"] for m in kept if m.get("tool_name") == "cancel_order")
    assert "같은 요청을 다시 호출하지 마세요" in told_the_agent  # 모델은 여전히 안내 전체를 읽는다
    for text in [*replies, *shown]:
        assert "호출" not in text and "알리세요" not in text and "알려드리겠습니다" not in text
    assert LEAKED_REPLY not in shown
    # 고객은 모델의 첫 문장을 읽고, 이어서 서비스의 안내(주문, 금액, 승인 번호)를 읽는다.
    assert len(replies) == 1 and replies[0] == shown[-1]
    assert replies[0].startswith(
        "주문 O-10097의 환불액 108,200원은 담당자 승인이 필요하여 승인 대기열에 올렸습니다.\n"
    )
    assert all(part in replies[0] for part in ("O-10097", "취소", "108,200원", "AP-1", "담당자"))
    # 대화 기록(모델이 다음에 읽는 것)에는 고객이 받은 것만 있고 전달되지 않은 것은 없다. 감사 로그에는
    # 둘 다 있다.
    held_at = next(i for i, m in enumerate(kept) if m.get("tool_name") == "cancel_order")
    assert [(m["role"], m["content"]) for m in kept[held_at + 1 :]] == [("assistant", replies[0])]
    assert not [m for m in kept if m.get("delivered") is False]
    reply_event = next(e["payload"] for e in audit if e["kind"] == "agent_reply")
    assert (reply_event["text"], reply_event["not_delivered"]) == (replies[0], LEAKED_REPLY)


# 승인 대기 턴이 지난 뒤의 답은 안내를 붙이지 않은 모델 자신의 답이다.
def test_a_later_reply_after_a_held_write_is_the_models_own(engine):
    later = "담당자 확인을 기다리고 있습니다. 결과는 이 대화창으로 알려 드리겠습니다."
    with make_client(engine, [*verify_and_cancel(BIG), LEAKED_REPLY, later]) as client:
        session_id, _ = ask_for_big_cancel(client)
        assert dict(say(client, session_id, "언제 처리되나요?"))["reply"]["text"] == later


# 큰 취소에 붙는 서비스의 정해진 안내 문장.
BIG_NOTICE = (
    "주문 O-10097의 취소 요청(환불 예정 108,200원)은 담당자 승인이 필요해 아직 처리되지 않았습니다. "
    "승인 번호는 AP-1이며, 담당자가 확인한 뒤 결과를 이 대화창으로 안내해 드립니다."
)


# 회귀 테스트(972317b 리뷰): 조회와 질문이 있던 승인 대기 턴에서 모델 답 전체를 버려 조회 결과가 고객에게 가지
# 않던 문제. 이제 지시만 뺀 모델 답 + 정해진 안내가 고객에게 가고, 다음 턴의 모델도 정확히 그 글을 자기 답으로
# 읽는다.
def test_a_held_turn_keeps_what_the_model_looked_up_and_asks(engine):
    """972317b 리뷰: 한 턴에 find_customer, get_order, 승인 대기 취소가 있으면 모델 답 전체를 버려 조회
    결과가 고객에게 가지 않았는데, 다음 턴의 모델은 그 답을 말한 것으로 읽었다. 이제 고객은 지시를 뺀 답과
    안내를 받고, 다음 턴의 모델은 정확히 그것을 읽는다."""
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
    # 다음 턴의 모델은 대기된 호출 뒤에 고객이 받은 글을 읽고, 그다음 고객의 말을 읽는다.
    seen = provider.requests[-1]["messages"]
    held_at = next(i for i, m in enumerate(seen) if m.tool_name == "cancel_order")
    assert [(m.role, m.content) for m in seen[held_at + 1 :]] == [
        ("assistant", delivered),
        ("user", "언제 처리되나요?"),
    ]
    assert not [m for m in kept if m.get("delivered") is False]
    told = [e["payload"] for e in audit if e["kind"] == "agent_reply"]
    assert [(p["text"], p.get("not_delivered")) for p in told] == [(delivered, said), (later, None)]


# without_agent_only 표 테스트의 재료: 조회 결과 문장, 실제 턴에서 나온 지시 문장, 다시 호출하지 말라는 지시.
LOOKED_UP = "주문 O-10097은 배송준비중입니다."
REAL_TURN = "고객님께 담당자 확인 후 처리되며 결과는 이 대화창으로 안내된다고 알려드리겠습니다."
DONT_CALL = "같은 요청을 다시 호출하지 마세요."


# 지시를 빼는 규칙(without_agent_only)의 사례 표: 실제 턴의 문장, 도구의 원문, 띄어쓰기·줄바꿈·마크다운 목록이
# 달라진 것은 빼고, 쉼표로 나뉜 절은 지시와 같은 절만 뺀다. 조건(…면)으로 이어진 절은 떼어 내면 남은 말이
# 원래보다 많은 것을 말하므로 문장째 뺀다. 뺄 것이 없으면 줄바꿈까지 그대로 둔다 (docs/service.md "빼는
# 규칙").
@pytest.mark.parametrize(
    ("said", "kept"),
    [
        # 실제 턴의 두 문장(docs/service.md)과 도구 자신의 문구.
        (f"{LOOKED_UP} {REAL_TURN} {DONT_CALL}", LOOKED_UP),
        (
            f"{LOOKED_UP} 고객에게 담당자 확인 후 처리되며 결과는 이 대화창으로 "
            f"안내된다고 알리세요. {DONT_CALL}",
            LOOKED_UP,
        ),
        # 마침표 뒤에 공백이 없어 한 문장인 경우, 띄어쓰기가 바뀌고 어미가 다른 경우.
        (f"{LOOKED_UP} {REAL_TURN}{DONT_CALL}", LOOKED_UP),
        (f"{LOOKED_UP} 담당자확인 후처리되며 결과는 이대화창으로 안내된다고 알려드릴게요.", LOOKED_UP),
        # 두 줄에 걸쳐 끊긴 경우, 마크다운과 마침표 없는 목록 항목.
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
        # 모두 지시인 절들, 인용 어미(…다고,), 쉼표가 남긴 조각.
        (
            f"{LOOKED_UP} 고객님, 담당자 확인 후 처리되며, 결과는 이 대화창으로 안내된다고 알려 드립니다.",
            LOOKED_UP,
        ),
        (f"{LOOKED_UP} 결과는 이 대화창으로 안내된다고, 고객님께 알려드리겠습니다.", LOOKED_UP),
        (f"{LOOKED_UP} 확인 후 처리되며, 결과는 이 대화창으로 안내됩니다.", LOOKED_UP),
        # 조건절은 잘라 내지 않는다: 주절만 남기면 원래 문장보다 더 많은 것을 말하게 된다.
        (
            f"{LOOKED_UP} 담당자 확인 후 처리되며 결과는 이 대화창으로 안내된다면, 환불은 3일 안에 됩니다.",
            LOOKED_UP,
        ),
        # 조회 결과와 '…고,'로 이어진 경우: 조회 절은 남고 지시는 빠진다.
        (
            "주문 O-10097은 배송준비중이고, 고객님께 담당자 확인 후 처리되며 결과는 이 대화창으로 "
            "안내된다고 알려드리겠습니다.",
            "주문 O-10097은 배송준비중이고.",
        ),
        (
            "2번 상품은 담당자 확인 후 처리되며, 1번 상품 반품은 접수되었습니다.",
            "1번 상품 반품은 접수되었습니다.",
        ),
        # 접수된 줄의 보고가 먼저 오고 지시가 그 쉼표 바로 뒤에서 시작하는 경우: 옮겨 쓴 구간이 쉼표를
        # 넘지만 빠지는 것은 지시의 절뿐이다.
        (
            "1번 상품 반품은 접수되었으며, 결과는 이 대화창으로 안내됩니다.",
            "1번 상품 반품은 접수되었으며.",
        ),
        (
            "1번 상품 반품은 바로 처리되며, 결과는 이 대화창으로 안내됩니다.",
            "1번 상품 반품은 바로 처리되며.",
        ),
        # 뺄 것이 없으면 답은 줄바꿈까지 그대로 남는다.
        (f"{LOOKED_UP}\n\n다른 주문도 확인해 드릴까요?", f"{LOOKED_UP}\n\n다른 주문도 확인해 드릴까요?"),
    ],
)
def test_the_instructions_for_the_agent_never_reach_the_customer(said, kept):
    assert without_agent_only(said) == kept


# 관리자가 승인하면 그때 도구가 실행되어 주문이 취소되고, 결과(환불액)가 정해진 문장으로 고객 대화에 들어가며
# 대기 목록에서 빠진다. 이미 결정된 승인을 다시 결정하면 409이고, 감사 로그에 승인 요청과 결정이 남는다.
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


# 거절하면 아무것도 쓰지 않고, 담당자의 메모가 고객 대화에 안내로 들어간다. 없는 승인 번호는 409다.
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


# 승인을 기다리는 사이 주문 상태가 바뀌면(출고됨) 승인 시점에 도구가 거절해 failed로 남고, 고객에게는 처리하지
# 못했다는 안내가 간다. 승인 행은 하나뿐이다.
def test_an_approval_that_comes_too_late_fails_cleanly(engine):
    with make_client(engine, [*verify_and_cancel(BIG), "담당자 확인 후 처리됩니다."]) as client:
        session_id, _ = ask_for_big_cancel(client)
        with Session(engine) as session:  # 그사이 상품이 출고됐다
            session.get(db.Order, "O-10097").status = db.OrderStatus.SHIPPED
            session.commit()
        decided = client.post("/api/admin/approvals/1/decision", headers=ADMIN, json={"approve": True}).json()
        assert decided["status"] == "failed" and decided["result"].startswith("Error")
        assert (
            "처리하지 못했습니다" in client.get(f"/api/sessions/{session_id}").json()["messages"][-1]["text"]
        )
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(store.Approval)) == 1


# 기준을 20만 원으로 올리면 같은 취소(108,200원)는 승인 없이 바로 실행된다. 관문이 설정값을 따르는지 본다.
def test_a_refund_below_the_threshold_needs_nobody(engine):
    with make_client(
        engine, [*verify_and_cancel(BIG), "취소했습니다."], approval_refund_won=200_000
    ) as client:
        _, events = ask_for_big_cancel(client)
        assert not [d for n, d in events if n == "approval"]
        assert order_status(engine, "O-10097") == "cancelled"


# 줄 두 개인 수령 주문(1번 90,400원, 2번 36,700원, 합계 127,100원). 반품을 줄마다 나눠 접수하는 시나리오에
# 쓴다.
SPLIT = ("조시우", "01000007605", "O-10086")  # 배송 완료, 반품 기한 안: 1번 줄 90,400원 + 2번 줄 36,700원


# 반품은 같은 주문에 이미 접수된 반품의 환불액을 더해 기준과 비교한다. 1번 줄(90,400원)은 기준 아래라 바로
# 접수되고, 2번 줄은 합계 127,100원이 기준 이상이라 승인 대기로 간다. 감사 로그에 줄 환불액과 주문 합계가 함께
# 남고, 같은 턴에 1번 줄을 접수한 모델의 말이 안내 앞에 남는다. 승인하면 관문이 다시 막지 않고 2번 줄이
# 접수된다. 줄을 나눠 승인을 피하는 길을 막는 규칙이다 (docs/service.md "사람 승인"). test_postgres.py가 이
# 함수를 PostgreSQL 엔진으로 다시 부른다.
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
        assert notice.startswith("접수했습니다.")  # 이 턴에 1번 줄이 접수됐으므로 모델이 그렇게 말한다
        decided = client.post(
            f"/api/admin/approvals/{pending[0]['id']}/decision", headers=ADMIN, json={"approve": True}
        ).json()
        assert decided["status"] == "approved"  # 사람이 결정했으므로 관문이 다시 붙잡지 않는다
    with Session(engine) as session:
        assert session.get(db.ServiceRequest, "RT-O-10086-2").refund_won == 36_700


# 한 턴에 1번 줄은 접수되고 2번 줄은 승인 대기일 때, 1번 줄에 대한 모델의 말은 고객에게 가고 지시를 옮긴
# 문장은 빠지며, 2번 줄의 정해진 안내가 뒤에 붙는다.
def test_a_turn_that_also_wrote_keeps_the_models_words_without_the_agent_only_ones(engine):
    """한 턴에 1번 줄은 접수되고 2번 줄은 대기된다: 1번 줄에 대한 모델의 말은 고객에게 가고, 에이전트용
    안내의 지시를 옮긴 문장은 가지 않으며, 2번 줄에 대한 서비스의 안내가 뒤따른다."""
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


# 회귀 테스트(972317b 리뷰): 1번 줄 접수를 알리는 절과 2번 줄의 지시를 되풀이한 절이 '…고,'로 이어진 문장이
# 통째로 버려지던 문제. 이제 지시가 든 절만 빠지고 접수를 알린 절은 남는다.
def test_a_sentence_that_reports_a_filed_write_keeps_that_clause(engine):
    """972317b 리뷰: 1번 줄을 보고하면서 2번 줄의 지시를 되풀이한 문장이 통째로 버려졌다. 그 절들은 '…고,'의
    쉼표로 나뉘므로 지시만 빠진다."""
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


# 모델이 읽는 승인 대기 도구 결과 문구는 고객용 안내를 만들기 전과 바이트까지 같아야 한다. 그래야 서비스
# 모델의 입력과 완료 주장 가드가 근거로 읽는 형식이 바뀌지 않는다 (docs/service.md "이렇게 한 이유"). 반품이면
# 같은 주문의 합계도 함께 적힌다.
def test_the_notice_the_agent_reads_is_unchanged():
    """대기된 쓰기에 대해 모델이 보는 글은 고객용 안내가 따로 생기기 전(2026-10-05 이전의 core.py)과 같다.
    그래서 에이전트의 입력과 완료 주장 가드의 근거가 그대로다."""
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


# 대본을 다 쓴 다음 호출에서 정해진 오류를 내는 대역. 모델 연결 오류(ProviderError)는 서비스의 예외 경로를,
# 컨텍스트 넘침(ContextOverflowError)은 에이전트 루프의 context_limit 종료를 만든다.
class FailsAfterTheScript(ScriptedProvider):
    def __init__(self, script, error: Exception):
        super().__init__(script)
        self.left, self.error = len(script), error

    def chat(self, *args, **kwargs):
        if self.left == 0:
            raise self.error
        self.left -= 1
        return super().chat(*args, **kwargs)


# 큰 취소(BIG)가 승인 대기에 걸린 뒤 같은 턴이 `ending`으로 끝나게 하는 (제공자, 설정 덮어쓰기, 안내 앞에 오는
# 고정 문장, 끝난 세션 상태). get_order 되풀이는 턴당 LLM 호출 한도(4)를 다 쓰게 한다. 한도·오류 종료의 고정
# 문장은 '처리하지 못했습니다'(FALLBACK_REPLY)가 아니라 턴이 끝까지 가지 못했다는 HELD_FALLBACK_REPLY다.
def held_then(ending: str):
    held = verify_and_cancel(BIG)
    stopped = HELD_FALLBACK_REPLY
    if ending == "handoff":
        transfer = ToolCall(
            "transfer_to_human", {"reason": "customer_request", "summary": "상담원 연결 요청"}
        )
        return ScriptedProvider([*held, transfer]), {}, HANDOFF_MESSAGE, "handoff"
    if ending == "max_agent_calls":
        look = ToolCall("get_order", {"order_id": BIG[2]})
        return ScriptedProvider([*held, look, look]), {"max_agent_calls_per_turn": 4}, stopped, "open"
    if ending == "provider_error":
        return FailsAfterTheScript(held, ProviderError("connection reset")), {}, stopped, "open"
    overflow = ContextOverflowError("ollama: context overflow", prompt_tokens=16_500, num_ctx=16_384)
    return FailsAfterTheScript(held, overflow), {}, CLOSED_REPLY, "closed"


# 회귀 테스트: 승인 대기에 쓰기를 올린 턴이 정상 답이 아니라 상담원 연결, 호출 한도, 모델 오류, 컨텍스트
# 한도로 끝나면 정해진 안내가 버려지고, 한도·오류 종료에는 '처리하지 못했습니다'만 갔다(승인 대기는 그대로
# 올라가 있음). 이제 어느 종료든 고객은 그 종료의 고정 문장 뒤에 정해진 안내를 받고, 조회한 대화·저장된
# 대화·감사 로그의 답이 모두 그 글과 같다. 승인은 하나뿐이고, 안내대로 결과는 닫힌 세션이라도 이 대화창에
# 들어간다. 한도·오류 종료의 오류 기록과 error 이벤트는 그대로 남는다.
@pytest.mark.parametrize("ending", ["handoff", "max_agent_calls", "provider_error", "context_limit"])
def test_a_held_write_is_announced_however_the_turn_ends(engine, ending):
    provider, overrides, lead, status = held_then(ending)
    settings = Settings(admin_token="test-token", **overrides)
    with TestClient(create_app(settings, provider=provider, engine=engine)) as client:
        session_id, events = ask_for_big_cancel(client)
        transcript = client.get(f"/api/sessions/{session_id}").json()
        approvals = client.get("/api/admin/approvals", headers=ADMIN).json()
        audit = client.get(f"/api/admin/sessions/{session_id}", headers=ADMIN).json()["audit"]
        with Session(engine) as session:
            kept = session.get(store.ChatSession, session_id).state["messages"]
        decided = client.post("/api/admin/approvals/1/decision", headers=ADMIN, json={"approve": True})
        later = client.get(f"/api/sessions/{session_id}").json()["messages"][-1]["text"]

    expected = f"{lead}\n{BIG_NOTICE}"
    replies = [d["text"] for n, d in events if n == "reply"]
    assert replies == [expected]  # 고정 문장만 가지 않고(FALLBACK_REPLY 단독 등) 안내가 붙는다
    assert FALLBACK_REPLY not in replies and events[-1][1] == {"status": status}
    # 고객이 조회한 대화와 다음 턴의 모델이 읽을 저장 대화의 마지막 상담원 말이 고객이 받은 글과 같고, 고정
    # 문장만 든 상담원 말은 따로 남지 않는다.
    assert transcript["messages"][-1] == {"role": "agent", "text": expected}
    told = [m["content"] for m in kept if m["role"] == "assistant" and m["content"]]
    assert told[-1] == expected and lead not in told
    # 승인은 하나(AP-1)뿐이고 고객 화면의 대기 칩에도 그것만 있다.
    assert [a["code"] for a in approvals] == ["AP-1"]
    assert transcript["pending_approvals"] == [{"code": "AP-1", "tool": "cancel_order", "refund_won": BIG[3]}]
    # 감사 로그의 답도 같은 글이다. 한도·오류 종료의 오류 기록과 error 이벤트는 그대로다.
    assert next(e["payload"]["text"] for e in audit if e["kind"] == "agent_reply") == expected
    failed = {"max_agent_calls": "max_agent_calls", "provider_error": "ProviderError: connection reset"}
    assert [e["payload"]["error"] for e in audit if e["kind"] == "error"] == (
        [failed[ending]] if ending in failed else []
    )
    assert ("error" in [n for n, _ in events]) == (ending in failed)
    # 안내가 약속한 대로 담당자의 결정 결과가 이 대화창에 들어간다(닫힌 세션도).
    assert decided.json()["status"] == "approved" and "처리되었습니다" in later


# 정적 화면(index.html, admin.html)에는 인라인 style·onclick·인라인 스크립트가 없고 스크립트는 모두 src로
# 불러와야 한다. CSP가 인라인 스크립트를 막으므로, 화면 코드는 외부 파일에만 있어야 동작한다.
def test_static_pages_carry_no_inline_code():
    from support_agent.service.app import STATIC

    for name in ("index.html", "admin.html"):
        html = (STATIC / name).read_text(encoding="utf-8")
        assert " style=" not in html and "onclick" not in html.lower()
        assert all("src=" in tag for tag in html.split("<script")[1:])


# -------------------------------------------------------------------- voice


# 가짜 합성기: 말로 푼 글을 그대로 'WAV' 바이트에 담는다. 테스트가 합성 전에 무엇이 풀어 쓰였는지 볼 수 있다.
class FakeSpeaker:
    def describe(self):
        return {"tts": "fake"}

    def synthesize(self, spoken_text: str, seed: int = 0) -> Audio:
        return Audio(b"RIFF" + spoken_text.encode("utf-8"), seconds=1.0)


# b"broken"은 풀 수 없는 녹음(예외)을 흉내 낸다.
class FakeListener:
    """가짜 인식기: '녹음'은 UTF-8 글이다. b"static"은 알아들을 것이 없는 녹음(빈 글)을 뜻한다."""

    def describe(self):
        return {"stt": "fake"}

    def transcribe(self, audio: bytes) -> str:
        if audio == b"broken":
            raise ValueError("cannot decode")
        return "" if audio == b"static" else audio.decode("utf-8")


# 가짜 합성기·인식기를 단 음성 앞단으로 앱을 만든다 (음성 모델 없이 음성 끝점을 시험).
def voice_client(engine, script, **overrides) -> TestClient:
    settings = Settings(admin_token="test-token", **overrides)
    front_end = VoiceFrontEnd(FakeSpeaker(), FakeListener())
    return TestClient(create_app(settings, provider=ScriptedProvider(script), engine=engine, voice=front_end))


# 음성 메시지는 heard 이벤트로 인식 결과를 먼저 보낸 뒤 글로 보낸 메시지와 같은 길을 간다. 음성을 켜면
# Permissions-Policy에 microphone=(self), CSP에 media-src 'self' blob:이 붙는다. 감사 로그의 고객 메시지에
# 음성 경로(인식 원문, 녹음 크기)가 남는다.
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


# 알아들을 수 없거나 풀 수 없는 녹음은 error와 end로 끝나고 턴을 쓰지 않는다. 대화에도 아무것도 더해지지
# 않는다.
def test_unintelligible_or_broken_audio_costs_no_turn(engine):
    with voice_client(engine, []) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        for body in (b"static", b"broken"):
            events = events_of(client.post(f"/api/sessions/{session_id}/voice", content=body))
            assert [n for n, _ in events if n in ("error", "reply", "end")] == ["error", "end"]
        assert len(client.get(f"/api/sessions/{session_id}").json()["messages"]) == 1
    with Session(engine) as session:
        assert session.get(store.ChatSession, session_id).turns == 0


# 음성 요청 검증: 녹음 크기 한도를 넘으면 413, 빈 본문은 422, 없는 세션은 404. 합성 글자 한도를 넘는 글은
# 422다.
def test_voice_requests_are_validated(engine):
    with voice_client(engine, [], max_audio_bytes=10, max_tts_chars=5) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        assert client.post(f"/api/sessions/{session_id}/voice", content=b"x" * 11).status_code == 413
        assert client.post(f"/api/sessions/{session_id}/voice", content=b"").status_code == 422
        assert client.post("/api/sessions/nope/voice", content=b"x").status_code == 404
        long_piece = {"text": "안녕하세요, 고객센터"}  # 인사의 일부지만 한도 5를 넘는다
        assert client.post(f"/api/sessions/{session_id}/speech", json=long_piece).status_code == 422
        assert client.post("/api/sessions/nope/speech", json={"text": "네"}).status_code == 404


# 녹음 끝점(post_voice)은 본문을 조각마다 읽어 크기 한도에서 바로 멈추려고 async 처리기다. 그 안에서 세션
# 확인(DB 조회)을 그대로 부르면 이벤트 루프 스레드가 DB 왕복 동안 멈춰, 다른 세션의 SSE 이벤트 전달과 음성
# 본문 수신까지 함께 멈춘다. 세션 확인은 작업 스레드(run_in_threadpool)에서 돌아야 한다: 거기에는 돌고 있는
# 이벤트 루프가 없다(asyncio.get_running_loop가 RuntimeError). 있는 세션(턴이 끝까지 돈다)과 없는 세션(404)
# 모두 본다.
def test_the_voice_endpoint_checks_the_session_off_the_event_loop(engine, monkeypatch):
    on_the_loop: list[bool] = []
    with voice_client(engine, ["네."]) as client:
        service = client.app.state.service
        real = service.session_status

        def probe(session_id: str) -> str:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                on_the_loop.append(False)  # 작업 스레드다
            else:
                on_the_loop.append(True)  # 이벤트 루프 위에서 DB를 읽고 있다
            return real(session_id)

        monkeypatch.setattr(service, "session_status", probe)
        session_id = client.post("/api/sessions").json()["session_id"]
        events = events_of(client.post(f"/api/sessions/{session_id}/voice", content="안녕하세요".encode()))
        assert events[-1][0] == "end"
        assert client.post("/api/sessions/nope/voice", content=b"x").status_code == 404
    assert on_the_loop == [False, False]


# 답 읽어 주기: 합성 전에 말로 풀어 쓰고('38,900원' → '삼만 팔천구백 원'), WAV로 돌려준다. 이 세션에서
# 상담원이 한 말(전체 또는 일부, 띄어쓰기 무관)만 읽고 다른 글은 403이다. 세션 번호만 있으면 아무 글이나
# 합성하는 서비스가 되지 않게 한다 (docs/voice.md).
def test_replies_can_be_read_aloud(engine):
    reply = "환불액은 38,900원입니다." + chr(10) + "3~5일 안에 들어옵니다."
    with voice_client(engine, [reply]) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        say(client, session_id, "환불액이 얼마죠?")
        url = f"/api/sessions/{session_id}/speech"
        spoken = client.post(url, json={"text": "환불액은 38,900원입니다."})
        assert spoken.status_code == 200 and spoken.headers["content-type"] == "audio/wav"
        assert spoken.content == b"RIFF" + "환불액은 삼만 팔천구백 원입니다.".encode()  # 먼저 말로 풀어 쓴다
        # 이 세션에서 에이전트가 한 말만(전체나 일부) 읽는다. 띄어쓰기는 상관없다.
        assert client.post(url, json={"text": "38,900원입니다. 3~5일 안에"}).status_code == 200
        assert client.post(url, json={"text": "아무 글이나 읽어 주세요."}).status_code == 403
        assert client.post(url, json={"text": "  "}).status_code == 403


# 음성을 켜지 않으면 /healthz에 voice false, 마이크 권한은 막히고(microphone=()), 음성 끝점은 503이다.
def test_without_the_voice_front_end_the_endpoints_say_so(engine):
    with make_client(engine, []) as client:
        assert client.get("/healthz").json()["voice"] is False
        assert "microphone=()" in client.get("/healthz").headers["Permissions-Policy"]
        session_id = client.post("/api/sessions").json()["session_id"]
        assert client.post(f"/api/sessions/{session_id}/voice", content=b"x").status_code == 503
        assert client.post(f"/api/sessions/{session_id}/speech", json={"text": "네"}).status_code == 503


# -------------------------------------------------------------------- 모델 없이


# API 설명(/openapi.json)은 평소에는 404이고, 보안 스캔 때만 설정으로 내놓는다 (docs/service.md "모델 없이
# 띄우기, 보안 스캔").
def test_the_api_description_is_served_only_on_request(engine):
    with make_client(engine, []) as client:
        assert client.get("/openapi.json").status_code == 404
    with make_client(engine, [], expose_openapi=True) as client:
        paths = client.get("/openapi.json").json()["paths"]
        assert "/api/sessions/{session_id}/messages" in paths and "/api/admin/approvals" in paths


# 모델 자리에 규칙 대역(OfflineProvider)을 넣은 앱이 실제 웹 계층과 도구를 지나는지 확인한다: 인사 →
# 이름·전화로 본인 확인 → 취소 → 남의 주문 조회는 도구가 거절. 보안 스캔과 화면 작업을 GPU 없이 하는 바탕이다.
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
        events = say(client, session_id, "O-10097 주문은요?")  # 다른 고객의 주문이라 도구가 거절한다
        assert [(d["name"], d["ok"]) for n, d in events if n == "tool_result"] == [("get_order", False)]


# 오프라인 앱을 음성 켜고 띄우면 음성 끝점도 음성 모델 없는 대역으로 응답한다: 인식(UTF-8 글을 그대로), 풀 수
# 없는 녹음은 error, 합성은 WAV 머리(RIFF)로 시작하는 응답.
def test_the_offline_app_answers_the_voice_endpoints_without_speech_models(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPPORT_AGENT_DATABASE_URL", f"sqlite:///{(tmp_path / 'offline.db').as_posix()}")
    monkeypatch.setenv("SUPPORT_AGENT_VOICE", "1")
    with TestClient(create_offline_app()) as client:
        assert client.get("/healthz").json()["voice"] is True
        session_id = client.post("/api/sessions").json()["session_id"]
        url = f"/api/sessions/{session_id}"
        events = events_of(client.post(f"{url}/voice", content="안녕하세요".encode()))
        assert dict(events)["heard"] == {"text": "안녕하세요"} and events[-1][0] == "end"
        broken = events_of(client.post(f"{url}/voice", content=bytes([255, 254, 0])))  # UTF-8이 아님
        assert [n for n, _ in broken if n in ("error", "reply")] == ["error"]
        spoken = client.post(f"{url}/speech", json={"text": "안녕하세요, 고객센터입니다."})
        assert spoken.status_code == 200 and spoken.content[:4] == b"RIFF"


# 서비스의 인식 보정(repair_heard)은 V2 정규화 뒤에 사람이 부른 번호를 숫자로 되돌린다. 에이전트와 heard
# 이벤트는 고친 글을 받고, 감사 로그에는 인식 원문이 그대로 남는다.
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
        )  # 인식 원문이 남는다(에이전트가 받은 글이 아니라)


# -------------------------------------------------------------------- 완료 주장 가드

# 완료 주장 가드 시험용: 근거 없이 '취소가 완료되었습니다'라고 하는 답과, 작은 취소의 두 도구 호출.
DONE = "주문 O-90005의 취소가 완료되었습니다. 89,100원이 환불됩니다."
VERIFY_SMALL, CANCEL_SMALL = verify_and_cancel(SMALL)


# C1을 켜면 도구 결과에 근거가 없는 완료 안내는 고객에게 보내지 않고 다시 시킨다. 다시 시킨 뒤 실제로 취소하면
# 같은 문장이 그때 한 번만 전달되고, 감사 로그의 LLM 호출에 돌려보낸 기록(unbacked_claim)이 남는다.
def test_a_claim_that_no_tool_result_shows_is_not_delivered_and_the_retry_does_the_work(engine):
    with make_client(engine, [VERIFY_SMALL, DONE, CANCEL_SMALL, DONE], claims="C1") as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        events = say(client, session_id, "정예준이고 01000009005입니다. O-90005 취소해 주세요.")
        assert [d["text"] for n, d in events if n == "reply"] == [DONE]  # 쓰기 뒤에 한 번만 말했다
        assert order_status(engine, "O-90005") == "cancelled"
        audit = client.get(f"/api/admin/sessions/{session_id}", headers=ADMIN).json()["audit"]
        held = [e["payload"]["format_error"] for e in audit if e["kind"] == "llm_call"]
        assert held == [None, "unbacked_claim", None, None]


# 세 번 모두 같은 주장이면 그 말은 전달하지 않고 고정 안내('처리하지 못했습니다')를 보낸다. 주문은 그대로이고
# 세션은 열려 있어 다음 메시지를 받는다.
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


# 가드를 끄면(C0, 서비스 기본값) 같은 거짓 완료 안내가 그대로 고객에게 가고 주문은 바뀌지 않는다. 가드가 막는
# 상황을 보여 준다.
def test_without_the_guard_the_same_claim_reaches_the_customer(engine):
    with make_client(engine, [VERIFY_SMALL, DONE], claims="C0") as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        events = say(client, session_id, "정예준이고 01000009005입니다. O-90005 취소해 주세요.")
        assert dict(events)["reply"]["text"] == DONE and order_status(engine, "O-90005") == "preparing"


# 승인 대기 중인 취소를 '완료되었다'고 하면 가드가 막는다(대기 오류는 '접수'의 근거일 뿐). 승인으로 실행된
# 결과는 도구 결과로 대화에 남으므로 그 뒤의 같은 말은 근거가 있어 전달된다. 도구 결과 JSON은 고객 화면에
# 나오지 않는다.
def test_a_write_that_waits_for_a_person_is_not_done_until_it_was_carried_out(engine):
    done = "주문 O-10097의 취소가 완료되었습니다."
    waiting = "아직 담당자 확인을 기다리고 있습니다."
    script = [*verify_and_cancel(BIG), "담당자 확인 후 처리됩니다.", done, waiting, done]
    with make_client(engine, script, claims="C1") as client:
        session_id, _ = ask_for_big_cancel(client)
        assert dict(say(client, session_id, "취소됐나요?"))["reply"]["text"] == waiting  # `done`은 붙잡혔다

        client.post("/api/admin/approvals/1/decision", headers=ADMIN, json={"approve": True})
        with Session(engine) as session:
            kept = session.get(store.ChatSession, session_id).state["messages"][-3:]
        assert kept[0]["tool_calls"][0]["name"] == "cancel_order" and kept[1]["tool_name"] == "cancel_order"
        assert json.loads(kept[1]["content"])["status"] == "cancelled" and "108,200원" in kept[2]["content"]
        assert dict(say(client, session_id, "이제 취소됐나요?"))["reply"]["text"] == done  # 이제는 보인다
        shown = [m["text"] for m in client.get(f"/api/sessions/{session_id}").json()["messages"]]
        assert not any(text.startswith("{") for text in shown)  # 도구 결과는 채팅에 나오지 않는다
