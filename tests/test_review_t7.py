"""Regressions for the findings of the read-only review of stages 3 to 5 (work/relay, round 1, T7)."""

# [테스트 묶음] 3~5단계(서비스, MCP, 음성)를 읽기 전용으로 검토한 결과에서 나온 지적마다 회귀
# 테스트를 둔다. 구역 제목의 번호는 지적 번호다: 1-3 한 트랜잭션, 4-6 웹 계층의 한도, 8-9 MCP, 10-13 말로 풀어
# 쓰기와 지표, 14-15 측정 절차, 그리고 보안 스캔(HawkScan)이 찾은 것.
# [왜 중요한가] 이 지적들은 돈이 나가는 쓰기와 감사 로그의 일관성, 동시 승인, 자원 한도, 측정 규칙(시험용
# 기록을 보지 않는다)에 걸린 것이다. 고친 동작이 다시 깨지면 서비스가 감사 기록 없이 주문을 바꾸거나, 측정
# 절차가 어겨질 수 있다.
# [방법] tests/test_service.py의 fixture(마이그레이션한 SQLite 파일)와 도우미(대본, 채팅 클라이언트, 큰·작은
# 취소)를 그대로 가져와 쓴다.

from __future__ import annotations

# threading: 동시 결정과 동시 턴을 흉내 낸다. TestClient: FastAPI 앱을 실제 서버 없이 부른다. test_service의
# engine·seeded_template은 fixture라 import해 두어야 이 파일에서도 쓸 수 있다(F401·F811 경고는 그래서 끈다).
import json
import threading

import pytest
from conftest import NOW
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_service import (
    ADMIN,
    BIG,
    SMALL,
    ask_for_big_cancel,
    engine,  # noqa: F401 - fixture
    events_of,
    make_client,
    order_status,
    say,
    seeded_template,  # noqa: F401 - fixture
    verify_and_cancel,
    voice_client,
)
from toy_tools import TOY_REGISTRY

from support_agent import analyze, db
from support_agent.chat import ScriptedProvider
from support_agent.mcp_server import MAX_ARGUMENT_BYTES, build_server
from support_agent.service import core, store
from support_agent.service.app import create_app
from support_agent.service.core import ApprovalStateError, ChatService
from support_agent.service.settings import Settings
from support_agent.toolkit import ToolBugError, ToolContext, ToolError, execute
from support_agent.voice import metrics
from support_agent.voice.speech import AudioTooLongError
from support_agent.voice.verbalize import verbalize

# -------------------------------------------------------------------- 1-3: one transaction


# execute()의 on_success 훅은 도구와 같은 트랜잭션 안에서 돈다: 훅이 받은 정보(도구, 정규화한 인자, 결과)가
# 맞고, 훅이 바꾼 행이 도구의 변경과 함께 커밋된다. 서비스는 이 훅으로 감사 행과 승인 표시를 도구의 변경과
# 같은 트랜잭션에 넣는다.
def test_the_on_success_hook_shares_the_tools_transaction(tiny_engine):
    engine = db.memory_engine(tiny_engine)  # noqa: F811
    ctx = ToolContext(now=NOW)
    seen = []

    def note(session, spec, clean, answer):
        seen.append((spec.name, clean, answer["status"]))
        session.get(db.Order, "O-2").ship_recipient = "hook"

    assert execute(TOY_REGISTRY, engine, ctx, "toy_cancel_order", {"order_id": "O-1"}, on_success=note).ok
    assert seen == [("toy_cancel_order", {"order_id": "O-1"}, "cancelled")]
    orders = {o["id"]: o for o in db.dump_db(engine)["orders"]}
    assert (orders["O-1"]["status"], orders["O-2"]["ship_recipient"]) == ("cancelled", "hook")


# 훅이 ToolError로 거절하면(승인 관문의 approval_required) 도구의 변경이 롤백되고 정상적인 도구 오류로
# 돌아간다. 훅에서 예상 못 한 예외가 나면 ToolBugError이고 역시 롤백된다. 어느 경우에도 DB는 그대로다.
def test_a_refusal_or_a_failure_in_the_hook_undoes_the_tool(tiny_engine):
    engine = db.memory_engine(tiny_engine)  # noqa: F811
    ctx = ToolContext(now=NOW)
    before = db.dump_db(engine)

    def refuse(session, spec, clean, answer):
        raise ToolError("approval_required", "held")

    result = execute(TOY_REGISTRY, engine, ctx, "toy_cancel_order", {"order_id": "O-1"}, on_success=refuse)
    assert (result.ok, result.error_code, result.args) == (False, "approval_required", {"order_id": "O-1"})
    assert db.dump_db(engine) == before

    def broken(session, spec, clean, answer):
        raise RuntimeError("audit table is gone")

    with pytest.raises(ToolBugError):
        execute(TOY_REGISTRY, engine, ctx, "toy_cancel_order", {"order_id": "O-1"}, on_success=broken)
    assert db.dump_db(engine) == before


# 채팅에서 취소의 감사 행을 만들 때 오류가 나면 취소도 일어나지 않아야 한다(주문이 preparing 그대로). 감사
# 기록 없이 주문만 바뀐 상태가 생기지 않는다는 서비스의 약속이다 (docs/service.md).
def test_no_change_of_the_shop_without_its_audit_row(engine, monkeypatch):  # noqa: F811
    """The audit row is written in the tool's transaction: when it cannot be written, nothing is cancelled."""

    class NoAudit:
        def __init__(self, **fields):
            if fields.get("kind") == "tool_call" and fields["payload"]["name"] == "cancel_order":
                raise RuntimeError("audit storage failed")
            self.row = store.AuditEvent(**fields)

    real = core.AuditEvent
    monkeypatch.setattr(core, "AuditEvent", lambda **fields: NoAudit(**fields).row)
    script = [*verify_and_cancel(SMALL), "취소했습니다."]
    with make_client(engine, script) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        events = say(client, session_id, "정예준, 01000009005입니다. O-90005 취소해 주세요.")
        assert "error" in [name for name, _ in events]
        assert order_status(engine, "O-90005") == "preparing"
    monkeypatch.setattr(core, "AuditEvent", real)


# 성공한 쓰기의 감사 행은 트랜잭션 안에서 한 번만 쓰이고, 뒤에서 다시 쓰여 중복되지 않는다.
def test_the_audit_row_of_a_successful_write_is_in_the_same_transaction(engine):  # noqa: F811
    with make_client(engine, [*verify_and_cancel(SMALL), "취소했습니다."]) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        say(client, session_id, "정예준, 01000009005입니다. O-90005 취소해 주세요.")
        audit = client.get(f"/api/admin/sessions/{session_id}", headers=ADMIN).json()["audit"]
    calls = [e["payload"] for e in audit if e["kind"] == "tool_call"]
    assert [(c["name"], c["ok"]) for c in calls] == [("find_customer", True), ("cancel_order", True)]
    assert len([e for e in audit if e["kind"] == "tool_call"]) == 2  # written once, not again afterwards


# 승인 관문은 도구가 실제로 계산한 환불액을 그 트랜잭션 안에서 본다. 큰 취소는 approval_required로 끝나고 DB는
# 그대로이며, 감사 로그에 승인 번호(AP-1)가 담긴 거절 결과가 남는다. 미리 계산해 보는 단계가 없으니 '확인할
# 때와 실행할 때 금액이 다른' 틈이 없다.
def test_the_gate_reads_the_refund_the_tool_really_computed(engine):  # noqa: F811
    """The amount is checked inside the transaction of the real call, so nothing can change in between."""
    with make_client(engine, [*verify_and_cancel(BIG), "담당자 확인 후 처리됩니다."]) as client:
        before = db.dump_db(engine)
        session_id, events = ask_for_big_cancel(client)
        assert [d["error_code"] for n, d in events if n == "tool_result"][-1] == "approval_required"
        assert db.dump_db(engine) == before
        audit = client.get(f"/api/admin/sessions/{session_id}", headers=ADMIN).json()["audit"]
        held = [
            e["payload"] for e in audit if e["kind"] == "tool_call" and e["payload"]["name"] == "cancel_order"
        ]
        assert [(c["ok"], c["error_code"]) for c in held] == [(False, "approval_required")]
        assert "AP-1" in held[0]["content"]


# 두 서비스 프로세스가 같은 승인을 같은 순간에 승인해도 한쪽만 실행하고 다른 쪽은 거절된다(조건부 UPDATE 한
# 번으로 선점). 실행된 취소는 하나이고 승인 결과의 환불액이 맞다. PostgreSQL에서의 같은 시험은
# test_postgres.py에 있다.
def test_of_two_deciders_one_carries_the_write_out(engine):  # noqa: F811
    """Two service processes share the database; the claim is one conditional UPDATE."""
    with make_client(engine, [*verify_and_cancel(BIG), "담당자 확인 후 처리됩니다."]) as client:
        ask_for_big_cancel(client)
    services = [ChatService(Settings(), engine, ScriptedProvider([])) for _ in range(2)]
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def decide(service: ChatService) -> None:
        barrier.wait()
        try:
            outcomes.append(service.decide(1, approve=True, by="kim")["status"])
        except ApprovalStateError:
            outcomes.append("refused")

    threads = [threading.Thread(target=decide, args=(service,)) for service in services]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert sorted(outcomes) == ["approved", "refused"]
    with Session(engine) as session:
        approval = session.get(store.Approval, 1)
        assert (approval.status, json.loads(approval.result)["refund_won"]) == ("approved", BIG[3])
        calls = session.scalars(select(store.AuditEvent).where(store.AuditEvent.kind == "tool_call")).all()
        carried_out = [e.payload for e in calls if e.payload.get("approval") == "AP-1"]
        assert [(c["name"], c["ok"]) for c in carried_out] == [("cancel_order", True)]
    assert order_status(engine, "O-10097") == "cancelled"


# 어떤 승인 번호로 실행되어 성공한 도구 호출의 감사 기록만 모은다.
def carried_out(engine, code: str) -> list[dict]:  # noqa: F811
    with Session(engine) as session:
        calls = session.scalars(select(store.AuditEvent).where(store.AuditEvent.kind == "tool_call")).all()
        return [e.payload for e in calls if e.payload.get("approval") == code and e.payload["ok"]]


# 승인 실행 중에 감사 행을 쓰지 못하면 도구 트랜잭션이 롤백되어 아무것도 바뀌지 않았으므로, 승인은 pending으로
# 돌아가 다시 결정할 수 있어야 한다. 오류는 감사 로그에 승인 번호와 함께 남고, 고친 뒤 다시 승인하면 취소가 한
# 번만 실행된다.
def test_a_failed_approved_write_gives_the_claim_back(engine, monkeypatch):  # noqa: F811
    """The tool's transaction was rolled back, so nothing changed: the approval can be decided again."""
    with make_client(engine, [*verify_and_cancel(BIG), "담당자 확인 후 처리됩니다."]) as client:
        session_id, _ = ask_for_big_cancel(client)
    service = ChatService(Settings(), engine, ScriptedProvider([]))

    real = core.AuditEvent

    def failing(**fields):
        if fields.get("kind") == "tool_call" and fields["payload"]["name"] == "cancel_order":
            raise RuntimeError("audit storage failed")
        return real(**fields)

    monkeypatch.setattr(core, "AuditEvent", failing)
    with pytest.raises(ToolBugError):
        service.decide(1, approve=True, by="kim")
    with Session(engine) as session:
        approval = session.get(store.Approval, 1)
        assert (approval.status, approval.decided_by) == ("pending", None)
    assert order_status(engine, "O-10097") == "preparing"
    assert [a["code"] for a in service.transcript(session_id)["pending_approvals"]] == ["AP-1"]

    monkeypatch.setattr(core, "AuditEvent", real)
    errors = [e["payload"] for e in service.audit(session_id) if e["kind"] == "error"]
    assert [e["approval"] for e in errors] == ["AP-1"] and "ToolBugError" in errors[0]["error"]
    assert service.decide(1, approve=True, by="kim")["status"] == "approved"
    assert order_status(engine, "O-10097") == "cancelled"
    assert [c["name"] for c in carried_out(engine, "AP-1")] == ["cancel_order"]


# 반대로 도구의 커밋이 끝난 뒤(이미 approved로 기록됨)에 오류가 나면 승인을 되돌리지 않는다. 되돌리면 같은
# 취소가 두 번 실행될 수 있다.
def test_a_write_that_committed_is_not_given_back(engine, monkeypatch):  # noqa: F811
    """An error after the commit (the tool's transaction already said "approved") leaves it approved."""
    with make_client(engine, [*verify_and_cancel(BIG), "담당자 확인 후 처리됩니다."]) as client:
        ask_for_big_cancel(client)
    service = ChatService(Settings(), engine, ScriptedProvider([]))
    real = core.execute

    def commits_then_fails(*args, **kwargs):
        real(*args, **kwargs)
        raise ToolBugError("the connection dropped after the commit")

    monkeypatch.setattr(core, "execute", commits_then_fails)
    with pytest.raises(ToolBugError):
        service.decide(1, approve=True, by="kim")
    with Session(engine) as session:
        assert session.get(store.Approval, 1).status == "approved"
    assert order_status(engine, "O-10097") == "cancelled"


# -------------------------------------------------------------------- 4-6: limits of the web layer


# 동시에 도는 턴 수 한도: 한도 1에서 첫 턴이 모델 호출에 머무는 동안 다른 세션의 메시지는 503과 Retry-After:
# 5로 거절된다. 첫 턴이 끝나면 자리가 돌아와 다음 메시지가 처리된다.
def test_turns_over_the_limit_are_turned_away(engine):  # noqa: F811
    release, entered = threading.Event(), threading.Event()

    class Slow(ScriptedProvider):
        def chat(self, *args, **kwargs):
            entered.set()
            release.wait(timeout=30)
            return super().chat(*args, **kwargs)

    settings = Settings(max_concurrent_turns=1)
    with TestClient(create_app(settings, provider=Slow(["네."]), engine=engine)) as client:
        first, second = (client.post("/api/sessions").json()["session_id"] for _ in range(2))
        done: list[list] = []
        worker = threading.Thread(target=lambda: done.append(say(client, first, "안녕하세요")))
        worker.start()
        assert entered.wait(timeout=30)
        refused = client.post(f"/api/sessions/{second}/messages", json={"text": "안녕하세요"})
        assert refused.status_code == 503 and refused.headers["Retry-After"] == "5"
        release.set()
        worker.join(timeout=30)
        assert done[0][-1][1] == {"status": "open"}
        assert say(client, second, "안녕하세요")[-1][0] == "end"  # the slot came back


# 인식된 문장이 메시지 길이 한도를 넘으면 자르지 않고 다시 말해 달라고 하며, 턴을 쓰지 않는다. 문장 끝의
# '취소하지 마세요'가 잘리면 뜻이 뒤집히기 때문이다 (docs/service.md "한도"). 가짜 인식기는 녹음 바이트를 글로
# 그대로 돌려준다.
def test_a_recognised_sentence_is_never_cut(engine):  # noqa: F811
    with voice_client(engine, [], max_message_chars=20) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        recording = ("주문을 취소해 주세요. 아니, 다시 생각해 보니 취소하지 마세요.").encode()
        events = events_of(client.post(f"/api/sessions/{session_id}/voice", content=recording))
        assert [n for n, _ in events if n in ("heard", "error", "reply", "end")] == ["error", "end"]
    with Session(engine) as session:
        assert session.get(store.ChatSession, session_id).turns == 0


# 인식기가 AudioTooLongError를 내면 '녹음이 너무 깁니다' 안내가 오류 이벤트로 간다.
def test_a_recording_that_is_too_long_is_refused(engine, monkeypatch):  # noqa: F811
    with voice_client(engine, []) as client:
        front_end = client.app.state.voice

        def too_long(audio: bytes) -> str:
            raise AudioTooLongError("3600 s of audio")

        monkeypatch.setattr(front_end.listener, "transcribe", too_long)
        session_id = client.post("/api/sessions").json()["session_id"]
        events = events_of(client.post(f"/api/sessions/{session_id}/voice", content=b"x"))
        assert "너무 깁니다" in dict(events)["error"]["message"]


# 지킬 수 없는 설정(메시지 한도 4,000자 초과, 음성 합성 글자 한도 0)은 서비스가 시작할 때 거절한다. 실행 중에
# 이상하게 동작하기보다 처음에 멈추는 편이 낫다.
def test_limits_the_request_model_cannot_honour_are_refused_at_start():
    with pytest.raises(ValueError):
        Settings(max_message_chars=5000)
    with pytest.raises(ValueError):
        Settings(max_tts_chars=0)


# -------------------------------------------------------------------- 8-9: MCP


# anyio 테스트의 백엔드를 asyncio로 정한다.
@pytest.fixture
def anyio_backend():
    return "asyncio"


# MCP 경계는 한 호출의 인자를 JSON 20,000바이트까지만 받는다. 넘으면 아무것도 실행하기 전에(세션도 만들지
# 않고) '너무 큽니다'로 거절한다. 도구의 자유 서술 인자에는 길이 제한이 없어서 MCP 쪽에서 막는다 (docs/mcp.md
# "크기 제한").
@pytest.mark.anyio
async def test_mcp_refuses_oversized_arguments_before_anything_runs(engine):  # noqa: F811
    from mcp import Client

    service = ChatService(Settings(), engine, ScriptedProvider([]))
    async with Client(build_server(service, "write")) as client:
        body = "가" * MAX_ARGUMENT_BYTES
        result = await client.call_tool("create_ticket", {"category": "etc", "body": body})
        assert result.is_error and "너무 큽니다" in result.content[0].text
    with Session(engine) as session:
        assert session.scalars(select(store.ChatSession)).all() == []  # not even a session was made


# -------------------------------------------------------------------- 10-13: spoken forms and metrics


# 검토자가 찾은, 말로 잘못 풀던 표기들: +82와 괄호가 든 전화번호, 이메일의 %, 한 글자 영문 식별자(G는 그램이
# 아니라 글자), 음수 금액, 점으로 쓴 날짜, 16자리를 넘는 수.
@pytest.mark.parametrize(
    ("written", "spoken"),
    [
        ("+82 10-1234-5678입니다", "공일공, 일이삼사, 오육칠팔 입니다"),
        ("(010) 1234-5678", "공일공, 일이삼사, 오육칠팔"),
        ("a%b@example.com", "에이 퍼센트 비 골뱅이 이그잼플 닷컴"),
        ("G-123 주문", "지 다시 일이삼 주문"),
        ("-100원", "마이너스 백 원"),
        ("2026.09.13에", "이천이십육 년 구월 십삼일에"),
        ("12345678901234567원", "일이삼사오육칠팔구공일이삼사오육칠 원"),
    ],
)
def test_forms_the_reviewer_found_misread(written, spoken):
    assert verbalize(written) == spoken


# 엔티티 생존 판정에서 더 긴 값의 일부는 살아남은 것이 아니다 (O-12와 O-123, 1,000원과 11,000원, 1월 2일과
# 11월 2일, a@x.com과 ba@x.com, 전화번호 뒤에 숫자가 더 붙은 것). 같은 값의 다른 표기는 살아남은 것이다.
@pytest.mark.parametrize(
    ("kind", "entity", "text", "alive"),
    [
        ("id", "O-12", "O-123 주문이요", False),
        ("id", "O-12", "O-12이고요", True),
        ("amount", "1,000원", "11,000원입니다", False),
        ("amount", "1,000원", "1000 원입니다", True),
        ("date", "1월 2일", "11월 2일", False),
        ("date", "1월 2일", "1월2일에", True),
        ("email", "a@x.com", "ba@x.com", False),
        ("phone", "010-0000-9103", "010 0000 9103 2개", True),
        ("phone", "010-0000-9103", "0100000910321", False),
    ],
)
def test_an_entity_inside_a_longer_one_did_not_survive(kind, entity, text, alive):
    assert metrics.survived(kind, entity, text) is alive


# -------------------------------------------------------------------- 14-15: the measurement procedure


# analyze가 읽는 최소한의 실행 폴더(manifest.json과 episodes.jsonl)를 만든다. 과제 파일 이름·과제 번호
# 접두사·음성 조건·규정·시도 번호를 바꿔 가며 쓴다. 발화 하나는 주문 번호를 잃은(O-1 → 5-1) 음성 기록이다.
def write_run(root, name, *, tasks_file="dev", task_prefix="dev", voice="V0", policy="P0", trials=(0, 1)):
    run_dir = root / name
    run_dir.mkdir()
    config = {"model": "m", "policy": policy, "voice": voice}
    manifest = {
        "config": config,
        "trials": 2,
        "tasks_file": tasks_file,
        "task_sha256": {f"{task_prefix}-001": "abc"},
        "seed_hash": "s",
        "prompt_sha256": {"policy": "p"},
        "agent_provider": {"model": "m", "digest": "d", "ollama_version": "0.34.2"},
        "user_provider": {"model": "m"},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    lost = {"said": "O-1 취소", "heard": "5-1 취소", "text": "5-1 취소", "audio_seconds": 1.0,
            "tts_ms": 1.0, "stt_ms": 1.0, "cached": False}  # fmt: skip
    lines = [
        json.dumps({"task_id": f"{task_prefix}-001", "trial": t, "status": "completed", "voice": [lost]})
        for t in trials
    ]
    (run_dir / "episodes.jsonl").write_text("\n".join(lines), encoding="utf-8")
    return run_dir


# analyze same-setup은 두 실행을 짝지어도 되는지 본다: 음성 축만 다르면 차이가 없고, 규정이나 판정된 시도
# 번호가 다르면 그 차이를 모두 이름으로 보고한다. 무시할 축을 주면 그 축의 차이만 빠진다. V0을 다시 재지 않고
# 쓰는 조건을 눈이 아니라 코드로 확인하려는 것이다 (docs/experiments.md 5단계 "V0은 다시 재지 않는다").
def test_same_setup_names_every_difference_but_the_ignored_axis(tmp_path):
    text = write_run(tmp_path, "text")
    voice = write_run(tmp_path, "voice", voice="V1")
    assert analyze.setup_differences(text, voice) == []
    other = write_run(tmp_path, "other", voice="V1", policy="P1", trials=(0,))
    found = analyze.setup_differences(text, other)
    assert "config.policy: 'P0' != 'P1'" in found
    assert "judged trials of dev-001: [0, 1] != [0]" in found
    assert analyze.setup_differences(text, other, ignore=("voice", "policy")) == [found[-1]]


# V2 정규화 규칙은 개발용 기록만 보고 만든다는 측정 규칙을 코드로 막는다: 과제 파일 이름이나 과제 번호가
# 시험용이면 is_test_run이 알아보고, voice-worst 명령은 그런 실행을 거부한다. 개발용 실행에는 엔티티를 잃은
# 발화가 그대로 나온다.
def test_the_normaliser_cannot_be_built_from_test_records(tmp_path, monkeypatch, capsys):
    dev = write_run(tmp_path, "dev-v1", voice="V1")
    by_name = write_run(tmp_path, "renamed", tasks_file="eval_copy", task_prefix="test", voice="V1")
    by_file = write_run(tmp_path, "by-file", tasks_file="tasks/test.yaml", voice="V1")
    assert not analyze.is_test_run(dev) and analyze.is_test_run(by_name) and analyze.is_test_run(by_file)
    monkeypatch.setattr("sys.argv", ["analyze", "voice-worst", str(by_name)])
    with pytest.raises(SystemExit, match="development tasks"):
        analyze.main()
    monkeypatch.setattr("sys.argv", ["analyze", "voice-worst", str(dev)])
    analyze.main()
    assert "lost : ['O-1']" in capsys.readouterr().out


# -------------------------------------------------------------------- found by the security scan (HawkScan)


# HawkScan 1차 스캔이 찾은 결함: 승인 번호에 DB 정수 범위를 넘는 수를 넣으면 500이 났다. 이제 경로 인자에
# 범위를 두어 아주 큰 수, 0, 음수는 422이고, 범위 안이지만 없는 번호는 409다 (docs/service.md 스캔 표 1번).
def test_an_approval_number_beyond_the_database_range_is_a_client_error(engine):  # noqa: F811
    with make_client(engine, []) as client:
        for approval_id in ("92173292616343605102041285171235936212706764", "0", "-1"):
            answer = client.post(
                f"/api/admin/approvals/{approval_id}/decision", headers=ADMIN, json={"approve": False}
            )
            assert answer.status_code == 422, approval_id
        assert (
            client.post("/api/admin/approvals/2147483647/decision", headers=ADMIN, json={"approve": False})
        ).status_code == 409  # a valid number that names no approval
