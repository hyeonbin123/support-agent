"""Regressions for the findings of the read-only review of stages 3 to 5 (work/relay, round 1, T7)."""

from __future__ import annotations

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


def test_the_audit_row_of_a_successful_write_is_in_the_same_transaction(engine):  # noqa: F811
    with make_client(engine, [*verify_and_cancel(SMALL), "취소했습니다."]) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        say(client, session_id, "정예준, 01000009005입니다. O-90005 취소해 주세요.")
        audit = client.get(f"/api/admin/sessions/{session_id}", headers=ADMIN).json()["audit"]
    calls = [e["payload"] for e in audit if e["kind"] == "tool_call"]
    assert [(c["name"], c["ok"]) for c in calls] == [("find_customer", True), ("cancel_order", True)]
    assert len([e for e in audit if e["kind"] == "tool_call"]) == 2  # written once, not again afterwards


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


# -------------------------------------------------------------------- 4-6: limits of the web layer


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


def test_a_recognised_sentence_is_never_cut(engine):  # noqa: F811
    with voice_client(engine, [], max_message_chars=20) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        recording = ("주문을 취소해 주세요. 아니, 다시 생각해 보니 취소하지 마세요.").encode()
        events = events_of(client.post(f"/api/sessions/{session_id}/voice", content=recording))
        assert [n for n, _ in events if n in ("heard", "error", "reply", "end")] == ["error", "end"]
    with Session(engine) as session:
        assert session.get(store.ChatSession, session_id).turns == 0


def test_a_recording_that_is_too_long_is_refused(engine, monkeypatch):  # noqa: F811
    with voice_client(engine, []) as client:
        front_end = client.app.state.voice

        def too_long(audio: bytes) -> str:
            raise AudioTooLongError("3600 s of audio")

        monkeypatch.setattr(front_end.listener, "transcribe", too_long)
        session_id = client.post("/api/sessions").json()["session_id"]
        events = events_of(client.post(f"/api/sessions/{session_id}/voice", content=b"x"))
        assert "너무 깁니다" in dict(events)["error"]["message"]


def test_limits_the_request_model_cannot_honour_are_refused_at_start():
    with pytest.raises(ValueError):
        Settings(max_message_chars=5000)
    with pytest.raises(ValueError):
        Settings(max_tts_chars=0)


# -------------------------------------------------------------------- 8-9: MCP


@pytest.fixture
def anyio_backend():
    return "asyncio"


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


def test_same_setup_names_every_difference_but_the_ignored_axis(tmp_path):
    text = write_run(tmp_path, "text")
    voice = write_run(tmp_path, "voice", voice="V1")
    assert analyze.setup_differences(text, voice) == []
    other = write_run(tmp_path, "other", voice="V1", policy="P1", trials=(0,))
    found = analyze.setup_differences(text, other)
    assert "config.policy: 'P0' != 'P1'" in found
    assert "judged trials of dev-001: [0, 1] != [0]" in found
    assert analyze.setup_differences(text, other, ignore=("voice", "policy")) == [found[-1]]


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
