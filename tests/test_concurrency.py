"""쓰기 도구의 행 잠금·재고 조건부 갱신(2026-10-15)이 SQLite 평가를 바꾸지 않는다는 것과, 서비스의 승인
대기열이 같은 대상의 재요청을 하나로 묶는다는 것을 시험한다. 동시성 자체는 tests/test_postgres.py가 실제
PostgreSQL의 두 트랜잭션으로 본다."""

# [테스트 묶음]
# (1) 평가 경로 그대로: 커밋된 reports/의 에피소드 기록에 남은 도구 호출을 지금 코드로 순서대로 다시 실행하면
#     도구 결과 글, 성공 여부, 에피소드가 끝난 DB의 해시가 기록과 같다. 바뀐 SQL을 쓰는
#     교환(request_exchange)이 든 에피소드 전부와 8단계 qwen3.5:4b 개발용 기록 전체를 CI에서 돈다. 기록
#     전체(36개 실행, 에피소드 3,814, 도구 호출 12,974)는 2026-10-15에 한 번 돌려 모두
#     같았다(docs/experiments.md 공통 규칙의 2026-10-15 메모).
# (2) SQLite에는 FOR UPDATE가 가지 않는다: 쓰기 도구가 SQLite에 보낸 SQL에 잠금 구문이 없다.
# (3) 승인 대기열: 같은 세션에서 같은 주문의 같은 쓰기를 사유만 바꿔 다시 요청해도 승인은 하나다.

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import event, func, select
from sqlalchemy.orm import Session
from test_service import (  # noqa: F401 - engine·seeded_template은 fixture
    ADMIN,
    BIG,
    engine,
    make_client,
    say,
    seeded_template,
)

from support_agent import db
from support_agent.chat import ToolCall
from support_agent.episode import caller_number
from support_agent.paths import REPORTS
from support_agent.seed import build_seed_engine
from support_agent.service import store
from support_agent.tasks import load_tasks
from support_agent.toolkit import ConversationState, ToolContext, execute
from support_agent.tools import build_registry

# CI에서 전부 다시 실행하는 기록: 8단계 qwen3.5:4b 개발용(P1, 서비스의 기본 구성).
FULL_REPLAY = ("20261003-200306-qwen3.5-4b-P1-R0-G0-F0-m2b",)


@cache
def seed_engine():
    return build_seed_engine()


@cache
def tasks_of(tasks_file: str) -> dict[str, Any]:
    return {task.id: task for task in load_tasks(Path(tasks_file).stem)}


def recorded_runs() -> list[str]:
    return sorted(p.name for p in REPORTS.iterdir() if (p / "episodes.jsonl").exists())


# 기록 하나의 에피소드 가운데 다시 실행할 것: FULL_REPLAY면 전부, 아니면 교환 도구를 부른 것만.
def episodes_to_replay(run: str) -> list[dict[str, Any]]:
    out = []
    for line in (REPORTS / run / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        episode = json.loads(line)
        if episode["status"] == "infra_error":
            continue
        if run in FULL_REPLAY or any(c["name"] == "request_exchange" for c in episode["tool_calls"]):
            out.append(episode)
    return out


# 에피소드의 도구 호출을 기록된 순서와 인자(raw_arguments) 그대로 새 DB 복사본에 실행한다. 문맥은 그 실행의
# 설정(P0/P1, V4의 발신 번호)과 과제의 고정 시각이다. 등록부는 그 실행의 V4 여부와 15단계 읽기 도구 출력
# 축(reads, 없는 옛 기록은 X0)으로 만든다(X1 기록의 get_order 결과에는 shipment가 있다). 모델은 부르지 않는다.
@pytest.mark.parametrize("run", recorded_runs())
def test_recorded_tool_calls_replay_to_the_same_results_and_database(run):
    manifest = json.loads((REPORTS / run / "manifest.json").read_text(encoding="utf-8"))
    seed = seed_engine()
    assert manifest["seed_hash"] == db.state_hash(db.dump_db(seed))
    config = manifest["config"]
    caller_id = config.get("voice") == "V4"
    registry = build_registry(caller_id=caller_id, reads=config.get("reads", "X0"))
    tasks = tasks_of(manifest["tasks_file"])
    for episode in episodes_to_replay(run):
        task = tasks[episode["task_id"]]
        shop = db.memory_engine(seed)
        ctx = ToolContext(
            now=task.now,
            enforce_policy=config.get("policy") == "P1",
            caller_phone=caller_number(task, seed) if caller_id else None,
            error_facts=config.get("errors") == "E1",  # 13단계 E1 실행은 거절 오류에 사실이 붙어 있다
        )
        where = (episode["task_id"], episode["trial"])
        for call in episode["tool_calls"]:
            result = execute(registry, shop, ctx, call["name"], call["raw_arguments"])
            assert (result.ok, result.content) == (call["ok"], call["content"]), (*where, call["name"])
        assert db.state_hash(db.dump_db(shop)) == episode["final_hash"], where
        shop.dispose()


# SQLite(평가와 서비스의 SQLite 모드)에 간 SQL에는 FOR UPDATE가 없다: SQLAlchemy의 SQLite 방언이 잠금 요청을
# 빼고 보낸다. 개발용 과제의 정답 쓰기(반품·교환·취소·배송지·쿠폰·티켓·이관)를 모두 실행하며 SQL을 모은다.
def test_sqlite_never_receives_for_update():
    registry = build_registry()
    statements: list[str] = []
    tools_run: set[str] = set()
    for task in tasks_of("dev").values():
        shop = db.memory_engine(seed_engine())
        event.listen(shop, "before_cursor_execute", lambda *a: statements.append(a[2]))
        ctx = ToolContext(
            now=task.now,
            enforce_policy=True,
            state=ConversationState(verified_customer_id=task.customer_id if task.gold_verified else None),
        )
        for action in task.gold_actions:
            assert execute(registry, shop, ctx, action.tool, action.args).ok
            tools_run.add(action.tool)
        shop.dispose()
    assert {"request_return", "request_exchange", "cancel_order", "issue_compensation_coupon"} <= tools_run
    assert statements
    assert not [s for s in statements if "FOR UPDATE" in s.upper()]


# 사유만 다른 재요청(단순 변심 → 주문 실수)이 같은 주문의 승인을 하나 더 만들지 않는다. 감사 로그에는 두 번째
# 요청이 같은 승인 번호를 다시 쓴 것(reused)으로 남는다.
def test_a_rerequest_with_another_reason_reuses_the_pending_approval(engine):  # noqa: F811
    name, phone, order_id, _ = BIG
    script = [
        ToolCall("find_customer", {"name": name, "contact": phone}),
        ToolCall("cancel_order", {"order_id": order_id, "reason": "changed_mind"}),
        ToolCall("cancel_order", {"order_id": order_id, "reason": "ordered_by_mistake"}),
        "담당자 확인 후 처리됩니다.",
    ]
    with make_client(engine, script) as client:
        session_id = client.post("/api/sessions").json()["session_id"]
        say(client, session_id, f"{name}, {phone}입니다. {order_id} 취소해 주세요.")
        audit = client.get(f"/api/admin/sessions/{session_id}", headers=ADMIN).json()["audit"]
    with Session(engine) as session:
        approvals = session.scalars(select(store.Approval)).all()
        assert session.scalar(select(func.count()).select_from(store.Approval)) == 1
    assert approvals[0].args["reason"] == "changed_mind"  # 처음 요청의 인자 그대로
    asked = [e["payload"] for e in audit if e["kind"] == "approval_requested"]
    assert [(a["code"], a.get("reused", False)) for a in asked] == [("AP-1", False), ("AP-1", True)]
