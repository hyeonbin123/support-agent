# [파일 역할] toolkit.py의 단일 실행 경로 execute()와 도구 정의 계약(@tool, make_registry, canonical_args,
# llm_schema)을 검증하는 테스트 묶음이다.
# [왜 중요한가] 에이전트 루프, 정답 재실행(judge.py), 서비스, MCP 서버가 모두 같은 execute()를
# 지난다(docs/design.md "실행 경로 하나"). 여기서 오류를 문자열로 돌려주는 방식, P0/P1 규정 처리, 롤백, 버그
# 예외 구분이 어긋나면 모든 측정과 서비스가 함께 틀어진다.
# [방법] 실제 도구 대신 toy_tools.py의 장난감 등록부와 conftest.py의 작은 DB(tiny_engine: 고객 1명, 주문 O-1
# 결제 완료·O-2 배송 중)를 써서 도구 종류별 동작만 본다.

import json

# conftest의 NOW는 과제처럼 고정된 현재 시각(2026-09-14 10:00 KST)이다. 도구가 현재 시각 대신 ctx.now를 쓰므로
# 테스트도 고정 시각을 넣는다. toy_tools는 실제 도메인 도구와 무관한 장난감 등록부다.
import pytest
from conftest import NOW
from sqlalchemy.orm import Session
from toy_tools import TOY_REGISTRY, OrderArgs, toy_get_order

# db: 에피소드용 복사본(memory_engine), 덤프·차이·해시로 DB 상태를 비교한다. toolkit: 검증 대상인 실행 경로와
# 도구 계약이다.
from support_agent import db
from support_agent.toolkit import (
    ToolArgs,
    ToolBugError,
    ToolContext,
    ToolResult,
    canonical_args,
    execute,
    make_registry,
    tool,
)


# 장난감 등록부로 도구 하나를 실행하고 결과와 컨텍스트를 함께 돌려주는 도우미. kw로 enforce_policy(P0/P1) 같은
# ToolContext 필드를 바꾼다. 컨텍스트를 돌려주는 이유는 기록된 위반(ctx.violations)까지 확인하기 위해서다.
def run(engine, name, args, **kw):
    ctx = ToolContext(now=NOW, **kw)
    return execute(TOY_REGISTRY, engine, ctx, name, args), ctx


# 읽기 도구가 성공하면 content가 JSON 문자열이고 값(O-1 결제 금액 38,900원)을 담는지 본다. 모델에 돌려주는
# 도구 결과의 기본 형식이다.
def test_a_read_tool_returns_json(tiny_engine):
    result, _ = run(tiny_engine, "toy_get_order", {"order_id": "O-1"})
    assert result.ok and json.loads(result.content)["total_won"] == 38900


# 없는 도구 이름과 지어낸 인자(invented)는 예외가 아니라 'Error: [code] 메시지' 결과로 돌아오는지 본다. 모델의
# 실수는 에이전트 실패로 남아야지 하니스를 멈추면 안 되기 때문이다. 지어낸 인자 거절은 ToolArgs의
# extra=forbid에서 온다.
def test_unknown_tool_and_bad_arguments_are_reported_not_raised(tiny_engine):
    result, _ = run(tiny_engine, "nope", {})
    assert (result.ok, result.error_code) == (False, "unknown_tool")
    result, _ = run(tiny_engine, "toy_get_order", {"order_id": "O-1", "invented": 1})
    assert result.error_code == "invalid_arguments"
    assert result.content.startswith("Error: [invalid_arguments]")


# P1(enforce_policy=True)에서는 규정 위반(배송 중인 O-2 취소)이 policy_blocked=True로 거절되고, DB가
# 그대로이며 위반 기록도 남지 않는지 본다. 거절된 호출이 DB를 바꾸면 판정(최종 DB 비교)이 틀어진다.
def test_p1_refuses_a_policy_violation_and_leaves_the_db_alone(tiny_engine):
    engine = db.memory_engine(tiny_engine)
    before = db.dump_db(engine)
    result, ctx = run(engine, "toy_cancel_order", {"order_id": "O-2"}, enforce_policy=True)
    assert (result.ok, result.error_code, result.policy_blocked) == (False, "cancel_not_allowed_status", True)
    assert db.dump_db(engine) == before and ctx.violations == []


# P0(규정은 프롬프트에만)에서는 같은 위반이 실행되고, 결과의 violations에 코드가 남는지 본다. 이 기록이
# '통과된 규정 위반' 지표의 근거이고, P0와 P1의 차이가 2단계 후보 P1의 비교 축이다.
# diff_dumps로 원본과 복사본을 비교해 주문 상태가 실제로 cancelled로 바뀌었는지까지 확인한다.
def test_p0_lets_the_violation_through_and_records_it(tiny_engine):
    engine = db.memory_engine(tiny_engine)
    result, ctx = run(engine, "toy_cancel_order", {"order_id": "O-2"}, enforce_policy=False)
    assert result.ok and result.violations == ("cancel_not_allowed_status",)
    changed = db.diff_dumps(db.dump_db(tiny_engine), db.dump_db(engine))
    assert changed["orders"]["only_actual"][0]["status"] == "cancelled"


# 무결성 검사(ctx.require: 없는 주문)는 P0에서도 막히고 policy_blocked가 False인지 본다. 규정 검사와 달리
# 무결성·본인 확인은 모드와 무관하게 항상 막는다는 설계(docs/design.md)를 지킨다.
def test_integrity_checks_refuse_in_every_mode(tiny_engine):
    result, _ = run(tiny_engine, "toy_get_order", {"order_id": "O-404"}, enforce_policy=False)
    assert (result.error_code, result.policy_blocked) == ("order_not_found", False)


# 핸들러의 예상 못 한 예외(KeyError)가 ToolBugError로 올라오는지 본다. 이것을 오류 문자열로 삼켜 모델에
# 돌려주면 우리 버그가 에이전트 실패로 세어지므로, 예외로 올려 에피소드를 infra_error로 빼게 한다.
def test_a_handler_bug_is_raised_as_tool_bug(tiny_engine):
    with pytest.raises(ToolBugError):
        run(tiny_engine, "toy_bug", {"order_id": "O-1"})


# before_write 훅은 쓰기 도구에만 불리고, 훅이 돌려준 ToolResult가 그대로 거절 결과가 되는지 본다. 읽기 도구
# 호출은 훅을 지나지 않아야 한다(seen에 쓰기 호출 하나만 남음). 훅이 받는 인자는 검증·정규화된 인자다.
def test_before_write_can_refuse_a_write_but_is_not_asked_about_reads(tiny_engine):
    seen = []

    # 모든 쓰기를 confirmation_required로 거절하고, 어떤 호출로 불렸는지 seen에 적는 가짜 훅.
    def gate(spec, args):
        seen.append((spec.name, args))
        return ToolResult.error("confirmation_required", "고객 확인이 필요합니다.")

    ctx = ToolContext(now=NOW)
    assert execute(TOY_REGISTRY, tiny_engine, ctx, "toy_get_order", {"order_id": "O-1"}, before_write=gate).ok
    refused = execute(
        TOY_REGISTRY, tiny_engine, ctx, "toy_cancel_order", {"order_id": "O-1"}, before_write=gate
    )
    assert refused.error_code == "confirmation_required"
    assert seen == [("toy_cancel_order", {"order_id": "O-1"})]


# memory_engine(tiny_engine)로 만든 복사본에 쓴 것이 원본에 새지 않는지 해시로 본다. 에피소드마다 가상 DB의
# 복사본을 하나씩 쓰는 평가 방식(README "평가 방식")의 전제다.
def test_the_episode_copy_is_private(tiny_engine):
    engine = db.memory_engine(tiny_engine)
    run(engine, "toy_cancel_order", {"order_id": "O-1"})
    assert db.state_hash(db.dump_db(engine)) != db.state_hash(db.dump_db(tiny_engine))


# 같은 호출을 두 복사본에서 돌리면 최종 상태 해시가 같은지 본다. 판정은 '에피소드 끝 DB == 새 복사본에 정답
# 동작만 실행한 DB'로 하므로, 같은 호출이 같은 상태를 만든다는 결정성이 깨지면 정답도 실패로 판정된다.
def test_two_copies_that_run_the_same_calls_end_in_the_same_state(tiny_engine):
    hashes = []
    for _ in range(2):
        engine = db.memory_engine(tiny_engine)
        run(engine, "toy_cancel_order", {"order_id": "O-1"})
        hashes.append(db.state_hash(db.dump_db(engine)))
    assert hashes[0] == hashes[1]


# 필수 인자가 하나도 없는 도구를 make_registry가 거절하는지 본다. toolkit.py의 주석대로 Ollama의 범용 도구
# 호출 파서가 인자가 모두 선택인 호출을 처리하지 못하기 때문이다.
def test_registry_refuses_tools_without_a_required_argument():
    class Loose(ToolArgs):
        note: str = ""

    @tool(write=False)
    def loose(session, ctx, args: Loose) -> dict:
        """doc"""
        return {}

    with pytest.raises(ValueError, match="required"):
        make_registry(loose)


# unordered_args로 표시한 리스트 인자(반품할 줄 번호처럼 순서에 뜻이 없는 것)를 canonical_args가 정렬하는지
# 본다. [2, 1]과 [1, 2]를 같은 호출로 봐야 정답에 없는 쓰기·빠진 쓰기 비교가 순서 차이로 틀리지 않는다.
def test_canonical_args_sorts_order_free_lists():
    class LinesArgs(ToolArgs):
        line_nos: list[int]

    @tool(write=True, unordered_args=("line_nos",))
    def lines(session, ctx, args: LinesArgs) -> dict:
        """doc"""
        return {}

    assert canonical_args(lines, LinesArgs(line_nos=[2, 1])) == {"line_nos": [1, 2]}


# ToolSpec을 함수처럼 직접 부를 수 있고(테스트가 handler를 바로 쓸 수 있게), schema()가 평평한 JSON
# 스키마(required 있음, title 없음)를 내는지 본다. llm_schema가 pydantic 스키마에서 $ref·Optional·title을 걷어
# 내는 것은 작은 모델이 스키마를 덜 헷갈리게 하려는 것이다(toolkit.py docstring).
def test_a_tool_spec_is_callable_and_has_a_flat_schema(tiny_engine):
    with Session(tiny_engine) as session:
        assert toy_get_order(session, ToolContext(now=NOW), OrderArgs(order_id="O-1"))["status"] == "paid"
    schema = toy_get_order.schema()
    assert schema["parameters"]["required"] == ["order_id"] and "title" not in json.dumps(schema)
