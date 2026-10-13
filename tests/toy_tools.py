"""A toy registry for tests that must not depend on the real domain tools."""

# [파일 역할] 실제 도메인 도구(tools.py의 14개) 대신 테스트에서 쓰는 작은 장난감 도구 등록부다. 조회 1개, 쓰기
# 1개, 항상 예외를 내는 도구 1개, 대화를 끝내는 이관 도구 1개, think 도구 1개로 이루어진다.
# [왜 필요한가] 에이전트 루프(test_agent.py), 판정(test_judge.py), 완료 주장 가드(test_claims.py), 실행
# 경로(test_toolkit.py) 같은 테스트는 도구의 '종류'(읽기/쓰기/종료/버그)에 따른 동작만 확인하면 된다. 실제
# 도구를 쓰면 규정·본인 확인 같은 도메인 규칙이 섞여 테스트가 무엇을 검증하는지 흐려지고, 도구가 바뀔 때마다
# 무관한 테스트가 깨진다.
# [연결] TOY_REGISTRY를 test_agent.py, test_claims.py, test_judge.py, test_review_t7.py, test_stage2.py,
# test_toolkit.py, test_voice.py가 가져다 쓴다. 이 파일은 toolkit.py의 @tool·make_registry와 db.py의
# Order·Handoff 테이블만 쓴다.

from __future__ import annotations

# pydantic Field: 인자 설명(description)이 도구 스키마에 들어가므로 실제 도구와 같은 방식으로 적는다.
# Session은 핸들러가 받는 DB 세션의 타입 표기다.
from pydantic import Field
from sqlalchemy.orm import Session

# db: 테스트용 작은 DB(conftest.py의 tiny_engine)에 있는 Order·Handoff 행을 읽고 쓴다. toolkit: 실제 도구와
# 똑같은 계약(ToolArgs, ToolContext, @tool, make_registry)으로 도구를 만들어 execute() 경로를 그대로 지나게
# 한다.
from support_agent import db
from support_agent.toolkit import ToolArgs, ToolContext, make_registry, tool


# 주문 번호 하나만 받는 인자 모델. ToolArgs를 상속해 모르는 인자는 거절된다(extra=forbid). test_toolkit.py와
# test_judge.py가 직접 가져다 쓴다.
class OrderArgs(ToolArgs):
    order_id: str = Field(description="주문 번호")


# think 도구의 인자. make_registry는 필수 인자가 하나도 없는 도구를 거절하므로(Ollama 파서 제약) thought를
# 필수로 둔다.
class NoteArgs(ToolArgs):
    thought: str = Field(description="생각")


# 이관 도구의 인자. 실제 transfer_to_human처럼 자유 입력 요약(summary)을 받는다.
class HandoffArgs(ToolArgs):
    summary: str = Field(description="요약")


# 읽기 도구: 주문을 조회한다. 없는 주문이면 ctx.require로 order_not_found 오류를 낸다. 읽기 도구는
# before_write 훅을 지나지 않으므로, 쓰기 도구와 다르게 다뤄지는지 확인할 때 쓰인다.
@tool(write=False)
def toy_get_order(session: Session, ctx: ToolContext, args: OrderArgs) -> dict:
    """주문 상태를 조회한다."""
    order = session.get(db.Order, args.order_id)
    ctx.require(order is not None, "order_not_found", "주문을 찾을 수 없습니다.")
    return {"order_id": order.id, "status": order.status.value, "total_won": order.total_won}


# 쓰기 도구: 주문을 취소한다. 결제(PAID) 상태가 아닐 때의 거절은 ctx.check_policy라서 P1이면 막히고 P0이면
# 위반으로 기록된 채 통과한다. 그래서 P0/P1 차이, 정답 재실행, 금지 동작 검증 같은 테스트의 기준 도구가 된다.
# 취소 시각은 ctx.now(과제에 고정된 시각)를 쓴다. 현재 시각을 쓰지 않는 것은 판정이 언제 돌려도 같게 나오게
# 하려는 프로젝트 규칙과 같다.
@tool(write=True)
def toy_cancel_order(session: Session, ctx: ToolContext, args: OrderArgs) -> dict:
    """주문을 취소한다."""
    order = session.get(db.Order, args.order_id)
    ctx.require(order is not None, "order_not_found", "주문을 찾을 수 없습니다.")
    ctx.check_policy(
        order.status == db.OrderStatus.PAID, "cancel_not_allowed_status", "취소할 수 없는 상태입니다."
    )
    order.status = db.OrderStatus.CANCELLED
    order.cancelled_at = ctx.now
    return {"order_id": order.id, "status": "cancelled", "refund_won": order.total_won}


# 항상 KeyError를 내는 도구. execute()는 ToolError가 아닌 예외를 ToolBugError로 바꿔 올리는데, 이것은 에이전트
# 실패가 아니라 '우리 코드의 버그'로 보고 에피소드를 infra_error로 처리해야 한다(docs/design.md). 그 경로를
# 시험하려고 둔다.
@tool(write=True)
def toy_bug(session: Session, ctx: ToolContext, args: OrderArgs) -> dict:
    """항상 예상 밖의 예외를 낸다."""
    raise KeyError("boom")


# terminates=True인 쓰기 도구: 이관을 흉내 낸다. 에이전트 루프(agent.py)는 terminates 표시가 있는 도구가
# 성공하면 고정 안내문을 보내고 대화를 handoff로 끝낸다. 이름과 출력 키(handoff_id)를 실제 도구와 같게 둬서,
# 완료 주장 가드(claims.py)도 이 결과를 이관의 근거로 알아본다. id를 HO-1로 고정해 DB 비교가 언제나 같게
# 나온다.
@tool(write=True, terminates=True)
def transfer_to_human(session: Session, ctx: ToolContext, args: HandoffArgs) -> dict:
    """상담원에게 넘긴다."""
    session.add(
        db.Handoff(
            id="HO-1",
            customer_id=ctx.state.verified_customer_id,
            reason=db.HandoffReason.CUSTOMER_REQUEST,
            summary=args.summary,
            created_at=ctx.now,
        )
    )
    return {"handoff_id": "HO-1"}


# think 도구: 아무것도 바꾸지 않고 ok만 돌려준다. agent.visible_tools는 이 이름(THINK_TOOL)의 도구를 R1
# 구성에서만 모델에 보여 주므로, 실제 이름과 같게 둔다.
@tool(write=False)
def think(session: Session, ctx: ToolContext, args: NoteArgs) -> dict:
    """생각을 적는다."""
    return {"ok": True}


# 다섯 도구를 make_registry로 묶은 등록부. make_registry가 이름 형식, 중복, extra=forbid, 필수 인자, 평평한
# 스키마를 검사하므로 장난감 도구도 실제 도구와 같은 제약을 지킨다.
TOY_REGISTRY = make_registry(toy_get_order, toy_cancel_order, toy_bug, transfer_to_human, think)
