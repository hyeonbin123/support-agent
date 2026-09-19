"""A toy registry for tests that must not depend on the real domain tools."""

from __future__ import annotations

from pydantic import Field
from sqlalchemy.orm import Session

from support_agent import db
from support_agent.toolkit import ToolArgs, ToolContext, make_registry, tool


class OrderArgs(ToolArgs):
    order_id: str = Field(description="주문 번호")


class NoteArgs(ToolArgs):
    thought: str = Field(description="생각")


class HandoffArgs(ToolArgs):
    summary: str = Field(description="요약")


@tool(write=False)
def toy_get_order(session: Session, ctx: ToolContext, args: OrderArgs) -> dict:
    """주문 상태를 조회한다."""
    order = session.get(db.Order, args.order_id)
    ctx.require(order is not None, "order_not_found", "주문을 찾을 수 없습니다.")
    return {"order_id": order.id, "status": order.status.value, "total_won": order.total_won}


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


@tool(write=True)
def toy_bug(session: Session, ctx: ToolContext, args: OrderArgs) -> dict:
    """항상 예상 밖의 예외를 낸다."""
    raise KeyError("boom")


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


@tool(write=False)
def think(session: Session, ctx: ToolContext, args: NoteArgs) -> dict:
    """생각을 적는다."""
    return {"ok": True}


TOY_REGISTRY = make_registry(toy_get_order, toy_cancel_order, toy_bug, transfer_to_human, think)
