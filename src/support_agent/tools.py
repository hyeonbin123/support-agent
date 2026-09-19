"""The 14 domain tools. Every caller runs them through toolkit.execute().

Check order inside a handler: identity -> existence -> ownership -> integrity (`ctx.require`) -> policy
(`ctx.check_policy`). Error messages give the reason only, never a hint about what to do next.
Nothing here reads the wall clock, uuid or random numbers: time is `ctx.now`, new ids come from content.
"""

from __future__ import annotations

from datetime import timedelta

from pydantic import Field, field_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from support_agent import db, rules
from support_agent.clock import format_kst, format_kst_date
from support_agent.config import HANDOFF_MESSAGE
from support_agent.labels import choices, label
from support_agent.toolkit import Registry, ToolArgs, ToolContext, make_registry, tool

CANCELLABLE = (db.OrderStatus.PAID, db.OrderStatus.PREPARING)


# ---------------------------------------------------------------------------------------------- arguments


class FindCustomerArgs(ToolArgs):
    name: str = Field(description="고객 이름")
    contact: str = Field(description="가입한 전화번호 또는 이메일")


class CustomerArgs(ToolArgs):
    customer_id: str = Field(description="고객 번호 (예: C-0001)")


class OrderArgs(ToolArgs):
    order_id: str = Field(description="주문 번호 (예: O-10001)")


class ProductArgs(ToolArgs):
    product_id: str = Field(description="상품 번호 (예: P-0001)")


class CancelOrderArgs(ToolArgs):
    order_id: str = Field(description="주문 번호")
    reason: db.CancelReason = Field(description=f"취소 사유. {choices(db.CancelReason)}")


class RequestReturnArgs(ToolArgs):
    order_id: str = Field(description="주문 번호")
    line_nos: list[int] = Field(min_length=1, description="반품할 주문 상품의 줄 번호(line_no) 목록")
    reason: db.RequestReason = Field(description=f"반품 사유. {choices(db.RequestReason)}")

    @field_validator("line_nos")
    @classmethod
    def no_duplicates(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError("line_nos must not contain duplicates")
        return value


class RequestExchangeArgs(ToolArgs):
    order_id: str = Field(description="주문 번호")
    line_no: int = Field(description="교환할 주문 상품의 줄 번호(line_no)")
    new_variant_id: str = Field(description="새로 받을 옵션 번호 (예: V-0001-02)")
    reason: db.RequestReason = Field(description=f"교환 사유. {choices(db.RequestReason)}")


class ChangeAddressArgs(ToolArgs):
    order_id: str = Field(description="주문 번호")
    address_id: str = Field(description="고객 주소록의 주소 번호 (예: AD-C-0001-2)")


class IssueCouponArgs(ToolArgs):
    order_id: str = Field(description="보상 대상 주문 번호")
    reason: db.CompensationReason = Field(description=f"보상 사유. {choices(db.CompensationReason)}")


class CreateTicketArgs(ToolArgs):
    category: db.TicketCategory = Field(description=f"문의 분류. {choices(db.TicketCategory)}")
    body: str = Field(min_length=1, description="문의 내용")
    order_id: str = Field(default="", description="관련 주문 번호. 없으면 빈 문자열")


class TransferArgs(ToolArgs):
    reason: db.HandoffReason = Field(description=f"이관 사유. {choices(db.HandoffReason)}")
    summary: str = Field(min_length=1, description="상담원에게 남길 대화 요약")


class ThinkArgs(ToolArgs):
    thought: str = Field(description="생각한 내용")


# ------------------------------------------------------------------------------------------------ helpers


def _verified(ctx: ToolContext) -> str:
    customer_id = ctx.state.verified_customer_id
    ctx.require(customer_id is not None, "identity_not_verified", "본인 확인이 되지 않았습니다.")
    return customer_id


def _own_customer(session: Session, ctx: ToolContext, customer_id: str) -> db.Customer:
    verified = _verified(ctx)
    customer = session.get(db.Customer, customer_id)
    ctx.require(customer is not None, "customer_not_found", "고객을 찾을 수 없습니다.")
    ctx.require(customer.id == verified, "not_same_customer", "본인 확인된 고객의 정보가 아닙니다.")
    return customer


def _own_order(session: Session, ctx: ToolContext, order_id: str) -> db.Order:
    verified = _verified(ctx)
    order = session.get(db.Order, order_id)
    ctx.require(order is not None, "order_not_found", "주문을 찾을 수 없습니다.")
    ctx.require(order.customer_id == verified, "not_order_owner", "본인 확인된 고객의 주문이 아닙니다.")
    return order


def _line(ctx: ToolContext, order: db.Order, line_no: int) -> db.OrderItem:
    item = next((x for x in order.items if x.line_no == line_no), None)
    ctx.require(item is not None, "line_not_found", f"주문에 {line_no}번 줄이 없습니다.")
    return item


def _require_open(ctx: ToolContext, order: db.Order) -> None:
    ctx.require(order.status != db.OrderStatus.CANCELLED, "already_cancelled", "이미 취소된 주문입니다.")


def _require_line_free(ctx: ToolContext, item: db.OrderItem) -> None:
    ctx.require(item.status != db.ItemStatus.CANCELLED, "already_cancelled", "이미 취소된 주문 상품입니다.")
    ctx.require(
        item.status == db.ItemStatus.ORDERED,
        "already_requested",
        f"{item.line_no}번 줄은 이미 반품 또는 교환이 접수되었습니다.",
    )


def _check_delivered_in_window(ctx: ToolContext, order: db.Order, kind: str, word: str) -> None:
    """The two policy rules that returns and exchanges share. `kind` is the code prefix."""
    ctx.check_policy(
        order.status == db.OrderStatus.DELIVERED,
        f"{kind}_not_delivered",
        f"배송완료 상태의 주문만 {word}할 수 있습니다.",
    )
    delivered_at = order.shipment.delivered_at if order.shipment is not None else None
    if delivered_at is not None:
        ctx.check_policy(
            rules.within_return_window(ctx.now, delivered_at),
            f"{kind}_window_expired",
            f"{word} 기한(수령일 다음 날부터 {rules.RETURN_WINDOW_DAYS}일)이 지났습니다.",
        )


def _free_id(session: Session, model: type[db.Base], prefix: str, start: int) -> str:
    """`{prefix}{n}` with the first unused n from `start`: a count-based id that never collides."""
    n = start
    while session.get(model, f"{prefix}{n}") is not None:
        n += 1
    return f"{prefix}{n}"


def _compensation_coupons(session: Session, **where: str) -> list[db.Coupon]:
    query = select(db.Coupon).filter_by(kind=db.CouponKind.COMPENSATION, **where).order_by(db.Coupon.id)
    return list(session.scalars(query))


def _normalise_contact(contact: str) -> tuple[str, str]:
    """(column, value): an e-mail in lower case, or a phone number reduced to its digits."""
    text = contact.strip()
    if "@" in text:
        return "email", text.lower()
    return "phone", "".join(ch for ch in text if ch.isdigit())


# -------------------------------------------------------------------------------------------------- reads


@tool(write=False)
def find_customer(session: Session, ctx: ToolContext, args: FindCustomerArgs) -> dict:
    """이름과 가입 연락처(전화번호 또는 이메일)로 고객 본인 확인을 한다. 둘 다 일치해야 한다."""
    column, value = _normalise_contact(args.contact)
    name = "".join(args.name.split())
    found = None
    if value and name:
        candidates = session.scalars(
            select(db.Customer).filter_by(**{column: value}).order_by(db.Customer.id)
        )
        found = next((c for c in candidates if "".join(c.name.split()) == name), None)
    ctx.require(found is not None, "customer_not_found", "이름과 연락처가 모두 일치하는 고객이 없습니다.")
    verified = ctx.state.verified_customer_id
    ctx.require(
        verified is None or verified == found.id,
        "already_verified",
        "이 대화에서는 이미 다른 고객의 본인 확인을 마쳤습니다.",
    )
    ctx.state.verified_customer_id = found.id
    return {
        "customer_id": found.id,
        "name": found.name,
        "grade": found.grade.value,
        "grade_label": label(found.grade),
    }


@tool(write=False)
def get_customer(session: Session, ctx: ToolContext, args: CustomerArgs) -> dict:
    """본인 확인된 고객의 연락처, 등급, 주소록을 조회한다."""
    customer = _own_customer(session, ctx, args.customer_id)
    return {
        "customer_id": customer.id,
        "name": customer.name,
        "phone": customer.phone,
        "email": customer.email,
        "grade": customer.grade.value,
        "grade_label": label(customer.grade),
        "joined_at": format_kst(customer.joined_at),
        "addresses": [
            {
                "address_id": a.id,
                "label": a.label,
                "recipient": a.recipient,
                "postal_code": a.postal_code,
                "address": a.address,
                "is_default": a.is_default,
            }
            for a in customer.addresses
        ],
    }


@tool(write=False)
def list_orders(session: Session, ctx: ToolContext, args: CustomerArgs) -> dict:
    """본인 확인된 고객의 주문 목록을 최근 주문부터 조회한다."""
    customer = _own_customer(session, ctx, args.customer_id)
    query = (
        select(db.Order).filter_by(customer_id=customer.id).order_by(db.Order.ordered_at.desc(), db.Order.id)
    )
    orders = []
    for order in session.scalars(query):
        summary = order.items[0].product_name if order.items else ""
        if len(order.items) > 1:
            summary += f" 외 {len(order.items) - 1}건"
        orders.append(
            {
                "order_id": order.id,
                "ordered_at": format_kst(order.ordered_at),
                "status": order.status.value,
                "status_label": label(order.status),
                "total_won": order.total_won,
                "item_summary": summary,
            }
        )
    return {"customer_id": customer.id, "orders": orders}


@tool(write=False)
def get_order(session: Session, ctx: ToolContext, args: OrderArgs) -> dict:
    """주문 한 건의 상품, 금액, 결제, 배송지, 반품·교환 접수 내역을 조회한다."""
    order = _own_order(session, ctx, args.order_id)
    payment = order.payment
    items = []
    for item in order.items:
        variant = session.get(db.ProductVariant, item.variant_id)
        items.append(
            {
                "line_no": item.line_no,
                "product_id": variant.product_id,
                "variant_id": item.variant_id,
                "product_name": item.product_name,
                "option_label": item.option_label,
                "quantity": item.quantity,
                "unit_price_won": item.unit_price_won,
                "status": item.status.value,
                "status_label": label(item.status),
            }
        )
    return {
        "order_id": order.id,
        "status": order.status.value,
        "status_label": label(order.status),
        "ordered_at": format_kst(order.ordered_at),
        "items": items,
        "items_won": order.items_won,
        "shipping_fee_won": order.shipping_fee_won,
        "discount_won": order.discount_won,
        "total_won": order.total_won,
        "payment": {
            "method": payment.method.value,
            "method_label": label(payment.method),
            "amount_won": payment.amount_won,
            "status": payment.status.value,
            "status_label": label(payment.status),
            "refund_won": payment.refund_won,
        },
        "shipping": {
            "address_id": order.ship_address_id,
            "recipient": order.ship_recipient,
            "postal_code": order.ship_postal_code,
            "address": order.ship_address,
        },
        "requests": [
            {
                "request_id": r.id,
                "kind": r.kind.value,
                "kind_label": label(r.kind),
                "reason": r.reason.value,
                "reason_label": label(r.reason),
                "line_nos": [x.line_no for x in r.items],
                "refund_won": r.refund_won,
                "return_fee_won": r.return_fee_won,
                "created_at": format_kst(r.created_at),
            }
            for r in order.requests
        ],
        "cancelled_at": format_kst(order.cancelled_at),
        "cancel_reason": order.cancel_reason.value if order.cancel_reason else None,
        "cancel_reason_label": label(order.cancel_reason) if order.cancel_reason else None,
    }


@tool(write=False)
def get_product(session: Session, ctx: ToolContext, args: ProductArgs) -> dict:
    """상품의 옵션별 가격과 재고를 조회한다."""
    product = session.get(db.Product, args.product_id)
    ctx.require(product is not None, "product_not_found", "상품을 찾을 수 없습니다.")
    return {
        "product_id": product.id,
        "name": product.name,
        "category": product.category,
        "variants": [
            {"variant_id": v.id, "option_label": v.option_label, "price_won": v.price_won, "stock": v.stock}
            for v in product.variants
        ],
    }


@tool(write=False)
def track_shipment(session: Session, ctx: ToolContext, args: OrderArgs) -> dict:
    """주문의 택배사, 송장 번호, 배송 상태, 출고·수령 시각, 도착 예정일을 조회한다."""
    order = _own_order(session, ctx, args.order_id)
    shipment = order.shipment
    ctx.require(shipment is not None, "shipment_not_found", "이 주문의 배송 정보가 없습니다.")
    return {
        "order_id": order.id,
        "carrier": shipment.carrier,
        "tracking_no": shipment.tracking_no,
        "status": shipment.status.value,
        "status_label": label(shipment.status),
        "shipped_at": format_kst(shipment.shipped_at),
        "delivered_at": format_kst(shipment.delivered_at),
        "promised_by": format_kst_date(shipment.promised_by),
    }


# ------------------------------------------------------------------------------------------------- writes


@tool(write=True)
def cancel_order(session: Session, ctx: ToolContext, args: CancelOrderArgs) -> dict:
    """주문 전체를 취소하고 결제 금액 전액을 원 결제 수단으로 환불 처리한다."""
    order = _own_order(session, ctx, args.order_id)
    _require_open(ctx, order)
    ctx.check_policy(
        order.status in CANCELLABLE,
        "cancel_not_allowed_status",
        f"{label(order.status)} 상태의 주문은 취소할 수 없습니다.",
    )
    order.status = db.OrderStatus.CANCELLED
    order.cancelled_at = ctx.now
    order.cancel_reason = args.reason
    for item in order.items:
        item.status = db.ItemStatus.CANCELLED
    payment = order.payment
    payment.status = db.PaymentStatus.REFUND_PENDING
    payment.refund_won = payment.amount_won
    return {
        "order_id": order.id,
        "status": order.status.value,
        "status_label": label(order.status),
        "refund_won": payment.refund_won,
        "refund_method": payment.method.value,
        "refund_method_label": label(payment.method),
    }


@tool(write=True, unordered_args=("line_nos",))
def request_return(session: Session, ctx: ToolContext, args: RequestReturnArgs) -> dict:
    """배송완료된 주문의 상품 줄을 반품 접수한다. 환불액과 반품 배송비는 규정대로 계산된다."""
    order = _own_order(session, ctx, args.order_id)
    line_nos = sorted(args.line_nos)
    items = [_line(ctx, order, line_no) for line_no in line_nos]
    _require_open(ctx, order)
    for item in items:
        _require_line_free(ctx, item)
    request_id = f"RT-{order.id}-{line_nos[0]}"
    ctx.require(
        session.get(db.ServiceRequest, request_id) is None,
        "already_requested",
        f"{line_nos[0]}번 줄은 이미 반품 또는 교환이 접수되었습니다.",
    )
    _check_delivered_in_window(ctx, order, "return", "반품")

    fee = rules.return_fee_won(args.reason)
    refund = rules.return_refund_won([(x.unit_price_won, x.quantity) for x in items], args.reason)
    session.add(
        db.ServiceRequest(
            id=request_id,
            order_id=order.id,
            kind=db.RequestKind.RETURN,
            reason=args.reason,
            refund_won=refund,
            return_fee_won=fee,
            created_at=ctx.now,
            items=[
                db.ServiceRequestItem(line_no=x.line_no, quantity=x.quantity, exchange_variant_id=None)
                for x in items
            ],
        )
    )
    for item in items:
        item.status = db.ItemStatus.RETURN_REQUESTED
    return {
        "request_id": request_id,
        "order_id": order.id,
        "line_nos": line_nos,
        "refund_won": refund,
        "return_fee_won": fee,
    }


@tool(write=True)
def request_exchange(session: Session, ctx: ToolContext, args: RequestExchangeArgs) -> dict:
    """배송완료된 주문의 상품 줄 하나를 같은 상품의 다른 옵션으로 교환 접수한다."""
    order = _own_order(session, ctx, args.order_id)
    item = _line(ctx, order, args.line_no)
    new_variant = session.get(db.ProductVariant, args.new_variant_id)
    ctx.require(new_variant is not None, "variant_not_found", "옵션을 찾을 수 없습니다.")
    _require_open(ctx, order)
    _require_line_free(ctx, item)
    request_id = f"EX-{order.id}-{item.line_no}"
    ctx.require(
        session.get(db.ServiceRequest, request_id) is None,
        "already_requested",
        f"{item.line_no}번 줄은 이미 반품 또는 교환이 접수되었습니다.",
    )
    ctx.require(new_variant.id != item.variant_id, "same_variant", "주문한 옵션과 같은 옵션입니다.")
    ctx.require(new_variant.stock >= item.quantity, "out_of_stock", "해당 옵션의 재고가 부족합니다.")
    _check_delivered_in_window(ctx, order, "exchange", "교환")
    current = session.get(db.ProductVariant, item.variant_id)
    ctx.check_policy(
        current.product_id == new_variant.product_id,
        "exchange_different_product",
        "다른 상품의 옵션으로는 교환할 수 없습니다.",
    )

    session.add(
        db.ServiceRequest(
            id=request_id,
            order_id=order.id,
            kind=db.RequestKind.EXCHANGE,
            reason=args.reason,
            refund_won=0,
            return_fee_won=0,
            created_at=ctx.now,
            items=[
                db.ServiceRequestItem(
                    line_no=item.line_no, quantity=item.quantity, exchange_variant_id=new_variant.id
                )
            ],
        )
    )
    item.status = db.ItemStatus.EXCHANGE_REQUESTED
    new_variant.stock -= item.quantity
    return {
        "request_id": request_id,
        "order_id": order.id,
        "line_no": item.line_no,
        "new_variant_id": new_variant.id,
        "new_option_label": new_variant.option_label,
    }


@tool(write=True)
def change_shipping_address(session: Session, ctx: ToolContext, args: ChangeAddressArgs) -> dict:
    """아직 출고되지 않은 주문의 배송지를 고객 주소록에 있는 다른 주소로 바꾼다."""
    order = _own_order(session, ctx, args.order_id)
    address = session.get(db.CustomerAddress, args.address_id)
    ctx.require(address is not None, "address_not_found", "주소를 찾을 수 없습니다.")
    ctx.require(
        address.customer_id == order.customer_id, "address_not_owned", "고객의 주소록에 있는 주소가 아닙니다."
    )
    _require_open(ctx, order)
    ctx.require(order.ship_address_id != address.id, "same_address", "이미 이 주소로 배송될 주문입니다.")
    ctx.check_policy(
        order.status in CANCELLABLE,
        "address_change_not_allowed_status",
        f"{label(order.status)} 상태의 주문은 배송지를 바꿀 수 없습니다.",
    )
    order.ship_address_id = address.id
    order.ship_recipient = address.recipient
    order.ship_postal_code = address.postal_code
    order.ship_address = address.address
    return {
        "order_id": order.id,
        "address_id": address.id,
        "recipient": address.recipient,
        "postal_code": address.postal_code,
        "address": address.address,
    }


@tool(write=True)
def issue_compensation_coupon(session: Session, ctx: ToolContext, args: IssueCouponArgs) -> dict:
    """배송 지연이나 상품 불량을 겪은 주문에 보상 쿠폰을 발급한다. 금액은 규정이 정한다."""
    order = _own_order(session, ctx, args.order_id)

    if args.reason == db.CompensationReason.DELIVERY_DELAY:
        amount = None
        shipment = order.shipment
        if shipment is not None and order.status != db.OrderStatus.CANCELLED:
            late = rules.delay_days(shipment.promised_by, shipment.delivered_at, ctx.now)
            amount = rules.compensation_amount_won(args.reason, late)
        ctx.check_policy(amount is not None, "coupon_not_eligible", "도착 예정일보다 늦어진 주문이 아닙니다.")
    else:
        defective = any(r.reason == db.RequestReason.DEFECTIVE for r in order.requests)
        amount = rules.compensation_amount_won(args.reason, 0) if defective else None
        ctx.check_policy(
            amount is not None,
            "coupon_not_eligible",
            "이 주문에는 상품 불량으로 접수된 반품·교환이 없습니다.",
        )
    if amount is None:  # let through under P0
        amount = rules.fallback_compensation_won(args.reason)

    for_order = _compensation_coupons(session, order_id=order.id)
    ctx.check_policy(
        not for_order, "coupon_already_issued_for_order", "이 주문에는 이미 보상 쿠폰이 발급되었습니다."
    )
    window_start = ctx.now - timedelta(days=rules.COUPON_WINDOW_DAYS)
    recent = [
        c
        for c in _compensation_coupons(session, customer_id=order.customer_id)
        if window_start < c.issued_at <= ctx.now
    ]
    days, limit = rules.COUPON_WINDOW_DAYS, rules.COUPON_WINDOW_LIMIT
    ctx.check_policy(
        len(recent) < limit,
        "coupon_limit_exceeded",
        f"최근 {days}일 동안 보상 쿠폰을 이미 {limit}장 받은 고객입니다.",
    )

    coupon = db.Coupon(
        id=_free_id(session, db.Coupon, f"CP-{order.id}-", len(for_order) + 1),
        customer_id=order.customer_id,
        kind=db.CouponKind.COMPENSATION,
        amount_won=amount,
        reason=args.reason,
        order_id=order.id,
        issued_at=ctx.now,
        expires_at=ctx.now + timedelta(days=rules.COUPON_VALID_DAYS),
        used_at=None,
    )
    session.add(coupon)
    return {"coupon_id": coupon.id, "amount_won": amount, "expires_at": format_kst(coupon.expires_at)}


@tool(write=True, uncompared_args=("body",))
def create_ticket(session: Session, ctx: ToolContext, args: CreateTicketArgs) -> dict:
    """바로 처리할 수 없는 문의를 담당 부서가 확인하도록 상담 티켓으로 남긴다."""
    customer_id = _verified(ctx)
    order_id = args.order_id.strip()
    if order_id:
        _own_order(session, ctx, order_id)
    count = len(list(session.scalars(select(db.Ticket.id).filter_by(customer_id=customer_id))))
    ticket = db.Ticket(
        id=_free_id(session, db.Ticket, f"TK-{customer_id}-", count + 1),
        customer_id=customer_id,
        order_id=order_id or None,
        category=args.category,
        body=args.body,
        created_at=ctx.now,
    )
    session.add(ticket)
    return {
        "ticket_id": ticket.id,
        "category": ticket.category.value,
        "category_label": label(ticket.category),
    }


@tool(write=True, terminates=True, uncompared_args=("reason", "summary"))
def transfer_to_human(session: Session, ctx: ToolContext, args: TransferArgs) -> dict:
    """대화를 사람 상담원에게 넘기고 상담을 끝낸다."""
    count = len(list(session.scalars(select(db.Handoff.id))))
    handoff = db.Handoff(
        id=_free_id(session, db.Handoff, "HO-", count + 1),
        customer_id=ctx.state.verified_customer_id,
        reason=args.reason,
        summary=args.summary,
        created_at=ctx.now,
    )
    session.add(handoff)
    return {"handoff_id": handoff.id, "message": HANDOFF_MESSAGE}


@tool(write=False)
def think(session: Session, ctx: ToolContext, args: ThinkArgs) -> dict:
    """고객에게 보이지 않는 메모장에 생각을 적는다. 정보를 조회하거나 DB를 바꾸지 않는다."""
    return {"ok": True}


def build_registry() -> Registry:
    return make_registry(
        find_customer,
        get_customer,
        list_orders,
        get_order,
        get_product,
        track_shipment,
        cancel_order,
        request_return,
        request_exchange,
        change_shipping_address,
        issue_compensation_coupon,
        create_ticket,
        transfer_to_human,
        think,
    )
