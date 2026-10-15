"""도메인 도구 14개와, 음성 V4(9단계)에서 find_customer 자리를 대신하는 verify_caller.

모든 호출자는 이 도구들을 toolkit.execute()로 실행한다.

핸들러 안의 검사 순서: 본인 확인 -> 대상 존재 -> 소유 -> 무결성(`ctx.require`) -> 규정
(`ctx.check_policy`). 오류 메시지는 이유만 말하고, 다음에 무엇을 하라는 힌트는 주지 않는다.
여기서는 현재 시각(벽시계), uuid, 난수를 읽지 않는다: 시각은 `ctx.now`이고 새 id는 내용에서 만든다.
"""

# [파일 역할] 가상 쇼핑몰 고객센터의 도구 14개(조회 6, 쓰기 7, think)와 음성 V4용 verify_caller를 정의한다. 각
# 도구는 인자 모델(ToolArgs) + 핸들러 함수이고, @tool이 docstring을 모델에게 보이는 도구 설명으로 쓴다.
# [설계 원칙] (docs/design.md "도구")
# - 이름은 영문, 설명·오류 메시지는 한국어. enum 인자는 ASCII 코드이고 설명에 "코드=한국어" 목록을 붙인다
# - 읽기 도구는 사실만 준다. "반품 가능 여부" 같은 계산된 판단은 주지 않아, 규정을 적용하는 일은 에이전트(P0)
#   또는 쓰기 도구의 검사(P1)에 남긴다
# - 검사 순서는 본인 확인 → 대상 존재 → 소유 → 무결성(ctx.require) → 규정(ctx.check_policy)
# - 결정성: 현재 시각·uuid·난수를 쓰지 않는다(테스트가 ast로 검사). 시각은 ctx.now, 새 행 id는 내용이나
#   개수에서 만든다
# [연결] 평가(run.py, judge.py의 정답 재실행), 서비스(service/core.py), MCP 서버가 build_registry()로 같은
# 도구를 쓴다. 규정 검사를 바꾸면 prompts/policy.md, docs/design.md의 표, rules.py를 같이 고친다.

from __future__ import annotations

import re
import unicodedata
from datetime import timedelta
from typing import Annotated

from pydantic import Field, field_validator
from sqlalchemy import select, update
from sqlalchemy.orm import Session

# db: 쇼핑몰 테이블과 enum. rules: 반품 기한·환불액·보상 금액 같은 규정 계산(순수 함수). clock: 시각을 KST로
# 표기하고 KST 날짜를 구한다. labels: enum 코드의 한국어 표기와 "코드=한국어" 선택지 문자열. HANDOFF_MESSAGE:
# 상담원 이관 때 고객에게 가는 고정 안내.
from support_agent import db, rules
from support_agent.clock import format_kst, format_kst_date, kst_date
from support_agent.config import HANDOFF_MESSAGE
from support_agent.labels import choices, label
from support_agent.toolkit import Registry, ToolArgs, ToolContext, make_registry, tool

# 취소와 배송지 변경이 허용되는 주문 상태(결제완료, 상품 준비 중). 출고된 뒤에는 둘 다 규정상 안 된다.
CANCELLABLE = (db.OrderStatus.PAID, db.OrderStatus.PREPARING)
# 음성 V4: 에이전트가 들은 이름과 발신 번호로 가입한 고객의 이름 사이에 허용하는 자모 편집 수. 처음 규칙은
# 1로 정했다가 이름 충돌 관문에서 떨어졌고(seed 이름 가운데 자모 하나 차이인 쌍이 여섯), 2026-10-04 규칙
# 변경에서 측정 전에 0(정규화 뒤 이름 완전 일치)으로 바꿨다(docs/experiments.md 9단계).
NAME_TOLERANCE = 0


# ---------------------------------------------------------------------------------------------- arguments


# --- 인자 모델 ---
# Field의 description은 모델에게 보이는 인자 설명이 된다(한국어). 모든 모델이 ToolArgs를 상속해 지어낸 인자를
# 거절한다.
class FindCustomerArgs(ToolArgs):
    name: str = Field(description="고객 이름")
    contact: str = Field(description="가입한 전화번호 또는 이메일")


# 음성 V4의 본인 확인 인자. 연락처는 받지 않고 이름만 받는다(번호는 하니스가 문맥의 caller_phone으로 넣는다).
class VerifyCallerArgs(ToolArgs):
    name: str = Field(description="고객이 말한 이름")


# 주문 상품의 줄 번호. 1~999로 묶어 터무니없는 값은 검증 단계에서 거절한다.
LineNo = Annotated[int, Field(ge=1, le=999)]


class CustomerArgs(ToolArgs):
    customer_id: str = Field(description="고객 번호 (예: C-0001)")


class OrderArgs(ToolArgs):
    order_id: str = Field(description="주문 번호 (예: O-10001)")


class ProductArgs(ToolArgs):
    product_id: str = Field(description="상품 번호 (예: P-0001)")


class CancelOrderArgs(ToolArgs):
    order_id: str = Field(description="주문 번호")
    # choices()는 "changed_mind=단순 변심, …" 같은 코드=한국어 목록을 만든다. 모델은 ASCII 코드를 보내고
    # 고객에게는 한국어로 말한다.
    reason: db.CancelReason = Field(description=f"취소 사유. {choices(db.CancelReason)}")


# 반품은 같은 주문의 여러 줄을 한 번에 접수한다. 줄 번호 순서는 의미가 없어 unordered_args로 등록하고(정렬해
# 비교), 중복은 검증에서 막는다.
class RequestReturnArgs(ToolArgs):
    order_id: str = Field(description="주문 번호")
    line_nos: list[LineNo] = Field(
        min_length=1,
        max_length=50,
        description="반품할 주문 상품의 줄 번호(line_no) 목록. 같은 주문의 여러 줄은 한 번에 접수한다",
    )
    reason: db.RequestReason = Field(description=f"반품 사유. {choices(db.RequestReason)}")

    @field_validator("line_nos")
    @classmethod
    def no_duplicates(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError("line_nos must not contain duplicates")
        return value


class RequestExchangeArgs(ToolArgs):
    order_id: str = Field(description="주문 번호")
    line_no: LineNo = Field(description="교환할 주문 상품의 줄 번호(line_no)")
    new_variant_id: str = Field(description="새로 받을 옵션 번호 (예: V-0001-02)")
    reason: db.RequestReason = Field(description=f"교환 사유. {choices(db.RequestReason)}")


class ChangeAddressArgs(ToolArgs):
    order_id: str = Field(description="주문 번호")
    address_id: str = Field(description="고객 주소록의 주소 번호 (예: AD-C-0001-2)")


class IssueCouponArgs(ToolArgs):
    order_id: str = Field(description="보상 대상 주문 번호")
    reason: db.CompensationReason = Field(description=f"보상 사유. {choices(db.CompensationReason)}")


# 티켓 인자. 주문과 관계없는 문의면 order_id에 None이 아니라 빈 문자열을 보낸다(기본값도 빈 문자열). 핸들러가
# 빈 문자열을 "주문 없음"으로 저장한다.
class CreateTicketArgs(ToolArgs):
    category: db.TicketCategory = Field(description=f"문의 분류. {choices(db.TicketCategory)}")
    body: str = Field(min_length=1, description="문의 내용")
    order_id: str = Field(default="", description="관련 주문 번호. 없으면 빈 문자열")


class TransferArgs(ToolArgs):
    reason: db.HandoffReason = Field(description=f"이관 사유. {choices(db.HandoffReason)}")
    summary: str = Field(min_length=1, description="상담원에게 남길 대화 요약")


# think 도구의 인자. R1(추론 절차 실험)에서만 모델에게 보인다(agent.py).
class ThinkArgs(ToolArgs):
    thought: str = Field(description="생각한 내용")


# ------------------------------------------------------------------------------------------------ helpers


# --- 공통 검사 도우미 ---
# 본인 확인이 끝났는지 확인하고 확인된 고객 번호를 돌려준다. 확인 전에 고객·주문 정보에 접근하면
# identity_not_verified로 막는다.
def _verified(ctx: ToolContext) -> str:
    customer_id = ctx.state.verified_customer_id
    ctx.require(customer_id is not None, "identity_not_verified", "본인 확인이 되지 않았습니다.")
    return customer_id


# 본인 확인 → 고객 존재 → 확인된 고객 본인인지 순으로 검사한다. 다른 고객의 정보를 조회하지 못하게 한다.
def _own_customer(session: Session, ctx: ToolContext, customer_id: str) -> db.Customer:
    verified = _verified(ctx)
    customer = session.get(db.Customer, customer_id)
    ctx.require(customer is not None, "customer_not_found", "고객을 찾을 수 없습니다.")
    ctx.require(customer.id == verified, "not_same_customer", "본인 확인된 고객의 정보가 아닙니다.")
    return customer


# 본인 확인 → 주문 존재 → 확인된 고객의 주문인지. 주문을 다루는 모든 도구가 처음에 부른다.
# lock=True(주문을 바꾸거나 주문에 딸린 행을 만드는 쓰기 도구): 주문 행을 SELECT … FOR UPDATE로 읽어 이
# 트랜잭션이 끝날 때까지 잠근다. PostgreSQL(서비스)에서 두 세션이 같은 주문을 같은 순간에 바꾸면 늦은 쪽이
# 앞쪽의 커밋을 기다린 뒤 바뀐 상태(이미 접수된 줄, 이미 취소된 주문, 이미 발급된 쿠폰)를 보고 판단한다.
# 잠금 뒤의 줄·결제·접수 조회는 지연 로딩이라 잠금을 얻은 다음에 읽힌다(READ COMMITTED). SQLite(평가)는
# FOR UPDATE를 SQL에 넣지 않으므로(SQLAlchemy의 SQLite 방언) 평가가 보내는 SQL과 결과는 그대로다. 잠금
# 순서는 늘 주문 → 고객이라 두 도구가 서로를 기다리는 교착이 생기지 않는다.
def _own_order(session: Session, ctx: ToolContext, order_id: str, *, lock: bool = False) -> db.Order:
    verified = _verified(ctx)
    order = session.get(db.Order, order_id, with_for_update=True if lock else None)
    ctx.require(order is not None, "order_not_found", "주문을 찾을 수 없습니다.")
    ctx.require(order.customer_id == verified, "not_order_owner", "본인 확인된 고객의 주문이 아닙니다.")
    return order


# 주문의 줄 하나를 찾는다. 없으면 line_not_found.
def _line(ctx: ToolContext, order: db.Order, line_no: int) -> db.OrderItem:
    item = next((x for x in order.items if x.line_no == line_no), None)
    ctx.require(item is not None, "line_not_found", f"주문에 {line_no}번 줄이 없습니다.")
    return item


# 이미 취소된 주문에는 취소·반품·교환·배송지 변경을 하지 않는다(무결성 검사라 P0에서도 막는다).
def _require_open(ctx: ToolContext, order: db.Order) -> None:
    ctx.require(order.status != db.OrderStatus.CANCELLED, "already_cancelled", "이미 취소된 주문입니다.")


# 줄이 취소됐거나 이미 반품·교환 접수된 상태면 막는다. 같은 줄을 두 번 접수하지 않게 한다.
def _require_line_free(ctx: ToolContext, item: db.OrderItem) -> None:
    ctx.require(item.status != db.ItemStatus.CANCELLED, "already_cancelled", "이미 취소된 주문 상품입니다.")
    ctx.require(
        item.status == db.ItemStatus.ORDERED,
        "already_requested",
        f"{item.line_no}번 줄은 이미 반품 또는 교환이 접수되었습니다.",
    )


# 배송완료인데 수령 시각 기록이 없는 것은 데이터 무결성 문제라 규정이 아니라 require로 막는다.
def _check_delivered_in_window(ctx: ToolContext, order: db.Order, kind: str, word: str) -> None:
    """반품과 교환이 함께 쓰는 규정 두 가지. `kind`는 오류 코드의 접두사(return/exchange)다.

    규정: 배송완료 상태일 것, 수령일 다음 날부터 7일째 되는 날(KST 날짜)까지일 것. 접두사만 바꿔 같은 검사를
    쓴다.
    """
    ctx.check_policy(
        order.status == db.OrderStatus.DELIVERED,
        f"{kind}_not_delivered",
        f"배송완료 상태의 주문만 {word}할 수 있습니다.",
    )
    delivered_at = order.shipment.delivered_at if order.shipment is not None else None
    ctx.require(
        order.status != db.OrderStatus.DELIVERED or delivered_at is not None,
        "shipment_not_found",
        "배송완료 주문인데 수령 시각 기록이 없습니다.",
    )
    if delivered_at is not None:
        ctx.check_policy(
            rules.within_return_window(ctx.now, delivered_at),
            f"{kind}_window_expired",
            f"{word} 기한(수령일 다음 날부터 {rules.RETURN_WINDOW_DAYS}일)이 지났습니다.",
        )


def _free_id(session: Session, model: type[db.Base], prefix: str, start: int) -> str:
    """`start`부터 아직 쓰이지 않은 첫 n으로 만든 `{prefix}{n}`: 개수로 만들지만 충돌하지 않는 id.

    쿠폰, 티켓, 이관 id에 쓴다. 행이 지워져 개수와 번호가 어긋나도 충돌하지 않는다.
    """
    n = start
    while session.get(model, f"{prefix}{n}") is not None:
        n += 1
    return f"{prefix}{n}"


# 조건(고객 또는 주문)에 맞는 보상 쿠폰을 id 순으로. 판촉 쿠폰은 세지 않는다.
def _compensation_coupons(session: Session, **where: str) -> list[db.Coupon]:
    query = select(db.Coupon).filter_by(kind=db.CouponKind.COMPENSATION, **where).order_by(db.Coupon.id)
    return list(session.scalars(query))


# 연락처 글에서 이메일 주소를 찾는 패턴. "이메일은 a@b.com이에요"처럼 말 속에 섞여 와도 주소만 뽑는다.
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _coupon_fact(coupon: db.Coupon) -> dict:
    """보상 쿠폰 한 장의 사실만(번호, 주문, 사유, 금액, 발급 시각) 준다.

    판단은 넣지 않아 규정의 쿠폰 한도(주문당 1장, 고객당 30일 2장)를 에이전트가 직접 적용하게 한다.
    get_order·get_customer가 이것을 보여 준다(docs/design.md).
    """
    return {
        "coupon_id": coupon.id,
        "order_id": coupon.order_id,
        "reason": coupon.reason.value if coupon.reason else None,
        "reason_label": label(coupon.reason) if coupon.reason else None,
        "amount_won": coupon.amount_won,
        "issued_at": format_kst(coupon.issued_at),
    }


def _normalise_contact(contact: str) -> tuple[str, str]:
    """연락처를 (비교할 열, 값)으로 바꾼다: 소문자 이메일, 또는 숫자만 남긴 전화번호.

    NFKC로 전각 문자를 바꾼 뒤 이메일이 있으면 이메일로, 없으면 전화번호로 비교한다. 국가 번호로 시작하는
    번호(+82 10…)는 010…으로 바꾼다.
    """
    text = unicodedata.normalize("NFKC", contact)
    email = _EMAIL.search(text)
    if email:
        return "email", email.group(0).lower()
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits.startswith("8210"):  # +82 10-0000-0000
        digits = "0" + digits[2:]
    return "phone", digits


# 이름 정규화: NFKC 후 공백을 모두 빼고 끝의 "고객님"·"고객"·"님"을 하나 뗀다. 뗀 뒤 두 글자 이상 남을 때만
# 떼므로 이름 자체가 지워지지 않는다. find_customer와 verify_caller가 같은 기준을 쓴다.
def _normalise_name(name: str) -> str:
    text = "".join(unicodedata.normalize("NFKC", name).split())
    for suffix in ("고객님", "고객", "님"):
        if len(text) > len(suffix) + 1 and text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def name_distance(said: str, registered: str) -> int:
    """find_customer와 같은 이름 정규화 뒤 두 이름의 자모 단위 편집 거리(레벤슈타인, "배성분"·"배성훈": 1).

    NFD로 한글 음절을 자모로 풀어 계산한다. verify_caller의 허용 오차 비교와 analyze.py의 이름 충돌
    검사(9단계 관문)가 쓴다.
    """
    a, b = (unicodedata.normalize("NFD", _normalise_name(x)) for x in (said, registered))
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def _identified(ctx: ToolContext, customer: db.Customer) -> dict:
    """고객을 이 대화의 본인 확인된 고객으로 표시한다(대화당 고객 1명).

    이미 다른 고객을 확인한 대화면 already_verified로 막는다. 호출이 거절되면 execute()가 이 상태 변경을
    되돌린다.
    """
    verified = ctx.state.verified_customer_id
    ctx.require(
        verified is None or verified == customer.id,
        "already_verified",
        "이 대화에서는 이미 다른 고객의 본인 확인을 마쳤습니다.",
    )
    ctx.state.verified_customer_id = customer.id
    return {
        "customer_id": customer.id,
        "name": customer.name,
        "grade": customer.grade.value,
        "grade_label": label(customer.grade),
    }


# -------------------------------------------------------------------------------------------------- reads


# --- 읽기 도구 ---
# 본인 확인: 이름과 연락처가 모두 맞는 고객을 찾는다. 연락처로 후보를 고르고(같은 번호의 고객이 여럿일 수
# 있음) 그중 이름이 같은 첫 고객을 쓴다. 어느 한쪽이 비면 찾지 않는다.
@tool(write=False)
def find_customer(session: Session, ctx: ToolContext, args: FindCustomerArgs) -> dict:
    """이름과 가입 연락처(전화번호 또는 이메일)로 고객 본인 확인을 한다. 둘 다 일치해야 한다."""
    column, value = _normalise_contact(args.contact)
    name = _normalise_name(args.name)
    found = None
    if value and name:
        candidates = session.scalars(
            select(db.Customer).filter_by(**{column: value}).order_by(db.Customer.id)
        )
        found = next((c for c in candidates if "".join(c.name.split()) == name), None)
    ctx.require(found is not None, "customer_not_found", "이름과 연락처가 모두 일치하는 고객이 없습니다.")
    return _identified(ctx, found)


# 음성 V4(9단계)의 본인 확인: 연락처를 받아 적는 대신, 발신 번호로 가입한 고객 한 명과 이름만 맞춘다. 번호가
# 없거나, 그 번호의 고객이 없거나, 여럿이면 각각 다른 코드로 막는다.
# 이름은 그 한 명의 이름과만 비교하므로 다른 고객 이름과 우연히 가까워도 통과하지 않는다. 허용 오차는
# NAME_TOLERANCE(0)라 정규화 뒤 글자까지 같아야 한다.
@tool(write=False)
def verify_caller(session: Session, ctx: ToolContext, args: VerifyCallerArgs) -> dict:
    """전화 상담의 본인 확인. 고객이 말한 이름이 발신 번호로 가입한 고객의 이름과 맞는지 확인한다."""
    ctx.require(ctx.caller_phone is not None, "no_caller_number", "발신 번호가 없는 상담입니다.")
    digits = "".join(ch for ch in ctx.caller_phone or "" if ch.isdigit())
    owners = list(session.scalars(select(db.Customer).filter_by(phone=digits).order_by(db.Customer.id)))
    ctx.require(bool(owners), "caller_not_registered", "발신 번호로 가입한 고객이 없습니다.")
    # 일대일: 번호가 고객 한 명을 가리키고, 이름은 그 고객의 이름과만 비교한다.
    ctx.require(len(owners) == 1, "caller_shared", "발신 번호로 가입한 고객이 여러 명입니다.")
    owner = owners[0]
    ctx.require(
        bool(_normalise_name(args.name)) and name_distance(args.name, owner.name) <= NAME_TOLERANCE,
        "name_mismatch",
        "발신 번호로 가입한 고객의 이름과 일치하지 않습니다.",
    )
    return _identified(ctx, owner)


# 본인 확인된 고객의 정보와 주소록, 발급된 보상 쿠폰. 배송지 변경 때 address_id를 고르는 근거가 된다.
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
        "compensation_coupons": [
            _coupon_fact(c) for c in _compensation_coupons(session, customer_id=customer.id)
        ],
    }


# 주문 목록(최근 주문부터). 상품 요약은 "첫 상품명 외 N건" 형식이다.
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


# 주문 한 건의 전체 사실. 환불액·반품 가능 여부 같은 판단은 넣지 않고, 결제·줄 상태·접수 내역·보상 쿠폰만
# 준다. 출고·수령 시각은 track_shipment로 따로 본다.
@tool(write=False)
def get_order(session: Session, ctx: ToolContext, args: OrderArgs) -> dict:
    """주문 한 건의 상품, 금액, 결제, 배송지, 반품·교환 접수, 보상 쿠폰 내역을 조회한다.

    출고·수령 시각은 track_shipment로 본다."""
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
        "compensation_coupons": [_coupon_fact(c) for c in _compensation_coupons(session, order_id=order.id)],
        "cancelled_at": format_kst(order.cancelled_at),
        "cancel_reason": order.cancel_reason.value if order.cancel_reason else None,
        "cancel_reason_label": label(order.cancel_reason) if order.cancel_reason else None,
    }


# 상품의 옵션별 가격·재고. 교환할 새 옵션(new_variant_id)을 고르는 데 쓴다. 본인 확인 없이 조회할 수 있는 예외
# 도구다(docs/design.md).
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


# 배송 조회. 시각은 KST 표기로, 도착 예정일은 날짜만 준다.
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


# --- 쓰기 도구 ---
# 주문 취소: 결제완료·상품 준비 중일 때만(규정). 주문과 모든 줄을 cancelled로, 결제를 refund_pending으로
# 바꾸고 결제 금액 전액을 환불액으로 적는다(docs/design.md 금액 규칙). 취소 시각은 고정 시계 ctx.now다.
@tool(write=True)
def cancel_order(session: Session, ctx: ToolContext, args: CancelOrderArgs) -> dict:
    """주문 전체를 취소하고 결제 금액 전액을 원 결제 수단으로 환불 처리한다."""
    order = _own_order(session, ctx, args.order_id, lock=True)
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


# 반품 접수: 접수 id는 RT-주문-가장 작은 줄 번호로 내용에서 만든다(호출 순서와 무관하게 같은 id). 환불액 =
# Σ(단가 × 수량) − 반품 배송비(단순 변심 3,000원, 불량·오배송 0원)이고 결제는 건드리지 않는다(회수 뒤 환불).
# 반환하는 refund_won은 서비스의 승인 관문이 기준(10만 원)과 비교하는 값이다(service/core.py).
@tool(write=True, unordered_args=("line_nos",))
def request_return(session: Session, ctx: ToolContext, args: RequestReturnArgs) -> dict:
    """배송완료된 주문의 상품 줄을 반품 접수한다. 환불액과 반품 배송비는 규정대로 계산된다."""
    # 주문 행 잠금: 같은 주문의 줄 [1, 3]과 [3]을 두 세션이 같은 순간에 반품해도 접수 id(첫 줄 번호)가 달라
    # 둘 다 통과하던 경합을 막는다. 늦은 쪽은 3번 줄이 이미 접수된 것을 보고 already_requested로 거절된다.
    # 서비스의 승인 관문이 같은 주문의 반품 합계를 셀 때도 앞쪽의 접수를 본다(service/core.py).
    order = _own_order(session, ctx, args.order_id, lock=True)
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


# 교환 접수: 같은 상품의 다른 옵션으로만(규정), 재고가 그 줄의 수량 이상일 때만(무결성). 환불액은 0원이고 새
# 옵션의 재고를 줄인다. 접수 id는 EX-주문-줄 번호.
@tool(write=True)
def request_exchange(session: Session, ctx: ToolContext, args: RequestExchangeArgs) -> dict:
    """배송완료된 주문의 상품 줄 하나를 같은 상품의 다른 옵션으로 교환 접수한다."""
    order = _own_order(session, ctx, args.order_id, lock=True)
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
    # 재고는 읽고-빼서-쓰지 않고 조건부 UPDATE 한 번으로 줄인다: "재고가 수량 이상일 때만 수량만큼 뺀다".
    # 같은 옵션을 두 세션이 같은 순간에 교환하면, 읽고-빼기는 둘 다 같은 재고를 읽어 하나의 차감이 사라졌다
    # (갱신 유실). 이 UPDATE는 늦은 쪽이 앞쪽의 커밋을 기다린 뒤 새 재고로 조건을 다시 보므로, 차감이 모두
    # 남고 재고가 모자라면 바뀐 행이 0이라 out_of_stock으로 거절된다. 위의 재고 검사(같은 코드)는 오류의
    # 순서를 지금과 같게 두려고 그대로 둔다. 한 세션만 쓰는 평가(SQLite)에서는 그 검사를 통과했으면 이
    # UPDATE도 반드시 한 행을 바꾸므로 결과가 같다.
    taken = session.execute(
        update(db.ProductVariant)
        .where(db.ProductVariant.id == new_variant.id, db.ProductVariant.stock >= item.quantity)
        .values(stock=db.ProductVariant.stock - item.quantity)
    ).rowcount
    ctx.require(taken == 1, "out_of_stock", "해당 옵션의 재고가 부족합니다.")
    return {
        "request_id": request_id,
        "order_id": order.id,
        "line_no": item.line_no,
        "new_variant_id": new_variant.id,
        "new_option_label": new_variant.option_label,
    }


# 배송지 변경: 고객 주소록의 주소만, 출고 전(결제완료·상품 준비 중)에만 된다. 주문의 배송지 열에 주소를 복사해
# 넣는다(주문 당시 주소의 사본 방식).
@tool(write=True)
def change_shipping_address(session: Session, ctx: ToolContext, args: ChangeAddressArgs) -> dict:
    """아직 출고되지 않은 주문의 배송지를 고객 주소록에 있는 다른 주소로 바꾼다."""
    order = _own_order(session, ctx, args.order_id, lock=True)
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


# 보상 쿠폰 발급. 금액은 인자가 아니라 규정이 정한다: 배송 지연은 늦은 일수(1~2일 2,000원, 3일 이상 5,000원),
# 상품 불량은 그 주문에 불량 사유의 반품·교환 접수가 있을 때 3,000원. 주문당 1장, 고객당 최근 30일(KST 날짜,
# 오늘 포함) 2장까지이고 유효 기간은 발급 + 30일이다(docs/design.md).
@tool(write=True)
def issue_compensation_coupon(session: Session, ctx: ToolContext, args: IssueCouponArgs) -> dict:
    """배송 지연이나 상품 불량을 겪은 주문에 보상 쿠폰을 발급한다. 금액은 규정이 정한다."""
    order = _own_order(session, ctx, args.order_id, lock=True)
    # 주문 다음에 고객 행도 잠근다(잠금 순서: 주문 → 고객). "고객당 최근 30일 2장" 한도는 고객의 여러 주문에
    # 걸치므로, 한 고객의 다른 주문 두 개에 같은 순간 발급하면 둘 다 1장만 보고 3장째를 냈다. 이제 늦은 쪽은
    # 앞쪽의 쿠폰이 커밋된 뒤에 센다. 평가(SQLite)에서는 고객 행을 한 번 더 읽을 뿐이다.
    session.get(db.Customer, order.customer_id, with_for_update=True)
    ctx.check_policy(
        order.status != db.OrderStatus.CANCELLED,
        "coupon_order_cancelled",
        "취소된 주문에는 보상 쿠폰을 발급하지 않습니다.",
    )

    if args.reason == db.CompensationReason.DELIVERY_DELAY:
        amount = None
        shipment = order.shipment
        if shipment is not None:
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
    # P0에서 자격 없이 통과된 쿠폰은 사유별 최소 금액으로 발급한다(위반은 ctx.violations에 남는다).
    if amount is None:  # P0에서 막지 않고 통과된 경우
        amount = rules.fallback_compensation_won(args.reason)

    for_order = _compensation_coupons(session, order_id=order.id)
    ctx.check_policy(
        not for_order, "coupon_already_issued_for_order", "이 주문에는 이미 보상 쿠폰이 발급되었습니다."
    )
    # 반품 기한처럼 KST 날짜로 센다: 오늘을 포함한 최근 30일 안에 발급된 쿠폰.
    first_day = kst_date(ctx.now) - timedelta(days=rules.COUPON_WINDOW_DAYS - 1)
    recent = [
        c
        for c in _compensation_coupons(session, customer_id=order.customer_id)
        if first_day <= kst_date(c.issued_at) <= kst_date(ctx.now)
    ]
    days, limit = rules.COUPON_WINDOW_DAYS, rules.COUPON_WINDOW_LIMIT
    ctx.check_policy(
        len(recent) < limit,
        "coupon_limit_exceeded",
        f"최근 {days}일 동안 보상 쿠폰을 이미 {limit}장 받은 고객입니다.",
    )

    # 쿠폰 id는 CP-주문-n. 과제는 같은 부모 아래 새 행을 1개까지만 만들므로 개수 기반 id도
    # 결정적이다(docs/design.md 결정성).
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


# 상담 티켓: 바로 처리할 수 없는 문의를 남긴다. 본문은 모델이 쓰는 자유 글이라 판정의 호출·DB 비교에서 뺀다.
# 주문 번호를 주면 본인 주문인지 확인한다. id는 TK-고객-n.
@tool(write=True, uncompared_args=("body",))
def create_ticket(session: Session, ctx: ToolContext, args: CreateTicketArgs) -> dict:
    """바로 처리할 수 없는 문의를 담당 부서가 확인하도록 상담 티켓으로 남긴다."""
    customer_id = _verified(ctx)
    order_id = args.order_id.strip()
    if order_id:
        _own_order(session, ctx, order_id)
    # 티켓 id는 고객의 티켓 수로 만든다(TK-고객-n). 같은 고객의 두 세션이 같은 순간에 티켓을 남기면 같은 n을
    # 골라 기본 키가 겹쳐 늦은 쪽이 DB 오류로 실패했다. 고객 행을 잠가 늦은 쪽이 앞쪽의 티켓을 본 뒤 센다.
    # 주문은 읽기만 하므로 잠그지 않는다(주문 → 고객 순서를 지키는 다른 도구와 교착하지 않는다).
    session.get(db.Customer, customer_id, with_for_update=True)
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


# 상담원 이관: 성공하면 대화가 끝난다(terminates). 본인 확인 전에도 쓸 수 있어 고객 번호가 비어 있을 수 있다.
# 사유와 요약은 둘 다 타당할 수 있어 비교에서 뺀다. 쓸데없는 이관이 DB 차이로 드러나도록 행으로 남긴다.
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


# think: 아무것도 조회하거나 바꾸지 않는 메모장 도구. 2단계 추론 절차 후보(R1)에서만 보인다. 서비스의 MCP
# 서버는 이 도구를 노출하지 않는다.
@tool(write=False)
def think(session: Session, ctx: ToolContext, args: ThinkArgs) -> dict:
    """고객에게 보이지 않는 메모장에 생각을 적는다. 정보를 조회하거나 DB를 바꾸지 않는다."""
    return {"ok": True}


def build_registry(*, caller_id: bool = False) -> Registry:
    """대화 하나가 쓰는 도구 등록부. caller_id=True(음성 V4)면 find_customer 자리에 verify_caller가 들어간다.

    나머지 도구는 같다. run.py가 실행 설정(config.caller_id)에 따라 고르고, 서비스와 MCP 서버는 기본값을
    쓴다."""
    return make_registry(
        verify_caller if caller_id else find_customer,
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
