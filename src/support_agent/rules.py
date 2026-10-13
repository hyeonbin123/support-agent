"""Policy arithmetic: return deadline, refund amounts, compensation amounts.

Pure functions. No DB, no clock reads: every "now" is an argument. Calendar rules use KST dates.
The numbers here are the ones written in prompts/policy.md (a test keeps the two in step).
"""

# [파일 역할] 규정이 정한 계산(반품 기한, 반품 환불액, 보상 쿠폰 금액)만 모은 순수 함수 모듈이다.
# [설계 이유] 계산을 도구 코드(tools.py)에서 떼어 DB·시계 없이 테스트할 수 있게 했다. '지금' 시각은 항상
# 인자로 받으므로 같은 과제는 언제 돌려도 같은 결과가 나온다(docs/design.md 결정성).
# [규정과의 일치] 여기 숫자는 prompts/policy.md에 적힌 숫자와 같아야 한다. 규정 문서에 없는 규칙을 코드가
# 막으면 에이전트가 알 수 없는 실패가 되기 때문이다. tests/test_tools.py가 policy.md 본문에 이 상수들로 만든
# 문장이 있는지 검사한다. 규정 검사를 바꿀 때는 policy.md, docs/design.md의 표, 이 파일을 같이 고친다.
# [연결] tools.py의 반품·교환 기한 검사, request_return의 환불액, issue_compensation_coupon의 금액·기간·한도가
# 이 함수와 상수를 쓴다.

from __future__ import annotations

from datetime import date, datetime, timedelta

# 반품 사유·보상 사유 enum(db.RequestReason, db.CompensationReason)과 KST 날짜 변환을 가져온다.
from support_agent import db
from support_agent.clock import kst_date

# 반품·교환 기한: 수령일 다음 날부터 7일째 되는 날까지(KST 날짜). 반품 배송비 3,000원은 단순 변심일 때만 받고,
# 상품 불량·오배송은 0원이다(policy.md 반품 절).
RETURN_WINDOW_DAYS = 7
RETURN_FEE_WON = 3000  # charged only when the customer changed their mind

# 보상 쿠폰 규정(policy.md 보상 쿠폰 표, docs/design.md 금액 규칙):
# - 배송 지연: 1~2일 늦으면 2,000원, 3일 이상이면 5,000원. 상품 불량: 3,000원.
# - 유효 기간 30일(발급 시각 + 30일).
# - 한도: 고객당 최근 30일(KST 발급일 기준, 오늘 포함) 동안 2장까지. 주문당 1장 제한은 tools.py가 따로
#   검사한다.
COUPON_DELAY_SHORT_WON = 2000  # 1-2 days late
COUPON_DELAY_LONG_WON = 5000  # 3 days late or more
COUPON_DELAY_LONG_FROM_DAYS = 3
COUPON_DEFECTIVE_WON = 3000
COUPON_VALID_DAYS = 30
COUPON_WINDOW_DAYS = 30
COUPON_WINDOW_LIMIT = 2


# 반품·교환을 접수할 수 있는 마지막 KST 날짜. 수령일에 7일을 더한 날이다(9월 2일 수령 → 9월 9일까지). 시각이
# 아니라 날짜로 계산해 수령 시각이 오후든 오전이든 같은 날 끝나게 한다.
def return_deadline(delivered_at: datetime) -> date:
    """The last KST date on which a return or exchange can be requested."""
    return kst_date(delivered_at) + timedelta(days=RETURN_WINDOW_DAYS)


# 오늘(KST 날짜)이 기한 마지막 날 이하인지. 기한 마지막 날은 처리, 다음 날은 거절이라는 경계 사례를 과제가
# 직접 시험한다(docs/experiments.md 1단계 과제 묶음).
def within_return_window(now: datetime, delivered_at: datetime) -> bool:
    return kst_date(now) <= return_deadline(delivered_at)


# 반품 사유별 반품 배송비. 단순 변심만 고객 부담이다.
def return_fee_won(reason: db.RequestReason) -> int:
    return RETURN_FEE_WON if reason == db.RequestReason.CHANGED_MIND else 0


# 반품 환불액 = Σ(단가 × 수량) − 반품 배송비. items는 반품할 줄들의 (단가, 수량) 목록이다.
# 여러 줄을 한 번에 접수해야 배송비가 한 번만 빠진다(나눠 접수하면 두 번 빠져 DB가 달라진다.
# docs/experiments.md 공통 규칙 과제). 할인 안분 규칙은 없어서 쓰기 대상 주문은 할인이 0원인 것만
# 쓴다(judge.validate_task가 검사). max(0, ...)로 음수 환불을 막는다.
def return_refund_won(items: list[tuple[int, int]], reason: db.RequestReason) -> int:
    """Sum of unit price x quantity minus the return shipping fee, never below zero."""
    goods = sum(unit_price * quantity for unit_price, quantity in items)
    return max(0, goods - return_fee_won(reason))


# 배송 지연 일수 = (수령일, 아직 못 받았으면 오늘) − 도착 예정일. KST 날짜로 센다. 0 이하면 늦지 않은 것이다.
def delay_days(promised_by: date, delivered_at: datetime | None, now: datetime) -> int:
    """Days past the promised date, counted to the delivery date or, if not delivered yet, to today (KST).

    Zero or negative means on time.
    """
    reached = kst_date(delivered_at) if delivered_at is not None else kst_date(now)
    return (reached - promised_by).days


# 규정이 정한 쿠폰 금액. 금액은 도구 인자가 아니라 규정이 정한다(docs/design.md 금액 규칙). 그래서 모델이
# 금액을 정해 넣을 길이 없다.
# 지연 쿠폰 자격이 없으면 None을 돌려주고, 그 처리(P1이면 coupon_not_eligible로 거절, P0이면 위반 기록 뒤
# 통과)는 tools.py가 정한다.
def compensation_amount_won(reason: db.CompensationReason, delay_days: int) -> int | None:
    """The coupon amount the policy fixes, or None when a delivery-delay coupon is not due.

    `defective_item` always has an amount; whether a defective request exists is a DB fact the tool checks.
    """
    # 불량 보상은 금액이 고정이다. 불량 반품·교환 접수가 실제로 있는지는 DB를 봐야 알 수 있어 도구가 따로
    # 확인한다.
    if reason == db.CompensationReason.DEFECTIVE_ITEM:
        return COUPON_DEFECTIVE_WON
    if delay_days <= 0:
        return None
    return COUPON_DELAY_LONG_WON if delay_days >= COUPON_DELAY_LONG_FROM_DAYS else COUPON_DELAY_SHORT_WON


# P0에서 자격 없는 쿠폰이 통과됐을 때 발급할 금액: 사유별 가장 낮은 금액(지연 2,000원, 불량 3,000원). P0은
# 위반을 기록만 하고 쿠폰 행을 실제로 만드는데, 규정상 정해진 금액이 없으니 이 값으로 채운다(docs/design.md
# 금액 규칙). 호출 위치: tools.issue_compensation_coupon.
def fallback_compensation_won(reason: db.CompensationReason) -> int:
    """The amount of a coupon that is not due but was let through (P0 only): the lowest tier of the reason."""
    if reason == db.CompensationReason.DEFECTIVE_ITEM:
        return COUPON_DEFECTIVE_WON
    return COUPON_DELAY_SHORT_WON
