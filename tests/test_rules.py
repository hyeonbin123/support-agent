# [테스트 묶음] rules.py의 규정 계산(반품 기한, 반품 배송비와 환불액, 배송 지연 일수, 보상 쿠폰 금액, 쿠폰
# 상수)을 순수 함수 단위로 확인한다.
# [왜 중요한가] 도구의 규정 검사와 판정이 모두 이 계산에 기댄다. 값은 규정 문서(prompts/policy.md)와
# docs/design.md "금액 규칙" 표와 같아야 한다. 규정 문서에 없는 규칙을 코드가 막으면 에이전트가 알 수 없는
# 실패가 된다.
# [핵심] 날짜 비교는 UTC가 아니라 KST 날짜로 한다. 저장은 UTC라서, 자정 근처 시각에서 하루가 어긋나는 버그를
# 여기서 막는다.

from datetime import UTC, date, datetime

import pytest

from support_agent import db, rules
from support_agent.clock import KST

# 보상 쿠폰 사유 두 가지(배송 지연, 불량)를 짧은 이름으로 둔다.
DELAY = db.CompensationReason.DELIVERY_DELAY
DEFECT = db.CompensationReason.DEFECTIVE_ITEM


# 2026년 KST 시각을 만드는 도우미 (기본 자정).
def kst(month, day, hour=0, minute=0):
    return datetime(2026, month, day, hour, minute, tzinfo=KST)


# docs/design.md의 예: 9월 2일 수령이면 다음 날부터 7일째인 9월 9일까지 반품할 수 있다. 9월 9일 23:59는 되고
# 9월 10일 0시는 안 된다.
def test_the_design_example_delivered_sep_2_can_be_returned_until_sep_9():
    delivered = kst(9, 2, 14, 20)
    assert rules.return_deadline(delivered) == date(2026, 9, 9)
    assert rules.within_return_window(kst(9, 9, 23, 59), delivered)
    assert not rules.within_return_window(kst(9, 10, 0, 0), delivered)


# 수령 시각의 날짜는 KST로 정한다. UTC 9월 2일 15:30은 KST 9월 3일 00:30이므로 기한이 9월 10일이다.
def test_the_deadline_uses_the_kst_date_of_delivery_not_the_utc_date():
    # 9월 3일 00:30 KST는 UTC로는 아직 9월 2일이다.
    delivered = datetime(2026, 9, 2, 15, 30, tzinfo=UTC)
    assert rules.return_deadline(delivered) == date(2026, 9, 10)
    assert rules.return_deadline(kst(9, 2, 23, 30)) == date(2026, 9, 9)


# '지금'의 날짜도 KST로 정한다. KST 9월 10일 0시(UTC 9월 9일 15:00)는 이미 기한이 지났다. UTC 날짜로 비교하면
# 하루를 더 허용하는 버그가 된다. 수령 당일도 기한 안이다.
def test_the_window_uses_the_kst_date_of_now_not_the_utc_date():
    delivered = kst(9, 2, 14, 20)
    # 9월 10일 00:00 KST는 UTC로 9월 9일 15:00이다: UTC로 비교하면 아직 허용하게 된다.
    assert not rules.within_return_window(datetime(2026, 9, 9, 15, 0, tzinfo=UTC), delivered)
    assert rules.within_return_window(datetime(2026, 9, 9, 14, 59, tzinfo=UTC), delivered)
    assert rules.within_return_window(delivered, delivered)


# 시간대가 없는 datetime은 KST인지 UTC인지 알 수 없으므로 ValueError로 거절해야 한다.
def test_naive_datetimes_are_rejected():
    with pytest.raises(ValueError):
        rules.return_deadline(datetime(2026, 9, 2, 14, 20))


# 반품 배송비: 단순 변심 3,000원, 불량·오배송 0원. 환불액 = Σ(단가 × 수량) − 배송비이고, 배송비가 상품값보다
# 커도 음수가 되지 않는다.
def test_return_fee_and_refund():
    assert rules.return_fee_won(db.RequestReason.CHANGED_MIND) == 3000
    assert rules.return_fee_won(db.RequestReason.DEFECTIVE) == 0
    assert rules.return_fee_won(db.RequestReason.WRONG_ITEM) == 0
    items = [(38900, 1), (14700, 2)]
    assert rules.return_refund_won(items, db.RequestReason.CHANGED_MIND) == 65300
    assert rules.return_refund_won(items, db.RequestReason.DEFECTIVE) == 68300
    assert rules.return_refund_won([(2000, 1)], db.RequestReason.CHANGED_MIND) == 0  # 음수가 되지 않는다


# 늦은 일수 = (수령일, 아직 못 받았으면 오늘) − 도착 예정일, 모두 KST 날짜. 예정일에 받으면 0, 일찍 받으면
# 음수이고, UTC로는 전날인 KST 자정 직후 수령도 KST 날짜로 센다.
def test_delay_days_counts_to_the_delivery_date_or_to_today():
    promised = date(2026, 9, 13)
    assert rules.delay_days(promised, None, kst(9, 14, 10)) == 1
    assert rules.delay_days(promised, None, kst(9, 13, 23, 59)) == 0
    assert rules.delay_days(promised, kst(9, 13, 23, 0), kst(9, 20)) == 0  # 도착 예정일에 배송됨
    assert rules.delay_days(promised, kst(9, 11, 9, 0), kst(9, 20)) == -2
    # 9월 14일 00:10 KST는 UTC로는 아직 9월 13일이다.
    assert rules.delay_days(promised, datetime(2026, 9, 13, 15, 10, tzinfo=UTC), kst(9, 20)) == 1
    assert rules.delay_days(promised, None, datetime(2026, 9, 13, 15, 10, tzinfo=UTC)) == 1


# 배송 지연 보상 금액 구간: 늦지 않았으면 자격 없음(None), 1~2일 2,000원, 3일 이상 5,000원 (docs/design.md
# 보상 표).
@pytest.mark.parametrize(
    ("days", "amount"), [(-1, None), (0, None), (1, 2000), (2, 2000), (3, 5000), (4, 5000), (30, 5000)]
)
def test_delivery_delay_tiers(days, amount):
    assert rules.compensation_amount_won(DELAY, days) == amount


# 불량 보상은 3,000원. 대체 금액은 P0(규정을 프롬프트에만 둠)에서 자격 없이 통과된 쿠폰에 쓰는 사유별 최소
# 금액(지연 2,000원, 불량 3,000원)이다.
def test_defective_item_amount_and_the_fallback_amounts():
    assert rules.compensation_amount_won(DEFECT, 0) == 3000
    assert rules.fallback_compensation_won(DELAY) == 2000
    assert rules.fallback_compensation_won(DEFECT) == 3000


# 규정 상수: 쿠폰 유효 30일, 고객당 최근 30일에 2장까지, 반품 기한 7일. 숫자가 바뀌면 규정 문서와 과제 정답도
# 함께 바뀌어야 하므로 값 자체를 고정한다.
def test_coupon_constants():
    assert (rules.COUPON_VALID_DAYS, rules.COUPON_WINDOW_DAYS, rules.COUPON_WINDOW_LIMIT) == (30, 30, 2)
    assert rules.RETURN_WINDOW_DAYS == 7
