from datetime import UTC, date, datetime

import pytest

from support_agent import db, rules
from support_agent.clock import KST

DELAY = db.CompensationReason.DELIVERY_DELAY
DEFECT = db.CompensationReason.DEFECTIVE_ITEM


def kst(month, day, hour=0, minute=0):
    return datetime(2026, month, day, hour, minute, tzinfo=KST)


def test_the_design_example_delivered_sep_2_can_be_returned_until_sep_9():
    delivered = kst(9, 2, 14, 20)
    assert rules.return_deadline(delivered) == date(2026, 9, 9)
    assert rules.within_return_window(kst(9, 9, 23, 59), delivered)
    assert not rules.within_return_window(kst(9, 10, 0, 0), delivered)


def test_the_deadline_uses_the_kst_date_of_delivery_not_the_utc_date():
    # 00:30 KST on Sep 3 is still Sep 2 in UTC.
    delivered = datetime(2026, 9, 2, 15, 30, tzinfo=UTC)
    assert rules.return_deadline(delivered) == date(2026, 9, 10)
    assert rules.return_deadline(kst(9, 2, 23, 30)) == date(2026, 9, 9)


def test_the_window_uses_the_kst_date_of_now_not_the_utc_date():
    delivered = kst(9, 2, 14, 20)
    # 00:00 KST on Sep 10 is 15:00 UTC on Sep 9: a UTC comparison would still allow it.
    assert not rules.within_return_window(datetime(2026, 9, 9, 15, 0, tzinfo=UTC), delivered)
    assert rules.within_return_window(datetime(2026, 9, 9, 14, 59, tzinfo=UTC), delivered)
    assert rules.within_return_window(delivered, delivered)


def test_naive_datetimes_are_rejected():
    with pytest.raises(ValueError):
        rules.return_deadline(datetime(2026, 9, 2, 14, 20))


def test_return_fee_and_refund():
    assert rules.return_fee_won(db.RequestReason.CHANGED_MIND) == 3000
    assert rules.return_fee_won(db.RequestReason.DEFECTIVE) == 0
    assert rules.return_fee_won(db.RequestReason.WRONG_ITEM) == 0
    items = [(38900, 1), (14700, 2)]
    assert rules.return_refund_won(items, db.RequestReason.CHANGED_MIND) == 65300
    assert rules.return_refund_won(items, db.RequestReason.DEFECTIVE) == 68300
    assert rules.return_refund_won([(2000, 1)], db.RequestReason.CHANGED_MIND) == 0  # never negative


def test_delay_days_counts_to_the_delivery_date_or_to_today():
    promised = date(2026, 9, 13)
    assert rules.delay_days(promised, None, kst(9, 14, 10)) == 1
    assert rules.delay_days(promised, None, kst(9, 13, 23, 59)) == 0
    assert rules.delay_days(promised, kst(9, 13, 23, 0), kst(9, 20)) == 0  # delivered on the promised date
    assert rules.delay_days(promised, kst(9, 11, 9, 0), kst(9, 20)) == -2
    # 00:10 KST on Sep 14 is still Sep 13 in UTC.
    assert rules.delay_days(promised, datetime(2026, 9, 13, 15, 10, tzinfo=UTC), kst(9, 20)) == 1
    assert rules.delay_days(promised, None, datetime(2026, 9, 13, 15, 10, tzinfo=UTC)) == 1


@pytest.mark.parametrize(
    ("days", "amount"), [(-1, None), (0, None), (1, 2000), (2, 2000), (3, 5000), (4, 5000), (30, 5000)]
)
def test_delivery_delay_tiers(days, amount):
    assert rules.compensation_amount_won(DELAY, days) == amount


def test_defective_item_amount_and_the_fallback_amounts():
    assert rules.compensation_amount_won(DEFECT, 0) == 3000
    assert rules.fallback_compensation_won(DELAY) == 2000
    assert rules.fallback_compensation_won(DEFECT) == 3000


def test_coupon_constants():
    assert (rules.COUPON_VALID_DAYS, rules.COUPON_WINDOW_DAYS, rules.COUPON_WINDOW_LIMIT) == (30, 30, 2)
    assert rules.RETURN_WINDOW_DAYS == 7
