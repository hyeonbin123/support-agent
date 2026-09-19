"""Policy arithmetic: return deadline, refund amounts, compensation amounts.

Pure functions. No DB, no clock reads: every "now" is an argument. Calendar rules use KST dates.
The numbers here are the ones written in prompts/policy.md (a test keeps the two in step).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from support_agent import db
from support_agent.clock import kst_date

RETURN_WINDOW_DAYS = 7
RETURN_FEE_WON = 3000  # charged only when the customer changed their mind

COUPON_DELAY_SHORT_WON = 2000  # 1-2 days late
COUPON_DELAY_LONG_WON = 5000  # 3 days late or more
COUPON_DELAY_LONG_FROM_DAYS = 3
COUPON_DEFECTIVE_WON = 3000
COUPON_VALID_DAYS = 30
COUPON_WINDOW_DAYS = 30
COUPON_WINDOW_LIMIT = 2


def return_deadline(delivered_at: datetime) -> date:
    """The last KST date on which a return or exchange can be requested."""
    return kst_date(delivered_at) + timedelta(days=RETURN_WINDOW_DAYS)


def within_return_window(now: datetime, delivered_at: datetime) -> bool:
    return kst_date(now) <= return_deadline(delivered_at)


def return_fee_won(reason: db.RequestReason) -> int:
    return RETURN_FEE_WON if reason == db.RequestReason.CHANGED_MIND else 0


def return_refund_won(items: list[tuple[int, int]], reason: db.RequestReason) -> int:
    """Sum of unit price x quantity minus the return shipping fee, never below zero."""
    goods = sum(unit_price * quantity for unit_price, quantity in items)
    return max(0, goods - return_fee_won(reason))


def delay_days(promised_by: date, delivered_at: datetime | None, now: datetime) -> int:
    """Days past the promised date, counted to the delivery date or, if not delivered yet, to today (KST).

    Zero or negative means on time.
    """
    reached = kst_date(delivered_at) if delivered_at is not None else kst_date(now)
    return (reached - promised_by).days


def compensation_amount_won(reason: db.CompensationReason, delay_days: int) -> int | None:
    """The coupon amount the policy fixes, or None when a delivery-delay coupon is not due.

    `defective_item` always has an amount; whether a defective request exists is a DB fact the tool checks.
    """
    if reason == db.CompensationReason.DEFECTIVE_ITEM:
        return COUPON_DEFECTIVE_WON
    if delay_days <= 0:
        return None
    return COUPON_DELAY_LONG_WON if delay_days >= COUPON_DELAY_LONG_FROM_DAYS else COUPON_DELAY_SHORT_WON


def fallback_compensation_won(reason: db.CompensationReason) -> int:
    """The amount of a coupon that is not due but was let through (P0 only): the lowest tier of the reason."""
    if reason == db.CompensationReason.DEFECTIVE_ITEM:
        return COUPON_DEFECTIVE_WON
    return COUPON_DELAY_SHORT_WON
