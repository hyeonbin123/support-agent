"""Korean time helpers. Every "now" is passed in by the caller; nothing here reads the wall clock."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

# A fixed offset instead of zoneinfo("Asia/Seoul"): Windows has no tz database without the tzdata package,
# and Korea has had no daylight saving time since 1988.
KST = timezone(timedelta(hours=9), "KST")


def to_kst(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("naive datetime is not allowed")
    return value.astimezone(KST)


def kst_date(value: datetime) -> date:
    return to_kst(value).date()


def format_kst(value: datetime | None) -> str | None:
    """`2026-09-02 14:20`, the form every tool output uses."""
    return None if value is None else to_kst(value).strftime("%Y-%m-%d %H:%M")


def format_kst_date(value: date | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        value = kst_date(value)
    return value.isoformat()
