"""Korean time helpers. Every "now" is passed in by the caller; nothing here reads the wall clock."""

# [파일 역할] 한국 시간(KST) 변환과 도구 출력용 시각 표기를 맡는 작은 도우미 모듈이다.
# [설계 이유] 도구·seed·판정 코드는 현재 시각을 읽지 않는다(README 구조 절, docs/design.md 결정성). 시각은
# 과제에 적힌 고정 시각(ctx.now)뿐이라 "수령 후 7일" 같은 규정이 언제 돌려도 같게 판정된다. 그래서 이 모듈의
# 함수는 모두 시각을 인자로 받기만 하고 벽시계를 보지 않는다.
# [주요 구성] KST 상수, to_kst(시간대 변환), kst_date(KST 날짜), format_kst(분 단위 표기),
# format_kst_date(날짜 표기).
# [연결] tools.py(출력 표기, 쿠폰 기간 계산), rules.py(반품 기한·지연 일수), seed.py(도착 예정일),
# agent.py(시스템 프롬프트의 오늘 날짜), judge.py(과제 검증 메시지), user_sim.py가 쓴다.

from __future__ import annotations

# 날짜 계산은 표준 라이브러리 datetime만으로 한다. 외부 시간대 패키지에 기대지 않으려는 것이다(아래 KST 주석).
from datetime import date, datetime, timedelta, timezone

# A fixed offset instead of zoneinfo("Asia/Seoul"): Windows has no tz database without the tzdata package,
# and Korea has had no daylight saving time since 1988.
# UTC+9 고정 오프셋 시간대. DB에는 UTC로 저장하고 도구 출력과 날짜 경계(반품 기한 등)는 이 KST로
# 계산한다(docs/design.md 결정성).
KST = timezone(timedelta(hours=9), "KST")


# 시간대가 붙은 datetime을 KST로 바꾼다.
# 시간대 없는(naive) 값은 UTC인지 KST인지 알 수 없어 날짜 경계가 9시간 어긋날 수 있으므로, 조용히 추측하지
# 않고 ValueError로 막는다.
def to_kst(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("naive datetime is not allowed")
    return value.astimezone(KST)


# KST 기준 날짜만 꺼낸다. "9월 2일 수령 → 9월 9일까지"처럼 규정이 날짜 단위로 정해져 있어서(docs/design.md
# 규정 검사 표) UTC 날짜가 아니라 KST 날짜로 비교해야 한다. rules.py의 반품 기한·지연 일수, tools.py의 쿠폰
# 30일 집계가 이것을 쓴다.
def kst_date(value: datetime) -> date:
    return to_kst(value).date()


# 도구 출력에 넣는 시각 문자열. 초와 시간대 표기를 빼 모델과 고객이 읽기 쉬운 짧은 형태로 통일한다. None(아직
# 출고 안 됨 등)은 그대로 None을 돌려줘 JSON에서 null이 된다.
def format_kst(value: datetime | None) -> str | None:
    """`2026-09-02 14:20`, the form every tool output uses."""
    return None if value is None else to_kst(value).strftime("%Y-%m-%d %H:%M")


# 날짜만 필요한 출력용(예: track_shipment의 promised_by). date는 그대로, datetime은 KST 날짜로 바꾼 뒤 ISO
# 형식(2026-09-15)으로 적는다.
def format_kst_date(value: date | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        value = kst_date(value)
    return value.isoformat()
