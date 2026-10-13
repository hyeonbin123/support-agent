"""Tables of the service: chat sessions, the audit log and the approval queue.

They live in the same database as the shop tables but in their own metadata, so `db.dump_db` (the judged
state) never sees them.
"""

# [파일 역할] 서비스에만 있는 테이블 3개(chat_sessions, audit_events, approvals)를 SQLAlchemy ORM으로
# 선언한다.
# [설계 이유] 쇼핑몰 테이블 13개(db.py)와 같은 DB에 두지만 메타데이터(ServiceBase)를 따로 둔다. 평가의 판정은
# db.Base의 테이블만 덤프해 비교하므로(db.dump_db), 서비스 기록이 판정 대상 상태에 섞이지
# 않는다(docs/service.md 파일 표).
# [연결] service/core.py가 세션 저장·감사 로그·승인 대기열에 쓰고, migrations/env.py가 db.Base와 함께
# 마이그레이션 대상으로 묶는다. tests/test_postgres.py는 마이그레이션 결과가 이 선언과 같은지 확인한다.

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

# SQLAlchemy 2의 타입 힌트 기반 선언(Mapped, mapped_column)을 쓴다. JSON 열은 SQLite와 PostgreSQL 양쪽에서
# 같은 코드로 dict를 저장하려는 것이다.
from sqlalchemy import JSON, ForeignKey, Integer, MetaData, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# 쇼핑몰 테이블과 같은 제약 이름 규칙과 UTC 시각 타입을 써서, 두 메타데이터를 한 Alembic 마이그레이션으로 다룰
# 때 이름·시각 처리가 어긋나지 않게 한다. UTCDateTime은 시간대 없는 시각을 거절한다(db.py).
from support_agent.db import NAMING_CONVENTION, UTCDateTime


# 서비스 테이블 전용 선언 기반 클래스. db.Base와 다른 MetaData를 가지므로 db.dump_db가 이 테이블들을 보지
# 않는다.
# naming_convention을 주면 인덱스·외래 키 이름이 정해진 규칙으로 만들어져, 마이그레이션 파일과 선언을 비교할
# 때 이름이 일치한다.
class ServiceBase(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


# 상담 세션의 상태. open일 때만 새 메시지를 받는다. handoff는 상담원 이관(transfer_to_human)으로, closed는 턴
# 한도·컨텍스트 한도로 닫힌 것이다(docs/service.md "한도").
# StrEnum이라 값이 그대로 문자열 열에 들어간다.
class SessionStatus(enum.StrEnum):
    OPEN = "open"
    HANDOFF = "handoff"  # the agent handed the conversation to a person
    CLOSED = "closed"  # too long to go on (turn or context limit)


# 사람 승인 대기열 한 건의 상태 흐름: pending → executing(한 결정자가 선점) → approved/rejected/failed.
# 선점은 core.py에서 조건부 UPDATE 한 번으로 해서, 두 담당자나 두 프로세스가 같은 승인을 동시에 눌러도 한쪽만
# 실행한다(docs/service.md "사람 승인").
class ApprovalStatus(enum.StrEnum):
    PENDING = "pending"
    # Claimed by one decider. A write that raises gives it back (pending); it stays only if the process died
    # meanwhile or the database could not be reached to give it back.
    EXECUTING = "executing"
    APPROVED = "approved"  # approved and carried out
    REJECTED = "rejected"
    FAILED = "failed"  # approved, but the tool refused by then (the order had moved on)


# 고객 상담 한 건. 고객 쪽에는 로그인이 없고 세션 id 자체가 열쇠이므로, id는 추측할 수 없는
# 난수다(docs/service.md "세션 id가 곧 열쇠다").
# turns는 세션당 턴 한도(기본 40)를 세는 데 쓰고, verified_customer_id는 find_customer로 본인 확인된 고객
# 번호다. state에는 에이전트 대화 상태 전체를 JSON으로 저장해 다음 요청에서 그대로 이어 간다.
class ChatSession(ServiceBase):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # unguessable: it is the only credential
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime)
    status: Mapped[str] = mapped_column(String(16), default=SessionStatus.OPEN.value)
    turns: Mapped[int] = mapped_column(Integer, default=0)
    verified_customer_id: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # AgentState.to_dict(): every message the model sees, including tool results and harness notices.
    state: Mapped[dict[str, Any]] = mapped_column(JSON)


# 감사 로그. 행을 고치거나 지우지 않고 추가만 한다. kind로 사건 종류(고객 메시지, LLM 호출, 도구 호출, 응답,
# 승인 요청·결정 등)를 나누고 세부는 payload JSON에 담는다.
# session_id와 kind에 인덱스를 두어 관리 화면이 세션별·종류별로 찾을 수 있게 한다.
class AuditEvent(ServiceBase):
    """Append-only. One row per customer message, LLM call, tool call, reply and approval step."""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    at: Mapped[datetime] = mapped_column(UTCDateTime)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


# 사람 승인 대기열의 한 건. 환불액이 기준(기본 10만 원) 이상인 취소·반품 호출이 여기에 올라간다.
# args는 검증을 거친 정규 인자이고, 승인되면 이 인자 그대로 도구를 실행한다. customer_id는 요청 당시 본인
# 확인된 고객이다. 승인 실행 때(core.py의 _decide_locked) 이 고객을 본인 확인된 상태로 넣고 도구를 다시
# 돌리므로 소유·규정 검사를 다시 거친다.
# status에 인덱스를 두어 관리 화면이 대기 중인 건만 빠르게 고른다. result에는 실행 뒤 도구의 응답 문자열이
# 남는다.
class Approval(ServiceBase):
    __tablename__ = "approvals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)
    tool: Mapped[str] = mapped_column(String(64))
    args: Mapped[dict[str, Any]] = mapped_column(JSON)  # canonical arguments, run as they are on approval
    customer_id: Mapped[str | None] = mapped_column(String(20), nullable=True)  # verified when it was asked
    refund_won: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default=ApprovalStatus.PENDING.value, index=True)
    decided_at: Mapped[datetime | None] = mapped_column(UTCDateTime, nullable=True)
    decided_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")
    result: Mapped[str] = mapped_column(Text, default="")  # the tool's answer once it ran


# 승인 번호의 화면 표기(AP-1 등). 고객 안내 문장, 관리 화면, 감사 로그가 모두 이 함수로 같은 표기를
# 쓴다(core.py에서 호출).
def approval_code(approval_id: int) -> str:
    return f"AP-{approval_id}"
