"""Tables of the service: chat sessions, the audit log and the approval queue.

They live in the same database as the shop tables but in their own metadata, so `db.dump_db` (the judged
state) never sees them.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, ForeignKey, Integer, MetaData, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from support_agent.db import NAMING_CONVENTION, UTCDateTime


class ServiceBase(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class SessionStatus(enum.StrEnum):
    OPEN = "open"
    HANDOFF = "handoff"  # the agent handed the conversation to a person
    CLOSED = "closed"  # too long to go on (turn or context limit)


class ApprovalStatus(enum.StrEnum):
    PENDING = "pending"
    EXECUTING = "executing"  # claimed by one decider; stays so only if the process died meanwhile
    APPROVED = "approved"  # approved and carried out
    REJECTED = "rejected"
    FAILED = "failed"  # approved, but the tool refused by then (the order had moved on)


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


class AuditEvent(ServiceBase):
    """Append-only. One row per customer message, LLM call, tool call, reply and approval step."""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    at: Mapped[datetime] = mapped_column(UTCDateTime)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


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


def approval_code(approval_id: int) -> str:
    return f"AP-{approval_id}"
