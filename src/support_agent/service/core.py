"""One chat turn, the audit log and the approval queue. No web code here: app.py is a thin layer over this."""

from __future__ import annotations

import copy
import json
import secrets
import threading
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, select, update
from sqlalchemy.orm import Session

from support_agent.agent import AgentState, agent_turn, build_system_prompt, load_policy, new_state
from support_agent.chat import ChatProvider, Message, ProviderError, ToolCall
from support_agent.config import RunConfig
from support_agent.service.settings import Settings
from support_agent.service.store import (
    Approval,
    ApprovalStatus,
    AuditEvent,
    ChatSession,
    SessionStatus,
    approval_code,
)
from support_agent.toolkit import (
    ConversationState,
    Registry,
    ToolContext,
    ToolError,
    ToolResult,
    ToolSpec,
    execute,
)
from support_agent.tools import build_registry

# Writes whose answer carries `refund_won`. The amount is known only after the tool's own rules ran, so it
# is read from the handler's answer inside the transaction, which is rolled back when a person must decide.
APPROVAL_TOOLS = {"cancel_order": "취소", "request_return": "반품"}
TOOL_LABELS = {
    "find_customer": "본인 확인",
    "get_customer": "고객 정보 조회",
    "list_orders": "주문 목록 조회",
    "get_order": "주문 조회",
    "get_product": "상품 조회",
    "track_shipment": "배송 조회",
    "cancel_order": "주문 취소",
    "request_return": "반품 접수",
    "request_exchange": "교환 접수",
    "change_shipping_address": "배송지 변경",
    "issue_compensation_coupon": "보상 쿠폰 발급",
    "create_ticket": "상담 티켓 생성",
    "transfer_to_human": "상담원 연결",
    "think": "검토",
}
FALLBACK_REPLY = (
    "죄송합니다. 방금 요청을 처리하지 못했습니다. 다시 한 번 말씀해 주시거나 상담원 연결을 요청해 주세요."
)
CLOSED_REPLY = "대화가 길어져 이 상담은 여기서 마칩니다. 이어서 도움이 필요하시면 새 상담을 시작해 주세요."

Emit = Callable[[str, dict[str, Any]], None]


def wall_clock() -> datetime:
    """When something was recorded. `Settings.clock()` is the shop's clock and may be fixed for the demo."""
    return datetime.now(UTC)


class SessionNotFoundError(LookupError):
    pass


class SessionClosedError(RuntimeError):
    pass


class BusyError(RuntimeError):
    """The session is already answering a message (or an approval of it is being carried out)."""


class ApprovalStateError(RuntimeError):
    """The approval does not exist or was decided already."""


def _state_from(data: dict[str, Any]) -> AgentState:
    return AgentState(
        messages=[Message.from_dict(m) for m in data["messages"]],
        tool_errors=data.get("tool_errors", 0),
        format_errors=data.get("format_errors", 0),
        dropped_calls=data.get("dropped_calls", 0),
        stalls=data.get("stalls", 0),
        held_claims=data.get("held_claims", 0),
    )


def visible_messages(state: dict[str, Any]) -> list[dict[str, str]]:
    """What the customer saw: their own words and the replies that were delivered."""
    out = []
    for m in state["messages"]:
        if m["role"] == "user" and not m.get("harness"):
            out.append({"role": "customer", "text": m["content"]})
        elif (
            m["role"] == "assistant" and m["content"] and m.get("delivered", True) and not m.get("tool_calls")
        ):
            out.append({"role": "agent", "text": m["content"]})
    return out


class ChatService:
    def __init__(
        self,
        settings: Settings,
        engine: Engine,
        provider: ChatProvider,
        registry: Registry | None = None,
    ):
        self.settings = settings
        self.engine = engine
        self.provider = provider
        self.registry = registry or build_registry()
        self.policy_text = load_policy()
        self.config = RunConfig(
            model=settings.model,
            policy=settings.policy,
            num_ctx=settings.num_ctx,
            max_agent_calls=settings.max_agent_calls_per_turn,
            language="L1",  # never show a customer a reply that drifted into Chinese
            claims=settings.claims,
        )
        # One process serves the chat (see docs/design.md): a lock per session is enough.
        self._locks: dict[str, threading.Lock] = {}
        self._decisions = threading.Lock()
        self._locks_guard = threading.Lock()

    # ---------------------------------------------------------------- sessions

    def new_session(self, kind: str = "chat") -> dict[str, Any]:
        """`kind` "mcp" is a session without a conversation: an MCP client calls the tools itself."""
        now = wall_clock()
        state = new_state(build_system_prompt(self.policy_text, self.settings.clock()))
        if kind != "chat":
            state.messages.clear()
        session_id = ("" if kind == "chat" else f"{kind}-") + secrets.token_urlsafe(24)
        with Session(self.engine) as db_session:
            db_session.add(
                ChatSession(id=session_id, created_at=now, updated_at=now, turns=0, state=state.to_dict())
            )
            db_session.commit()
        self._audit(
            session_id,
            "session_started",
            {"kind": kind, "model": self.settings.model, "policy": self.settings.policy},
        )
        return self.transcript(session_id)

    def transcript(self, session_id: str) -> dict[str, Any]:
        with Session(self.engine) as db_session:
            row = self._row(db_session, session_id)
            pending = db_session.scalars(
                select(Approval)
                .where(Approval.session_id == session_id, Approval.status == ApprovalStatus.PENDING.value)
                .order_by(Approval.id)
            ).all()
            return {
                "session_id": row.id,
                "status": row.status,
                "messages": visible_messages(row.state),
                "pending_approvals": [
                    {"code": approval_code(a.id), "tool": a.tool, "refund_won": a.refund_won} for a in pending
                ],
            }

    # ---------------------------------------------------------------- one turn

    def handle(self, session_id: str, text: str, emit: Emit, *, via: dict[str, Any] | None = None) -> None:
        """Answer one customer message. Progress goes to `emit(event, data)`; the last event is `end`.
        `via` says how the message arrived when it was not typed (kept in the audit log)."""
        text = text.strip()
        if not text or len(text) > self.settings.max_message_chars:
            raise ValueError("the message is empty or too long")
        lock = self._lock(session_id)
        if not lock.acquire(blocking=False):
            raise BusyError(session_id)
        try:
            self._handle_locked(session_id, text, emit, via)
        finally:
            lock.release()

    def _handle_locked(
        self, session_id: str, text: str, emit: Emit, via: dict[str, Any] | None = None
    ) -> None:
        with Session(self.engine) as db_session:
            row = self._row(db_session, session_id)
            if row.status != SessionStatus.OPEN.value:
                raise SessionClosedError(row.status)
            state = _state_from(row.state)
            customer_id, turns = row.verified_customer_id, row.turns

        self._audit(session_id, "customer_message", {"text": text, **({"via": via} if via else {})})
        ctx = ToolContext(
            now=self.settings.clock(),
            enforce_policy=self.config.enforce_policy,
            state=ConversationState(verified_customer_id=customer_id),
        )

        def run_tool(name: str, arguments: dict) -> ToolResult:
            return self._run_tool(session_id, ctx, name, arguments, emit)

        status, reply, error, closing = SessionStatus.OPEN, None, "", ""
        try:
            turn = agent_turn(
                state,
                text,
                provider=self.provider,
                registry=self.registry,
                run_tool=run_tool,
                config=self.config,
                task_id=session_id,
            )
            reply = turn.reply
            if turn.stop == "handoff":
                status = SessionStatus.HANDOFF
            elif turn.stop == "context_limit":
                status, reply = SessionStatus.CLOSED, CLOSED_REPLY
            elif turn.stop:  # the model ran out of calls, retries or tool errors in this turn
                error, reply = turn.stop, FALLBACK_REPLY
        except Exception as exc:  # noqa: BLE001
            # Whatever the tools wrote before the failure is committed and audited; keep the conversation
            # consistent with it and tell the customer that this message was not answered.
            error = f"{type(exc).__name__}: {exc}"
            if not isinstance(exc, ProviderError):
                error += "\n" + traceback.format_exc()
            reply = FALLBACK_REPLY
            if state.messages[-1].tool_calls:  # the call that failed has no answer; do not keep half of it
                state.messages.pop()
        if reply is not None and (error or status is SessionStatus.CLOSED):
            state.messages.append(Message("assistant", reply))
        if turns + 1 >= self.settings.max_turns_per_session and status is SessionStatus.OPEN:
            status = SessionStatus.CLOSED
            closing = CLOSED_REPLY  # its own message, as in the stored conversation
            state.messages.append(Message("assistant", CLOSED_REPLY))

        for call in state.llm_log:
            self._audit(
                session_id,
                "llm_call",
                {
                    "format_error": call.format_error,
                    "finish_reason": call.finish_reason,
                    "tool_calls": [c["name"] for c in call.tool_calls],
                    "prompt_tokens": call.prompt_tokens,
                    "completion_tokens": call.completion_tokens,
                    "wall_ms": round(call.wall_ms, 1),
                },
            )
        if error:
            self._audit(session_id, "error", {"error": error})
        self._audit(session_id, "agent_reply", {"text": reply or "", "status": status.value})
        with Session(self.engine) as db_session:
            row = self._row(db_session, session_id)
            row.state = state.to_dict()
            row.turns = turns + 1
            row.status = status.value
            row.verified_customer_id = ctx.state.verified_customer_id
            row.updated_at = wall_clock()
            db_session.commit()
        if error:
            emit("error", {"message": "처리 중 문제가 생겼습니다."})
        for reply_text in (reply, closing):
            if reply_text:
                emit("reply", {"text": reply_text})
        emit("end", {"status": status.value})

    def _run_tool(
        self,
        session_id: str,
        ctx: ToolContext,
        name: str,
        arguments: dict[str, Any],
        emit: Emit,
        *,
        approval_id: int | None = None,
    ) -> ToolResult:
        """The one way a tool runs in the service.

        Inside the tool's own transaction (toolkit's on_success hook): the refund the handler really computed
        is held against the approval threshold, the audit row is written, and an approval that is being
        carried out is marked as done. So a change of the shop data never exists without its audit row, and
        no amount above the threshold is committed without a person. `approval_id` is set when that person
        has decided.
        """
        emit("tool", {"name": name, "label": TOOL_LABELS.get(name, name)})
        held: dict[str, Any] = {}
        seen = len(ctx.violations)

        def payload(args, ok, content, error_code=None, policy_blocked=False) -> dict[str, Any]:
            spec = self.registry.get(name)
            return {
                "name": name,
                "arguments": arguments,
                "args": args,
                "write": bool(spec and spec.write),
                "ok": ok,
                "error_code": error_code,
                "policy_blocked": policy_blocked,
                "violations": list(ctx.violations[seen:]) if ok else [],
                "content": content,
                "customer_id": ctx.state.verified_customer_id,
                **({"approval": approval_code(approval_id)} if approval_id is not None else {}),
            }

        def in_transaction(db_session: Session, spec: ToolSpec, clean: dict[str, Any], answer: dict) -> None:
            refund = answer.get("refund_won")
            if (
                approval_id is None
                and spec.name in APPROVAL_TOOLS
                and isinstance(refund, int)
                and refund >= self.settings.approval_refund_won
            ):
                held.update(refund=refund, args=clean)
                raise ToolError("approval_required", "held for approval")  # rolls the call back
            content = json.dumps(answer, ensure_ascii=False)
            db_session.add(
                AuditEvent(
                    session_id=session_id,
                    at=wall_clock(),
                    kind="tool_call",
                    payload=payload(clean, True, content),
                )
            )
            if approval_id is not None:
                approval = db_session.get(Approval, approval_id)
                approval.status = ApprovalStatus.APPROVED.value
                approval.result = content

        result = execute(self.registry, self.engine, ctx, name, arguments, on_success=in_transaction)
        if held:
            result = self._queue_approval(session_id, ctx, name, held["args"], held["refund"], emit)
        if not result.ok:  # nothing was changed, so this row may stand alone
            self._audit(
                session_id,
                "tool_call",
                payload(result.args, False, result.content, result.error_code, result.policy_blocked),
            )
        emit("tool_result", {"name": name, "ok": result.ok, "error_code": result.error_code})
        return result

    def call_tool(self, session_id: str, name: str, arguments: dict[str, Any]) -> ToolResult:
        """A tool call that comes from outside the agent loop (the MCP server). Same gate, same audit."""
        lock = self._lock(session_id)
        if not lock.acquire(timeout=30):
            raise BusyError(session_id)
        try:
            with Session(self.engine) as db_session:
                customer_id = self._row(db_session, session_id).verified_customer_id
            ctx = ToolContext(
                now=self.settings.clock(),
                enforce_policy=self.config.enforce_policy,
                state=ConversationState(verified_customer_id=customer_id),
            )
            result = self._run_tool(session_id, ctx, name, arguments, lambda _event, _data: None)
            with Session(self.engine) as db_session:
                row = self._row(db_session, session_id)
                row.verified_customer_id = ctx.state.verified_customer_id
                row.turns += 1
                row.updated_at = wall_clock()
                db_session.commit()
            return result
        finally:
            lock.release()

    # ---------------------------------------------------------------- approvals

    def _queue_approval(
        self, session_id: str, ctx: ToolContext, tool: str, clean: dict[str, Any], refund: int, emit: Emit
    ) -> ToolResult:
        """Put the write that was just rolled back into the queue (once per session and arguments)."""
        with Session(self.engine) as db_session:
            pending = db_session.scalars(
                select(Approval).where(
                    Approval.session_id == session_id,
                    Approval.tool == tool,
                    Approval.status == ApprovalStatus.PENDING.value,
                )
            ).all()
            approval = next((a for a in pending if a.args == clean), None)
            if approval is None:
                approval = Approval(
                    session_id=session_id,
                    created_at=wall_clock(),
                    tool=tool,
                    args=clean,
                    customer_id=ctx.state.verified_customer_id,
                    refund_won=refund,
                )
                db_session.add(approval)
                db_session.commit()
            new_id = approval.id
        code = approval_code(new_id)
        self._audit(
            session_id,
            "approval_requested",
            {"code": code, "tool": tool, "args": clean, "refund_won": refund},
        )
        emit("approval", {"code": code, "tool": tool, "refund_won": refund})
        return ToolResult.error(
            "approval_required",
            f"환불액 {refund:,}원은 담당자 승인이 필요해 아직 처리되지 않았고, 승인 대기열에 올렸습니다"
            f"(승인 번호 {code}). 고객에게 담당자 확인 후 처리되며 결과는 이 대화창으로 안내된다고 알리세요. "
            "같은 요청을 다시 호출하지 마세요.",
            args=clean,
        )

    def decide(self, approval_id: int, *, approve: bool, by: str, note: str = "") -> dict[str, Any]:
        """Approve (and carry out) or reject a waiting write. The customer reads the outcome in the chat."""
        with Session(self.engine) as db_session:
            approval = db_session.get(Approval, approval_id)
            if approval is None:
                raise ApprovalStateError("no such approval")
            session_id = approval.session_id
        lock = self._lock(session_id)  # a running turn of that session would overwrite the conversation
        if not lock.acquire(timeout=30):
            raise BusyError(session_id)
        try:
            with self._decisions:  # one approved write at a time, whatever the session
                return self._decide_locked(approval_id, approve=approve, by=by, note=note.strip())
        finally:
            lock.release()

    def _decide_locked(self, approval_id: int, *, approve: bool, by: str, note: str) -> dict[str, Any]:
        now = wall_clock()
        claimed = ApprovalStatus.EXECUTING if approve else ApprovalStatus.REJECTED
        with Session(self.engine) as db_session:
            # The claim is one conditional UPDATE: of two deciders, in this process or another, one wins.
            won = db_session.execute(
                update(Approval)
                .where(Approval.id == approval_id, Approval.status == ApprovalStatus.PENDING.value)
                .values(status=claimed.value, decided_at=now, decided_by=by, note=note)
            ).rowcount
            db_session.commit()
            if won != 1:
                raise ApprovalStateError("decided already")
            approval = db_session.get(Approval, approval_id)
            session_id, tool, args = approval.session_id, approval.tool, dict(approval.args)
            customer_id = approval.customer_id

        kind, order_id = APPROVAL_TOOLS[tool], args.get("order_id", "")
        status, result = claimed, None
        if approve:
            ctx = ToolContext(
                now=self.settings.clock(),
                enforce_policy=self.config.enforce_policy,
                state=ConversationState(verified_customer_id=customer_id),
            )
            # The tool's changes, its audit row and the approval's "approved" are one transaction.
            result = self._run_tool(session_id, ctx, tool, args, lambda _e, _d: None, approval_id=approval_id)
            status = ApprovalStatus.APPROVED if result.ok else ApprovalStatus.FAILED

        if status is ApprovalStatus.APPROVED:
            answer = json.loads(result.content)
            text = (
                f"담당자 확인이 끝나 주문 {order_id}의 {kind} 요청이 처리되었습니다. "
                f"환불 예정 금액은 {answer['refund_won']:,}원입니다."
            )
        elif status is ApprovalStatus.FAILED:
            text = (
                f"담당자가 주문 {order_id}의 {kind} 요청을 승인했지만, 그 사이 주문 상태가 바뀌어 처리하지 "
                "못했습니다. 다시 확인이 필요하시면 말씀해 주세요."
            )
        else:
            text = f"담당자 검토 결과 주문 {order_id}의 {kind} 요청은 승인되지 않았습니다."
            if note:
                text += f" 사유: {note}"

        with Session(self.engine) as db_session:
            if status is ApprovalStatus.FAILED:
                approval = db_session.get(Approval, approval_id)
                approval.status = status.value
                approval.result = result.content
            row = self._row(db_session, session_id)
            state = copy.deepcopy(row.state)
            if status is ApprovalStatus.APPROVED and state["messages"]:  # an MCP session has no conversation
                # The conversation keeps the write like any other tool call: the model reads what was done,
                # and the claim guard finds the evidence when the agent later says that it is done.
                state["messages"] += [
                    Message("assistant", "", (ToolCall(tool, args),)).to_dict(),
                    Message("tool", result.content, tool_name=tool).to_dict(),
                ]
            state["messages"].append(Message("assistant", text).to_dict())
            row.state = state
            row.updated_at = now
            db_session.commit()
        self._audit(
            session_id,
            "approval_decided",
            {
                "code": approval_code(approval_id),
                "status": status.value,
                "by": by,
                "note": note,
                "tool": tool,
                "args": args,
                "content": result.content if result else "",
            },
        )
        return self.approval(approval_id)

    # ---------------------------------------------------------------- admin reads

    def approval(self, approval_id: int) -> dict[str, Any]:
        with Session(self.engine) as db_session:
            approval = db_session.get(Approval, approval_id)
            if approval is None:
                raise ApprovalStateError("no such approval")
            return _approval_dict(approval)

    def approvals(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        query = select(Approval).order_by(Approval.id.desc()).limit(limit)
        if status:
            query = query.where(Approval.status == status)
        with Session(self.engine) as db_session:
            return [_approval_dict(a) for a in db_session.scalars(query)]

    def sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        query = select(ChatSession).order_by(ChatSession.updated_at.desc()).limit(limit)
        with Session(self.engine) as db_session:
            return [
                {
                    "session_id": row.id,
                    "status": row.status,
                    "turns": row.turns,
                    "customer_id": row.verified_customer_id,
                    "created_at": row.created_at.isoformat(),
                    "updated_at": row.updated_at.isoformat(),
                }
                for row in db_session.scalars(query)
            ]

    def audit(self, session_id: str) -> list[dict[str, Any]]:
        with Session(self.engine) as db_session:
            self._row(db_session, session_id)
            events = db_session.scalars(
                select(AuditEvent).where(AuditEvent.session_id == session_id).order_by(AuditEvent.id)
            )
            return [
                {"id": e.id, "at": e.at.isoformat(), "kind": e.kind, "payload": e.payload} for e in events
            ]

    # ---------------------------------------------------------------- helpers

    def _row(self, db_session: Session, session_id: str) -> ChatSession:
        row = db_session.get(ChatSession, session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        return row

    def _lock(self, session_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(session_id, threading.Lock())

    def _audit(self, session_id: str, kind: str, payload: dict[str, Any]) -> None:
        with Session(self.engine) as db_session:
            db_session.add(AuditEvent(session_id=session_id, at=wall_clock(), kind=kind, payload=payload))
            db_session.commit()


def _approval_dict(a: Approval) -> dict[str, Any]:
    return {
        "id": a.id,
        "code": approval_code(a.id),
        "session_id": a.session_id,
        "created_at": a.created_at.isoformat(),
        "tool": a.tool,
        "label": TOOL_LABELS.get(a.tool, a.tool),
        "args": a.args,
        "customer_id": a.customer_id,
        "refund_won": a.refund_won,
        "status": a.status,
        "decided_at": a.decided_at.isoformat() if a.decided_at else None,
        "decided_by": a.decided_by,
        "note": a.note,
        "result": a.result,
    }
