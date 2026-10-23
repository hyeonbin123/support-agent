"""채팅 한 턴, 감사 로그, 승인 대기열. 웹 코드는 여기 없다: app.py는 이 위의 얇은 층이다."""

# [파일 역할] 채팅 서비스의 핵심. 고객 메시지 한 개에 답하는 한 턴(handle), 모든 도구 실행이 지나는
# 관문(_run_tool), 감사 로그, 사람 승인 대기열(_queue_approval, decide), 관리 화면용 조회를 맡는다.
# [설계 이유] 웹 코드(FastAPI)를 두지 않아 테스트가 이 클래스를 직접 부르고, 웹(app.py)과 MCP
# 서버(mcp_server.py)가 같은 객체를 공유한다. 에이전트 루프(agent.agent_turn)와 도구 실행
# 경로(toolkit.execute)는 평가와 바이트까지 같은 것을 쓰고, 서비스에만 필요한 것(승인 관문, 감사 행, 고객에게
# 보내는 글 고치기)은 그 바깥 훅으로만 붙인다. 그래서 평가가 잰 것과 서비스가 도는 것이 달라지지
# 않는다(docs/service.md "사람 승인"의 "이렇게 한 이유").
# [주요 구성] (1) 승인 대기 턴에서 에이전트용 지시를 고객 답에서 빼는 함수들(without_agent_only 등) (2)
# ChatService: 세션, 한 턴, 도구 실행, 승인, 관리자 조회.

from __future__ import annotations

# copy: 승인 결과를 대화에 덧붙일 때 저장된 상태 dict를 깊은 복사해 고친다(SQLAlchemy가 JSON 열의 변경을
# 알아채도록 새 객체로 바꿔 넣음). secrets: 세션 id 난수. threading: 세션별 잠금과 승인 결정 잠금. traceback:
# 예상 못 한 예외의 위치를 감사 로그에 남긴다. contextmanager: 세션별 잠금을 얻고 놓고 다 쓴 잠금을 지우는
# 일을 with 블록 하나(ChatService._hold)로 묶는다.
import copy
import json
import re
import secrets
import threading
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Engine, func, select, update
from sqlalchemy.orm import Session

# 평가와 공유하는 모듈: 에이전트 루프와 시스템 프롬프트(agent), 메시지 타입(chat), 실행 설정(RunConfig), 도구
# 실행 경로(toolkit), 도구 등록부(tools). 서비스 전용은 settings와 store(서비스 테이블)뿐이다.
from support_agent import db
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

# 승인 관문에 걸릴 수 있는 쓰기 도구와 고객 안내에 쓸 한국어 이름. 환불액은 도구가 규정 계산을 마쳐야 알 수
# 있으므로, 미리 계산해 보는 단계 없이 도구의 트랜잭션 안에서 실제 refund_won을 읽어 판단한다. 그래서 "확인할
# 때는 기준 아래, 실행할 때는 위"인 틈이 없다(docs/service.md). 대상은 답에 `refund_won`이 들어 있는 쓰기
# 도구다. 사람이 결정해야 하면 그 트랜잭션은 되돌린다.
APPROVAL_TOOLS = {"cancel_order": "취소", "request_return": "반품"}
# 도구 이름 → 화면 표기. 고객 화면의 진행 표시("주문 조회 중" 같은 tool 이벤트), 관리 화면의 승인 카드, MCP
# 서버의 도구 제목(mcp_server.py)이 쓴다.
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
# 턴이 한도(LLM 호출 수, 형식 재시도, 도구 오류 수)에 닿거나 예외가 났을 때 고객에게 가는 고정 안내. 모델이
# 아무 말도 못 했을 때도 고객이 다음에 할 일을 알게 한다. 그 턴에 승인 대기에 걸린 쓰기가 있으면 이것 대신
# 아래 HELD_FALLBACK_REPLY와 그 쓰기의 안내가 간다.
FALLBACK_REPLY = (
    "죄송합니다. 방금 요청을 처리하지 못했습니다. 다시 한 번 말씀해 주시거나 상담원 연결을 요청해 주세요."
)
# 한도·예외로 끝난 턴인데 그 턴에 승인 대기에 걸린 쓰기가 있을 때의 고정 문장. 뒤에 그 쓰기의 서비스
# 안내(HELD_NOTICE)가 붙는다. FALLBACK_REPLY의 '처리하지 못했습니다'만 가면, 실제로는 담당자 승인을 기다리는
# 쓰기가 올라가 있는데 고객은 아무것도 되지 않았다고 읽는다. 그래서 턴이 끝까지 가지 못했다고만 말한다.
HELD_FALLBACK_REPLY = (
    "죄송합니다. 방금 요청을 끝까지 처리하지 못했습니다. "
    "다시 한 번 말씀해 주시거나 상담원 연결을 요청해 주세요."
)
# 컨텍스트가 차거나 세션당 턴 한도(기본 40)에 닿아 세션을 닫을 때의 고정 안내.
CLOSED_REPLY = "대화가 길어져 이 상담은 여기서 마칩니다. 이어서 도움이 필요하시면 새 상담을 시작해 주세요."

# 승인 대기 도구 결과 끝에 붙는, 에이전트에게만 하는 지시 두 문장. 사람을 기다리는 쓰기는 에이전트에게 사실과
# 이 지시를 함께 돌려준다. 문구는 관문을 만들 때부터 그대로이고(모델의 입력을 바꾸지 않으려고), 고객에게 가는
# 답에서만 빼낸다. 모델이 이 지시를 글자 그대로 고객에게 옮길 수 있기 때문이다(docs/service.md, 2026-10-04의
# qwen3.5:4b 턴). 그래서 쓰기가 대기에 걸린 턴에서 고객은 지시를 뺀 모델의 답(without_agent_only)과 그 뒤의
# 서비스 고정 안내(HELD_NOTICE)를 읽는다.
HELD_AGENT_ONLY = (
    "고객에게 담당자 확인 후 처리되며 결과는 이 대화창으로 안내된다고 알리세요.",
    "같은 요청을 다시 호출하지 마세요.",
)
# 승인 대기 턴에 서비스가 직접 붙이는 고객 안내 문장. 고객이 꼭 들어야 하는 것(주문, 금액, 승인 번호, 결과를
# 이 대화창으로 알린다는 것)은 모델의 답과 상관없이 이 문장으로 전달한다(docs/service.md).
HELD_NOTICE = (
    "주문 {order_id}의 {kind} 요청(환불 예정 {amount})은 담당자 승인이 필요해 아직 처리되지 않았습니다. "
    "승인 번호는 {code}이며, 담당자가 확인한 뒤 결과를 이 대화창으로 안내해 드립니다."
)
# 지시를 옮겼다고 보는 기준: 공백·문장부호를 뺀 글자·숫자가 지시와 10자 이상 이어서 같으면 그 부분을 뺀다.
# 문장을 나눌 때는 절 하나가 지시와 6자 이상 같으면 그 절을 뺀다(2026-10-05에 정한 값, docs/service.md "빼는
# 규칙").
_COPIED_RUN = 10  # 답이 HELD_AGENT_ONLY와 이만큼 이어서 같은 글자·숫자를 가지면 그 부분을 뺀다
_CLAUSE_RUN = 6  # ... 잘린 문장에서 남기는 절은 지시와 이만큼 이어서 같으면 안 된다
# 지시 문장에서 글자·숫자만 남긴 비교용 열쇠. 띄어쓰기나 문장부호를 바꿔 옮겨도 잡히게 한다.
_AGENT_KEYS = tuple("".join(ch for ch in s if ch.isalnum()) for s in HELD_AGENT_ONLY)
# 답을 줄 → 문장 → 쉼표 뒤 조각 순으로 나누는 구분 패턴. 문장 끝은 마침표·느낌표·물음표 뒤의 공백이다.
_LINE_END = re.compile(r"\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_AFTER_COMMA = re.compile(r"(?<=,)\s+")
# 쉼표 앞 조각이 "…고, …며, …지만, …는데"('그리고', '하면서', '하지만'으로 잇는 어미)로 끝나면 절의 경계로
# 믿는다(뒤에 괄호 하나가 붙어도 된다). "…다고/라고" 같은 인용(지시 자체의 "안내된다고"가 이 꼴이다), "…면"
# 조건, "…서" 이유는 떼어 내면 남은 절이 원래 문장보다 많은 것을 말하게 되므로 경계로 보지 않고 문장째 뺀다.
_JOINED = re.compile(r"(?:(?<![다라자냐])고|며|지만|데)(?:\s*\([^()]*\))?$")

# 진행 이벤트를 내보내는 콜백의 타입: emit(이벤트 이름, 데이터). 웹에서는 SSE 큐에 넣고, 승인 실행·MCP에서는
# 아무것도 하지 않는 함수를 넘긴다.
Emit = Callable[[str, dict[str, Any]], None]


# 에이전트에게 주는 금액 표기. 같은 주문에 이미 접수된 반품이 있어 합계가 다르면 합계를 괄호로 덧붙인다.
def _amount(refund: int, order_total: int) -> str:
    return f"{refund:,}원" + (
        f"(같은 주문의 반품 환불액 합계 {order_total:,}원)" if order_total != refund else ""
    )


# 승인 대기에 걸린 쓰기에 대해 에이전트가 받는 도구 오류 문구. 관문을 만든 뒤로 바꾸지 않았다. 바꾸면 서비스의
# 모델 입력과 완료 주장 가드(claims.py)가 근거로 읽는 형식이 평가와 달라지기 때문이다(테스트가 문구를
# 고정한다, docs/service.md).
def held_for_agent(refund: int, order_total: int, code: str) -> str:
    """쓰기가 대기에 걸렸을 때 에이전트가 읽는 글. 관문을 만든 뒤로 바꾸지 않았다."""
    return (
        f"환불액 {_amount(refund, order_total)}은 담당자 승인이 필요해 아직 처리되지 않았고, 승인 대기열에 "
        f"올렸습니다(승인 번호 {code}). " + " ".join(HELD_AGENT_ONLY)
    )


# 승인 대기 턴의 모델 답에서 HELD_AGENT_ONLY를 옮긴 부분만 뺀다. 같은 턴에 모델이 조회해 알려 준 내용이나
# 고객에게 묻는 말은 모델만 말할 수 있으므로 남긴다(처음 고친 972317b가 답 전체를 버려 조회 결과까지 사라졌던
# 회귀를 같은 날 이렇게 고침).
# 순서: 지시와 겹치는 글자 위치를 먼저 모두 구하고(_copied), 줄마다 겹침이 없으면 그대로 두고, 겹치면 문장
# 단위로 _sentence_without에 맡긴다. 빈 줄이 셋 이상 이어지면 둘로 줄인다.
# 한계: 지시를 다른 말로 바꿔 옮기면(10자 이상 같은 글자가 없으면) 잡지 못한다(docs/service.md "알려진 틈").
def without_agent_only(text: str) -> str:
    """`text`에서 HELD_AGENT_ONLY를 고객에게 옮기는 부분을 뺀 글.

    옮긴 것으로 보는 것은 `text`가 지시와 공유하는, _COPIED_RUN개 글자·숫자가 이어진 조각(공백과 문장부호는
    빼고 보며, 문장 끝이나 줄 끝을 넘어도 된다)과 낱말 호출(도구에 대해서만 쓰고 고객의 요청에는 쓰지 않는
    말)이다. 그런 것이 없는 줄은 그대로 둔다. 그런 것이 있는 문장은 뺀다. 단, 이음 어미(_JOINED) 뒤의 쉼표가
    문장을 절로 나누면, 지시와 _CLAUSE_RUN개 이어진 조각을 공유하는 절만 빼고 나머지는 남긴다. 그런 쉼표로
    끝났다가 이제 문장 끝이 된 절은 마침표로 끝낸다. 지시를 다른 말로 바꿔 옮긴 것은 잡지 못한다."""
    copied = _copied(text)
    lines = []
    for at, line in _pieces(text, 0, _LINE_END):
        if not copied.intersection(range(at, at + len(line))):
            lines.append(line)
            continue
        said = [_sentence_without(s_at, s, copied) for s_at, s in _pieces(line, at, _SENTENCE_END)]
        if any(said):
            lines.append(" ".join(s for s in said if s))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


# 구분 패턴으로 글을 자르되, 각 조각이 전체 답의 몇 번째 글자에서 시작하는지도 함께 돌려준다. 겹침 위치(전체
# 답 기준)와 조각을 맞대어 보기 위해서다.
def _pieces(text: str, start: int, sep: re.Pattern[str]) -> list[tuple[int, str]]:
    """전체 답의 `start` 위치에서 시작하는 `text`를 `sep`에서 자른 것: (조각의 시작 위치, 조각)."""
    out, at = [], 0
    for match in sep.finditer(text):
        out.append((start + at, text[at : match.start()]))
        at = match.end()
    out.append((start + at, text[at:]))
    return out


# 공백·문장부호를 뺀 압축 글에서 지시의 10자 조각(모든 시작 위치)과 "호출"을 찾고, 찾은 글자를 원래 답의
# 위치로 되돌려 모은다. 압축 글에서 찾으므로 문장이나 줄을 넘어 이어진 복사도 잡는다.
# "호출"은 도구에 대해서만 쓰는 말이라(고객의 요청을 "호출"이라 하지 않음) 그 낱말만으로도 뺀다.
def _copied(text: str) -> set[int]:
    """`text`가 HELD_AGENT_ONLY에서 옮긴 글자·숫자의 `text` 안 위치들."""
    where = [i for i, ch in enumerate(text) if ch.isalnum()]
    compact = "".join(text[i] for i in where)
    runs = {key[i : i + _COPIED_RUN] for key in _AGENT_KEYS for i in range(len(key) - _COPIED_RUN + 1)}
    out: set[int] = set()
    for run in [*runs, "호출"]:
        at = compact.find(run)
        while at != -1:
            out.update(where[at : at + len(run)])
            at = compact.find(run, at + 1)
    return out


# 글(절 하나)이 지시와 n자 이상 이어서 같은 부분이나 "호출"을 가지는지. _sentence_without이 절마다 판정할 때
# 쓴다.
def _shares_a_run(text: str, n: int) -> bool:
    compact = "".join(ch for ch in text if ch.isalnum())
    return "호출" in compact or any(
        key[i : i + n] in compact for key in _AGENT_KEYS for i in range(len(key) - n + 1)
    )


# 겹침이 있는 문장 하나를 처리한다: 이음 어미 + 쉼표로 나뉜 절이 둘 이상이면 지시와 6자 이상 같은 절만 빼고
# 나머지를 남기고, 절로 나뉘지 않거나 남는 절이 없으면 ""(문장째 뺌)를 돌려준다.
def _sentence_without(at: int, sentence: str, copied: set[int]) -> str:
    """답의 `at` 위치에 있는 `sentence`에서 옮긴 글이 든 절을 뺀 것, 또는 ""(without_agent_only 참고)."""
    if not copied.intersection(range(at, at + len(sentence))):
        return sentence
    # 쉼표 뒤마다 자른 조각을 이어 붙여 절을 만든다. 조각이 _JOINED로 끝나는 곳(또는 문장 끝)에서만 절을 끊고,
    # 그 밖의 쉼표는 같은 절 안의 쉼표로 둔다.
    pieces = _pieces(sentence, 0, _AFTER_COMMA)  # 마지막 조각 말고는 모두 쉼표로 끝난다
    clauses, start = [], 0
    for i, (piece_at, piece) in enumerate(pieces):
        last = i == len(pieces) - 1
        if last or _JOINED.search(piece[:-1]):
            clauses.append((start, piece_at + len(piece)))
            if not last:
                start = pieces[i + 1][0]
    # 절 자신의 글로 판정한다: 잇는 쉼표를 넘어 찾은 복사 조각으로 판정하면 그 앞 절까지 빼게
    # 된다("접수되었으며, 결과는 이 대화창으로 …"). 절 안에 있는 복사 조각은 _CLAUSE_RUN보다 길므로 여기서도
    # 잡힌다.
    kept = [sentence[s:e] for s, e in clauses if not _shares_a_run(sentence[s:e], _CLAUSE_RUN)]
    if len(clauses) < 2 or not kept:
        return ""
    said = " ".join(kept)
    # 문장을 끝맺던 절이 빠져 남은 글이 쉼표로 끝나면, 쉼표를 원래 문장의 끝 부호(마침표 등)로 바꾼다. 남은
    # 절은 "…고."처럼 이음 어미로 끝날 수 있다(docs/service.md).
    if said.endswith(","):  # 문장을 끝맺던 절이 빠졌다
        said = said[:-1] + ("." if sentence.rstrip()[-1:] in (".", "!", "?") else "")
    return said


# 기록용 실제 시각(UTC). 세션·감사 로그·승인 기록에 쓴다. 도구·규정은 이것이 아니라 설정의 고정
# 시계(Settings.clock)를 본다.
def wall_clock() -> datetime:
    """무언가가 기록된 시각. `Settings.clock()`은 쇼핑몰의 시계로, 시연용으로 고정될 수 있다."""
    return datetime.now(UTC)


# 없는 세션 번호. app.py가 404로 바꾼다.
class SessionNotFoundError(LookupError):
    pass


# 닫힌 세션(handoff·closed)에 메시지가 옴. 예외 값이 세션 상태다.
class SessionClosedError(RuntimeError):
    pass


# 같은 세션의 잠금을 얻지 못함. 웹은 "앞선 메시지에 답하는 중" 안내나 409로 바꾼다.
class BusyError(RuntimeError):
    """세션이 이미 메시지에 답하는 중이다(또는 그 세션의 승인을 실행하는 중이다)."""


# 없는 승인이거나 이미 결정된 승인. app.py가 409로 바꾼다.
class ApprovalStateError(RuntimeError):
    """승인이 없거나 이미 결정되었다."""


# DB에 저장한 JSON에서 에이전트 상태를 되살린다. agent_calls는 되살리지 않아 0부터 다시 세므로, 루프의 LLM
# 호출 한도(max_agent_calls)가 세션 전체가 아니라 턴당 한도가 된다. 기록용 tool_log·llm_log도 턴마다 새로
# 시작한다.
def _state_from(data: dict[str, Any]) -> AgentState:
    return AgentState(
        messages=[Message.from_dict(m) for m in data["messages"]],
        tool_errors=data.get("tool_errors", 0),
        format_errors=data.get("format_errors", 0),
        dropped_calls=data.get("dropped_calls", 0),
        stalls=data.get("stalls", 0),
        held_claims=data.get("held_claims", 0),
        repeats=data.get("repeats", 0),
    )


# 고객이 실제로 본 대화만 고른다: 하니스 안내문(harness=True인 user 메시지)을 뺀 고객 말과, 전달된(delivered)
# 상담원 답 가운데 도구 호출이 아닌 것. 형식 오류나 가드로 돌려보낸 답은 기록에는 남지만 여기서 빠진다.
# 고객 대화 조회와 관리 화면의 대화 표시가 이것을 쓰고, app.py의 /speech는 이 목록에 있는 답만 소리로 만든다.
def visible_messages(state: dict[str, Any]) -> list[dict[str, str]]:
    """고객이 본 것: 고객 자신의 말과 전달된 답."""
    out = []
    for m in state["messages"]:
        if m["role"] == "user" and not m.get("harness"):
            out.append({"role": "customer", "text": m["content"]})
        elif (
            m["role"] == "assistant" and m["content"] and m.get("delivered", True) and not m.get("tool_calls")
        ):
            out.append({"role": "agent", "text": m["content"]})
    return out


# 세션 하나의 잠금과, 그 잠금을 지금 쥐고 있거나 기다리는(또는 바로 얻어 보려는) 요청의 수. users는
# ChatService._hold가 _locks_guard 안에서만 고치고, 0이 되면 그 항목을 세션별 잠금 dict에서 지운다.
class _SessionLock:
    __slots__ = ("lock", "users")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.users = 0


# 서비스 하나를 이루는 상태: 설정, DB 엔진, 모델 제공자, 도구 등록부, 규정 문서, 실행 설정(RunConfig), 잠금들.
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
        # 평가 실행기와 같은 RunConfig를 서비스 설정으로 채운다. 평가의 기본값과 다른 것은 언어 가드(L1)다.
        # 한자·가나가 섞인 답을 고객에게 보내지 않고 한국어로 다시 답하게 한다(qwen2.5가 중국어로 넘어가는
        # 일이 있어서, docs/service.md).
        self.config = RunConfig(
            model=settings.model,
            policy=settings.policy,
            num_ctx=settings.num_ctx,
            max_agent_calls=settings.max_agent_calls_per_turn,
            language="L1",  # 중국어로 넘어간 답은 절대 고객에게 보이지 않는다
            claims=settings.claims,
            # 13단계 후보(멈춤 G1, 오류 사실 E1, 되풀이 D1)는 설정으로만 있고 기본값은 모두 끔이다. 켤지는
            # 13단계 규칙의 서비스 기본값 조건으로 정한다(docs/experiments.md 13단계).
            guard=settings.guard,
            errors=settings.errors,
            repeat=settings.repeat,
        )
        # 잠금 셋: 세션별 잠금(한 세션은 한 번에 메시지 하나), 승인 결정 잠금(승인된 쓰기는 세션과 상관없이 한
        # 번에 하나), 그리고 세션별 잠금 dict와 그 항목의 참조 수(users)를 보호하는 잠금. 세션별 잠금은 그
        # 세션의 요청이 쥐거나 기다리는 동안만 dict에 있고 마지막 요청이 끝나면 지운다(_hold). 프로세스 안의
        # 잠금이라 앱은 프로세스 하나로 띄운다(여러 프로세스로 늘리려면 DB 행 잠금이 필요하다). 프로세스
        # 하나가 채팅을 맡으므로 세션별 잠금으로 충분하다(docs/service.md "한 세션은 한 번에 메시지 하나").
        self._locks: dict[str, _SessionLock] = {}
        self._decisions = threading.Lock()
        self._locks_guard = threading.Lock()

    # ---------------------------------------------------------------- 세션 (sessions)

    # 새 세션을 만들고 고객용 대화 형태로 돌려준다. 시스템 프롬프트는 규정 문서와 쇼핑몰의 지금(고정 시계)으로
    # 만든다.
    # kind가 chat이 아니면(MCP 서버의 "mcp") 대화 메시지를 비우고 id 앞에 "mcp-"를 붙인다. MCP 클라이언트가
    # 도구를 직접 부르므로 모델과의 대화가 없다.
    # 세션 id는 secrets.token_urlsafe(24)라 추측할 수 없다. 고객 쪽에 로그인이 없으므로 이 값이 곧 열쇠다.
    # 감사 로그 첫 줄에 이 세션을 어떤 모델·think·규정 모드로 처리했는지 남긴다.
    def new_session(self, kind: str = "chat") -> dict[str, Any]:
        """`kind`가 "mcp"면 대화 없는 세션이다: MCP 클라이언트가 도구를 직접 부른다."""
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
            {
                "kind": kind,
                "model": self.settings.model,
                "think": self.settings.think,
                "policy": self.settings.policy,
            },
        )
        return self.transcript(session_id)

    # 고객 화면용 세션 상태: 상태, 고객이 본 대화, 대기 중인 승인(번호·도구·환불액).
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

    # ---------------------------------------------------------------- 한 턴 (one turn)

    # 고객 메시지 하나에 답한다(웹의 글·음성 메시지가 모두 여기로 온다). 빈 메시지나 한도를 넘는 메시지는
    # ValueError다.
    # 세션 잠금은 기다리지 않고 바로 시도한다(_hold의 wait=0). 같은 세션에서 메시지가 겹치면 뒤의 것은
    # BusyError로 돌려보낸다.
    def handle(self, session_id: str, text: str, emit: Emit, *, via: dict[str, Any] | None = None) -> None:
        """고객 메시지 하나에 답한다. 진행 상황은 `emit(event, data)`로 보내고 마지막 이벤트는 `end`다.
        `via`는 메시지가 입력이 아닌 방식으로 들어왔을 때 그 방식을 알린다(감사 로그에 남는다)."""
        text = text.strip()
        if not text or len(text) > self.settings.max_message_chars:
            raise ValueError("the message is empty or too long")
        with self._hold(session_id):  # 기다리지 않는다: 겹친 메시지는 바로 BusyError
            self._handle_locked(session_id, text, emit, via)

    # 잠금을 쥔 채 한 턴을 처리한다.
    # 1) 세션을 읽어 열려 있는지 확인하고 에이전트 상태를 되살린다 2) 고객 메시지를 감사 로그에 남긴다 3)
    #    평가와 같은 agent_turn을 돌리되 도구 실행은 self._run_tool(승인 관문 + 감사 행)을 거치게 한다 4) 종료
    #    사유에 따라 고객에게 보낼 답을 정한다(이번 턴에 승인 대기에 걸린 쓰기가 있으면 어느 사유로 끝나든 그
    #    안내를 붙인다) 5) LLM 호출·오류·답을 감사 로그에 남기고 대화 상태를 저장한다 6) 마지막에 reply·end
    #    이벤트를 보낸다.
    def _handle_locked(
        self, session_id: str, text: str, emit: Emit, via: dict[str, Any] | None = None
    ) -> None:
        with Session(self.engine) as db_session:
            row = self._row(db_session, session_id)
            if row.status != SessionStatus.OPEN.value:
                raise SessionClosedError(row.status)
            state = _state_from(row.state)
            customer_id, turns = row.verified_customer_id, row.turns
        # max_tool_errors는 max_agent_calls처럼 턴당 한도다: 앞선 턴의 오류가 이번 턴을 끝내면 안 된다. 저장된
        # 도구 오류 수는 세션 누계라서, 이번 턴은 0부터 세고 끝에서 다시 더해 저장한다.
        earlier_errors, state.tool_errors = state.tool_errors, 0

        self._audit(session_id, "customer_message", {"text": text, **({"via": via} if via else {})})
        # 도구 문맥: 쇼핑몰의 지금(고정 시계), 규정을 막을지(P1), 이 세션에서 본인 확인된 고객. 본인 확인은
        # 턴을 넘어 세션에 저장된다.
        ctx = self._context(customer_id)

        # 이번 턴에 승인 대기에 걸린 쓰기의 {승인 번호: 고객 안내 문장}. _run_tool이 채운다.
        held: dict[str, str] = {}  # 승인 번호 -> 고객 안내 문장(이번 턴에 대기에 걸린 쓰기마다)

        def run_tool(name: str, arguments: dict) -> ToolResult:
            return self._run_tool(session_id, ctx, name, arguments, emit, notices=held)

        status, reply, error, closing = SessionStatus.OPEN, None, "", ""
        not_delivered = ""
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
            # 종료 사유별 처리: handoff는 세션을 상담원 연결로 닫고, context_limit는 세션을 닫으며 닫는 안내를
            # 보내고, 그 밖의 사유(호출·재시도·도구 오류 한도)는 오류로 기록하고 고정 안내를 보낸다. 사유 없이
            # 끝났는데 승인 대기가 있으면 모델 답에서 지시를 빼고 서비스 안내를 붙인다. 사유가 있는데 승인
            # 대기가 있으면 아래(try 뒤)에서 그 사유의 고정 문장 뒤에 서비스 안내를 붙인다.
            reply = turn.reply
            if turn.stop == "handoff":
                status = SessionStatus.HANDOFF
            elif turn.stop == "context_limit":
                status, reply = SessionStatus.CLOSED, CLOSED_REPLY
            elif turn.stop:  # 모델이 이번 턴의 호출·재시도·도구 오류 한도를 다 썼다
                error, reply = turn.stop, FALLBACK_REPLY
            elif held:
                # 모델은 방금 대기에 걸린 쓰기의 에이전트용 지시를 읽었다. 고객은 지시를 뺀 모델의 답(조회한
                # 것, 물은 것, 그 밖에 한 일)과 그 뒤의 서비스 안내를 읽는다. 대화에도 정확히 그것을 남겨
                # 모델의 다음 턴이 고객이 읽은 것을 읽게 하고, 모델이 원래 한 말은 감사 로그에 남긴다.
                not_delivered = reply
                said = without_agent_only(reply)
                reply = "\n".join([*([said] if said else []), *held.values()])
                state.messages[-1] = Message("assistant", reply)
        except Exception as exc:  # noqa: BLE001
            # 모델 연결 오류(ProviderError)는 메시지만, 그 밖의 예외는 스택까지 감사 로그에 남긴다. 실패 전에
            # 도구가 쓴 것은 이미 커밋·감사됐으므로, 대화를 그것과 맞게 두고 고객에게는 이번 메시지에 답하지
            # 못했다고 알린다.
            error = f"{type(exc).__name__}: {exc}"
            if not isinstance(exc, ProviderError):
                error += "\n" + traceback.format_exc()
            reply = FALLBACK_REPLY
            # 마지막 메시지가 답 없는 도구 호출이면 빼서, 다음 턴의 모델이 짝이 맞지 않는 호출을 읽지 않게
            # 한다.
            if state.messages[-1].tool_calls:  # 실패한 호출에는 답이 없다: 반쪽만 남기지 않는다
                state.messages.pop()
        # 정상 답이 아닌 사유(상담원 연결, 컨텍스트 한도, 호출·재시도·도구 오류 한도, 예외)로 끝난 턴에도 이번
        # 턴에 승인 대기에 걸린 쓰기가 있으면, 그 사유의 고정 문장 뒤에 서비스 안내(held)를 붙인다. 고객이 꼭
        # 들어야 하는 것(주문, 금액, 승인 번호, 결과를 이 대화창으로 알린다는 것)은 턴이 어떻게 끝나든 정해진
        # 문장으로 간다(docs/service.md "사람 승인"). 승인 행은 대기열에 올릴 때 이미 커밋됐으므로 예외로 끝난
        # 턴에서도 안내가 사실이다. 한도·예외의 고정 문장은 FALLBACK_REPLY 대신 HELD_FALLBACK_REPLY다.
        # 저장된 대화가 고객이 본 글과 같아야 하므로, 상담원 연결은 agent_turn이 대화 끝에 이미 넣은 이관
        # 안내(HANDOFF_MESSAGE)를 합친 글로 바꾸고, 한도·예외·닫힘은 아래 append가 합친 글을 넣는다. error
        # 감사 행과 error 이벤트는 그대로 둔다(예전에는 이 경우들에 안내가 빠지고 한도·예외에는 '처리하지
        # 못했습니다'만 갔다).
        if held and (error or status is not SessionStatus.OPEN):
            reply = "\n".join([HELD_FALLBACK_REPLY if error else reply, *held.values()])
            if status is SessionStatus.HANDOFF:
                state.messages[-1] = Message("assistant", reply)
        state.tool_errors += earlier_errors  # 저장하는 상태에는 세션 누계를 둔다
        # 고정 안내(오류·닫힘)는 agent_turn이 대화에 넣지 않았으므로 여기서 상담원 답으로 넣는다. 저장된
        # 대화가 고객이 본 것과 같아야 하기 때문이다.
        if reply is not None and (error or status is SessionStatus.CLOSED):
            state.messages.append(Message("assistant", reply))
        # 이 턴으로 세션당 턴 한도에 닿으면 세션을 닫고, 닫는 안내를 별도 메시지로 덧붙인다.
        if turns + 1 >= self.settings.max_turns_per_session and status is SessionStatus.OPEN:
            status = SessionStatus.CLOSED
            closing = CLOSED_REPLY  # 저장된 대화처럼 별도 메시지로 보낸다
            state.messages.append(Message("assistant", CLOSED_REPLY))

        # 이번 턴의 LLM 호출마다 감사 행 하나: 형식 오류, 종료 이유, 부른 도구 이름, 토큰 수, 걸린 시간(ms).
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
        # 고객에게 보낸 답. 승인 대기 턴이면 모델이 쓴 원래 답은 not_delivered에만 남는다.
        self._audit(
            session_id,
            "agent_reply",
            {
                "text": reply or "",
                "status": status.value,
                **({"not_delivered": not_delivered} if not_delivered else {}),
            },
        )
        # 대화 상태, 턴 수, 세션 상태, 본인 확인된 고객을 턴이 끝날 때 한 번에 저장한다. 도구의 변경과 감사
        # 행은 도구마다 이미 커밋됐으므로, 턴 도중 프로세스가 죽으면 대화에는 그 호출이 없고 감사 로그에는
        # 있다(docs/service.md).
        with Session(self.engine) as db_session:
            row = self._row(db_session, session_id)
            row.state = state.to_dict()
            row.turns = turns + 1
            row.status = status.value
            row.verified_customer_id = ctx.state.verified_customer_id
            row.updated_at = wall_clock()
            db_session.commit()
        # 이벤트는 저장이 끝난 뒤에 보낸다. 화면이 end를 받고 대화를 다시 읽어도 저장된 것과 같다.
        if error:
            emit("error", {"message": "처리 중 문제가 생겼습니다."})
        for reply_text in (reply, closing):
            if reply_text:
                emit("reply", {"text": reply_text})
        emit("end", {"status": status.value})

    # 서비스에서 도구가 실행되는 유일한 길(에이전트 루프, 승인 실행, MCP가 모두 여기로 온다).
    def _run_tool(
        self,
        session_id: str,
        ctx: ToolContext,
        name: str,
        arguments: dict[str, Any],
        emit: Emit,
        *,
        approval_id: int | None = None,
        notices: dict[str, str] | None = None,
    ) -> ToolResult:
        """서비스에서 도구가 실행되는 유일한 길.

        도구 자신의 트랜잭션 안에서(toolkit의 on_success 훅): 핸들러가 실제로 계산한 환불액(반품이면 그
        주문에 이미 접수된 반품의 환불액을 더한 것)을 승인 기준과 비교하고, 감사 행을 쓰고, 실행 중인 승인을
        끝남으로 표시한다. 그래서 쇼핑몰 데이터의 변경은 감사 행 없이 존재하지 않고, 기준 이상의 금액은
        사람 없이 커밋되지 않는다. `approval_id`는 그 사람이 결정했을 때 주어진다.
        """
        # 화면에 "무엇을 하는 중"을 바로 알린다. seen은 이 호출 전까지 쌓인 규정 위반 기록 수다(이 호출에서
        # 생긴 것만 감사 행에 넣으려고).
        emit("tool", {"name": name, "label": TOOL_LABELS.get(name, name)})
        held: dict[str, Any] = {}
        seen = len(ctx.violations)

        # 감사 행 tool_call의 내용: 도구 이름, 모델이 보낸 원래 인자와 검증된 인자, 쓰기 여부, 성공 여부, 오류
        # 코드, 규정으로 막혔는지, 이번 호출이 통과시킨 규정 위반(P0일 때), 결과 글, 본인 확인된 고객, 승인
        # 실행이면 승인 번호.
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

        # toolkit.execute의 on_success 훅. 핸들러가 성공한 직후, 같은 DB 트랜잭션 안에서 돈다.
        # 여기서 예외를 내면 도구의 변경이 통째로 롤백된다. 그래서 승인 관문은 "계산된 실제 환불액이 기준
        # 이상이면 ToolError로 롤백"으로, 감사 행은 "변경과 같은 트랜잭션에 추가"로 구현한다.
        def in_transaction(db_session: Session, spec: ToolSpec, clean: dict[str, Any], answer: dict) -> None:
            refund = answer.get("refund_won")
            earlier = 0
            # 반품은 같은 주문에 이미 접수된 반품의 환불액을 더해 기준과 비교한다. 줄을 하나씩 나눠 접수해
            # 기준 아래로 빠져나가는 것을 막는다. 방금 이 호출이 추가한 행은 합계에서 뺀다. 승인
            # 실행(approval_id 있음)은 이미 사람이 본 것이므로 다시 재지 않는다.
            if approval_id is None and spec.name == "request_return" and isinstance(refund, int):
                earlier = db_session.scalar(
                    select(func.coalesce(func.sum(db.ServiceRequest.refund_won), 0)).where(
                        db.ServiceRequest.order_id == answer["order_id"],
                        db.ServiceRequest.kind == db.RequestKind.RETURN,
                        db.ServiceRequest.id != answer["request_id"],  # 이 호출이 방금 추가한 행
                    )
                )
            # 기준 이상이면 실행 정보를 held에 담고 approval_required로 롤백한다. 사람이 결정하기 전에는 기준
            # 이상의 금액이 커밋되지 않는다.
            if (
                approval_id is None
                and spec.name in APPROVAL_TOOLS
                and isinstance(refund, int)
                and refund + earlier >= self.settings.approval_refund_won
            ):
                held.update(refund=refund, args=clean, order_total=refund + earlier)
                raise ToolError("approval_required", "held for approval")  # 호출을 롤백한다
            # 성공한 호출의 감사 행을 도구의 변경과 같은 트랜잭션에 넣는다. 감사 행을 쓰지 못하면 주문도
            # 바뀌지 않는다(tests/test_postgres.py가 확인한다).
            content = json.dumps(answer, ensure_ascii=False)
            db_session.add(
                AuditEvent(
                    session_id=session_id,
                    at=wall_clock(),
                    kind="tool_call",
                    payload=payload(clean, True, content),
                )
            )
            # 승인 실행이면 승인 행의 approved 표시와 결과도 같은 트랜잭션에서 남긴다.
            if approval_id is not None:
                approval = db_session.get(Approval, approval_id)
                approval.status = ApprovalStatus.APPROVED.value
                approval.result = content

        result = execute(self.registry, self.engine, ctx, name, arguments, on_success=in_transaction)
        # 롤백된 쓰기를 승인 대기열에 올리고, 에이전트에게는 approval_required 오류를, 고객 안내는 notices로
        # 돌려준다.
        if held:
            result, notice = self._queue_approval(
                session_id, ctx, name, held["args"], held["refund"], emit, order_total=held["order_total"]
            )
            if notices is not None:
                notices.update(notice)
        # 실패·거절된 호출은 바뀐 것이 없으므로 감사 행을 따로(별도 트랜잭션으로) 적는다.
        if not result.ok:  # 바뀐 것이 없으므로 이 행은 따로 있어도 된다
            self._audit(
                session_id,
                "tool_call",
                payload(result.args, False, result.content, result.error_code, result.policy_blocked),
            )
        emit("tool_result", {"name": name, "ok": result.ok, "error_code": result.error_code})
        return result

    # MCP 서버에서 온 도구 호출. 에이전트 루프 없이 _run_tool만 거치므로 규정 검사, 승인 관문, 감사 로그가
    # 채팅과 같다(docs/mcp.md).
    # 채팅과 달리 잠금을 30초까지 기다린다. 본인 확인 결과는 세션에 저장되고, 호출 하나를 턴 하나로 센다.
    def call_tool(self, session_id: str, name: str, arguments: dict[str, Any]) -> ToolResult:
        """에이전트 루프 밖(MCP 서버)에서 온 도구 호출. 관문도 감사도 같다."""
        with self._hold(session_id, wait=30):
            with Session(self.engine) as db_session:
                customer_id = self._row(db_session, session_id).verified_customer_id
            ctx = self._context(customer_id)
            result = self._run_tool(session_id, ctx, name, arguments, lambda _event, _data: None)
            with Session(self.engine) as db_session:
                row = self._row(db_session, session_id)
                row.verified_customer_id = ctx.state.verified_customer_id
                row.turns += 1
                row.updated_at = wall_clock()
                db_session.commit()
            return result

    # ---------------------------------------------------------------- 승인 (approvals)

    # 롤백된 쓰기를 승인 대기열에 올린다. 같은 세션에서 같은 대상(도구 + 주문, 반품이면 + 줄 번호)의 대기 건이
    # 이미 있으면 새로 만들지 않고 그것을 쓴다(모델이 같은 요청을 다시 불러도 대기열에는 하나만). 대상은
    # _approval_target으로 비교한다: 예전에는 인자 전체를 비교해, 사유(reason)만 바꾼 재요청이 같은 주문의
    # 승인을 하나 더 만들었다. 두 건이 모두 승인되면 둘째는 주문 상태가 바뀌어 failed가 될 뿐이지만, 담당자가
    # 같은 요청을 두 번 보게 된다. 대기 건의 인자(처음 요청의 사유)는 그대로 둔다.
    # 돌려주는 것: 에이전트가 읽을 approval_required 결과(held_for_agent 문구)와 {승인 번호: 고객 안내 문장}.
    # approval 이벤트로 화면에도 알린다.
    def _queue_approval(
        self,
        session_id: str,
        ctx: ToolContext,
        tool: str,
        clean: dict[str, Any],
        refund: int,
        emit: Emit,
        *,
        order_total: int,
    ) -> tuple[ToolResult, dict[str, str]]:
        """방금 롤백된 쓰기를 대기열에 올린다(세션과 인자마다 한 번). `order_total`은 그 주문에 이미 접수된
        반품을 더한 값이고, 승인하면 `refund`만 커밋된다. 에이전트에게 줄 답과 {승인 번호: 고객 안내}를
        돌려준다."""
        with Session(self.engine) as db_session:
            pending = db_session.scalars(
                select(Approval).where(
                    Approval.session_id == session_id,
                    Approval.tool == tool,
                    Approval.status == ApprovalStatus.PENDING.value,
                )
            ).all()
            target = _approval_target(tool, clean)
            approval = next((a for a in pending if _approval_target(a.tool, a.args) == target), None)
            reused = approval is not None
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
            {
                "code": code,
                "tool": tool,
                "args": clean,
                "refund_won": refund,
                "order_refund_won": order_total,
                # 같은 대상의 대기 건을 다시 쓴 요청이면 표시한다(새 승인을 만들지 않았다).
                **({"reused": True} if reused else {}),
            },
        )
        emit("approval", {"code": code, "tool": tool, "refund_won": refund})
        amount = f"{refund:,}원" + (
            f", 같은 주문의 반품 환불액 합계 {order_total:,}원" if order_total != refund else ""
        )
        notice = HELD_NOTICE.format(
            order_id=clean.get("order_id", ""), kind=APPROVAL_TOOLS[tool], amount=amount, code=code
        )
        result = ToolResult.error("approval_required", held_for_agent(refund, order_total, code), args=clean)
        return result, {code: notice}

    # 관리자의 승인·거절. 그 세션의 잠금을 먼저 잡는다(30초까지 기다림. 진행 중인 턴이 대화 상태를 덮어쓰지
    # 않게). 이어 결정 잠금으로 승인된 쓰기를 프로세스 안에서 한 번에 하나만 실행한다.
    def decide(self, approval_id: int, *, approve: bool, by: str, note: str = "") -> dict[str, Any]:
        """기다리는 쓰기를 승인(하고 실행)하거나 거절한다. 고객은 결과를 채팅에서 읽는다."""
        with Session(self.engine) as db_session:
            approval = db_session.get(Approval, approval_id)
            if approval is None:
                raise ApprovalStateError("no such approval")
            session_id = approval.session_id
        with self._hold(session_id, wait=30):  # 그 세션에서 도는 턴이 대화를 덮어쓰지 않게
            with self._decisions:  # 세션과 상관없이 승인된 쓰기는 한 번에 하나
                return self._decide_locked(approval_id, approve=approve, by=by, note=note.strip())

    # 결정을 처리한다: 선점 → (승인이면) 도구 실행 → 결과 문장을 대화에 추가 → 감사 로그.
    def _decide_locked(self, approval_id: int, *, approve: bool, by: str, note: str) -> dict[str, Any]:
        now = wall_clock()
        claimed = ApprovalStatus.EXECUTING if approve else ApprovalStatus.REJECTED
        with Session(self.engine) as db_session:
            # pending인 행만 바꾸는 조건부 UPDATE 한 번으로 선점한다. 바뀐 행이 1이 아니면 다른 결정자가 먼저
            # 가져간 것이다. DB가 판정하므로 같은 프로세스든 다른 프로세스든 두 결정자 가운데 한쪽만 이긴다.
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
            # 요청 당시 본인 확인된 고객을 문맥에 넣고 같은 인자로 도구를 다시 실행한다. 그 사이 주문 상태가
            # 바뀌었으면 도구가 거절하고 failed가 된다.
            ctx = self._context(customer_id)
            # 도구의 변경, 그 감사 행, 승인의 "approved" 표시는 한 트랜잭션이다.
            try:
                result = self._run_tool(
                    session_id, ctx, tool, args, lambda _e, _d: None, approval_id=approval_id
                )
            except Exception as exc:
                # 실행 중 예외(감사 행 쓰기 실패, DB 오류 등)면 그 트랜잭션은 롤백되어 쇼핑몰이 그대로이므로,
                # 선점을 풀어 승인을 pending으로 되돌려 다시 결정할 수 있게 한다. 커밋이 실제로 됐다면
                # "approved"도 함께 기록됐으므로 아래 조건부 UPDATE는 아무 행에도 맞지 않는다. 되돌리기도
                # 실패하거나 프로세스가 죽으면 executing으로 남는다(docs/service.md).
                with Session(self.engine) as db_session:
                    db_session.execute(
                        update(Approval)
                        .where(Approval.id == approval_id, Approval.status == claimed.value)
                        .values(
                            status=ApprovalStatus.PENDING.value, decided_at=None, decided_by=None, note=""
                        )
                    )
                    db_session.commit()
                error = f"{type(exc).__name__}: {exc}"
                self._audit(session_id, "error", {"approval": approval_code(approval_id), "error": error})
                raise
            status = ApprovalStatus.APPROVED if result.ok else ApprovalStatus.FAILED

        # 결과는 모델이 아니라 서비스가 정해진 문장으로 대화에 넣는다. 고객 화면은 주기적으로 대화를 다시 읽어
        # 이 문장을 보여 준다.
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
            # JSON 열을 제자리에서 고치면 SQLAlchemy가 변경을 알아채지 못하므로, 복사본을 고쳐 통째로 다시
            # 넣는다.
            state = copy.deepcopy(row.state)
            if status is ApprovalStatus.APPROVED and not session_id.startswith(
                "mcp-"
            ):  # MCP 세션은 대화가 없다
                # 승인된 쓰기를 대화에 일반 도구 호출과 결과로 남긴다. 다음 턴의 모델이 무엇이 처리됐는지
                # 읽고, 완료 주장 가드도 "처리되었습니다"의 근거를 찾을 수 있다. MCP 세션은 대화가 없으므로
                # 넣지 않는다.
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

    # ---------------------------------------------------------------- 관리자 조회 (admin reads)

    # --- 관리 화면용 조회 ---
    # 승인 한 건(결정 뒤 관리 화면이 카드를 갱신할 때).
    def approval(self, approval_id: int) -> dict[str, Any]:
        with Session(self.engine) as db_session:
            approval = db_session.get(Approval, approval_id)
            if approval is None:
                raise ApprovalStateError("no such approval")
            return _approval_dict(approval)

    # 승인 목록(최근 것부터 최대 limit건, status로 거름).
    def approvals(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        query = select(Approval).order_by(Approval.id.desc()).limit(limit)
        if status:
            query = query.where(Approval.status == status)
        with Session(self.engine) as db_session:
            return [_approval_dict(a) for a in db_session.scalars(query)]

    # 최근에 바뀐 세션 목록(최대 50건).
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

    # 한 세션의 감사 로그 전체를 기록 순서대로. 없는 세션이면 SessionNotFoundError.
    def audit(self, session_id: str) -> list[dict[str, Any]]:
        with Session(self.engine) as db_session:
            self._row(db_session, session_id)
            events = db_session.scalars(
                select(AuditEvent).where(AuditEvent.session_id == session_id).order_by(AuditEvent.id)
            )
            return [
                {"id": e.id, "at": e.at.isoformat(), "kind": e.kind, "payload": e.payload} for e in events
            ]

    # ---------------------------------------------------------------- 도우미 (helpers)

    # 도구 문맥: 쇼핑몰의 지금(고정 시계), 규정을 막을지(P1), 오류 글에 사실을 붙일지(E1), 이 세션에서 본인
    # 확인된 고객. 채팅 턴, MCP 호출, 승인 실행이 같은 문맥을 쓴다.
    def _context(self, customer_id: str | None) -> ToolContext:
        return ToolContext(
            now=self.settings.clock(),
            enforce_policy=self.config.enforce_policy,
            state=ConversationState(verified_customer_id=customer_id),
            error_facts=self.config.errors == "E1",
        )

    # 세션 행을 읽고, 없으면 SessionNotFoundError.
    def _row(self, db_session: Session, session_id: str) -> ChatSession:
        row = db_session.get(ChatSession, session_id)
        if row is None:
            raise SessionNotFoundError(session_id)
        return row

    # 세션별 잠금을 쥔 채 with 블록을 실행한다. 한 턴(handle), MCP 호출(call_tool), 승인 결정(decide)이 모두
    # 이것을 지난다. `wait`가 0이면 기다리지 않고(채팅: 겹친 메시지는 바로 BusyError), 0보다 크면 그 초만큼
    # 기다린 뒤에도 못 얻으면 BusyError를 낸다. 블록이 예외로 끝나도 잠금을 풀고 참조 수를 돌려준다.
    # 잠금은 그 세션을 처음 쓰는 요청이 만들고, 그 세션의 마지막 요청(잠금을 쥔 것도 기다리는 것도 없음)이
    # 끝나면 dict에서 지운다. 세션 id는 다시 쓰이지 않는 난수라서, 예전처럼 setdefault로 만들기만 하면 닫힌
    # 세션·MCP 상담·없는 세션 번호의 잠금까지 프로세스가 사는 동안 늘기만 했다.
    # 지워도 한 세션의 직렬화가 깨지지 않는 이유: 요청은 _locks_guard 안에서 잠금 항목을 얻으며 users를 하나
    # 늘리고, 잠금을 푼 뒤 _locks_guard 안에서 하나 줄인다. 그래서 users가 0이 되어 지울 때는 그 잠금을 쥐거나
    # 기다리는 요청이 없고, 그 뒤에 온 요청은 새 항목을 만든다. 한 세션에서 쓰이는 잠금 객체는 언제나 하나다.
    # 턴이 끝날 때 승인 결정이 그 잠금을 기다리고 있으면 users가 남아 있으므로 지우지 않고 결정이 이어받는다.
    # 같은 승인을 두 결정자가 함께 실행하는 경합은 이 잠금이 아니라 _decide_locked의 조건부 UPDATE(pending →
    # executing, 다른 프로세스와도)가 막는다. 이 잠금이 지키는 것은 한 세션의 대화 상태를 두 요청이 함께
    # 고치지 않는 것이다.
    @contextmanager
    def _hold(self, session_id: str, *, wait: float = 0) -> Iterator[None]:
        with self._locks_guard:
            entry = self._locks.get(session_id)
            if entry is None:
                entry = self._locks[session_id] = _SessionLock()
            entry.users += 1
        try:
            got = entry.lock.acquire(timeout=wait) if wait > 0 else entry.lock.acquire(blocking=False)
            if not got:
                raise BusyError(session_id)
            try:
                yield
            finally:
                entry.lock.release()
        finally:
            with self._locks_guard:
                entry.users -= 1
                if entry.users == 0:  # 쥔 요청도 기다리는 요청도 없다
                    self._locks.pop(session_id, None)

    # 감사 행 하나를 자기 트랜잭션으로 추가한다(추가만 하고 고치지 않는다). 도구 성공 행은 이것이 아니라
    # in_transaction이 도구와 같은 트랜잭션에 넣는다.
    def _audit(self, session_id: str, kind: str, payload: dict[str, Any]) -> None:
        with Session(self.engine) as db_session:
            db_session.add(AuditEvent(session_id=session_id, at=wall_clock(), kind=kind, payload=payload))
            db_session.commit()


# 승인 대기 건의 대상: (도구, 주문 번호, 반품 줄 번호). 같은 세션에서 대상이 같은 대기 건은 하나만 둔다
# (_queue_approval). 사유처럼 대상을 바꾸지 않는 인자는 보지 않는다. 반품은 줄 번호가 다르면 다른 요청이다(줄
# 번호는 도구가 정렬해 둔 canonical_args 형태라 순서와 무관하다).
def _approval_target(tool: str, args: dict[str, Any]) -> tuple[str, str, tuple[int, ...]]:
    return tool, str(args.get("order_id", "")), tuple(args.get("line_nos") or ())


# 승인 행 → 관리자 API의 JSON. 화면 표기용 승인 번호와 도구의 한국어 이름을 함께 넣는다.
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
