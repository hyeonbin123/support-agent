"""Does a reply say that something was done which no tool result of the conversation shows?

"취소가 완료되었습니다" with no cancelled order in any tool result is the failure this is for: the customer is
told that the work is done and the database says otherwise. A reply that reports what a tool showed ("이
주문은 9월 1일에 이미 취소되었습니다", after get_order) has its evidence and passes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from support_agent.chat import Message
from support_agent.toolkit import ERROR_PREFIX


@dataclass(frozen=True)
class ClaimKind:
    name: str
    label: str  # how the notice names it
    tools: str  # the write tool(s) that would really do it, named in the notice
    said: re.Pattern[str]  # the sentence is about this kind of work
    shown: re.Pattern[str]  # a successful tool result ("<tool name>\n<content>") that shows it


# The patterns of `shown` read the JSON that tools.py writes (json.dumps, default separators).
KINDS: tuple[ClaimKind, ...] = (
    ClaimKind(
        "cancel", "주문 취소", "cancel_order", re.compile(r"취소"), re.compile(r'"status": "cancelled"')
    ),
    ClaimKind(
        "return",
        "반품 접수",
        "request_return",
        re.compile(r"반품"),
        re.compile(r'"request_id": "RT-|"kind": "return"'),
    ),
    ClaimKind(
        "exchange",
        "교환 접수",
        "request_exchange",
        re.compile(r"교환"),
        re.compile(r'"request_id": "EX-|"kind": "exchange"'),
    ),
    # A read shows the address an order ships to, never that it was changed: only the write is evidence.
    ClaimKind(
        "address",
        "배송지 변경",
        "change_shipping_address",
        re.compile(r"배송지|(?<!이메일 )(?<!이메일)주소"),
        re.compile(r"\Achange_shipping_address\n"),
    ),
    ClaimKind(
        "coupon",
        "보상 쿠폰 발급",
        "issue_compensation_coupon",
        re.compile(r"쿠폰"),
        re.compile(r'"coupon_id": "'),
    ),
    ClaimKind(
        "ticket", "상담 티켓 접수", "create_ticket", re.compile(r"티켓"), re.compile(r'"ticket_id": "')
    ),
    ClaimKind(
        "handoff", "상담원 연결", "transfer_to_human", re.compile(r"상담원"), re.compile(r'"handoff_id": "')
    ),
    ClaimKind(
        "refund",
        "환불",
        "cancel_order 또는 request_return",
        re.compile(r"환불"),
        re.compile(r'"refund_won": [1-9]|"status": "(cancelled|refund_pending|refunded)"'),
    ),
)
# In the service a cancel or return above the approval threshold is rolled back and queued, and the tool
# answers with this error. "취소 요청이 접수되었습니다" is then true ("취소되었습니다" is not), so the error
# is evidence for those kinds when the sentence speaks of receiving or registering the request.
_QUEUED = re.compile(
    r"\A(cancel_order|request_return)\n" + re.escape(ERROR_PREFIX) + r": \[approval_required\]"
)
_QUEUED_KINDS = frozenset({"cancel", "return", "refund"})
_RECEIVED = re.compile(r"(접수|신청|등록)\s?(?:[가이를은는도]\s?)?(?:되었|됐|했|하였|해\s?드렸|드렸|마쳤)")

# "…되었습니다 / 했습니다 / 해 드렸습니다": the work is said to be over. Future and conditional forms
# ("취소해 드리겠습니다", "완료되면", "취소했는지") do not match.
_DONE = re.compile(
    r"(?:(?:완료|접수|처리|변경|발급|제출|등록|신청|연결|생성|진행|취소|반품|교환|환불)\s?(?:[가이를은는도]\s?)?"
    r"(?:되었|됐|했|하였|해\s?드렸|드렸|마쳤)|남겼|남겨\s?드렸|이루어졌)(?:습니다|어요|으며|고(?![가-힣]))"
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
# Clauses: "본인 확인이 완료되었고, 취소는 가능합니다" says nothing about a cancellation being done. A claim
# is judged inside its own clause, and the kind must stand before (or be) the verb (Korean word order).
_CLAUSE_END = re.compile(r"(?<=[고며만데]),?\s+")  # a bare comma is not a boundary ("미르로 210, 새길빌딩")
_DELIVERY = re.compile(
    r"배송[이은도]?\s*$"
)  # "배송이 완료되었습니다" is about the parcel, not about our work


@dataclass(frozen=True)
class Claim:
    sentence: str
    kinds: tuple[str, ...]  # every kind the sentence is about; none of them has evidence
    label: str  # of the first kind
    tools: str  # of the first kind


def shown_results(messages: Iterable[Message]) -> list[str]:
    """The successful tool results of the conversation, each as "<tool name>\\n<content>" (plus the
    approval_required errors of the service, which record a queued request)."""
    out = []
    for m in messages:
        if m.role != "tool":
            continue
        entry = f"{m.tool_name or ''}\n{m.content}"
        if not m.content.startswith(f"{ERROR_PREFIX}: [") or _QUEUED.match(entry):
            out.append(entry)
    return out


def unbacked_claim(text: str, messages: Iterable[Message]) -> Claim | None:
    """The first sentence of `text` that says some work is done while no tool result shows it."""
    results: list[str] | None = None
    for sentence in (s.strip() for s in _SENTENCE_END.split(text)):
        if not sentence or sentence.endswith("?"):
            continue
        for clause in _CLAUSE_END.split(sentence):
            kinds = _claimed_kinds(clause)
            if not kinds:
                continue
            if results is None:
                results = shown_results(messages)
            received = bool(_RECEIVED.search(clause))
            if not any(_shows(kind, result, received) for kind in kinds for result in results):
                return Claim(sentence, tuple(kind.name for kind in kinds), kinds[0].label, kinds[0].tools)
    return None


def _claimed_kinds(clause: str) -> list[ClaimKind]:
    """The kinds of work a clause says are done: named before the verb ("반품 접수를 완료했습니다") or by
    the verb itself ("취소되었습니다"). A delivery being complete is not our work."""
    for done in _DONE.finditer(clause):
        if _DELIVERY.search(clause[: done.start()]):
            continue
        head = clause[: done.end()]
        kinds = [kind for kind in KINDS if kind.said.search(head)]
        if kinds:
            return kinds
    return []


def _shows(kind: ClaimKind, result: str, received: bool) -> bool:
    if kind.shown.search(result):
        return True
    return received and kind.name in _QUEUED_KINDS and bool(_QUEUED.match(result))
