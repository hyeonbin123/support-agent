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
    said: re.Pattern[str]  # the sentence is about this kind of work
    shown: re.Pattern[str]  # a successful tool result ("<tool name>\n<content>") that shows it


# The patterns of `shown` read the JSON that tools.py writes (json.dumps, default separators).
KINDS: tuple[ClaimKind, ...] = (
    ClaimKind("cancel", "주문 취소", re.compile(r"취소"), re.compile(r'"status": "cancelled"')),
    ClaimKind("return", "반품 접수", re.compile(r"반품"), re.compile(r'"request_id": "RT-|"kind": "return"')),
    ClaimKind(
        "exchange", "교환 접수", re.compile(r"교환"), re.compile(r'"request_id": "EX-|"kind": "exchange"')
    ),
    # A read shows the address an order ships to, never that it was changed: only the write is evidence.
    ClaimKind(
        "address", "배송지 변경", re.compile(r"배송지|주소"), re.compile(r"\Achange_shipping_address\n")
    ),
    ClaimKind("coupon", "보상 쿠폰 발급", re.compile(r"쿠폰"), re.compile(r'"coupon_id": "')),
    ClaimKind("ticket", "상담 티켓 접수", re.compile(r"티켓"), re.compile(r'"ticket_id": "')),
    ClaimKind("handoff", "상담원 연결", re.compile(r"상담원"), re.compile(r'"handoff_id": "')),
    ClaimKind(
        "refund",
        "환불",
        re.compile(r"환불"),
        re.compile(r'"refund_won": [1-9]|"status": "(cancelled|refund_pending|refunded)"'),
    ),
)

# "…되었습니다 / 했습니다 / 해 드렸습니다": the work is said to be over. Future and conditional forms
# ("취소해 드리겠습니다", "완료되면", "취소했는지") do not match.
_DONE = re.compile(
    r"(?:(?:완료|접수|처리|변경|발급|제출|등록|신청|연결|생성|취소|반품|교환|환불)\s?"
    r"(?:되었|됐|했|하였|해\s?드렸|드렸|마쳤)|남겼|남겨\s?드렸)(?:습니다|어요|으며|고(?![가-힣]))"
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass(frozen=True)
class Claim:
    sentence: str
    kinds: tuple[str, ...]  # every kind the sentence is about; none of them has evidence
    label: str  # of the first kind


def shown_results(messages: Iterable[Message]) -> list[str]:
    """The successful tool results of the conversation, each as "<tool name>\\n<content>"."""
    return [
        f"{m.tool_name or ''}\n{m.content}"
        for m in messages
        if m.role == "tool" and not m.content.startswith(f"{ERROR_PREFIX}: [")
    ]


def unbacked_claim(text: str, messages: Iterable[Message]) -> Claim | None:
    """The first sentence of `text` that says some work is done while no tool result shows it."""
    results: list[str] | None = None
    for sentence in (s.strip() for s in _SENTENCE_END.split(text)):
        if not sentence or sentence.endswith("?") or not _DONE.search(sentence):
            continue
        kinds = [kind for kind in KINDS if kind.said.search(sentence)]
        if not kinds:
            continue
        if results is None:
            results = shown_results(messages)
        if not any(kind.shown.search(result) for kind in kinds for result in results):
            return Claim(sentence, tuple(kind.name for kind in kinds), kinds[0].label)
    return None
