"""답이 이 대화의 어떤 도구 결과도 보여 주지 않는 일을 끝냈다고 말하는가?

이 모듈이 겨냥하는 실패는 도구 결과 어디에도 취소된 주문이 없는데 "취소가 완료되었습니다"라고 말하는
것이다. 고객은 일이 끝났다고 듣지만 데이터베이스는 그렇지 않다. 도구가 보여 준 사실을 전하는 답(get_order
뒤의 "이 주문은 9월 1일에 이미 취소되었습니다")은 근거가 있으므로 통과한다.
"""

# [파일 역할] 7단계 완료 주장 가드의 검출기. 상담원 답에 '취소되었습니다'처럼 일이 끝났다는 말이 있는데, 이
# 대화의 성공한 도구 결과에 그 근거가 없으면 그 문장을 찾아 돌려준다.
# [왜 필요한가] 서비스를 실제 모델로 써 보았을 때 7B가 취소 도구를 부르지 않고 '취소되었습니다. 12,000원이
# 환불됩니다'라고 답했다(DB는 그대로). 평가에서는 DB가 정답과 달라 실패로 잡히지만, 서비스에는 그 말이
# 고객에게 가는 것을 막을 장치가 없었다(docs/experiments.md 7단계 왜 하는가).
# [방식] 정규식으로 (1) 끝났다는 말과 (2) 무엇에 대한 말인지(취소·반품·교환·배송지·쿠폰·티켓·상담원·환불)를
# 찾고, 근거는 그 종류의 흔적이 있는 성공한 도구 결과다. 조회 결과도 근거가 되므로 '이 주문은 이미
# 취소되었습니다'처럼 조회한 사실을 전하는 말은 통과한다. 개발용 기준 기록만 보고 만들었고, 문장을 절로 나누는
# 규칙 등은 코드 검토 뒤 본 측정 전에 고쳤다(같은 절).
# [결과] C1·C2 모두 판정 '개선 없음'. 거짓 완료의 전달은 0이 되지만 7B가 안내문을 받고도 도구를 부르지 않아
# 성공률은 오르지 않았다. 서비스에는 꺼진 설정(SUPPORT_AGENT_CLAIMS=C1)으로만 남았다(README 7단계).
# [연결] agent.py(C1·C2일 때 고객에게 보낼 답마다), analyze.py(analyze claims: 가드 없이 돈 기록에도 같은
# 검출기를 대 본다)가 unbacked_claim을 부른다.

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

# is_confirmation: R2 미리보기는 '하면 이렇게 된다'일 뿐 했다는 근거가 아니라 빼야 한다. ERROR_PREFIX: 실패한
# 도구 결과('Error: [code] …')를 근거에서 빼는 데 쓴다.
from support_agent.chat import Message
from support_agent.confirm import is_confirmation
from support_agent.toolkit import ERROR_PREFIX


# 완료 주장의 종류 하나. said는 문장이 이 종류의 일을 말하는지, shown은 도구 결과가 그 일을 보여 주는지 찾는
# 정규식이다. label과 tools는 하니스 안내문(agent.CLAIM_NOTICE)에 '무엇을 끝냈다고 했는지'와 '실제로 하려면
# 부를 도구'로 들어간다.
@dataclass(frozen=True)
class ClaimKind:
    name: str
    label: str  # 안내문에서 이 종류를 부르는 이름
    tools: str  # 이 일을 실제로 하는 쓰기 도구(들). 안내문에 이름으로 들어간다
    said: re.Pattern[str]  # 문장이 이 종류의 일에 대한 말인지
    shown: re.Pattern[str]  # 이 일을 보여 주는 성공한 도구 결과("<도구 이름>\n<내용>")인지


# 종류별 근거 표(docs/experiments.md 7단계 검출기 표): 취소는 status cancelled, 반품·교환은 접수
# 번호(RT-/EX-)나 kind, 쿠폰·티켓·상담원 연결은 새로 생긴 id, 환불은 0이 아닌 refund_won이나 취소·환불 상태.
# shown 정규식은 tools.py가 json.dumps 기본 구분자로 쓴 결과를 읽으므로 출력 형식이 바뀌면 함께 고쳐야 한다.
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
    # 조회 결과는 주문의 배송지를 보여 줄 뿐 바뀌었다는 것을 보여 주지 않는다. 그래서 쓰기 도구의 성공만
    # 근거로 본다.
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
    # 환불은 취소(전액 환불)와 반품(회수 뒤 환불) 어느 쪽으로도 생기므로 안내문에 두 도구를 함께 적는다.
    ClaimKind(
        "refund",
        "환불",
        "cancel_order 또는 request_return",
        re.compile(r"환불"),
        re.compile(r'"refund_won": [1-9]|"status": "(cancelled|refund_pending|refunded)"'),
    ),
)
# 서비스 전용 예외: 환불액이 승인 기준(10만 원) 이상인 취소·반품은 실행되지 않고 대기열에 들어가며, 도구는
# approval_required 오류로 답한다. 이때 '취소 요청이 접수되었습니다'는 사실이므로, 이 오류를
# '접수·신청·등록되었다'는 말에 한해 취소·반품·환불의 근거로 인정한다. '취소가 완료되었습니다'는 여전히
# 막힌다. 평가(승인 없음)에는 영향이 없다(docs/experiments.md 7단계 코드 검토 뒤에 고친 것).
# 정리하면: 기준을 넘는 취소·반품은 되돌려진 뒤 대기열에 들어가고 도구는 이 오류로 답한다. 그때 "취소 요청이
# 접수되었습니다"는 참이고 "취소되었습니다"는 거짓이므로, 요청을 받았다·등록했다는 문장일 때만 이 오류가
# 그 종류들의 근거가 된다.
_QUEUED = re.compile(
    r"\A(cancel_order|request_return)\n" + re.escape(ERROR_PREFIX) + r": \[approval_required\]"
)
_QUEUED_KINDS = frozenset({"cancel", "return", "refund"})
_RECEIVED = re.compile(r"(접수|신청|등록)\s?(?:[가이를은는도]\s?)?(?:되었|됐|했|하였|해\s?드렸|드렸|마쳤)")

# _DONE: 끝났다는 말. (완료·접수·처리·변경·발급·… 같은 명사) + (조사 하나 허용) + (되었/됐/했/해 드렸/마쳤) +
# (습니다/어요/으며/고), 또는 '남겼습니다', '이루어졌습니다'. 미래('취소해 드리겠습니다'), 조건('완료되면'),
# 의문('취소했는지')은 걸리지 않는다. 마지막의 '고(?![가-힣])'는 연결어미 '…했고'만 잡고, '고' 뒤에 한글이
# 바로 붙어 다른 낱말이 되는 경우는 잡지 않는다.
_DONE = re.compile(
    r"(?:(?:완료|접수|처리|변경|발급|제출|등록|신청|연결|생성|진행|취소|반품|교환|환불)\s?(?:[가이를은는도]\s?)?"
    r"(?:되었|됐|했|하였|해\s?드렸|드렸|마쳤)|남겼|남겨\s?드렸|이루어졌)(?:습니다|어요|으며|고(?![가-힣]))"
)
# 문장 경계: 마침표·느낌표·물음표 뒤의 공백, 또는 줄바꿈. 물음표로 끝나는 문장은 unbacked_claim에서 따로
# 건너뛴다.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
# 절 경계: '…고/…며/…만/…데' 뒤. '본인 확인이 완료되었고, 취소는 가능합니다'가 취소 주장으로 걸리던 오탐을
# 코드 검토에서 찾아 넣었다. 이 문장은 취소가 끝났다는 말이 아니다. 그래서 주장은 자기 절 안에서만
# 판단하고, 종류를 나타내는 말은 동사 앞에 있거나 동사 자체여야 한다(한국어 어순). 쉼표만으로는 끊지
# 않는다(주소의 쉼표 때문).
_CLAUSE_END = re.compile(r"(?<=[고며만데]),?\s+")  # 쉼표만으로는 경계가 아니다("미르로 210, 새길빌딩")
# '배송이 완료되었습니다'는 택배가 도착했다는 말이지 상담원이 한 일이 아니므로, 끝났다는 말 바로 앞이
# '배송(이/은/도)'이면 건너뛴다.
_DELIVERY = re.compile(
    r"배송[이은도]?\s*$"
)  # "배송이 완료되었습니다"는 택배 이야기이지 상담원이 한 일이 아니다


# 검출 결과. 근거가 없는 문장과 그 문장이 말한 종류들. 안내문에는 첫 종류의 label과 tools를 쓴다.
@dataclass(frozen=True)
class Claim:
    sentence: str
    kinds: tuple[str, ...]  # 문장이 말하는 모든 종류. 그중 어느 것도 근거가 없다
    label: str  # 첫 종류의 label
    tools: str  # 첫 종류의 tools


# 대화에서 근거로 쓸 도구 결과를 '도구 이름\n내용' 형태로 모은다. 도구 이름을 앞에 붙이는 것은 배송지처럼
# '어느 도구가 성공했는지'가 근거인 종류(\Achange_shipping_address)를 같은 정규식으로 검사하기 위해서다.
def shown_results(messages: Iterable[Message]) -> list[str]:
    """대화의 성공한 도구 결과들. 각각 "<도구 이름>\\n<내용>" 꼴이다(서비스의 approval_required 오류도
    넣는다. 대기열에 들어간 요청의 기록이기 때문이다). R2 미리보기는 쓰기를 하면 어떻게 되는지를 보여 줄
    뿐 했다는 것을 보여 주지 않으므로 뺀다."""
    out = []
    for m in messages:
        # 도구 결과가 아니거나 R2 미리보기면 건너뛴다.
        if m.role != "tool" or is_confirmation(m.content):
            continue
        entry = f"{m.tool_name or ''}\n{m.content}"
        # 오류 결과는 근거가 아니다. 단, 서비스의 승인 대기(approval_required) 오류는 위의 예외대로 남긴다.
        if not m.content.startswith(f"{ERROR_PREFIX}: [") or _QUEUED.match(entry):
            out.append(entry)
    return out


# 답에서 근거 없는 완료 주장이 든 첫 문장을 찾는다. 없으면 None.
# 순서: 문장으로 나눔 → 물음표 문장 제외 → 절로 나눔 → 절마다 주장한 종류를 찾음 → 그 종류 중 하나라도 근거가
# 있으면 통과(관대한 쪽, 주문 번호는 비교하지 않음). 도구 결과 목록은 주장이 처음 나왔을 때 한 번만 만든다.
def unbacked_claim(text: str, messages: Iterable[Message]) -> Claim | None:
    """`text`에서 어떤 일이 끝났다고 말하지만 그것을 보여 주는 도구 결과가 없는 첫 문장."""
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
            # '접수/신청/등록되었다'는 말인지. 승인 대기 오류를 근거로 쓸 수 있는지가 여기에 달려 있다.
            received = bool(_RECEIVED.search(clause))
            if not any(_shows(kind, result, received) for kind in kinds for result in results):
                return Claim(sentence, tuple(kind.name for kind in kinds), kinds[0].label, kinds[0].tools)
    return None


# 한 절이 끝났다고 말하는 일의 종류들. 한국어는 동사가 뒤에 오므로, 끝났다는 말이 끝나는 곳까지(head)에 나오는
# 종류만 본다. 그래서 '반품 접수를 완료했습니다'는 반품, '취소되었습니다'는 동사 자체가 취소다. 동사 뒤에
# 나오는 낱말은 그 주장의 대상이 아니다.
def _claimed_kinds(clause: str) -> list[ClaimKind]:
    """절이 끝났다고 말하는 일의 종류들: 동사 앞에서 이름이 나오거나("반품 접수를 완료했습니다") 동사
    자체가 그 일이다("취소되었습니다"). 배송 완료는 상담원이 한 일이 아니다."""
    for done in _DONE.finditer(clause):
        if _DELIVERY.search(clause[: done.start()]):
            continue
        head = clause[: done.end()]
        kinds = [kind for kind in KINDS if kind.said.search(head)]
        if kinds:
            return kinds
    return []


# 도구 결과 하나가 이 종류의 근거인지. 기본은 shown 정규식이고, 승인 대기 오류는 '접수되었다'는 말이고
# 취소·반품·환불일 때만 근거가 된다.
def _shows(kind: ClaimKind, result: str, received: bool) -> bool:
    if kind.shown.search(result):
        return True
    return received and kind.name in _QUEUED_KINDS and bool(_QUEUED.match(result))
