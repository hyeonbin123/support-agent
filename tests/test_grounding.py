"""상담원 답의 '근거 없는 값' 보조 지표(analyze grounding)를 모델 없이 시험한다: 손으로 만든 작은 대화,
8단계 M2b 개발용 기록의 dev-008 시도 3, 합성한 실행 폴더, 등록한 규칙 문서."""

# [테스트 묶음] docs/experiments.md 끝 절 "보조 지표: 상담원 답의 근거 없는 값"이 정한 것을 코드가 그대로
# 하는지 고정한다. 판정이 아니라 보고만 하는 지표다.
# - 계산 값 출처: 도구 날짜 + 7·+8일(반품·교환 마감, rules.return_deadline과 같은 셈), 발급일·오늘 + 30일(쿠폰
#   만료), 오늘 − 29일(쿠폰 한도를 세는 '최근 30일'의 첫날), 한 도구 결과 안의 금액 2~3개의 합·차(규정 금액
#   포함), 단가 × 수량.
# - 출처가 아닌 것: 규정 금액끼리만의 조합, 서로 다른 도구 결과의 금액을 섞은 조합, 그 답 뒤에 온 도구 결과,
#   상담원 자신이 앞서 한 말.
# - 형식 예시('예를 들어 "P-0001" 형식'), 자리 표시 꼴 숫자(0001), 더 긴 숫자의 일부, 이메일의 대소문자.
# - 전달된 답의 기준이 판정(judge.delivered_texts)과 같다(첫 인사·전달하지 않은 답·도구 호출 메시지 제외).
# - 기록: 개발용 dev-008 시도 3에서 115,400원은 근거 없음으로 잡히고 112,300원은 잡히지 않는다.
# - 실행 단위 표, 시험용 기록은 수만, 손 확인 표본은 개발용만(고정 seed, 같은 대화는 한 번), CLI 연결, 문서.
# [방법] GPU·모델 없이 메모리의 대화 dict와 임시 폴더의 합성 기록, 커밋된 reports/ 개발용 기록 하나를 읽는다.

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from support_agent import analyze, grounding, rules, sim_report
from support_agent.agent import build_system_prompt, load_policy
from support_agent.chat import Message
from support_agent.clock import kst_date
from support_agent.config import FIRST_AGENT_MESSAGE
from support_agent.judge import delivered_texts
from support_agent.paths import REPORTS, ROOT
from support_agent.tasks import load_tasks

EXPERIMENTS = (ROOT / "docs" / "experiments.md").read_text(encoding="utf-8")
DEV = {task.id: task for task in load_tasks("dev")}
M2B = REPORTS / "20261003-200306-qwen3.5-4b-P1-R0-G0-F0-m2b"
# 개발용 과제는 모두 2026-09-14 10:00 KST(월요일)에 고정돼 있어 시스템 프롬프트가 하나다(tasks/dev.yaml
# 머리말).
TODAY = date(2026, 9, 14)
SYSTEM = build_system_prompt(load_policy(), DEV["dev-008"].now)
SYSTEM_SOURCES = sim_report.source_tokens([SYSTEM])

# get_order 결과 꼴(dev-008의 O-91011과 같은 금액): 단가 48,700 × 1, 9,800 × 1, 33,300 × 2, 합계 125,100원.
ORDER = {
    "order_id": "O-91011",
    "status": "delivered",
    "items": [
        {"line_no": 1, "product_name": "니트 가디건", "quantity": 1, "unit_price_won": 48700},
        {"line_no": 2, "product_name": "면 양말 5켤레", "quantity": 1, "unit_price_won": 9800},
        {"line_no": 3, "product_name": "기모 후드티", "quantity": 2, "unit_price_won": 33300},
    ],
    "items_won": 125100,
    "shipping_fee_won": 0,
    "discount_won": 0,
    "total_won": 125100,
    "payment": {"method": "card", "amount_won": 125100, "status": "paid"},
}


# ==================================================================== 대화 만들기


def user(text: str, **extra: Any) -> dict[str, Any]:
    return {"role": "user", "content": text, **extra}


def say(text: str, **extra: Any) -> dict[str, Any]:
    return {"role": "assistant", "content": text, **extra}


def call(name: str) -> dict[str, Any]:
    return {"role": "assistant", "content": "", "tool_calls": [{"name": name, "arguments": {}, "id": ""}]}


def tool(name: str, content: dict[str, Any] | str) -> dict[str, Any]:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return {"role": "tool", "tool_name": name, "content": text}


# 첫 인사 뒤의 메시지들로 대화를 만들어, 전달된 답마다 근거 없는 값 목록(없으면 빈 튜플)을 돌려준다.
def check(*messages: dict[str, Any]) -> list[tuple[str, ...]]:
    answers = grounding.check_answers([say(FIRST_AGENT_MESSAGE), *messages], SYSTEM, TODAY)
    return [answer.missing for answer in answers]


# 전제: 그 값이 시스템 프롬프트(규정 문서와 오늘 시각)에 우연히 있지 않다. 있으면 테스트가 보려는 규칙이
# 아니라 프롬프트 출처 때문에 통과하므로 먼저 확인한다.
def not_in_prompt(*tokens: str) -> None:
    for token in tokens:
        assert not sim_report.grounded(token, SYSTEM_SOURCES), token


# ==================================================================== 계산 값 출처


# 반품·교환 마감일(수령일 + 7)과 그다음 날(+8, 규정 예시의 '1월 13일부터 불가')은 출처가 있다. 둘 다 아닌 날은
# 근거 없음이다. +7은 rules.return_deadline과 같은 셈이다(KST 날짜에 7일).
def test_dates_counted_from_a_tool_date_by_the_policy_offsets():
    not_in_prompt("date:09-18", "date:09-19", "date:09-20")
    shipment = {
        "order_id": "O-91013",
        "status": "delivered",
        "delivered_at": "2026-09-11 14:00",
        "promised_by": "2026-09-10",
    }
    answers = check(
        user("교환하고 싶어요. 주문은 O-91013입니다."),
        call("track_shipment"),
        tool("track_shipment", shipment),
        say("9월 11일에 받으셔서 9월 18일까지 교환하실 수 있고, 9월 19일부터는 접수가 안 됩니다."),
        say("다시 말씀드리면 마감은 9월 20일입니다."),
    )
    assert answers == [(), ("date:09-20",)]
    assert grounding.DATE_OFFSETS_DAYS[:2] == (rules.RETURN_WINDOW_DAYS, rules.RETURN_WINDOW_DAYS + 1)
    # 도구는 KST 시각을 적는다. UTC로 9월 11일 15:30은 KST 9월 12일 00:30이라 마감은 9월 19일이다.
    for delivered in (datetime(2026, 9, 11, 5, 0, tzinfo=UTC), datetime(2026, 9, 11, 15, 30, tzinfo=UTC)):
        offset = timedelta(days=grounding.DATE_OFFSETS_DAYS[0])
        assert kst_date(delivered) + offset == rules.return_deadline(delivered)


# 쿠폰은 발급일부터 30일(규정 8절): 도구가 보여 준 발급일 + 30, 오늘 발급하면 오늘 + 30(검토의 거친 셈에서
# 걸린 M2b dev-012 꼴 '오늘이 9월 14일이므로 10월 14일까지'). 한도를 세는 '최근 30일(오늘 포함)'의 첫날은
# 오늘 − 29(규정 예시: 3월 30일이면 3월 1일부터).
def test_coupon_dates_from_the_issue_date_and_today():
    not_in_prompt("date:10-01", "date:10-14", "date:08-16", "date:10-15")
    customer = {
        "customer_id": "C-9112",
        "compensation_coupons": [
            {"coupon_id": "CP-O-91019-1", "amount_won": 2000, "issued_at": "2026-09-01 10:00"},
        ],
    }
    answers = check(
        user("쿠폰을 더 받을 수 있나요?"),
        call("get_customer"),
        tool("get_customer", customer),
        say("9월 1일에 발급된 쿠폰은 10월 1일까지 쓰실 수 있습니다."),
        say("오늘 발급해 드리면 10월 14일까지 사용 가능합니다."),
        say("최근 30일은 8월 16일부터 셉니다."),
        say("쿠폰은 10월 15일까지입니다."),
    )
    assert answers == [(), (), (), ("date:10-15",)]
    assert grounding.DATE_OFFSETS_DAYS[2:] == (rules.COUPON_VALID_DAYS, -(rules.COUPON_WINDOW_DAYS - 1))


# 단가 × 수량(33,300 × 2 = 66,600), 두 금액의 합(48,700 + 66,600)·차(125,100 − 9,800), 규정의 반품 배송비를
# 넣은 세 금액(48,700 + 66,600 − 3,000)은 출처가 있다. 어느 것으로도 만들 수 없는 115,400원은 근거 없음이다.
def test_amounts_from_sums_differences_and_unit_price_times_quantity():
    not_in_prompt("num:66600", "num:115300", "num:112300", "num:115400")
    answers = check(
        user("O-91011 가디건과 후드티를 반품할게요."),
        call("get_order"),
        tool("get_order", ORDER),
        say("후드티 2개는 66,600원이고 가디건과 합치면 115,300원입니다."),
        say("반품 배송비 3,000원을 빼면 112,300원이 환불됩니다."),
        say("양말을 뺀 금액은 115,300원입니다."),
        say("환불액은 115,400원입니다."),
    )
    assert answers == [(), (), (), ("num:115400",)]
    assert grounding.POLICY_AMOUNTS == {
        rules.RETURN_FEE_WON,
        rules.COUPON_DELAY_SHORT_WON,
        rules.COUPON_DELAY_LONG_WON,
        rules.COUPON_DEFECTIVE_WON,
    }


# 조합에는 그 도구 결과의 금액이 하나 이상 들어가야 한다: 도구 결과가 없을 때 규정 금액끼리의 합(5,000 +
# 2,000)은 출처가 아니다. 서로 다른 도구 결과의 금액은 섞지 않는다(두 get_order 합계의 합). 한 결과 안의 두
# 금액(한 list_orders의 두 주문 합계)은 섞는다.
def test_combinations_need_an_amount_of_one_tool_result():
    not_in_prompt("num:7000", "num:54100")
    assert check(user("쿠폰은 얼마까지 받을 수 있나요?"), say("최대 7,000원까지 받으실 수 있습니다.")) == [
        ("num:7000",)
    ]
    first = {"order_id": "O-91020", "total_won": 31400}
    second = {"order_id": "O-91021", "total_won": 22700}
    separate = check(
        user("두 주문 합계가 얼마예요?"),
        call("get_order"),
        tool("get_order", first),
        call("get_order"),
        tool("get_order", second),
        say("두 주문을 합치면 54,100원입니다."),
    )
    assert separate == [("num:54100",)]
    listed = check(
        user("두 주문 합계가 얼마예요?"),
        call("list_orders"),
        tool("list_orders", {"customer_id": "C-9120", "orders": [first, second]}),
        say("두 주문 합계는 54,100원입니다."),
    )
    assert listed == [()]


# ==================================================================== 형식 예시, 더 긴 숫자, 이메일


# 형식 예시는 값으로 읽지 않는다: '예를 들어 …', '(예: …)' 뒤의 글(문장·괄호 끝까지), '… 형식'·'…처럼' 앞의
# 낱말(검토의 거친 셈에서 걸린 M2b dev-004 꼴), 표지가 없어도 자리 표시 꼴인 숫자(0001). 출처에 없는 진짜 번호
# 꼴은 근거 없음이다.
def test_format_examples_are_not_values():
    not_in_prompt("num:0001", "num:12345", "num:91234", "num:9150")
    answers = check(
        user("칫솔 상품을 찾고 있어요."),
        say('상품 번호를 알려 주시면 확인해 드리겠습니다. 예를 들어 "P-0001" 형식으로 말씀해 주시면 됩니다.'),
        say("주문 번호(예: O-12345)를 알려 주세요."),
        say("O-91234처럼 O로 시작하는 번호입니다."),
        say("P-0001 번호로는 상품이 없습니다."),
        say("상품 P-9150은 품절입니다."),
    )
    assert answers == [(), (), (), (), ("num:9150",)]
    assert all(
        grounding.is_placeholder(d) for d in ("0001", "0012", "1234", "56789", "9876", "1111", "000000")
    )
    assert not any(grounding.is_placeholder(d) for d in ("9150", "91234", "0101", "115400", "04524"))


# 숫자는 더 긴 출처 숫자의 일부여도 출처가 있다(sim_report.grounded와 같은 규칙: 전화번호 끝 네 자리).
def test_part_of_a_longer_source_number_is_grounded():
    not_in_prompt("num:9108", "num:9180")
    customer = {"customer_id": "C-9108", "phone": "01000009108", "email": "mingyu.song@example.com"}
    answers = check(
        user("송민규입니다."),
        call("get_customer"),
        tool("get_customer", customer),
        say("끝자리 9108 번호로 확인했습니다."),
        say("등록된 번호 끝자리는 9180입니다."),
    )
    assert answers == [(), ("num:9180",)]


# 이메일은 소문자로 바꿔 통째로 비교한다: 도구가 보여 준 주소와 고객이 말한 주소는 출처가 있고, 형식 예시는
# 읽지 않으며, 도메인이 다른 주소는 근거 없음이다.
def test_emails_compare_without_case_and_come_from_the_customer_or_tools():
    customer = {"customer_id": "C-9108", "email": "mingyu.song@example.com"}
    answers = check(
        user("제 메일은 song.mk@example.net 입니다."),
        call("get_customer"),
        tool("get_customer", customer),
        say("Mingyu.Song@Example.com 주소가 등록되어 있습니다."),
        say("말씀하신 song.mk@example.net 주소와 다릅니다."),
        say("이메일(예: hong@example.com)을 알려 주세요."),
        say("mingyu.song@example.co.kr 로 보내 드렸습니다."),
    )
    assert answers == [(), (), (), ("email:mingyu.song@example.co.kr",)]


# ==================================================================== 출처의 범위


# 출처는 그 답 앞까지 에이전트가 받은 글뿐이다: 도구 결과보다 먼저 말한 접수 번호는 근거 없음이고(결과가 온
# 뒤에는 출처가 있음), 상담원이 앞서 한 말은 출처가 아니다(같은 틀린 금액을 되풀이해도 매번 근거 없음).
def test_only_what_the_agent_had_received_before_the_answer_counts():
    not_in_prompt("num:910111", "num:115400")
    receipt = {
        "request_id": "RT-O-91011-1",
        "order_id": "O-91011",
        "line_nos": [1, 3],
        "refund_won": 112300,
        "return_fee_won": 3000,
    }
    answers = check(
        user("가디건과 후드티를 반품할게요. 주문은 O-91011입니다."),
        say("반품 번호는 RT-O-91011-1이 될 예정입니다."),
        call("request_return"),
        tool("request_return", receipt),
        say("반품 번호는 RT-O-91011-1입니다."),
        say("환불액은 115,400원입니다."),
        say("다시 말씀드리면 115,400원입니다."),
    )
    assert answers == [("num:910111",), (), ("num:115400",), ("num:115400",)]


# 고객의 말(음성 실행이면 인식된 글)과 하니스 안내문도 에이전트가 받은 글이라 출처다.
def test_customer_words_and_harness_notices_are_sources():
    not_in_prompt("num:01024681357", "date:09-30")
    answers = check(
        user("010-2468-1357로 연락 주세요."),
        say("010-2468-1357로 연락드리겠습니다."),
        user("(9월 30일까지 답하세요)", harness=True),
        say("9월 30일까지 답하겠습니다."),
    )
    assert answers == [(), ()]


# 전달된 답만 센다: 첫 인사, 고객에게 가지 않은 답(delivered=False), 도구 호출 메시지는 빼고, 그 글은 출처도
# 아니다. 기준은 판정(judge.delivered_texts)과 같다.
def test_only_delivered_answers_are_read():
    messages = [
        say(FIRST_AGENT_MESSAGE),
        user("주문 O-91011 환불액이 궁금해요."),
        say("환불액은 61,234원입니다.", delivered=False),
        user("(형식 안내)", harness=True),
        {**call("get_order"), "content": "조회하겠습니다 52,345원"},
        tool("get_order", ORDER),
        say("환불액은 71,234원입니다."),
    ]
    assert grounding.delivered_answers(messages) == [(6, "환불액은 71,234원입니다.")]
    assert [text for _, text in grounding.delivered_answers(messages)] == delivered_texts(
        [Message.from_dict(m) for m in messages]
    )
    answers = grounding.check_answers(messages, SYSTEM, TODAY)
    assert [(a.index, a.missing) for a in answers] == [(6, ("num:71234",))]


# 기록에서도 전달된 답의 목록이 판정이 전달 값을 찾는 글과 같다(지표 값은 보지 않고 목록만 맞춘다).
def test_delivered_answers_are_the_judge_texts_on_a_record():
    for episode in analyze.load_episodes(M2B):
        messages = episode["messages"]
        expected = delivered_texts([Message.from_dict(m) for m in messages])
        assert [text for _, text in grounding.delivered_answers(messages)] == expected


# ==================================================================== 기록: dev-008 시도 3


# 개발용 M2b 기록의 dev-008 시도 3(성공 판정): 상담원이 환불액을 115,400원이라고 세 번 말했다(도구 결과는
# 112,300원). 115,400원이 든 세 답이 모두 그 값 하나로 잡히고, 112,300원(도구 결과에 있음)은 잡히지 않는다.
# 66,600원(33,300 × 2)과 3,000원(규정)도 잡히지 않는다.
def test_dev008_trial3_flags_115400_and_not_112300():
    episode = next(e for e in analyze.load_episodes(M2B) if e["task_id"] == "dev-008" and e["trial"] == 3)
    assert analyze.is_success(episode)
    answers = grounding.check_answers(episode["messages"], SYSTEM, TODAY)
    assert len(answers) == 4
    flagged = [a for a in answers if a.missing]
    assert [a.missing for a in flagged] == [("num:115400",)] * 3
    assert all("115,400원" in a.text for a in flagged)
    assert any("112,300원" in a.text for a in flagged)
    assert any("66,600원" in a.text for a in flagged)


# ==================================================================== 실행 단위 표, 표본, CLI


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# 판정 요소만 갖춘 에피소드 dict. 판정 값이 없는 infra 오류와 과제 해시가 다른 에피소드도 만든다.
def ep(
    trial: int,
    messages: list[dict[str, Any]],
    *,
    success: bool = True,
    infra: bool = False,
    task_sha: str | None = None,
) -> dict[str, Any]:
    return {
        "task_id": "dev-008",
        "task_sha256": task_sha or DEV["dev-008"].sha256(),
        "trial": trial,
        "status": "infra_error" if infra else "completed",
        "termination": "infra_error" if infra else "user_stop",
        "verdict": None if infra else {"success": success, "db_match": success, "values": {}},
        "messages": [say(FIRST_AGENT_MESSAGE), *messages],
    }


# 임시 폴더에 manifest(과제 파일, 과제·시스템 프롬프트 해시, 음성 축)와 episodes.jsonl을 쓴다.
def write_run(
    tmp_path: Path,
    name: str,
    episodes: list[dict[str, Any]] | None,
    *,
    tasks_file: str = "tasks/dev.yaml",
    agent_system: str | None = None,
) -> Path:
    folder = tmp_path / name
    folder.mkdir()
    manifest = {
        "tasks_file": tasks_file,
        "task_sha256": {"dev-008": DEV["dev-008"].sha256()},
        "config": {"voice": "V0"},
        "prompt_sha256": {"policy": sha256(load_policy()), "agent_system": agent_system or sha256(SYSTEM)},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    if episodes is not None:
        lines = [json.dumps(e, ensure_ascii=False) for e in episodes]
        (folder / "episodes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder


ASKED = user("가디건과 후드티를 반품할게요.")
WRONG = ep(0, [ASKED, call("get_order"), tool("get_order", ORDER), say("환불액은 115,400원입니다.")])
RIGHT = ep(
    1,
    [ASKED, call("get_order"), tool("get_order", ORDER), say("후드티 2개는 66,600원입니다.")],
    success=False,
)
BROKEN = ep(2, [], infra=True)
CHANGED = ep(3, [ASKED, say("환불액은 115,400원입니다.")], task_sha="0" * 64)


# 실행 하나의 표 행: 판정된 에피소드 2(infra 1, 과제 바뀜 1은 빠짐), 전달된 답 2 중 1, 근거 없는 값 1(숫자),
# 그런 에피소드 1/2, 성공 에피소드 1개 중 1. 시험용 기록은 이름에 '(시험용)'을 붙여 같은 수만 낸다. 시스템
# 프롬프트 해시가 지금 코드와 다른 기록과 에피소드 기록이 없는 폴더는 표 아래에 이유를 적는다.
def test_table_counts_answers_episodes_and_successes(tmp_path):
    dev = write_run(tmp_path, "dev", [WRONG, RIGHT, BROKEN, CHANGED])
    test = write_run(tmp_path, "t", [WRONG, RIGHT], tasks_file="tasks/test.yaml")
    old = write_run(tmp_path, "old", [WRONG], agent_system="0" * 64)
    empty = write_run(tmp_path, "empty", None)
    table = grounding.grounding_table([dev, test, old, empty])
    lines = table.splitlines()
    assert lines[0] == grounding.HEADER
    assert "| dev | 2 | 2 | 1/2 (50.0%) | 1 (1·0·0) | 1/2 (50.0%) | 1/1 (100.0%) | 1 · 1 |" in lines
    assert "| t (시험용) | 2 | 2 | 1/2 (50.0%) | 1 (1·0·0) | 1/2 (50.0%) | 1/1 (100.0%) | 0 · 0 |" in lines
    assert (
        "- old: 시스템 프롬프트가 지금 코드로 만든 것과 달라 세지 않음 (manifest의 agent_system 해시)"
        in lines
    )
    assert "- empty: 에피소드 기록 없음" in lines
    # 개발용 합계: 같은 대화는 한 번만 센다(시험용은 넣지 않는다).
    assert (
        "- 개발용 기록 1개 합계 (같은 대화는 한 번): 에피소드 2, 전달된 답 2, "
        "근거 없는 값이 든 답 1/2 (50.0%), 그런 에피소드 1/2 (50.0%), 성공 에피소드 중 1/1 (100.0%)"
    ) in lines


# 개발용 합계와 손 확인 표본은 같은 대화를 한 번만 넣는다: 끼워 넣은 기록처럼 다른 기록에 같은 에피소드가
# 있으면 한 번. 시험용 기록은 표본에 넣지 않고 그 수를 적는다. seed가 고정이라 같은 입력은 같은 표본이다.
def test_hand_sheet_samples_development_answers_only(tmp_path):
    dev = write_run(tmp_path, "dev", [WRONG, RIGHT])
    copy = write_run(tmp_path, "copy", [WRONG])
    test = write_run(tmp_path, "t", [WRONG], tasks_file="tasks/test.yaml")
    sheet = grounding.hand_sheet([dev, copy, test], n=20)
    assert sheet.count("\n## ") == 1 and "num:115400" in sheet and "환불액은 115,400원입니다." in sheet
    assert "시험용 기록 1개는 표본에 넣지 않았다" in sheet
    assert grounding.hand_sheet([dev, copy, test], n=20) == sheet
    table = grounding.grounding_table([dev, copy])
    assert "- 개발용 기록 2개 합계 (같은 대화는 한 번): 에피소드 2, 전달된 답 2," in table


# 모집단이 표본 수보다 크면 고정 seed로 n개를 뽑는다(뽑은 순서가 아니라 모집단 순서로 적음). 다른 대화 세 개
# 중 둘.
def test_hand_sheet_draws_n_with_the_fixed_seed(tmp_path):
    wrong = [
        ep(k, [ASKED, call("get_order"), tool("get_order", ORDER), say(f"환불액은 {amount}원입니다.")])
        for k, amount in enumerate(("115,400", "116,400", "117,400"))
    ]
    run_dir = write_run(tmp_path, "dev", wrong)
    pool = grounding.flagged_pool([run_dir])
    assert [(item.run, item.trial, item.answer.missing) for item in pool] == [
        ("dev", 0, ("num:115400",)),
        ("dev", 1, ("num:116400",)),
        ("dev", 2, ("num:117400",)),
    ]
    picked = grounding.draw(pool, 2)
    assert len(picked) == 2 and picked == grounding.draw(pool, 2)
    assert [pool.index(item) for item in picked] == sorted(pool.index(item) for item in picked)
    assert grounding.GROUNDING_SEED == 20261024 and grounding.SHEET_SIZE == 20
    sheet = grounding.hand_sheet([run_dir], n=2)
    assert sheet.count("\n## ") == 2


# 대화를 다른 실행에서 다시 돌렸을 때처럼 도구 호출 id(assistant의 tool_calls[].id, tool의 tool_call_id)만
# 바꾼 사본.
def with_call_ids(episode: dict[str, Any], prefix: str) -> dict[str, Any]:
    copy = json.loads(json.dumps(episode, ensure_ascii=False))
    for number, message in enumerate(copy["messages"]):
        for item in message.get("tool_calls", ()):
            item["id"] = f"{prefix}-{number}"
        if message["role"] == "tool":
            message["tool_call_id"] = f"{prefix}-{number - 1}"
    return copy


# '같은 대화'는 공통 규칙 "음성 대조"의 '대화'와 같이 도구 호출 id를 빼고 본다: id는 실행마다 새로 붙는
# 값이라, 장치가 작동하지 않아 대화가 같은 두 실행(예: P1·7B의 0.34.2·0.35.1 두 기록)도 id까지 비교하면
# 다른 대화로 세어진다. 개발용 합계와 손 확인 모집단에 한 번만 들어가야 하고, 기록마다의 행은 그대로다.
def test_the_same_conversation_with_other_tool_call_ids_counts_once(tmp_path):
    first = write_run(tmp_path, "a", [with_call_ids(WRONG, "call_a")])
    second = write_run(tmp_path, "b", [with_call_ids(WRONG, "call_b")])
    table = grounding.grounding_table([first, second])
    assert "| a | 1 | 1 | 1/1 (100.0%) | 1 (1·0·0) | 1/1 (100.0%) | 1/1 (100.0%) | 0 · 0 |" in table
    assert "| b | 1 | 1 | 1/1 (100.0%) | 1 (1·0·0) | 1/1 (100.0%) | 1/1 (100.0%) | 0 · 0 |" in table
    assert "- 개발용 기록 2개 합계 (같은 대화는 한 번): 에피소드 1, 전달된 답 1," in table
    assert [item.run for item in grounding.flagged_pool([first, second])] == ["a"]
    # 글이 하나라도 다르면 다른 대화다(도구 호출 인자가 다른 경우).
    other = with_call_ids(WRONG, "call_c")
    other["messages"][2]["tool_calls"][0]["arguments"] = {"order_id": "O-91012"}
    third = write_run(tmp_path, "c", [other])
    assert len(grounding.flagged_pool([first, second, third])) == 2


# analyze grounding: 표를 내고, --sheet를 주면 손 확인 표본 파일을 쓴다.
def test_analyze_grounding_command(tmp_path, monkeypatch, capsys):
    run_dir = write_run(tmp_path, "dev", [WRONG, RIGHT])
    sheet = tmp_path / "sheet.md"
    monkeypatch.setattr("sys.argv", ["analyze", "grounding", str(run_dir), "--sheet", str(sheet), "--n", "5"])
    analyze.main()
    out = capsys.readouterr().out
    assert out.startswith(grounding.HEADER) and "| dev | 2 | 2 | 1/2 (50.0%) |" in out
    assert sheet.read_text(encoding="utf-8") == grounding.hand_sheet([run_dir], n=5)
    assert f"wrote {sheet}" in out


# ==================================================================== 등록한 규칙 문서


SECTION = "## 보조 지표: 상담원 답의 근거 없는 값"


# 규칙은 기록에 계산하기 전에 docs/experiments.md에 커밋한다. 문서의 정의가 코드의 상수와 같고, 명령·표본·
# 문턱·한계가 적혀 있는지 본다.
@pytest.mark.parametrize(
    "phrase",
    [
        "보고만, 판정 아님",
        "judge.delivered_texts",
        "sim_report.value_tokens",
        "4자리 이상",
        "+7·+8·+30·−29일",
        "단가 × 수량",
        "한 도구 결과 안",
        "규정 금액",
        "형식 예시",
        "더 긴 숫자의 일부",
        "seed 20261024",
        "20개",
        "0.5",
        "uv run --no-sync python -m support_agent.analyze grounding reports/2026*",
        "시험용 기록은 수만",
        "한계",
    ],
)
def test_the_rule_is_registered(phrase):
    assert SECTION in EXPERIMENTS
    assert phrase in EXPERIMENTS.split(SECTION, 1)[1]
