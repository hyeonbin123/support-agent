"""2026-09-19 검토에서 찾은 결함들. 하나마다 테스트로 고정한다."""

# [테스트 묶음] 2026-09-19 리뷰에서 찾은 결함마다 회귀 테스트를 하나씩 고정한다: 전달 값이 시스템 프롬프트에
# 새는 과제, 본문에 샌 도구 호출의 여러 모양, 시뮬레이터 종료 토큰의 변형, 종료 토큰과 함께 온 동의, 예상 못
# 한 예외, 하이픈 닮은 문자, 이름·연락처 표기, 이상한 인자, 취소된 주문의 보상 쿠폰, 발급된 쿠폰의 노출.
# [왜 중요한가] 이 결함들은 측정 수치를 조용히 바꾸는 종류다(성공을 실패로, 실패를 성공으로, 기반 오류를
# 에이전트 실패로). 고친 뒤 다시 생기지 않게 한다.
# [방법] seed DB와 실제 도구 등록부(test_tasks.py의 REGISTRY), 대본 제공자·대본 고객을 쓴다. 모델은 쓰지
# 않는다.

from __future__ import annotations

# test_tasks의 전체 과제 목록, 고객 연락처 표, 실제 도구 등록부, 대본 에피소드 도우미를 다시 쓴다. 에피소드
# 실행(run_episode)과 종료 토큰 분리(split_ending), 판정의 과제 검사와 값 찾기를 직접 부른다.
import json

import pytest
from conftest import NOW
from test_tasks import ALL_TASKS, CONTACTS, REGISTRY, episode, say_values

from support_agent import db
from support_agent.agent import build_system_prompt, format_problem, load_policy
from support_agent.chat import ChatResponse, ScriptedProvider, ToolCall
from support_agent.config import RunConfig
from support_agent.episode import run_episode, split_ending
from support_agent.judge import validate_task, value_found
from support_agent.seed import build_seed_engine
from support_agent.tasks import RequiredValue
from support_agent.toolkit import ToolContext, execute
from support_agent.user_sim import ScriptedUser


# 모든 과제 파일에 대해, 꼭 전달할 값이 에이전트 시스템 프롬프트(규정 문서와 그 예시 포함)에 들어 있지 않아야
# 한다. 프롬프트의 예시 금액이 정답과 같으면 조회하지 않고 예시를 읊어도 성공이 된다.
@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda task: task.id)
def test_no_required_value_is_given_away_by_the_agent_prompt(task):
    prompt = build_system_prompt(load_policy(), task.now)
    assert validate_task(task, build_seed_engine(), REGISTRY, prompt_texts=[prompt]) == []


# 반대 방향 확인: 프롬프트 글에 전달 값('9월 2일')이 있으면 validate_task가 'agent prompt'를 이유로 문제를
# 보고한다.
def test_validate_task_reports_a_value_that_the_prompt_contains():
    task = next(t for t in ALL_TASKS if t.id == "smoke-refusal-01")
    problems = validate_task(task, build_seed_engine(), REGISTRY, prompt_texts=["예: 9월 2일 수령"])
    assert any("agent prompt" in p for p in problems)


# qwen2.5가 도구 호출을 본문에 쓰는 여러 모양(공백·줄바꿈, 코드 펜스, 앞에 붙은 문장, 키 순서, parameters
# 키)을 모두 형식 오류(leaked_tool_call)로 잡아야 한다. 놓치면 그 JSON이 고객에게 전달된다.
@pytest.mark.parametrize(
    "text",
    [
        '{ "name": "get_order", "arguments": {"order_id": "O-1"}}',
        '{\n  "name": "get_order",\n  "arguments": {"order_id": "O-1"}\n}',
        '```json\n{"name": "get_order", "arguments": {"order_id": "O-1"}}\n```',
        '조회하겠습니다.\n{"name": "get_order", "arguments": {"order_id": "O-1"}}',
        '{"arguments": {"order_id": "O-1"}, "name": "get_order"}',
        '{"name": "get_order", "parameters": {"order_id": "O-1"}}',
    ],
)
def test_a_tool_call_written_into_the_text_is_a_format_error(text):
    assert format_problem(ChatResponse(text=text)) == "leaked_tool_call"


# 반대로 금액 문장, 중괄호가 든 보통 문장, 호출 모양이 아닌 JSON은 형식 오류가 아니다. 잘못 걸면 멀쩡한 답을
# 돌려보내게 된다.
@pytest.mark.parametrize(
    "text", ["환불 금액은 32,400원입니다.", "주문 {O-1}을 확인했습니다.", '{"note": "json이지만 호출 아님"}']
)
def test_ordinary_text_is_not_a_format_error(text):
    assert format_problem(ChatResponse(text=text)) is None


# 시뮬레이터의 종료 토큰은 공백, 대소문자, # 개수, 전각 문자가 달라도 찾아야 하고, 토큰 앞의 말은 남긴다.
# OUT-OF-SCOPE도 같다. 토큰이 없으면 말 그대로다. 토큰을 놓치면 대화가 고객 턴 한도까지 이어져 판정하지 않는
# 종료가 된다.
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("###STOP###", ("", "user_stop")),
        ("네, 진행해 주세요. 감사합니다 ###STOP###", ("네, 진행해 주세요. 감사합니다", "user_stop")),
        ("### stop ###", ("", "user_stop")),
        ("##STOP##", ("", "user_stop")),
        ("＃＃＃STOP＃＃＃", ("", "user_stop")),
        ("###OUT-OF-SCOPE###", ("", "out_of_scope")),
        ("### out of scope ###", ("", "out_of_scope")),
        ("그만할게요", ("그만할게요", None)),
    ],
)
def test_end_tokens_are_found_tolerantly(text, expected):
    assert split_ending(text) == expected


# 고객이 '네, 진행해 주세요 ###STOP###'처럼 동의와 종료 토큰을 한 번에 보내면, 토큰을 떼어 낸 동의를
# 에이전트에게 먼저 전해 취소를 실행하게 한 뒤 user_stop으로 끝나야 한다. 토큰만 보고 바로 끝내면 에이전트가
# 하려던 일을 못 해 실패로 잘못 세어진다.
def test_consent_that_arrives_with_the_stop_token_is_still_acted_on():
    task = next(t for t in ALL_TASKS if t.id == "smoke-action-01")
    name, contact = CONTACTS[task.customer_id]
    script = [
        ToolCall("find_customer", {"name": name, "contact": contact}),
        "O-90003 주문을 단순 변심으로 취소할까요?",
        ToolCall("cancel_order", {"order_id": "O-90003", "reason": "changed_mind"}),
        say_values(task),
    ]
    result = run_episode(
        task,
        0,
        config=RunConfig(policy="P1"),
        provider=ScriptedProvider(script),
        user=ScriptedUser(["취소해 주세요.", "네, 진행해 주세요 ###STOP###"]),
        registry=REGISTRY,
        seed_engine=build_seed_engine(),
        policy_text=load_policy(),
    )
    assert result.termination == "user_stop" and result.verdict.success


# 제공자 쪽에서 예상 못 한 예외(AttributeError 같은 우리 코드의 버그)가 나도 실행 전체가 죽지 않고 그
# 에피소드만 infra_error가 되며, 원인을 찾을 수 있게 Traceback이 기록된다. 대본이 떨어진 경우도 마찬가지로
# 결과가 나온다.
def test_an_unexpected_exception_is_an_infra_error_with_a_traceback():
    class Broken(ScriptedProvider):
        def chat(self, *args, **kwargs):
            raise AttributeError("'dict' object has no attribute 'strip'")

    result = episode(ALL_TASKS[0], [])
    assert result is not None  # 대본이 떨어지기만 한 경우도 멈춤(예외)이 아니라 infra 오류다
    result = run_episode(
        ALL_TASKS[0],
        0,
        config=RunConfig(),
        provider=Broken([]),
        user=ScriptedUser(["안녕하세요"]),
        registry=REGISTRY,
        seed_engine=build_seed_engine(),
        policy_text=load_policy(),
    )
    assert result.status == "infra_error" and "Traceback" in result.error


# 이관 행의 고객 번호는 판정 비교에서 뺀다. 규정은 본인 확인 전후 어느 쪽의 이관도 허용하므로, 그 차이로 DB
# 일치가 갈리면 안 된다.
def test_a_handoff_matches_whether_or_not_the_customer_was_verified_first():
    assert ("handoffs", "customer_id") in db.IGNORED_COLUMNS


# 글 전달 값은 하이픈 닮은 문자(줄바꿈 없는 하이픈, en 대시)와 폭 없는 공백을 같게 보되, 앞뒤에 숫자가 더 붙은
# 더 긴 번호는 인정하지 않는다. 모델이 다른 하이픈 문자를 쓰면 맞는 송장 번호를 놓치던 결함의 회귀 테스트다.
@pytest.mark.parametrize(
    ("value", "text", "found"),
    [
        ("5550-1207-9001", "송장 번호는 5550‑1207‑9001입니다.", True),  # 줄바꿈 없는 하이픈
        ("5550-1207-9001", "송장 번호는 5550–1207–9001입니다.", True),  # en 대시
        ("5550-1207-9001", "15550-1207-90011", False),
        ("한빛택배", "한빛​택배로 배송 중입니다.", True),
    ],
)
def test_text_values_tolerate_hyphen_lookalikes_but_not_longer_numbers(value, text, found):
    assert value_found(RequiredValue(kind="text", value=value, label="x"), [text]) is found


# seed DB 복사본에서 확인된 고객으로 도구 하나를 실행하는 도우미 (enforce=False면 P0).
def run_tool(engine, customer_id, name, args, *, enforce=True):
    ctx = ToolContext(now=NOW, enforce_policy=enforce)
    ctx.state.verified_customer_id = customer_id
    return execute(REGISTRY, engine, ctx, name, args)


# find_customer는 사람들이 쓰는 여러 표기를 받아야 한다: +82 국가번호, 이름 끝의 '님'·'고객님'·'고객', 전각
# 숫자, 이름 안의 공백, 대소문자가 섞인 이메일과 앞말. 표기만 다른 정답 고객을 '없는 고객'으로 거절하면 본인
# 확인에서 과제가 끝난다 (docs/design.md 본인 확인 규칙).
@pytest.mark.parametrize(
    ("name", "contact"),
    [
        ("김하준", "+82 10-0000-9001"),
        ("김하준님", "010-0000-9001"),
        ("김하준 고객님", "０１０－００００－９００１"),
        ("김하준 고객", "010-0000-9001"),
        ("김 하준", "이메일: Hajun.Kim@example.com"),
    ],
)
def test_find_customer_accepts_the_ways_people_write_names_and_contacts(name, contact):
    engine = db.memory_engine(build_seed_engine())
    result = execute(
        REGISTRY, engine, ToolContext(now=NOW), "find_customer", {"name": name, "contact": contact}
    )
    assert result.ok and json.loads(result.content)["customer_id"] == "C-9001"


# 모델이 보낸 이상한 인자(짝 없는 서로게이트, NUL 문자, 아주 큰 정수, 0번 줄)는 도구가 예외로 죽지 않고
# invalid_arguments 오류로 돌려줘야 한다. 예외가 나면 도구 버그(infra_error)로 세어져 에이전트의 실수가
# 분모에서 빠진다.
@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("get_order", {"order_id": "abc\ud83d"}),
        ("get_order", {"order_id": "O-9\x000001"}),
        (
            "request_exchange",
            {"order_id": "O-90004", "line_no": 10**30, "new_variant_id": "V", "reason": "defective"},
        ),
        ("request_return", {"order_id": "O-90004", "line_nos": [0], "reason": "defective"}),
    ],
)
def test_hostile_arguments_are_argument_errors_not_crashes(tool, args):
    result = run_tool(db.memory_engine(build_seed_engine()), "C-9004", tool, args)
    assert result.error_code == "invalid_arguments"


# 취소된 주문에는 사유와 상관없이 보상 쿠폰을 발급하지 않는다(coupon_order_cancelled). P1에서는 막히고,
# P0에서는 통과하되 위반으로 기록돼야 한다.
def test_a_cancelled_order_gets_no_coupon_whatever_the_reason():
    for reason in ("delivery_delay", "defective_item"):
        engine = db.memory_engine(build_seed_engine())
        args = {"order_id": "O-90002", "reason": reason}
        blocked = run_tool(engine, "C-9002", "issue_compensation_coupon", args)
        assert (blocked.error_code, blocked.policy_blocked) == ("coupon_order_cancelled", True)
        passed = run_tool(engine, "C-9002", "issue_compensation_coupon", args, enforce=False)
        assert passed.ok and "coupon_order_cancelled" in passed.violations


# 이미 발급된 보상 쿠폰이 get_order와 get_customer 결과에 보여야 한다. 에이전트가 '주문당 1장', '30일에 2장'
# 한도를 스스로 셀 수 있게 하려는 것이다 (docs/design.md 규정 검사 표).
def test_existing_compensation_coupons_are_visible_to_the_agent():
    engine = db.memory_engine(build_seed_engine())
    issued = run_tool(
        engine,
        "C-9002",
        "issue_compensation_coupon",
        {"order_id": "O-90002", "reason": "delivery_delay"},
        enforce=False,
    )
    coupon_id = json.loads(issued.content)["coupon_id"]
    order = json.loads(run_tool(engine, "C-9002", "get_order", {"order_id": "O-90002"}).content)
    customer = json.loads(run_tool(engine, "C-9002", "get_customer", {"customer_id": "C-9002"}).content)
    assert [c["coupon_id"] for c in order["compensation_coupons"]] == [coupon_id]
    assert [c["coupon_id"] for c in customer["compensation_coupons"]] == [coupon_id]
