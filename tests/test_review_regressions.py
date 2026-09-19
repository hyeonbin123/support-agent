"""Defects found by review on 2026-09-19, each pinned by a test."""

from __future__ import annotations

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


@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda task: task.id)
def test_no_required_value_is_given_away_by_the_agent_prompt(task):
    prompt = build_system_prompt(load_policy(), task.now)
    assert validate_task(task, build_seed_engine(), REGISTRY, prompt_texts=[prompt]) == []


def test_validate_task_reports_a_value_that_the_prompt_contains():
    task = next(t for t in ALL_TASKS if t.id == "smoke-refusal-01")
    problems = validate_task(task, build_seed_engine(), REGISTRY, prompt_texts=["예: 9월 2일 수령"])
    assert any("agent prompt" in p for p in problems)


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


@pytest.mark.parametrize(
    "text", ["환불 금액은 32,400원입니다.", "주문 {O-1}을 확인했습니다.", '{"note": "json이지만 호출 아님"}']
)
def test_ordinary_text_is_not_a_format_error(text):
    assert format_problem(ChatResponse(text=text)) is None


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


def test_an_unexpected_exception_is_an_infra_error_with_a_traceback():
    class Broken(ScriptedProvider):
        def chat(self, *args, **kwargs):
            raise AttributeError("'dict' object has no attribute 'strip'")

    result = episode(ALL_TASKS[0], [])
    assert result is not None  # a script that only runs out is still an infra error, not a crash
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


def test_a_handoff_matches_whether_or_not_the_customer_was_verified_first():
    assert ("handoffs", "customer_id") in db.IGNORED_COLUMNS


@pytest.mark.parametrize(
    ("value", "text", "found"),
    [
        ("5550-1207-9001", "송장 번호는 5550‑1207‑9001입니다.", True),  # non-breaking hyphens
        ("5550-1207-9001", "송장 번호는 5550–1207–9001입니다.", True),  # en dashes
        ("5550-1207-9001", "15550-1207-90011", False),
        ("한빛택배", "한빛​택배로 배송 중입니다.", True),
    ],
)
def test_text_values_tolerate_hyphen_lookalikes_but_not_longer_numbers(value, text, found):
    assert value_found(RequiredValue(kind="text", value=value, label="x"), [text]) is found


def run_tool(engine, customer_id, name, args, *, enforce=True):
    ctx = ToolContext(now=NOW, enforce_policy=enforce)
    ctx.state.verified_customer_id = customer_id
    return execute(REGISTRY, engine, ctx, name, args)


@pytest.mark.parametrize(
    ("name", "contact"),
    [
        ("김하준", "+82 10-0000-9001"),
        ("김하준님", "010-0000-9001"),
        ("김하준 고객님", "０１０－００００－９００１"),
        ("김 하준", "이메일: Hajun.Kim@example.com"),
    ],
)
def test_find_customer_accepts_the_ways_people_write_names_and_contacts(name, contact):
    engine = db.memory_engine(build_seed_engine())
    result = execute(
        REGISTRY, engine, ToolContext(now=NOW), "find_customer", {"name": name, "contact": contact}
    )
    assert result.ok and json.loads(result.content)["customer_id"] == "C-9001"


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


def test_a_cancelled_order_gets_no_coupon_whatever_the_reason():
    for reason in ("delivery_delay", "defective_item"):
        engine = db.memory_engine(build_seed_engine())
        args = {"order_id": "O-90002", "reason": reason}
        blocked = run_tool(engine, "C-9002", "issue_compensation_coupon", args)
        assert (blocked.error_code, blocked.policy_blocked) == ("coupon_order_cancelled", True)
        passed = run_tool(engine, "C-9002", "issue_compensation_coupon", args, enforce=False)
        assert passed.ok and "coupon_order_cancelled" in passed.violations


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
