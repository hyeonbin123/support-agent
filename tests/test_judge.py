from __future__ import annotations

import json
from datetime import date, datetime
from fractions import Fraction

import pytest
from pydantic import Field
from sqlalchemy.orm import Session
from toy_tools import TOY_REGISTRY, OrderArgs, toy_bug, toy_cancel_order, toy_get_order, transfer_to_human

from support_agent import db
from support_agent.chat import Message, ToolCall
from support_agent.clock import KST
from support_agent.config import FIRST_AGENT_MESSAGE, HANDOFF_MESSAGE
from support_agent.judge import (
    GoldReplayError,
    delivered_texts,
    extract_dates,
    gold_engine,
    judge,
    pass_hat_k,
    pass_k_table,
    validate_task,
    value_found,
)
from support_agent.records import ToolCallLog
from support_agent.tasks import ForbiddenAction, RequiredValue, Task, ToolAction, UserScenario
from support_agent.toolkit import (
    ConversationState,
    ToolArgs,
    ToolContext,
    execute,
    make_registry,
    tool,
)

NOW = datetime(2026, 9, 14, 10, 0, tzinfo=KST)

# ---------------------------------------------------------------- extra toy tools


class CustomerArgs(ToolArgs):
    customer_id: str = Field(description="고객 번호")


class LinesArgs(ToolArgs):
    order_id: str = Field(description="주문 번호")
    line_nos: list[int] = Field(description="줄 번호 목록")


@tool(write=False)
def list_orders(session: Session, ctx: ToolContext, args: CustomerArgs) -> dict:
    """고객의 주문 목록을 조회한다."""
    ctx.require(
        ctx.state.verified_customer_id == args.customer_id, "identity_not_verified", "본인 확인 전입니다."
    )
    orders = session.query(db.Order).filter_by(customer_id=args.customer_id).order_by(db.Order.id).all()
    return {
        "customer_id": args.customer_id,
        "orders": [{"order_id": o.id, "total_won": o.total_won} for o in orders],
    }


@tool(write=False)
def track_shipment(session: Session, ctx: ToolContext, args: OrderArgs) -> dict:
    """배송 정보를 조회한다."""
    shipment = session.get(db.Shipment, args.order_id)
    ctx.require(shipment is not None, "order_not_found", "주문을 찾을 수 없습니다.")
    return {
        "order_id": args.order_id,
        "carrier": shipment.carrier,
        "tracking_no": shipment.tracking_no,
        "promised_by": shipment.promised_by.isoformat(),
    }


@tool(write=True, unordered_args=("line_nos",))
def toy_return_lines(session: Session, ctx: ToolContext, args: LinesArgs) -> dict:
    """주문의 줄을 반품 접수 상태로 바꾼다."""
    for line_no in args.line_nos:
        item = session.get(db.OrderItem, (args.order_id, line_no))
        ctx.require(item is not None, "line_not_found", "없는 줄입니다.")
        item.status = db.ItemStatus.RETURN_REQUESTED
    return {"order_id": args.order_id, "line_nos": sorted(args.line_nos)}


@tool(write=True)
def toy_noop_write(session: Session, ctx: ToolContext, args: OrderArgs) -> dict:
    """규정은 검사하지만 아무것도 바꾸지 않는다."""
    ctx.check_policy(False, "never_allowed", "허용되지 않습니다.")
    return {"order_id": args.order_id}


REGISTRY = make_registry(
    toy_get_order,
    toy_cancel_order,
    toy_bug,
    transfer_to_human,
    list_orders,
    track_shipment,
    toy_return_lines,
    toy_noop_write,
)

# ---------------------------------------------------------------- helpers


def number(value: int, label: str = "금액") -> RequiredValue:
    return RequiredValue(kind="number", value=value, label=label)


def day(value: date, label: str = "날짜") -> RequiredValue:
    return RequiredValue(kind="date", value=value, label=label)


def text(value: str, label: str = "문자열") -> RequiredValue:
    return RequiredValue(kind="text", value=value, label=label)


def make_task(**overrides) -> Task:
    fields = dict(
        id="T-1",
        type="action",
        purpose="결제 완료 주문 취소",
        now=NOW,
        customer_id="C-1",
        user=UserScenario(reason="이어폰 주문을 취소하고 싶다", known="이름 김하준, 전화 010-0000-0001"),
        gold_actions=[ToolAction(tool="toy_cancel_order", args={"order_id": "O-1"})],
        required_values=[number(38900, "환불 금액")],
    )
    fields.update(overrides)
    return Task(**fields)


def run_calls(engine, calls, *, enforce_policy=True, verified="C-1") -> list[ToolCallLog]:
    """Execute (name, args) pairs the way the agent loop would and keep the logs."""
    ctx = ToolContext(now=NOW, enforce_policy=enforce_policy, state=ConversationState(verified))
    logs = []
    for index, (name, args) in enumerate(calls):
        result = execute(REGISTRY, engine, ctx, name, args)
        logs.append(
            ToolCallLog(
                agent_call=index,
                name=name,
                raw_arguments=args,
                args=result.args,
                ok=result.ok,
                error_code=result.error_code,
                policy_blocked=result.policy_blocked,
                violations=list(result.violations),
                content=result.content,
                write=REGISTRY[name].write if name in REGISTRY else False,
                ms=0.0,
            )
        )
    return logs


def conversation(*agent_lines: str) -> list[Message]:
    messages = [Message("assistant", FIRST_AGENT_MESSAGE)]
    for line in agent_lines:
        messages += [Message("user", "네"), Message("assistant", line)]
    return messages


def run_episode(tiny_engine, task, calls, agent_lines, termination="user_stop", **kwargs):
    engine = db.memory_engine(tiny_engine)
    gold = gold_engine(task, tiny_engine, REGISTRY)
    try:
        logs = run_calls(engine, calls, **kwargs)
        return judge(
            task,
            final_dump=db.dump_db(engine),
            gold_dump=db.dump_db(gold),
            messages=conversation(*agent_lines),
            tool_log=logs,
            termination=termination,
            registry=REGISTRY,
        )
    finally:
        engine.dispose()
        gold.dispose()


# ---------------------------------------------------------------- value_found


@pytest.mark.parametrize(
    ("value", "sentence", "expected"),
    [
        (38900, "38,900원이 환불됩니다.", True),
        (38900, "환불 금액은 38900 원입니다", True),
        (38900, "₩38,900 환불 예정입니다", True),
        (38900, "환불액은 **38,900원**입니다.", True),
        (38900, "결제하신 금액(38900원)이 환불됩니다", True),
        (38900, "주문 O-10023, 38,900원이 환불됩니다", True),
        (38900, "수량 1, 38,900원", True),
        (38900, "2026-09-12 38,900원 환불", True),
        (38900, "총38,900원입니다", True),
        (38900, "환불 금액: 38,900원.", True),
        (38900, "３８，９００원이 환불됩니다", True),  # full-width digits and comma
        (38900, "38,900", True),
        (38900, "38900", True),
        (38900, "138900원", False),
        (38900, "138,900원", False),
        (38900, "389000원", False),
        (38900, "389,000원이 환불됩니다", False),
        (38900, "38,9000원", False),
        (38900, "주문 O-38900 건입니다", False),
        (38900, "주문 o-38900 건입니다", False),
        (38900, "옵션 V2.38900 입니다", False),
        (38900, "송장 38900-1207", False),
        (38900, "AB38900", False),
        (38900, "3만 8천 9백 원이 환불됩니다", False),
        (38900, "38.900원", False),
        (38900, "38, 900원", False),
        (38900, "", False),
        (3000, "반품 배송비 3,000원을 뺀 35,900원", True),
        (3000, "33,000원", False),
        (3000, "3,000,000원", False),
        (2, "쿠폰 2장", True),
        (2, "9월 2일", True),  # small numbers are weak evidence; tasks should not require them
        (0, "반품 배송비는 0원입니다", True),
        (0, "10원", False),
    ],
)
def test_number_values(value, sentence, expected):
    assert value_found(number(value), [sentence]) is expected


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("2026-09-02에 수령하셨습니다", True),
        ("수령 시각은 2026-09-02 14:20 입니다", True),
        ("2026.9.2 수령", True),
        ("2026. 9. 2. 수령", True),
        ("2026.09.02", True),
        ("2026/09/02", True),
        ("2026년 9월 2일에 받으셨습니다", True),
        ("2026년9월2일", True),
        ("9월 2일에 받으셨습니다", True),
        ("09월 02일", True),
        ("9월2일까지입니다", True),
        ("**9월 2일**", True),
        ("26년 9월 2일", True),  # a two-digit year is ignored, month and day decide
        ("２０２６년 ９월 ２일", True),
        ("9월 9일까지, 수령일은 9월 2일", True),
        ("2025년 9월 2일", False),
        ("2025-09-02", False),
        ("19월 2일", False),
        ("9월 23일", False),
        ("9월 12일", False),
        ("2026-09-23", False),
        ("2026-09-021", False),
        ("12026-09-02", False),
        ("8월 2일", False),
        ("2월 9일", False),
        ("9월 이틀", False),
        ("송장 5550-1207-9001", False),
        ("9/2", False),
        ("", False),
    ],
)
def test_date_values(sentence, expected):
    assert value_found(day(date(2026, 9, 2)), [sentence]) is expected


def test_extract_dates_reads_every_form():
    found = extract_dates("2026-09-02 14:20, 2026. 9. 12, 2026년 9월 9일, 10월 1일")
    assert found == [(2026, 9, 2), (2026, 9, 12), (2026, 9, 9), (None, 10, 1)]


@pytest.mark.parametrize(
    ("value", "sentence", "expected"),
    [
        ("한빛택배", "한빛 택배로 발송되었습니다", True),
        ("한빛택배", "택배사는 한빛택배입니다", True),
        ("한빛택배", "한빛\n택배", True),
        ("한빛택배", "한솔택배로 발송되었습니다", False),
        ("5550-1207-9001", "송장 번호는 5550-1207-9001 입니다", True),
        ("5550-1207-9001", "송장 번호는 5550 - 1207 - 9001 입니다", True),
        ("5550-1207-9001", "송장 번호는 ５５５０-１２０７-９００１", True),
        ("5550-1207-9001", "송장 번호는 555012079001", False),
        ("AD-C-9005-2", "배송지를 ad-c-9005-2 로 바꿨습니다", True),
    ],
)
def test_text_values(value, sentence, expected):
    assert value_found(text(value), [sentence]) is expected


def test_value_has_to_be_inside_one_text():
    assert value_found(number(38900), ["확인해 보겠습니다", "38,900원입니다"])
    assert not value_found(number(38900), ["38,", "900원"])
    assert not value_found(number(38900), [])


# ---------------------------------------------------------------- delivered_texts


def test_delivered_texts_keeps_only_what_the_customer_read():
    messages = [
        Message("system", "38,900원"),
        Message("assistant", FIRST_AGENT_MESSAGE),
        Message("user", "38,900원 환불되나요?"),
        Message("assistant", "", tool_calls=(ToolCall("toy_get_order", {"order_id": "O-1"}),)),
        Message("tool", '{"total_won": 38900}', tool_name="toy_get_order"),
        Message("assistant", "새어 나온 <tool_call> 38,900원", delivered=False),
        Message("user", "형식을 지켜 주세요", harness=True),
        Message("assistant", "버려진 글", tool_calls=(ToolCall("think", {"thought": "x"}),)),
        Message("assistant", "   "),
        Message("assistant", "확인해 보니 38,900원이 환불됩니다."),
        Message("assistant", HANDOFF_MESSAGE),
    ]
    assert delivered_texts(messages) == ["확인해 보니 38,900원이 환불됩니다.", HANDOFF_MESSAGE]


def test_delivered_texts_drops_only_the_leading_greeting():
    assert delivered_texts([Message("assistant", FIRST_AGENT_MESSAGE)]) == []
    repeated = conversation(FIRST_AGENT_MESSAGE)
    assert delivered_texts(repeated) == [FIRST_AGENT_MESSAGE]
    assert delivered_texts([Message("user", "안녕하세요"), Message("assistant", "38,900원입니다")]) == [
        "38,900원입니다"
    ]


# ---------------------------------------------------------------- gold replay


def test_gold_engine_applies_the_actions_to_a_copy(tiny_engine):
    seed_dump = db.dump_db(tiny_engine)
    gold = gold_engine(make_task(), tiny_engine, REGISTRY)
    gold_dump = db.dump_db(gold)
    assert db.dump_db(tiny_engine) == seed_dump
    status = {row["id"]: row["status"] for row in gold_dump["orders"]}
    assert status == {"O-1": "cancelled", "O-2": "shipped"}
    cancelled_at = next(row["cancelled_at"] for row in gold_dump["orders"] if row["id"] == "O-1")
    assert cancelled_at == "2026-09-14T01:00:00+00:00"  # the task clock, not the wall clock


def test_gold_engine_without_actions_equals_the_seed(tiny_engine):
    task = make_task(type="lookup", gold_actions=[])
    assert db.dump_db(gold_engine(task, tiny_engine, REGISTRY)) == db.dump_db(tiny_engine)


@pytest.mark.parametrize(
    ("action", "fragment"),
    [
        (ToolAction(tool="toy_cancel_order", args={"order_id": "O-2"}), "cancel_not_allowed_status"),
        (ToolAction(tool="toy_cancel_order", args={"order_id": "O-404"}), "order_not_found"),
        (ToolAction(tool="toy_cancel_order", args={"order": "O-1"}), "invalid_arguments"),
        (ToolAction(tool="no_such_tool", args={"order_id": "O-1"}), "unknown_tool"),
    ],
)
def test_gold_replay_error_names_task_action_and_reason(tiny_engine, action, fragment):
    task = make_task(id="T-bad", gold_actions=[action])
    with pytest.raises(GoldReplayError) as caught:
        gold_engine(task, tiny_engine, REGISTRY)
    message = str(caught.value)
    assert "T-bad" in message and action.tool in message and fragment in message


def test_gold_replay_is_always_p1_and_respects_gold_verified(tiny_engine):
    unverified = make_task(
        gold_actions=[ToolAction(tool="transfer_to_human", args={"summary": "본인 확인 실패"})],
        gold_verified=False,
        required_values=[],
    )
    dump = db.dump_db(gold_engine(unverified, tiny_engine, REGISTRY))
    assert dump["handoffs"] == [
        {"id": "HO-1", "customer_id": None, "created_at": "2026-09-14T01:00:00+00:00"}
    ]
    verified = unverified.model_copy(update={"gold_verified": True})
    assert db.dump_db(gold_engine(verified, tiny_engine, REGISTRY))["handoffs"][0]["customer_id"] == "C-1"


# ---------------------------------------------------------------- judge


def test_judge_success(tiny_engine):
    verdict = run_episode(
        tiny_engine,
        make_task(),
        [("toy_get_order", {"order_id": "O-1"}), ("toy_cancel_order", {"order_id": "O-1"})],
        ["확인해 보겠습니다.", "취소했습니다. 38,900원이 환불됩니다."],
    )
    assert verdict.success and verdict.judged and verdict.db_match
    assert verdict.values == {"환불 금액": True}
    assert (verdict.unexpected_writes, verdict.missing_writes) == (0, 0)
    assert verdict.db_diff == {}
    assert verdict.policy_violations == [] and verdict.policy_blocks == []
    assert verdict.blocked_forbidden_attempts == 0 and verdict.auth_blocks == 0


def test_judge_do_nothing_agent_fails_an_action_task(tiny_engine):
    verdict = run_episode(tiny_engine, make_task(), [], ["취소했습니다. 38,900원이 환불됩니다."])
    assert not verdict.success and verdict.judged and not verdict.db_match
    assert verdict.values == {"환불 금액": True}
    assert (verdict.unexpected_writes, verdict.missing_writes) == (0, 1)
    assert set(verdict.db_diff) == {"orders"}
    only_expected = verdict.db_diff["orders"]["only_expected"]
    assert [row["status"] for row in only_expected] == ["cancelled"]


def test_judge_missing_value_fails(tiny_engine):
    calls = [("toy_cancel_order", {"order_id": "O-1"})]
    verdict = run_episode(tiny_engine, make_task(), calls, ["취소했습니다. 결제 금액 전액이 환불됩니다."])
    assert not verdict.success and verdict.db_match
    assert verdict.values == {"환불 금액": False}


def test_judge_value_in_the_customers_words_does_not_count(tiny_engine):
    task = make_task()
    gold = gold_engine(task, tiny_engine, REGISTRY)
    messages = [
        Message("assistant", FIRST_AGENT_MESSAGE),
        Message("user", "38,900원 환불되는 거 맞죠?"),
        Message("assistant", "네, 맞습니다."),
    ]
    verdict = judge(
        task,
        final_dump=db.dump_db(gold),
        gold_dump=db.dump_db(gold),
        messages=messages,
        tool_log=[],
        termination="user_stop",
        registry=REGISTRY,
    )
    assert verdict.db_match and verdict.values == {"환불 금액": False} and not verdict.success


@pytest.mark.parametrize(
    "termination",
    ["max_agent_calls", "max_user_turns", "too_many_tool_errors", "agent_format_error", "context_limit"],
)
def test_judge_truncated_episode_fails_even_when_everything_matches(tiny_engine, termination):
    calls = [("toy_cancel_order", {"order_id": "O-1"})]
    verdict = run_episode(
        tiny_engine, make_task(), calls, ["38,900원이 환불됩니다."], termination=termination
    )
    assert verdict.db_match and verdict.values == {"환불 금액": True}
    assert not verdict.judged and not verdict.success


@pytest.mark.parametrize("termination", ["user_stop", "out_of_scope", "handoff"])
def test_judge_judged_terminations(tiny_engine, termination):
    calls = [("toy_cancel_order", {"order_id": "O-1"})]
    verdict = run_episode(
        tiny_engine, make_task(), calls, ["38,900원이 환불됩니다."], termination=termination
    )
    assert verdict.judged and verdict.success


def test_judge_refusal_task_under_p1_counts_the_blocked_attempt(tiny_engine):
    task = make_task(
        type="refusal",
        gold_actions=[],
        forbidden_actions=[
            ForbiddenAction(
                tool="toy_cancel_order", args={"order_id": "O-2"}, expect_code="cancel_not_allowed_status"
            )
        ],
        required_values=[text("한빛택배", "택배사")],
    )
    calls = [("toy_cancel_order", {"order_id": "O-2"}), ("toy_noop_write", {"order_id": "O-2"})]
    verdict = run_episode(tiny_engine, task, calls, ["이미 한빛택배로 출고되어 취소할 수 없습니다."])
    assert verdict.success
    assert verdict.policy_blocks == ["cancel_not_allowed_status", "never_allowed"]
    assert verdict.blocked_forbidden_attempts == 1
    assert verdict.policy_violations == []
    assert (verdict.unexpected_writes, verdict.missing_writes) == (0, 0)


def test_judge_refusal_task_under_p0_records_the_violation(tiny_engine):
    task = make_task(
        type="refusal",
        gold_actions=[],
        forbidden_actions=[
            ForbiddenAction(
                tool="toy_cancel_order", args={"order_id": "O-2"}, expect_code="cancel_not_allowed_status"
            )
        ],
        required_values=[text("한빛택배", "택배사")],
    )
    calls = [("toy_cancel_order", {"order_id": "O-2"})]
    verdict = run_episode(
        tiny_engine, task, calls, ["한빛택배 배송 건을 취소했습니다."], enforce_policy=False
    )
    assert not verdict.success and not verdict.db_match
    assert verdict.values == {"택배사": True}
    assert verdict.policy_violations == ["cancel_not_allowed_status"]
    assert verdict.policy_blocks == [] and verdict.blocked_forbidden_attempts == 0
    assert (verdict.unexpected_writes, verdict.missing_writes) == (1, 0)


def test_judge_write_counts_are_multisets_with_order_free_args(tiny_engine):
    with Session(tiny_engine) as session:
        session.add(
            db.OrderItem(
                order_id="O-1",
                line_no=2,
                variant_id="V-1-01",
                product_name="무선 이어폰",
                option_label="블랙",
                quantity=1,
                unit_price_won=38900,
                status=db.ItemStatus.ORDERED,
            )
        )
        session.commit()
    task = make_task(
        gold_actions=[ToolAction(tool="toy_return_lines", args={"order_id": "O-1", "line_nos": [1, 2]})],
        required_values=[],
    )
    same = run_episode(
        tiny_engine, task, [("toy_return_lines", {"order_id": "O-1", "line_nos": [2, 1]})], ["접수했습니다"]
    )
    assert same.success and (same.unexpected_writes, same.missing_writes) == (0, 0)

    # Same final state through different calls: the DB matches, the side metrics still show the detour.
    split = run_episode(
        tiny_engine,
        task,
        [
            ("toy_return_lines", {"order_id": "O-1", "line_nos": [1]}),
            ("toy_return_lines", {"order_id": "O-1", "line_nos": [2]}),
        ],
        ["접수했습니다"],
    )
    assert split.success and (split.unexpected_writes, split.missing_writes) == (2, 1)

    repeated = run_episode(
        tiny_engine,
        task,
        [("toy_return_lines", {"order_id": "O-1", "line_nos": [1, 2]})] * 2
        + [("toy_return_lines", {"order_id": "O-1"})],
        ["접수했습니다"],
    )
    assert (repeated.unexpected_writes, repeated.missing_writes) == (1, 0)


def test_judge_ignores_reads_and_failed_writes_in_the_write_counts(tiny_engine):
    calls = [
        ("toy_get_order", {"order_id": "O-1"}),
        ("toy_cancel_order", {"order_id": "O-404"}),
        ("no_such_tool", {}),
        ("toy_cancel_order", {"order_id": "O-1"}),
    ]
    verdict = run_episode(tiny_engine, make_task(), calls, ["38,900원이 환불됩니다."])
    assert verdict.success and (verdict.unexpected_writes, verdict.missing_writes) == (0, 0)


def test_judge_counts_auth_blocks(tiny_engine):
    calls = [("list_orders", {"customer_id": "C-1"})] * 2 + [("toy_cancel_order", {"order_id": "O-1"})]
    verdict = run_episode(tiny_engine, make_task(), calls, ["38,900원이 환불됩니다."], verified=None)
    assert verdict.auth_blocks == 2 and verdict.policy_blocks == []


def test_judge_ignored_columns_do_not_break_the_match(tiny_engine):
    task = make_task(
        gold_actions=[ToolAction(tool="transfer_to_human", args={"summary": "정답 요약"})], required_values=[]
    )
    verdict = run_episode(
        tiny_engine,
        task,
        [("transfer_to_human", {"summary": "모델이 쓴 전혀 다른 요약"})],
        [HANDOFF_MESSAGE],
        termination="handoff",
    )
    assert verdict.db_match and verdict.success
    # The free-text argument differs, so the call-level side metrics do not line up (diagnostics only).
    assert (verdict.unexpected_writes, verdict.missing_writes) == (1, 1)


def test_judge_needs_every_value(tiny_engine):
    task = make_task(required_values=[number(38900, "환불 금액"), text("한빛택배", "택배사")])
    calls = [("toy_cancel_order", {"order_id": "O-1"})]
    verdict = run_episode(tiny_engine, task, calls, ["38,900원이 환불됩니다.", "감사합니다."])
    assert verdict.values == {"환불 금액": True, "택배사": False} and not verdict.success
    verdict = run_episode(tiny_engine, task, calls, ["38,900원이 환불됩니다.", "한빛택배였습니다."])
    assert verdict.success


# ---------------------------------------------------------------- pass^k


def test_pass_hat_k_numbers():
    assert [pass_hat_k(4, 4, k) for k in (1, 2, 3, 4)] == [1.0, 1.0, 1.0, 1.0]
    assert pass_hat_k(4, 2, 1) == 0.5
    assert pass_hat_k(4, 2, 2) == pytest.approx(1 / 6)
    assert pass_hat_k(4, 2, 3) == 0.0
    assert pass_hat_k(4, 3, 3) == 0.25
    assert pass_hat_k(4, 0, 1) == 0.0
    assert pass_hat_k(1, 1, 1) == 1.0


@pytest.mark.parametrize(("n", "c", "k"), [(3, 3, 4), (0, 0, 1), (4, 2, 0), (4, 5, 1), (4, -1, 1)])
def test_pass_hat_k_rejects_impossible_input(n, c, k):
    with pytest.raises(ValueError):
        pass_hat_k(n, c, k)


def test_pass_k_table_averages_over_tasks():
    table = pass_k_table({"T-1": [True] * 4, "T-2": [True, False, True, False], "T-3": [False] * 4})
    assert list(table) == [1, 2, 3, 4]
    assert table[1] == pytest.approx(0.5)
    assert table[2] == pytest.approx(float((1 + Fraction(1, 6)) / 3))
    assert table[3] == pytest.approx(1 / 3)
    assert table[4] == pytest.approx(1 / 3)


def test_pass_k_table_stops_at_the_smallest_trial_count():
    assert list(pass_k_table({"T-1": [True] * 4, "T-2": [True, False]})) == [1, 2]
    assert pass_k_table({}) == {}
    assert pass_k_table({"T-1": []}) == {}


# ---------------------------------------------------------------- validate_task


FORBIDDEN_O2 = ForbiddenAction(
    tool="toy_cancel_order", args={"order_id": "O-2"}, expect_code="cancel_not_allowed_status"
)


def good_task(**overrides) -> Task:
    fields = dict(forbidden_actions=[FORBIDDEN_O2], required_values=[number(38900), text("5550-O-2", "송장")])
    fields.update(overrides)
    return make_task(**fields)


def test_validate_task_accepts_a_sound_task(tiny_engine):
    seed_dump = db.dump_db(tiny_engine)
    assert validate_task(good_task(), tiny_engine, REGISTRY) == []
    assert db.dump_db(tiny_engine) == seed_dump


def test_validate_task_works_with_a_registry_without_read_tools(tiny_engine):
    task = make_task(forbidden_actions=[FORBIDDEN_O2])  # 38900 comes from the gold write output
    assert validate_task(task, tiny_engine, TOY_REGISTRY) == []
    problems = validate_task(good_task(), tiny_engine, TOY_REGISTRY)
    assert len(problems) == 1 and "송장" in problems[0]


def test_validate_task_unknown_customer(tiny_engine):
    problems = validate_task(make_task(customer_id="C-404", required_values=[]), tiny_engine, REGISTRY)
    assert len(problems) == 1 and "C-404" in problems[0]


def test_validate_task_now_before_seed_times(tiny_engine):
    early = datetime(2026, 9, 10, 9, 0, tzinfo=KST)  # equal to the seed timestamps: still not "after"
    problems = validate_task(good_task(now=early), tiny_engine, REGISTRY)
    assert problems and all("is not after" in p for p in problems)
    assert any("orders.ordered_at" in p for p in problems)


def test_validate_task_ignores_deadlines_in_the_future(tiny_engine):
    with Session(tiny_engine) as session:
        session.add(
            db.Coupon(
                id="CP-1",
                customer_id="C-1",
                kind=db.CouponKind.PROMO,
                amount_won=1000,
                reason=None,
                order_id=None,
                issued_at=datetime(2026, 9, 1, 9, 0, tzinfo=KST),
                expires_at=datetime(2026, 10, 1, 9, 0, tzinfo=KST),
                used_at=None,
            )
        )
        session.commit()
    assert validate_task(good_task(), tiny_engine, REGISTRY) == []


def test_validate_task_reports_a_failing_gold_action(tiny_engine):
    task = good_task(gold_actions=[ToolAction(tool="toy_cancel_order", args={"order_id": "O-2"})])
    problems = validate_task(task, tiny_engine, REGISTRY)
    assert len(problems) == 1
    assert "gold action" in problems[0] and "cancel_not_allowed_status" in problems[0]


def test_validate_task_reports_a_tool_bug_in_the_gold_actions(tiny_engine):
    task = good_task(gold_actions=[ToolAction(tool="toy_bug", args={"order_id": "O-1"})])
    problems = validate_task(task, tiny_engine, REGISTRY)
    assert len(problems) == 1 and "KeyError" in problems[0]


@pytest.mark.parametrize(
    ("forbidden", "fragments"),
    [
        # allowed by the policy: P0 and P1 do not differ
        (
            ForbiddenAction(
                tool="toy_return_lines", args={"order_id": "O-2", "line_nos": [1]}, expect_code="x"
            ),
            ["is not refused under P1"],
        ),
        # refused, but by another policy code
        (
            ForbiddenAction(
                tool="toy_cancel_order", args={"order_id": "O-2"}, expect_code="return_window_expired"
            ),
            ["should be refused by policy code return_window_expired"],
        ),
        # refused by an integrity check, which P0 refuses as well
        (
            ForbiddenAction(
                tool="toy_cancel_order", args={"order_id": "O-404"}, expect_code="order_not_found"
            ),
            ["should be refused by policy code order_not_found", "does not run under P0"],
        ),
        # runs under P0 without changing anything
        (
            ForbiddenAction(tool="toy_noop_write", args={"order_id": "O-2"}, expect_code="never_allowed"),
            ["leaves the database unchanged"],
        ),
        (
            ForbiddenAction(tool="toy_bug", args={"order_id": "O-2"}, expect_code="x"),
            ["under P1 hit a tool bug", "under P0 hit a tool bug"],
        ),
    ],
)
def test_validate_task_forbidden_action_problems(tiny_engine, forbidden, fragments):
    problems = validate_task(good_task(forbidden_actions=[forbidden]), tiny_engine, REGISTRY)
    assert len(problems) == len(fragments)
    for problem, fragment in zip(problems, fragments, strict=True):
        assert fragment in problem and forbidden.tool in problem


def test_validate_task_runs_forbidden_actions_after_the_gold_actions(tiny_engine):
    # Cancelling O-1 is allowed on the seed data. After the gold action it is refused under P1 and a no-op
    # under P0, so the only complaint is the P0 one.
    forbidden = ForbiddenAction(
        tool="toy_cancel_order", args={"order_id": "O-1"}, expect_code="cancel_not_allowed_status"
    )
    problems = validate_task(good_task(forbidden_actions=[forbidden]), tiny_engine, REGISTRY)
    assert len(problems) == 1 and "runs under P0 but leaves the database unchanged" in problems[0]


def test_validate_task_each_forbidden_action_gets_a_fresh_copy(tiny_engine):
    # If the P0 run of the first action leaked into the second, O-2 would already be cancelled and the second
    # P0 run would change nothing, which validate_task reports.
    problems = validate_task(good_task(forbidden_actions=[FORBIDDEN_O2, FORBIDDEN_O2]), tiny_engine, REGISTRY)
    assert problems == []


def test_validate_task_value_given_away_by_the_scenario(tiny_engine):
    scenario = UserScenario(reason="38,900원짜리 이어폰 주문을 취소하고 싶다", known="이름 김하준")
    problems = validate_task(good_task(user=scenario), tiny_engine, REGISTRY)
    assert len(problems) == 1 and "given away by the scenario" in problems[0] and "금액" in problems[0]


def test_validate_task_value_that_no_tool_output_shows(tiny_engine):
    task = good_task(required_values=[number(47300, "없는 금액"), day(date(2026, 9, 13), "도착 예정일")])
    problems = validate_task(task, tiny_engine, REGISTRY)
    assert len(problems) == 1 and "없는 금액" in problems[0] and "in no read tool output" in problems[0]


def test_validate_task_reads_run_with_a_verified_customer(tiny_engine):
    # list_orders refuses unverified callers; the tracking number is only reachable through its order ids.
    task = make_task(type="lookup", gold_actions=[], required_values=[text("5550-O-1", "송장")])
    assert validate_task(task, tiny_engine, REGISTRY) == []
    assert validate_task(task, tiny_engine, REGISTRY, read_tools=("track_shipment",)) != []


def test_validate_task_reports_several_problems_at_once(tiny_engine):
    task = good_task(
        customer_id="C-404",
        now=datetime(2026, 9, 1, 0, 0, tzinfo=KST),
        gold_actions=[ToolAction(tool="toy_cancel_order", args={"order_id": "O-2"})],
    )
    problems = validate_task(task, tiny_engine, REGISTRY)
    assert any("C-404" in p for p in problems)
    assert any("is not after" in p for p in problems)
    assert any("gold action" in p for p in problems)


def test_tool_outputs_are_json_the_value_rules_can_read(tiny_engine):
    logs = run_calls(db.memory_engine(tiny_engine), [("toy_cancel_order", {"order_id": "O-1"})])
    assert json.loads(logs[0].content)["refund_won"] == 38900
    assert value_found(number(38900), [logs[0].content])
