"""Stage 2 candidates: the stall guard (G1) and the rescue of leaked tool calls (F1)."""

from __future__ import annotations

import pytest
from conftest import NOW
from toy_tools import TOY_REGISTRY

from support_agent import analyze, db
from support_agent.agent import STALL_NOTICE, agent_turn, is_stall, leaked_tool_call, new_state
from support_agent.chat import ScriptedProvider, ToolCall
from support_agent.config import RunConfig
from support_agent.toolkit import ToolContext, execute


def turn(tiny_engine, script, **config):
    engine = db.memory_engine(tiny_engine)
    ctx = ToolContext(now=NOW)
    state = new_state("system prompt")
    provider = ScriptedProvider(script)
    result = agent_turn(
        state,
        "O-1 주문을 취소해 주세요.",
        provider=provider,
        registry=TOY_REGISTRY,
        run_tool=lambda name, args: execute(TOY_REGISTRY, engine, ctx, name, args),
        config=RunConfig(**config),
    )
    turn.provider = provider
    return result, state, engine


@pytest.mark.parametrize(
    "text",
    [
        "확인해 보겠습니다. 잠시만 기다려 주세요.",
        "네, 주문 내역을 조회해 보겠습니다.",
        "취소 처리해 드리겠습니다.",
        "상담원에게 연결하겠습니다.",
    ],
)
def test_a_reply_that_only_promises_is_a_stall(text):
    assert is_stall(text)


@pytest.mark.parametrize(
    "text",
    [
        "확인해 보겠습니다. 성함과 가입하신 전화번호를 알려 주세요.",
        "O-1 주문을 단순 변심으로 취소할까요?",
        "취소되었습니다. 38,900원이 환불됩니다.",
        "수령 후 7일이 지나 반품이 어렵습니다.",
    ],
)
def test_asking_or_reporting_is_not_a_stall(text):
    assert not is_stall(text)


def test_g1_holds_a_stall_back_and_the_retry_acts(tiny_engine):
    script = [
        "확인해 보겠습니다. 잠시만 기다려 주세요.",
        ToolCall("toy_cancel_order", {"order_id": "O-1"}),
        "취소했습니다.",
    ]
    result, state, engine = turn(tiny_engine, script, guard="G1")
    assert result.reply == "취소했습니다." and state.stalls == 1 and state.format_errors == 0
    held, notice = state.messages[3], state.messages[4]
    assert (held.delivered, notice.harness, notice.content) == (False, True, STALL_NOTICE)
    assert [log.format_error for log in state.llm_log] == ["stall", None, None]
    assert db.dump_db(engine)["orders"][0]["status"] == "cancelled"


def test_g0_delivers_the_same_reply(tiny_engine):
    result, state, _ = turn(tiny_engine, ["확인해 보겠습니다. 잠시만 기다려 주세요."])
    assert result.reply.startswith("확인해 보겠습니다") and state.stalls == 0


def test_g1_gives_up_after_its_retries_and_delivers_the_reply(tiny_engine):
    result, state, _ = turn(tiny_engine, ["잠시만 기다려 주세요."] * 3, guard="G1")
    assert result.reply == "잠시만 기다려 주세요." and result.stop is None and state.stalls == 2


def test_leaked_tool_call_parsing():
    call = leaked_tool_call(
        'portun\n{"name": "toy_get_order", "arguments": {"order_id": "O-1"}}\n</tool_call>'
    )
    assert call == ToolCall("toy_get_order", {"order_id": "O-1"})
    assert leaked_tool_call('{"name": "toy_get_order"}') is None
    assert leaked_tool_call("주문 {O-1}을 확인했습니다.") is None


def test_f1_runs_the_leaked_call_and_still_counts_the_format_error(tiny_engine):
    leaked = 'portun\n{"name": "toy_cancel_order", "arguments": {"order_id": "O-1"}}\n</tool_call>'
    result, state, engine = turn(tiny_engine, [leaked, "취소했습니다."], rescue="F1")
    assert result.reply == "취소했습니다." and state.format_errors == 1
    assert state.llm_log[0].format_error == "rescued_tool_call"
    assert [log.name for log in state.tool_log] == ["toy_cancel_order"]
    assert db.dump_db(engine)["orders"][0]["status"] == "cancelled"
    assert all(m.delivered for m in state.messages if m.role == "assistant")  # nothing leaked to the customer


def test_f0_sends_the_leaked_call_back(tiny_engine):
    leaked = '{"name": "toy_cancel_order", "arguments": {"order_id": "O-1"}}'
    _, state, engine = turn(tiny_engine, [leaked, "취소하려면 도구가 필요합니다."])
    assert state.llm_log[0].format_error == "leaked_tool_call" and state.tool_log == []
    assert db.dump_db(engine)["orders"][0]["status"] == "paid"


def test_paired_difference_resamples_tasks_and_keeps_pairs():
    base = {"a": [True, False, False, False], "b": [False] * 4, "c": [True] * 4}
    better = {"a": [True, True, True, False], "b": [True, False, False, False], "c": [True] * 4}
    diff, low, high = analyze.paired_difference(base, better, 1)
    assert diff == pytest.approx((0.5 + 0.25 + 0.0) / 3) and low <= diff <= high and low >= 0.0
    same = analyze.paired_difference(base, base, 1)
    assert same == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="same tasks"):
        analyze.paired_difference(base, {"a": [True] * 4}, 1)


def test_g2_samples_the_retry_but_not_the_first_attempt(tiny_engine):
    script = ["확인해 보겠습니다.", ToolCall("toy_cancel_order", {"order_id": "O-1"}), "취소했습니다."]
    _, state, _ = turn(tiny_engine, script, guard="G2")
    temperatures = [request["temperature"] for request in turn.provider.requests]
    assert temperatures == [0.0, 0.7, 0.7] and state.stalls == 1
    _, _, _ = turn(tiny_engine, script, guard="G1")
    assert [request["temperature"] for request in turn.provider.requests] == [0.0, 0.0, 0.0]
