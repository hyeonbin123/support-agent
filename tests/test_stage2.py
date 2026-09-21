"""Stage 2 candidates: the stall guard (G1) and the rescue of leaked tool calls (F1)."""

from __future__ import annotations

import pytest
from conftest import NOW
from toy_tools import TOY_REGISTRY

from support_agent import analyze, db
from support_agent.agent import (
    LANGUAGE_NOTICE,
    STALL_NOTICE,
    agent_turn,
    in_another_language,
    is_stall,
    leaked_tool_call,
    new_state,
)
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


def test_compare_pairs_pass_k_over_the_tasks_that_have_k_trials_in_both_runs(tmp_path):
    import json

    def write(name, trials_by_task):
        run_dir = tmp_path / name
        run_dir.mkdir()
        lines = [
            json.dumps(
                {
                    "task_id": task,
                    "trial": i,
                    "status": "completed",
                    "termination": "user_stop",
                    "verdict": {"db_match": ok, "values": {}},
                }
            )
            for task, oks in trials_by_task.items()
            for i, ok in enumerate(oks)
        ]
        (run_dir / "episodes.jsonl").write_text(chr(10).join(lines), encoding="utf-8")
        return run_dir

    base = write("base", {"a": [True] * 4, "b": [False] * 4})
    other = write("other", {"a": [True] * 4, "b": [True, True, True]})  # one trial of b was an infra error
    table = analyze.compare(base, [other])
    assert "| other | 100.0% | +50.0%p" in table  # pass^1 uses both tasks
    assert "100.0% (1과제)" in table  # pass^4 only the task with four judged trials in both


# -------------------------------------------------------------------- the language guard (service only)

CHINESE = "顾客ID为C-9003，您的订单O-90003的状态是已发货。"


def test_a_reply_in_chinese_is_held_back_and_asked_again_in_korean(tiny_engine):
    result, state, _ = turn(tiny_engine, [CHINESE, "주문 O-1은 결제 완료 상태입니다."], language="L1")
    assert result.reply == "주문 O-1은 결제 완료 상태입니다." and state.format_errors == 1
    held, notice = state.messages[3], state.messages[4]
    assert (held.content, held.delivered, notice.harness, notice.content) == (
        CHINESE,
        False,
        True,
        LANGUAGE_NOTICE,
    )
    assert [log.format_error for log in state.llm_log] == ["wrong_language", None]


def test_when_it_keeps_answering_in_chinese_nothing_is_delivered(tiny_engine):
    result, state, _ = turn(tiny_engine, [CHINESE] * 3, language="L1")
    assert (result.reply, result.stop) == (None, "agent_format_error")
    assert not [m for m in state.messages if m.role == "assistant" and m.delivered and m.content == CHINESE]


def test_the_evaluation_default_delivers_the_reply_as_it_is(tiny_engine):
    result, state, _ = turn(tiny_engine, [CHINESE])
    assert result.reply == CHINESE and state.format_errors == 0 and RunConfig().language == "L0"


def test_one_stray_character_is_not_another_language():
    assert not in_another_language("김하준(金) 고객님, 주문 O-1은 결제 완료 상태입니다.")
    assert in_another_language("ご注文は発送済みです") and in_another_language(CHINESE)
