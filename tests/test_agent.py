"""The agent loop against a scripted provider, the toy registry and a real closure over toolkit.execute."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from conftest import NOW
from sqlalchemy.orm import Session
from toy_tools import TOY_REGISTRY

from support_agent import db
from support_agent.agent import (
    CUT_OFF_NOTICE,
    FORMAT_NOTICE,
    AgentState,
    agent_turn,
    build_system_prompt,
    format_problem,
    new_state,
    visible_tools,
)
from support_agent.chat import ChatResponse, ProviderError, ScriptedProvider, ToolCall, Usage
from support_agent.config import FIRST_AGENT_MESSAGE, HANDOFF_MESSAGE, RunConfig, derive_seed
from support_agent.toolkit import ToolBugError, ToolContext, execute

GET_O1 = ToolCall("toy_get_order", {"order_id": "O-1"}, id="call-1")
CANCEL_O1 = ToolCall("toy_cancel_order", {"order_id": "O-1"})
MISSING = ToolCall("toy_get_order", {"order_id": "O-404"})


class Harness:
    """One conversation: a private DB copy, a tool closure and the agent state."""

    def __init__(self, seed_engine, script, config: RunConfig | None = None):
        self.engine = db.memory_engine(seed_engine)
        self.ctx = ToolContext(now=NOW, enforce_policy=True)
        self.provider = ScriptedProvider(script)
        self.config = config or RunConfig()
        self.state: AgentState = new_state("system prompt")

    def run_tool(self, name: str, arguments: dict):
        return execute(TOY_REGISTRY, self.engine, self.ctx, name, arguments)

    def turn(self, text: str = "주문 O-1 상태 알려 주세요"):
        return agent_turn(
            self.state,
            text,
            provider=self.provider,
            registry=TOY_REGISTRY,
            run_tool=self.run_tool,
            config=self.config,
            task_id="T-1",
            trial=3,
        )


def test_tool_then_answer(tiny_engine):
    h = Harness(tiny_engine, [GET_O1, "결제 완료 상태이고 38,900원입니다."])
    result = h.turn()

    assert result.reply == "결제 완료 상태이고 38,900원입니다."
    assert result.stop is None
    assert [m.role for m in h.state.messages] == [
        "system",
        "assistant",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert h.state.messages[1].content == FIRST_AGENT_MESSAGE
    call_message, tool_message = h.state.messages[3], h.state.messages[4]
    assert call_message.content == "" and call_message.tool_calls == (GET_O1,)
    assert tool_message.tool_name == "toy_get_order" and tool_message.tool_call_id == "call-1"
    assert json.loads(tool_message.content)["total_won"] == 38900

    assert h.state.agent_calls == 2 and h.state.tool_errors == 0
    (log,) = h.state.tool_log
    assert (log.agent_call, log.name, log.ok, log.write) == (0, "toy_get_order", True, False)
    assert log.raw_arguments == log.args == {"order_id": "O-1"}
    assert log.ms >= 0
    # the second request saw the tool result
    assert h.provider.requests[1]["messages"][-1].role == "tool"


def test_write_tool_is_logged_as_write_and_tool_message_without_id(tiny_engine):
    h = Harness(tiny_engine, [CANCEL_O1, "취소했습니다."])
    h.turn("O-1 취소해 주세요")
    assert h.state.tool_log[0].write is True
    assert h.state.messages[4].tool_call_id is None


def test_text_with_tool_call_drops_the_text(tiny_engine):
    both = ChatResponse(text="잠시만요, 확인하겠습니다.", tool_calls=(GET_O1,))
    h = Harness(tiny_engine, [both, "확인했습니다."])
    result = h.turn()

    assert result.reply == "확인했습니다."
    assert all("잠시만요" not in m.content for m in h.state.messages)
    assert h.state.llm_log[0].dropped_text == "잠시만요, 확인하겠습니다."
    assert h.state.llm_log[0].format_error is None
    assert h.state.llm_log[1].dropped_text == ""


def test_only_the_first_of_several_calls_runs(tiny_engine):
    many = ChatResponse(tool_calls=(GET_O1, CANCEL_O1, MISSING))
    h = Harness(tiny_engine, [many, "확인했습니다."])
    h.turn()

    assert [log.name for log in h.state.tool_log] == ["toy_get_order"]
    assert h.state.dropped_calls == 2
    assert h.state.llm_log[0].dropped_calls == 2
    assert len(h.state.llm_log[0].tool_calls) == 3
    assert h.state.messages[3].tool_calls == (GET_O1,)
    with Session(h.engine) as session:
        assert session.get(db.Order, "O-1").status == db.OrderStatus.PAID


BAD_REPLIES = {
    "empty": ChatResponse(text="  "),
    "leaked_tool_call": ChatResponse(text='<tool_call>{"name": "toy_get_order"}</tool_call>'),
    "cut_off": ChatResponse(text="고객님의 주문은", finish_reason="length"),
}


def test_format_problem_kinds():
    for kind, response in BAD_REPLIES.items():
        assert format_problem(response) == kind
    assert format_problem(ChatResponse(text=' {"name": "x", "arguments": {}}')) == "leaked_tool_call"
    assert format_problem(ChatResponse(text="안녕하세요")) is None
    assert format_problem(ChatResponse(tool_calls=(GET_O1,))) is None
    assert format_problem(ChatResponse(text="<tool_call>", tool_calls=(GET_O1,))) is None


@pytest.mark.parametrize("kind", list(BAD_REPLIES))
def test_format_problem_retries_with_notice(tiny_engine, kind):
    bad = BAD_REPLIES[kind]
    h = Harness(tiny_engine, [bad, "네, 확인해 드리겠습니다."])
    result = h.turn()

    assert result.reply == "네, 확인해 드리겠습니다." and result.stop is None
    undelivered, notice = h.state.messages[3], h.state.messages[4]
    assert undelivered.role == "assistant" and undelivered.delivered is False
    assert undelivered.content == bad.text
    expected = CUT_OFF_NOTICE if kind == "cut_off" else FORMAT_NOTICE
    assert (notice.role, notice.content, notice.harness) == ("user", expected, True)
    assert h.provider.requests[1]["messages"][-1] == notice
    assert h.state.format_errors == 1
    assert [log.format_error for log in h.state.llm_log] == [kind, None]


@pytest.mark.parametrize("kind", list(BAD_REPLIES))
def test_format_problem_ends_after_the_retries(tiny_engine, kind):
    h = Harness(tiny_engine, [BAD_REPLIES[kind]] * 3 + ["never reached"])
    result = h.turn()

    assert (result.reply, result.stop) == (None, "agent_format_error")
    assert h.state.agent_calls == 3 and h.state.format_errors == 3
    assert sum(m.harness for m in h.state.messages) == 2
    assert all(
        m.delivered is False
        for m in h.state.messages
        if m.role == "assistant" and m is not h.state.messages[1]
    )


def test_format_retries_are_per_turn(tiny_engine):
    empty = ChatResponse()
    h = Harness(tiny_engine, [empty, empty, "첫 답", empty, empty, "둘째 답"])
    assert h.turn().reply == "첫 답"
    assert h.turn("하나 더요").reply == "둘째 답"
    assert h.state.format_errors == 4


def test_zero_format_retries_fails_at_once(tiny_engine):
    h = Harness(tiny_engine, [ChatResponse(), "never reached"], RunConfig(max_format_retries=0))
    result = h.turn()
    assert result.stop == "agent_format_error"
    assert h.state.agent_calls == 1
    assert not any(m.harness for m in h.state.messages)
    assert h.state.messages[-1].delivered is False


def test_tool_errors_accumulate_until_the_limit(tiny_engine):
    h = Harness(tiny_engine, [MISSING, MISSING, MISSING, "never reached"], RunConfig(max_tool_errors=3))
    result = h.turn()

    assert (result.reply, result.stop) == (None, "too_many_tool_errors")
    assert h.state.tool_errors == 3 and h.state.agent_calls == 3
    assert {log.error_code for log in h.state.tool_log} == {"order_not_found"}
    assert h.state.messages[-1].content.startswith("Error: [order_not_found]")


def test_tool_error_below_the_limit_goes_on(tiny_engine):
    h = Harness(tiny_engine, [MISSING, "주문을 찾지 못했습니다."])
    result = h.turn()
    assert result.reply == "주문을 찾지 못했습니다." and h.state.tool_errors == 1


def test_policy_block_is_logged(tiny_engine):
    h = Harness(tiny_engine, [ToolCall("toy_cancel_order", {"order_id": "O-2"}), "취소할 수 없습니다."])
    h.turn("O-2 취소해 주세요")
    (log,) = h.state.tool_log
    assert (log.ok, log.error_code, log.policy_blocked) == (False, "cancel_not_allowed_status", True)
    assert log.args == {"order_id": "O-2"}


def test_unknown_tool_counts_as_error_and_is_not_a_write(tiny_engine):
    h = Harness(tiny_engine, [ToolCall("refund_everything", {"x": 1}), "죄송합니다."])
    h.turn()
    (log,) = h.state.tool_log
    assert (log.name, log.ok, log.error_code, log.write, log.args) == (
        "refund_everything",
        False,
        "unknown_tool",
        False,
        None,
    )
    assert h.state.tool_errors == 1


def test_handoff_ends_the_conversation(tiny_engine):
    h = Harness(tiny_engine, [ToolCall("transfer_to_human", {"summary": "상담원 요청"}), "never reached"])
    result = h.turn("사람이랑 통화할래요")

    assert (result.reply, result.stop) == (HANDOFF_MESSAGE, "handoff")
    assert h.state.messages[-1].role == "assistant" and h.state.messages[-1].content == HANDOFF_MESSAGE
    assert h.state.agent_calls == 1
    with Session(h.engine) as session:
        assert session.get(db.Handoff, "HO-1") is not None


def test_failed_handoff_does_not_end_the_conversation(tiny_engine):
    h = Harness(tiny_engine, [ToolCall("transfer_to_human", {}), "다시 확인하겠습니다."])
    result = h.turn()
    assert result.stop is None and h.state.tool_errors == 1
    assert h.state.tool_log[0].error_code == "invalid_arguments"


def test_max_agent_calls(tiny_engine):
    h = Harness(tiny_engine, [GET_O1] * 5, RunConfig(max_agent_calls=3))
    result = h.turn()
    assert (result.reply, result.stop) == (None, "max_agent_calls")
    assert h.state.agent_calls == 3 and len(h.provider.requests) == 3

    # the budget is per episode: a later turn makes no further call
    assert h.turn("여보세요?").stop == "max_agent_calls"
    assert len(h.provider.requests) == 3


def test_context_limit(tiny_engine):
    big = ChatResponse(text="답변", usage=Usage(prompt_tokens=960, completion_tokens=5))
    h = Harness(tiny_engine, [big], RunConfig(num_ctx=1000))
    result = h.turn()
    assert (result.reply, result.stop) == (None, "context_limit")
    assert h.state.messages[-1].role == "user"
    assert h.state.llm_log[0].prompt_tokens == 960

    at_limit = ChatResponse(text="답변", usage=Usage(prompt_tokens=950))
    h = Harness(tiny_engine, [at_limit], RunConfig(num_ctx=1000))
    assert h.turn().reply == "답변"


def test_think_is_hidden_unless_r1(tiny_engine):
    def names(reasoning):
        return [t["name"] for t in visible_tools(TOY_REGISTRY, RunConfig(reasoning=reasoning))]

    assert "think" not in names("R0") and "think" not in names("R2")
    assert "think" in names("R1")
    assert len(names("R0")) == len(TOY_REGISTRY) - 1
    assert set(visible_tools(TOY_REGISTRY, RunConfig())[0]) == {"name", "description", "parameters"}

    h = Harness(tiny_engine, ["네"], RunConfig(reasoning="R1"))
    h.turn()
    assert "think" in [t["name"] for t in h.provider.requests[0]["tools"]]


def test_seeds_differ_per_call_and_are_recorded(tiny_engine):
    h = Harness(tiny_engine, [GET_O1, "첫 답", "둘째 답"], RunConfig(base_seed=7, temperature=0.3))
    h.turn()
    h.turn("감사합니다")

    sent = [r["seed"] for r in h.provider.requests]
    assert sent == [derive_seed(7, "T-1", 3, "agent", i) for i in range(3)]
    assert len(set(sent)) == 3
    assert [log.seed for log in h.state.llm_log] == sent
    assert [log.index for log in h.state.llm_log] == [0, 1, 2]
    assert all(log.who == "agent" for log in h.state.llm_log)
    assert h.provider.requests[0]["temperature"] == 0.3


def test_provider_error_propagates(tiny_engine):
    class Broken(ScriptedProvider):
        def chat(self, messages, tools=(), **options):
            raise ProviderError("timeout")

    h = Harness(tiny_engine, [])
    h.provider = Broken([])
    with pytest.raises(ProviderError):
        h.turn()


def test_tool_bug_propagates(tiny_engine):
    h = Harness(tiny_engine, [ToolCall("toy_bug", {"order_id": "O-1"})])
    with pytest.raises(ToolBugError):
        h.turn()


def test_system_prompt_has_the_policy_and_kst_time():
    prompt = build_system_prompt("반품은 수령 후 7일까지 가능합니다.\n", NOW)
    assert "<policy>\n반품은 수령 후 7일까지 가능합니다.\n</policy>" in prompt
    assert "현재 시각: 2026-09-14 10:00 (월요일)" in prompt
    assert "{" not in prompt and "}" not in prompt

    # a UTC instant is shown in KST, across the date line
    late = build_system_prompt("규정", datetime(2026, 9, 19, 16, 30, tzinfo=UTC))
    assert "2026-09-20 01:30 (일요일)" in late


def test_new_state_and_to_dict_are_json_serialisable(tiny_engine):
    both = ChatResponse(text="버려질 말", tool_calls=(GET_O1, MISSING))
    h = Harness(tiny_engine, [ChatResponse(), both, MISSING, "답변"])
    h.turn()

    data = json.loads(json.dumps(h.state.to_dict(), ensure_ascii=False))
    assert data["agent_calls"] == 4
    assert (data["tool_errors"], data["format_errors"], data["dropped_calls"]) == (1, 1, 1)
    assert data["messages"][0] == {"role": "system", "content": "system prompt"}
    assert data["messages"][3]["delivered"] is False
    assert data["messages"][4]["harness"] is True
    assert data["messages"][5]["tool_calls"][0]["name"] == "toy_get_order"
    assert set(data) == {"messages", "agent_calls", "tool_errors", "format_errors", "dropped_calls"}
