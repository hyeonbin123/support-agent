"""에이전트 루프를 대본 제공자, 장난감 도구 등록부, toolkit.execute를 실제로 감싼 도구 함수로 시험한다."""

# [테스트 묶음] 에이전트 루프(agent.agent_turn)의 규칙을 모델 없이 확인한다. 대본대로 답하는 가짜
# 제공자(ScriptedProvider), 실제 도메인 도구 대신 장난감 도구 등록부(toy_tools.py), 실제 단일 실행
# 경로(toolkit.execute)를 묶어 한 턴을 돌린다.
# [왜 중요한가] 루프 규칙(텍스트와 도구 호출이 함께 오면 도구를 따름, 여러 호출 중 첫 번째만, 형식 오류
# 재시도, 도구 오류 한도, 이관 종료, 컨텍스트 한도, seed)은 측정 결과의 해석을 바꾼다 (docs/design.md
# "에이전트 루프 규칙"). 규칙이 조용히 바뀌면 단계 사이의 수치를 비교할 수 없게 되므로 하나하나 고정한다.

from __future__ import annotations

# conftest의 고정 시각 NOW와 작은 DB fixture, 장난감 도구 등록부를 쓴다. agent 모듈에서는 루프와 형식
# 검사·안내문 상수를, chat·config·toolkit에서는 메시지 타입과 실행 설정·실행 경로를 가져온다.
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

# 자주 쓰는 도구 호출: 조회(id가 있어 tool 메시지의 tool_call_id 연결을 확인), 취소(쓰기, id 없음), 없는 주문
# 조회(order_not_found 오류를 일으킨다).
GET_O1 = ToolCall("toy_get_order", {"order_id": "O-1"}, id="call-1")
CANCEL_O1 = ToolCall("toy_cancel_order", {"order_id": "O-1"})
MISSING = ToolCall("toy_get_order", {"order_id": "O-404"})


# 대화 하나를 위한 도구 묶음. 테스트마다 작은 DB의 복사본을 만들고, 규정을 막는(P1, enforce_policy=True) 도구
# 문맥과 대본 제공자를 둔다. run_tool은 평가·서비스와 같은 execute()를 지나므로 루프와 실제 실행 경로를 함께
# 시험한다. task_id와 trial을 고정해 seed 계산을 확인할 수 있다.
class Harness:
    """대화 하나: 전용 DB 복사본, 도구 호출 함수, 에이전트 상태."""

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


# 도구 한 번 → 답의 기본 흐름. 메시지 순서(system, 첫 인사, 고객, 호출, 도구 결과, 답), 호출 메시지는 본문이
# 비고 호출만 담는 것, 도구 결과가 호출 id로 이어지는 것, 호출·도구 로그의 값, 두 번째 요청이 도구 결과를 보고
# 있는지를 확인한다. 기록 형식이 바뀌면 episodes.jsonl과 분석 코드가 함께 깨지므로 고정한다.
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
    # 두 번째 요청은 도구 결과를 보았다
    assert h.provider.requests[1]["messages"][-1].role == "tool"


# 쓰기 도구 호출은 로그에 write=True로 남아야 한다 (정답에 없는 쓰기 같은 보조 지표가 이 표시를 센다). 호출에
# id가 없으면 tool 메시지의 tool_call_id도 None이다.
def test_write_tool_is_logged_as_write_and_tool_message_without_id(tiny_engine):
    h = Harness(tiny_engine, [CANCEL_O1, "취소했습니다."])
    h.turn("O-1 취소해 주세요")
    assert h.state.tool_log[0].write is True
    assert h.state.messages[4].tool_call_id is None


# 텍스트와 도구 호출이 함께 오면 도구 호출을 따르고, 텍스트는 고객에게 보내지 않으며 대화 기록에서도 뺀다.
# 버린 말은 LLM 호출 로그의 dropped_text에만 남는다. 형식 오류로 세지 않는다 (docs/design.md 루프 규칙 첫 줄).
def test_text_with_tool_call_drops_the_text(tiny_engine):
    both = ChatResponse(text="잠시만요, 확인하겠습니다.", tool_calls=(GET_O1,))
    h = Harness(tiny_engine, [both, "확인했습니다."])
    result = h.turn()

    assert result.reply == "확인했습니다."
    assert all("잠시만요" not in m.content for m in h.state.messages)
    assert h.state.llm_log[0].dropped_text == "잠시만요, 확인하겠습니다."
    assert h.state.llm_log[0].format_error is None
    assert h.state.llm_log[1].dropped_text == ""


# 한 응답에 도구 호출이 여러 개면 첫 번째만 실행하고 나머지는 dropped_calls로 센다. 두 번째인 취소가 실행되지
# 않았음을 DB(O-1이 그대로 paid)로 확인한다. 한 번에 여러 쓰기가 실행되면 규칙 밖의 일이 생기기 때문이다.
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


# 형식 오류 세 종류: 빈 응답, 본문에 샌 도구 호출(<tool_call> 태그), 길이 제한으로 잘린
# 응답(finish_reason=length).
BAD_REPLIES = {
    "empty": ChatResponse(text="  "),
    "leaked_tool_call": ChatResponse(text='<tool_call>{"name": "toy_get_order"}</tool_call>'),
    "cut_off": ChatResponse(text="고객님의 주문은", finish_reason="length"),
}


# format_problem의 분류를 확인한다. name과 arguments를 가진 JSON이 본문에 있어도 샌 호출로 보고, 정상 문장과
# 정상 도구 호출은 문제가 없다. 실제 호출이 함께 있으면 본문에 태그가 있어도 샌 호출로 보지 않는다.
def test_format_problem_kinds():
    for kind, response in BAD_REPLIES.items():
        assert format_problem(response) == kind
    assert format_problem(ChatResponse(text=' {"name": "x", "arguments": {}}')) == "leaked_tool_call"
    assert format_problem(ChatResponse(text="안녕하세요")) is None
    assert format_problem(ChatResponse(tool_calls=(GET_O1,))) is None
    assert format_problem(ChatResponse(text="<tool_call>", tool_calls=(GET_O1,))) is None


# 형식 오류가 나면 그 응답을 전달하지 않은 것(delivered=False)으로 기록하고, 하니스 안내문(harness=True인 user
# 메시지)을 넣어 다시 부른다. 잘린 응답에는 '더 짧게' 안내가 간다. 다시 부를 때 마지막 메시지가 그
# 안내문이어야 하고, 형식 오류 수와 로그의 종류가 맞아야 한다.
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


# 재시도 한도(기본 max_format_retries = 2)를 넘기면 agent_format_error로 끝난다. 호출은 3번, 안내문은 2번이고,
# 첫 인사 말고는 어떤 상담원 응답도 전달되지 않는다.
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


# 형식 재시도 횟수는 턴마다 새로 센다. 앞 턴에서 두 번 다시 시켰어도 다음 턴은 다시 두 번까지 된다. 형식 오류
# 누계(4)는 에피소드 전체로 남는다.
def test_format_retries_are_per_turn(tiny_engine):
    empty = ChatResponse()
    h = Harness(tiny_engine, [empty, empty, "첫 답", empty, empty, "둘째 답"])
    assert h.turn().reply == "첫 답"
    assert h.turn("하나 더요").reply == "둘째 답"
    assert h.state.format_errors == 4


# 재시도 0이면 첫 형식 오류에서 바로 끝나고 안내문을 넣지 않는다. 마지막 메시지는 전달되지 않은 응답이다.
def test_zero_format_retries_fails_at_once(tiny_engine):
    h = Harness(tiny_engine, [ChatResponse(), "never reached"], RunConfig(max_format_retries=0))
    result = h.turn()
    assert result.stop == "agent_format_error"
    assert h.state.agent_calls == 1
    assert not any(m.harness for m in h.state.messages)
    assert h.state.messages[-1].delivered is False


# 도구 오류가 한도(max_tool_errors)에 닿으면 too_many_tool_errors로 끝난다. 모델은 'Error: [코드] 메시지'
# 형태의 결과를 받는다 (예상한 실패는 문자열로 돌려준다는 설계).
def test_tool_errors_accumulate_until_the_limit(tiny_engine):
    h = Harness(tiny_engine, [MISSING, MISSING, MISSING, "never reached"], RunConfig(max_tool_errors=3))
    result = h.turn()

    assert (result.reply, result.stop) == (None, "too_many_tool_errors")
    assert h.state.tool_errors == 3 and h.state.agent_calls == 3
    assert {log.error_code for log in h.state.tool_log} == {"order_not_found"}
    assert h.state.messages[-1].content.startswith("Error: [order_not_found]")


# 한도 아래의 도구 오류는 대화를 끝내지 않고, 모델이 오류를 읽고 답할 수 있다.
def test_tool_error_below_the_limit_goes_on(tiny_engine):
    h = Harness(tiny_engine, [MISSING, "주문을 찾지 못했습니다."])
    result = h.turn()
    assert result.reply == "주문을 찾지 못했습니다." and h.state.tool_errors == 1


# P1에서 규정 검사(출고된 주문 O-2의 취소)에 막힌 호출은 로그에 오류 코드와 policy_blocked=True로 남는다.
# 'P1에서 막힌 코드' 보조 지표가 이 값을 센다.
def test_policy_block_is_logged(tiny_engine):
    h = Harness(tiny_engine, [ToolCall("toy_cancel_order", {"order_id": "O-2"}), "취소할 수 없습니다."])
    h.turn("O-2 취소해 주세요")
    (log,) = h.state.tool_log
    assert (log.ok, log.error_code, log.policy_blocked) == (False, "cancel_not_allowed_status", True)
    assert log.args == {"order_id": "O-2"}


# 없는 도구를 부르면 unknown_tool 오류로 세고, 쓰기로 세지 않으며, 검증된 인자도 없다(None). 모델이 지어낸
# 도구 이름이 정답에 없는 쓰기로 잘못 세어지지 않게 한다.
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


# 상담원 이관 도구가 성공하면 고정 안내문(HANDOFF_MESSAGE)을 고객에게 전달한 것으로 치고 handoff로 끝난다. 그
# 뒤 모델을 더 부르지 않고(대본의 'never reached'), 이관 행이 DB에 남는다.
def test_handoff_ends_the_conversation(tiny_engine):
    h = Harness(tiny_engine, [ToolCall("transfer_to_human", {"summary": "상담원 요청"}), "never reached"])
    result = h.turn("사람이랑 통화할래요")

    assert (result.reply, result.stop) == (HANDOFF_MESSAGE, "handoff")
    assert h.state.messages[-1].role == "assistant" and h.state.messages[-1].content == HANDOFF_MESSAGE
    assert h.state.agent_calls == 1
    with Session(h.engine) as session:
        assert session.get(db.Handoff, "HO-1") is not None


# 이관 호출이 인자 검증에 실패하면(summary 없음) 대화가 끝나지 않고 도구 오류로 센다. 끝나는 것은 성공한
# 이관뿐이다.
def test_failed_handoff_does_not_end_the_conversation(tiny_engine):
    h = Harness(tiny_engine, [ToolCall("transfer_to_human", {}), "다시 확인하겠습니다."])
    result = h.turn()
    assert result.stop is None and h.state.tool_errors == 1
    assert h.state.tool_log[0].error_code == "invalid_arguments"


# 에이전트 LLM 호출 한도는 에피소드 전체에 대한 것이다. 한도에 닿으면 max_agent_calls로 끝나고, 다음 턴에서도
# 모델을 더 부르지 않는다.
def test_max_agent_calls(tiny_engine):
    h = Harness(tiny_engine, [GET_O1] * 5, RunConfig(max_agent_calls=3))
    result = h.turn()
    assert (result.reply, result.stop) == (None, "max_agent_calls")
    assert h.state.agent_calls == 3 and len(h.provider.requests) == 3

    # 한도는 에피소드 단위다: 뒤의 턴은 더 이상 호출하지 않는다
    assert h.turn("여보세요?").stop == "max_agent_calls"
    assert len(h.provider.requests) == 3


# 프롬프트 토큰이 num_ctx의 95%를 넘으면 context_limit로 끝난다(960 > 950). 정확히 95%면 계속된다. Ollama는
# 컨텍스트가 넘치면 오류 없이 앞을 잘라 버리므로, 잘린 대화로 판정하지 않으려고 미리 멈춘다 (docs/design.md
# 루프 규칙).
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


# think 도구는 R1일 때만 모델에게 보인다 (R0·R2에서는 숨김). 도구 스키마는 name, description, parameters 세
# 키뿐이다. 실제 요청에도 R1이면 think가 들어가는지 본다.
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


# 모델 호출마다 seed가 derive_seed(기본 seed, 과제, 시도, "agent", 호출 순번)로 정해지고, 요청과 로그에 같은
# 값이 남는지 확인한다. 같은 명령이 같은 seed로 도는 재현성의 근거다 (README "다시 재는 방법"). temperature도
# 설정대로 전달돼야 한다.
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


# 제공자 오류(시간 초과 등)는 루프가 삼키지 않고 위로 올린다. 에피소드 실행기가 이를 infra_error로 기록해
# 분모에서 뺀다 (에이전트의 실패가 아니라 기반의 실패).
def test_provider_error_propagates(tiny_engine):
    class Broken(ScriptedProvider):
        def chat(self, messages, tools=(), **options):
            raise ProviderError("timeout")

    h = Harness(tiny_engine, [])
    h.provider = Broken([])
    with pytest.raises(ProviderError):
        h.turn()


# 도구 핸들러의 예상 못 한 예외는 ToolBugError로 위로 올라간다. 우리 코드의 버그를 에이전트의 실패로 세지 않기
# 위해서다.
def test_tool_bug_propagates(tiny_engine):
    h = Harness(tiny_engine, [ToolCall("toy_bug", {"order_id": "O-1"})])
    with pytest.raises(ToolBugError):
        h.turn()


# 시스템 프롬프트에 규정 문서가 <policy> 태그 안에 들어가고, 현재 시각이 KST와 요일로 적히며, 틀에 채우지 않은
# 중괄호가 남지 않는지 확인한다. UTC 시각은 날짜가 바뀌는 경우에도 KST로 옮겨 적어야 한다.
def test_system_prompt_has_the_policy_and_kst_time():
    prompt = build_system_prompt("반품은 수령 후 7일까지 가능합니다.\n", NOW)
    assert "<policy>\n반품은 수령 후 7일까지 가능합니다.\n</policy>" in prompt
    assert "현재 시각: 2026-09-14 10:00 (월요일)" in prompt
    assert "{" not in prompt and "}" not in prompt

    # UTC 시각은 날짜가 바뀌더라도 KST로 보여 준다
    late = build_system_prompt("규정", datetime(2026, 9, 19, 16, 30, tzinfo=UTC))
    assert "2026-09-20 01:30 (일요일)" in late


# AgentState.to_dict()가 JSON으로 왕복되고 키 집합이 정해져 있는지 확인한다. 서비스는 이 dict를 세션에 저장해
# 다음 요청에서 대화를 이어 가므로, 전달하지 않은 응답(delivered=False)과 하니스 안내문(harness=True) 표시까지
# 그대로 남아야 한다.
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
    assert set(data) == {
        "messages",
        "agent_calls",
        "tool_errors",
        "format_errors",
        "dropped_calls",
        "stalls",
        "held_claims",
    }
