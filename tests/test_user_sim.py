# [파일 역할] 고객 역할(user_sim.py)의 두 구현을 검증하는 테스트 묶음이다. ScriptedUser는 테스트용 대본
# 고객이고, LLMUser는 실제 측정에서 시나리오를 받아 고객을 연기하는 LLM 시뮬레이터다.
# [왜 중요한가] τ-bench 방식에서 고객 역할은 LLM이 맡으므로(README "평가 방식"), 시뮬레이터가 과제의 정답(고객
# id, 정답 동작, 전달 값)을 보거나, 호출마다 seed가 흔들리거나, 기록이 지저분해지면 측정 자체를 믿을 수 없다.
# 1단계 시뮬레이터 점검 뒤 고친 내용(매 턴 리마인더, 기록에는 리마인더를 남기지 않음, docs/experiments.md
# 1단계 "결과 1")도 여기서 지킨다.
# [방법] 실제 모델 대신 ScriptedProvider로 응답을 재생하고, 요청에 실린 메시지·seed·temperature·max_tokens를
# 확인한다.

from __future__ import annotations

from datetime import datetime

import pytest

# chat: 메시지 타입과 대본 재생 제공자. config: 종료 토큰(STOP, OUT-OF-SCOPE), 실행 설정, LLM 호출마다 seed를
# 만드는 derive_seed. tasks: 테스트용 과제를 직접 만든다. user_sim: 검증 대상.
from support_agent.chat import ChatResponse, Message, ScriptedProvider, ToolCall, Usage
from support_agent.clock import KST
from support_agent.config import OUT_OF_SCOPE_TOKEN, STOP_TOKEN, RunConfig, derive_seed
from support_agent.tasks import RequiredValue, Task, ToolAction, UserScenario
from support_agent.user_sim import TURN_REMINDER, LLMUser, ScriptedUser, SimulatorError, build_user_prompt

# 에이전트의 첫 인사말. 시뮬레이터가 처음 받는 상담원 발화로 쓴다.
GREETING = "안녕하세요, 고객센터입니다. 무엇을 도와드릴까요?"


# 결제 완료 주문을 취소하는 테스트 과제를 만든다. overrides로 필드를 바꾼다.
# 고객 id(C-1), 정답 동작(toy_cancel_order, O-1), 전달 값(38900, 라벨-환불), 목적(purpose)은 시뮬레이터
# 프롬프트에 새면 안 되는 값으로, 아래 프롬프트 테스트가 이 값들이 없는지 찾는다.
def make_task(**overrides) -> Task:
    fields = dict(
        id="T-cancel",
        type="action",
        purpose="결제 완료 주문 취소",
        now=datetime(2026, 9, 14, 10, 0, tzinfo=KST),
        customer_id="C-1",
        user=UserScenario(
            persona="말이 짧고 급한 편",
            reason="어제 주문한 이어폰을 취소하고 싶다",
            known="이름은 김하준, 전화번호는 010-0000-0001",
            unknown="주문 번호",
            rules="취소가 접수되고 환불 금액을 들으면 끝낸다",
        ),
        gold_actions=[ToolAction(tool="toy_cancel_order", args={"order_id": "O-1"})],
        required_values=[RequiredValue(kind="number", value=38900, label="라벨-환불")],
    )
    fields.update(overrides)
    return Task(**fields)


# max_tokens까지 기록하는 대본 제공자. 시뮬레이터 답의 길이 상한(USER_MAX_TOKENS 256)이 실제로 요청에 실리는지
# 보려고 둔다.
class RecordingProvider(ScriptedProvider):
    """ScriptedProvider does not keep max_tokens, so record it here."""

    def __init__(self, script):
        super().__init__(script)
        self.max_tokens: list[int] = []

    def chat(self, messages, tools=(), *, temperature=0.0, seed=None, max_tokens=1024):
        self.max_tokens.append(max_tokens)
        return super().chat(messages, tools, temperature=temperature, seed=seed, max_tokens=max_tokens)


# ScriptedUser가 준비된 말을 차례로 하고, 다 쓰면 계속 STOP 토큰을 내는지 본다. LLM을 부르지 않으므로
# llm_log는 비어 있고, 기록은 시뮬레이터 쪽 관점(상담원 말 = user, 고객 말 = assistant)으로 쌓인다. 다른
# 테스트들이 이 대본 고객으로 에피소드를 끝내므로 동작이 바뀌면 안 된다.
def test_scripted_user_says_its_lines_then_stops():
    user = ScriptedUser(["주문을 취소하고 싶어요", "네 맞아요"])
    assert user.reply(GREETING) == "주문을 취소하고 싶어요"
    assert user.reply("성함을 알려 주세요") == "네 맞아요"
    assert user.reply("취소했습니다") == STOP_TOKEN
    assert user.reply("더 필요한 것이 있으신가요?") == STOP_TOKEN
    assert user.llm_log == []
    assert [m.role for m in user.messages] == ["user", "assistant"] * 4
    assert user.messages[0] == Message("user", GREETING)
    assert user.messages[-1] == Message("assistant", STOP_TOKEN)


# 대본이 비어 있으면 첫 답부터 STOP인지 본다.
def test_scripted_user_without_lines_stops_at_once():
    assert ScriptedUser([]).reply(GREETING) == STOP_TOKEN


# 시뮬레이터 시스템 프롬프트에 시나리오 다섯 칸, 오늘 날짜, 종료 토큰 두 개, <scenario> 표지가 들어가고, 고객
# id·정답 도구·주문 번호·전달 값(두 표기)·값 라벨·과제 목적은 들어가지 않는지 본다.
# 전달 값이 시뮬레이터에 새면 고객이 상담원보다 먼저 값을 말해 판정이 오염된다. 중괄호가 남지 않았는지도
# 보는데, 템플릿의 자리 표시자가 채워지지 않고 남는 실수를 잡는다.
def test_prompt_holds_the_scenario_and_nothing_else_of_the_task():
    task = make_task()
    prompt = build_user_prompt(task)
    for shown in (
        task.user.persona,
        task.user.reason,
        task.user.known,
        task.user.unknown,
        task.user.rules,
        "2026년 9월 14일",
        STOP_TOKEN,
        OUT_OF_SCOPE_TOKEN,
        "<scenario>",
    ):
        assert shown in prompt
    for hidden in ("C-1", "toy_cancel_order", "O-1", "38900", "38,900", "라벨-환불", task.purpose, "{", "}"):
        assert hidden not in prompt


# now가 UTC로 9월 13일 16:30이어도 프롬프트의 오늘 날짜는 한국 시각 날짜(9월 14일)인지, 빈 칸은 '(없음)'으로
# 표시되는지 본다. 반품 기한처럼 날짜로 판단하는 규정이 KST 날짜 기준이라 고객이 아는 '오늘'도 같아야 한다.
def test_prompt_today_is_the_kst_date_and_empty_fields_are_marked():
    late_utc = datetime.fromisoformat("2026-09-13T16:30:00+00:00")  # 01:30 on the 14th in Korea
    task = make_task(
        now=late_utc, user=UserScenario(reason="배송이 언제 오는지 궁금하다", known="이름은 김하준")
    )
    prompt = build_user_prompt(task)
    assert "오늘 날짜: 2026년 9월 14일" in prompt
    assert "모르는 정보: (없음)" in prompt


# 시나리오 글에 중괄호가 있어도 프롬프트 생성이 깨지지 않고 글이 그대로 들어가는지 본다. 템플릿만
# str.format으로 채우고 시나리오 값은 다시 해석하지 않는다는 것을 확인한다.
def test_prompt_survives_braces_in_the_scenario():
    task = make_task(user=UserScenario(reason="문의 {중괄호} 포함", known="{known} 그대로"))
    prompt = build_user_prompt(task)
    assert "문의 {중괄호} 포함" in prompt
    assert "{known} 그대로" in prompt


# LLMUser의 요청 형식을 세 차례 대화로 확인한다.
# - 역할 뒤집기: 상담원 말은 user, 시뮬레이터 자신의 말은 assistant로 쌓인다.
# - 리마인더: 가장 새 메시지에만 역할 지침(TURN_REMINDER)과 '당신이 아는 정보'(known)가 붙고, 이전 메시지와
#   기록(user.messages)에는 남지 않는다. 7B 시뮬레이터가 몇 턴 뒤 규칙을 잊어서 넣은 장치이고, 기록에 남기면
#   대화 기록이 오염된다.
# - 도구 없음, 설정의 temperature, derive_seed(기준 seed, 과제, 시도, 'user', 호출 순번)로 만든 서로 다른
#   seed, 답 길이 상한 256.
# - 답 앞뒤 공백은 지운다.
def test_llm_user_flips_roles_and_passes_seed_and_temperature():
    task = make_task()
    config = RunConfig(base_seed=77, user_temperature=0.6)
    provider = RecordingProvider(["  주문 취소하려고요.\n", "김하준이고 010-0000-0001이에요", STOP_TOKEN])
    user = LLMUser(provider, task, config, trial=2)

    assert user.reply(GREETING) == "주문 취소하려고요."
    assert user.reply("성함과 연락처를 알려 주세요.") == "김하준이고 010-0000-0001이에요"
    assert user.reply("취소했습니다. 38,900원이 환불됩니다.") == STOP_TOKEN

    first, second, third = provider.requests
    assert [m.role for m in first["messages"]] == ["system", "user"]
    assert first["messages"][0].content == build_user_prompt(task)
    assert first["messages"][1].content.startswith(f"{GREETING}\n\n{TURN_REMINDER}")  # sent with the reminder
    assert task.user.known in first["messages"][1].content
    assert [m.role for m in second["messages"]] == ["system", "user", "assistant", "user"]
    assert second["messages"][2].content == "주문 취소하려고요."
    assert second["messages"][3].content.startswith("성함과 연락처를 알려 주세요.\n\n(고객 역할 지침")
    assert TURN_REMINDER not in second["messages"][1].content  # only the newest message carries it
    assert all(TURN_REMINDER not in m.content for m in user.messages)  # the record stays clean
    assert len(third["messages"]) == 6

    for index, request in enumerate(provider.requests):
        assert request["tools"] == []
        assert request["temperature"] == 0.6
        assert request["seed"] == derive_seed(77, task.id, 2, "user", index)
    assert provider.max_tokens == [256, 256, 256]
    assert len({r["seed"] for r in provider.requests}) == 3

    assert [m.role for m in user.messages] == ["system", "user", "assistant"] * 1 + ["user", "assistant"] * 2
    assert user.messages[-1].content == STOP_TOKEN


# 시도 번호나 과제가 다르면 첫 호출의 seed가 달라지는지 본다. 시도마다 고객의 말이 달라져야 pass^k(같은 과제를
# 여러 번 시켰을 때 모두 성공하는 비율)가 의미를 가진다.
def test_llm_user_seeds_differ_between_trials_and_tasks():
    seeds = set()
    for task_id, trial in (("T-a", 0), ("T-a", 1), ("T-b", 0)):
        provider = ScriptedProvider(["네"])
        LLMUser(provider, make_task(id=task_id), RunConfig(), trial).reply(GREETING)
        seeds.add(provider.requests[0]["seed"])
    assert len(seeds) == 3


# 시뮬레이터 호출마다 LLMCallLog가 남고, 순번·역할(user)·seed·글·토큰 수·시간·종료 이유가 제공자 응답 그대로
# 기록되는지 본다. 기본 seed(1000)로 만든 seed 값까지 확인한다. 이 기록으로 reports/에서 시뮬레이터 점검과
# 시간 수치를 다시 계산한다.
def test_llm_user_logs_every_call():
    usage = Usage(prompt_tokens=321, completion_tokens=12, wall_ms=40.0, eval_ms=30.0)
    provider = ScriptedProvider([ChatResponse(text="취소해 주세요", usage=usage, finish_reason="stop"), "네"])
    user = LLMUser(provider, make_task(), RunConfig(), trial=0)
    user.reply(GREETING)
    user.reply("확인했습니다.")
    assert [log.index for log in user.llm_log] == [0, 1]
    log = user.llm_log[0]
    assert log.who == "user"
    assert log.seed == derive_seed(1000, "T-cancel", 0, "user", 0)
    assert log.text == "취소해 주세요"
    assert log.tool_calls == []
    assert log.format_error is None
    assert (log.prompt_tokens, log.completion_tokens, log.wall_ms, log.eval_ms) == (321, 12, 40.0, 30.0)
    assert log.finish_reason == "stop"


# 종료 토큰이 든 답은 앞뒤 공백만 지우고 그대로 돌려주는지 본다. 토큰을 해석해 대화를 끝내는 것은 에피소드
# 실행기의 일이고, 시뮬레이터는 답을 바꾸지 않는다.
@pytest.mark.parametrize(
    "reply",
    [STOP_TOKEN, OUT_OF_SCOPE_TOKEN, f"감사합니다. {STOP_TOKEN}", f"그건 잘 모르겠네요 {OUT_OF_SCOPE_TOKEN}"],
)
def test_llm_user_returns_token_replies_as_they_are(reply):
    user = LLMUser(ScriptedProvider([f" {reply} "]), make_task(), RunConfig(), trial=0)
    assert user.reply(GREETING) == reply


# 빈 답, 공백뿐인 답, 도구 호출만 있는 답은 SimulatorError(과제 id가 든 메시지)로 올라오는지 본다.
# 시뮬레이터가 쓸 말을 내지 못한 것은 에이전트 실패가 아니라 infra_error로 빼야 하기 때문이다. 그래도 호출
# 기록은 format_error='empty'로 남고, 답은 기록(messages)에 넣지 않는다.
@pytest.mark.parametrize(
    "response", ["", "  \n ", ChatResponse(tool_calls=(ToolCall("toy_get_order", {"order_id": "O-1"}),))]
)
def test_llm_user_raises_on_empty_output(response):
    user = LLMUser(ScriptedProvider([response]), make_task(), RunConfig(), trial=3)
    with pytest.raises(SimulatorError, match="T-cancel"):
        user.reply(GREETING)
    assert len(user.llm_log) == 1
    assert user.llm_log[0].format_error == "empty"
    assert [m.role for m in user.messages] == ["system", "user"]
