from __future__ import annotations

from datetime import datetime

import pytest

from support_agent.chat import ChatResponse, Message, ScriptedProvider, ToolCall, Usage
from support_agent.clock import KST
from support_agent.config import OUT_OF_SCOPE_TOKEN, STOP_TOKEN, RunConfig, derive_seed
from support_agent.tasks import RequiredValue, Task, ToolAction, UserScenario
from support_agent.user_sim import TURN_REMINDER, LLMUser, ScriptedUser, SimulatorError, build_user_prompt

GREETING = "안녕하세요, 고객센터입니다. 무엇을 도와드릴까요?"


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


class RecordingProvider(ScriptedProvider):
    """ScriptedProvider does not keep max_tokens, so record it here."""

    def __init__(self, script):
        super().__init__(script)
        self.max_tokens: list[int] = []

    def chat(self, messages, tools=(), *, temperature=0.0, seed=None, max_tokens=1024):
        self.max_tokens.append(max_tokens)
        return super().chat(messages, tools, temperature=temperature, seed=seed, max_tokens=max_tokens)


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


def test_scripted_user_without_lines_stops_at_once():
    assert ScriptedUser([]).reply(GREETING) == STOP_TOKEN


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


def test_prompt_today_is_the_kst_date_and_empty_fields_are_marked():
    late_utc = datetime.fromisoformat("2026-09-13T16:30:00+00:00")  # 01:30 on the 14th in Korea
    task = make_task(
        now=late_utc, user=UserScenario(reason="배송이 언제 오는지 궁금하다", known="이름은 김하준")
    )
    prompt = build_user_prompt(task)
    assert "오늘 날짜: 2026년 9월 14일" in prompt
    assert "모르는 정보: (없음)" in prompt


def test_prompt_survives_braces_in_the_scenario():
    task = make_task(user=UserScenario(reason="문의 {중괄호} 포함", known="{known} 그대로"))
    prompt = build_user_prompt(task)
    assert "문의 {중괄호} 포함" in prompt
    assert "{known} 그대로" in prompt


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


def test_llm_user_seeds_differ_between_trials_and_tasks():
    seeds = set()
    for task_id, trial in (("T-a", 0), ("T-a", 1), ("T-b", 0)):
        provider = ScriptedProvider(["네"])
        LLMUser(provider, make_task(id=task_id), RunConfig(), trial).reply(GREETING)
        seeds.add(provider.requests[0]["seed"])
    assert len(seeds) == 3


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


@pytest.mark.parametrize(
    "reply",
    [STOP_TOKEN, OUT_OF_SCOPE_TOKEN, f"감사합니다. {STOP_TOKEN}", f"그건 잘 모르겠네요 {OUT_OF_SCOPE_TOKEN}"],
)
def test_llm_user_returns_token_replies_as_they_are(reply):
    user = LLMUser(ScriptedProvider([f" {reply} "]), make_task(), RunConfig(), trial=0)
    assert user.reply(GREETING) == reply


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
