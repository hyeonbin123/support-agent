"""에피소드의 고객 쪽: 시나리오를 연기하는 LLM과, 테스트용으로 대본을 읽는 대역."""

# [파일 역할] 에피소드의 고객 쪽. 과제의 시나리오를 받아 고객을 연기하는 LLM(LLMUser)과, 테스트용으로 정해진
# 대사를 차례로 말하는 대역(ScriptedUser)을 둔다.
# [설계 이유] τ-bench 방식이라 고객 역할을 LLM 시뮬레이터가 맡는다(README "평가 방식"). 시뮬레이터에게는
# 시나리오만 주고 고객 번호·정답 동작·전달 값은 주지 않아, 고객이 정답을 흘려 에이전트를 돕지 못하게 한다.
# 시뮬레이터의 모델과 프롬프트는 한 단계 안에서 바꾸지 않는다(docs/plan.md "시뮬레이터 잡음").
# [연결] run.py가 과제·시도마다 LLMUser를 만들어 episode.py에 넘기고, 테스트는 ScriptedUser를 쓴다. run.py와
# sim_report.py는 build_user_prompt의 결과로 프롬프트 해시·출처 토큰을 만든다.

from __future__ import annotations

from collections.abc import Sequence
from importlib.resources import files
from typing import Protocol

# chat: 제공자 중립 메시지 타입과 제공자 인터페이스. config: 종료 토큰(STOP_TOKEN), 실행 설정, 호출마다의 seed
# 계산. records.LLMCallLog: 시뮬레이터 호출도 에이전트 호출과 같은 형식으로 기록한다.
from support_agent.chat import ChatProvider, Message
from support_agent.clock import kst_date
from support_agent.config import STOP_TOKEN, RunConfig, derive_seed
from support_agent.records import LLMCallLog
from support_agent.tasks import Task

# 고객 한 번의 답에 허용하는 최대 토큰. 고객은 한두 문장만 말하므로 짧게 둔다(이 한도에 잘린 답은
# sim_report.py가 cut_off로 센다).
USER_MAX_TOKENS = 256
# 매 턴 모델에 보내는 마지막 메시지 끝에 붙이는 역할 지침. 7B 시뮬레이터가 몇 턴 지나면 시스템 프롬프트의
# 규칙을 잊어서 2026-09-19에 넣었다(docs/experiments.md 1단계 "시뮬레이터를 바꿨다").
# "원하던 일이 모두 끝난 뒤에는"이라고 쓴 이유: "끝내는 조건이 채워졌으면 STOP"으로 쓰면 첫 턴에 바로 STOP을
# 냈다(같은 기록).
TURN_REMINDER = (
    "(고객 역할 지침: 위는 상담원의 말입니다. 당신은 고객입니다. 한국어로 한두 문장만 답하세요. "
    "<scenario>에 없는 번호·날짜·금액·주소는 말하지 마세요. "
    "상담원에게 기다리라고 하거나 상담원처럼 안내하지 마세요. "
    "상담원이 묻는 것 가운데 아는 것은 바로 알려 주고, 모르는 것은 모른다고 하세요. "
    "원하는 일이 아직 끝나지 않았으면 대화를 계속하세요. "
    "원하던 일이 모두 끝난 뒤에는 인사 대신 ###STOP### 만 출력하세요.)"
)
# 시나리오의 빈 항목(성격, 아는 것, 모르는 것, 규칙) 자리에 넣는 표기.
_EMPTY_FIELD = "(없음)"


# 모델에 보내는 가장 새 메시지의 글: 상담원의 말 뒤에 역할 지침(TURN_REMINDER)과 시나리오의 known("당신이 아는
# 정보")을 붙인다. LLMUser.reply가 매 턴 이 글을 보내고(기록에는 상담원의 말만 남긴다), run.py는 상담원 말과
# known 자리를 비운 이 글의 해시를 manifest(harness_sha256.turn_reminder)에 남겨 지침과 형식이 바뀌면 옛
# 기록과 짝지을 때 드러나게 한다. 글자 하나도 바꾸지 않아야 한다: 기록된 모든 측정(2026-09-19 e86aa05 이후)이
# 이 형식으로 쟀다.
def reminded_message(agent_text: str, known: str) -> str:
    """상담원의 말에 매 턴 역할 지침과 '당신이 아는 정보'를 붙인, 시뮬레이터 모델이 실제로 받는 글."""
    return f"{agent_text}\n\n{TURN_REMINDER}\n(당신이 아는 정보: {known})"


class SimulatorError(RuntimeError):
    """시뮬레이터 고객이 쓸 수 있는 답을 내지 못했다(빈 답). 에이전트의 실패가 아니다(infra_error).

    episode.py가 infra_error로 기록해 성공률의 분모에서 뺀다.
    """


# episode.py가 고객에게 기대하는 인터페이스. messages는 시뮬레이터 쪽 대화 기록으로, 역할이 뒤집혀 있다(상담원
# 말이 user, 고객 자신의 말이 assistant).
class User(Protocol):
    messages: list[Message]  # 시뮬레이터 쪽 대화 기록(역할이 뒤집힘)
    llm_log: list[LLMCallLog]

    def reply(self, agent_text: str) -> str:
        """상담원이 방금 한 말에 대한 고객의 답. 종료 토큰이 붙어 있을 수 있다."""
        ...


# 테스트용 고객. GPU 없이 에피소드 전체 경로를 시험할 수 있다.
class ScriptedUser:
    """모델 없이 준비된 대사를 순서대로 말하고, 대사가 떨어지면 그 뒤로는 계속 STOP_TOKEN만 낸다."""

    def __init__(self, lines: Sequence[str]):
        self._lines = list(lines)
        self.messages: list[Message] = []
        self.llm_log: list[LLMCallLog] = []

    def reply(self, agent_text: str) -> str:
        text = self._lines.pop(0) if self._lines else STOP_TOKEN
        self.messages.append(Message("user", agent_text))
        self.messages.append(Message("assistant", text))
        return text


# prompts/user_sim.md 템플릿에 시나리오(성격, 이유, 아는 것, 모르는 것, 행동 규칙)와 오늘 날짜(과제의 고정
# 시계, KST)를 채운다. 빈 항목은 "(없음)"으로 넣는다.
def build_user_prompt(task: Task) -> str:
    """시뮬레이터의 시스템 프롬프트. 시나리오만 넣고 고객 번호, 정답 동작, 전달 값은 넣지 않는다."""
    today = kst_date(task.now)
    scenario = task.user
    template = (files("support_agent") / "prompts" / "user_sim.md").read_text(encoding="utf-8")
    return template.format(
        persona=scenario.persona.strip() or _EMPTY_FIELD,
        reason=scenario.reason.strip(),
        known=scenario.known.strip() or _EMPTY_FIELD,
        unknown=scenario.unknown.strip() or _EMPTY_FIELD,
        rules=scenario.rules.strip() or _EMPTY_FIELD,
        today=f"{today.year}년 {today.month}월 {today.day}일",
    )


# 도구는 주지 않는다(빈 도구 목록).
class LLMUser:
    """고객을 연기하는 LLM. 상담원의 말은 `user` 역할로 들어오고, 자기 답은 `assistant`로 쌓인다."""

    def __init__(self, provider: ChatProvider, task: Task, config: RunConfig, trial: int):
        self._provider = provider
        self._task = task
        self._config = config
        self._trial = trial
        self.messages: list[Message] = [Message("system", build_user_prompt(task))]
        self.llm_log: list[LLMCallLog] = []

    def reply(self, agent_text: str) -> str:
        # seed는 (기본 seed, 과제, 시도, "user", 호출 순번)에서 만든다. 같은 명령은 같은 seed로 돌아 시도마다
        # 다른 대화가 재현된다(README "다시 재는 방법"). temperature는 설정의
        # user_temperature다(docs/experiments.md: 0.7에서 0.3으로 낮춤).
        index = len(self.llm_log)
        seed = derive_seed(self._config.base_seed, self._task.id, self._trial, "user", index)
        self.messages.append(Message("user", agent_text))
        # 7B 시뮬레이터는 몇 턴 지나면 규칙을 잊으므로, 가장 새 메시지에 짧은 리마인더를 붙인다. 리마인더에는
        # 시나리오의 known("당신이 아는 정보")도 함께 붙인다(reminded_message). 리마인더는 모델에 보내는
        # 마지막 메시지에만 붙이고, 기록(self.messages)에는 상담원의 말을 그대로 남긴다.
        reminded = [*self.messages[:-1], Message("user", reminded_message(agent_text, self._task.user.known))]
        response = self._provider.chat(
            reminded,
            (),
            temperature=self._config.user_temperature,
            seed=seed,
            max_tokens=USER_MAX_TOKENS,
        )
        text = response.text.strip()
        usage = response.usage
        # 에이전트 호출과 같은 형식으로 시뮬레이터 호출을 기록한다(토큰 수, 시간, 종료 이유). 빈 답은 형식
        # 오류 "empty"로 표시한다.
        self.llm_log.append(
            LLMCallLog(
                who="user",
                index=index,
                seed=seed,
                text=text,
                tool_calls=[
                    {"name": c.name, "arguments": c.arguments, "id": c.id} for c in response.tool_calls
                ],
                format_error=None if text else "empty",
                dropped_text="",
                dropped_calls=0,
                finish_reason=response.finish_reason,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                wall_ms=usage.wall_ms,
                load_ms=usage.load_ms,
                prompt_eval_ms=usage.prompt_eval_ms,
                eval_ms=usage.eval_ms,
            )
        )
        # 빈 답은 기록을 남긴 뒤 SimulatorError로 에피소드를 멈춘다. 대화 기록에는 넣지 않는다.
        if not text:
            raise SimulatorError(
                f"{self._task.id} trial {self._trial}: empty reply from the simulator (call {index})"
            )
        self.messages.append(Message("assistant", text))
        return text
