"""The customer side of an episode: an LLM that plays the scenario, and a scripted stand-in for tests."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.resources import files
from typing import Protocol

from support_agent.chat import ChatProvider, Message
from support_agent.clock import kst_date
from support_agent.config import STOP_TOKEN, RunConfig, derive_seed
from support_agent.records import LLMCallLog
from support_agent.tasks import Task

USER_MAX_TOKENS = 256
TURN_REMINDER = (
    "(고객 역할 지침: 위는 상담원의 말입니다. 당신은 고객입니다. 한국어로 한두 문장만 답하세요. "
    "<scenario>에 없는 번호·날짜·금액·주소는 말하지 마세요. "
    "상담원에게 기다리라고 하거나 상담원처럼 안내하지 마세요. "
    "상담원이 묻는 것 가운데 아는 것은 바로 알려 주고, 모르는 것은 모른다고 하세요. "
    "원하는 일이 아직 끝나지 않았으면 대화를 계속하세요. "
    "원하던 일이 모두 끝난 뒤에는 인사 대신 ###STOP### 만 출력하세요.)"
)
_EMPTY_FIELD = "(없음)"


class SimulatorError(RuntimeError):
    """The simulated customer produced nothing usable. Not the agent's failure (infra_error)."""


class User(Protocol):
    messages: list[Message]  # simulator-side history (role-flipped)
    llm_log: list[LLMCallLog]

    def reply(self, agent_text: str) -> str:
        """The customer's answer to what the agent just said; may carry a termination token."""
        ...


class ScriptedUser:
    """Says the prepared lines in order, then STOP_TOKEN forever."""

    def __init__(self, lines: Sequence[str]):
        self._lines = list(lines)
        self.messages: list[Message] = []
        self.llm_log: list[LLMCallLog] = []

    def reply(self, agent_text: str) -> str:
        text = self._lines.pop(0) if self._lines else STOP_TOKEN
        self.messages.append(Message("user", agent_text))
        self.messages.append(Message("assistant", text))
        return text


def build_user_prompt(task: Task) -> str:
    """The simulator's system prompt. Only the scenario goes in: no customer id, gold actions or values."""
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


class LLMUser:
    """An LLM playing the customer. The agent's words arrive as `user`, its own go back as `assistant`."""

    def __init__(self, provider: ChatProvider, task: Task, config: RunConfig, trial: int):
        self._provider = provider
        self._task = task
        self._config = config
        self._trial = trial
        self.messages: list[Message] = [Message("system", build_user_prompt(task))]
        self.llm_log: list[LLMCallLog] = []

    def reply(self, agent_text: str) -> str:
        index = len(self.llm_log)
        seed = derive_seed(self._config.base_seed, self._task.id, self._trial, "user", index)
        self.messages.append(Message("user", agent_text))
        # A 7B simulator forgets its rules after a few turns, so the newest message carries a short reminder.
        # The reminder is sent to the model only; the recorded history holds the agent's words as they were.
        reminder = f"{TURN_REMINDER}\n(당신이 아는 정보: {self._task.user.known})"
        reminded = [*self.messages[:-1], Message("user", f"{agent_text}\n\n{reminder}")]
        response = self._provider.chat(
            reminded,
            (),
            temperature=self._config.user_temperature,
            seed=seed,
            max_tokens=USER_MAX_TOKENS,
        )
        text = response.text.strip()
        usage = response.usage
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
        if not text:
            raise SimulatorError(
                f"{self._task.id} trial {self._trial}: empty reply from the simulator (call {index})"
            )
        self.messages.append(Message("assistant", text))
        return text
