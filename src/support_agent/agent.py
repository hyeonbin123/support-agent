"""The agent loop: one customer turn in, tool calls until the agent answers or the episode must stop."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from importlib.resources import files
from typing import Any

from support_agent.chat import ChatProvider, ChatResponse, Message, ToolCall
from support_agent.clock import to_kst
from support_agent.config import (
    FIRST_AGENT_MESSAGE,
    HANDOFF_MESSAGE,
    THINK_TOOL,
    RunConfig,
    Termination,
    derive_seed,
)
from support_agent.records import LLMCallLog, ToolCallLog
from support_agent.toolkit import Registry, ToolResult

STALL_NOTICE = (
    "[시스템 안내] 방금 응답은 고객에게 전달되지 않았습니다. "
    "하겠다는 말만 보내면 고객은 기다릴 수밖에 없습니다. "
    "지금 필요한 도구를 바로 호출하거나, 고객에게 물어볼 것이 있으면 그것을 물어보세요."
)
CUT_OFF_NOTICE = (
    "[시스템 안내] 방금 응답은 너무 길어 중간에 끊겼고 고객에게 전달되지 않았습니다. 더 짧게 다시 답하세요."
)
FORMAT_NOTICE = (
    "[시스템 안내] 방금 응답은 형식이 잘못되어 고객에게 전달되지 않았습니다. "
    "도구를 쓰려면 정해진 도구 호출 형식으로 호출하고, 아니면 고객에게 보낼 말을 일반 문장으로 답하세요."
)
LANGUAGE_NOTICE = (
    "[시스템 안내] 방금 응답은 한국어가 아니어서 고객에게 전달되지 않았습니다. "
    "같은 내용을 한국어로만 다시 답하세요."
)
_WEEKDAYS = "월화수목금토일"

# The loop never sees the DB or the ToolContext; the caller passes a closure over toolkit.execute.
RunTool = Callable[[str, dict], ToolResult]


@dataclass
class AgentState:
    messages: list[Message]
    agent_calls: int = 0  # LLM calls so far in this episode
    tool_errors: int = 0
    format_errors: int = 0
    dropped_calls: int = 0
    stalls: int = 0  # replies held back by the stall guard (G1)
    tool_log: list[ToolCallLog] = field(default_factory=list)
    llm_log: list[LLMCallLog] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [m.to_dict() for m in self.messages],
            "agent_calls": self.agent_calls,
            "tool_errors": self.tool_errors,
            "format_errors": self.format_errors,
            "dropped_calls": self.dropped_calls,
            "stalls": self.stalls,
        }


@dataclass(frozen=True)
class TurnResult:
    reply: str | None  # what the customer is told; None when the episode stops without a reply
    stop: Termination | None  # None = the conversation goes on


def _prompt_file(name: str) -> str:
    return (files("support_agent") / "prompts" / name).read_text(encoding="utf-8")


def load_policy() -> str:
    return _prompt_file("policy.md")


def build_system_prompt(policy_text: str, now: datetime) -> str:
    local = to_kst(now)
    now_text = f"{local:%Y-%m-%d %H:%M} ({_WEEKDAYS[local.weekday()]}요일)"
    return _prompt_file("agent.md").format(policy=policy_text.strip(), now=now_text)


def new_state(system_prompt: str) -> AgentState:
    return AgentState([Message("system", system_prompt), Message("assistant", FIRST_AGENT_MESSAGE)])


def visible_tools(registry: Registry, config: RunConfig) -> list[dict[str, Any]]:
    """Tool schemas shown to the model. The think tool exists only under R1."""
    return [
        spec.schema() for spec in registry.values() if spec.name != THINK_TOOL or config.reasoning == "R1"
    ]


def _looks_like_tool_call(text: str) -> bool:
    """True when the text holds a JSON object shaped like a tool call (qwen2.5 sometimes writes the call
    into the message, with or without a code fence or a lead-in sentence)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return False
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return False
    return isinstance(data, dict) and "name" in data and ("arguments" in data or "parameters" in data)


def leaked_tool_call(text: str) -> ToolCall | None:
    """The tool call that the model wrote into its message, if the text holds exactly such a JSON object."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return None
    if not (isinstance(data, dict) and isinstance(data.get("name"), str)):
        return None
    arguments = data.get("arguments", data.get("parameters"))
    return ToolCall(data["name"], arguments) if isinstance(arguments, dict) else None


_PROMISE = re.compile(
    r"(확인|조회|처리|접수|진행|발급|변경|취소|연결|검토)\s*(해|하)\s*(보|드리)?\s*겠습니다|잠시만|기다려\s*주"
)
_ASKS = re.compile(r"[?？]|알려\s*주|말씀해\s*주|불러\s*주|입력해\s*주")


def is_stall(text: str) -> bool:
    """A reply that only promises to do something ("확인해 보겠습니다, 잠시만 기다려 주세요") and asks the
    customer nothing. Delivered as it is, the customer can only say "네" and the turn is wasted."""
    return bool(_PROMISE.search(text)) and not _ASKS.search(text)


_OTHER_SCRIPT = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")  # Han characters, kana


def in_another_language(text: str) -> bool:
    """qwen2.5 sometimes drifts into Chinese in the middle of a Korean conversation. One stray character
    (a name, a symbol) is let through; two or more mean the sentence is not Korean."""
    return len(_OTHER_SCRIPT.findall(text)) >= 2


def format_problem(response: ChatResponse) -> str | None:
    """Why a reply cannot be used as it is: empty | leaked_tool_call | cut_off, or None."""
    text = response.text.strip()
    if not text and not response.tool_calls:
        return "empty"
    if not response.tool_calls and (
        "<tool_call>" in text or "</tool_call>" in text or _looks_like_tool_call(text)
    ):
        return "leaked_tool_call"  # the model wrote the call into the text instead of calling
    if response.finish_reason == "length":
        return "cut_off"
    return None


def agent_turn(
    state: AgentState,
    user_text: str,
    *,
    provider: ChatProvider,
    registry: Registry,
    run_tool: RunTool,
    config: RunConfig,
    task_id: str = "",
    trial: int = 0,
) -> TurnResult:
    """Handle one customer message. ProviderError and ToolBugError pass through to the episode runner."""
    state.messages.append(Message("user", user_text))
    tools = visible_tools(registry, config)
    retries = 0  # format retries are counted per turn
    stall_retries = 0
    while state.agent_calls < config.max_agent_calls:
        index = state.agent_calls
        seed = derive_seed(config.base_seed, task_id, trial, "agent", index)
        # G2: a retry after a held-back stall is sampled, because at temperature 0 the model tends to
        # answer the notice with the very same sentence.
        retrying_stall = config.guard == "G2" and stall_retries > 0
        temperature = config.stall_retry_temperature if retrying_stall else config.temperature
        response = provider.chat(
            state.messages, tools, temperature=temperature, seed=seed, max_tokens=config.max_tokens
        )
        state.agent_calls += 1
        problem = format_problem(response)
        calls = response.tool_calls
        if problem == "leaked_tool_call" and config.rescue == "F1":
            rescued = leaked_tool_call(response.text)
            if rescued is not None:  # run it as if it had been a proper call; still counted as a format error
                calls, problem = (rescued,), None
                state.format_errors += 1
        if not problem and not calls and config.language == "L1" and in_another_language(response.text):
            problem = "wrong_language"
        if (
            not problem
            and not calls
            and config.guard in ("G1", "G2")
            and stall_retries < config.max_stall_retries
        ):
            if is_stall(response.text):
                problem = "stall"
        usage = response.usage
        state.llm_log.append(
            LLMCallLog(
                who="agent",
                index=index,
                seed=seed,
                text=response.text,
                tool_calls=[{"name": c.name, "arguments": c.arguments, "id": c.id} for c in calls],
                format_error=problem or ("rescued_tool_call" if calls and not response.tool_calls else None),
                dropped_text=response.text if calls and not problem else "",
                dropped_calls=max(len(calls) - 1, 0) if not problem else 0,
                finish_reason=response.finish_reason,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                wall_ms=usage.wall_ms,
                load_ms=usage.load_ms,
                prompt_eval_ms=usage.prompt_eval_ms,
                eval_ms=usage.eval_ms,
            )
        )
        # Ollama silently cuts the front of an overlong prompt, so stop before that happens.
        if usage.prompt_tokens > 0.95 * config.num_ctx:
            return TurnResult(None, "context_limit")

        if problem == "stall":
            # Not a format error: the reply is well formed, it just does nothing. Same remedy, own budget.
            stall_retries += 1
            state.stalls += 1
            state.messages.append(Message("assistant", response.text, delivered=False))
            state.messages.append(Message("user", STALL_NOTICE, harness=True))
            continue
        if problem:
            state.format_errors += 1
            state.messages.append(Message("assistant", response.text, delivered=False))
            if retries >= config.max_format_retries:
                return TurnResult(None, "agent_format_error")
            retries += 1
            notice = {"cut_off": CUT_OFF_NOTICE, "wrong_language": LANGUAGE_NOTICE}.get(
                problem, FORMAT_NOTICE
            )
            state.messages.append(Message("user", notice, harness=True))
            continue

        if not calls:
            state.messages.append(Message("assistant", response.text))
            return TurnResult(response.text, None)

        # Only the first call runs; text that came with it is not delivered and not kept in history.
        call = calls[0]
        state.dropped_calls += len(calls) - 1
        state.messages.append(Message("assistant", "", (call,)))
        spec = registry.get(call.name)
        started = time.perf_counter()
        result = run_tool(call.name, call.arguments)
        state.tool_log.append(
            ToolCallLog(
                agent_call=index,
                name=call.name,
                raw_arguments=call.arguments,
                args=result.args,
                ok=result.ok,
                error_code=result.error_code,
                policy_blocked=result.policy_blocked,
                violations=list(result.violations),
                content=result.content,
                write=spec.write if spec else False,
                ms=(time.perf_counter() - started) * 1000,
            )
        )
        state.messages.append(
            Message("tool", result.content, tool_name=call.name, tool_call_id=call.id or None)
        )
        if not result.ok:
            state.tool_errors += 1
            if state.tool_errors >= config.max_tool_errors:
                return TurnResult(None, "too_many_tool_errors")
        elif spec is not None and spec.terminates:
            state.messages.append(Message("assistant", HANDOFF_MESSAGE))
            return TurnResult(HANDOFF_MESSAGE, "handoff")
    return TurnResult(None, "max_agent_calls")
