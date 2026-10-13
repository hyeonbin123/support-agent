"""The agent loop: one customer turn in, tool calls until the agent answers or the episode must stop."""

# [파일 역할] 에이전트 루프. 고객 메시지 하나를 받아, 모델이 고객에게 줄 답을 낼 때까지(또는 에피소드를 끝내야
# 할 때까지) LLM 호출 → 도구 실행 → 결과를 대화에 넣기를 되풀이한다.
# [설계 이유] LangChain 같은 프레임워크 없이 직접 구현했다. 가드(형식 오류·멈춤·언어·완료 주장), 사람 승인,
# 추적 로그를 루프 안에 넣어야 하고 분량이 작기 때문이다(docs/plan.md 구조). 루프는 DB와 ToolContext를 직접
# 보지 않고 호출자가 넘긴 run_tool만 부른다. 그래서 같은 루프가 평가(episode.py: 메모리 SQLite, R2 확인
# 관문)와 서비스(service/core.py: PostgreSQL, 승인 관문·감사 로그)에서 그대로 돈다.
# [루프 규칙] docs/design.md '에이전트 루프 규칙' 표와 같다: 본문과 도구 호출이 함께 오면 도구 호출만 따르고,
# 도구 호출이 여럿이면 첫 번째만 실행하며, 형식 오류는 고객에게 보내지 않고 안내문을 넣어 한 턴에 2번까지 다시
# 시킨다. 한도(LLM 호출 30, 도구 오류 10, 컨텍스트 95%)에 닿으면 그 이유로 끝낸다.
# [연결] episode.py와 service/core.py가 agent_turn·new_state·build_system_prompt를 부르고, run.py는 manifest의
# 프롬프트·도구 해시를 만들 때 build_system_prompt·visible_tools를 쓴다. in_another_language는
# analyze.py·sim_report.py가 한자 섞인 답을 셀 때도 쓴다.

from __future__ import annotations

# importlib.resources.files: 프롬프트 파일(prompts/*.md)을 패키지 데이터로 읽는다. 설치 위치나 현재 작업
# 폴더와 상관없이 같은 파일을 찾는다.
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from importlib.resources import files
from typing import Any

# 내부 모듈: 제공자 중립 대화 타입(chat), 완료 주장 검출기(claims, 7단계), KST 변환(clock), 설정·고정
# 문자열·seed(config), R2 미리보기 코드(confirm, 10단계), 기록 타입(records), 도구 등록부와 결과
# 타입(toolkit).
from support_agent.chat import ChatProvider, ChatResponse, Message, ToolCall
from support_agent.claims import unbacked_claim
from support_agent.clock import to_kst
from support_agent.config import (
    FIRST_AGENT_MESSAGE,
    HANDOFF_MESSAGE,
    THINK_TOOL,
    RunConfig,
    Termination,
    derive_seed,
)
from support_agent.confirm import CONFIRM_CODE
from support_agent.records import LLMCallLog, ToolCallLog
from support_agent.toolkit import Registry, ToolResult

# 하니스가 다시 시킬 때 대화에 넣는 안내문들(harness=True인 user 메시지가 된다). 오류 메시지와 마찬가지로
# 이유를 알리고 다시 답하게 하는 것이 목적이다.
# STALL_NOTICE: 멈춤 가드(G1·G2). 도구 호출 없이 '확인해 보겠습니다'만 하는 답을 돌려보낼 때.
STALL_NOTICE = (
    "[시스템 안내] 방금 응답은 고객에게 전달되지 않았습니다. "
    "하겠다는 말만 보내면 고객은 기다릴 수밖에 없습니다. "
    "지금 필요한 도구를 바로 호출하거나, 고객에게 물어볼 것이 있으면 그것을 물어보세요."
)
# CUT_OFF_NOTICE: max_tokens에 잘린 답. FORMAT_NOTICE: 빈 답이나 본문에 쓴 도구 호출. LANGUAGE_NOTICE:
# 중국어·일본어 문자로 넘어간 답(L1).
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
# CLAIM_NOTICE: 완료 주장 가드(C1·C2). 무엇을 끝냈다고 말했는지(label)와 그 일을 하는 도구 이름(tools)을
# claims.py가 채운다. 7단계 결과로는 7B가 이 안내를 받고도 도구를 부르지 않았다(README 7단계).
CLAIM_NOTICE = (
    "[시스템 안내] 방금 응답은 고객에게 전달되지 않았습니다. {label}에 대해 이미 끝난 것처럼 말했지만, "
    "이 대화의 도구 결과에는 그것을 보여 주는 기록이 없습니다. 실제로 처리하려면 지금 {tools} 도구를 "
    "호출하세요. 처리하지 않았거나 처리할 수 없는 일이면 끝났다고 말하지 말고 사실대로 안내하세요."
)
# 시스템 프롬프트의 '오늘' 표기에 붙일 요일. datetime.weekday()가 월요일=0이라 이 순서다.
_WEEKDAYS = "월화수목금토일"

# RunTool: (도구 이름, 인자) → ToolResult. 평가는 toolkit.execute(또는 R2 관문)를, 서비스는 승인 관문과 감사
# 행을 거치는 함수를 넘긴다.
# The loop never sees the DB or the ToolContext; the caller passes a closure over toolkit.execute.
RunTool = Callable[[str, dict], ToolResult]


# 한 대화(에피소드 또는 서비스 세션)의 에이전트 쪽 상태. messages는 모델이 보는 전체 기록(시스템 프롬프트,
# 도구 결과, 하니스 안내 포함)이고, 나머지는 한도 검사와 보고용 카운터와 로그다.
# 서비스는 요청 사이에 이 상태를 to_dict()로 DB에 저장했다가 되살린다(service/core.py). to_dict에는
# tool_log·llm_log가 들어가지 않는다.
@dataclass
class AgentState:
    messages: list[Message]
    agent_calls: int = 0  # LLM calls so far in this episode
    tool_errors: int = 0
    format_errors: int = 0
    dropped_calls: int = 0
    stalls: int = 0  # replies held back by the stall guard (G1)
    held_claims: int = 0  # replies held back by the claim guard (C1)
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
            "held_claims": self.held_claims,
        }


# agent_turn의 결과. reply는 고객에게 전달할 말, stop은 에피소드를 끝내야 하는 이유(config.Termination). 둘 다
# 있으면(이관) 안내문을 전달하고 끝낸다.
@dataclass(frozen=True)
class TurnResult:
    reply: str | None  # what the customer is told; None when the episode stops without a reply
    stop: Termination | None  # None = the conversation goes on


# 패키지 안 prompts/ 폴더의 파일을 UTF-8로 읽는다.
def _prompt_file(name: str) -> str:
    return (files("support_agent") / "prompts" / name).read_text(encoding="utf-8")


# 규정 문서(policy.md) 원문. 시스템 프롬프트에 그대로 들어가고, 과제 검증은 전달 값이 이 문서(예시 포함)에
# 새어 있지 않은지 검사한다.
def load_policy() -> str:
    return _prompt_file("policy.md")


# 에이전트 시스템 프롬프트 = agent.md 틀에 규정 문서, 현재 시각(KST, 요일), 음성 V4 규칙 한 줄을 끼운 것.
# now는 과제의 고정 시각(평가) 또는 서비스 설정의 시계라, 같은 과제는 언제 돌려도 같은 프롬프트가 된다.
# caller_id가 아니면 빈 문자열을 넣어 9단계 이전과 글자까지 같은 프롬프트를 만든다. manifest에 프롬프트 해시가
# 남으므로, 이렇게 해야 이전 기록과 짝지을 수 있다.
def build_system_prompt(policy_text: str, now: datetime, *, caller_id: bool = False) -> str:
    """The agent's system prompt. caller_id (voice V4) adds one rule line: identify by the caller's number and
    the name. Without it the text is the same as before stage 9 (the manifests' hashes)."""
    local = to_kst(now)
    now_text = f"{local:%Y-%m-%d %H:%M} ({_WEEKDAYS[local.weekday()]}요일)"
    caller_rule = "\n" + _prompt_file("caller_id.md").strip() if caller_id else ""
    return _prompt_file("agent.md").format(policy=policy_text.strip(), now=now_text, caller_rule=caller_rule)


# 새 대화의 상태. 시스템 프롬프트와 고정 첫 인사로 시작한다. 첫 인사는 모델을 부르지 않고 넣는다.
def new_state(system_prompt: str) -> AgentState:
    return AgentState([Message("system", system_prompt), Message("assistant", FIRST_AGENT_MESSAGE)])


# 모델에게 보여 줄 도구 스키마 목록. think 도구는 R1에서만 보인다. 등록부는 구성과 상관없이 같고, 보이는
# 목록만 구성에 따라 달라진다. run.py는 이 목록의 해시를 manifest에 남긴다.
def visible_tools(registry: Registry, config: RunConfig) -> list[dict[str, Any]]:
    """Tool schemas shown to the model. The think tool exists only under R1."""
    return [
        spec.schema() for spec in registry.values() if spec.name != THINK_TOOL or config.reasoning == "R1"
    ]


# 본문에 도구 호출 모양의 JSON이 들어 있는지 본다. 첫 '{'부터 마지막 '}'까지를 JSON으로 읽어 name과
# arguments(또는 parameters)가 있으면 샌 도구 호출로 본다. 코드 블록이나 앞 문장이 붙어 있어도 잡힌다.
# 형식 오류 판정(format_problem)에 쓰인다. 기준 구성은 이것을 찾기만 하고 실행하지 않는다(docs/experiments.md
# 공통 규칙 에이전트 루프).
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


# F1(본문에 샌 도구 호출 구제, 2단계 후보)에서 쓰는 해석기. 본문의 JSON을 ToolCall로 바꿔 돌려주고, 이름이
# 문자열이 아니거나 인자가 dict가 아니면 None이다. 구제해서 실행해도 형식 오류로는 그대로 센다.
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


# 멈춤 판정 정규식(2단계 G1. 개발용 기준선 에피소드만 보고 정했다, docs/experiments.md 2단계 후보 표).
# _PROMISE: '확인/조회/처리… + 하겠습니다·해 보겠습니다·해 드리겠습니다', '잠시만', '기다려 주'. _ASKS:
# 물음표나 '알려 주/말씀해 주' 같은 요청이 있으면 고객에게 묻는 답이므로 멈춤이 아니다.
_PROMISE = re.compile(
    r"(확인|조회|처리|접수|진행|발급|변경|취소|연결|검토)\s*(해|하)\s*(보|드리)?\s*겠습니다|잠시만|기다려\s*주"
)
_ASKS = re.compile(r"[?？]|알려\s*주|말씀해\s*주|불러\s*주|입력해\s*주")


# 하겠다는 말만 하고 고객에게 묻는 것도 없는 답인지. 이런 답이 고객에게 가면 고객은 '네'라고만 할 수 있어 한
# 턴이 버려진다. 1단계 실패 분류에서 멈춤 31건 + 쓰기 없이 끝남 19건이 실패 96건 중 50건이라 G1 후보가 됐다.
# 결과는 개선 없음이었다(README 2단계).
def is_stall(text: str) -> bool:
    """A reply that only promises to do something ("확인해 보겠습니다, 잠시만 기다려 주세요") and asks the
    customer nothing. Delivered as it is, the customer can only say "네" and the turn is wasted."""
    return bool(_PROMISE.search(text)) and not _ASKS.search(text)


# 한자(CJK 통합 한자)와 일본어 가나의 유니코드 범위.
_OTHER_SCRIPT = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")  # Han characters, kana


# 답이 중국어·일본어 문자로 넘어갔는지. 이름이나 기호에 한 글자 섞이는 것은 넘기고 두 글자 이상이면 한국어
# 답이 아니라고 본다. 서비스는 L1로 이 검사를 켠다(측정은 하지 않음, config.py language 주석).
def in_another_language(text: str) -> bool:
    """qwen2.5 sometimes drifts into Chinese in the middle of a Korean conversation. One stray character
    (a name, a symbol) is let through; two or more mean the sentence is not Korean."""
    return len(_OTHER_SCRIPT.findall(text)) >= 2


# 답을 그대로 쓸 수 없는 형식상의 이유를 돌려준다: empty(본문도 도구 호출도 없음), leaked_tool_call(도구
# 호출을 본문에 씀), cut_off(길이 제한으로 잘림). 문제가 없으면 None.
# 검사 순서가 의미가 있다. 도구 호출이 있으면 본문에 무엇이 있든 샌 호출로 보지 않는다(본문은 어차피 버린다).
def format_problem(response: ChatResponse) -> str | None:
    """Why a reply cannot be used as it is: empty | leaked_tool_call | cut_off, or None."""
    text = response.text.strip()
    if not text and not response.tool_calls:
        return "empty"
    # qwen2.5가 쓰는 <tool_call> 태그가 본문에 보이거나, 태그 없이 호출 모양 JSON만 있을 때.
    if not response.tool_calls and (
        "<tool_call>" in text or "</tool_call>" in text or _looks_like_tool_call(text)
    ):
        return "leaked_tool_call"  # the model wrote the call into the text instead of calling
    if response.finish_reason == "length":
        return "cut_off"
    return None


# 고객 메시지 하나를 처리한다. 고객 메시지를 기록에 넣고, LLM을 불러 답이 나올 때까지 도는 루프다.
# 흐름: LLM 호출 → (F1 구제) → 언어·완료 주장·멈춤 검사 → 호출 로그 → 컨텍스트 한도 검사 → 문제가 있으면
# 안내문을 넣고 다시 → 도구 호출이 없으면 답을 전달하고 턴 끝 → 있으면 첫 호출만 실행하고 결과를 넣은 뒤 다시.
# 재시도 예산은 턴마다 따로다(형식 오류, 멈춤, 완료 주장 각각). LLM 호출 수(agent_calls)는 에피소드 전체로
# 센다.
# 모델 서버 오류(ProviderError)와 도구 버그(ToolBugError)는 잡지 않고 위로 올린다. 에피소드 실행기가
# infra_error로 기록한다.
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
    claim_retries = 0
    # LLM 호출 한도에 닿을 때까지 돈다. 평가에서는 에피소드당 30번이다. 서비스는 상태를 되살릴 때
    # agent_calls를 0부터 다시 세므로 턴당 한도가 된다(service/core.py).
    while state.agent_calls < config.max_agent_calls:
        index = state.agent_calls
        # 호출마다 (과제, 시도, 'agent', 순번)에서 seed를 만든다. temperature 0에서는 의미가 없고, G2·C2의
        # 표본 추출 때 쓰인다.
        seed = derive_seed(config.base_seed, task_id, trial, "agent", index)
        # G2: a retry after a held-back stall is sampled, because at temperature 0 the model tends to
        # answer the notice with the very same sentence. C2 does the same after a held-back claim.
        temperature = config.temperature
        if config.guard == "G2" and stall_retries > 0:
            temperature = config.stall_retry_temperature
        if config.claims == "C2" and claim_retries > 0:
            temperature = config.claim_retry_temperature
        # 대화 전체와 보이는 도구 목록을 보내 한 턴을 받는다.
        response = provider.chat(
            state.messages, tools, temperature=temperature, seed=seed, max_tokens=config.max_tokens
        )
        state.agent_calls += 1
        problem = format_problem(response)
        calls = response.tool_calls
        # F1: 본문에 샌 호출을 해석할 수 있으면 정식 호출처럼 실행한다. 형식 오류 수는 그래도 하나 늘린다(기준
        # 구성과 비교할 수 있게).
        if problem == "leaked_tool_call" and config.rescue == "F1":
            rescued = leaked_tool_call(response.text)
            if rescued is not None:  # run it as if it had been a proper call; still counted as a format error
                calls, problem = (rescued,), None
                state.format_errors += 1
        # 아래 세 검사(언어, 완료 주장, 멈춤)는 도구 호출 없이 고객에게 갈 본문 답에만 한다. 도구 호출이
        # 있으면 본문은 어차피 전달되지 않는다.
        if not problem and not calls and config.language == "L1" and in_another_language(response.text):
            problem = "wrong_language"
        claim = None
        # C1·C2: 끝났다고 말하는데 이 대화의 성공한 도구 결과에 근거가 없으면 claims.py가 그 주장을 돌려준다.
        if not problem and not calls and config.claims in ("C1", "C2"):
            claim = unbacked_claim(response.text, state.messages)
            if claim:
                problem = "unbacked_claim"
        # G1·G2: 멈춤 재시도 예산(턴당 2번)이 남아 있을 때만 검사한다. 예산을 다 쓰면 같은 답이라도 그대로
        # 전달한다(docs/experiments.md 2단계 G1 정의).
        if (
            not problem
            and not calls
            and config.guard in ("G1", "G2")
            and stall_retries < config.max_stall_retries
        ):
            if is_stall(response.text):
                problem = "stall"
        usage = response.usage
        # 모든 LLM 호출을 기록한다. format_error는 문제의 이유, 구제한 호출이면 rescued_tool_call이다.
        # dropped_text는 도구 호출과 함께 와서 전달하지 않은 본문, dropped_calls는 실행하지 않은 두 번째 이후
        # 호출 수다.
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
        # Ollama는 프롬프트가 num_ctx를 넘으면 오류 없이 앞부분(시스템 프롬프트 포함)을 잘라 버린다. 0단계에서
        # prompt_eval_count가 캐시된 토큰까지 포함한 전체 길이임을 확인했으므로 이 값으로 95%를 넘기 전에
        # 멈춘다(docs/experiments.md 0단계).
        if usage.prompt_tokens > 0.95 * config.num_ctx:
            return TurnResult(None, "context_limit")

        # 멈춤: 형식은 맞으므로 형식 오류로 세지 않고 자기 예산(stall_retries)을 쓴다. 답은 전달하지
        # 않음(delivered=False)으로 남기고 안내문을 넣어 다시 시킨다.
        if problem == "stall":
            # Not a format error: the reply is well formed, it just does nothing. Same remedy, own budget.
            stall_retries += 1
            state.stalls += 1
            state.messages.append(Message("assistant", response.text, delivered=False))
            state.messages.append(Message("user", STALL_NOTICE, harness=True))
            continue
        # 근거 없는 완료 주장: 이것도 형식 오류가 아니다. 멈춤과 달리 예산을 다 써도 전달하지 않고
        # unbacked_claim으로 끝낸다. 고객이 '취소되었습니다'를 들으면 더 묻지 않기 때문이다.
        # max_claim_retries=2면 세 번째 주장에서 끝난다.
        if claim:
            # Not a format error either. Unlike a stall it is never delivered: when the model keeps saying
            # it, the episode ends, because a customer who hears "취소되었습니다" stops asking.
            state.held_claims += 1
            state.messages.append(Message("assistant", response.text, delivered=False))
            if claim_retries >= config.max_claim_retries:
                return TurnResult(None, "unbacked_claim")
            claim_retries += 1
            notice = CLAIM_NOTICE.format(label=claim.label, tools=claim.tools)
            state.messages.append(Message("user", notice, harness=True))
            continue
        # 형식 오류(empty, leaked_tool_call, cut_off, wrong_language): 전달하지 않은 답과 알맞은 안내문을
        # 기록에 넣고 다시 시킨다. 재시도 2번을 넘으면 agent_format_error로 끝낸다.
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

        # 문제없는 본문 답: 고객에게 전달하고 이 턴을 끝낸다.
        if not calls:
            state.messages.append(Message("assistant", response.text))
            return TurnResult(response.text, None)

        # Only the first call runs; text that came with it is not delivered and not kept in history.
        # 도구 호출이 있으면 첫 번째만 실행하고, 대화 기록에도 그 호출 하나만 빈 본문으로 남긴다. 함께 온
        # 본문은 고객에게 보내지 않고 기록에서도 뺀다(LLM 로그의 dropped_text에만 남음, docs/design.md
        # 에이전트 루프 규칙).
        call = calls[0]
        state.dropped_calls += len(calls) - 1
        state.messages.append(Message("assistant", "", (call,)))
        spec = registry.get(call.name)
        started = time.perf_counter()
        # 실제 실행은 호출자가 넘긴 run_tool이 한다. 결과는 성공이든 오류든 문자열(content)로 모델에게
        # 돌아간다.
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
        # 도구 결과 메시지. Ollama는 도구 이름으로, OpenAI식 API는 호출 id로 결과를 짝지으므로 둘 다
        # 넣는다(id가 없으면 None).
        state.messages.append(
            Message("tool", result.content, tool_name=call.name, tool_call_id=call.id or None)
        )
        # R2의 미리보기는 고객의 답을 기다리는 정상 단계라 도구 오류 한도에 세지 않는다(docs/experiments.md
        # 10단계).
        if result.error_code == CONFIRM_CODE:
            continue  # R2: a preview that waits for the customer's answer, not a tool error
        # 도구 오류는 누적해 max_tool_errors(평가는 에피소드당 10)에 닿으면 too_many_tool_errors로 끝낸다.
        if not result.ok:
            state.tool_errors += 1
            if state.tool_errors >= config.max_tool_errors:
                return TurnResult(None, "too_many_tool_errors")
        # 대화를 끝내는 도구(transfer_to_human)가 성공하면 고정 이관 안내문을 고객에게 전달한 것으로 치고
        # handoff로 끝낸다. 판정은 이 안내문도 전달된 말로 본다.
        elif spec is not None and spec.terminates:
            state.messages.append(Message("assistant", HANDOFF_MESSAGE))
            return TurnResult(HANDOFF_MESSAGE, "handoff")
    return TurnResult(None, "max_agent_calls")
