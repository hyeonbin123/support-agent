"""제공자 중립 대화 타입, ChatProvider 인터페이스, 테스트용 대본 재생 제공자."""

# [파일 역할] LLM 제공자(Ollama 등)와 무관한 대화 타입(Message, ToolCall, ChatResponse, Usage)과 제공자
# 인터페이스(ChatProvider), 테스트용 대본 재생 제공자(ScriptedProvider)를 정의한다.
# [왜 필요한가] 에이전트 루프·시뮬레이터·판정·서비스가 특정 API의 응답 형식에 묶이지 않게 하려는 것이다.
# 제공자마다 다른 JSON(도구 호출을 이름으로 보내는지 id로 보내는지 등)은 각 제공자 클래스가 이 타입으로 바꿔
# 준다. 계획에 있던 OpenAI 호환·Anthropic 연결은 하지 않았지만 붙일 자리는 이 인터페이스로 남아
# 있다(docs/plan.md 계획과 달라진 것).
# [설계 이유] GPU 없이 루프·환경·판정을 결정적으로 테스트하려고 ScriptedProvider를 같은 인터페이스로
# 둔다(docs/plan.md 구조). 다른 모듈이 모두 기대는 계약 파일이라 고치면 전체 테스트를 돌린다.
# [연결] ollama.py(OllamaProvider)가 구현하고,
# agent.py·user_sim.py·episode.py·service/core.py·mcp_server.py·analyze.py가 쓴다.

from __future__ import annotations

# abc: ChatProvider를 추상 클래스로 두어 chat()을 구현하지 않은 제공자는 만들 수 없게 한다.
# dataclasses: 메시지·응답을 불변(frozen) 값 객체로 둬, 기록에 남긴 대화가 나중에 바뀌지 않게 한다.
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

# 대화 역할 네 가지. tool은 도구 실행 결과를 모델에 돌려주는 메시지다.
Role = Literal["system", "user", "assistant", "tool"]


# 모델이 요청한 도구 호출 하나. arguments는 이미 dict로 바뀐 인자다. 문자열로 보내는 제공자의 JSON을 해석하지
# 못하면 {"_raw": 원문}으로 남긴다(ollama.py). 도구 인자 모델은 모르는 필드를 금지하므로(toolkit.py
# extra="forbid") 이런 호출은 invalid_arguments 오류로 모델에 돌아간다.
@dataclass(frozen=True)
class ToolCall:
    name: str
    # 인자를 JSON 문자열로 보내는 제공자는 그것을 해석해 넣는다. 해석할 수 없는 글은 {"_raw": 글}이 된다.
    arguments: dict[str, Any]
    id: str = ""  # 제공자가 id를 보내지 않으면 ""


# 대화의 메시지 한 개. 평가 기록(episodes.jsonl)과 서비스 세션 저장이 모두 이 형태다.
# delivered와 harness는 이 프로젝트가 덧붙인 표시다. delivered=False는 형식 오류·멈춤·근거 없는 완료 주장
# 때문에 고객에게 보내지 않은 상담원 답이고, harness=True는 하니스가 다시 시키려고 넣은 안내문(고객의 말이
# 아님)이다. 판정(judge.py)은 delivered인 답에서만 전달 값을 찾고, confirm.py는 harness가 아닌 고객의 말만
# 동의 판정에 쓴다.
@dataclass(frozen=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()  # assistant 전용
    tool_name: str | None = None  # tool 전용(Ollama 고유 API는 이름으로 짝짓는다)
    tool_call_id: str | None = None  # tool 전용(OpenAI식·Anthropic API는 id로 짝짓는다)
    delivered: bool = True  # assistant 전용: 글이 고객에게 가지 않았으면 False
    harness: bool = False  # user 전용: 고객이 아니라 하니스가 쓴 글이면 True

    # 기록용 JSON으로 바꾼다. 기본값인 필드(빈 tool_calls, delivered=True, harness=False 등)는 적지 않아
    # 기록을 짧게 유지하고, 표시가 붙은 경우만 키가 생긴다.
    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [
                {"name": c.name, "arguments": c.arguments, "id": c.id} for c in self.tool_calls
            ]
        if self.tool_name:
            out["tool_name"] = self.tool_name
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        if not self.delivered:
            out["delivered"] = False
        if self.harness:
            out["harness"] = True
        return out

    # to_dict의 역함수. 서비스(service/core.py)는 요청 사이에 대화를 JSON으로 DB에 두었다가 이것으로 되살리고,
    # analyze.py는 기록된 에피소드를 다시 읽을 때 쓴다. 빠진 키는 to_dict에서 생략한 기본값으로 채운다.
    @staticmethod
    def from_dict(data: dict[str, Any]) -> Message:
        """to_dict의 역함수(서비스는 요청 사이에 대화를 JSON으로 보관한다)."""
        calls = tuple(
            ToolCall(c["name"], c["arguments"], c.get("id", "")) for c in data.get("tool_calls", ())
        )
        return Message(
            data["role"],
            data.get("content", ""),
            calls,
            data.get("tool_name"),
            data.get("tool_call_id"),
            data.get("delivered", True),
            data.get("harness", False),
        )


# LLM 호출 한 번의 사용량과 시간. 토큰 수는 컨텍스트 한도 검사(prompt_tokens가 num_ctx의 95%를 넘는지)에, 시간
# 값들은 에이전트 호출당 시간 같은 보조 지표에 쓰인다. wall_ms는 클라이언트가 HTTP 호출 앞뒤로 잰 시간,
# 나머지는 서버가 보고한 값이다.
@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_ms: float = 0.0  # 클라이언트가 HTTP 호출 앞뒤로 잰 시간
    load_ms: float = 0.0  # 서버가 보고한 모델 적재 시간
    prompt_eval_ms: float = 0.0
    eval_ms: float = 0.0


# 제공자가 돌려주는 상담원(또는 시뮬레이터) 턴 하나. 본문과 도구 호출이 함께 올 수 있고, 그때 어느 쪽을
# 따를지는 agent.py가 정한다(도구 호출을 따르고 본문은 버림).
# finish_reason이 "length"면 max_tokens에 잘린 답이라 형식 오류로 처리된다. thinking은 사고 모드 모델의 추론
# 글로, 다음 요청에 다시 넣지 않는다.
@dataclass(frozen=True)
class ChatResponse:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None  # "length"면 답이 잘렸다는 뜻
    thinking: str = ""  # 사고 모드 모델의 추론 글. 다음 요청에 다시 보내지 않는다


# 모델 서버의 실패(시간 초과, HTTP 오류, 깨진 JSON). 에이전트의 잘못이 아니므로 episode.py가 잡아
# infra_error로 기록하고 분모에서 뺀다. 서비스(service/core.py)는 다른 예외처럼 대체 안내문으로 답하되 이
# 오류는 트레이스백 없이 기록한다.
class ProviderError(RuntimeError):
    """모델 서버가 실패했다(시간 초과, HTTP 오류, 깨진 JSON). 에이전트의 잘못이 아니다."""


# 프롬프트가 모델의 컨텍스트(num_ctx)보다 길어 서버가 요청을 거절했다는 오류. ProviderError의 하위
# 클래스라서, 이것을 따로 다루지 않는 호출자(시뮬레이터 호출 → episode.py)는 지금처럼 infra_error로 기록한다.
# 에이전트 루프(agent.py)만 이것을 잡아 컨텍스트 한도 종료(context_limit)로 바꾼다. 평가에서는 실패로 세고,
# 서비스는 세션을 닫으며 새 상담을 권한다.
# [왜 따로 두나] Ollama는 요청에 truncate: false를 주지 않으면 넘친 프롬프트의 앞부분(시스템 프롬프트 포함)을
# 조용히 잘라 HTTP 200으로 답한다. ollama.py는 2026-10-15부터 truncate: false를 보내 넘침을 오류로 받는다.
# 같은 PC·같은 Ollama 0.35.1에서 이 동작을 확인한 기록이 rag-doc-qa 프로젝트의 v11 스모크에 있다(HTTP 400,
# type exceed_context_size_error). prompt_tokens·num_ctx는 서버가 알려 준 값이고, 모르면 0이다.
class ContextOverflowError(ProviderError):
    """프롬프트가 모델의 컨텍스트보다 길어 서버가 요청을 거절했다(잘라서 답하지 않았다)."""

    def __init__(self, message: str, *, prompt_tokens: int = 0, num_ctx: int = 0):
        super().__init__(message)
        self.prompt_tokens = prompt_tokens
        self.num_ctx = num_ctx


# LLM 제공자 인터페이스. 구현은 대화 전체와 도구 목록을 받아 한 턴을 돌려주는 chat() 하나와, 실행 manifest에
# 남길 정보를 주는 describe()다.
# 대화 상태를 제공자 쪽에 두지 않고 매번 전체 기록을 보내는 방식이라, 같은 기록으로 같은 요청을 다시 만들 수
# 있다.
class ChatProvider(ABC):
    name: str = "provider"

    @abstractmethod
    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] = (),
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """대화 기록 전체를 보내고 assistant 턴 하나를 돌려준다.

        `tools`의 항목은 중립 형식 {"name", "description", "parameters"}이고, 제공자마다 자기 형식으로
        감싼다. 서버의 실패는 ProviderError로 올린다.
        """

    # run.py가 manifest.json의 agent_provider·user_provider에 넣는 값. analyze same-setup은 두 실행의 이
    # 값(모델 digest, 서버 버전, 옵션)이 같은지 비교해 짝지어도 되는지 판단한다.
    def describe(self) -> dict[str, Any]:
        """실행 manifest에 들어갈 정보(모델, digest, 옵션)."""
        return {"provider": self.name}


# 테스트가 준비한 대본보다 많은 호출이 왔다는 뜻이다. 루프가 예상보다 LLM을 더 부르는 회귀를 테스트에서 바로
# 드러낸다.
class ScriptExhaustedError(RuntimeError):
    """대본 재생 제공자가 테스트가 준비한 것보다 많은 턴을 요청받았다."""


# 미리 정한 응답을 순서대로 돌려주는 가짜 제공자. GPU·모델 없이 에이전트 루프, 환경, 판정을 결정적으로
# 테스트하는 데 쓴다(tests/의 여러 테스트).
# requests에 받은 요청(메시지, 도구 목록, temperature, seed)을 모두 남겨, 테스트가 루프가 무엇을 보냈는지
# 확인할 수 있게 한다.
# mcp_server.py는 빈 대본으로 이것을 넣는다. MCP에서는 외부 클라이언트가 직접 도구를 부르므로 모델이 필요
# 없다.
class ScriptedProvider(ChatProvider):
    """준비한 턴을 차례로 재생한다. str은 본문 답, ToolCall은 도구 호출 하나다."""

    name = "scripted"

    def __init__(self, script: Sequence[ChatResponse | ToolCall | str]):
        self._script = list(script)
        self.requests: list[dict[str, Any]] = []

    def chat(self, messages, tools=(), *, temperature=0.0, seed=None, max_tokens=1024):
        self.requests.append(
            {"messages": list(messages), "tools": list(tools), "temperature": temperature, "seed": seed}
        )
        # 대본이 떨어졌는데 또 불리면 조용히 빈 답을 주지 않고 예외를 낸다. 루프가 몇 번 불렀는지 메시지에
        # 남긴다.
        if not self._script:
            raise ScriptExhaustedError(f"script ran out after {len(self.requests) - 1} calls")
        item = self._script.pop(0)
        # 대본 항목의 세 형태를 ChatResponse로 맞춘다: 이미 만든 응답은 그대로, ToolCall은 도구 호출 하나만
        # 있는 응답, 문자열은 본문 답.
        if isinstance(item, ChatResponse):
            return item
        if isinstance(item, ToolCall):
            return ChatResponse(tool_calls=(item,))
        return ChatResponse(text=item)
