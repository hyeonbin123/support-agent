"""Tool contracts: argument base class, context with the policy checks, the @tool decorator and execute().

execute() is the single path of every caller (agent loop, gold replay, the service and the MCP server):
look up -> validate -> before_write hook -> handler and on_success hook in one DB session -> commit or
roll back. Both hooks run only when a caller passes them: the service passes on_success (approval gate and
audit row, service/core.py); no caller in this package passes before_write. A dry run (R2's preview,
confirm.py) rolls back what the handler did and skips the hooks.
"""

# [파일 역할] 도구 계약 모듈. 도구 인자의 기반 클래스(ToolArgs), 대화 상태·규정 검사를 담은 문맥(ToolContext),
# 함수를 도구로 바꾸는 @tool, 등록부 검사(make_registry), 모든 도구 호출이 지나는 단일 실행 경로 execute()를
# 둔다.
# [설계 이유] 에이전트 루프, 정답 재실행(judge.py), 서비스(service/core.py), MCP 서버가 모두 같은 execute()를
# 지나므로, 인자 검증·규정 검사·트랜잭션 처리가 호출하는 쪽마다 달라지지 않는다(README "구조"). 규정 검사를
# require(항상 막음)와 check_policy(P1이면 막고 P0면 기록만)로 나눠, 같은 도구 코드로 "규정을
# 프롬프트에만"(P0)과 "도구에서도 막음"(P1)을 잴 수 있다(docs/design.md).
# [연결] tools.py가 @tool과 make_registry로 도구 14개를 정의하고, agent.py는 ToolSpec.schema()로 모델에 줄
# 도구 설명을 만든다. 다른 모듈이 모두 기대는 계약이라 고치면 전체 테스트를 돌린다.

from __future__ import annotations

import copy
import inspect
import json
import re
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# pydantic: 모델이 보낸 인자를 검증하고, 같은 모델에서 JSON Schema(도구 설명)를 뽑는다. SQLAlchemy Session:
# 도구 호출 하나가 DB 세션 하나 안에서 돌고 commit 또는 rollback된다.
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from sqlalchemy import Engine
from sqlalchemy.orm import Session

# 실패한 도구 결과의 머리말("Error: [code] 메시지"). claims.py, confirm_report.py 등이 이 접두사로 실패한
# 결과를 가려낸다.
ERROR_PREFIX = "Error"
# 도구 이름 규칙: 소문자로 시작하는 영문 소문자·숫자·밑줄 64자 이하. 작은 모델과 MCP SDK가 다루기 쉬운 이름만
# 받는다(make_registry).
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


# 모든 도구 인자 모델의 기반. extra="forbid"라 모델이 지어낸 인자 이름은 invalid_arguments로 거절된다.
class ToolArgs(BaseModel):
    """Base class of every tool argument model. Invented arguments are rejected."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    # 인자 안의 모든 문자열(중첩된 dict·list 포함)을 검사한다. NUL 문자와 짝 없는 서로게이트는 PostgreSQL
    # 드라이버나 기록 파일(JSONL) 쓰기에서 터지므로, 도구가 실행되기 전에 검증 오류로 돌려보낸다.
    def _storable_text(cls, data: Any) -> Any:
        """Lone surrogates and NUL bytes would crash the database driver or the report writer."""

        def check(value: Any) -> None:
            if isinstance(value, str):
                if "\x00" in value:
                    raise ValueError("text must not contain NUL characters")
                try:
                    value.encode("utf-8")
                except UnicodeEncodeError as error:
                    raise ValueError("text is not valid Unicode") from error
            elif isinstance(value, dict):
                for item in value.values():
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)

        check(data)
        return data


# 예상한 실패(본인 확인 안 됨, 규정 위반 등). execute()가 잡아 "Error: [code] 메시지"로 모델에 돌려준다.
# 메시지에는 이유만 적고 "다음에 무엇을 하라"는 힌트는 넣지 않는다(P1의 효과에 가르쳐 주는 효과가 섞이지 않게,
# docs/design.md).
class ToolError(Exception):
    """An expected failure that is reported back to the model as `Error: [code] message`."""

    def __init__(self, code: str, message: str, *, policy: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.policy = policy  # True when a policy rule (P axis) refused the call


# 핸들러의 예상 못 한 예외. 에이전트 실패가 아니라 코드 버그이므로 에피소드는 infra_error가 되어 성공률의
# 분모에서 빠진다(docs/design.md).
class ToolBugError(RuntimeError):
    """A handler raised something unexpected. This is our bug, never the agent's failure."""


# 한 대화에서 도구 호출을 넘어 기억하는 상태. 지금은 본인 확인된 고객 번호 하나다(대화당 고객 1명). 서비스는
# 이것을 세션 행에 저장한다.
@dataclass
class ConversationState:
    """What tools remember across calls of one conversation. The service will store this per chat session."""

    verified_customer_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"verified_customer_id": self.verified_customer_id}


# 도구 호출마다 넘기는 문맥.
# - now: 고정 시계(평가는 과제의 now, 서비스는 설정의 시계). 도구 코드는 datetime.now를 쓰지 않고 이것만 본다
# - enforce_policy: P1이면 True
# - violations: P0에서 통과시킨 규정 위반 코드(판정의 보조 지표)
# - caller_phone: 음성 V4에서만 쓰는 발신 번호(verify_caller)
@dataclass
class ToolContext:
    """Per-conversation input of every tool call. Evaluation injects a fixed `now`."""

    now: datetime  # timezone-aware
    enforce_policy: bool = True  # P1 = True, P0 = False
    state: ConversationState = field(default_factory=ConversationState)
    violations: list[str] = field(default_factory=list)  # policy codes that were let through (P0 only)
    caller_phone: str | None = None  # voice V4: the number the call comes from (digits); None elsewhere

    # 무결성·본인 확인 검사. P0·P1과 상관없이 항상 막는다.
    def require(self, ok: bool, code: str, message: str) -> None:
        """Integrity and identity checks. Refused in every mode."""
        if not ok:
            raise ToolError(code, message)

    # 코드로 확인할 수 있는 규정. P1이면 policy=True인 ToolError로 막고, P0면 위반 코드만 기록하고 통과시킨다.
    def check_policy(self, ok: bool, code: str, message: str) -> None:
        """Policy rules that code can verify. P1 refuses; P0 records the violation and lets the call pass."""
        if ok:
            return
        if self.enforce_policy:
            raise ToolError(code, message, policy=True)
        self.violations.append(code)


# 도구 핸들러의 모양: (DB 세션, 문맥, 검증된 인자) → JSON으로 바꿀 수 있는 dict.
Handler = Callable[[Session, ToolContext, Any], dict[str, Any]]


# 등록된 도구 하나의 정의. write는 DB를 바꾸는 도구다(R2의 쓰기 전 확인, MCP의 쓰기 범위, 판정의 쓰기 집계,
# 서비스 감사 로그의 write 표시가 이 값을 본다). unordered_args는 순서가 의미 없는 목록 인자(반품 줄 번호),
# uncompared_args는 자유 글처럼 호출을 비교할 때 빼는 인자다(판정의 정답에 없는 쓰기 계산 등). terminates는
# 성공하면 대화가 끝나는 도구(상담원 이관)다.
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[ToolArgs]
    handler: Handler
    write: bool  # True = changes the DB (confirmation, approval and the MCP write scope apply)
    unordered_args: tuple[str, ...] = ()  # list arguments whose order carries no meaning
    terminates: bool = False  # True = the conversation ends after a successful call (hand-off)
    uncompared_args: tuple[str, ...] = ()  # free text and the like: ignored when calls are compared

    # 모델과 MCP에 주는 도구 설명: 이름, 설명(핸들러의 docstring), 평탄하게 만든 인자 JSON Schema.
    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": llm_schema(self.args_model)}

    def __call__(self, session: Session, ctx: ToolContext, args: Any) -> dict[str, Any]:
        return self.handler(session, ctx, args)


# 도구 호출 한 번의 결과. content는 모델이 읽는 글(성공이면 JSON, 실패면 "Error: [code] 메시지")이고, args는
# 검증·정규화된 인자(검증 실패면 None)다.
@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str  # JSON text on success, "Error: [code] message" on failure
    error_code: str | None = None  # unknown_tool | invalid_arguments | a code raised by the handler or a hook
    policy_blocked: bool = False
    args: dict[str, Any] | None = None  # canonical arguments (None when validation failed)
    violations: tuple[str, ...] = ()  # policy codes this call violated while being let through (P0)

    @staticmethod
    def error(
        code: str, message: str, *, policy: bool = False, args: dict[str, Any] | None = None
    ) -> ToolResult:
        return ToolResult(False, f"{ERROR_PREFIX}: [{code}] {message}", code, policy, args)


# 도구 이름 → 정의. BeforeWrite는 쓰기 도구의 핸들러 실행 전에 부르는 훅으로, ToolResult를 돌려주면 그것이
# 거절 결과가 된다. 지금 이 패키지에서 이 훅을 넘기는 호출자는 없고 계약은 테스트(test_toolkit.py)가 확인한다.
# 서비스의 승인 관문은 핸들러가 계산한 실제 환불액을 봐야 해서 이 훅이 아니라 OnSuccess 훅에 있다(a5717ce).
Registry = dict[str, ToolSpec]
BeforeWrite = Callable[[ToolSpec, dict[str, Any]], ToolResult | None]
# Runs after the handler, inside its transaction: (session, spec, canonical args, the handler's answer).
# Rows it adds are committed with the tool's changes or not at all; a ToolError it raises rolls the
# call back like a refusal of the tool itself.
OnSuccess = Callable[[Session, ToolSpec, dict[str, Any], dict[str, Any]], None]


# 핸들러 함수를 ToolSpec으로 바꾸는 데코레이터. args 매개변수의 타입 힌트에서 인자 모델을, docstring에서 도구
# 설명을 가져온다. 둘 중 하나라도 없으면 정의 시점에 TypeError로 멈춘다.
def tool(
    *,
    write: bool,
    unordered_args: tuple[str, ...] = (),
    terminates: bool = False,
    uncompared_args: tuple[str, ...] = (),
) -> Callable[[Handler], ToolSpec]:
    """Turn `def name(session, ctx, args: SomeArgs) -> dict` with a docstring into a ToolSpec.

    The ToolSpec is callable with the same arguments, so tests may call `cancel_order(session, ctx, args)`.
    """

    def wrap(fn: Handler) -> ToolSpec:
        args_model = typing.get_type_hints(fn).get("args")
        if not (isinstance(args_model, type) and issubclass(args_model, ToolArgs)):
            raise TypeError(f"{fn.__name__}: annotate the args parameter with a ToolArgs subclass")
        if not fn.__doc__:
            raise TypeError(f"{fn.__name__}: the docstring is the tool description and is required")
        return ToolSpec(
            fn.__name__,
            inspect.cleandoc(fn.__doc__),
            args_model,
            fn,
            write,
            unordered_args,
            terminates,
            uncompared_args,
        )

    return wrap


# 등록부를 만들며 작은 모델이나 MCP SDK가 잘 다루지 못하는 정의를 거절한다: 이름 규칙 위반·중복, 추가 인자를
# 허용하는 모델, 필수 인자가 하나도 없는 도구(Ollama의 도구 호출 파서가 처리하지 못함), $ref·anyOf가 남은
# 스키마, 없는 필드를 가리키는 unordered_args·uncompared_args.
def make_registry(*specs: ToolSpec) -> Registry:
    """Build a registry and refuse tool definitions that small models or the MCP SDK handle badly."""
    registry: Registry = {}
    for spec in specs:
        if not _NAME_RE.match(spec.name):
            raise ValueError(f"bad tool name: {spec.name!r}")
        if spec.name in registry:
            raise ValueError(f"duplicate tool name: {spec.name}")
        if spec.args_model.model_config.get("extra") != "forbid":
            raise ValueError(f"{spec.name}: the args model must forbid extra fields")
        schema = llm_schema(spec.args_model)
        if not schema.get("required"):
            # Ollama's generic tool-call parser cannot handle a call whose arguments are all optional.
            raise ValueError(f"{spec.name}: at least one argument must be required")
        flat = json.dumps(schema)
        if "$ref" in flat or "anyOf" in flat:
            raise ValueError(f"{spec.name}: the argument schema must be flat (no $ref, no anyOf)")
        for arg in (*spec.unordered_args, *spec.uncompared_args):
            if arg not in spec.args_model.model_fields:
                raise ValueError(f"{spec.name}: unordered_args/uncompared_args name an unknown field {arg!r}")
        registry[spec.name] = spec
    return registry


# pydantic의 JSON Schema를 작은 모델이 읽기 쉽게 평탄하게 만든다: $ref로 따로 둔 정의(enum 등)를 제자리에 풀어
# 넣고, Optional(X | None)의 anyOf를 X 하나로 접고, title과 기본값 None을 지운다. 인자 모델의 docstring은
# 스키마에 넣지 않는다.
def llm_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Flatten pydantic's JSON Schema for small models: inline $ref, fold Optional, drop titles."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(x) for x in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            target = copy.deepcopy(defs[node["$ref"].rsplit("/", 1)[-1]])
            return walk({**target, **{k: v for k, v in node.items() if k != "$ref"}})
        if "anyOf" in node:
            non_null = [x for x in node["anyOf"] if x != {"type": "null"}]
            if len(non_null) == 1:
                return walk({**non_null[0], **{k: v for k, v in node.items() if k != "anyOf"}})
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "title" or (key == "default" and value is None):
                continue
            out[key] = {k: walk(v) for k, v in value.items()} if key == "properties" else walk(value)
        return out

    flat = walk(schema)
    flat.pop("description", None)  # the docstring of the args model is not part of the schema
    return flat


# 검증 오류를 "필드: 이유" 몇 개(기본 3개)로 줄인 글. 모델에 돌려줄 오류 메시지를 짧게 유지한다. 입력값은 넣지
# 않는다.
def short_errors(error: ValidationError, limit: int = 3) -> str:
    items = error.errors(include_url=False, include_input=False)
    parts = [(".".join(map(str, e["loc"])) or "(root)") + ": " + e["msg"] for e in items[:limit]]
    more = len(items) - limit
    return "; ".join(parts) + (f" (+{more} more)" if more > 0 else "")


# 검증된 인자를 비교할 수 있는 형태로: JSON 타입, None 제거, 순서 없는 목록은 정렬. 판정(정답 동작과 비교),
# 서비스 승인 대기열(같은 요청 찾기), R2 확인(confirm.py)이 이 형태를 쓴다.
def canonical_args(spec: ToolSpec, args: ToolArgs) -> dict[str, Any]:
    """Validated arguments in a comparable form: JSON types, no None, order-free lists sorted."""
    out = args.model_dump(mode="json", exclude_none=True)
    for name in spec.unordered_args:
        if isinstance(out.get(name), list):
            out[name] = sorted(out[name])
    return out


# 도구 호출 한 번을 실행한다. 순서: 이름 확인 → 인자 검증 → (쓰기 도구면) before_write 훅 → DB 세션 하나에서
# 핸들러 → on_success 훅 → commit, 실패하면 rollback. 두 훅은 넘겼을 때만 돈다. 평가(에이전트 루프, 정답
# 재실행)는 둘 다 넘기지 않고, 서비스는 on_success만 넘긴다(service/core.py _run_tool).
# 예상한 실패는 ToolResult로 돌려주고, 핸들러 버그는 ToolBugError로 올린다.
def execute(
    registry: Registry,
    engine: Engine,
    ctx: ToolContext,
    name: str,
    raw_args: dict[str, Any],
    *,
    before_write: BeforeWrite | None = None,
    on_success: OnSuccess | None = None,
    dry_run: bool = False,
) -> ToolResult:
    """Run one tool call. Expected failures come back as ToolResult; a handler bug raises ToolBugError.

    dry_run (R2's preview): the handler runs and everything it did is rolled back, the conversation state
    and the recorded violations included; the hooks do not run. The answer and the violations the call would
    have are returned."""
    spec = registry.get(name)
    if spec is None:
        return ToolResult.error("unknown_tool", f"{name} 도구는 없습니다.")
    try:
        args = spec.args_model.model_validate(raw_args)
    except ValidationError as error:
        return ToolResult.error("invalid_arguments", f"인자가 잘못되었습니다. {short_errors(error)}")
    clean = canonical_args(spec, args)
    if spec.write and before_write is not None and not dry_run:
        refusal = before_write(spec, clean)
        if refusal is not None:
            return refusal
    # 이 호출 전의 위반 기록 수와 대화 상태를 기억해 둔다. 호출이 거절되거나 실패하면 이 호출이 바꾼 상태(예:
    # 본인 확인)와 위반 기록을 되돌린다.
    seen = len(ctx.violations)
    saved_state = copy.copy(ctx.state)
    with Session(engine) as session:
        try:
            result = spec.handler(session, ctx, args)
            content = json.dumps(result, ensure_ascii=False)
            # 미리보기(R2): 핸들러가 한 일을 모두 롤백하고 대화 상태·위반 기록도 되돌린 뒤, "했다면 이렇게
            # 된다"는 답과 위반 코드만 돌려준다. 훅은 부르지 않는다.
            if dry_run:
                session.rollback()
                ctx.state = saved_state
                found = tuple(ctx.violations[seen:])
                del ctx.violations[seen:]
                return ToolResult(True, content, args=clean, violations=found)
            # on_success는 같은 트랜잭션 안에서 돈다. 서비스는 여기서 감사 행을 추가하고, 환불액이 기준
            # 이상이면 ToolError를 내 호출 전체를 롤백한다(승인 관문).
            if on_success is not None:
                on_success(session, spec, clean, result)
            session.commit()
        except ToolError as error:
            # 거절된 호출은 아무것도 바꾸지 않았으므로 위반도 없던 것으로 지운다.
            session.rollback()
            ctx.state = saved_state
            del ctx.violations[seen:]  # a refused call changed nothing, so it violated nothing
            return ToolResult.error(error.code, error.message, policy=error.policy, args=clean)
        except Exception as error:
            # 예상 못 한 예외: 롤백하고 상태를 되돌린 뒤 ToolBugError로 감싸 올린다. 메시지에 도구 이름과
            # 인자를 넣어 기록에서 원인을 찾을 수 있게 한다.
            session.rollback()
            ctx.state = saved_state
            raise ToolBugError(f"{name}({clean}) raised {type(error).__name__}: {error}") from error
    return ToolResult(True, content, args=clean, violations=tuple(ctx.violations[seen:]))
