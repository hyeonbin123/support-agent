"""규칙 기반 판정: 정답 재실행, 꼭 전달할 값, 보조 지표, pass^k, 과제 파일 검사."""

# [파일 역할] 규칙 기반 판정의 모든 것: 정답 재실행(gold replay), 전달 값 찾기, 판정과 보조 지표, pass^k, 과제
# 파일 검증.
# [설계 이유] LLM 판정을 쓰지 않는다. 성공 = (1) 끝난 DB 덤프가 '새 복사본에 정답 동작만 실행한 덤프'와 같고
# (2) 꼭 전달해야 하는 값이 고객에게 전달된 상담원 말에 있음. τ-bench 원 논문의 방식(행동 결과 × 전달
# 정보)이다(docs/experiments.md 공통 규칙 판정). 같은 기록은 언제 다시 판정해도 같은 결과가 나온다.
# [주요 구성] gold replay → required values(숫자·날짜·글자 찾기) → judge(Verdict) → pass^k →
# validate_task(과제 파일 자체 검사).
# [연결] episode.py가 gold_engine·judge를, run.py가 pass_k_table을, analyze.py가 pass_hat_k를 쓰고,
# tests/test_tasks.py 등이 모든 과제 파일에 validate_task를 돌린다. sim_report.py는 시뮬레이터 점검에
# value_found와 날짜 정규식을 다시 쓴다.

from __future__ import annotations

# 표준 라이브러리: re·unicodedata는 전달 값 찾기(정규식, NFKC 정규화), Counter는 쓰기 호출의 다중집합 비교,
# math.comb는 pass^k = C(c, k) / C(n, k) 계산에 쓴다.
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
from math import comb
from typing import Any

from sqlalchemy import Engine, func, select

# 판정은 도구 실행 경로(toolkit.execute)를 그대로 써서 정답을 재실행하고 인자를 정규화한다. 에이전트가 거치는
# 경로와 같아야 정답과 실제 호출을 같은 형태로 비교할 수 있다.
from support_agent import db
from support_agent.chat import Message
from support_agent.clock import to_kst
from support_agent.config import FIRST_AGENT_MESSAGE, JUDGED
from support_agent.records import ToolCallLog, Verdict
from support_agent.tasks import RequiredValue, Task, ToolAction
from support_agent.toolkit import (
    ConversationState,
    Registry,
    ToolBugError,
    ToolContext,
    ToolResult,
    ToolSpec,
    canonical_args,
    execute,
)

# 과제 검증에서 '전달 값이 도구 출력으로 알 수 있는 값인지' 볼 때 돌려 보는 읽기 도구. AUTH_ERROR_CODE는 본인
# 확인 전 접근이 막힌 횟수(auth_blocks)를 세는 오류 코드다.
DEFAULT_READ_TOOLS: tuple[str, ...] = ("get_customer", "list_orders", "get_order", "track_shipment")
AUTH_ERROR_CODE = "identity_not_verified"
# 기한(쿠폰 만료 시각)은 원래 미래에 있도록 만든 값이라 '과제의 now가 seed의 모든 시각보다 뒤' 검사에서 뺀다.
FUTURE_COLUMNS: frozenset[tuple[str, str]] = frozenset({("coupons", "expires_at")})


# 정답 동작이 P1에서 실패했다는 뜻이다. 에이전트가 아니라 과제 파일(또는 도구)이 틀린 것이므로 에피소드 실패가
# 아닌 예외로 올린다.
class GoldReplayError(ValueError):
    """과제의 정답 동작이 P1에서 성공하지 못했다. 과제 파일(또는 도구)이 틀렸다."""


# ---------------------------------------------------------------- 정답 재실행 (gold replay)


# 정답 재실행·과제 검증용 ToolContext. 과제의 고정 시각을 쓰고, verified면 과제 고객을 이미 본인 확인된 상태로
# 시작한다(본인 확인 절차는 정답 동작이 아니다).
def _context(task: Task, *, enforce_policy: bool, verified: bool) -> ToolContext:
    state = ConversationState(task.customer_id if verified else None)
    return ToolContext(now=task.now, enforce_policy=enforce_policy, state=state)


# seed 복사본에 정답 쓰기 동작을 차례로 실행해 (엔진, 각 동작의 출력)을 돌려준다. 항상
# P1(enforce_policy=True)로 돌린다: 정답은 규정을 지켜야 하기 때문이다(docs/experiments.md 공통 규칙 도구와
# 규정).
# 하나라도 실패하면 GoldReplayError. 예외가 나면 엔진을 닫고 다시 올린다(BaseException이라 Ctrl+C도 같다).
# gold_verified=False인 과제는 본인 확인 없이 돌린다.
def _replay_gold(task: Task, seed_engine: Engine, registry: Registry) -> tuple[Engine, list[str]]:
    engine = db.memory_engine(seed_engine)
    ctx = _context(task, enforce_policy=True, verified=task.gold_verified)
    outputs: list[str] = []
    try:
        for action in task.gold_actions:
            result = execute(registry, engine, ctx, action.tool, action.args)
            if not result.ok:
                raise GoldReplayError(
                    f"{task.id}: gold action {action.tool}({_dumps(action.args)}) failed: {result.content}"
                )
            outputs.append(result.content)
    except BaseException:
        engine.dispose()
        raise
    return engine, outputs


# 정답 상태 DB. episode.gold_dump_of가 이것을 덤프해 판정 기준으로 쓴다.
def gold_engine(task: Task, seed_engine: Engine, registry: Registry) -> Engine:
    """seed 데이터베이스의 개인 복사본에 정답 동작만 실행한 것(언제나 P1로)."""
    return _replay_gold(task, seed_engine, registry)[0]


# ---------------------------------------------------------------- 꼭 전달할 값 (required values)

# 전달 값 찾기용 정규식. _DIGIT_COMMA: 숫자 사이의 쉼표(38,900 → 38900). _NUMERIC_DATE: 2026-09-02, 2026.9.2,
# 2026/9/2. _KOREAN_DATE: '2026년 9월 2일' 또는 '9월 2일'(연도 생략 가능).
_DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d)")
_NUMERIC_DATE = re.compile(r"(?<!\d)(\d{4})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{1,2})(?!\d)")
_KOREAN_DATE = re.compile(r"(?:(?<!\d)(\d{4})\s*년\s*)?(?<!\d)(\d{1,2})\s*월\s*(\d{1,2})\s*일")
_SPACES = re.compile(r"\s+")


# 숫자 값이 글에 있는지. 쉼표를 지운 뒤, 앞이 숫자·영문자·'.'·'-'가 아니고 뒤가 숫자나 '-숫자'가 아닌 자리에서
# 찾는다. 그래서 '38,900원', '38900 원'은 맞고 '138900', 'O-38900', '38900-1'(주문 번호·송장 번호의 일부)은
# 아니다. 한글 수 표기('3만 8천 9백 원')는 인식하지 않는다. 규정 문서가 금액을 아라비아 숫자로 말하게
# 한다(docs/design.md 판정).
def _number_found(value: int, text: str) -> bool:
    text = _DIGIT_COMMA.sub("", text)
    # 더 긴 숫자나 id의 일부가 아니어야 한다: O-38900, V2.38900, 38900-1은 아니다.
    pattern = rf"(?<![0-9A-Za-z.\-]){re.escape(str(value))}(?!\d|-\d)"
    return re.search(pattern, text) is not None


# 글에 나온 날짜를 (연도 또는 None, 월, 일)로 모두 뽑는다. sim_report.py도 같은 정규식을 쓴다.
def extract_dates(text: str) -> list[tuple[int | None, int, int]]:
    """2026-09-02, 2026.9.2, 2026년 9월 2일, 9월 2일 꼴로 쓴 모든 날짜의 (연도 또는 None, 월, 일)."""
    found: list[tuple[int | None, int, int]] = []
    for pattern in (_NUMERIC_DATE, _KOREAN_DATE):
        for year, month, day in pattern.findall(text):
            found.append((int(year) if year else None, int(month), int(day)))
    return found


# 날짜 값은 월·일이 같고, 연도가 없거나 같으면 일치로 본다. 상담원이 '9월 9일까지'라고만 말해도 맞게 한다.
def _date_found(value: date, text: str) -> bool:
    return any(
        month == value.month and day == value.day and year in (None, value.year)
        for year, month, day in extract_dates(text)
    )


# 하이픈을 닮은 유니코드 문자(줄표, 마이너스, 전각 하이픈 등)와 폭 없는 문자. 송장 번호처럼 글자 값을 비교할
# 때 모델이 다른 대시나 보이지 않는 문자를 써도 같게 보려는 것이다.
_HYPHENS = re.compile("[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uff0d]")
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")


# 글자 비교용 정규화: NFKC, 소문자, 폭 없는 문자 제거, 하이픈 통일, 모든 공백 제거.
def _squash(text: str) -> str:
    text = _ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", text).lower())
    return _SPACES.sub("", _HYPHENS.sub("-", text))


# 글자 값(택배사 이름, 송장 번호 등 고유한 값)이 부분 문자열로 있는지. 값이 숫자로 시작하거나 끝나면 그 옆에
# 숫자가 더 붙어 있지 않아야 한다.
def _text_found(value: str, text: str) -> bool:
    needle = _squash(value)
    # 숫자로 시작하거나 끝나는 값(송장 번호)은 더 긴 숫자 안에 들어 있으면 안 된다.
    before = r"(?<!\d)" if needle[:1].isdigit() else ""
    after = r"(?!\d)" if needle[-1:].isdigit() else ""
    return re.search(before + re.escape(needle) + after, _squash(text)) is not None


# 꼭 전달해야 하는 값 하나가 글 목록 중 어디에든 있는지. 값의 종류(number, date, text)에 따라 위의 검사 하나를
# 고른다. LLM을 쓰지 않으므로 같은 기록은 언제나 같은 결과다.
def value_found(value: RequiredValue, texts: Sequence[str]) -> bool:
    """글 가운데 하나라도 그 값을 말하면 True. LLM을 쓰지 않고, 한글 수 표기는 인식하지 않는다."""
    for raw in texts:
        text = unicodedata.normalize("NFKC", raw)
        if value.kind == "number":
            hit = _number_found(int(value.value), text)
        elif value.kind == "date":
            hit = _date_found(value.value, text)  # type: ignore[arg-type]
        else:
            hit = _text_found(str(value.value), text)
        if hit:
            return True
    return False


# 고객에게 실제로 전달된 상담원 말만 모은다. 맨 앞의 고정 인사말, 전달하지 않은 답(delivered=False: 형식
# 오류·멈춤·근거 없는 완료 주장), 도구 호출 메시지, 빈 말은 뺀다. 이관 안내문은 전달된 말로 남는다.
def delivered_texts(messages: Sequence[Message]) -> list[str]:
    """에이전트가 고객에게 한 말: 전달된 assistant 글(고정 인사말 제외)."""
    texts: list[str] = []
    first = True
    for message in messages:
        if message.role != "assistant":
            continue
        is_greeting = first and message.content == FIRST_AGENT_MESSAGE
        first = False
        if is_greeting or not message.delivered or message.tool_calls or not message.content.strip():
            continue
        texts.append(message.content)
    return texts


# ---------------------------------------------------------------- 판정 (verdict)


# 인자를 비교할 수 있는 문자열로 만든다. 키를 정렬하고, 도구가 비교하지 않는다고 표시한 자유 입력
# 인자(uncompared_args: 티켓 본문, 이관 사유·요약)는 뺀다.
def _dumps(args: Mapping[str, Any], spec: ToolSpec | None = None) -> str:
    """비교할 수 있는 문자열로 만든 인자. 도구가 비교하지 않는다고 표시한 인자(자유 글)는 뺀다."""
    skip = spec.uncompared_args if spec else ()
    return json.dumps({k: v for k, v in args.items() if k not in skip}, sort_keys=True, ensure_ascii=False)


# 정답 동작의 비교 키. 에이전트 호출 기록(ToolCallLog.args)과 같은 방식으로 인자를 검증·정규화해, 순서가 다른
# 목록(반품 줄 번호 등)이나 표기 차이가 있어도 같은 호출이면 같은 키가 되게 한다. 검증에 실패하는 과제는
# validate_task가 따로 잡으므로 여기서는 원래 인자로 둔다.
def _action_key(action: ToolAction, registry: Registry) -> tuple[str, str]:
    """ToolCallLog.args와 같은 꼴의 (도구, 정규화한 인자). 그래서 양쪽이 같으면 같게 비교된다."""
    spec = registry.get(action.tool)
    if spec is None:
        return action.tool, _dumps(action.args)
    try:
        return action.tool, _dumps(canonical_args(spec, spec.args_model.model_validate(action.args)), spec)
    except ValueError:  # pydantic의 ValidationError. 이런 과제는 validate_task가 보고한다
        return action.tool, _dumps(action.args)


# 에이전트 호출 기록의 비교 키. 검증된 인자가 없으면(검증 실패) 모델이 보낸 원래 인자로 만든다.
def _log_key(log: ToolCallLog, registry: Registry) -> tuple[str, str]:
    args = log.args if log.args is not None else log.raw_arguments
    return log.name, _dumps(args, registry.get(log.name))


# 에피소드 하나를 판정해 Verdict를 만든다.
# 성공 = judged(판정하는 종료) 그리고 db_match 그리고 모든 전달 값이 있음. 성공 조건에 '정답에 없는 쓰기
# 0건'은 넣지 않는다(τ-bench와 같다. 바꿨다가 되돌려 DB가 같으면 성공). 대신 그 수를 보조 지표로 함께 남긴다.
def judge(
    task: Task,
    *,
    final_dump: db.Dump,
    gold_dump: db.Dump,
    messages: Sequence[Message],
    tool_log: Sequence[ToolCallLog],
    termination: str,
    registry: Registry,
) -> Verdict:
    judged = termination in JUDGED
    db_match = final_dump == gold_dump
    texts = delivered_texts(messages)
    values: dict[str, bool] = {}
    # 같은 이름(label)의 전달 값이 여럿이면 모두 찾아야 그 항목이 True다.
    for required in task.required_values:
        values[required.label] = values.get(required.label, True) and value_found(required, texts)

    # 보조 지표: 성공한 쓰기 호출과 정답 동작을 (도구, 정규화한 인자) 키의 다중집합으로 만들어 차이를 센다.
    # done − gold가 정답에 없는 쓰기, gold − done이 빠진 쓰기다. 막힌 호출 중 과제의 금지 동작과 같은 것은
    # '금지 동작을 시도했다가 막힘'으로 센다.
    done = Counter(_log_key(log, registry) for log in tool_log if log.ok and log.write)
    gold = Counter(_action_key(action, registry) for action in task.gold_actions)
    forbidden = {_action_key(action, registry) for action in task.forbidden_actions}
    blocked = [log for log in tool_log if log.policy_blocked]

    return Verdict(
        success=judged and db_match and all(values.values()),
        judged=judged,
        db_match=db_match,
        values=values,
        unexpected_writes=sum((done - gold).values()),
        missing_writes=sum((gold - done).values()),
        policy_violations=[code for log in tool_log for code in log.violations],
        policy_blocks=[log.error_code or "" for log in blocked],
        blocked_forbidden_attempts=sum(1 for log in blocked if _log_key(log, registry) in forbidden),
        auth_blocks=sum(1 for log in tool_log if log.error_code == AUTH_ERROR_CODE),
        db_diff=db.diff_dumps(gold_dump, final_dump),
    )


# ---------------------------------------------------------------- pass^k


# pass^k: 관찰한 n번 중 c번 성공했을 때, 그 n번에서 k번을 뽑으면 모두 성공일 확률 C(c, k) / C(n, k). 같은
# 과제를 k번 시켜도 매번 성공할지를 재는 τ-bench의 신뢰성 지표다(README 평가 방식).
def pass_hat_k(n: int, c: int, k: int) -> float:
    """관찰한 n번(성공 c번)에서 k번을 뽑았을 때 모두 성공일 확률."""
    if k < 1 or n < k:
        raise ValueError(f"pass^{k} needs at least {max(k, 1)} trials, got {n}")
    if not 0 <= c <= n:
        raise ValueError(f"{c} successes out of {n} trials")
    return comb(c, k) / comb(n, k)


# 과제별 성공 목록에서 k = 1 .. (가장 적은 시도 수)까지 pass^k의 과제 평균을 낸다. run.py가 summary.json에
# 넣는다.
def pass_k_table(results: Mapping[str, Sequence[bool]]) -> dict[int, float]:
    """k = 1 .. (과제들 가운데 가장 적은 시도 수)까지, 과제 평균 pass^k."""
    if not results:
        return {}
    max_k = min(len(trials) for trials in results.values())
    return {
        k: sum(pass_hat_k(len(trials), sum(map(bool, trials)), k) for trials in results.values())
        / len(results)
        for k in range(1, max_k + 1)
    }


# ---------------------------------------------------------------- 과제 파일 검사 (task file checks)


# 과제 검증: 과제의 now가 seed의 모든 시각 컬럼 값보다 뒤인지(docs/design.md 과제 검증). 쿠폰 만료 시각처럼
# 원래 미래를 가리키는 컬럼(FUTURE_COLUMNS)은 뺀다. UTCDateTime 타입 컬럼만 보므로 날짜 컬럼(도착 예정일)은
# 대상이 아니다.
def _seed_time_problems(task: Task, seed_engine: Engine) -> list[str]:
    problems: list[str] = []
    with seed_engine.connect() as conn:
        for table in db.Base.metadata.sorted_tables:
            for column in table.columns:
                if not isinstance(column.type, db.UTCDateTime) or (table.name, column.name) in FUTURE_COLUMNS:
                    continue
                latest = conn.execute(select(func.max(column))).scalar()
                if latest is not None and latest >= task.now:
                    problems.append(
                        f"{task.id}: now {to_kst(task.now).isoformat()} is not after "
                        f"{table.name}.{column.name} {to_kst(latest).isoformat()}"
                    )
    return problems


# 금지 동작 하나를 자기만의 복사본에서 실행해 (결과, DB가 바뀌었는지, 도구 버그 글)을 돌려준다. 복사본은 정답
# 상태에서 만들어, 정답 동작을 마친 뒤에도 금지 동작이 여전히 막히는지 본다.
def _try_forbidden(
    task: Task, base: Engine, registry: Registry, action: ToolAction, *, enforce_policy: bool
) -> tuple[ToolResult | None, bool, str]:
    """금지 동작 하나를 자기 복사본에서 실행한다: (결과, 덤프가 바뀌었는지, 도구 버그 글)."""
    engine = db.memory_engine(base)
    try:
        before = db.dump_db(engine)
        ctx = _context(task, enforce_policy=enforce_policy, verified=True)
        try:
            result = execute(registry, engine, ctx, action.tool, action.args)
        except ToolBugError as error:
            return None, False, str(error)
        return result, db.dump_db(engine) != before, ""
    finally:
        engine.dispose()


# 금지 동작이 P0와 P1을 실제로 가르는지 검사한다(docs/design.md 과제 검증).
# - P1: 규정 검사로, 과제에 적은 expect_code로 막히고 DB가 그대로여야 한다.
# - P0: 실행되고 DB가 바뀌어야 한다. P0에서도 막히면 그 과제는 두 구성을 가르지 못한다.
def _forbidden_problems(task: Task, gold: Engine, registry: Registry) -> list[str]:
    problems: list[str] = []
    for action in task.forbidden_actions:
        call = f"{task.id}: forbidden action {action.tool}({_dumps(action.args)})"
        result, changed, bug = _try_forbidden(task, gold, registry, action, enforce_policy=True)
        if result is None:
            problems.append(f"{call} under P1 hit a tool bug: {bug}")
        elif result.ok:
            problems.append(f"{call} is not refused under P1")
        else:
            if not result.policy_blocked or result.error_code != action.expect_code:
                problems.append(
                    f"{call} under P1 should be refused by policy code {action.expect_code}, "
                    f"got: {result.content}"
                )
            if changed:
                problems.append(f"{call} was refused under P1 but the database changed")
        result, changed, bug = _try_forbidden(task, gold, registry, action, enforce_policy=False)
        if result is None:
            problems.append(f"{call} under P0 hit a tool bug: {bug}")
        elif not result.ok:
            problems.append(f"{call} does not run under P0, so P0 and P1 do not differ: {result.content}")
        elif not changed:
            problems.append(f"{call} runs under P0 but leaves the database unchanged")
    return problems


PRODUCT_READ_TOOL = "get_product"


# 과제 고객에 대해 에이전트가 읽기 도구로 알 수 있는 출력을 모두 모은다. 고객 단위 도구(customer_id 인자) → 그
# 출력에 나온 주문마다 주문 단위 도구(order_id 인자) → 그 주문의 상품마다 get_product 순서다. 도구를 어떻게
# 부를지는 인자 이름으로 정하므로 도구 목록이 바뀌어도 그대로 돈다.
def _read_outputs(
    task: Task, seed_engine: Engine, registry: Registry, read_tools: Sequence[str]
) -> list[str]:
    """과제 고객에 대한 읽기 도구의 출력들: 고객 단위 도구를 먼저 부르고, 그 출력에서 찾은 주문 번호마다
    한 번씩 부른다. 도구를 어떻게 부를지는 인자 이름으로 정한다."""
    engine = db.memory_engine(seed_engine)
    ctx = _context(task, enforce_policy=True, verified=True)
    outputs: list[str] = []
    order_ids: list[str] = []
    product_ids: list[str] = []
    try:
        specs = [registry[name] for name in read_tools if name in registry]
        for spec in specs:
            if "customer_id" not in spec.args_model.model_fields:
                continue
            result = execute(registry, engine, ctx, spec.name, {"customer_id": task.customer_id})
            if not result.ok:
                continue
            outputs.append(result.content)
            orders = json.loads(result.content).get("orders", [])
            for order in orders if isinstance(orders, list) else []:
                if isinstance(order, dict) and order.get("order_id") not in (None, *order_ids):
                    order_ids.append(order["order_id"])
        for spec in specs:
            fields = spec.args_model.model_fields
            if "customer_id" in fields or "order_id" not in fields:
                continue
            for order_id in order_ids:
                result = execute(registry, engine, ctx, spec.name, {"order_id": order_id})
                if result.ok:
                    outputs.append(result.content)
                    items = json.loads(result.content).get("items", [])
                    for item in items if isinstance(items, list) else []:
                        if isinstance(item, dict) and item.get("product_id") not in (None, *product_ids):
                            product_ids.append(item["product_id"])
        # 고객이 주문한 상품들: 그 상품의 다른 옵션, 가격, 재고도 에이전트가 알아낼 수 있는 값이다.
        if PRODUCT_READ_TOOL in registry:
            for product_id in product_ids:
                result = execute(registry, engine, ctx, PRODUCT_READ_TOOL, {"product_id": product_id})
                if result.ok:
                    outputs.append(result.content)
    finally:
        engine.dispose()
    return outputs


# 과제 파일이 seed 데이터와 도구에 맞는지 검사해 문제 목록을 돌려준다(빈 목록 = 문제없음). 테스트가 모든 과제
# 파일에 대해 돌린다.
# 검사: 고객이 seed에 있음 / now가 seed 시각보다 뒤 / 전달 값이 시나리오나 에이전트 프롬프트에 새지 않음 /
# 쓰기 대상 주문에 할인이 없음 / 정답이 P1에서 모두 성공 / 금지 동작이 P0·P1을 가름 / 전달 값이 읽기 도구나
# 정답 쓰기의 출력에 있음.
def validate_task(
    task: Task,
    seed_engine: Engine,
    registry: Registry,
    read_tools: Sequence[str] = DEFAULT_READ_TOOLS,
    prompt_texts: Sequence[str] = (),
) -> list[str]:
    """seed 데이터와 도구에 비춘 과제 파일의 문제들. 빈 목록이면 과제에 문제가 없다."""
    problems: list[str] = []
    with seed_engine.connect() as conn:
        customers = db.Customer.__table__
        known = conn.execute(select(customers.c.id).where(customers.c.id == task.customer_id)).first()
    if known is None:
        problems.append(f"{task.id}: customer {task.customer_id} is not in the seed data")
    problems += _seed_time_problems(task, seed_engine)

    # 전달 값이 시나리오나 에이전트 프롬프트(규정 문서의 예시 포함)에 이미 있으면, 조회 없이 그 글을 따라
    # 말해도 통과하므로 아무것도 증명하지 못한다. 2026-09-19 검토에서 규정 문서의 예시 날짜가 거절 과제의 전달
    # 값과 같은 것을 찾아 고친 일이 있다(docs/experiments.md 공통 규칙 과제). 규정 문서의 예시도 꼭 본다.
    for required in task.required_values:
        for source, texts in (("scenario", [task.user.all_text()]), ("agent prompt", list(prompt_texts))):
            if value_found(required, texts):
                problems.append(
                    f"{task.id}: required value {required.label!r} ({required.value}) "
                    f"is given away by the {source}"
                )

    # 할인이 있는 주문은 쓰기 대상으로 쓰지 않는다. 반품 환불액의 할인 안분 규칙이 아직 없기 때문이다.
    with seed_engine.connect() as conn:
        orders = db.Order.__table__
        for action in [*task.gold_actions, *task.forbidden_actions]:
            order_id = action.args.get("order_id")
            row = conn.execute(select(orders.c.discount_won).where(orders.c.id == order_id)).first()
            if row is not None and row[0] != 0:
                problems.append(f"{task.id}: {order_id} has a discount; refund rules do not cover that yet")

    # 정답 재실행이 실패하면 뒤의 검사(금지 동작, 전달 값 출처)는 정답 상태에 기대므로 여기서 멈춘다.
    try:
        gold, gold_outputs = _replay_gold(task, seed_engine, registry)
    except (GoldReplayError, ToolBugError) as error:
        problems.append(str(error))
        return problems  # 남은 검사는 정답 상태 위에서 하므로 여기서 멈춘다
    try:
        problems += _forbidden_problems(task, gold, registry)
    finally:
        gold.dispose()

    try:
        outputs = _read_outputs(task, seed_engine, registry, read_tools) + gold_outputs
    except ToolBugError as error:
        problems.append(f"{task.id}: a read tool hit a bug: {error}")
        return problems
    # 전달 값은 도구로 조회하거나 정답 쓰기를 실행해야만 알 수 있는 값이어야 한다. 어느 출력에도 없으면
    # 에이전트가 알 길이 없는 값이다.
    for required in task.required_values:
        if not value_found(required, outputs):
            problems.append(
                f"{task.id}: required value {required.label!r} ({required.value}) is in no read tool output "
                "for this customer and in no gold action output"
            )
    return problems
