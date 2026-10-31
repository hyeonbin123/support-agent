"""보조 지표: 상담원 답의 '근거 없는 값'. 여기서는 모델을 부르지 않는다. 보고만 하고 판정은 하지 않는다.

고객에게 전달된 상담원 답의 값(이메일 주소, 날짜, 하이픈과 천 단위 쉼표를 뺀 4자리 이상 숫자) 가운데, 그 답
앞까지 에이전트가 받은 글 어디에도 없고 규정이 정한 계산으로도 나오지 않는 것을 센다(docs/experiments.md 끝 절
"보조 지표: 상담원 답의 근거 없는 값"). 정의와 테스트는 어떤 기록에 계산하기 전에 커밋했다:

- 대상: 전달된 상담원 답(judge.delivered_texts와 같은 기준: 첫 인사, 고객에게 가지 않은 답, 도구 호출 메시지,
  빈 답은 뺀다). 값은 sim_report.value_tokens와 같은 방법으로 뽑는다(날짜는 월·일만 비교).
- 출처: 시스템 프롬프트(규정 문서와 오늘 시각. 기록의 manifest 해시로 지금 코드가 만든 글과 같은지 확인한다),
  그 답 앞까지의 고객 말(음성 실행은 인식된 글)·하니스 안내문·도구 결과. 상담원 자신이 앞서 한 말은 출처가
  아니다. 숫자는 더 긴 출처 숫자의 일부여도 출처가 있다(sim_report.grounded).
- 계산 값: 출처의 날짜 + 7일(반품·교환 마감일, rules.return_deadline), + 8일(마감 다음 날), + 30일(쿠폰
  만료), − 29일(쿠폰 한도를 세는 '최근 30일, 오늘 포함'의 첫날). 한 도구 결과 안의 금액(`_won` 값, 단가 ×
  수량)과 규정 금액(반품 배송비, 쿠폰 금액) 2~3개의 합·차. 조합에는 그 도구 결과의 금액이 하나 이상 들어간다.
- 형식 예시('예를 들어 "P-0001" 형식', '(예: O-12345)')와 자리 표시 꼴 숫자(0001, 1234, 0000)는 값으로 읽지
  않는다.
"""

# [파일 역할] 상담원 답의 값이 그 답 앞까지의 출처로 설명되는지 보는 보조 지표의 계산·보고 코드. 실행
# 기록(episodes.jsonl)만 읽고 모델은 부르지 않으므로 CPU에서 돈다.
# [왜 필요한가] 판정(judge.py)은 끝난 DB와 꼭 전달할 값이 있는지만 본다. 그래서 상담원이 고객에게 틀린 금액을
# 함께 말해도(8단계 M2b 개발용 dev-008 시도 3: 도구 결과는 112,300원인데 115,400원이라고도 말함) 성공으로
# 세어진다. 12단계의 시뮬레이터 '지어낸 값' 계수(sim_report.invented_values)와 같은 꼴의 계수를 상담원 쪽으로
# 옮겨, 판정이 보지 못하는 이 잘못이 얼마나 있는지 기록마다 보고한다. 성공률과 판정 규칙은 바꾸지 않는다.
# [설계 이유] 값 뽑기·출처 비교는 sim_report의 함수를 그대로 써서 두 계수가 같은 기준으로 센다. 거친 값
# 비교는 규정대로 계산한 정당한 값(수령일 + 7일, 단가 × 수량, 배송비를 뺀 환불액)을 근거 없음으로 잘못 세므로,
# 규정이 정한 계산을 출처에 더하고, 손 확인(고정 seed 표본 20개)의 정밀도로 정의를 점검한다(docs의 규칙).
# 금액 조합을 한 도구 결과 안으로 묶은 이유: 결과를 가리지 않고 섞으면 금액이 20개만 되어도 세 금액 조합이
# 2천 개를 넘어 100원 단위 금액의 대부분(무작위 금액 20개로 세면 1,000~200,000원의 약 78%)이 우연히 맞아, 실제
# 잘못을 세지 못한다. 한 결과로 묶어도 dev-008의 get_order 결과 하나와 규정 금액으로 약 200개가 만들어진다.
# [주요 구성] (1) 값과 형식 예시 (2) 계산 값(날짜, 금액) (3) 대화 하나: 전달된 답마다 근거 없는 값 (4) 실행
# 단위 표와 개발용 합계 (5) 손 확인 표본(개발용 기록만, 시험용은 수만).
# [연결] analyze.py의 grounding 명령이 grounding_table과 hand_sheet를 부른다. 기록 읽기·성공 판정·시험용
# 판별은 analyze, 값 뽑기·출처 비교·과제 목록은 sim_report, 날짜 꼴은 judge, 규정 상수는 rules를 쓴다.

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

# 다른 모듈과 같은 정의를 그대로 쓴다: 규정 상수(rules), 시스템 프롬프트를 만드는 함수와 규정 문서(agent),
# 기록 읽기·지금 규칙의 성공·시험용 판별(analyze), KST 날짜(clock), 고정 첫 인사(config), 판정의 날짜
# 꼴(judge), 값 뽑기·출처 비교·정규화·과제 목록·비율 표기(sim_report). 계수가 판정·12단계 계수와 다른 기준으로
# 세지 않게 하려는 것이다.
from support_agent import rules
from support_agent.agent import build_system_prompt, load_policy
from support_agent.analyze import is_success, is_test_run, load_episodes, load_manifest
from support_agent.clock import kst_date
from support_agent.config import FIRST_AGENT_MESSAGE

# 같은 대화를 알아보는 메시지 비교 값: 공통 규칙 "음성 대조"의 '대화'와 같은 기준(역할, 글, 도구 이름, 전달
# 여부, 하니스 안내문 여부, 도구 호출 이름과 인자. 실행마다 새로 붙는 도구 호출 id는 뺀다).
from support_agent.guard_report import _message as comparable_message
from support_agent.judge import extract_dates
from support_agent.sim_report import _all_tasks, _normal, _share, grounded, source_tokens, value_tokens
from support_agent.tasks import Task

# 손 확인 표본의 seed와 크기(docs의 규칙: seed 20261024로 무작위 20개). seed는 compare의 순열 검정 seed와 같은
# 값이다.
GROUNDING_SEED = 20261024
SHEET_SIZE = 20
# 출처 날짜에서 계산하는 날(일). 규정 5·6·8절의 셈만 넣는다: 반품·교환 마감일(수령일 + 7,
# rules.return_deadline), 마감 다음 날(규정 예시 '1월 13일부터 불가'), 쿠폰 만료일(발급일 + 30), 쿠폰 한도를
# 세는 '최근 30일(오늘 포함)'의 첫날(규정 예시 '오늘이 3월 30일이면 3월 1일부터'). 이 밖의 셈(내일, 다음 주)은
# 출처로 치지 않는다.
DATE_OFFSETS_DAYS: tuple[int, ...] = (
    rules.RETURN_WINDOW_DAYS,
    rules.RETURN_WINDOW_DAYS + 1,
    rules.COUPON_VALID_DAYS,
    -(rules.COUPON_WINDOW_DAYS - 1),
)
# 금액 조합에 넣는 규정 금액: 반품 배송비, 보상 쿠폰 금액(배송 지연 1~2일·3일 이상, 상품 불량). 환불액 =
# 단가 × 수량의 합 − 반품 배송비(규정 5절)를 반품 접수 전에 미리 계산한 답이 근거 없음으로 세어지지 않게 한다.
# 규정 문서의 다른 숫자(형식 예시 '12,000원')는 조합에 넣지 않는다(글 그대로는 시스템 프롬프트 출처에 있다).
POLICY_AMOUNTS: frozenset[int] = frozenset(
    {
        rules.RETURN_FEE_WON,
        rules.COUPON_DELAY_SHORT_WON,
        rules.COUPON_DELAY_LONG_WON,
        rules.COUPON_DEFECTIVE_WON,
    }
)

# 형식 예시. 정규화한 글(소문자, NFKC)에 쓴다.
# - 앞 표지: '예:', '예)', '예를 들어', '예를 들면', '예시', '예컨대', 'e.g.' 뒤의 글을 문장 끝(. ! ?
#   줄바꿈)이나 닫는 괄호 앞까지, 최대 40자(걸친 낱말은 끝까지) 지운다. 예시 뒤에 같은 문장으로 진짜 값을
#   말하면 그 값도 지워져 덜 세는 쪽으로 틀린다.
# - 뒤 표지: 숫자가 든 낱말 바로 뒤(닫는 따옴표·괄호와 빈칸만 사이에 둠)에 '형식', '꼴', '처럼'이 오면 그
#   낱말을 지운다('"P-0001" 형식으로', 'O-91234처럼'). '지난번처럼'처럼 숫자가 없는 낱말은 건드리지 않는다.
_EXAMPLE_LEAD = re.compile(
    r"(?:예\s*[:)]|예를\s*들(?:어|면)|예시|예컨대|e\.g\.)[^.!?\n)\]」』]{0,40}[^\s.!?)\]」』]*"
)
_EXAMPLE_TRAIL = re.compile(r"[^\s]*\d[^\s]*?[\"'”’」』)\]]*\s*(?:형식|꼴|처럼)")
# 자리 표시 꼴의 앞자리 0 채움 숫자(0001, 0012, 00001).
_ZERO_PADDED = re.compile(r"0{2,}\d{1,2}")


# ---------------------------------------------------------------- 값과 형식 예시


def is_placeholder(digits: str) -> bool:
    """자리 표시 꼴 숫자: 한 숫자만 되풀이(0000, 1111), 이어지는 오름·내림(1234, 56789, 9876), 앞을 0으로
    채운 작은 수(0001, 0012). 출처로 설명되지 않는 숫자에만 쓴다(출처에 있는 진짜 값은 이 검사 전에 출처가
    있다)."""
    return (
        len(set(digits)) == 1
        or digits in "0123456789"
        or digits in "1234567890"
        or digits in "9876543210"
        or _ZERO_PADDED.fullmatch(digits) is not None
    )


def answer_values(text: str) -> set[str]:
    """상담원 답 하나에서 볼 값(email:…, date:MM-DD, num:<4자리 이상 숫자>). 형식 예시 부분을 지운 뒤
    sim_report.value_tokens로 뽑는다."""
    normal = _normal(text)
    for pattern in (_EXAMPLE_LEAD, _EXAMPLE_TRAIL):
        normal = pattern.sub(" ", normal)
    return value_tokens(normal)


# ---------------------------------------------------------------- 계산 값


def computed_dates(text: str, today: date) -> set[str]:
    """글 속 날짜마다 DATE_OFFSETS_DAYS만큼 옮긴 날을 date:MM-DD로. 연도가 없는 날짜('9월 11일')는 오늘의
    연도로 보고, 있을 수 없는 날('2월 30일')은 건너뛴다. 날짜 꼴은 판정과 같다(judge.extract_dates)."""
    out: set[str] = set()
    for year, month, day in extract_dates(_normal(text)):
        try:
            base = date(year or today.year, month, day)
        except ValueError:
            continue
        for days in DATE_OFFSETS_DAYS:
            moved = base + timedelta(days=days)
            out.add(f"date:{moved.month:02d}-{moved.day:02d}")
    return out


# bool은 int의 하위 형이라 따로 뺀다(JSON의 true가 금액 1로 세어지지 않게).
def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def tool_amounts(content: str) -> frozenset[int]:
    """도구 결과 하나의 금액: 이름이 `_won`으로 끝나는 키의 정수 값과, 주문 줄(unit_price_won과 quantity가
    함께 있는 dict)의 단가 × 수량. 0원은 뺀다. JSON이 아닌 결과(`Error: [코드] …` 오류 글)에는 금액이
    없다(E1이 붙이는 사실 괄호도 날짜만 담는다)."""
    try:
        data = json.loads(content)
    except ValueError:
        return frozenset()
    found: set[int] = set()
    stack = [data]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            price, quantity = node.get("unit_price_won"), node.get("quantity")
            if _whole(price) and _whole(quantity):
                found.add(price * quantity)
            for key, value in node.items():
                if key.endswith("_won") and _whole(value):
                    found.add(value)
                stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return frozenset(amount for amount in found if amount > 0)


def by_combination(value: int, results: Iterable[frozenset[int]]) -> bool:
    """금액 value가 한 도구 결과의 금액(단가 × 수량 포함)이거나, 그 결과의 금액 하나 이상과 규정 금액으로 만든
    서로 다른 2~3개 금액의 합·차(±a ± b, ±a ± b ± c)인가. 결과끼리는 섞지 않는다(위 설계 이유)."""
    for own in results:
        if value in own:
            return True
        pool = own | POLICY_AMOUNTS
        for first in own:
            for second in pool - {first}:
                # value − (±first) − (±second)가 0이면 두 금액, 남은 값이 pool의 다른 금액이면 세 금액의
                # 조합이다.
                for rest in (
                    value - first - second,
                    value - first + second,
                    value + first - second,
                    value + first + second,
                ):
                    if rest == 0 or (abs(rest) in pool and abs(rest) not in (first, second)):
                        return True
    return False


# ---------------------------------------------------------------- 대화 하나


class Answer(NamedTuple):
    """전달된 상담원 답 하나: 메시지 번호, 글, 근거 없는 값(정렬, 없으면 빈 튜플)."""

    index: int
    text: str
    missing: tuple[str, ...]


def delivered_answers(messages: Sequence[dict[str, Any]]) -> list[tuple[int, str]]:
    """고객에게 전달된 상담원 답의 (메시지 번호, 글). judge.delivered_texts와 같은 기준이다: 맨 앞 고정 첫
    인사, 전달하지 않은 답(delivered=False), 도구 호출 메시지, 빈 답은 뺀다(테스트가 기록에서 둘을 맞춰
    본다)."""
    out: list[tuple[int, str]] = []
    first = True
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        content = message.get("content") or ""
        greeting = first and content == FIRST_AGENT_MESSAGE
        first = False
        if greeting or not message.get("delivered", True) or message.get("tool_calls") or not content.strip():
            continue
        out.append((index, content))
    return out


# 시스템 프롬프트의 출처 값(글 그대로와 그 날짜에서 계산한 날). 개발용 과제는 시각이 모두 같아 프롬프트가
# 하나이므로 기억해 두고 다시 쓴다.
@lru_cache(maxsize=16)
def _prompt_sources(system_prompt: str, today: date) -> frozenset[str]:
    return frozenset(source_tokens([system_prompt]) | computed_dates(system_prompt, today))


# 값 하나가 출처로 설명되는가: 출처에 있음(더 긴 숫자의 일부 포함, 계산한 날짜 포함), 자리 표시 꼴 숫자, 금액
# 조합. 0으로 시작하는 숫자(전화번호 등)는 금액이 아니므로 조합을 보지 않는다.
def _explained(token: str, sources: set[str], results: list[frozenset[int]]) -> bool:
    if grounded(token, sources):
        return True
    if not token.startswith("num:"):
        return False
    digits = token[4:]
    return is_placeholder(digits) or (not digits.startswith("0") and by_combination(int(digits), results))


def check_answers(messages: Sequence[dict[str, Any]], system_prompt: str, today: date) -> list[Answer]:
    """대화 하나(시스템 프롬프트를 뺀 에이전트 기록의 메시지)의 전달된 답마다 근거 없는 값. 출처는 시스템
    프롬프트와, 그 답 앞까지의 고객 말·하니스 안내문(role user)과 도구 결과(role tool)다. today는 과제의 KST
    날짜로, 연도 없는 날짜를 읽을 때 쓴다."""
    sources = set(_prompt_sources(system_prompt, today))
    results: list[frozenset[int]] = []
    delivered = dict(delivered_answers(messages))
    answers: list[Answer] = []
    for index, message in enumerate(messages):
        if index in delivered:
            text = delivered[index]
            missing = sorted(v for v in answer_values(text) if not _explained(v, sources, results))
            answers.append(Answer(index, text, tuple(missing)))
        elif message["role"] in ("user", "tool"):
            content = message.get("content") or ""
            sources |= source_tokens([content]) | computed_dates(content, today)
            if message["role"] == "tool":
                results.append(tool_amounts(content))
    return answers


# ---------------------------------------------------------------- 실행 하나


class Checked(NamedTuple):
    """판정된 에피소드 하나의 결과. key는 같은 대화를 알아보는 값(과제, 시도, 메시지마다 comparable_message.
    도구 호출 id는 빼므로 다른 실행에서 같은 대화가 나와도 같은 값이 된다)이다."""

    task_id: str
    trial: int
    success: bool
    answers: list[Answer]
    messages: list[dict[str, Any]]
    key: str


class RunResult(NamedTuple):
    """실행 하나: 이름, 시험용인가, 판정된 에피소드들, infra 오류 수, 과제 해시가 지금 파일과 다른 에피소드
    수."""

    name: str
    test: bool
    episodes: list[Checked]
    infra: int
    changed_tasks: int


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# 같은 대화의 판별 값: (과제, 시도, 메시지마다 comparable_message)를 JSON으로. 도구 호출 id를 넣으면 실행마다
# 새로 붙는 id 때문에 장치가 작동하지 않아 대화가 같은 두 실행(P1·7B의 Ollama 0.34.2·0.35.1 두 기록 등)이
# 다른 대화로 세어진다(docs/experiments.md 이 지표 절의 2026-11-01 바로잡음). tail은 판별에 함께 넣을 값(답
# 글)이다.
def _conversation_key(task_id: str, trial: int, messages: Sequence[dict[str, Any]], *tail: Any) -> str:
    return json.dumps([task_id, trial, [comparable_message(m) for m in messages], *tail], ensure_ascii=False)


def check_run(run_dir: Path, tasks: dict[str, Task] | None = None) -> RunResult | None:
    """실행 하나의 판정된 에피소드마다 전달된 답을 본다. 시스템 프롬프트를 지금 코드(규정 문서, agent.md 틀,
    과제 시각, V4 규칙 줄)로 다시 만들어 쓰므로, manifest의 agent_system 해시가 그 글과 다르면 None(세지
    않음). infra 오류는 세기만 하고, 기록의 과제 해시가 지금 과제 파일과 다르면(과제 시각이 달랐을 수 있음)
    빼고 '과제 바뀜'으로 센다."""
    tasks = tasks if tasks is not None else _all_tasks()
    manifest = load_manifest(run_dir)
    caller_id = (manifest.get("config") or {}).get("voice") == "V4"
    policy = load_policy()
    first = next(iter(manifest.get("task_sha256") or {}), None)
    recorded = (manifest.get("prompt_sha256") or {}).get("agent_system")
    if (
        first not in tasks
        or _sha256(build_system_prompt(policy, tasks[first].now, caller_id=caller_id)) != recorded
    ):
        return None
    hashes = {task_id: task.sha256() for task_id, task in tasks.items()}
    prompts: dict[str, str] = {}
    episodes: list[Checked] = []
    infra = changed = 0
    for episode in load_episodes(run_dir):
        if episode["status"] == "infra_error":
            infra += 1
            continue
        task = tasks.get(episode["task_id"])
        if task is None or hashes[task.id] != episode.get("task_sha256"):
            changed += 1
            continue
        if task.id not in prompts:
            prompts[task.id] = build_system_prompt(policy, task.now, caller_id=caller_id)
        messages = episode["messages"]
        answers = check_answers(messages, prompts[task.id], kst_date(task.now))
        key = _conversation_key(task.id, episode["trial"], messages)
        episodes.append(Checked(task.id, episode["trial"], is_success(episode), answers, messages, key))
    return RunResult(run_dir.name, is_test_run(run_dir), episodes, infra, changed)


# 여러 실행을 한 번에 본다. 에피소드 기록이 없는 폴더(음성 인식만 잰 11단계 실행 등)와 시스템 프롬프트가 다른
# 기록은 이유 한 줄로 돌려준다. 과제 목록은 한 번만 읽는다.
def check_runs(run_dirs: Sequence[Path]) -> tuple[list[RunResult], list[str]]:
    tasks = _all_tasks()
    results: list[RunResult] = []
    notes: list[str] = []
    for run_dir in run_dirs:
        if not (run_dir / "episodes.jsonl").exists():
            notes.append(f"- {run_dir.name}: 에피소드 기록 없음")
            continue
        result = check_run(run_dir, tasks)
        if result is None:
            notes.append(
                f"- {run_dir.name}: 시스템 프롬프트가 지금 코드로 만든 것과 달라 세지 않음 "
                "(manifest의 agent_system 해시)"
            )
            continue
        results.append(result)
    return results, notes


# 에피소드 묶음의 수: 에피소드, 전달된 답, 근거 없는 값이 든 답, 값의 종류별 수, 그런 에피소드, 성공
# 에피소드와 그 가운데 근거 없는 값이 든 것.
def _counts(episodes: Iterable[Checked]) -> dict[str, Any]:
    c: dict[str, Any] = dict.fromkeys(
        ["episodes", "answers", "flagged_answers", "flagged_episodes", "successes", "flagged_successes"], 0
    )
    kinds: Counter[str] = Counter()
    for episode in episodes:
        flagged = [a for a in episode.answers if a.missing]
        c["episodes"] += 1
        c["answers"] += len(episode.answers)
        c["flagged_answers"] += len(flagged)
        c["flagged_episodes"] += bool(flagged)
        c["successes"] += episode.success
        c["flagged_successes"] += episode.success and bool(flagged)
        kinds.update(token.split(":", 1)[0] for a in flagged for token in a.missing)
    c["kinds"] = kinds
    return c


# 표의 머리줄(한국어, docs/experiments.md에 그대로 붙여 넣는 형식). 근거 없는 값 칸은 값 수와
# (숫자·날짜·이메일).
HEADER = (
    "| 실행 | 에피소드 | 전달된 답 | 근거 없는 값이 든 답 | 근거 없는 값 (숫자·날짜·이메일) | 그런 에피소드 "
    "| 성공 에피소드 중 | 과제 바뀜 · infra |"
)


def _row(result: RunResult) -> str:
    c = _counts(result.episodes)
    kinds = c["kinds"]
    values = sum(kinds.values())
    name = result.name + (" (시험용)" if result.test else "")
    return (
        f"| {name} | {c['episodes']} | {c['answers']} | {_share(c['flagged_answers'], c['answers'])} "
        f"| {values} ({kinds['num']}·{kinds['date']}·{kinds['email']}) "
        f"| {_share(c['flagged_episodes'], c['episodes'])} "
        f"| {_share(c['flagged_successes'], c['successes'])} "
        f"| {result.changed_tasks} · {result.infra} |"
    )


# 개발용 기록을 합친 수. 같은 대화(과제, 시도, 도구 호출 id를 뺀 메시지 전체가 같음. _conversation_key)는 한
# 번만 센다: 끼워 넣은 기록(analyze stitch)은 원래 기록과 다시 잰 기록의 에피소드를 그대로 담고, 설정이 같거나
# 다른 장치가 한 번도 작동하지 않아 대화가 같은 기록도 있다.
def _development_total(results: Sequence[RunResult]) -> str | None:
    dev = [r for r in results if not r.test]
    if not dev:
        return None
    unique: dict[str, Checked] = {}
    for result in dev:
        for episode in result.episodes:
            unique.setdefault(episode.key, episode)
    c = _counts(unique.values())
    return (
        f"- 개발용 기록 {len(dev)}개 합계 (같은 대화는 한 번): 에피소드 {c['episodes']}, "
        f"전달된 답 {c['answers']}, 근거 없는 값이 든 답 {_share(c['flagged_answers'], c['answers'])}, "
        f"그런 에피소드 {_share(c['flagged_episodes'], c['episodes'])}, "
        f"성공 에피소드 중 {_share(c['flagged_successes'], c['successes'])}"
    )


def grounding_table(
    run_dirs: Sequence[Path], checked: tuple[list[RunResult], list[str]] | None = None
) -> str:
    """analyze grounding: 기록마다 한 행. 시험용 기록은 이름에 '(시험용)'을 붙여 수만 낸다(대화 글은 내지
    않음). 표 아래에 세지 않은 폴더의 이유와 개발용 합계를 적는다. checked는 check_runs의 결과(이미
    계산했으면)."""
    results, notes = checked if checked is not None else check_runs(run_dirs)
    lines = [HEADER, "|---|" + "---|" * (HEADER.count("|") - 2)]
    lines += [_row(result) for result in results]
    total = _development_total(results)
    below = notes + ([total] if total else [])
    return "\n".join(lines + ([""] + below if below else []))


# ---------------------------------------------------------------- 손 확인 표본 (개발용 기록만)


class PoolItem(NamedTuple):
    """손 확인 모집단의 한 항목: 근거 없는 값이 든 전달된 답 하나와, 그 답 앞까지의 메시지."""

    run: str
    task_id: str
    trial: int
    success: bool
    answer: Answer
    context: tuple[dict[str, Any], ...]


def flagged_pool(
    run_dirs: Sequence[Path], checked: tuple[list[RunResult], list[str]] | None = None
) -> list[PoolItem]:
    """개발용 기록의 근거 없는 값이 든 답 전부를 실행 이름 → 기록 순서 → 메시지 순서로. 시험용 기록은 넣지
    않는다(시험용 기록으로는 아무것도 만들지 않는다). 같은 대화의 같은 답(과제·시도와 그 답까지의 메시지가
    도구 호출 id를 빼고 같음)은 한 번만 넣는다(끼워 넣은 기록, 대화가 같은 기록)."""
    results, _ = checked if checked is not None else check_runs(run_dirs)
    seen: set[str] = set()
    pool: list[PoolItem] = []
    for result in sorted((r for r in results if not r.test), key=lambda r: r.name):
        for episode in result.episodes:
            for answer in episode.answers:
                if not answer.missing:
                    continue
                context = tuple(episode.messages[: answer.index])
                key = _conversation_key(episode.task_id, episode.trial, context, answer.text)
                if key in seen:
                    continue
                seen.add(key)
                pool.append(
                    PoolItem(result.name, episode.task_id, episode.trial, episode.success, answer, context)
                )
    return pool


def draw(pool: Sequence[PoolItem], n: int, seed: int = GROUNDING_SEED) -> list[PoolItem]:
    """모집단에서 고정 seed로 n개(모집단이 작으면 전부)를 뽑아 모집단 순서로 돌려준다. 모듈 수준 random이
    아니라 seed를 준 random.Random을 써서 같은 기록은 언제나 같은 표본이 된다."""
    picked = sorted(random.Random(seed).sample(range(len(pool)), min(n, len(pool))))
    return [pool[i] for i in picked]


# 표본 항목에 붙이는 앞 대화 한 줄: 누가 한 말인지와 글(줄바꿈은 ' / '). 상담원의 앞선 답은 출처가 아니라고
# 표시한다. 도구 호출 메시지와 전달하지 않은 답은 에이전트 안의 일이라 적지 않는다.
def _context_line(message: dict[str, Any]) -> str | None:
    text = " / ".join((message.get("content") or "").splitlines())
    if message["role"] == "tool":
        return f"  - 도구 {message.get('tool_name') or '?'}: {text}"
    if message["role"] == "user":
        return f"  - {'안내문' if message.get('harness') else '고객'}: {text}"
    if message.get("delivered", True) and not message.get("tool_calls") and text.strip():
        return f"  - 상담원 (출처 아님): {text}"
    return None


def hand_sheet(
    run_dirs: Sequence[Path], n: int = SHEET_SIZE, checked: tuple[list[RunResult], list[str]] | None = None
) -> str:
    """손 확인 표본(마크다운): 개발용 기록의 근거 없는 값이 든 답에서 고정 seed로 n개. 항목마다 걸린 값, 답,
    그 답 앞까지 에이전트가 받은 글을 적는다. 시험용 기록은 표본에 넣지 않고 그 수만 적는다."""
    checked = checked if checked is not None else check_runs(run_dirs)
    results = checked[0]
    pool = flagged_pool(run_dirs, checked)
    tests = sum(r.test for r in results)
    lines = [
        f"# 손 확인 표본: 상담원 답의 근거 없는 값 (analyze grounding --sheet, seed {GROUNDING_SEED}, n={n})",
        "",
        f"- 모집단: 개발용 기록 {len(results) - tests}개에서 근거 없는 값이 든 전달된 답 {len(pool)}개 "
        "(같은 대화의 같은 답은 한 번)",
    ]
    if tests:
        lines.append(f"- 시험용 기록 {tests}개는 표본에 넣지 않았다 (세기만 한다)")
    lines += [
        "- 적는 법: 항목마다 '근거 없음'(걸린 값 가운데 하나라도 그 답 앞까지 에이전트가 받은 글에 없고, "
        "그 글에서 맞게 셈한 값(금액의 합·차·곱, 날짜 셈)도 아니며, 형식 예시나 가정으로 든 예도 아님) 또는 "
        "'오탐'(걸린 값이 모두 그 글에 있거나 그 글에서 맞게 셈한 값, 그 이유)을 적는다. 값이 그 자리에 맞는 "
        "말인지는 보지 않는다(틀리게 셈한 값은 근거 없음). 정밀도 = 근거 없음 / 항목 수",
    ]
    for number, item in enumerate(draw(pool, n), start=1):
        outcome = "성공" if item.success else "실패"
        place = f"{item.run} · {item.task_id} 시도 {item.trial} · 메시지 {item.answer.index}"
        lines += [
            "",
            f"## {number}. {place} ({outcome})",
            f"- 걸린 값: {', '.join(item.answer.missing)}",
            f"- 답: {' / '.join(item.answer.text.splitlines())}",
            "- 그 앞까지의 대화:",
        ]
        lines += [line for line in map(_context_line, item.context) if line]
    return "\n".join(lines) + "\n"
