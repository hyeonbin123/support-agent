"""Stage 12: how much the simulated customer decides a score. Nothing here calls a model.

Automatic counts of simulator slips (P5(b) of the 2026-10-02 review), read from the episode records of any
run, and the report that pairs a run with another simulator against the same setup with the 7B simulator
(docs/experiments.md, stage 12). The counts were written down, with tests, before they were computed on any
record:

- stop before values: a task with required values (전달 값) that the simulator ended with its STOP token while
  some required value was not yet in what the agent had said to it (the judge's matcher, `value_found`).
- invented values: a value in a simulator utterance (an e-mail address, a date, a number of MIN_DIGITS or more
  digits once hyphens and thousands commas are dropped) that is in none of the simulator's own prompt (its
  scenario and today's date), what the agent had said to it so far, and the tool results of the episode so
  far. A number is also found when it is part of a longer one (the last digits of a phone number).

Reported only: a STOP with a gold write left, utterances in another script, simulator format problems.

Rule change of 2026-10-09 (after the counts on the earlier records, before the GPU run): the next-round rule
reads "stop before values" as a share of the episodes that never heard every value (an agent that delivers
more values lowers the plain count by itself), and asks U1 for no more invented values than U0 (U0 has none).
"""

# [파일 역할] 12단계(시뮬레이터 민감도 점검)의 집계·보고 코드. 실행 기록(episodes.jsonl)만 읽어 시뮬레이터
# 고객의 잘못을 자동으로 세고, 고객 역할 모델만 바꾼 실행(U1)을 7B 시뮬레이터 기록(U0)과 과제 단위로 짝지어
# 비교표를 만든다. 모델은 부르지 않으므로 CPU에서 돈다.
# [왜 필요한가] 같은 에이전트라도 고객 역할 모델에 따라 점수가 움직일 수 있다(Lost in Simulation 보고). 이
# 프로젝트의 수치가 시뮬레이터 선택에 얼마나 기대는지 보려고, 사람이 읽는 점검 대신 미리 정의한 자동 계수를
# 1차로 쓴다(docs/experiments.md 12단계). 계수의 정의와 테스트는 어떤 기록에 계산하기 전에
# 커밋했다(docstring).
# [주요 구성] (1) 값 뽑기: 발화에서 이메일·날짜·4자리 이상 숫자를 찾고 출처에 있는지 본다 (2) 에피소드 단위
# 계수: 값 전 STOP, 값을 끝내 못 들음, 쓰기가 남은 채 STOP, 지어낸 값, 형식 문제 (3) 실행 단위 집계와 표 (4)
# 스모크 관문 (5) U0·U1 짝 보고서와 다음 라운드 후보 규칙.
# [연결] analyze.py의 sim-counts, sim-smoke, sim-check 명령이 counts_table, smoke_report, check를 부른다.

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# 다른 모듈과 같은 정의를 그대로 쓴다: 한자·가나 판정(agent의 언어 가드), 기록 읽기·짝지은
# 부트스트랩·pass^k·설정 차이(analyze), 종료 토큰 떼기(episode.split_ending), 판정의 전달 값 일치와 날짜
# 꼴(judge), 과제 해시와 시뮬레이터 프롬프트(tasks, user_sim). 계수가 판정과 다른 기준으로 세지 않게 하려는
# 것이다.
from support_agent.agent import in_another_language
from support_agent.analyze import (
    is_test_run,
    load_episodes,
    load_manifest,
    load_task_types,
    paired_difference,
    pass_k,
    setup_differences,
    successes_by_task,
)
from support_agent.analyze import (
    smoke_gates as stage8_smoke_gates,
)
from support_agent.episode import split_ending
from support_agent.judge import _KOREAN_DATE, _NUMERIC_DATE, value_found  # the judge's date forms
from support_agent.paths import TASKS
from support_agent.tasks import Task, load_tasks
from support_agent.user_sim import build_user_prompt

# 지어낸 값으로 셀 숫자의 최소 자릿수. 1~3자리(수량, 날짜의 일, 신발 크기, 시각)는 정당한 경우가 많아 오탐이
# 잦으므로 읽지 않는다. 그래서 이 계수는 지어낸 값을 덜 센다(docs/experiments.md 12단계 "자동 계수").
MIN_DIGITS = 4  # shorter numbers (quantities, days, sizes, times) are too often legitimate to count
# 시뮬레이터 프롬프트가 num_ctx의 95%를 넘으면 앞부분이 잘렸을 수 있다고 보고 그 호출 수를 보고한다.
# 시뮬레이터에는 컨텍스트 한도 종료가 없어서다.
PROMPT_LIMIT_SHARE = 0.95  # a simulator prompt above this share of its num_ctx may have lost its start
# 값 패턴: 이메일(소문자로 바꾼 뒤 찾음), 하이픈이나 천 단위 쉼표로 이어진 숫자 묶음(전화·주문·송장 번호,
# 금액). _HYPHENS는 여러 종류의 하이픈 닮은 문자를 '-' 하나로 맞추는 데 쓴다.
_EMAIL = re.compile(r"[0-9a-z._%+\-]+@[0-9a-z\-]+(?:\.[0-9a-z\-]+)*\.[a-z]{2,}")
# Digit groups joined by hyphens (phone, order and tracking numbers) or by thousands commas (amounts).
_NUMBER = re.compile(r"\d+(?:-\d+|,\d{3}(?!\d))*")
_HYPHENS = re.compile("[\u2010-\u2015\u2212\ufe63\uff0d]")
# 시뮬레이터 답에 사고 태그가 새어 나왔는지(think=false가 먹지 않음).
_THINK_TAG = re.compile(r"</?think>")
# What is left of an end token after the valid ones are taken off: "네 감사합니다 ###", "STOP".
_END_LEFTOVER = re.compile(r"#{2,}|(?<![A-Za-z])STOP(?![A-Za-z])|OUT[-_ ]?OF[-_ ]?SCOPE", re.IGNORECASE)
# 보고하는 형식 문제의 종류: 사고 태그, 256토큰 한도에서 잘림, 도구 호출(시뮬레이터에는 도구를 주지 않음),
# 떼고 남은 종료 토큰 조각.
FORMAT_KINDS = ("think_tag", "cut_off", "tool_call", "broken_end")


# ---------------------------------------------------------------- values


# 비교 전 정규화: NFKC(전각 숫자 등), 하이픈 통일, 소문자.
def _normal(text: str) -> str:
    return _HYPHENS.sub("-", unicodedata.normalize("NFKC", text)).lower()


# 판정과 같은 날짜 꼴(2026-09-15, 2026년 9월 15일, 9월 15일)을 "date:MM-DD"로 뽑고, 날짜를 지운 나머지 글도
# 돌려준다. 날짜 속 숫자가 다시 숫자 값으로 세어지지 않게 하려는 것이다. 있을 수 없는 월·일은 버린다.
def _dates_and_rest(text: str) -> tuple[set[str], str]:
    """Dates as month-day (the year is left out: a customer rarely says it) and the text without them."""
    dates: set[str] = set()
    for pattern in (_NUMERIC_DATE, _KOREAN_DATE):
        for match in pattern.finditer(text):
            month, day = int(match.group(2)), int(match.group(3))
            if 1 <= month <= 12 and 1 <= day <= 31:
                dates.add(f"date:{month:02d}-{day:02d}")
        text = pattern.sub(" ", text)
    return dates, text


# 숫자 묶음에서 하이픈·쉼표를 뺀 숫자열이 min_digits 이상이면 "num:숫자"로 모은다.
def _numbers(text: str, min_digits: int) -> set[str]:
    out = set()
    for raw in _NUMBER.findall(text):
        digits = raw.replace("-", "").replace(",", "")
        if len(digits) >= min_digits:
            out.add(f"num:{digits}")
    return out


# 시뮬레이터 발화 하나에서 "지어냈는지" 볼 값들. 숫자는 MIN_DIGITS 이상만 본다.
def value_tokens(text: str) -> set[str]:
    """The values in one simulator utterance: email:..., date:MM-DD, num:<digits> (MIN_DIGITS or more)."""
    text = _normal(text)
    emails = {f"email:{e}" for e in _EMAIL.findall(text)}
    dates, rest = _dates_and_rest(_EMAIL.sub(" ", text))
    return emails | dates | _numbers(rest, MIN_DIGITS)


# 발화가 되풀이해도 되는 값(출처)들. 출처 쪽은 숫자를 길이 제한 없이 모으고, 날짜 안의 숫자도 숫자로 함께 남겨
# 넓게 잡는다(오탐을 줄이는 쪽).
def source_tokens(texts: Iterable[str]) -> set[str]:
    """The values an utterance may repeat: numbers of any length, the digits inside dates kept as well."""
    out: set[str] = set()
    for text in texts:
        text = _normal(text)
        out |= {f"email:{e}" for e in _EMAIL.findall(text)}
        rest = _EMAIL.sub(" ", text)
        out |= _dates_and_rest(rest)[0]
        out |= _numbers(rest, 1)
    return out


# 값이 출처에 있는지. 숫자는 더 긴 출처 숫자의 일부여도 있는 것으로 본다(고객이 전화번호 끝 네 자리만 말하는
# 경우).
def grounded(token: str, sources: set[str]) -> bool:
    if token in sources:
        return True
    if token.startswith("num:"):
        digits = token[4:]
        return any(s.startswith("num:") and digits in s[4:] for s in sources)
    return False


# ---------------------------------------------------------------- one episode


# 시뮬레이터 호출마다 (호출 기록, 그때까지 상담원이 시뮬레이터에게 한 말들, 그 발화가 에이전트 기록에 들어가기
# 전까지의 도구 결과)를 만든다.
# user_messages는 시뮬레이터 쪽 기록이라 role이 user인 것이 상담원의 말이다. k번째 호출은 상담원 말 k+1개를 본
# 상태에서 답했다.
def _simulator_turns(episode: dict[str, Any]) -> list[tuple[dict[str, Any], list[str], list[str]]]:
    """Each simulator call with the agent texts it had seen and the tool results that came before it."""
    calls = sorted((c for c in episode["llm_calls"] if c["who"] == "user"), key=lambda c: c["index"])
    seen = [m["content"] for m in episode["user_messages"] if m["role"] == "user"]
    messages = episode["messages"]
    # Where each simulator reply entered the agent's history; a bare end token (the last reply) never did.
    # 에이전트 기록에서 고객 발화가 들어간 자리(하니스 안내문 제외). k번째 시뮬레이터 답은 arrived[k]에
    # 들어갔으므로 그 앞의 도구 결과만 출처로 친다. 종료 토큰만 있던 마지막 답은 들어가지 않았으므로 기록
    # 끝까지를 본다.
    arrived = [i for i, m in enumerate(messages) if m["role"] == "user" and not m.get("harness")]
    results = [(i, m["content"]) for i, m in enumerate(messages) if m["role"] == "tool"]
    turns = []
    for k, call in enumerate(calls):
        before = arrived[k] if k < len(arrived) else len(messages)
        turns.append((call, seen[: k + 1], [content for i, content in results if i < before]))
    return turns


# 시뮬레이터 답에서 종료 토큰을 뗀 말 부분.
def _said(call: dict[str, Any]) -> str:
    return split_ending(call.get("text") or "")[0]


# 지어낸 값 계수(12단계 1차 계수 둘): 발화의 값 가운데 출처 세 곳 — 시뮬레이터 자신의 프롬프트(시나리오와 오늘
# 날짜), 그때까지 상담원이 한 말, 그 전까지의 도구 결과 — 어디에도 없는 것을 발화마다 모은다. 도구 결과는
# 시뮬레이터가 보지 않지만 오탐을 줄이려고 출처에 넣는다(docs/experiments.md).
def invented_values(
    episode: dict[str, Any], task: Task, prompt: set[str] | None = None
) -> list[tuple[int, list[str]]]:
    """(simulator call index, values found nowhere it could have them from) for each utterance with any."""
    if prompt is None:
        prompt = source_tokens([build_user_prompt(task)])
    out = []
    for call, seen, results in _simulator_turns(episode):
        values = value_tokens(_said(call))
        if not values:
            continue
        sources = prompt | source_tokens(seen) | source_tokens(results)
        missing = sorted(v for v in values if not grounded(v, sources))
        if missing:
            out.append((call["index"], missing))
    return out


# 값 전 STOP(1차 계수 하나): 전달 값이 있는 과제에서 시뮬레이터가 STOP으로 끝냈는데, 그때까지 상담원이
# 시뮬레이터에게 한 말에 전달 값이 하나라도 없었는가. 판정과 같은 value_found로 본다. 전달 값이 없는 과제나
# infra 오류는 None(분모에서 뺌).
def stop_before_values(episode: dict[str, Any], task: Task) -> bool | None:
    """None when the task has no required value (or the episode was an infra error)."""
    if episode["status"] == "infra_error" or not task.required_values:
        return None
    if episode["termination"] != "user_stop":
        return False
    seen = [m["content"] for m in episode["user_messages"] if m["role"] == "user"]
    return not all(value_found(value, seen) for value in task.required_values)


# 끝난 이유와 상관없이 시뮬레이터가 전달 값을 끝내 다 듣지 못했는가. 2026-10-09 규칙 변경으로 "값을 끝내 못
# 들은 에피소드 중 STOP" 비율의 분모가 됐다. 상담원이 값을 더 많이 전하기만 해도 값 전 STOP의 단순 비율이
# 내려가는 문제를 피하려는 것이다.
def values_never_heard(episode: dict[str, Any], task: Task) -> bool | None:
    """True when the simulator never read every required value, however the episode ended."""
    if episode["status"] == "infra_error" or not task.required_values:
        return None
    seen = [m["content"] for m in episode["user_messages"] if m["role"] == "user"]
    return not all(value_found(value, seen) for value in task.required_values)


# 보고용: 정답 쓰기가 있는 과제에서 STOP으로 끝났는데 빠진 쓰기가 있는가(10단계 스모크에서 7B 고객이 상담원이
# 쓰기를 부르기 전에 대화를 끝낸 꼴).
def stop_with_writes_left(episode: dict[str, Any], task: Task) -> bool | None:
    """None when the task has no gold write (or the episode was an infra error)."""
    if episode["status"] == "infra_error" or not task.gold_actions:
        return None
    return episode["termination"] == "user_stop" and episode["verdict"]["missing_writes"] > 0


# 시뮬레이터 답 하나의 형식 문제 목록(FORMAT_KINDS 가운데 해당하는 것).
def format_problems(call: dict[str, Any]) -> list[str]:
    text = call.get("text") or ""
    problems = []
    if _THINK_TAG.search(text):
        problems.append("think_tag")
    if call.get("finish_reason") == "length":
        problems.append("cut_off")  # USER_MAX_TOKENS ran out
    if call.get("tool_calls"):
        problems.append("tool_call")  # the simulator is offered no tools
    if _END_LEFTOVER.search(unicodedata.normalize("NFKC", split_ending(text)[0])):
        problems.append("broken_end")
    return problems


# ---------------------------------------------------------------- one run


# tasks/의 모든 과제 파일(smoke, dev, test)을 id로 묶는다. 실행 기록이 어느 과제 파일에서 나왔는지와 상관없이
# 찾을 수 있게 한다.
def _all_tasks() -> dict[str, Task]:
    return {task.id: task for path in sorted(TASKS.glob("*.yaml")) for task in load_tasks(path)}


# 실행 하나의 계수를 모두 센다. 대상은 판정된 에피소드이고, 기록의 과제 해시가 지금 과제 파일과
# 다르면(시뮬레이터가 받은 시나리오가 지금 것과 다름) 빼고 "과제 바뀜"으로 센다.
# 시뮬레이터 발화 단위로는 한자·가나 섞임, 형식 문제, 프롬프트 토큰 최대값과 95% 한도를 넘은 호출 수를 센다.
def run_counts(run_dir: Path) -> dict[str, Any]:
    manifest = load_manifest(run_dir)
    user = manifest.get("user_provider") or {}
    limit = (user.get("num_ctx") or 0) * PROMPT_LIMIT_SHARE
    tasks = _all_tasks()
    hashes = {task_id: task.sha256() for task_id, task in tasks.items()}
    prompts: dict[str, set[str]] = {}
    counts: dict[str, Any] = {
        "simulator": user.get("model"),
        "num_ctx": user.get("num_ctx"),
        **dict.fromkeys(
            ["episodes", "infra", "changed_tasks", "value_tasks", "stop_before_values", "write_tasks"], 0
        ),
        **dict.fromkeys(
            ["stop_with_writes_left", "utterances", "invented_utterances", "invented_episodes"], 0
        ),
        **dict.fromkeys(["invented_values", "other_language", "format_utterances", "prompts_over_limit"], 0),
        "values_never_heard": 0,
        "max_prompt_tokens": 0,
    }
    formats: Counter[str] = Counter()
    for episode in load_episodes(run_dir):
        if episode["status"] == "infra_error":
            counts["infra"] += 1
            continue
        task = tasks.get(episode["task_id"])
        if task is None or hashes[task.id] != episode.get("task_sha256"):
            counts["changed_tasks"] += 1  # the scenario the simulator had is not the one in the file now
            continue
        counts["episodes"] += 1
        before = stop_before_values(episode, task)
        counts["value_tasks"] += before is not None
        counts["stop_before_values"] += bool(before)
        counts["values_never_heard"] += bool(values_never_heard(episode, task))
        left = stop_with_writes_left(episode, task)
        counts["write_tasks"] += left is not None
        counts["stop_with_writes_left"] += bool(left)
        if task.id not in prompts:
            prompts[task.id] = source_tokens([build_user_prompt(task)])
        invented = invented_values(episode, task, prompts[task.id])
        counts["invented_utterances"] += len(invented)
        counts["invented_episodes"] += bool(invented)
        counts["invented_values"] += sum(len(values) for _, values in invented)
        for call in episode["llm_calls"]:
            if call["who"] != "user":
                continue
            counts["utterances"] += 1
            counts["other_language"] += in_another_language(_said(call))
            problems = format_problems(call)
            formats.update(problems)
            counts["format_utterances"] += bool(problems)
            tokens = call.get("prompt_tokens") or 0
            counts["max_prompt_tokens"] = max(counts["max_prompt_tokens"], tokens)
            counts["prompts_over_limit"] += bool(limit) and tokens > limit
    counts["format"] = {kind: formats[kind] for kind in FORMAT_KINDS if formats[kind]}
    return counts


# "n/d (비율)" 표기와 0으로 나누지 않는 비율 계산.
def _share(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.1%})" if d else f"{n}/0"


def _rate(n: int, d: int) -> float:
    return n / d if d else 0.0


# 계수표의 머리줄(한국어, docs/experiments.md에 그대로 붙여 넣는 형식). count_row가 같은 순서로 칸을 채운다.
COUNT_HEADER = (
    "| 실행 | 시뮬레이터 | 에피소드 | 값 전 STOP | 값을 끝내 못 들은 에피소드 중 STOP "
    "| 지어낸 값이 든 발화 (에피소드당) | 그런 에피소드 "
    "| 지어낸 값 | 쓰기가 남은 채 STOP | 시뮬레이터 발화 | 한자·가나 2자 이상 | 형식 문제 "
    "| 프롬프트 최대 토큰 (한도 95% 넘음) | 과제 바뀜 · infra |"
)


def count_row(name: str, c: dict[str, Any]) -> str:
    formats = ", ".join(f"{kind} {n}" for kind, n in c["format"].items())
    return (
        f"| {name} | {c['simulator']} | {c['episodes']} "
        f"| {_share(c['stop_before_values'], c['value_tasks'])} "
        f"| {_share(c['stop_before_values'], c['values_never_heard'])} "
        f"| {c['invented_utterances']} ({_rate(c['invented_utterances'], c['episodes']):.2f}) "
        f"| {_share(c['invented_episodes'], c['episodes'])} | {c['invented_values']} "
        f"| {_share(c['stop_with_writes_left'], c['write_tasks'])} | {c['utterances']} "
        f"| {_share(c['other_language'], c['utterances'])} "
        f"| {c['format_utterances']}{f' ({formats})' if formats else ''} "
        f"| {c['max_prompt_tokens']} ({c['prompts_over_limit']}) | {c['changed_tasks']} · {c['infra']} |"
    )


# analyze sim-counts: 여러 실행의 계수표. 에피소드 기록이 없는 폴더(음성 인식만 잰 11단계 실행 등)는 표 아래에
# 따로 적는다.
def counts_table(run_dirs: list[Path]) -> str:
    lines = [COUNT_HEADER, "|---|" + "---|" * (COUNT_HEADER.count("|") - 2)]
    skipped = []
    for run_dir in run_dirs:
        if not (run_dir / "episodes.jsonl").exists():
            skipped.append(f"- {run_dir.name}: 에피소드 기록 없음")
            continue
        lines.append(count_row(run_dir.name, run_counts(run_dir)))
    return "\n".join(lines + ([""] + skipped if skipped else []))


# ---------------------------------------------------------------- the smoke run (not a measurement)


# 12단계 스모크 관문: 8단계와 같은 S0(infra 오류 0)·S1(두 모델 모두 GPU, 다시 읽기 없음)과, T1(시뮬레이터 답에
# 사고 태그 0건). 측정이 아니라 하니스 확인이다.
def smoke_gates(run_dir: Path) -> list[tuple[str, str, bool]]:
    """S0 (no infra error) and S1 (both models on the GPU, no reload) of stage 8, and T1: no reasoning tag in
    what the simulator said (think=false honoured)."""
    gates = [gate for gate in stage8_smoke_gates(run_dir) if gate[0] in ("S0", "S1")]
    calls = [
        c for e in load_episodes(run_dir) if e["status"] != "infra_error" for c in e["llm_calls"]
        if c["who"] == "user"
    ]  # fmt: skip
    leaks = sum(bool(_THINK_TAG.search(c.get("text") or "")) for c in calls)
    gates.append(("T1", f"시뮬레이터 답의 think 태그 {leaks}건 / {len(calls)}", leaks == 0))
    return gates


# analyze sim-smoke: 관문표, 계수표, 그리고 에피소드당 시간으로 추정한 개발용 96 에피소드의 예상 시간(GPU 시간
# 상한 1.5시간을 지키는지 보려고).
def smoke_report(run_dir: Path) -> str:
    gates = smoke_gates(run_dir)
    lines = ["| 관문 | 본 것 | 통과 |", "|---|---|---|"]
    lines += [f"| {name} | {detail} | {'예' if ok else '아니오'} |" for name, detail, ok in gates]
    episodes = [e for e in load_episodes(run_dir) if e["status"] != "infra_error"]
    seconds = sum(e["wall_seconds"] for e in episodes) / len(episodes) if episodes else 0.0
    failed = [name for name, _, ok in gates if not ok]
    lines += [
        "",
        COUNT_HEADER,
        "|---|" + "---|" * (COUNT_HEADER.count("|") - 2),
        count_row(run_dir.name, run_counts(run_dir)),
        "",
        f"- 에피소드당 {seconds:.1f}초 → 개발용 96 에피소드 약 {96 * seconds / 3600:.2f}시간",
        f"- 결과: {'모두 통과' if not failed else '통과하지 못한 관문 ' + ', '.join(failed)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- the paired report


# 두 실행을 짝지어도 되는지 확인한다: 둘 다 개발용이어야 하고, 시뮬레이터 모델은 달라야 하며, 시뮬레이터
# 설정(모델, num_ctx, 제공자 정보) 말고 다른 차이(에이전트 모델·옵션, Ollama 버전, 과제·seed·프롬프트 해시
# 등)가 있으면 거절한다. 판정된 시도 수 차이는 과제 단위 짝짓기에서 처리하므로 막지 않는다.
def _pairing(base_dir: Path, cand_dir: Path) -> None:
    for run_dir in (base_dir, cand_dir):
        if is_test_run(run_dir):
            raise ValueError(f"{run_dir.name}: the sensitivity check reads development records only")
    base_user = load_manifest(base_dir)["user_provider"].get("model")
    if base_user == load_manifest(cand_dir)["user_provider"].get("model"):
        raise ValueError(f"the same simulator in both runs ({base_user}): not a sensitivity check")
    blocking = [
        line
        for line in setup_differences(base_dir, cand_dir, ignore=("user_model", "user_num_ctx"))
        if not line.startswith(("user_provider differs", "judged trials of"))
    ]
    if blocking:
        raise ValueError(
            f"{cand_dir.name} differs from {base_dir.name} in more than the simulator: " + "; ".join(blocking)
        )


# 두 실행 모두에서 시도가 k번 이상 판정된 과제만 골라 pass^k 차이의 짝지은 부트스트랩 점 추정과 95% 구간, 쓴
# 과제 수를 돌려준다. 쓸 과제가 없으면 NaN.
def _paired(
    base: dict[str, list[bool]], other: dict[str, list[bool]], k: int
) -> tuple[float, float, float, int]:
    usable = [t for t in base if len(base[t]) >= k and len(other.get(t, ())) >= k]
    if not usable:
        return float("nan"), float("nan"), float("nan"), 0
    diff, low, high = paired_difference({t: base[t] for t in usable}, {t: other[t] for t in usable}, k)
    return diff, low, high, len(usable)


# analyze sim-check: 12단계 보고서. pass^1·pass^4와 U0 대비 차이, 유형별 pass^1, 종료 사유, 두 실행의 계수표,
# 그리고 다음 라운드 후보 규칙의 결과를 만든다. 판정은 하지 않는다(보고만).
def check(base_dir: Path, cand_dir: Path) -> tuple[str, dict[str, Any]]:
    """The stage 12 report: U1 (another simulator) against U0 (the 7B record), same agent, same tasks."""
    _pairing(base_dir, cand_dir)
    names = (base_dir.name, cand_dir.name)
    episodes = {name: load_episodes(d) for name, d in zip(names, (base_dir, cand_dir), strict=True)}
    by_task = {name: successes_by_task(eps) for name, eps in episodes.items()}
    counts = {name: run_counts(d) for name, d in zip(names, (base_dir, cand_dir), strict=True)}
    base, cand = names
    diff, low, high, n1 = _paired(by_task[base], by_task[cand], 1)
    diff4, low4, high4, n4 = _paired(by_task[base], by_task[cand], 4)

    # 짝지은 과제 수가 두 실행의 전체 과제 수와 다르면 표에 "(N과제)"를 붙인다.
    def note(n: int) -> str:
        return "" if n == len(by_task[base]) == len(by_task[cand]) else f" ({n}과제)"

    lines = [
        "| 실행 | 시뮬레이터 | pass^1 | U0과의 차이 [95% 구간] | pass^4 | 차이 [95% 구간] |",
        "|---|---|---|---|---|---|",
        f"| {base} (U0) | {counts[base]['simulator']} | {pass_k(by_task[base], 1):.1%} | "
        f"| {pass_k(by_task[base], 4):.1%} | |",
        f"| {cand} | {counts[cand]['simulator']} | {pass_k(by_task[cand], 1):.1%} "
        f"| {diff:+.1%}p [{low:+.1%}, {high:+.1%}]{note(n1)} | {pass_k(by_task[cand], 4):.1%} "
        f"| {diff4:+.1%}p [{low4:+.1%}, {high4:+.1%}]{note(n4)} |",
        "",
        "| 실행 | 조회 | 처리 | 거절 | 복합 |",
        "|---|---|---|---|---|",
    ]
    types = load_task_types()
    for name in names:
        cells = []
        for kind in ("lookup", "action", "refusal", "composite"):
            part = {t: v for t, v in by_task[name].items() if types.get(t) == kind}
            cells.append(f"{pass_k(part, 1):.1%}" if part else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "| 실행 | 종료 사유 | 건수 |", "|---|---|---|"]
    for name in names:
        for reason, n in sorted(Counter(e["termination"] for e in episodes[name]).items()):
            lines.append(f"| {name} | {reason} | {n} |")
    lines += ["", COUNT_HEADER, "|---|" + "---|" * (COUNT_HEADER.count("|") - 2)]
    lines += [count_row(name, counts[name]) for name in names]

    def stop_share(c: dict[str, Any]) -> float | None:
        """STOP among the episodes that never heard every value: the simulator's choice when they do not
        come. None when every episode heard them (nothing to read)."""
        return c["stop_before_values"] / c["values_never_heard"] if c["values_never_heard"] else None

    def invented_rate(c: dict[str, Any]) -> float:
        return _rate(c["invented_utterances"], c["episodes"])

    # 다음 라운드 후보 규칙(재기 전에 정하고 2026-10-09에 바꿈): (1) 큰 차이 = pass^1 차이의 95% 구간이 0을
    # 벗어남 (2) U1의 "값을 끝내 못 들은 에피소드 중 STOP" 비율이 U0보다 낮고, 에피소드당 지어낸 값이 든
    # 발화가 U0 이하. 어느 한 실행이라도 그 비율을 계산할 수 없으면 (2)는 만족하지 않은 것으로 본다. 둘 다
    # 만족할 때만 시뮬레이터를 바꾸는 새 측정 시대를 후보로 적는다.
    large = low > 0 or high < 0
    shares = {name: stop_share(counts[name]) for name in names}
    known = shares[base] is not None and shares[cand] is not None
    counts_lower = (
        known and shares[cand] < shares[base] and invented_rate(counts[cand]) <= invented_rate(counts[base])
    )
    next_round = large and counts_lower
    over = {name: counts[name]["prompts_over_limit"] for name in names}
    lines += [
        "",
        f"- 큰 차이 (pass^1 차이의 95% 구간이 0을 벗어남): {'예' if large else '아니오'}",
        "- 자동 계수가 U1에서 낮음 (값을 끝내 못 들은 에피소드 중 STOP으로 끝낸 비율이 U0보다 낮고, "
        "에피소드당 지어낸 값이 든 발화가 U0 이하. 2026-10-09 규칙 변경): "
        + ("예" if counts_lower else "아니오")
        + ("" if known else " (값을 끝내 못 들은 에피소드가 없는 실행이 있어 비율을 알 수 없음)"),
        "- 한도 95%를 넘은 시뮬레이터 호출: "
        + ", ".join(f"{name} {n}" for name, n in over.items())
        + (" (프롬프트 앞부분이 잘렸을 수 있다)" if any(over.values()) else ""),
        (
            f"- 다음 라운드 후보: 시뮬레이터를 {counts[cand]['simulator']}로 바꾸는 새 측정 시대를 "
            "따로 등록하는 안 (그 단계의 규칙·관문, 기준 다시 재기)"
            if next_round
            else "- 다음 라운드 후보 없음: 1차 계측기(시뮬레이터)는 그대로 둔다"
        ),
    ]
    outcome = {
        "difference": diff,
        "low": low,
        "high": high,
        "large": large,
        "counts_lower": counts_lower,
        "next_round": next_round,
        "prompts_over_limit": over,
        "stop_share": shares,
    }
    return "\n".join(lines), outcome
