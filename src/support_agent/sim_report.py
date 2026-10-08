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

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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

MIN_DIGITS = 4  # shorter numbers (quantities, days, sizes, times) are too often legitimate to count
PROMPT_LIMIT_SHARE = 0.95  # a simulator prompt above this share of its num_ctx may have lost its start
_EMAIL = re.compile(r"[0-9a-z._%+\-]+@[0-9a-z\-]+(?:\.[0-9a-z\-]+)*\.[a-z]{2,}")
# Digit groups joined by hyphens (phone, order and tracking numbers) or by thousands commas (amounts).
_NUMBER = re.compile(r"\d+(?:-\d+|,\d{3}(?!\d))*")
_HYPHENS = re.compile("[\u2010-\u2015\u2212\ufe63\uff0d]")
_THINK_TAG = re.compile(r"</?think>")
# What is left of an end token after the valid ones are taken off: "네 감사합니다 ###", "STOP".
_END_LEFTOVER = re.compile(r"#{2,}|(?<![A-Za-z])STOP(?![A-Za-z])|OUT[-_ ]?OF[-_ ]?SCOPE", re.IGNORECASE)
FORMAT_KINDS = ("think_tag", "cut_off", "tool_call", "broken_end")


# ---------------------------------------------------------------- values


def _normal(text: str) -> str:
    return _HYPHENS.sub("-", unicodedata.normalize("NFKC", text)).lower()


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


def _numbers(text: str, min_digits: int) -> set[str]:
    out = set()
    for raw in _NUMBER.findall(text):
        digits = raw.replace("-", "").replace(",", "")
        if len(digits) >= min_digits:
            out.add(f"num:{digits}")
    return out


def value_tokens(text: str) -> set[str]:
    """The values in one simulator utterance: email:..., date:MM-DD, num:<digits> (MIN_DIGITS or more)."""
    text = _normal(text)
    emails = {f"email:{e}" for e in _EMAIL.findall(text)}
    dates, rest = _dates_and_rest(_EMAIL.sub(" ", text))
    return emails | dates | _numbers(rest, MIN_DIGITS)


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


def grounded(token: str, sources: set[str]) -> bool:
    if token in sources:
        return True
    if token.startswith("num:"):
        digits = token[4:]
        return any(s.startswith("num:") and digits in s[4:] for s in sources)
    return False


# ---------------------------------------------------------------- one episode


def _simulator_turns(episode: dict[str, Any]) -> list[tuple[dict[str, Any], list[str], list[str]]]:
    """Each simulator call with the agent texts it had seen and the tool results that came before it."""
    calls = sorted((c for c in episode["llm_calls"] if c["who"] == "user"), key=lambda c: c["index"])
    seen = [m["content"] for m in episode["user_messages"] if m["role"] == "user"]
    messages = episode["messages"]
    # Where each simulator reply entered the agent's history; a bare end token (the last reply) never did.
    arrived = [i for i, m in enumerate(messages) if m["role"] == "user" and not m.get("harness")]
    results = [(i, m["content"]) for i, m in enumerate(messages) if m["role"] == "tool"]
    turns = []
    for k, call in enumerate(calls):
        before = arrived[k] if k < len(arrived) else len(messages)
        turns.append((call, seen[: k + 1], [content for i, content in results if i < before]))
    return turns


def _said(call: dict[str, Any]) -> str:
    return split_ending(call.get("text") or "")[0]


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


def stop_before_values(episode: dict[str, Any], task: Task) -> bool | None:
    """None when the task has no required value (or the episode was an infra error)."""
    if episode["status"] == "infra_error" or not task.required_values:
        return None
    if episode["termination"] != "user_stop":
        return False
    seen = [m["content"] for m in episode["user_messages"] if m["role"] == "user"]
    return not all(value_found(value, seen) for value in task.required_values)


def values_never_heard(episode: dict[str, Any], task: Task) -> bool | None:
    """True when the simulator never read every required value, however the episode ended."""
    if episode["status"] == "infra_error" or not task.required_values:
        return None
    seen = [m["content"] for m in episode["user_messages"] if m["role"] == "user"]
    return not all(value_found(value, seen) for value in task.required_values)


def stop_with_writes_left(episode: dict[str, Any], task: Task) -> bool | None:
    """None when the task has no gold write (or the episode was an infra error)."""
    if episode["status"] == "infra_error" or not task.gold_actions:
        return None
    return episode["termination"] == "user_stop" and episode["verdict"]["missing_writes"] > 0


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


def _all_tasks() -> dict[str, Task]:
    return {task.id: task for path in sorted(TASKS.glob("*.yaml")) for task in load_tasks(path)}


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


def _share(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d:.1%})" if d else f"{n}/0"


def _rate(n: int, d: int) -> float:
    return n / d if d else 0.0


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


def _paired(
    base: dict[str, list[bool]], other: dict[str, list[bool]], k: int
) -> tuple[float, float, float, int]:
    usable = [t for t in base if len(base[t]) >= k and len(other.get(t, ())) >= k]
    if not usable:
        return float("nan"), float("nan"), float("nan"), 0
    diff, low, high = paired_difference({t: base[t] for t in usable}, {t: other[t] for t in usable}, k)
    return diff, low, high, len(usable)


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
