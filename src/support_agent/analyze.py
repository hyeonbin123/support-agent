"""Recompute numbers from the episode records of a run. Nothing here calls a model.

Usage:
    uv run python -m support_agent.analyze table outputs/runs/<run_id> [more run dirs ...]
    uv run python -m support_agent.analyze compare reports/<baseline> reports/<candidate> [...]
    uv run python -m support_agent.analyze misses outputs/runs/<run_id>
    uv run python -m support_agent.analyze sample outputs/runs/<run_id> --n 20
    uv run python -m support_agent.analyze voice reports/<text run> reports/<voice run> [...]
    uv run python -m support_agent.analyze voice-worst reports/<voice run on the development tasks>
    uv run python -m support_agent.analyze claims reports/<run> [more run dirs ...]
    uv run python -m support_agent.analyze same-setup reports/<run> reports/<run to pair it with>
        [--ignore voice] [--ignore-prompt tools agent_system]   (stage 9: V2 and V4 differ in those two)
    uv run python -m support_agent.analyze smoke outputs/runs/<smoke run of a stage 8 candidate>
    uv run python -m support_agent.analyze select reports/<P1 7B dev> reports/<P0 7B dev> reports/<cand> ..
    uv run python -m support_agent.analyze verdict reports/<P1 7B test> reports/<P0 7B test> reports/<cand>
    uv run python -m support_agent.analyze blind reports/<dev run A> reports/<dev run B> .. --n 20 --out <dir>
    uv run python -m support_agent.analyze unblind <dir>
    uv run python -m support_agent.analyze name-collisions [--k K]   (default: the registered tolerance)
    uv run python -m support_agent.analyze caller-id reports/<V2 run> reports/<V4 run> [...]
    uv run python -m support_agent.analyze confirm reports/<R2 run> [...]   (stage 10: previews and answers)
    uv run python -m support_agent.analyze confirm-smoke outputs/runs/<R2 smoke run>
    uv run python -m support_agent.analyze confirm-check reports/<M2b run> reports/<R2 run> [--test]
    uv run python -m support_agent.analyze confirm-sheet reports/<R2 dev run> --n 20 --out <dir>
    uv run python -m support_agent.analyze confirm-marks <dir>
    uv run python -m support_agent.analyze sim-counts reports/<run> [...]   (stage 12: simulator slips, P5(b))
    uv run python -m support_agent.analyze sim-smoke outputs/runs/<smoke run with another simulator>
    uv run python -m support_agent.analyze sim-check reports/<run, 7B simulator> reports/<other simulator>

`table` prints the markdown tables that go into docs/experiments.md. `misses` lists episodes whose database
matched but whose required value was not found, so that a person can check the value matcher. `sample` draws
episodes with a fixed seed for the simulator check ("시뮬레이터 점검").
"""

# [파일 역할] 실행 기록(reports/·outputs/runs/의 episodes.jsonl, manifest.json, summary.json)에서 모든 수치를
# 다시 계산하는 분석 CLI다. 모델을 부르지 않는다.
# [왜 필요한가] 이 저장소의 수치는 모두 기록에서 다시 계산될 수 있어야 한다(README 다시 재는 방법: analyze
# table, compare, voice, claims는 모델 없이 된다). 판정 규칙이 바뀐 뒤에도 기록된 판정 요소(DB 일치, 전달 값,
# 종료 사유)로 성공을 다시 계산한다.
# [주요 구성]
# - 공통: pass^k와 과제 단위 부트스트랩 구간, 두 실행의 짝지은 차이(compare), 결과표(table), 값만 놓친
#   에피소드(misses), 점검용 표본(sample)
# - 7단계: 고객에게 전달된 근거 없는 완료 주장과 가드 뒤의 행동(claims)
# - 짝짓기 검사: 두 실행이 한 축 말고 같은 설정인지(same-setup)
# - 9단계: 이름 허용 오차 충돌 관문(name-collisions), 본인 확인 표(caller-id)
# - 8단계: 스모크 관문(smoke), 선택(select), 시험용 판정(verdict), 결과를 가린 시뮬레이터 점검(blind/unblind)
# - 10단계(confirm_report.py)·12단계(sim_report.py) 명령은 그 모듈로 넘긴다
# [설계 이유] 문턱값과 판정표는 docs/experiments.md에 측정 전에 적은 규칙을 그대로 코드로 옮긴 것이다.
# 부트스트랩은 고정 seed라 같은 기록은 언제나 같은 구간을 낸다. 시험용 기록으로는 무엇도 만들지 않는다는
# 규칙을 is_test_run 검사로 지킨다.

from __future__ import annotations

# statistics: 스모크 관문 S3의 prompt_eval 중앙값. random: 고정 seed 부트스트랩과 표본 추출.
import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# 판정하는 종료 목록(JUDGED), 설정 기본값(RunConfig, 옛 manifest에 없는 필드를 채움), pass^k 공식(judge), 음성
# 지표(voice.metrics)를 쓴다. 나머지 내부 모듈은 쓰는 명령 안에서 가져온다.
from support_agent.config import JUDGED, RunConfig
from support_agent.judge import pass_hat_k
from support_agent.voice import metrics as voice_metrics

# 부트스트랩 10,000회, 고정 seed(2단계 선택 규칙: 과제 단위 짝지은 부트스트랩 10,000회, 고정 seed). 점검용
# 표본 추출에도 같은 seed를 쓴다.
BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 20260919
# table의 보조 지표 열 이름(docs/experiments.md 결과표와 같은 순서).
_COUNT_COLUMNS = (
    "정답에 없는 쓰기",
    "성공인데 정답에 없는 쓰기",
    "통과된 규정 위반",
    "막힌 규정 위반",
    "본인 확인 전 차단",
    "형식 오류",
    "버린 호출",
    "에피소드당 초",
)
# 형식 오류로 세지 않는 format_error 값. 멈춤(stall)과 근거 없는 완료 주장(unbacked_claim)은 형식이 맞는 답을
# 가드가 돌려보낸 것이라 따로 센다. run.py의 summary와 같은 기준이다.
# Replies held back by a guard are well formed; they are counted on their own.
_NOT_A_FORMAT_ERROR = (None, "stall", "unbacked_claim")


# 실행 폴더의 episodes.jsonl을 에피소드 dict 목록으로 읽는다(빈 줄은 건너뜀).
def load_episodes(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "episodes.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def task_type(task_id: str, types: dict[str, str]) -> str:
    return types.get(task_id, "?")


# 지금 규칙의 성공을 기록된 판정 요소로 다시 계산한다: infra_error가 아니고, 판정하는 종료이고, DB가 정답과
# 같고, 전달 값이 모두 있음. 고객 턴 한도(max_user_turns)는 2026-09-19에 판정 대상으로 옮겼으므로, 그 전에 쓴
# 기록의 verdict.success를 그대로 쓰지 않고 이렇게 다시 계산한다.
def is_success(episode: dict[str, Any]) -> bool:
    """Success by the current rules, recomputed from the recorded parts of the verdict (older records were
    written when fewer endings were judged)."""
    verdict = episode["verdict"]
    return bool(
        verdict
        and episode["termination"] in JUDGED
        and verdict["db_match"]
        and all(verdict["values"].values())
    )


# 과제 id → 시도별 성공 목록. infra_error는 분모에서 빠진다.
def successes_by_task(episodes: list[dict[str, Any]]) -> dict[str, list[bool]]:
    by_task: dict[str, list[bool]] = {}
    for e in episodes:
        if e["status"] != "infra_error":
            by_task.setdefault(e["task_id"], []).append(is_success(e))
    return by_task


# 과제 평균 pass^k. 판정된 시도가 k번보다 적은 과제는 그 k에서 빼고 평균한다(infra_error로 시도를 잃은 과제).
def pass_k(by_task: dict[str, list[bool]], k: int) -> float:
    values = [pass_hat_k(len(v), sum(v), k) for v in by_task.values() if len(v) >= k]
    return sum(values) / len(values) if values else float("nan")


# pass^k의 95% 구간. 에피소드가 아니라 과제를 복원 추출한다. 한 과제의 시도들은 서로 독립이 아니기
# 때문이다(같은 시나리오·같은 DB). 과제별 점수를 10,000번 다시 뽑아 평균을 정렬하고 2.5%·97.5% 위치를 구간
# 끝으로 쓴다.
def bootstrap_interval(by_task: dict[str, list[bool]], k: int) -> tuple[float, float]:
    """95% interval of pass^k, resampling tasks (not episodes): trials of one task are not independent."""
    tasks = [v for v in by_task.values() if len(v) >= k]
    if not tasks:
        return float("nan"), float("nan")
    rng = random.Random(BOOTSTRAP_SEED)
    scores = [pass_hat_k(len(v), sum(v), k) for v in tasks]
    means = sorted(sum(rng.choice(scores) for _ in scores) / len(scores) for _ in range(BOOTSTRAP_ROUNDS))
    return means[int(0.025 * BOOTSTRAP_ROUNDS)], means[int(0.975 * BOOTSTRAP_ROUNDS) - 1]


# 두 구성의 pass^k 차이(other − base)와 95% 구간. 같은 과제를 두 구성에서 돌렸으므로 과제마다 차이를 먼저
# 구하고, 과제 단위로 다시 뽑아 짝을 유지한다. 과제 사이의 난이도 차이는 짝 안에서 빠진다. 두 실행의 과제
# 목록이 다르면 오류다.
def paired_difference(
    base: dict[str, list[bool]], other: dict[str, list[bool]], k: int
) -> tuple[float, float, float]:
    """pass^k of `other` minus `base` with a 95% interval. Tasks are resampled and each task keeps its pair,
    because the same tasks were run under both settings."""
    if set(base) != set(other):
        raise ValueError("both runs must cover the same tasks")
    diffs = [
        pass_hat_k(len(other[t]), sum(other[t]), k) - pass_hat_k(len(base[t]), sum(base[t]), k)
        for t in sorted(base)
    ]
    rng = random.Random(BOOTSTRAP_SEED)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(BOOTSTRAP_ROUNDS))
    low, high = means[int(0.025 * BOOTSTRAP_ROUNDS)], means[int(0.975 * BOOTSTRAP_ROUNDS) - 1]
    return sum(diffs) / len(diffs), low, high


# 기준 실행과 후보 실행들의 pass^1·pass^4와 짝지은 차이를 마크다운 표로 낸다(README·docs의 '기준과의 차이 [95%
# 구간]' 열).
def compare(base_dir: Path, other_dirs: list[Path]) -> str:
    base = successes_by_task(load_episodes(base_dir))
    lines = [
        "| 실행 | pass^1 | 기준과의 차이 [95% 구간] | pass^4 | 차이 [95% 구간] |",
        "|---|---|---|---|---|",
    ]
    lines.append(f"| {base_dir.name} (기준) | {pass_k(base, 1):.1%} | | {pass_k(base, 4):.1%} | |")
    for run_dir in other_dirs:
        other = successes_by_task(load_episodes(run_dir))
        cells = []
        for k in (1, 4):
            # A trial lost to an infra error leaves a task with fewer than k judged trials: pass^k is then
            # paired over the tasks that have k in both runs, and the cell says over how many.
            usable = [t for t in base if len(base[t]) >= k and len(other.get(t, ())) >= k]
            mine, theirs = {t: base[t] for t in usable}, {t: other[t] for t in usable}
            diff, low, high = paired_difference(mine, theirs, k)
            note = "" if len(usable) == len(base) == len(other) else f" ({len(usable)}과제)"
            cells += [f"{pass_k(theirs, k):.1%}{note}", f"{diff:+.1%}p [{low:+.1%}, {high:+.1%}]"]
        lines.append(f"| {run_dir.name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# 표 칸 하나: 'pass^k [구간 아래, 위]'.
def cell(by_task: dict[str, list[bool]], k: int) -> str:
    low, high = bootstrap_interval(by_task, k)
    return f"{pass_k(by_task, k):.1%} [{low:.1%}, {high:.1%}]"


# 모든 과제 파일의 과제 id → 유형(lookup, action, refusal, composite). 유형별 성공률 표에 쓴다.
def load_task_types() -> dict[str, str]:
    from support_agent.paths import TASKS
    from support_agent.tasks import load_tasks

    return {task.id: task.type for path in sorted(TASKS.glob("*.yaml")) for task in load_tasks(path)}


# docs/experiments.md에 붙이는 결과표 묶음을 만든다: (1) 실행별 과제·에피소드·infra·중단 수와
# pass^1..pass^k(구간 포함), (2) 유형별 pass^1, (3) 종료 사유별 건수, (4) 보조 지표(정답에 없는 쓰기, 성공인데
# 정답에 없는 쓰기, 통과·막힌 규정 위반, 본인 확인 전 차단, 형식 오류, 버린 호출, 에피소드당 초), (5) 통과된
# 규정 위반 코드별 횟수.
def table(run_dirs: list[Path]) -> str:
    types = load_task_types()
    runs = {d.name: load_episodes(d) for d in run_dirs}
    lines: list[str] = []
    # 표의 k 범위는 모든 실행에서 가장 적은 '과제당 판정된 시도 수'까지다.
    max_k = min(
        (min((len(v) for v in successes_by_task(eps).values()), default=0) for eps in runs.values()),
        default=0,
    )

    lines += [
        "| 실행 | 과제 | 에피소드 | infra | 중단 | "
        + " | ".join(f"pass^{k}" for k in range(1, max_k + 1))
        + " |"
    ]
    lines += ["|---|---|---|---|---|" + "---|" * max_k]
    for name, eps in runs.items():
        by_task = successes_by_task(eps)
        infra = sum(e["status"] == "infra_error" for e in eps)
        truncated = sum(e["status"] != "infra_error" and e["termination"] not in JUDGED for e in eps)
        cells = " | ".join(cell(by_task, k) for k in range(1, max_k + 1))
        lines.append(f"| {name} | {len(by_task)} | {len(eps)} | {infra} | {truncated} | {cells} |")

    lines += ["", "| 실행 | 유형 | 과제 | pass^1 |", "|---|---|---|---|"]
    for name, eps in runs.items():
        by_task = successes_by_task(eps)
        for kind in ("lookup", "action", "refusal", "composite"):
            part = {t: v for t, v in by_task.items() if task_type(t, types) == kind}
            if part:
                lines.append(f"| {name} | {kind} | {len(part)} | {cell(part, 1)} |")

    lines += ["", "| 실행 | 종료 사유 | 건수 |", "|---|---|---|"]
    for name, eps in runs.items():
        for reason, count in sorted(Counter(e["termination"] for e in eps).items()):
            lines.append(f"| {name} | {reason} | {count} |")

    lines += [
        "",
        "| 실행 | " + " | ".join(_COUNT_COLUMNS) + " |",
        "|---|" + "---|" * len(_COUNT_COLUMNS),
    ]
    for name, eps in runs.items():
        judged = [e for e in eps if e["verdict"]]
        agent_calls = [c for e in judged for c in e["llm_calls"] if c["who"] == "agent"]
        seconds = [e["wall_seconds"] for e in judged]
        lines.append(
            f"| {name} | {sum(e['verdict']['unexpected_writes'] for e in judged)} "
            f"| {sum(bool(is_success(e) and e['verdict']['unexpected_writes']) for e in judged)} "
            f"| {sum(len(e['verdict']['policy_violations']) for e in judged)} "
            f"| {sum(len(e['verdict']['policy_blocks']) for e in judged)} "
            f"| {sum(e['verdict']['auth_blocks'] for e in judged)} "
            f"| {sum(c['format_error'] not in _NOT_A_FORMAT_ERROR for c in agent_calls)} "
            f"| {sum(c['dropped_calls'] for c in agent_calls)} "
            f"| {sum(seconds) / len(seconds) if seconds else 0:.1f} |"
        )

    codes = Counter(
        (name, code) for name, eps in runs.items() for e in eps if e["verdict"]
        for code in e["verdict"]["policy_violations"]
    )  # fmt: skip
    if codes:
        lines += ["", "| 실행 | 통과된 규정 위반 코드 | 횟수 |", "|---|---|---|"]
        lines += [f"| {name} | {code} | {count} |" for (name, code), count in sorted(codes.items())]
    return "\n".join(lines)


# DB는 정답과 같은데 전달 값만 못 찾은 에피소드와 그 상담원 말을 늘어놓는다. 값 찾기 규칙(judge.value_found)이
# 실제로 말한 값을 놓친 것은 아닌지 사람이 확인하려는 용도다(docstring).
def misses(run_dir: Path) -> str:
    out: list[str] = []
    for e in load_episodes(run_dir):
        verdict = e["verdict"]
        if verdict and verdict["judged"] and verdict["db_match"] and not all(verdict["values"].values()):
            missing = [label for label, found in verdict["values"].items() if not found]
            out.append(f"## {e['task_id']} #{e['trial']}: missing {missing}")
            out += [f"- {m['content']}" for m in e["messages"] if m["role"] == "assistant" and m["content"]]
    return "\n".join(out) or "no episode failed on a required value alone"


# 시뮬레이터 점검용 표본: infra_error가 아닌 에피소드 n개를 고정 seed로 뽑아 고객·도구 호출·상담원 말을 읽기
# 쉬운 형태로 출력한다. 하니스 안내문과 전달하지 않은 답은 뺀다(1단계 시뮬레이터 점검).
def sample(run_dir: Path, n: int) -> str:
    episodes = [e for e in load_episodes(run_dir) if e["status"] != "infra_error"]
    chosen = random.Random(BOOTSTRAP_SEED).sample(episodes, min(n, len(episodes)))
    out: list[str] = []
    for e in sorted(chosen, key=lambda x: (x["task_id"], x["trial"])):
        verdict = "PASS" if is_success(e) else "fail"
        out.append(f"## {e['task_id']} #{e['trial']} ({verdict}, {e['termination']})")
        for m in e["messages"]:
            if m["role"] == "user" and not m.get("harness"):
                out.append(f"- 고객: {m['content']}")
            elif m["role"] == "assistant" and m.get("tool_calls"):
                call = m["tool_calls"][0]
                out.append(f"- (도구) {call['name']} {json.dumps(call['arguments'], ensure_ascii=False)}")
            elif m["role"] == "assistant" and m["content"] and m.get("delivered", True):
                out.append(f"- 상담원: {m['content']}")
        out.append("")
    return "\n".join(out)


# 7단계 지표: 고객에게 실제로 전달된 답 가운데 근거 없는 완료 주장 문장. 각 답을 그 앞까지의 대화만으로
# 검사한다. 검출기가 코드라 가드 없이 돈 실행의 기록에도 그대로 댈 수 있다(가드를 켜기 전 기준 기록에서 거짓
# 완료가 얼마나 전달됐는지).
def delivered_claims(episode: dict[str, Any]) -> list[str]:
    """Sentences that reached the customer saying work was done which no earlier tool result showed.
    The detector is code, so it reads the records of any run, also of one that ran without the guard."""
    from support_agent.chat import Message
    from support_agent.claims import unbacked_claim

    messages = [Message.from_dict(m) for m in episode["messages"]]
    found = []
    for i, m in enumerate(messages):
        if m.role == "assistant" and m.content and m.delivered and not m.tool_calls:
            claim = unbacked_claim(m.content, messages[:i])
            if claim:
                found.append(claim.sentence)
    return found


# 가드가 돌려보낸 답마다 그다음 에이전트 호출이 무엇이었는지 분류한다: 쓰기 성공, 쓰기 실패, 조회, 같은 주장,
# 다른 말, 형식 오류, 끝. 안내문을 받은 모델이 실제로 도구를 불렀는지 보려는 지표다(7단계 결과: 7B는 거의
# 부르지 않았다).
def after_a_held_claim(episode: dict[str, Any]) -> list[str]:
    """What the agent's next LLM call was after each reply the claim guard held back:
    write | write_failed | read | claim_again | reply | format_error | nothing (the episode ended)."""
    calls = [c for c in episode["llm_calls"] if c["who"] == "agent"]
    tools = {t["agent_call"]: t for t in episode["tool_calls"]}
    out = []
    for position, call in enumerate(calls):
        if call["format_error"] != "unbacked_claim":
            continue
        following = calls[position + 1] if position + 1 < len(calls) else None
        if following is None:
            out.append("nothing")
        elif following["format_error"] == "unbacked_claim":
            out.append("claim_again")
        # 다음 호출이 도구 호출을 냈으면 그 도구 기록으로 쓰기·조회와 성공 여부를 가른다(도구 기록은 그 호출의
        # 순번 agent_call로 찾는다).
        elif following["index"] in tools:
            tool = tools[following["index"]]
            out.append(("write" if tool["ok"] else "write_failed") if tool["write"] else "read")
        elif following["format_error"] in (None, "stall"):
            out.append("reply")
        else:
            out.append("format_error")
    return out


# 7단계 표(analyze claims): 실행별로 거짓 완료가 전달된 에피소드와 그중 성공, 전달된 거짓 완료 답 수, 가드가
# 돌려보낸 답과 그 뒤의 행동, 가드로 끝난 에피소드.
def claims_table(run_dirs: list[Path]) -> str:
    header = (
        "| 실행 | 에피소드 | 거짓 완료가 전달된 에피소드 | 그중 성공 | 전달된 거짓 완료 응답 "
        "| 가드가 돌려보낸 응답 | 그 뒤: 쓰기 성공 | 쓰기 실패 | 조회 | 같은 주장 | 다른 말 "
        "| 형식 오류 | 끝 | 가드로 끝난 에피소드 |"
    )
    lines = [header, "|---|" + "---|" * (header.count("|") - 2)]
    for run_dir in run_dirs:
        episodes = [e for e in load_episodes(run_dir) if e["status"] != "infra_error"]
        told = [(e, delivered_claims(e)) for e in episodes]
        after = Counter(step for e in episodes for step in after_a_held_claim(e))
        held = sum(c["format_error"] == "unbacked_claim" for e in episodes for c in e["llm_calls"])
        lines.append(
            f"| {run_dir.name} | {len(episodes)} | {sum(bool(s) for _, s in told)} "
            f"| {sum(bool(s) and is_success(e) for e, s in told)} | {sum(len(s) for _, s in told)} | {held} "
            f"| {after['write']} | {after['write_failed']} | {after['read']} | {after['claim_again']} "
            f"| {after['reply']} | {after['format_error']} | {after['nothing']} "
            f"| {sum(e['termination'] == 'unbacked_claim' for e in episodes)} |"
        )
    return "\n".join(lines)


# 실행 조건 기록(manifest.json)을 읽는다.
def load_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


# 시험용 과제의 실행인지. 과제 파일 이름이나 과제 id(test-…)로 판단한다. 시험용 기록으로는 아무것도 만들지
# 않는다는 규칙을 지키려고, 점검표·실패 분석 명령이 이 검사로 시험용을 거부한다.
def is_test_run(run_dir: Path) -> bool:
    """True when the run is on the test tasks: those records are not for building anything from."""
    manifest = load_manifest(run_dir)
    return "test" in Path(str(manifest.get("tasks_file", ""))).stem or any(
        task_id.startswith("test-") for task_id in manifest.get("task_sha256", {})
    )


# 두 실행을 '한 축만 다른 같은 설정'으로 짝지어도 되는지 확인하고, 안 되는 이유를 목록으로 돌려준다(빈 목록 =
# 짝지어도 됨). analyze same-setup과 8·10·12단계의 짝짓기 검사가 쓴다.
# 비교 항목: 무시할 축을 뺀 모든 설정(옛 manifest에 없는 필드는 기본값으로 봄), 시도 수, 과제 해시, seed 해시,
# 프롬프트 해시(V4는 에이전트 프롬프트와 도구 목록이 일부러 달라 뺄 수 있음), 두 제공자 정보(모델 digest,
# Ollama 버전, 옵션), 과제별 판정된 시도 번호.
def setup_differences(
    run_a: Path, run_b: Path, ignore: tuple[str, ...] = ("voice",), ignore_prompts: tuple[str, ...] = ()
) -> list[str]:
    """Why two runs may not be paired as "the same setup but for `ignore`". Empty when they may.

    Compared: every config field but the ignored axes (a field an older manifest lacks counts as its default),
    trials, task hashes, seed hash, prompt hashes but `ignore_prompts` (V4 changes the agent prompt and the
    tool list on purpose), both providers (model digest, server version, options), and per task the trials
    that were judged.
    """
    a, b = load_manifest(run_a), load_manifest(run_b)
    for manifest in (a, b):
        prompts = manifest.get("prompt_sha256")
        if isinstance(prompts, dict) and ignore_prompts:
            manifest["prompt_sha256"] = {k: v for k, v in prompts.items() if k not in ignore_prompts}
    # 새 필드가 생기기 전의 기록은 그 필드가 없다. 기본값과 같다고 보고 비교해, 필드가 추가됐다는 이유만으로
    # 옛 기록과 짝짓기가 막히지 않게 한다.
    defaults = RunConfig().to_dict()
    out = []
    for key in sorted((set(a["config"]) | set(b["config"]) | set(defaults)) - set(ignore)):
        left, right = a["config"].get(key, defaults.get(key)), b["config"].get(key, defaults.get(key))
        if left != right:
            out.append(f"config.{key}: {left!r} != {right!r}")
    for key in ("trials", "task_sha256", "seed_hash", "prompt_sha256", "agent_provider", "user_provider"):
        if a.get(key) != b.get(key):
            detail = ""
            if isinstance(a.get(key), dict) and isinstance(b.get(key), dict):
                changed = sorted(k for k in set(a[key]) | set(b[key]) if a[key].get(k) != b[key].get(k))
                detail = f" ({', '.join(changed[:6])})"
            out.append(f"{key} differs{detail}")

    # 과제별로 판정된(infra_error가 아닌) 시도 번호. 두 실행에서 다르면 짝지은 비교의 표본이 달라지므로 차이로
    # 보고한다.
    def judged_trials(run_dir: Path) -> dict[str, list[int]]:
        trials: dict[str, list[int]] = {}
        for e in load_episodes(run_dir):
            if e["status"] != "infra_error":
                trials.setdefault(e["task_id"], []).append(e["trial"])
        return {task: sorted(v) for task, v in trials.items()}

    left, right = judged_trials(run_a), judged_trials(run_b)
    for task in sorted(set(left) | set(right)):
        if left.get(task) != right.get(task):
            out.append(f"judged trials of {task}: {left.get(task)} != {right.get(task)}")
    return out


# ---------------------------------------------------------------- stage 9: voice V4, the caller's number


# 9단계 관문: 고객 id → 이름 표에서, A의 번호로 건 전화에서 B의 이름을 말했을 때 verify_caller가 받아들이는
# 순서쌍(충돌)을 센다. 이름이 정확히 같은 쌍(동명이인)은 따로 센다. 정확한 비교도 받아들이고 확인되는 사람은
# 번호의 주인 A라서 허용 오차 탓이 아니기 때문이다.
# 충돌이 1건이라도 있으면 측정으로 가지 않는다는 규칙이었다. 처음 규칙(자모 1개 허용)은 가상 고객 109명 안에서
# 다른 이름 6쌍을 받아들여 통과하지 못했고, 이름을 정확히 맞추는 규칙(k = 0)으로 다시 적었다(README 9단계).
def name_collisions(
    names: dict[str, str], k: int
) -> tuple[list[tuple[str, str, str, str, int]], list[tuple[str, str]]]:
    """The gate of stage 9, on customer id -> name: (collisions, same-name pairs).

    A collision is an ordered pair (A, B) where B's name is not A's name but verify_caller, called from A's
    number with B's name, would accept it (jamo distance <= k). Pairs with the very same name are set apart:
    exact matching accepts them as well, and the customer verified is A, the owner of the number.
    """
    from support_agent.tools import _normalise_name, name_distance

    collisions, same = [], []
    for a in sorted(names):
        for b in sorted(names):
            if a == b:
                continue
            if _normalise_name(names[a]) == _normalise_name(names[b]):
                same.append((a, b))
                continue
            distance = name_distance(names[b], names[a])
            if distance <= k:
                collisions.append((a, b, names[a], names[b], distance))
    return collisions, same


# seed DB의 모든 고객 id와 이름.
def seed_names() -> dict[str, str]:
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from support_agent import db
    from support_agent.seed import build_seed_engine

    with Session(build_seed_engine()) as session:
        return {c.id: c.name for c in session.scalars(select(db.Customer))}


# 관문 결과 보고: 고객 수, 서로 다른 이름 수, 동명이인 쌍, 충돌 목록, 통과 여부. main은 통과하지 못하면 종료
# 코드 1을 낸다.
def collision_report(k: int) -> tuple[str, bool]:
    names = seed_names()
    collisions, same = name_collisions(names, k)
    lines = [
        f"customers {len(names)}, distinct names {len(set(names.values()))}, k = {k}",
        f"same-name ordered pairs (set apart): {len(same)} " + " ".join(f"{a}/{b}" for a, b in same),
        f"collisions (another name accepted from a number): {len(collisions)}",
        *(f"  {a} {na} <- {b} {nb}: {d}" for a, b, na, nb, d in collisions),
        "gate: " + ("PASS (0 collisions)" if not collisions else "FAIL"),
    ]
    return "\n".join(lines), not collisions


# 하니스가 쓸 수 있는 발신 번호를 주지 못했다는 오류들. 에이전트 탓이 아니라 하니스 버그이므로 표에서 따로
# 센다(seed의 전화번호는 모두 달라 나오면 안 된다).
# verify_caller refusals that mean the harness gave no usable number: a bug, never the agent's doing.
CALLER_HARNESS_ERRORS = ("no_caller_number", "caller_not_registered", "caller_shared")


# 본인 확인 표(analyze caller-id): 실행별 본인 확인 성공 에피소드, 확인 도구를 아예 부르지 않은 에피소드,
# find_customer·verify_caller의 성공/호출, 허용 오차 덕에 통과한 호출, 없는 도구 호출(V4에서 find_customer를
# 부른 것), 발신 번호 오류. V2와 V4를 나란히 비교하는 데 쓴다.
def caller_id_table(runs: dict[str, list[dict[str, Any]]]) -> str:
    """How the customer got identified, per run: find_customer (V0..V2) or verify_caller (V4)."""
    from support_agent.tools import name_distance

    lines = [
        "| 실행 | 에피소드 | 본인 확인 성공 | 확인 도구를 부르지 않음 | find_customer 성공 / 호출 "
        "| verify_caller 성공 / 호출 | 허용 오차로 통과 | 없는 도구 호출 | 발신 번호 오류 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, episodes in runs.items():
        counted = [e for e in episodes if e["status"] != "infra_error"]
        calls = [c for e in counted for c in e["tool_calls"]]

        def tally(tool: str, calls: list[dict[str, Any]] = calls) -> str:
            mine = [c for c in calls if c["name"] == tool]
            return f"{sum(bool(c['ok']) for c in mine)} / {len(mine)}"

        # 성공한 verify_caller 중 에이전트가 넣은 이름과 등록된 이름의 자모 거리가 0보다 큰 것: 허용 오차로
        # 통과한 호출이다(k = 0 규칙에서는 0이어야 한다).
        tolerated = 0
        for c in calls:
            if c["name"] == "verify_caller" and c["ok"]:
                said = (c.get("args") or {}).get("name", "")
                tolerated += name_distance(said, json.loads(c["content"])["name"]) > 0
        identified = sum(voice_metrics.identified(e) for e in counted)
        silent = sum(
            not any(c["name"] in voice_metrics.IDENTITY_TOOLS for c in e["tool_calls"]) for e in counted
        )
        share = identified / len(counted) if counted else float("nan")
        codes = Counter(c.get("error_code") for c in calls)
        lines.append(
            f"| {name} | {len(counted)} | {identified} ({share:.1%}) | {silent} | {tally('find_customer')} "
            f"| {tally('verify_caller')} | {tolerated} | {codes['unknown_tool']} "
            f"| {sum(codes[code] for code in CALLER_HARNESS_ERRORS)} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- stage 8: a new-generation agent model
# 8단계 문턱값(docs/experiments.md 8단계, 측정 전에 커밋).
# - RELOAD_MS: load_duration이 1초를 넘는 호출은 모델을 다시 읽은 것으로 본다(S1).
# - S2: 에이전트 호출당 평균이 P1·7B 0.34.2 기록 589 ms의 3배(약 1,767 ms) 이하. 미리 거르는 값이다.
# - S3: prompt_eval 중앙값 500 ms 이하. 캐시를 다시 쓰면 수십 ms, 매번 처음부터 계산하면 수 초라 그 사이에 둔
#   값이다.
# - S4: 형식 오류 + think 태그가 에피소드당 1.0 이하(P1·7B 비율 48/96의 2배). S5: 한자·가나 섞인 답 10% 이하.
# - 선택: pass^1 +5%p 이상, 호출당 시간 기준의 3배 이하.
# Thresholds of the stage 8 rules in docs/experiments.md, written down and committed before measuring.

RELOAD_MS = 1_000.0  # a call whose load_duration is longer than this waited for its model to be loaded again
SMOKE_REFERENCE_AGENT_CALL_MS = 588.9  # P1·7B, reports/20260920-190957-...-p1 (Ollama 0.34.2): a pre-filter
SMOKE_MAX_CALL_RATIO = 3.0
SMOKE_MAX_PROMPT_EVAL_MEDIAN_MS = 500.0  # far below a prompt evaluated from scratch, far above a cached one
SMOKE_MAX_FORMAT_ERRORS_PER_EPISODE = 1.0  # twice the P1·7B rate (48 in 96 episodes)
SMOKE_MAX_OTHER_SCRIPT_SHARE = 0.10
SELECT_MIN_GAIN = 0.05
SELECT_MAX_CALL_RATIO = 3.0
# 사고 모델의 생각이 본문에서 떨어지지 않고 새어 나온 흔적(<think> 태그).
_THINK_TAG = re.compile(r"</?think>")


# 아래 짧은 도우미들은 여러 표·관문이 함께 쓴다: infra_error를 뺀 에피소드, 에이전트 LLM 호출, 형식 오류 수,
# think 태그 누출 수.
def _judged(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in episodes if e["status"] != "infra_error"]


def _agent_calls(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for e in _judged(episodes) for c in e["llm_calls"] if c["who"] == "agent"]


def _format_errors(calls: list[dict[str, Any]]) -> int:
    return sum(c["format_error"] not in _NOT_A_FORMAT_ERROR for c in calls)


def _think_leaks(calls: list[dict[str, Any]]) -> int:
    """Replies with a reasoning tag in their text: a thinking model whose thoughts were not split off."""
    return sum(bool(_THINK_TAG.search(c.get("text") or "")) for c in calls)


# 고객에게 실제로 전달된 상담원 본문 답만 모은다(도구 호출 메시지와 전달하지 않은 답 제외).
def delivered_replies(episode: dict[str, Any]) -> list[str]:
    """What the agent said to the customer: assistant text that was delivered and was not a tool call."""
    return [
        m["content"]
        for m in episode["messages"]
        if m["role"] == "assistant"
        and m.get("content")
        and m.get("delivered", True)
        and not m.get("tool_calls")
    ]


# (한자·가나가 2자 이상 섞인 전달 답 수, 전달 답 수). agent.in_another_language와 같은 기준이다.
def _other_script(episodes: list[dict[str, Any]]) -> tuple[int, int]:
    from support_agent.agent import in_another_language

    replies = [r for e in _judged(episodes) for r in delivered_replies(e)]
    return sum(in_another_language(r) for r in replies), len(replies)


# 모델을 다시 읽은 것으로 보이는 LLM 호출 수. 두 모델이 GPU에 함께 올라가지 못하면 턴마다 다시 읽어 시간이
# 크게 늘어난다.
def _reloads(episodes: list[dict[str, Any]]) -> int:
    return sum((c.get("load_ms") or 0) > RELOAD_MS for e in _judged(episodes) for c in e["llm_calls"])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# 정답에 없는 쓰기 합계, 통과된 규정 위반 합계, 성공한 쓰기 합계. 8단계의 쓰기 조건과 쓰기 정밀도에 쓴다.
def _unexpected(episodes: list[dict[str, Any]]) -> int:
    return sum(e["verdict"]["unexpected_writes"] for e in _judged(episodes))


def _violations(episodes: list[dict[str, Any]]) -> int:
    return sum(len(e["verdict"]["policy_violations"]) for e in _judged(episodes))


def _writes(episodes: list[dict[str, Any]]) -> int:
    return sum(1 for e in _judged(episodes) for t in e["tool_calls"] if t["ok"] and t["write"])


def load_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# 끝날 때의 /api/ps 기록에서 모델이 GPU에 올라간 비율(size_vram / size). 서버가 그 모델을 들고 있지 않았으면
# None이다.
def _gpu_share(ps: list[dict[str, Any]], model: str) -> float | None:
    """size_vram / size of a loaded model; None when the server did not hold it."""
    for m in ps:
        if m.get("name") == model:
            size = m.get("size") or 0
            return (m.get("size_vram") or 0) / size if size else 0.0
    return None


# 시뮬레이터 자리 표기: num_gpu가 없으면 GPU, 0이면 CPU, 그 밖은 일부 CPU. 8단계의 '시뮬레이터 자리 사다리'를
# 표에 적는 데 쓴다.
def _placement(user_provider: dict[str, Any]) -> str:
    num_gpu = user_provider.get("num_gpu")
    if num_gpu is None:
        return "GPU"
    return "CPU" if num_gpu == 0 else f"일부 CPU (num_gpu {num_gpu})"


# 8단계 스모크 관문 S0~S5를 (관문, 본 것, 통과) 목록으로 낸다. 스모크는 측정이 아니라 탈락 조건 확인이다.
# S0 infra 오류 0, S1 모델 자리와 다시 읽기 0, S2 호출당 시간, S3 prompt_eval 중앙값, S4 형식 오류, S5 다른
# 문자 섞인 답. 10단계(confirm_report.py)도 이 중 S0·S1·S4·S5를 다시 쓴다.
def smoke_gates(run_dir: Path) -> list[tuple[str, str, bool]]:
    """The stage 8 smoke gates S0-S5 as (gate, what was seen, passed). The smoke run is not a measurement."""
    episodes = load_episodes(run_dir)
    manifest = load_manifest(run_dir)
    ps = load_summary(run_dir).get("ollama_ps_at_end")
    judged = _judged(episodes)
    calls = _agent_calls(episodes)
    infra = len(episodes) - len(judged)
    out = [("S0", f"infra 오류 {infra}건", infra == 0)]

    user = manifest["user_provider"]
    reloads = _reloads(episodes)
    if ps is None:
        out.append(("S1", "/api/ps 기록 없음", False))
    else:
        agent_share = _gpu_share(ps, manifest["agent_provider"]["model"])
        user_share = _gpu_share(ps, user["model"])
        placement = _placement(user)
        # S1의 시뮬레이터 조건은 지정한 자리에 따라 다르다: GPU면 100%, CPU면 0%, 일부 CPU면 GPU에 조금이라도
        # 있음. 에이전트는 언제나 100% GPU여야 한다.
        if placement == "GPU":
            user_ok = user_share is not None and user_share >= 0.999
        elif placement == "CPU":
            user_ok = user_share == 0.0
        else:
            user_ok = user_share is not None and user_share > 0.0
        ok = agent_share is not None and agent_share >= 0.999 and user_ok and reloads == 0

        def pct(share: float | None) -> str:
            return "없음" if share is None else f"{share:.0%}"

        detail = (
            f"에이전트 {pct(agent_share)} GPU, 시뮬레이터 {pct(user_share)} GPU (지정: {placement}), "
            f"다시 읽기 {reloads}회"
        )
        out.append(("S1", detail, ok))

    mean_ms = _mean([c["wall_ms"] for c in calls])
    limit = SMOKE_MAX_CALL_RATIO * SMOKE_REFERENCE_AGENT_CALL_MS
    out.append(("S2", f"에이전트 호출당 {mean_ms:.0f} ms (상한 {limit:.0f})", mean_ms <= limit))
    median_eval = statistics.median([c["prompt_eval_ms"] for c in calls]) if calls else 0.0
    out.append(
        (
            "S3",
            f"prompt_eval 중앙값 {median_eval:.0f} ms (상한 {SMOKE_MAX_PROMPT_EVAL_MEDIAN_MS:.0f})",
            median_eval <= SMOKE_MAX_PROMPT_EVAL_MEDIAN_MS,
        )
    )
    # S4는 형식 오류에 think 태그 누출을 더해 에피소드당으로 본다.
    errors, leaks = _format_errors(calls), _think_leaks(calls)
    per_episode = (errors + leaks) / len(judged) if judged else 0.0
    out.append(
        (
            "S4",
            f"형식 오류 {errors} + think 태그 {leaks} = 에피소드당 {per_episode:.2f} "
            f"(상한 {SMOKE_MAX_FORMAT_ERRORS_PER_EPISODE:.1f})",
            per_episode <= SMOKE_MAX_FORMAT_ERRORS_PER_EPISODE,
        )
    )
    other, total = _other_script(episodes)
    share = other / total if total else 0.0
    out.append(
        (
            "S5",
            f"한자·가나 섞인 전달 답 {other}/{total} ({share:.0%}, 상한 {SMOKE_MAX_OTHER_SCRIPT_SHARE:.0%})",
            share <= SMOKE_MAX_OTHER_SCRIPT_SHARE,
        )
    )
    return out


# 스모크 보고서: 관문 표, 시뮬레이터 자리와 호출당 시간, 에피소드당 시간으로 낸 개발용(96 에피소드)·시험용(160
# 에피소드) 예상 시간, 시작할 때 다른 프로그램의 GPU 사용량. 단계의 GPU 시간 상한 안에 드는지 미리 보는 데
# 쓴다.
def smoke_report(run_dir: Path) -> str:
    gates = smoke_gates(run_dir)
    episodes = _judged(load_episodes(run_dir))
    manifest = load_manifest(run_dir)
    lines = ["| 관문 | 본 것 | 통과 |", "|---|---|---|"]
    lines += [f"| {name} | {detail} | {'예' if ok else '아니오'} |" for name, detail, ok in gates]
    seconds = _mean([e["wall_seconds"] for e in episodes if "wall_seconds" in e])
    user_ms = _mean([c["wall_ms"] for e in episodes for c in e["llm_calls"] if c["who"] == "user"])
    failed = [name for name, _, ok in gates if not ok]
    lines += [
        "",
        f"- 시뮬레이터 자리: {_placement(manifest['user_provider'])}, 시뮬레이터 호출당 {user_ms:.0f} ms",
        f"- 에피소드당 {seconds:.1f}초 → 개발용 96 에피소드 약 {96 * seconds / 3600:.2f}시간, "
        f"시험용 160 에피소드 약 {160 * seconds / 3600:.2f}시간",
        f"- GPU를 쓴 다른 프로그램 (시작할 때): {manifest.get('gpu_mib_used_by_others_at_start')} MiB",
        f"- 결과: {'모두 통과' if not failed else '통과하지 못한 관문 ' + ', '.join(failed)}",
    ]
    return "\n".join(lines)


# 8단계 규칙이 정한 실행인지 확인한다: 기준은 P1, 쓰기 참조는 P0, 후보는 P1이고, 모두 개발용(또는 모두
# 시험용)이며, 과제 해시와 Ollama 버전이 기준과 같아야 한다. Ollama가 0.35로 바뀌어 0.34.x 기록과 짝짓지
# 않기로 했기 때문이다(README 8단계).
def _check_runs(base_dir: Path, p0_dir: Path, cand_dirs: list[Path], *, test: bool) -> None:
    """The references are what the stage 8 rule names: the same tasks, the same Ollama, the right policy."""
    base = load_manifest(base_dir)
    expected = {base_dir: "P1", p0_dir: "P0"} | dict.fromkeys(cand_dirs, "P1")
    for run_dir, policy in expected.items():
        manifest = load_manifest(run_dir)
        if manifest["config"].get("policy") != policy:
            raise ValueError(f"{run_dir.name}: the rule needs policy {policy} here")
        if is_test_run(run_dir) != test:
            raise ValueError(f"{run_dir.name}: {'only test runs' if test else 'test runs are not'} read here")
        if manifest["task_sha256"] != base["task_sha256"]:
            raise ValueError(f"{run_dir.name}: other tasks than {base_dir.name}")
        version = manifest["agent_provider"].get("ollama_version")
        if version != base["agent_provider"].get("ollama_version"):
            raise ValueError(f"{run_dir.name}: Ollama {version} is not the version of {base_dir.name}")


# 후보를 기준과 짝지을 수 없는 이유와 시뮬레이터 자리. 에이전트 모델(think 값 포함)이 이 단계의 축이므로
# agent_provider 차이는 막지 않고, 시뮬레이터를 일부 CPU로 옮긴 것(num_gpu)도 사다리 규칙대로 허용해 표에
# 적기만 한다.
def _pairing(base_dir: Path, cand_dir: Path) -> tuple[list[str], str]:
    """Why a candidate cannot be paired with the baseline (empty when it can), and where its simulator sat.
    The agent model (and its think value) is the axis; a simulator moved to the CPU is recorded, not refused.
    """
    blocking = []
    for line in setup_differences(base_dir, cand_dir, ignore=("model",)):
        if line.startswith(("agent_provider differs", "judged trials of")):
            continue  # the axis itself; tasks lost to infra errors are paired over the rest
        if line == "user_provider differs (num_gpu)":
            continue
        blocking.append(line)
    return blocking, _placement(load_manifest(cand_dir)["user_provider"])


# 두 실행 모두에 판정된 시도가 있는 과제만으로 짝지은 pass^1 차이를 낸다.
def _paired_pass1(base: dict[str, list[bool]], other: dict[str, list[bool]]) -> tuple[float, float, float]:
    usable = [t for t in base if base[t] and other.get(t)]
    return paired_difference({t: base[t] for t in usable}, {t: other[t] for t in usable}, 1)


# 8단계 선택 규칙(analyze select). 후보마다 세 조건을 본다.
# 1. pass^1 차이(짝지은 부트스트랩의 점 추정) ≥ +5%p
# 2. 통과된 규정 위반 0, 그리고 정답에 없는 쓰기 ≤ P0·7B 참조. P1·7B는 거의 행동하지 않아 정답에 없는 쓰기가
#    2건 수준이라, 그것을 기준으로 삼으면 더 자주 행동하는 모델은 재기 전에 탈락이 정해지므로 P0·7B를 상한으로
#    정했다(docs/experiments.md 8단계 재기 전에 고친 것 3).
# 3. 에이전트 호출당 시간 ≤ 기준의 3배
# 만족한 후보 중 pass^1이 가장 높은 것을 고르고(같으면 앞에 준 작은 모델), 시뮬레이터를 CPU에 둔 후보는
# 시험용을 재지 않는다.
def select(base_dir: Path, p0_dir: Path, cand_dirs: list[Path]) -> str:
    """The stage 8 selection rule on the development tasks. base: P1·7B, p0: P0·7B (the write reference)."""
    _check_runs(base_dir, p0_dir, cand_dirs, test=False)
    base_eps = load_episodes(base_dir)
    base_by = successes_by_task(base_eps)
    base_ms = _mean([c["wall_ms"] for c in _agent_calls(base_eps)])
    limit = _unexpected(load_episodes(p0_dir))
    header = (
        "| 실행 | pass^1 | 기준과의 차이 [95% 구간] | pass^4 | 정답에 없는 쓰기 / P0 상한 | 통과된 규정 위반 "
        "| 에이전트 호출 ms (기준의 배수) | 쓰기 정밀도 | 버린 호출 | 형식 오류 | 한자 섞인 답 "
        "| 컨텍스트 한도 종료 "
        "| 시뮬레이터 | 조건 1·2·3 | 채택 가능 |"
    )
    lines = [header, "|---|" + "---|" * (header.count("|") - 2)]

    # 판정에 넣지 않고 보고만 하는 칸: 쓰기 정밀도(정답과 맞는 쓰기 / 성공한 쓰기), 버린 호출, 형식 오류, 한자
    # 섞인 답, 컨텍스트 한도 종료.
    def report_cells(eps: list[dict[str, Any]]) -> list[str]:
        calls = _agent_calls(eps)
        writes = _writes(eps)
        unexpected = _unexpected(eps)
        precision = (
            f"{(writes - unexpected) / writes:.0%} ({writes - unexpected}/{writes})" if writes else "-"
        )
        other, total = _other_script(eps)
        return [
            precision,
            str(sum(c["dropped_calls"] for c in calls)),
            str(_format_errors(calls)),
            f"{other}/{total}",
            str(sum(e["termination"] == "context_limit" for e in _judged(eps))),
        ]

    lines.append(
        f"| {base_dir.name} (기준) | {pass_k(base_by, 1):.1%} | | {pass_k(base_by, 4):.1%} "
        f"| {_unexpected(base_eps)} / {limit} | {_violations(base_eps)} | {base_ms:.0f} | "
        + " | ".join(report_cells(base_eps))
        + f" | {_placement(load_manifest(base_dir)['user_provider'])} | | |"
    )
    eligible: list[tuple[float, Path, str]] = []
    notes: list[str] = []
    for run_dir in cand_dirs:
        eps = load_episodes(run_dir)
        by_task = successes_by_task(eps)
        diff, low, high = _paired_pass1(base_by, by_task)
        unexpected, violations = _unexpected(eps), _violations(eps)
        ms = _mean([c["wall_ms"] for c in _agent_calls(eps)])
        ratio = ms / base_ms if base_ms else float("inf")
        blocking, placement = _pairing(base_dir, run_dir)
        # 조건 1의 1e-9는 부동소수점 오차로 정확히 +5%p인 값이 떨어지지 않게 하는 여유다.
        conditions = [
            diff >= SELECT_MIN_GAIN - 1e-9,
            unexpected <= limit and violations == 0,
            ratio <= SELECT_MAX_CALL_RATIO,
        ]
        ok = all(conditions) and not blocking
        if blocking:
            notes.append(f"- 짝지을 수 없음 ({run_dir.name}): " + "; ".join(blocking))
        if ok:
            eligible.append((pass_k(by_task, 1), run_dir, placement))
        lines.append(
            f"| {run_dir.name} | {pass_k(by_task, 1):.1%} | {diff:+.1%}p [{low:+.1%}, {high:+.1%}] "
            f"| {pass_k(by_task, 4):.1%} | {unexpected} / {limit} | {violations} "
            f"| {ms:.0f} ({ratio:.2f}배) | "
            + " | ".join(report_cells(eps))
            + f" | {placement} | {' · '.join('예' if c else '아니오' for c in conditions)} "
            f"| {'예' if ok else '아니오'} |"
        )
    lines += ["", *notes]
    infra = {d.name: len(load_episodes(d)) - len(_judged(load_episodes(d))) for d in [base_dir, *cand_dirs]}
    if any(infra.values()):
        lines.append("- infra 오류: " + ", ".join(f"{name} {n}" for name, n in infra.items() if n))
    if not eligible:
        lines.append("선택: 없음 (시험용은 재지 않고 '개선 없음'으로 적는다)")
    else:
        # the highest pass^1; max() keeps the first of equals, and the candidates are given smallest first
        _, chosen, placement = max(eligible, key=lambda item: item[0])
        if placement == "CPU":
            lines.append(
                f"선택: {chosen.name} (시뮬레이터를 CPU에서 돌렸으므로 규칙대로 시험용은 재지 않는다)"
            )
        else:
            lines.append(
                f"선택: {chosen.name} (시험용 40과제 × 4회로 P1·7B 기준, P0·7B 쓰기 참조와 함께 잰다)"
            )
    return "\n".join(lines)


# 8단계 시험용 판정(analyze verdict). 2단계 판정표와 같고, 쓰기 조건의 기준만 P0·7B 시험용 참조로 바꿨다. 1:
# pass^1 차이의 95% 구간 아래 끝 > 0, 2: 정답에 없는 쓰기 ≤ P0·7B. 둘 다면 개선, 1만이면 '성공률은 올랐지만
# 위험한 쓰기가 늘었음', 1이 아니면 '개선을 확인하지 못함'. 짝지을 수 없으면 판정하지 않고 오류를 낸다.
def verdict(base_dir: Path, p0_dir: Path, cand_dir: Path) -> str:
    """The stage 8 verdict on the test tasks: the stage 2 table, with the P0·7B count as the write limit."""
    _check_runs(base_dir, p0_dir, [cand_dir], test=True)
    blocking, placement = _pairing(base_dir, cand_dir)
    if blocking:
        raise ValueError(f"{cand_dir.name} cannot be paired with {base_dir.name}: " + "; ".join(blocking))
    base_by = successes_by_task(load_episodes(base_dir))
    cand_eps = load_episodes(cand_dir)
    diff, low, high = _paired_pass1(base_by, successes_by_task(cand_eps))
    unexpected, limit = _unexpected(cand_eps), _unexpected(load_episodes(p0_dir))
    raised, safe = low > 0, unexpected <= limit
    if raised and safe:
        outcome = "개선"
    elif raised:
        outcome = "성공률은 올랐지만 위험한 쓰기가 늘었음"
    else:
        outcome = "개선을 확인하지 못함"
    return "\n".join(
        [
            "| 후보 | pass^1 차이 [95% 구간] | 1: 구간이 0 위 | 정답에 없는 쓰기 / P0·7B | 2: 늘지 않음 "
            "| 시뮬레이터 |",
            "|---|---|---|---|---|---|",
            f"| {cand_dir.name} | {diff:+.1%}p [{low:+.1%}, {high:+.1%}] | {'예' if raised else '아니오'} "
            f"| {unexpected} / {limit} | {'예' if safe else '아니오'} | {placement} |",
            "",
            f"판정: {outcome}",
        ]
    )


# 시뮬레이터 점검에서 읽는 사람이 에피소드마다 적는 코드(1단계 점검표와 같은 항목). 상담원이 아니라
# 고객(시뮬레이터)의 잘못만 적는다. 도구 결과는 앞 300자만 보여 준다.
BLIND_CODES = {
    "outcome": "결과를 바꿨을 만한 잘못이 하나라도 있음",
    "invent": "시나리오에 없는 사실을 지어냄",
    "early_stop": "일이 끝나기 전이나 값을 듣기 전에 대화를 끝냄",
    "agent_role": "상담원처럼 말함",
    "asks_wait": "상담원에게 기다려 달라고 함",
    "wrong_accept": "틀린 안내나 확인 요청에 동의함",
    "off_script": "시나리오에 없는 제안을 받아들임",
    "language": "다른 언어가 섞임",
}
_TOOL_RESULT_CHARS = 300


# 결과와 구성 이름을 가린 시뮬레이터 점검표(8단계 해석 보조). 개발용 실행 둘 이상에서 같은 (과제, 시도) n개를
# 고정 seed로 뽑아 한 장에 섞는다. 한 점검의 실행은 모두 한 장에 넣는다: 추출과 섞기가 고정 seed라 실행 하나를
# 공유하는 두 장은 배치가 같아져 다른 실행이 드러나기 때문이다(docstring).
# 출력: sheet.md(읽는 사람용), key.json(어느 것이 어느 실행인지), marks.template.json(채울 양식).
def blind(*run_dirs: Path, n: int, out_dir: Path) -> None:
    """P5(a): the same n (task, trial) pairs from two or more development runs, shuffled into one sheet,
    with no run names or verdicts. Every run of one check goes into one sheet: the draw and the shuffle
    use fixed seeds, so two sheets that share a run have the same layout and its episodes give the
    other run away.
    sheet.md is for the reader, key.json says which is which, marks.json is what the reader fills in."""
    from support_agent.paths import TASKS
    from support_agent.tasks import load_tasks

    # 실행이 하나면 가릴 것이 없고, 같은 실행을 두 번 주면 key가 둘을 구별하지 못한다. 시험용 기록은 읽지
    # 않는다.
    if len(run_dirs) < 2:
        raise ValueError("the blind check needs two or more runs")
    if len({run_dir.name for run_dir in run_dirs}) != len(run_dirs):
        raise ValueError("the same run twice: the key would not tell them apart")
    for run_dir in run_dirs:
        if is_test_run(run_dir):
            raise ValueError(
                f"{run_dir.name}: the simulator check reads development records only, never test"
            )
    scenarios = {task.id: task.user for path in sorted(TASKS.glob("*.yaml")) for task in load_tasks(path)}
    # 모든 실행에 판정된 시도로 있는 (과제, 시도)만 공통 후보로 삼아 뽑고, 실행 × 표본을 다른 고정 seed로
    # 섞는다.
    keyed = [{(e["task_id"], e["trial"]): e for e in _judged(load_episodes(d))} for d in run_dirs]
    common = sorted(set.intersection(*(set(episodes) for episodes in keyed)))
    chosen = random.Random(BOOTSTRAP_SEED).sample(common, min(n, len(common)))
    items = [
        (run_dir.name, key, episodes[key])
        for run_dir, episodes in zip(run_dirs, keyed, strict=True)
        for key in chosen
    ]
    random.Random(BOOTSTRAP_SEED + 1).shuffle(items)

    sheet = [
        "# 시뮬레이터 점검 (구성 이름을 가림)",
        "",
        "각 에피소드에서 고객(시뮬레이터)의 잘못을 아래 코드로 적는다. 상담원의 잘못은 적지 않는다.",
        "",
        *[f"- `{code}`: {text}" for code, text in BLIND_CODES.items()],
    ]
    key: dict[str, dict[str, Any]] = {}
    for number, (run_name, (task_id, trial), episode) in enumerate(items, start=1):
        label = f"B{number:02d}"
        key[label] = {"run": run_name, "task_id": task_id, "trial": trial}
        scenario = scenarios.get(task_id)
        sheet += ["", f"## {label}", "", "시나리오"]
        if scenario is not None:
            sheet += [
                f"- 상황: {scenario.reason.strip()}",
                f"- 알고 있는 것: {scenario.known.strip()}",
                f"- 모르는 것: {scenario.unknown.strip() or '(없음)'}",
                f"- 지침: {scenario.rules.strip() or '(없음)'}",
            ]
        sheet += ["", "대화"]
        for m in episode["messages"]:
            if m["role"] == "user" and not m.get("harness"):
                sheet.append(f"- 고객: {m['content']}")
            elif m["role"] == "assistant" and m.get("tool_calls"):
                call = m["tool_calls"][0]
                sheet.append(f"- (도구) {call['name']} {json.dumps(call['arguments'], ensure_ascii=False)}")
            elif m["role"] == "tool":
                sheet.append(f"  - (결과) {m['content'][:_TOOL_RESULT_CHARS]}")
            elif m["role"] == "assistant" and m.get("content") and m.get("delivered", True):
                sheet.append(f"- 상담원: {m['content']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sheet.md").write_text("\n".join(sheet) + "\n", encoding="utf-8", newline="\n")
    (out_dir / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    template = json.dumps(dict.fromkeys(key, []), indent=1)
    (out_dir / "marks.template.json").write_text(template, encoding="utf-8")


# 읽는 사람이 채운 marks.json을 key.json과 합쳐 실행별 코드 수를 센다. 모든 항목에 표시가 있는지, 모르는
# 코드가 없는지 먼저 확인한다. 관찰만 하고 이것으로 무엇도 고르지 않는다.
def unblind(out_dir: Path) -> str:
    """Counts of each code per run from the reader's marks.json. Observation only, nothing is chosen by it."""
    key = json.loads((out_dir / "key.json").read_text(encoding="utf-8"))
    marks = json.loads((out_dir / "marks.json").read_text(encoding="utf-8"))
    missing = sorted(set(key) - set(marks))
    if missing:
        raise ValueError(f"no marks for {', '.join(missing)}")
    unknown = sorted({code for codes in marks.values() for code in codes} - set(BLIND_CODES))
    if unknown:
        raise ValueError(f"unknown codes: {', '.join(unknown)} (known: {', '.join(BLIND_CODES)})")
    lines = ["| 실행 | 에피소드 | " + " | ".join(BLIND_CODES) + " |", "|---|---|" + "---|" * len(BLIND_CODES)]
    for run_name in sorted({entry["run"] for entry in key.values()}):
        labels = [label for label, entry in key.items() if entry["run"] == run_name]
        counts = Counter(code for label in labels for code in set(marks[label]))
        lines.append(
            f"| {run_name} | {len(labels)} | " + " | ".join(str(counts[c]) for c in BLIND_CODES) + " |"
        )
    return "\n".join(lines)


# 10단계(R2) 하위 명령을 confirm_report.py로 넘긴다. confirm_report.py가 이 모듈을 가져다 쓰므로, 순환
# import를 피하려고 함수 안에서 가져온다.
def _confirm_command(args: argparse.Namespace) -> None:
    """Stage 10 (R2) commands; the code is in confirm_report.py."""
    from support_agent import confirm_report

    if args.command == "confirm":
        print(confirm_report.confirmation_table(args.run_dirs))
    elif args.command == "confirm-smoke":
        print(confirm_report.smoke_report(args.run_dir))
    elif args.command == "confirm-check":
        report, _ = confirm_report.check(args.base_dir, args.run_dir, test=args.test)
        print(report)
    elif args.command == "confirm-sheet":
        confirm_report.blind_sheet(args.run_dir, n=args.n, out_dir=args.out)
        sheet = args.out / "sheet.md"
        print(f"wrote {sheet}; fill in marks.json from marks.template.json, then confirm-marks")
    else:
        print(confirm_report.unblind(args.out_dir))


# 12단계(시뮬레이터 민감도) 하위 명령을 sim_report.py로 넘긴다. sim_report.py도 이 모듈을 가져다 쓰므로 같은
# 이유로 함수 안에서 가져온다.
def _sim_command(args: argparse.Namespace) -> None:
    """Stage 12 (simulator sensitivity) commands; the code is in sim_report.py."""
    from support_agent import sim_report

    if args.command == "sim-counts":
        print(sim_report.counts_table(args.run_dirs))
    elif args.command == "sim-smoke":
        print(sim_report.smoke_report(args.run_dir))
    else:
        print(sim_report.check(args.base_dir, args.run_dir)[0])


# 하위 명령 정의와 분기. 명령 목록과 쓰임은 모듈 docstring에 있다. name-collisions와 same-setup은 관문·검사
# 명령이라 통과하지 못하면 종료 코드 1을 내, 스크립트에서 바로 멈출 수 있게 한다.
def main() -> None:
    from support_agent.tools import NAME_TOLERANCE

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("table").add_argument("run_dirs", nargs="+", type=Path)
    comparer = commands.add_parser("compare")
    comparer.add_argument("base_dir", type=Path)
    comparer.add_argument("run_dirs", nargs="+", type=Path)
    commands.add_parser("misses").add_argument("run_dir", type=Path)
    sampler = commands.add_parser("sample")
    sampler.add_argument("run_dir", type=Path)
    sampler.add_argument("--n", type=int, default=20)
    commands.add_parser("voice").add_argument("run_dirs", nargs="+", type=Path)
    commands.add_parser("voice-worst").add_argument("run_dir", type=Path)
    commands.add_parser("claims").add_argument("run_dirs", nargs="+", type=Path)
    pairing = commands.add_parser("same-setup")
    pairing.add_argument("run_a", type=Path)
    pairing.add_argument("run_b", type=Path)
    pairing.add_argument("--ignore", nargs="*", default=["voice"], help="config axes that may differ")
    pairing.add_argument(
        "--ignore-prompt",
        nargs="*",
        default=[],
        help="prompt hashes that may differ (V4: tools agent_system)",
    )
    commands.add_parser("smoke").add_argument("run_dir", type=Path)
    selector = commands.add_parser("select")
    selector.add_argument("base_dir", type=Path, help="P1 7B on the development tasks")
    selector.add_argument("p0_dir", type=Path, help="P0 7B on the development tasks (the write reference)")
    selector.add_argument("run_dirs", nargs="+", type=Path, help="candidates, the smallest model first")
    judging = commands.add_parser("verdict")
    judging.add_argument("base_dir", type=Path, help="P1 7B on the test tasks")
    judging.add_argument("p0_dir", type=Path, help="P0 7B on the test tasks (the write reference)")
    judging.add_argument("run_dir", type=Path, help="the selected candidate on the test tasks")
    blinder = commands.add_parser("blind")
    blinder.add_argument("run_dirs", nargs="+", type=Path, help="every run of one check, in one sheet")
    blinder.add_argument("--n", type=int, default=20)
    blinder.add_argument("--out", type=Path, required=True)
    commands.add_parser("unblind").add_argument("out_dir", type=Path)
    colliding = commands.add_parser("name-collisions")
    colliding.add_argument(
        "--k", type=int, default=NAME_TOLERANCE, help="jamo edits verify_caller allows (default: registered)"
    )
    commands.add_parser("caller-id").add_argument("run_dirs", nargs="+", type=Path)
    commands.add_parser("confirm").add_argument("run_dirs", nargs="+", type=Path)
    commands.add_parser("confirm-smoke").add_argument("run_dir", type=Path)
    checking = commands.add_parser("confirm-check")
    checking.add_argument("base_dir", type=Path, help="the stage 8 M2b record (R0)")
    checking.add_argument("run_dir", type=Path, help="the R2 run on the same tasks")
    checking.add_argument("--test", action="store_true", help="the test verdict (default: development)")
    sheeting = commands.add_parser("confirm-sheet")
    sheeting.add_argument("run_dir", type=Path)
    sheeting.add_argument("--n", type=int, default=20)
    sheeting.add_argument("--out", type=Path, required=True)
    commands.add_parser("confirm-marks").add_argument("out_dir", type=Path)
    commands.add_parser("sim-counts").add_argument("run_dirs", nargs="+", type=Path)
    commands.add_parser("sim-smoke").add_argument("run_dir", type=Path)
    sim_checking = commands.add_parser("sim-check")
    sim_checking.add_argument("base_dir", type=Path, help="the development run with the 7B simulator (U0)")
    sim_checking.add_argument("run_dir", type=Path, help="the same setup with another simulator")
    args = parser.parse_args()

    # Windows 콘솔의 기본 인코딩으로는 한국어 표가 깨지거나 오류가 나므로 표준 출력을 UTF-8로 바꾼다.
    sys.stdout.reconfigure(encoding="utf-8")  # Korean on a Windows console
    if args.command == "table":
        print(table(args.run_dirs))
    elif args.command == "compare":
        print(compare(args.base_dir, args.run_dirs))
    elif args.command == "misses":
        print(misses(args.run_dir))
    elif args.command == "voice":
        print(voice_metrics.voice_table({d.name: load_episodes(d) for d in args.run_dirs}))
    elif args.command == "voice-worst":
        # voice-worst는 엔티티(주문 번호, 이메일 등)를 잃은 발화 목록이고, V2 정규화
        # 규칙(voice/normalize.py)은 이 목록을 개발용 기록에서만 보고 만들었다. 그래서 시험용 실행은
        # 거부한다(docs/experiments.md 5단계).
        if is_test_run(args.run_dir):
            sys.exit("voice-worst is for the development tasks: nothing is built from test records")
        print(voice_metrics.worst(load_episodes(args.run_dir)))
    elif args.command == "claims":
        print(claims_table(args.run_dirs))
    elif args.command == "smoke":
        print(smoke_report(args.run_dir))
    elif args.command == "select":
        print(select(args.base_dir, args.p0_dir, args.run_dirs))
    elif args.command == "verdict":
        print(verdict(args.base_dir, args.p0_dir, args.run_dir))
    elif args.command == "blind":
        blind(*args.run_dirs, n=args.n, out_dir=args.out)
        print(f"wrote {args.out / 'sheet.md'}; fill in marks.json from marks.template.json, then run unblind")
    elif args.command == "unblind":
        print(unblind(args.out_dir))
    elif args.command == "name-collisions":
        report, passed = collision_report(args.k)
        print(report)
        if not passed:
            sys.exit(1)
    elif args.command == "caller-id":
        print(caller_id_table({d.name: load_episodes(d) for d in args.run_dirs}))
    elif args.command.startswith("confirm"):
        _confirm_command(args)
    elif args.command.startswith("sim-"):
        _sim_command(args)
    elif args.command == "same-setup":
        differences = setup_differences(args.run_a, args.run_b, tuple(args.ignore), tuple(args.ignore_prompt))
        print("\n".join(differences) or "same setup")
        if differences:
            sys.exit(1)
    else:
        print(sample(args.run_dir, args.n))


if __name__ == "__main__":
    main()
