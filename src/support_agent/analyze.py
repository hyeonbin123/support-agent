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
    uv run python -m support_agent.analyze name-collisions [--k 1]
    uv run python -m support_agent.analyze caller-id reports/<V2 run> reports/<V4 run> [...]

`table` prints the markdown tables that go into docs/experiments.md. `misses` lists episodes whose database
matched but whose required value was not found, so that a person can check the value matcher. `sample` draws
episodes with a fixed seed for the simulator check ("시뮬레이터 점검").
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from support_agent.config import JUDGED, RunConfig
from support_agent.judge import pass_hat_k
from support_agent.voice import metrics as voice_metrics

BOOTSTRAP_ROUNDS = 10_000
BOOTSTRAP_SEED = 20260919
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
# Replies held back by a guard are well formed; they are counted on their own.
_NOT_A_FORMAT_ERROR = (None, "stall", "unbacked_claim")


def load_episodes(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "episodes.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def task_type(task_id: str, types: dict[str, str]) -> str:
    return types.get(task_id, "?")


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


def successes_by_task(episodes: list[dict[str, Any]]) -> dict[str, list[bool]]:
    by_task: dict[str, list[bool]] = {}
    for e in episodes:
        if e["status"] != "infra_error":
            by_task.setdefault(e["task_id"], []).append(is_success(e))
    return by_task


def pass_k(by_task: dict[str, list[bool]], k: int) -> float:
    values = [pass_hat_k(len(v), sum(v), k) for v in by_task.values() if len(v) >= k]
    return sum(values) / len(values) if values else float("nan")


def bootstrap_interval(by_task: dict[str, list[bool]], k: int) -> tuple[float, float]:
    """95% interval of pass^k, resampling tasks (not episodes): trials of one task are not independent."""
    tasks = [v for v in by_task.values() if len(v) >= k]
    if not tasks:
        return float("nan"), float("nan")
    rng = random.Random(BOOTSTRAP_SEED)
    scores = [pass_hat_k(len(v), sum(v), k) for v in tasks]
    means = sorted(sum(rng.choice(scores) for _ in scores) / len(scores) for _ in range(BOOTSTRAP_ROUNDS))
    return means[int(0.025 * BOOTSTRAP_ROUNDS)], means[int(0.975 * BOOTSTRAP_ROUNDS) - 1]


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


def cell(by_task: dict[str, list[bool]], k: int) -> str:
    low, high = bootstrap_interval(by_task, k)
    return f"{pass_k(by_task, k):.1%} [{low:.1%}, {high:.1%}]"


def load_task_types() -> dict[str, str]:
    from support_agent.paths import TASKS
    from support_agent.tasks import load_tasks

    return {task.id: task.type for path in sorted(TASKS.glob("*.yaml")) for task in load_tasks(path)}


def table(run_dirs: list[Path]) -> str:
    types = load_task_types()
    runs = {d.name: load_episodes(d) for d in run_dirs}
    lines: list[str] = []
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


def misses(run_dir: Path) -> str:
    out: list[str] = []
    for e in load_episodes(run_dir):
        verdict = e["verdict"]
        if verdict and verdict["judged"] and verdict["db_match"] and not all(verdict["values"].values()):
            missing = [label for label, found in verdict["values"].items() if not found]
            out.append(f"## {e['task_id']} #{e['trial']}: missing {missing}")
            out += [f"- {m['content']}" for m in e["messages"] if m["role"] == "assistant" and m["content"]]
    return "\n".join(out) or "no episode failed on a required value alone"


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
        elif following["index"] in tools:
            tool = tools[following["index"]]
            out.append(("write" if tool["ok"] else "write_failed") if tool["write"] else "read")
        elif following["format_error"] in (None, "stall"):
            out.append("reply")
        else:
            out.append("format_error")
    return out


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


def load_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def is_test_run(run_dir: Path) -> bool:
    """True when the run is on the test tasks: those records are not for building anything from."""
    manifest = load_manifest(run_dir)
    return "test" in Path(str(manifest.get("tasks_file", ""))).stem or any(
        task_id.startswith("test-") for task_id in manifest.get("task_sha256", {})
    )


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


def seed_names() -> dict[str, str]:
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from support_agent import db
    from support_agent.seed import build_seed_engine

    with Session(build_seed_engine()) as session:
        return {c.id: c.name for c in session.scalars(select(db.Customer))}


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


# verify_caller refusals that mean the harness gave no usable number: a bug, never the agent's doing.
CALLER_HARNESS_ERRORS = ("no_caller_number", "caller_not_registered", "caller_shared")


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
# Thresholds of the stage 8 rules in docs/experiments.md, written down and committed before measuring.

RELOAD_MS = 1_000.0  # a call whose load_duration is longer than this waited for its model to be loaded again
SMOKE_REFERENCE_AGENT_CALL_MS = 588.9  # P1·7B, reports/20260920-190957-...-p1 (Ollama 0.34.2): a pre-filter
SMOKE_MAX_CALL_RATIO = 3.0
SMOKE_MAX_PROMPT_EVAL_MEDIAN_MS = 500.0  # far below a prompt evaluated from scratch, far above a cached one
SMOKE_MAX_FORMAT_ERRORS_PER_EPISODE = 1.0  # twice the P1·7B rate (48 in 96 episodes)
SMOKE_MAX_OTHER_SCRIPT_SHARE = 0.10
SELECT_MIN_GAIN = 0.05
SELECT_MAX_CALL_RATIO = 3.0
_THINK_TAG = re.compile(r"</?think>")


def _judged(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in episodes if e["status"] != "infra_error"]


def _agent_calls(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [c for e in _judged(episodes) for c in e["llm_calls"] if c["who"] == "agent"]


def _format_errors(calls: list[dict[str, Any]]) -> int:
    return sum(c["format_error"] not in _NOT_A_FORMAT_ERROR for c in calls)


def _think_leaks(calls: list[dict[str, Any]]) -> int:
    """Replies with a reasoning tag in their text: a thinking model whose thoughts were not split off."""
    return sum(bool(_THINK_TAG.search(c.get("text") or "")) for c in calls)


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


def _other_script(episodes: list[dict[str, Any]]) -> tuple[int, int]:
    from support_agent.agent import in_another_language

    replies = [r for e in _judged(episodes) for r in delivered_replies(e)]
    return sum(in_another_language(r) for r in replies), len(replies)


def _reloads(episodes: list[dict[str, Any]]) -> int:
    return sum((c.get("load_ms") or 0) > RELOAD_MS for e in _judged(episodes) for c in e["llm_calls"])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _unexpected(episodes: list[dict[str, Any]]) -> int:
    return sum(e["verdict"]["unexpected_writes"] for e in _judged(episodes))


def _violations(episodes: list[dict[str, Any]]) -> int:
    return sum(len(e["verdict"]["policy_violations"]) for e in _judged(episodes))


def _writes(episodes: list[dict[str, Any]]) -> int:
    return sum(1 for e in _judged(episodes) for t in e["tool_calls"] if t["ok"] and t["write"])


def load_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _gpu_share(ps: list[dict[str, Any]], model: str) -> float | None:
    """size_vram / size of a loaded model; None when the server did not hold it."""
    for m in ps:
        if m.get("name") == model:
            size = m.get("size") or 0
            return (m.get("size_vram") or 0) / size if size else 0.0
    return None


def _placement(user_provider: dict[str, Any]) -> str:
    num_gpu = user_provider.get("num_gpu")
    if num_gpu is None:
        return "GPU"
    return "CPU" if num_gpu == 0 else f"일부 CPU (num_gpu {num_gpu})"


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


def _paired_pass1(base: dict[str, list[bool]], other: dict[str, list[bool]]) -> tuple[float, float, float]:
    usable = [t for t in base if base[t] and other.get(t)]
    return paired_difference({t: base[t] for t in usable}, {t: other[t] for t in usable}, 1)


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


def blind(*run_dirs: Path, n: int, out_dir: Path) -> None:
    """P5(a): the same n (task, trial) pairs from two or more development runs, shuffled into one sheet,
    with no run names or verdicts. Every run of one check goes into one sheet: the draw and the shuffle
    use fixed seeds, so two sheets that share a run have the same layout and its episodes give the
    other run away.
    sheet.md is for the reader, key.json says which is which, marks.json is what the reader fills in."""
    from support_agent.paths import TASKS
    from support_agent.tasks import load_tasks

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


def main() -> None:
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
    colliding.add_argument("--k", type=int, default=1, help="jamo edits verify_caller allows")
    commands.add_parser("caller-id").add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()

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
    elif args.command == "same-setup":
        differences = setup_differences(args.run_a, args.run_b, tuple(args.ignore), tuple(args.ignore_prompt))
        print("\n".join(differences) or "same setup")
        if differences:
            sys.exit(1)
    else:
        print(sample(args.run_dir, args.n))


if __name__ == "__main__":
    main()
