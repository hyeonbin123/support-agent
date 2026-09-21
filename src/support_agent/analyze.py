"""Recompute numbers from the episode records of a run. Nothing here calls a model.

Usage:
    uv run python -m support_agent.analyze table outputs/runs/<run_id> [more run dirs ...]
    uv run python -m support_agent.analyze compare reports/<baseline> reports/<candidate> [...]
    uv run python -m support_agent.analyze misses outputs/runs/<run_id>
    uv run python -m support_agent.analyze sample outputs/runs/<run_id> --n 20
    uv run python -m support_agent.analyze voice reports/<text run> reports/<voice run> [...]
    uv run python -m support_agent.analyze voice-worst reports/<voice run on the development tasks>
    uv run python -m support_agent.analyze same-setup reports/<run> reports/<run to pair it with>

`table` prints the markdown tables that go into docs/experiments.md. `misses` lists episodes whose database
matched but whose required value was not found, so that a person can check the value matcher. `sample` draws
episodes with a fixed seed for the simulator check ("시뮬레이터 점검").
"""

from __future__ import annotations

import argparse
import json
import random
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
            diff, low, high = paired_difference(base, other, k)
            cells += [f"{pass_k(other, k):.1%}", f"{diff:+.1%}p [{low:+.1%}, {high:+.1%}]"]
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
            f"| {sum(c['format_error'] not in (None, 'stall') for c in agent_calls)} "
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


def load_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def is_test_run(run_dir: Path) -> bool:
    """True when the run is on the test tasks: those records are not for building anything from."""
    manifest = load_manifest(run_dir)
    return "test" in Path(str(manifest.get("tasks_file", ""))).stem or any(
        task_id.startswith("test-") for task_id in manifest.get("task_sha256", {})
    )


def setup_differences(run_a: Path, run_b: Path, ignore: tuple[str, ...] = ("voice",)) -> list[str]:
    """Why two runs may not be paired as "the same setup but for `ignore`". Empty when they may.

    Compared: every config field but the ignored axes (a field an older manifest lacks counts as its default),
    trials, task hashes, seed hash, prompt hashes, both providers (model digest, server version, options), and
    per task the trials that were judged.
    """
    a, b = load_manifest(run_a), load_manifest(run_b)
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
    pairing = commands.add_parser("same-setup")
    pairing.add_argument("run_a", type=Path)
    pairing.add_argument("run_b", type=Path)
    pairing.add_argument("--ignore", nargs="*", default=["voice"], help="config axes that may differ")
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
    elif args.command == "same-setup":
        differences = setup_differences(args.run_a, args.run_b, tuple(args.ignore))
        print("\n".join(differences) or "same setup")
        if differences:
            sys.exit(1)
    else:
        print(sample(args.run_dir, args.n))


if __name__ == "__main__":
    main()
