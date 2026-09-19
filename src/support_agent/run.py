"""Run tasks x trials with a local model and write one JSONL record per episode.

Usage:
    uv run python -m support_agent.run --tasks smoke --trials 1
    uv run python -m support_agent.run --tasks smoke --model qwen2.5:14b-instruct --policy P1 --label p1-14b

Records go to outputs/runs/<run_id>/ (git-ignored). Pass --official to write to reports/ (committed); that
needs a clean working tree, and test task files also need --allow-test. The GPU is shared with other
projects: the run refuses to start when something else holds GPU memory (see --force).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from support_agent import db
from support_agent.agent import build_system_prompt, load_policy, visible_tools
from support_agent.config import RunConfig
from support_agent.episode import gold_dump_of, run_episode
from support_agent.judge import pass_k_table
from support_agent.ollama import OllamaProvider
from support_agent.paths import OUTPUTS, REPORTS, ROOT
from support_agent.records import EpisodeResult
from support_agent.seed import build_seed_engine
from support_agent.tasks import load_tasks
from support_agent.tools import build_registry
from support_agent.user_sim import LLMUser, build_user_prompt

OTHER_GPU_USE_LIMIT_MIB = 3000  # desktop apps take 1-2 GiB; more than this means another job is running


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def gpu_memory_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def gpu_used_by_others_mib(provider: OllamaProvider) -> int | None:
    """GPU memory held by anything but Ollama. `nvidia-smi --query-compute-apps` is useless on Windows
    (it lists every desktop app), so this subtracts what Ollama reports from the total in use."""
    used = gpu_memory_used_mib()
    if used is None:
        return None
    ollama_mib = sum(int(m.get("size_vram") or 0) for m in provider.loaded_models()) // (1024 * 1024)
    return max(used - ollama_mib, 0)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def summarise(results: list[EpisodeResult]) -> dict[str, Any]:
    counted = [r for r in results if r.status != "infra_error"]
    by_task: dict[str, list[bool]] = {}
    for r in counted:
        by_task.setdefault(r.task_id, []).append(bool(r.verdict and r.verdict.success))
    trials = min((len(v) for v in by_task.values()), default=0)
    agent_calls = [c for r in counted for c in r.llm_calls if c.who == "agent"]
    user_calls = [c for r in counted for c in r.llm_calls if c.who == "user"]

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 1) if values else 0.0

    return {
        "episodes": len(results),
        "infra_errors": sum(r.status == "infra_error" for r in results),
        "truncated": sum(r.status == "truncated" for r in results),
        "terminations": {
            t: sum(r.termination == t for r in results) for t in sorted({r.termination for r in results})
        },
        "successes": sum(sum(v) for v in by_task.values()),
        "pass_hat_k": pass_k_table({k: v[:trials] for k, v in by_task.items()}) if trials else {},
        "by_task": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_task.items())},
        "unexpected_writes": sum(r.verdict.unexpected_writes for r in counted if r.verdict),
        "policy_violations": sum(len(r.verdict.policy_violations) for r in counted if r.verdict),
        "policy_blocks": sum(len(r.verdict.policy_blocks) for r in counted if r.verdict),
        "auth_blocks": sum(r.verdict.auth_blocks for r in counted if r.verdict),
        "format_errors": sum(c.format_error is not None for c in agent_calls),
        "dropped_calls": sum(c.dropped_calls for c in agent_calls),
        "seconds_per_episode": mean([r.wall_seconds for r in counted]),
        "agent_calls_per_episode": mean([float(sum(c.who == "agent" for c in r.llm_calls)) for r in counted]),
        "agent_call_ms": mean([c.wall_ms for c in agent_calls]),
        "agent_prompt_eval_ms": mean([c.prompt_eval_ms for c in agent_calls]),
        "agent_prompt_tokens_max": max((c.prompt_tokens for c in agent_calls), default=0),
        "user_call_ms": mean([c.wall_ms for c in user_calls]),
        "model_load_ms_total": round(sum(c.load_ms for c in agent_calls + user_calls), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", default="smoke", help="task file name in tasks/ (without .yaml) or a path")
    parser.add_argument("--task-id", action="append", help="run only these task ids (repeatable)")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--model", default=RunConfig.model)
    parser.add_argument("--user-model", default=None, help="simulator model (default: same as --model)")
    parser.add_argument("--policy", choices=["P0", "P1"], default="P0")
    parser.add_argument("--reasoning", choices=["R0", "R1"], default="R0", help="R2 arrives in stage 2")
    parser.add_argument("--num-ctx", type=int, default=RunConfig.num_ctx)
    parser.add_argument("--label", default="", help="short name added to the run id")
    parser.add_argument("--official", action="store_true", help="write to reports/ (needs a clean tree)")
    parser.add_argument("--allow-test", action="store_true", help="required for test task files")
    parser.add_argument("--force", action="store_true", help="skip the check that the GPU is free")
    args = parser.parse_args()

    if "test" in Path(args.tasks).stem and not args.allow_test:
        parser.error("test tasks are measured once per stage; pass --allow-test when the stage is done")
    dirty = bool(_git("status", "--porcelain"))
    if args.official and dirty:
        parser.error("--official needs a clean working tree, so that the commit describes the code that ran")

    config = RunConfig(
        model=args.model,
        user_model=args.user_model or args.model,
        policy=args.policy,
        reasoning=args.reasoning,
        num_ctx=args.num_ctx,
    )
    tasks = load_tasks(args.tasks)
    if args.task_id:
        tasks = [t for t in tasks if t.id in set(args.task_id)]
    if not tasks:
        parser.error("no tasks selected")

    provider = OllamaProvider(config.model, num_ctx=config.num_ctx)
    # One provider object when both roles use the same model: equal runner options, so no reload.
    same = config.user_model == config.model
    user_provider = provider if same else OllamaProvider(config.user_model, num_ctx=config.num_ctx)

    others = gpu_used_by_others_mib(provider)
    if others is not None and others > OTHER_GPU_USE_LIMIT_MIB and not args.force:
        sys.exit(
            f"{others} MiB of GPU memory is in use by something other than Ollama (limit "
            f"{OTHER_GPU_USE_LIMIT_MIB}). Another job is probably running; not starting (--force overrides)."
        )

    registry = build_registry()
    seed_engine = build_seed_engine()
    seed_dump = db.dump_db(seed_engine)
    policy_text = load_policy()
    started = datetime.now(UTC)
    run_id = "-".join(
        p
        for p in [
            started.strftime("%Y%m%d-%H%M%S"),
            safe_name(config.model),
            config.policy,
            config.reasoning,
            safe_name(args.label),
        ]
        if p
    )
    run_dir = (REPORTS if args.official else OUTPUTS / "runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    tools_json = json.dumps(visible_tools(registry, config), ensure_ascii=False, sort_keys=True)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "started_at": started.isoformat(timespec="seconds"),
        "official": args.official,
        "commit": _git("rev-parse", "HEAD"),
        "dirty": dirty,
        "config": config.to_dict(),
        "trials": args.trials,
        "tasks_file": str(args.tasks),
        "task_sha256": {t.id: t.sha256() for t in tasks},
        "seed_hash": db.state_hash(seed_dump),
        "prompt_sha256": {
            "policy": text_sha256(policy_text),
            "agent_system": text_sha256(build_system_prompt(policy_text, tasks[0].now)),
            "user_sim": text_sha256(build_user_prompt(tasks[0])),
            "tools": text_sha256(tools_json),
        },
        "agent_provider": provider.describe(),
        "user_provider": user_provider.describe(),
        "gpu_mib_used_by_others_at_start": others,
        "versions": {"python": sys.version.split()[0]}
        | {name: version(name) for name in ("sqlalchemy", "pydantic", "httpx", "pyyaml")},
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )
    # The seed dump is kept once per hash, so that old records can be re-judged after the generator changes.
    seed_file = (REPORTS if args.official else OUTPUTS) / "seeds" / f"{manifest['seed_hash'][:16]}.json"
    if not seed_file.exists():
        seed_file.parent.mkdir(parents=True, exist_ok=True)
        seed_file.write_text(json.dumps(seed_dump, ensure_ascii=False), encoding="utf-8", newline="\n")

    print(f"run {run_id}: {len(tasks)} tasks x {args.trials} trials -> {run_dir}")
    print(f"loading {config.model} ... {provider.preload() / 1000:.1f} s")
    results: list[EpisodeResult] = []
    with (run_dir / "episodes.jsonl").open("a", encoding="utf-8", newline="\n") as out:
        for task in tasks:
            gold_dump = gold_dump_of(task, seed_engine, registry)
            for trial in range(args.trials):
                result = run_episode(
                    task,
                    trial,
                    config=config,
                    provider=provider,
                    user=LLMUser(user_provider, task, config, trial),
                    registry=registry,
                    seed_engine=seed_engine,
                    policy_text=policy_text,
                    gold_dump=gold_dump,
                    run_id=run_id,
                )
                results.append(result)
                out.write(result.to_json_line() + "\n")
                out.flush()
                mark = "infra" if result.verdict is None else ("PASS" if result.verdict.success else "fail")
                print(f"  {task.id} #{trial}: {mark:5} {result.termination:22} {result.wall_seconds:6.1f} s")

    summary = summarise(results)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
