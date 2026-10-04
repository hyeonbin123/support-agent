"""Run tasks x trials with a local model and write one JSONL record per episode.

Usage:
    uv run python -m support_agent.run --tasks smoke --trials 1
    uv run python -m support_agent.run --tasks smoke --model qwen2.5:14b-instruct --policy P1 --label p1-14b
    uv run python -m support_agent.run --tasks smoke --policy P1 --model qwen3:4b-instruct-2507-q4_K_M
        --think off --num-ctx 12288 --user-num-ctx 12288 --label smoke-m2a   (one command line)

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
from support_agent.confirm import CONFIRM_CODE
from support_agent.episode import gold_dump_of, run_episode
from support_agent.judge import pass_k_table
from support_agent.ollama import OllamaProvider
from support_agent.paths import OUTPUTS, REPORTS, ROOT
from support_agent.records import EpisodeResult
from support_agent.seed import build_seed_engine
from support_agent.tasks import load_tasks
from support_agent.tools import NAME_TOLERANCE, build_registry
from support_agent.user_sim import LLMUser, build_user_prompt
from support_agent.voice.speech import versions as speech_versions

MAX_CONSECUTIVE_INFRA_ERRORS = 3  # the model server is probably down: stop instead of burning the task list
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


def ollama_ps(provider: OllamaProvider) -> list[dict[str, Any]]:
    """What the server holds in memory (name, size, size_vram); [] for providers that cannot say."""
    loaded = getattr(provider, "loaded_models", None)
    return loaded() if callable(loaded) else []


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def summarise(results: list[EpisodeResult]) -> dict[str, Any]:
    counted = [r for r in results if r.status != "infra_error"]
    by_task: dict[str, list[bool]] = {}
    for r in counted:
        by_task.setdefault(r.task_id, []).append(bool(r.verdict and r.verdict.success))
    # A task whose every trial was an infra error must not vanish from the table without a trace.
    unmeasured = sorted({r.task_id for r in results} - set(by_task))
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
        # pass^k uses every valid trial of a task (C(c,k)/C(n,k) with the task's own n), for k up to the
        # smallest n. Trials lost to infra errors therefore shrink k, never the success counts.
        "pass_hat_k": pass_k_table(by_task) if by_task else {},
        "by_task": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_task.items())},
        "tasks_without_valid_trials": unmeasured,
        "successes_with_unexpected_writes": sum(
            bool(r.verdict and r.verdict.success and r.verdict.unexpected_writes) for r in counted
        ),
        "unexpected_writes": sum(r.verdict.unexpected_writes for r in counted if r.verdict),
        "policy_violations": sum(len(r.verdict.policy_violations) for r in counted if r.verdict),
        "policy_blocks": sum(len(r.verdict.policy_blocks) for r in counted if r.verdict),
        "auth_blocks": sum(r.verdict.auth_blocks for r in counted if r.verdict),
        "format_errors": sum(c.format_error not in (None, "stall", "unbacked_claim") for c in agent_calls),
        "held_claims": sum(c.format_error == "unbacked_claim" for c in agent_calls),
        "confirmation_requests": sum(t.error_code == CONFIRM_CODE for r in counted for t in r.tool_calls),
        "dropped_calls": sum(c.dropped_calls for c in agent_calls),
        "seconds_per_episode": mean([r.wall_seconds for r in counted]),
        "agent_calls_per_episode": mean([float(sum(c.who == "agent" for c in r.llm_calls)) for r in counted]),
        "agent_call_ms": mean([c.wall_ms for c in agent_calls]),
        "agent_prompt_eval_ms": mean([c.prompt_eval_ms for c in agent_calls]),
        "agent_prompt_tokens_max": max((c.prompt_tokens for c in agent_calls), default=0),
        # The context-limit check reads prompt_tokens; calls that report none are blind spots.
        "agent_calls_without_prompt_tokens": sum(c.prompt_tokens == 0 for c in agent_calls),
        "user_call_ms": mean([c.wall_ms for c in user_calls]),
        "model_load_ms_total": round(sum(c.load_ms for c in agent_calls + user_calls), 1),
    }


def build_channel(config: RunConfig, tts_device: str, stt_device: str):
    """The speech channel of V1/V2/V4 (needs the `voice` dependency group), or None for text."""
    if config.voice == "V0":
        return None
    from support_agent.voice.channel import SpeechChannel
    from support_agent.voice.speech import MeloSpeaker, WhisperListener

    normalizer = None
    if config.voice in ("V2", "V4"):  # V4 hears exactly as V2 does; only the identification differs
        from support_agent.voice.normalize import normalize_heard as normalizer
    print(f"loading the speech models (tts on {tts_device}, stt on {stt_device}) ...")
    return SpeechChannel(
        MeloSpeaker(device=tts_device),
        WhisperListener(device=stt_device),
        normalizer=normalizer,
        cache_path=OUTPUTS / "voice-cache" / "roundtrips.jsonl",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", default="smoke", help="task file name in tasks/ (without .yaml) or a path")
    parser.add_argument("--task-id", action="append", help="run only these task ids (repeatable)")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--model", default=RunConfig.model)
    parser.add_argument("--user-model", default=None, help="simulator model (default: qwen2.5:7b-instruct)")
    parser.add_argument("--policy", choices=["P0", "P1"], default="P0")
    parser.add_argument(
        "--reasoning",
        choices=["R0", "R1", "R2"],
        default="R0",
        help="R1: the think tool; R2: a write runs only after the customer agreed to its preview",
    )
    parser.add_argument(
        "--guard", choices=["G0", "G1", "G2"], default="G0", help="G1: hold back replies that only promise"
    )
    parser.add_argument(
        "--rescue", choices=["F0", "F1"], default="F0", help="F1: run tool calls leaked into text"
    )
    parser.add_argument(
        "--claims",
        choices=["C0", "C1", "C2"],
        default="C0",
        help="C1: hold back a reply that says work is done which no tool result shows; C2: sampled retry",
    )
    parser.add_argument(
        "--user-on-cpu",
        action="store_true",
        help="run the simulator model on the CPU (when the agent model leaves no GPU memory for it)",
    )
    parser.add_argument(
        "--user-num-gpu",
        type=int,
        default=None,
        help="layers of the simulator model on the GPU, the rest on the CPU (0 is --user-on-cpu)",
    )
    parser.add_argument(
        "--think",
        choices=["on", "off"],
        default=None,
        help="send think to the agent model only (default: not sent; models without thinking refuse it)",
    )
    parser.add_argument(
        "--voice",
        choices=["V0", "V1", "V2", "V4"],
        default="V0",
        help="V1: the customer is heard through speech synthesis and recognition; V2: and a normaliser; "
        "V4: V2, and verify_caller checks the name against the caller's number instead of find_customer",
    )
    parser.add_argument("--tts-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--stt-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num-ctx", type=int, default=RunConfig.num_ctx)
    parser.add_argument(
        "--user-num-ctx", type=int, default=None, help="the simulator's num_ctx (default: --num-ctx)"
    )
    parser.add_argument("--label", default="", help="short name added to the run id")
    parser.add_argument("--official", action="store_true", help="write to reports/ (needs a clean tree)")
    parser.add_argument("--allow-test", action="store_true", help="required for test task files")
    parser.add_argument("--force", action="store_true", help="skip the check that the GPU is free")
    args = parser.parse_args()

    if "test" in Path(args.tasks).stem and not args.allow_test:
        parser.error("test tasks are measured once per stage; pass --allow-test when the stage is done")
    commit = _git("rev-parse", "HEAD")
    # Records of earlier official runs wait in reports/ until they are committed; they are not code.
    dirty = bool(_git("status", "--porcelain", "--", ".", ":(exclude)reports"))
    if args.official and (dirty or not commit):
        parser.error(
            "--official needs git and a clean working tree, so that the commit describes the code that ran"
        )

    if args.user_on_cpu and args.user_num_gpu not in (None, 0):
        parser.error("--user-on-cpu means --user-num-gpu 0")
    user_num_gpu = 0 if args.user_on_cpu else args.user_num_gpu
    user_num_ctx = args.user_num_ctx or args.num_ctx
    user_model = args.user_model or RunConfig.user_model
    if user_model == args.model and (user_num_ctx != args.num_ctx or user_num_gpu is not None):
        # Ollama keeps one runner per model; other runner options reload it, and here that is every turn.
        parser.error(
            "the agent and the simulator are the same model, so they need the same runner options: "
            "drop --user-num-ctx/--user-on-cpu/--user-num-gpu or use another simulator model"
        )
    think = None if args.think is None else args.think == "on"

    config = RunConfig(
        model=args.model,
        user_model=user_model,
        policy=args.policy,
        reasoning=args.reasoning,
        guard=args.guard,
        rescue=args.rescue,
        voice=args.voice,
        claims=args.claims,
        num_ctx=args.num_ctx,
        user_num_ctx=user_num_ctx,
    )
    tasks = load_tasks(args.tasks)
    if args.task_id:
        unknown = sorted(set(args.task_id) - {t.id for t in tasks})
        if unknown:
            parser.error(f"unknown task ids: {unknown}")
        tasks = [t for t in tasks if t.id in set(args.task_id)]
    if not tasks:
        parser.error("no tasks selected")

    # think is a request field, not a runner option, so it never reloads a model; it goes to the agent only.
    provider = OllamaProvider(
        config.model, num_ctx=config.num_ctx, **({} if think is None else {"think": think})
    )
    # One provider object when both roles use the same model and nothing differs: equal runner options.
    same = config.user_model == config.model and think is None
    user_provider = (
        provider
        if same
        else OllamaProvider(config.user_model, num_ctx=config.user_num_ctx, num_gpu=user_num_gpu)
    )

    others = gpu_used_by_others_mib(provider)
    if others is not None and others > OTHER_GPU_USE_LIMIT_MIB and not args.force:
        sys.exit(
            f"{others} MiB of GPU memory is in use by something other than Ollama (limit "
            f"{OTHER_GPU_USE_LIMIT_MIB}). Another job is probably running; not starting (--force overrides)."
        )

    registry = build_registry(caller_id=config.caller_id)
    seed_engine = build_seed_engine()
    seed_dump = db.dump_db(seed_engine)
    policy_text = load_policy()
    # Everything that can fail without the model fails here, before hours of GPU time are spent.
    gold_dumps = {task.id: gold_dump_of(task, seed_engine, registry) for task in tasks}
    channel = build_channel(config, args.tts_device, args.stt_device)
    print(f"loading {config.model} ... {provider.preload() / 1000:.1f} s")
    if user_provider is not provider:
        print(f"loading {config.user_model} ... {user_provider.preload() / 1000:.1f} s")
        # The server may evict the first model to load the second even when the first then fits next to the
        # second (seen 2026-10-04: qwen2.5 7B evicted qwen3 4B, whose next load sat beside it). Load it again
        # here, once, so that the first episode does not pay for that reload.
        held = [m.get("name") for m in ollama_ps(provider)]
        if held and config.model not in held:
            seconds = provider.preload() / 1000
            print(f"{config.model} was evicted by that load; loaded it again ... {seconds:.1f} s")
    if config.user_model != config.model:
        print("warning: two models take turns; if both do not fit in GPU memory every turn reloads one")
    started = datetime.now(UTC)
    run_id = "-".join(
        p
        for p in [
            started.strftime("%Y%m%d-%H%M%S"),
            safe_name(config.model),
            config.policy,
            config.reasoning,
            config.guard,
            config.rescue,
            config.voice if channel else "",
            config.claims if config.claims != "C0" else "",
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
        "commit": commit,
        "dirty": dirty,
        "config": config.to_dict(),
        "trials": args.trials,
        "tasks_file": str(args.tasks),
        "task_sha256": {t.id: t.sha256() for t in tasks},
        "seed_hash": db.state_hash(seed_dump),
        "prompt_sha256": {
            "policy": text_sha256(policy_text),
            "agent_system": text_sha256(
                build_system_prompt(policy_text, tasks[0].now, caller_id=config.caller_id)
            ),
            "user_sim": text_sha256(build_user_prompt(tasks[0])),
            "tools": text_sha256(tools_json),
        },
        "agent_provider": provider.describe(),
        "user_provider": user_provider.describe(),
        "voice_channel": channel.describe() if channel else None,
        "caller_id": (
            {"tool": "verify_caller", "name_tolerance": NAME_TOLERANCE} if config.caller_id else None
        ),
        "gpu_mib_used_by_others_at_start": others,
        "ollama_ps_after_preload": ollama_ps(provider),
        "versions": {"python": sys.version.split()[0]}
        | {name: version(name) for name in ("sqlalchemy", "pydantic", "httpx", "pyyaml")}
        | (speech_versions("faster-whisper", "ctranslate2", "melotts", "torch") if channel else {}),
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
    results: list[EpisodeResult] = []
    try:
        _run_all(
            tasks,
            args.trials,
            results,
            run_dir,
            config,
            provider,
            user_provider,
            registry,
            seed_engine,
            policy_text,
            gold_dumps,
            run_id,
            channel,
        )
    finally:  # also after Ctrl-C or a crash: what was measured so far stays readable
        summary = summarise(results)
        summary["ollama_ps_at_end"] = ollama_ps(provider)  # a model evicted or moved to the CPU shows here
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=1))


def _run_all(
    tasks,
    trials,
    results,
    run_dir,
    config,
    provider,
    user_provider,
    registry,
    seed_engine,
    policy_text,
    gold_dumps,
    run_id,
    channel=None,
) -> None:
    infra_streak = 0
    with (run_dir / "episodes.jsonl").open("a", encoding="utf-8", newline="\n") as out:
        for task in tasks:
            gold_dump = gold_dumps[task.id]
            for trial in range(trials):
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
                    channel=channel,
                )
                results.append(result)
                out.write(result.to_json_line() + "\n")
                out.flush()
                mark = "infra" if result.verdict is None else ("PASS" if result.verdict.success else "fail")
                print(f"  {task.id} #{trial}: {mark:5} {result.termination:22} {result.wall_seconds:6.1f} s")
                infra_streak = infra_streak + 1 if result.status == "infra_error" else 0
                if infra_streak >= MAX_CONSECUTIVE_INFRA_ERRORS:
                    sys.exit(f"{infra_streak} infra errors in a row; stopping. Last: {result.error[:300]}")


if __name__ == "__main__":
    main()
