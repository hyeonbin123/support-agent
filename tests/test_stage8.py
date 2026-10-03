"""Stage 8 (a new-generation agent model): runner options, smoke gates, the selection rule, the test
verdict and the blind simulator check. No model, no GPU: the runner gets a fake provider and analyze
reads synthetic records.
"""

from __future__ import annotations

import json
import random

import pytest

from support_agent import analyze, run
from support_agent.chat import ChatProvider, ChatResponse
from support_agent.config import RunConfig

SIM = "qwen2.5:7b-instruct"
NEW = "qwen3:4b-instruct-2507-q4_K_M"


# ---------------------------------------------------------------- the runner


class RecordingOllama(ChatProvider):
    """Remembers how it was built. The customer asks once and stops; the agent only acknowledges."""

    made: list[RecordingOllama] = []

    def __init__(self, model: str, **options):
        self.model = model
        self.options = options
        self.customer_lines = ["O-1 주문 38,900원 맞나요?", "###STOP###"]
        RecordingOllama.made.append(self)

    def preload(self) -> float:
        return 0.0

    def describe(self):
        return {"provider": "fake", "model": self.model, **self.options}

    def loaded_models(self):
        return [{"name": self.model, "size": 100, "size_vram": 100}]

    def chat(self, messages, tools=(), *, temperature=0.0, seed=None, max_tokens=1024):
        if tools:
            return ChatResponse("네, 확인해 보겠습니다. 성함을 알려 주세요.")
        return ChatResponse(self.customer_lines.pop(0))


def start(tmp_path, monkeypatch, *extra: str):
    RecordingOllama.made = []
    monkeypatch.setattr(run, "OllamaProvider", RecordingOllama)
    monkeypatch.setattr(run, "gpu_used_by_others_mib", lambda provider: None)
    monkeypatch.setattr(run, "OUTPUTS", tmp_path)
    argv = ["run", "--tasks", "smoke", "--task-id", "smoke-lookup-01", "--label", "unit", *extra]
    monkeypatch.setattr("sys.argv", argv)
    run.main()
    (run_dir,) = (tmp_path / "runs").iterdir()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return RecordingOllama.made, manifest, summary


def test_the_simulator_gets_its_own_context_size(tmp_path, monkeypatch):
    made, manifest, _ = start(
        tmp_path, monkeypatch, "--model", NEW, "--num-ctx", "12288", "--user-num-ctx", "8192"
    )
    agent, user = made
    assert (agent.model, agent.options["num_ctx"]) == (NEW, 12288)
    assert (user.model, user.options["num_ctx"], user.options["num_gpu"]) == (SIM, 8192, None)
    assert manifest["config"]["num_ctx"] == 12288 and manifest["config"]["user_num_ctx"] == 8192
    assert manifest["user_provider"]["num_ctx"] == 8192


def test_without_the_flag_the_simulator_shares_the_agent_context_and_the_provider(tmp_path, monkeypatch):
    made, manifest, _ = start(tmp_path, monkeypatch)
    assert len(made) == 1  # one model, one set of runner options: Ollama never reloads between the roles
    assert manifest["config"]["user_num_ctx"] == manifest["config"]["num_ctx"] == RunConfig.num_ctx
    assert manifest["user_provider"] == manifest["agent_provider"]


@pytest.mark.parametrize("extra", [["--user-num-ctx", "8192"], ["--user-on-cpu"]])
def test_one_model_in_both_roles_cannot_have_two_sets_of_runner_options(tmp_path, monkeypatch, extra):
    # Ollama keeps one runner per model: different num_ctx or num_gpu would reload the model every turn.
    with pytest.raises(SystemExit):
        start(tmp_path, monkeypatch, *extra)


def test_part_of_the_simulator_can_sit_on_the_cpu(tmp_path, monkeypatch):
    made, manifest, _ = start(tmp_path, monkeypatch, "--model", NEW, "--user-num-gpu", "20")
    assert made[1].options["num_gpu"] == 20 and manifest["user_provider"]["num_gpu"] == 20


def test_on_cpu_and_a_layer_count_contradict_each_other(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        start(tmp_path, monkeypatch, "--model", NEW, "--user-on-cpu", "--user-num-gpu", "20")


def test_the_stage_2_m1_command_still_builds_the_same_simulator(tmp_path, monkeypatch):
    # README: --model qwen2.5:14b-instruct --user-on-cpu --num-ctx 8192 (both roles at 8192, simulator on CPU)
    made, manifest, _ = start(
        tmp_path, monkeypatch, "--model", "qwen2.5:14b-instruct", "--user-on-cpu", "--num-ctx", "8192"
    )
    agent, user = made
    assert agent.options == {"num_ctx": 8192}
    assert user.options == {"num_ctx": 8192, "num_gpu": 0}
    assert manifest["config"]["user_num_ctx"] == 8192


def test_think_goes_to_the_agent_and_never_to_the_simulator(tmp_path, monkeypatch):
    made, manifest, _ = start(tmp_path, monkeypatch, "--model", "qwen3.5:4b", "--think", "off")
    agent, user = made
    assert agent.options["think"] is False and user.options.get("think") is None
    assert manifest["agent_provider"]["think"] is False


def test_think_with_one_model_in_both_roles_keeps_the_runner_options_equal(tmp_path, monkeypatch):
    made, _, _ = start(tmp_path, monkeypatch, "--think", "on")
    agent, user = made  # two objects (think is per request), equal runner options (no reload)
    assert agent.options["think"] is True and user.options.get("think") is None
    assert agent.options["num_ctx"] == user.options["num_ctx"]
    assert agent.options.get("num_gpu") == user.options.get("num_gpu")


def test_without_think_nothing_is_sent(tmp_path, monkeypatch):
    made, _, _ = start(tmp_path, monkeypatch, "--model", NEW)
    assert made[0].options.get("think") is None  # a model without thinking answers 400 to any think value


class EvictingOllama(RecordingOllama):
    """One server for all providers: the first simulator load evicts the agent, which then fits beside it."""

    held: list[str] = []
    loads: list[str] = []

    def preload(self) -> float:
        EvictingOllama.loads.append(self.model)
        if self.model == SIM and EvictingOllama.loads.count(SIM) == 1:
            EvictingOllama.held = []
        if self.model not in EvictingOllama.held:
            EvictingOllama.held.append(self.model)
        return 0.0

    def loaded_models(self):
        return [{"name": name, "size": 100, "size_vram": 100} for name in EvictingOllama.held]


def test_an_agent_model_evicted_by_the_simulator_load_is_loaded_again(tmp_path, monkeypatch):
    EvictingOllama.held, EvictingOllama.loads = [], []
    monkeypatch.setattr(run, "OllamaProvider", EvictingOllama)
    monkeypatch.setattr(run, "gpu_used_by_others_mib", lambda provider: None)
    monkeypatch.setattr(run, "OUTPUTS", tmp_path)
    argv = ["run", "--tasks", "smoke", "--task-id", "smoke-lookup-01", "--model", NEW]
    monkeypatch.setattr("sys.argv", argv)
    run.main()
    assert EvictingOllama.loads == [NEW, SIM, NEW]
    (run_dir,) = (tmp_path / "runs").iterdir()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert {m["name"] for m in manifest["ollama_ps_after_preload"]} == {NEW, SIM}


def test_a_model_still_held_is_not_loaded_again(tmp_path, monkeypatch):
    start(tmp_path, monkeypatch, "--model", NEW)  # the plain fake always lists the agent model
    assert [m.model for m in RecordingOllama.made] == [NEW, SIM]


def test_the_records_say_where_the_models_sat(tmp_path, monkeypatch):
    _, manifest, summary = start(tmp_path, monkeypatch, "--model", NEW, "--user-num-ctx", "16384")
    assert manifest["ollama_ps_after_preload"] == [{"name": NEW, "size": 100, "size_vram": 100}]
    assert summary["ollama_ps_at_end"] == [{"name": NEW, "size": 100, "size_vram": 100}]


# ---------------------------------------------------------------- synthetic records


def call(who="agent", *, ms=500.0, prompt_eval=30.0, load=5.0, error=None, text="", dropped=0, index=0):
    return {
        "who": who,
        "index": index,
        "text": text,
        "format_error": error,
        "dropped_calls": dropped,
        "wall_ms": ms,
        "prompt_eval_ms": prompt_eval,
        "load_ms": load,
        "prompt_tokens": 1000,
    }


def ep(task, trial, success, *, calls=None, replies=("네, 확인했습니다.",), writes=0, unexpected=0,
       violations=(), termination="user_stop", status="completed"):  # fmt: skip
    return {
        "task_id": task,
        "trial": trial,
        "status": status,
        "termination": termination,
        "verdict": None
        if status == "infra_error"
        else {
            "success": success,
            "judged": True,
            "db_match": success,
            "values": {},
            "unexpected_writes": unexpected,
            "missing_writes": 0,
            "policy_violations": list(violations),
            "policy_blocks": [],
            "auth_blocks": 0,
        },
        "llm_calls": calls if calls is not None else [call(index=0), call("user", index=0)],
        "messages": [{"role": "assistant", "content": r} for r in replies],
        "tool_calls": [{"ok": True, "write": True, "name": "cancel_order", "agent_call": 0}] * writes,
        "user_messages": [{"role": "assistant", "content": "문의드립니다."}],
    }


def provider(model, *, num_ctx=12288, num_gpu=None, think=None, version="0.35.1"):
    return {
        "provider": "ollama",
        "model": model,
        "base_url": "http://127.0.0.1:11434",
        "num_ctx": num_ctx,
        "keep_alive": "60m",
        "think": think,
        "num_gpu": num_gpu,
        "digest": f"digest-of-{model}",
        "ollama_version": version,
    }


def write_run(tmp_path, name, episodes, *, model=SIM, policy="P1", num_ctx=12288, user=None, ps=None,
              tasks_file="dev", version="0.35.1", think=None):  # fmt: skip
    run_dir = tmp_path / name
    run_dir.mkdir()
    config = RunConfig(model=model, policy=policy, num_ctx=num_ctx).to_dict() | {"user_num_ctx": num_ctx}
    agent_provider = provider(model, num_ctx=num_ctx, version=version, think=think)
    manifest = {
        "run_id": name,
        "config": config,
        "trials": 4,
        "tasks_file": tasks_file,
        "task_sha256": {e["task_id"]: f"sha-{e['task_id']}" for e in episodes},
        "seed_hash": "seed",
        "prompt_sha256": {"policy": "p", "agent_system": "a", "user_sim": "u", "tools": "t"},
        "agent_provider": agent_provider,
        "user_provider": user or (agent_provider if model == SIM else provider(SIM, num_ctx=num_ctx)),
        "gpu_mib_used_by_others_at_start": 1200,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    summary = {} if ps is None else {"ollama_ps_at_end": ps}
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    lines = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in episodes)
    (run_dir / "episodes.jsonl").write_text(lines, encoding="utf-8")
    return run_dir


ON_GPU = [{"name": NEW, "size": 4_000, "size_vram": 4_000}, {"name": SIM, "size": 5_000, "size_vram": 5_000}]


def smoke_episodes(**overrides):
    calls = [call(index=0, prompt_eval=900.0), call(index=1), call(index=2), call("user", index=0)]
    return [ep(f"smoke-{i}", 0, i % 2 == 0, calls=overrides.get("calls", calls)) for i in range(5)]


def gates(run_dir):
    return {name: ok for name, _detail, ok in analyze.smoke_gates(run_dir)}


# ---------------------------------------------------------------- smoke gates


def test_a_clean_smoke_run_passes_every_gate(tmp_path):
    run_dir = write_run(tmp_path, "smoke", smoke_episodes(), model=NEW, ps=ON_GPU)
    assert gates(run_dir) == {"S0": True, "S1": True, "S2": True, "S3": True, "S4": True, "S5": True}
    assert "GPU" in analyze.smoke_report(run_dir)


def test_a_model_partly_on_the_cpu_fails_the_fit_gate(tmp_path):
    ps = [ON_GPU[0], {"name": SIM, "size": 5_000, "size_vram": 3_000}]
    assert gates(write_run(tmp_path, "smoke", smoke_episodes(), model=NEW, ps=ps))["S1"] is False


def test_a_simulator_that_was_evicted_fails_the_fit_gate(tmp_path):
    assert gates(write_run(tmp_path, "smoke", smoke_episodes(), model=NEW, ps=ON_GPU[:1]))["S1"] is False


def test_without_a_ps_record_the_fit_gate_fails(tmp_path):
    assert gates(write_run(tmp_path, "smoke", smoke_episodes(), model=NEW))["S1"] is False


def test_a_reload_in_the_middle_fails_the_fit_gate(tmp_path):
    calls = [call(index=0), call(index=1, load=4_000.0), call("user", index=0)]
    run_dir = write_run(tmp_path, "smoke", smoke_episodes(calls=calls), model=NEW, ps=ON_GPU)
    assert gates(run_dir)["S1"] is False


def test_a_simulator_on_the_cpu_on_purpose_passes_the_fit_gate_and_says_so(tmp_path):
    ps = [ON_GPU[0], {"name": SIM, "size": 5_000, "size_vram": 0}]
    user = provider(SIM, num_gpu=0)
    run_dir = write_run(tmp_path, "smoke", smoke_episodes(), model=NEW, ps=ps, user=user)
    assert gates(run_dir)["S1"] is True
    assert "CPU" in analyze.smoke_report(run_dir)


def test_slow_agent_calls_fail_the_latency_gate(tmp_path):
    calls = [call(index=0, ms=2_000.0), call(index=1, ms=1_900.0), call("user", index=0)]
    run_dir = write_run(tmp_path, "smoke", smoke_episodes(calls=calls), model=NEW, ps=ON_GPU)
    assert gates(run_dir)["S2"] is False


def test_a_prompt_cache_that_is_not_reused_fails_its_gate(tmp_path):
    calls = [call(index=i, prompt_eval=800.0) for i in range(3)] + [call("user", index=0)]
    run_dir = write_run(tmp_path, "smoke", smoke_episodes(calls=calls), model=NEW, ps=ON_GPU)
    assert gates(run_dir)["S3"] is False


def test_format_errors_and_leaked_thinking_count_against_the_format_gate(tmp_path):
    calls = [
        call(index=0, error="leaked_tool_call"),
        call(index=1, text="<think>음</think> 네"),
        call(index=2),
    ]
    run_dir = write_run(tmp_path, "smoke", smoke_episodes(calls=calls), model=NEW, ps=ON_GPU)
    assert gates(run_dir)["S4"] is False  # 2 per episode
    held = [call(index=0, error="stall"), call(index=1, error="unbacked_claim"), call(index=2)]
    run_dir = write_run(tmp_path, "held", smoke_episodes(calls=held), model=NEW, ps=ON_GPU)
    assert gates(run_dir)["S4"] is True  # replies held back by a guard are well formed


def test_replies_in_another_script_fail_the_language_gate(tmp_path):
    episodes = smoke_episodes()
    episodes[0]["messages"] = [{"role": "assistant", "content": "您好，订单已经取消了。"}] * 3
    assert gates(write_run(tmp_path, "smoke", episodes, model=NEW, ps=ON_GPU))["S5"] is False


def test_replies_that_were_held_back_or_called_a_tool_are_not_read_for_language(tmp_path):
    episodes = smoke_episodes()
    episodes[0]["messages"] = [
        {"role": "assistant", "content": "您好，订单已经取消了。", "delivered": False},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "get_order", "arguments": {}}]},
    ] * 3
    assert gates(write_run(tmp_path, "smoke", episodes, model=NEW, ps=ON_GPU))["S5"] is True


def test_an_infra_error_fails_the_first_gate(tmp_path):
    episodes = smoke_episodes()
    episodes[1] = ep("smoke-1", 0, False, status="infra_error", termination="infra_error")
    assert gates(write_run(tmp_path, "smoke", episodes, model=NEW, ps=ON_GPU))["S0"] is False


# ---------------------------------------------------------------- the selection rule


def dev_run(tmp_path, name, successes, *, model=SIM, policy="P1", unexpected=0, violations=(), ms=500.0,
            **kwargs):  # fmt: skip
    """24 tasks x 4 trials; `successes` successful trials spread over the first tasks."""
    episodes = []
    left = successes
    for t in range(24):
        for trial in range(4):
            ok = left > 0
            left -= ok
            episodes.append(ep(f"dev-{t + 1:03d}", trial, ok, calls=[call(index=0, ms=ms), call("user")]))
    episodes[0]["verdict"]["unexpected_writes"] = unexpected
    episodes[0]["verdict"]["policy_violations"] = list(violations)
    return write_run(tmp_path, name, episodes, model=model, policy=policy, **kwargs)


def rows(text):
    return {line.split(" | ")[0].strip("| "): line for line in text.splitlines() if line.startswith("| ")}


def test_a_candidate_that_meets_all_three_conditions_is_selected(tmp_path):
    base = dev_run(tmp_path, "base", 15)
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384, unexpected=11, violations=["x"] * 9)
    cand = dev_run(tmp_path, "m2a", 30, model=NEW, unexpected=11, ms=1_400.0)
    text = analyze.select(base, p0, [cand])
    assert "선택: m2a" in text and "시험용" in text


def test_a_gain_below_five_points_is_not_selected(tmp_path):
    base = dev_run(tmp_path, "base", 15)
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384, unexpected=11)
    cand = dev_run(tmp_path, "m2a", 19, model=NEW)  # +4 successes of 96 = +4.2%p
    assert "선택: 없음" in analyze.select(base, p0, [cand])


def test_more_unexpected_writes_than_the_p0_reference_are_not_selected(tmp_path):
    base = dev_run(tmp_path, "base", 15)
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384, unexpected=11)
    cand = dev_run(tmp_path, "m2a", 40, model=NEW, unexpected=12)
    assert "선택: 없음" in analyze.select(base, p0, [cand])


def test_a_violation_let_through_is_never_selected(tmp_path):
    base = dev_run(tmp_path, "base", 15)
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384, unexpected=11)
    cand = dev_run(tmp_path, "m2a", 40, model=NEW, unexpected=1, violations=["return_window_expired"])
    assert "선택: 없음" in analyze.select(base, p0, [cand])


def test_calls_more_than_three_times_slower_are_not_selected(tmp_path):
    base = dev_run(tmp_path, "base", 15, ms=500.0)
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384, unexpected=11)
    cand = dev_run(tmp_path, "m2a", 40, model=NEW, ms=1_501.0)
    assert "선택: 없음" in analyze.select(base, p0, [cand])


def test_of_two_qualified_candidates_the_higher_wins_and_a_tie_goes_to_the_first(tmp_path):
    base = dev_run(tmp_path, "base", 15)
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384, unexpected=11)
    m2a = dev_run(tmp_path, "m2a", 30, model=NEW)
    m2b = dev_run(tmp_path, "m2b", 35, model="qwen3.5:4b", think=False)
    assert "선택: m2b" in analyze.select(base, p0, [m2a, m2b])
    m2b_tie = dev_run(tmp_path, "m2b-tie", 30, model="qwen3.5:4b", think=False)
    assert "선택: m2a" in analyze.select(base, p0, [m2a, m2b_tie])


def test_a_candidate_on_another_agent_context_cannot_be_paired(tmp_path):
    base = dev_run(tmp_path, "base", 15)
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384, unexpected=11)
    cand = dev_run(tmp_path, "m2a", 40, model=NEW, num_ctx=16384)
    text = analyze.select(base, p0, [cand])
    assert "선택: 없음" in text and "num_ctx" in text


def test_a_candidate_with_its_simulator_on_the_cpu_stops_at_the_development_tasks(tmp_path):
    base = dev_run(tmp_path, "base", 15)
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384, unexpected=11)
    cand = dev_run(tmp_path, "m2a", 40, model=NEW, user=provider(SIM, num_gpu=0))
    text = analyze.select(base, p0, [cand])
    assert "선택: m2a" in text and "시험용은 재지 않는다" in text


def test_the_references_must_be_what_the_rule_names(tmp_path):
    base = dev_run(tmp_path, "base", 15, policy="P0")
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384, unexpected=11)
    cand = dev_run(tmp_path, "m2a", 40, model=NEW)
    with pytest.raises(ValueError, match="P1"):
        analyze.select(base, p0, [cand])
    old = dev_run(tmp_path, "p0-old", 19, policy="P0", num_ctx=16384, unexpected=11, version="0.34.2")
    with pytest.raises(ValueError, match="Ollama"):
        analyze.select(dev_run(tmp_path, "base2", 15), old, [cand])


def test_the_selection_table_shows_the_report_only_numbers(tmp_path):
    base = dev_run(tmp_path, "base", 15)
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384, unexpected=11)
    cand = dev_run(tmp_path, "m2a", 30, model=NEW, unexpected=3, ms=1_000.0)
    line = rows(analyze.select(base, p0, [cand]))["m2a"]
    assert "+15.6%p" in line and "2.00배" in line and "3 / 11" in line


# ---------------------------------------------------------------- the test verdict


def make_test_run(tmp_path, name, successes_per_task, **kwargs):
    episodes = [
        ep(f"test-{t + 1:03d}", trial, trial < successes_per_task[t])
        for t in range(len(successes_per_task))
        for trial in range(4)
    ]
    unexpected = kwargs.pop("unexpected", 0)
    episodes[0]["verdict"]["unexpected_writes"] = unexpected
    return write_run(tmp_path, name, episodes, tasks_file="test", **kwargs)


def test_the_verdict_follows_the_stage_2_table_with_the_p0_write_reference(tmp_path):
    base = make_test_run(tmp_path, "base", [1] * 10 + [0] * 30)
    p0 = make_test_run(tmp_path, "p0", [1] * 10 + [0] * 30, policy="P0", num_ctx=16384, unexpected=19)
    better = make_test_run(tmp_path, "m2a", [3] * 30 + [0] * 10, model=NEW, unexpected=19)
    assert analyze.verdict(base, p0, better).splitlines()[-1] == "판정: 개선"
    risky = make_test_run(tmp_path, "m2a-risky", [3] * 30 + [0] * 10, model=NEW, unexpected=20)
    assert analyze.verdict(base, p0, risky).splitlines()[-1] == "판정: 성공률은 올랐지만 위험한 쓰기가 늘었음"
    same = make_test_run(tmp_path, "m2a-same", [1] * 10 + [0] * 30, model=NEW)
    assert analyze.verdict(base, p0, same).splitlines()[-1] == "판정: 개선을 확인하지 못함"


def test_the_verdict_reads_only_test_runs(tmp_path):
    base = dev_run(tmp_path, "base", 15)
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384)
    cand = dev_run(tmp_path, "m2a", 40, model=NEW)
    with pytest.raises(ValueError, match="test"):
        analyze.verdict(base, p0, cand)


# ---------------------------------------------------------------- the blind simulator check (P5(a))


def blind_task_ids():
    from support_agent.tasks import load_tasks

    return [t.id for t in load_tasks("dev")][:6]


def talk_episodes(text):
    out = []
    for task in blind_task_ids():
        for trial in range(4):
            e = ep(task, trial, trial == 0, replies=(text,))
            e["messages"] = [
                {"role": "user", "content": "주문 취소하고 싶어요."},
                {"role": "assistant", "content": text},
            ]
            out.append(e)
    return out


def two_dev_runs(tmp_path):
    a = write_run(tmp_path, "run-a", talk_episodes("기준 상담원의 말"))
    b = write_run(tmp_path, "run-b", talk_episodes("후보 상담원의 말"), model=NEW)
    return a, b


def test_the_blind_sheet_hides_which_run_and_how_it_ended(tmp_path):
    a, b = two_dev_runs(tmp_path)
    out = tmp_path / "blind"
    analyze.blind(a, b, n=5, out_dir=out)
    sheet = (out / "sheet.md").read_text(encoding="utf-8")
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    assert len(key) == 10 and sheet.count("\n## B") == 10
    for hidden in ("run-a", "run-b", "PASS", "fail", "user_stop", NEW, SIM):
        assert hidden not in sheet
    assert "알고 있는 것" in sheet  # the scenario is shown to the reader
    assert {entry["run"] for entry in key.values()} == {"run-a", "run-b"}
    # the same draw every time, and the same (task, trial) pairs from both runs
    analyze.blind(a, b, n=5, out_dir=tmp_path / "again")
    assert (tmp_path / "again" / "sheet.md").read_text(encoding="utf-8") == sheet
    pairs = {}
    for entry in key.values():
        pairs.setdefault(entry["run"], set()).add((entry["task_id"], entry["trial"]))
    assert pairs["run-a"] == pairs["run-b"]


def test_two_runs_keep_the_layout_of_the_recorded_sheets(tmp_path):
    # the stage 8 sheets were made from two runs each; the same command must give the same sheet again
    a, b = two_dev_runs(tmp_path)
    out = tmp_path / "blind"
    analyze.blind(a, b, n=5, out_dir=out)
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    common = sorted((task, trial) for task in blind_task_ids() for trial in range(4))
    chosen = random.Random(analyze.BOOTSTRAP_SEED).sample(common, 5)
    items = [(name, pair) for name in ("run-a", "run-b") for pair in chosen]
    random.Random(analyze.BOOTSTRAP_SEED + 1).shuffle(items)
    assert [(entry["run"], (entry["task_id"], entry["trial"])) for entry in key.values()] == items


def test_every_run_of_one_check_goes_into_one_sheet(tmp_path):
    # two sheets that share a run have the same layout (fixed seeds), so its episodes give the other run
    # away; with every run in one sheet nothing is left to eliminate
    a, b = two_dev_runs(tmp_path)
    c = write_run(tmp_path, "run-c", talk_episodes("셋째 상담원의 말"), model="qwen3.5:4b")
    out = tmp_path / "three"
    analyze.blind(a, b, c, n=5, out_dir=out)
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    sheet = (out / "sheet.md").read_text(encoding="utf-8")
    assert len(key) == 15 and sheet.count("\n## B") == 15
    for hidden in ("run-a", "run-b", "run-c", "qwen3.5"):
        assert hidden not in sheet
    pairs = {}
    for entry in key.values():
        pairs.setdefault(entry["run"], set()).add((entry["task_id"], entry["trial"]))
    assert set(pairs) == {"run-a", "run-b", "run-c"}
    assert pairs["run-a"] == pairs["run-b"] == pairs["run-c"] and len(pairs["run-a"]) == 5


def test_the_blind_check_needs_two_different_runs(tmp_path):
    a, _ = two_dev_runs(tmp_path)
    with pytest.raises(ValueError, match="two or more"):
        analyze.blind(a, n=1, out_dir=tmp_path / "one")
    with pytest.raises(ValueError, match="same run twice"):
        analyze.blind(a, a, n=1, out_dir=tmp_path / "twice")


def test_unblinding_counts_the_marks_per_run(tmp_path):
    a, b = two_dev_runs(tmp_path)
    out = tmp_path / "blind"
    analyze.blind(a, b, n=2, out_dir=out)
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    marks = {
        label: (["invent", "outcome"] if entry["run"] == "run-b" else []) for label, entry in key.items()
    }
    (out / "marks.json").write_text(json.dumps(marks), encoding="utf-8")
    text = analyze.unblind(out)
    assert "| run-a | 2 | 0 |" in text and "| run-b | 2 | 2 |" in text


def test_unblinding_refuses_unknown_codes_and_missing_labels(tmp_path):
    a, b = two_dev_runs(tmp_path)
    out = tmp_path / "blind"
    analyze.blind(a, b, n=2, out_dir=out)
    key = json.loads((out / "key.json").read_text(encoding="utf-8"))
    (out / "marks.json").write_text(json.dumps(dict.fromkeys(key, ["made_up"])), encoding="utf-8")
    with pytest.raises(ValueError, match="made_up"):
        analyze.unblind(out)
    (out / "marks.json").write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValueError, match="B01"):
        analyze.unblind(out)


def test_the_blind_check_never_reads_test_records(tmp_path):
    a = write_run(tmp_path, "t-a", [ep("test-001", 0, True)], tasks_file="test")
    b = write_run(tmp_path, "t-b", [ep("test-001", 0, True)], tasks_file="test", model=NEW)
    with pytest.raises(ValueError, match="test"):
        analyze.blind(a, b, n=1, out_dir=tmp_path / "blind")
