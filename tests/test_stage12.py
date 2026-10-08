"""Stage 12 (simulator sensitivity): --user-think, the automatic simulator counts (P5(b)) and the report that
pairs a run with another simulator against the stage 8 M2b record.

No model, no GPU: the runner gets a fake provider, the provider a mock server, the counts read hand-made
records and records written by scripted episodes.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from test_stage8 import NEW, SIM, RecordingOllama
from test_tasks import CONTACTS, say_values

from support_agent import analyze, run, sim_report
from support_agent.agent import load_policy
from support_agent.chat import ChatResponse, ScriptedProvider, ToolCall
from support_agent.config import FIRST_AGENT_MESSAGE, RunConfig
from support_agent.episode import run_episode
from support_agent.ollama import OllamaProvider
from support_agent.paths import REPORTS, TASKS
from support_agent.seed import build_seed_engine
from support_agent.tasks import load_tasks
from support_agent.tools import build_registry
from support_agent.user_sim import LLMUser

AGENT = "qwen3.5:4b"
U1 = NEW  # qwen3:4b-instruct-2507-q4_K_M
DEV = {task.id: task for task in load_tasks(TASKS / "dev.yaml")}
REGISTRY = build_registry()
# The record U1 is paired with (docs/experiments.md, stage 12): stage 8 M2b, 7B simulator.
M2B_DEV = REPORTS / "20261003-200306-qwen3.5-4b-P1-R0-G0-F0-m2b"


# ==================================================================== the runner: --user-think


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
    return RecordingOllama.made, manifest


U1_COMMAND = [
    "--policy", "P1", "--model", AGENT, "--think", "off", "--num-ctx", "12288",
    "--user-model", U1, "--user-think", "off", "--user-num-ctx", "8192",
]  # fmt: skip


def test_user_think_goes_to_the_simulator_and_the_manifest(tmp_path, monkeypatch):
    made, manifest = start(tmp_path, monkeypatch, *U1_COMMAND)
    agent, user = made
    assert (agent.model, agent.options["think"], agent.options["num_ctx"]) == (AGENT, False, 12288)
    assert (user.model, user.options["think"], user.options["num_ctx"]) == (U1, False, 8192)
    assert user.options["num_gpu"] is None
    assert manifest["user_provider"]["think"] is False and manifest["agent_provider"]["think"] is False
    assert manifest["config"]["user_model"] == U1 and manifest["config"]["user_num_ctx"] == 8192


def test_without_user_think_nothing_is_sent_to_the_simulator(tmp_path, monkeypatch):
    # every earlier command: the 7B simulator never got a think value (it has no thinking)
    made, manifest = start(tmp_path, monkeypatch, "--model", AGENT, "--think", "off")
    assert "think" not in made[1].options
    assert manifest["user_provider"].get("think") is None


def test_user_think_with_one_model_in_both_roles_keeps_the_runner_options_equal(tmp_path, monkeypatch):
    made, _ = start(tmp_path, monkeypatch, "--user-think", "off")
    agent, user = made  # two objects (think is per request), equal runner options (no reload)
    assert user.options["think"] is False and "think" not in agent.options
    assert agent.options["num_ctx"] == user.options["num_ctx"]
    assert agent.options.get("num_gpu") == user.options.get("num_gpu")


def test_the_simulator_sends_think_false_on_the_wire():
    """The simulator's request body carries think: false at the top level, as the agent's does."""
    bodies = []

    def server(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        message = {"role": "assistant", "content": "환불 금액이 궁금해요."}
        return httpx.Response(200, json={"message": message, "done": True, "done_reason": "stop"})

    provider = OllamaProvider(U1, num_ctx=8192, think=False, transport=httpx.MockTransport(server))
    user = LLMUser(provider, DEV["dev-001"], RunConfig(), trial=0)
    assert user.reply(FIRST_AGENT_MESSAGE) == "환불 금액이 궁금해요."
    (body,) = bodies
    assert body["think"] is False and "think" not in body["options"]
    assert body["options"]["num_ctx"] == 8192 and body["options"]["temperature"] == 0.3
    assert provider.describe()["think"] is False


# ==================================================================== values in what the simulator says


def test_value_tokens_read_numbers_emails_and_dates_in_one_form():
    tokens = sim_report.value_tokens(
        "전화번호는 010-0000-9101, 주문 O-91001, 금액 58,300원, 메일 Sunghoon.Bae@Example.com, 9월 15일 도착"
    )
    assert tokens == {
        "num:01000009101",
        "num:91001",
        "num:58300",
        "email:sunghoon.bae@example.com",
        "date:09-15",
    }


def test_short_numbers_are_not_values():
    # quantities, days, sizes and times: too often legitimate to count as invented
    assert sim_report.value_tokens("2개 주문했고 3일 전에 10시쯤 받았어요. 260 사이즈요.") == set()


def test_a_date_written_with_digits_is_one_date_not_a_number():
    assert sim_report.value_tokens("2026-09-15에 주문했어요") == {"date:09-15"}
    assert sim_report.value_tokens("2026년 9월 15일") == {"date:09-15"}


def test_the_separators_inside_a_number_do_not_matter():
    assert sim_report.value_tokens("010 0000 9101") == {"num:0000", "num:9101"}
    assert sim_report.value_tokens("01000009101") == sim_report.value_tokens("010-0000-9101")
    assert sim_report.value_tokens("１２，３４５원") == {"num:12345"}  # full-width digits (NFKC)


def test_a_number_is_grounded_by_a_longer_source_number_that_contains_it():
    sources = sim_report.source_tokens(["가입한 전화번호는 010-0000-9101이다."])
    assert sim_report.grounded("num:9101", sources)  # the last digits of the phone number
    assert sim_report.grounded("num:01000009101", sources)
    assert not sim_report.grounded("num:01000009102", sources)
    assert not sim_report.grounded("num:010000091011", sources)  # longer than any source


def test_sources_keep_the_year_and_the_digits_of_a_date():
    sources = sim_report.source_tokens(["오늘 날짜: 2026년 9월 14일"])
    assert sim_report.grounded("num:2026", sources)
    assert sim_report.grounded("date:09-14", sources)
    assert not sim_report.grounded("date:09-15", sources)


def test_emails_and_dates_are_grounded_only_by_the_same_value():
    sources = sim_report.source_tokens(["메일은 jaehee.ahn@example.com, 2026-09-09 주문"])
    assert sim_report.grounded("email:jaehee.ahn@example.com", sources)
    assert not sim_report.grounded("email:jaehee@example.com", sources)
    assert sim_report.grounded("date:09-09", sources)


# ==================================================================== counts on hand-made records


def call(who, index, text, *, finish="stop", tools=(), prompt_tokens=1000):
    return {
        "who": who,
        "index": index,
        "seed": 1,
        "text": text,
        "tool_calls": list(tools),
        "format_error": None,
        "dropped_text": "",
        "dropped_calls": 0,
        "finish_reason": finish,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 10,
        "wall_ms": 300.0,
        "load_ms": 0.0,
        "prompt_eval_ms": 30.0,
        "eval_ms": 200.0,
    }


def episode(task_id, turns, *, trial=0, success=False, termination="user_stop", status="completed",
            missing_writes=0, values=None, heard=None):  # fmt: skip
    """turns: (agent text the simulator saw, simulator reply, [tool results the agent got after that reply]).
    The agent-side history is rebuilt from them; `heard` replaces what the agent heard (voice runs)."""
    task = DEV[task_id]
    messages, user_messages, sim_calls = [], [], []
    for i, (agent_text, reply, tools) in enumerate(turns):
        messages.append({"role": "assistant", "content": agent_text})
        user_messages += [{"role": "user", "content": agent_text}, {"role": "assistant", "content": reply}]
        sim_calls.append(call("user", i, reply))
        said = reply.replace("###STOP###", "").strip()
        if said:
            messages.append({"role": "user", "content": (heard or {}).get(i, said)})
        for content in tools:
            messages.append({"role": "assistant", "content": "", "tool_calls": [{"name": "get_order"}]})
            messages.append({"role": "tool", "content": content, "tool_name": "get_order"})
    return {
        "run_id": "unit",
        "task_id": task_id,
        "task_sha256": task.sha256(),
        "trial": trial,
        "status": status,
        "termination": termination,
        "verdict": None
        if status == "infra_error"
        else {
            "success": success,
            "judged": True,
            "db_match": success,
            "values": values if values is not None else {v.label: success for v in task.required_values},
            "unexpected_writes": 0,
            "missing_writes": missing_writes,
            "policy_violations": [],
            "policy_blocks": [],
            "auth_blocks": 0,
        },
        "messages": messages,
        "user_messages": user_messages,
        "tool_calls": [],
        "llm_calls": [call("agent", 0, "")] + sim_calls,
        "wall_seconds": 10.0,
    }


def test_known_values_and_what_the_agent_said_before_are_not_invented():
    e = episode(
        "dev-001",
        [
            (FIRST_AGENT_MESSAGE, "고은채이고 010-0000-9101이에요. 주문 번호는 O-91001이요.", []),
            ("환불 예정 금액은 58,300원입니다.", "58,300원 맞죠? 감사합니다. ###STOP###", []),
        ],
    )
    assert sim_report.invented_values(e, DEV["dev-001"]) == []


def test_a_value_from_nowhere_is_invented_and_counted_once_per_utterance():
    e = episode(
        "dev-001",
        [
            (FIRST_AGENT_MESSAGE, "주문 번호는 O-91999이고 금액은 61,000원이었어요.", []),
            ("확인해 보겠습니다.", "###STOP###", []),
        ],
    )
    assert sim_report.invented_values(e, DEV["dev-001"]) == [(0, ["num:61000", "num:91999"])]


def test_what_the_agent_says_later_does_not_ground_an_earlier_utterance():
    e = episode(
        "dev-001",
        [
            (FIRST_AGENT_MESSAGE, "금액이 58,300원 맞나요?", []),
            ("네, 환불 금액은 58,300원입니다.", "###STOP###", []),
        ],
    )
    assert sim_report.invented_values(e, DEV["dev-001"]) == [(0, ["num:58300"])]


def test_a_tool_result_before_the_utterance_grounds_a_value():
    order = json.dumps({"order_id": "O-91001", "ordered_at": "2026-09-11T10:20:00+09:00", "total_won": 58300})
    e = episode(
        "dev-001",
        [
            (FIRST_AGENT_MESSAGE, "환불 금액이 궁금해요.", [order]),
            ("확인했습니다.", "9월 11일에 주문했고 58,300원이죠?", []),
            ("네.", "###STOP###", []),
        ],
    )
    assert sim_report.invented_values(e, DEV["dev-001"]) == []


def test_the_simulator_prompt_grounds_its_own_fields_and_today():
    # dev-001 reason says "이틀 전", known gives the phone; today is 2026-09-14 in the simulator's prompt
    e = episode("dev-001", [(FIRST_AGENT_MESSAGE, "오늘이 9월 14일인데 2026년 주문이요. 9101 번호로요.", [])])
    assert sim_report.invented_values(e, DEV["dev-001"]) == []


def test_a_voice_run_is_read_by_what_the_simulator_said_not_what_was_heard():
    e = episode(
        "dev-001",
        [(FIRST_AGENT_MESSAGE, "주문 번호는 O-91001이에요.", []), ("네.", "###STOP###", [])],
        heard={0: "주문 번호는 O-91081이에요."},
    )
    assert sim_report.invented_values(e, DEV["dev-001"]) == []


def test_stop_before_values_counts_a_stop_the_value_never_preceded():
    early = episode(
        "dev-001",
        [(FIRST_AGENT_MESSAGE, "환불 금액이요.", []), ("본인 확인이 필요합니다.", "###STOP###", [])],
    )
    heard = episode(
        "dev-001", [(FIRST_AGENT_MESSAGE, "환불 금액이요.", []), ("58,300원입니다.", "###STOP###", [])]
    )
    assert sim_report.stop_before_values(early, DEV["dev-001"]) is True
    assert sim_report.stop_before_values(heard, DEV["dev-001"]) is False


def test_a_value_said_after_the_stop_does_not_count_as_heard():
    # "알겠습니다 ###STOP###" still gets an answer from the agent, but the simulator never read it
    e = episode("dev-001", [(FIRST_AGENT_MESSAGE, "알겠습니다. ###STOP###", [])], values={"환불 금액": True})
    e["messages"].append({"role": "assistant", "content": "환불 금액은 58,300원입니다."})
    assert sim_report.stop_before_values(e, DEV["dev-001"]) is True


def test_other_endings_and_tasks_without_values_are_not_stop_before_values():
    turns = [(FIRST_AGENT_MESSAGE, "환불 금액이요.", [])]
    assert (
        sim_report.stop_before_values(episode("dev-001", turns, termination="max_user_turns"), DEV["dev-001"])
        is False
    )
    no_values = next(t for t in DEV.values() if not t.required_values)
    assert sim_report.stop_before_values(episode(no_values.id, turns), no_values) is None


def test_a_stop_with_a_gold_write_left_is_counted_for_write_tasks_only():
    turns = [(FIRST_AGENT_MESSAGE, "취소하고 싶어요. ###STOP###", [])]
    assert (
        sim_report.stop_with_writes_left(episode("dev-007", turns, missing_writes=1), DEV["dev-007"]) is True
    )
    assert sim_report.stop_with_writes_left(episode("dev-007", turns), DEV["dev-007"]) is False
    assert sim_report.stop_with_writes_left(episode("dev-001", turns), DEV["dev-001"]) is None


@pytest.mark.parametrize(
    ("raw", "finish", "tools", "expected"),
    [
        ("네, 맞아요.", "stop", (), []),
        ("네, 맞아요. ###STOP###", "stop", (), []),
        ("<think>고객으로서</think>네.", "stop", (), ["think_tag"]),
        ("네 감사합니다 ###", "stop", (), ["broken_end"]),
        ("네 STOP", "stop", (), ["broken_end"]),
        ("길게 말하다가", "length", (), ["cut_off"]),
        ("", "stop", ({"name": "find_customer"},), ["tool_call"]),
    ],
)
def test_simulator_format_problems(raw, finish, tools, expected):
    assert sim_report.format_problems(call("user", 0, raw, finish=finish, tools=tools)) == expected


def write_run(tmp_path, name, episodes, *, user_model=SIM, user_num_ctx=12288, user_think=None,
              tasks_file="dev", agent_model=AGENT, ps=None):  # fmt: skip
    run_dir = tmp_path / name
    run_dir.mkdir()

    def provider(model, num_ctx, think):
        return {"provider": "ollama", "model": model, "base_url": "x", "num_ctx": num_ctx,
                "keep_alive": "60m", "think": think, "num_gpu": None, "digest": f"d-{model}",
                "ollama_version": "0.35.1"}  # fmt: skip

    config = RunConfig(model=agent_model, user_model=user_model, policy="P1", num_ctx=12288).to_dict()
    manifest = {
        "run_id": name,
        "config": config | {"user_num_ctx": user_num_ctx},
        "trials": 4,
        "tasks_file": tasks_file,
        "task_sha256": {e["task_id"]: e["task_sha256"] for e in episodes},
        "seed_hash": "seed",
        "prompt_sha256": {"policy": "p", "agent_system": "a", "user_sim": "u", "tools": "t"},
        "agent_provider": provider(agent_model, 12288, False),
        "user_provider": provider(user_model, user_num_ctx, user_think),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    summary = {} if ps is None else {"ollama_ps_at_end": ps}
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    lines = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in episodes)
    (run_dir / "episodes.jsonl").write_text(lines, encoding="utf-8")
    return run_dir


GOOD = [
    (FIRST_AGENT_MESSAGE, "환불 금액이 궁금해요. 고은채입니다.", []),
    ("58,300원입니다.", "###STOP###", []),
]
EARLY = [
    (FIRST_AGENT_MESSAGE, "환불 금액이요. 주문은 O-91999예요.", []),
    ("확인하겠습니다.", "###STOP###", []),
]


# The simulator keeps asking until the turn limit: it never heard the values and did not stop.
ASKING = [
    (FIRST_AGENT_MESSAGE, "환불 금액이 궁금해요.", []),
    ("확인 중입니다.", "환불 금액을 알려 주세요.", []),
]
ASKING_INVENTED = [
    (FIRST_AGENT_MESSAGE, "환불 금액이 궁금해요. 주문은 O-91999예요.", []),
    ("확인 중입니다.", "환불 금액을 알려 주세요.", []),
]


def heard(n):
    """Turns in which the agent says every required value of the n-th dev task before the STOP."""
    task = DEV[sorted(DEV)[n]]
    return [(FIRST_AGENT_MESSAGE, "문의드립니다.", []), (say_values(task), "###STOP###", [])]


def runs(tmp_path, base_pattern, cand_pattern, **cand):
    """Two runs over the same 24 dev tasks x 4 trials; a pattern says per task index
    (success, turns) or (success, turns, termination)."""
    tasks = sorted(DEV)

    def make(pattern):
        out = []
        for n, task_id in enumerate(tasks):
            success, turns, *ending = pattern(n)
            termination = ending[0] if ending else "user_stop"
            for trial in range(4):
                out.append(episode(task_id, turns, trial=trial, success=success, termination=termination))
        return out

    base = write_run(tmp_path, "u0", make(base_pattern))
    other = write_run(
        tmp_path, "u1", make(cand_pattern), user_model=U1, user_num_ctx=8192, user_think=False, **cand
    )
    return base, other


def test_the_counts_of_a_run(tmp_path):
    run_dir = write_run(
        tmp_path,
        "u0",
        [
            episode("dev-001", GOOD),
            episode("dev-001", EARLY, trial=1),
            episode("dev-007", [(FIRST_AGENT_MESSAGE, "취소요. ###STOP###", [])], missing_writes=1),
            episode("dev-001", GOOD, trial=2, status="infra_error"),
        ],
    )
    counts = sim_report.run_counts(run_dir)
    assert counts["episodes"] == 3 and counts["infra"] == 1 and counts["changed_tasks"] == 0
    assert (counts["stop_before_values"], counts["value_tasks"]) == (2, 3)  # dev-007 has a value too
    assert counts["values_never_heard"] == 2  # EARLY and dev-007; every one of them ended with a STOP
    assert (counts["invented_utterances"], counts["invented_episodes"], counts["invented_values"]) == (
        1,
        1,
        1,
    )
    assert (counts["stop_with_writes_left"], counts["write_tasks"]) == (1, 1)
    assert counts["utterances"] == 5 and counts["other_language"] == 0 and counts["format"] == {}


def test_a_record_of_a_changed_task_is_left_out_and_said_so(tmp_path):
    stale = episode("dev-001", GOOD)
    stale["task_sha256"] = "an-older-version"
    counts = sim_report.run_counts(write_run(tmp_path, "old", [stale, episode("dev-001", GOOD, trial=1)]))
    assert counts["changed_tasks"] == 1 and counts["episodes"] == 1


def test_the_counts_table_lists_runs_and_skips_folders_without_episodes(tmp_path):
    run_dir = write_run(tmp_path, "u0", [episode("dev-001", EARLY)])
    (tmp_path / "seeds").mkdir()
    text = sim_report.counts_table([run_dir, tmp_path / "seeds"])
    assert "| u0 | qwen2.5:7b-instruct | 1 | 1/1 (100.0%) |" in text
    assert "seeds: 에피소드 기록 없음" in text


def test_the_sensitivity_report_pairs_the_runs_and_reports_the_difference(tmp_path):
    base, other = runs(tmp_path, lambda n: (n % 2 == 0, GOOD), lambda n: (n % 4 == 0, GOOD))
    report, outcome = sim_report.check(base, other)
    assert outcome["difference"] == pytest.approx(-0.25)
    assert "−25.0%p" in report.replace("-", "−") and "qwen3:4b-instruct-2507-q4_K_M" in report
    assert outcome["large"] is True and outcome["counts_lower"] is False
    assert outcome["next_round"] is False and "다음 라운드 후보 없음" in report


def test_a_large_difference_with_lower_counts_on_u1_names_a_next_round_candidate(tmp_path):
    base, other = runs(tmp_path, lambda n: (n % 4 == 0, EARLY), lambda n: (True, ASKING, "max_user_turns"))
    report, outcome = sim_report.check(base, other)
    assert outcome["large"] and outcome["counts_lower"] and outcome["next_round"]
    assert "다음 라운드 후보: " in report


def test_lower_counts_without_a_large_difference_name_no_candidate(tmp_path):
    base, other = runs(
        tmp_path, lambda n: (n % 2 == 0, EARLY), lambda n: (n % 2 == 0, ASKING, "max_user_turns")
    )
    _, outcome = sim_report.check(base, other)
    assert outcome["difference"] == 0 and not outcome["large"] and outcome["counts_lower"]
    assert outcome["next_round"] is False


def test_the_stop_share_is_read_among_episodes_that_never_heard_the_values(tmp_path):
    # Rule change 2026-10-09: an agent that delivers more values lowers "stop before values" by itself.
    # Both simulators stop at once whenever the values do not come; U1's agent delivers half of them.
    base, other = runs(
        tmp_path, lambda n: (False, EARLY), lambda n: (True, heard(n)) if n % 2 == 0 else (False, GOOD)
    )
    _, outcome = sim_report.check(base, other)
    base_counts, cand_counts = (sim_report.run_counts(d) for d in (base, other))
    assert cand_counts["stop_before_values"] < base_counts["stop_before_values"]
    assert outcome["stop_share"] == {"u0": 1.0, "u1": 1.0}
    assert outcome["counts_lower"] is False


def test_when_u0_invents_nothing_u1_must_invent_nothing_too(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    asking = runs(tmp_path / "a", lambda n: (True, GOOD), lambda n: (True, ASKING, "max_user_turns"))
    inventing = runs(
        tmp_path / "b", lambda n: (True, GOOD), lambda n: (True, ASKING_INVENTED, "max_user_turns")
    )
    assert sim_report.check(*asking)[1]["counts_lower"] is True
    assert sim_report.check(*inventing)[1]["counts_lower"] is False


def test_without_an_episode_that_missed_a_value_the_stop_share_cannot_be_lower(tmp_path):
    base, other = runs(tmp_path, lambda n: (False, EARLY), lambda n: (True, heard(n)))
    report, outcome = sim_report.check(base, other)
    assert outcome["stop_share"]["u1"] is None and outcome["counts_lower"] is False
    assert "알 수 없음" in report


def test_only_the_simulator_may_differ(tmp_path):
    base, other = runs(tmp_path, lambda n: (True, GOOD), lambda n: (True, GOOD), agent_model="qwen3:4b-x")
    with pytest.raises(ValueError, match="agent_provider|config.model"):
        sim_report.check(base, other)


def test_the_same_simulator_is_not_a_sensitivity_check(tmp_path):
    episodes = [episode("dev-001", GOOD)]
    base, other = write_run(tmp_path, "a", episodes), write_run(tmp_path, "b", episodes)
    with pytest.raises(ValueError, match="same simulator"):
        sim_report.check(base, other)


def test_the_sensitivity_check_never_reads_test_records(tmp_path):
    episodes = [episode("dev-001", GOOD)]
    base = write_run(tmp_path, "a", episodes)
    other = write_run(tmp_path, "b", episodes, user_model=U1, tasks_file="test")
    with pytest.raises(ValueError, match="development"):
        sim_report.check(base, other)


def test_a_simulator_prompt_near_its_context_limit_is_reported(tmp_path):
    long = episode("dev-001", GOOD)
    long["llm_calls"][1]["prompt_tokens"] = 7_800  # > 95% of 8192
    base = write_run(tmp_path, "u0", [episode("dev-001", GOOD)])
    other = write_run(tmp_path, "u1", [long], user_model=U1, user_num_ctx=8192, user_think=False)
    report, outcome = sim_report.check(base, other)
    assert outcome["prompts_over_limit"] == {"u0": 0, "u1": 1}
    assert "한도 95%를 넘은 시뮬레이터 호출" in report


def test_the_smoke_report_has_the_fit_gate_and_the_think_gate(tmp_path):
    ps = [{"name": AGENT, "size": 10, "size_vram": 10}, {"name": U1, "size": 10, "size_vram": 10}]
    clean = write_run(tmp_path, "s1", [episode("dev-001", GOOD)], user_model=U1, ps=ps)
    gates = {name: ok for name, _, ok in sim_report.smoke_gates(clean)}
    assert gates == {"S0": True, "S1": True, "T1": True}
    leaked = episode("dev-001", GOOD)
    leaked["llm_calls"][1]["text"] = "<think>음</think>환불 금액이요."
    bad = write_run(tmp_path, "s2", [leaked], user_model=U1, ps=ps[:1])
    gates = {name: ok for name, _, ok in sim_report.smoke_gates(bad)}
    assert gates == {"S0": True, "S1": False, "T1": False}


def test_analyze_has_the_three_stage_12_commands(tmp_path, monkeypatch, capsys):
    base, other = runs(tmp_path, lambda n: (True, GOOD), lambda n: (True, GOOD))
    for argv in (["sim-counts", str(base), str(other)], ["sim-check", str(base), str(other)]):
        monkeypatch.setattr("sys.argv", ["analyze", *argv])
        analyze.main()
    out = capsys.readouterr().out
    assert "| u1 | qwen3:4b-instruct-2507-q4_K_M |" in out and "U0과의 차이" in out


# ==================================================================== a scripted episode, end to end


def scripted(task_id, agent_script, simulator_lines):
    """A real episode on the seed data: ScriptedProvider plays the agent and, through LLMUser, the customer
    (so the simulator's calls are recorded as a model's would be)."""
    task = DEV[task_id]
    result = run_episode(
        task,
        0,
        config=RunConfig(policy="P1"),
        provider=ScriptedProvider(agent_script),
        user=LLMUser(ScriptedProvider(simulator_lines), task, RunConfig(), trial=0),
        registry=REGISTRY,
        seed_engine=build_seed_engine(),
        policy_text=load_policy(),
    )
    return json.loads(result.to_json_line())


def test_a_scripted_episode_is_counted_from_its_own_record():
    name, phone = CONTACTS["C-9101"]
    record = scripted(
        "dev-001",
        [
            ToolCall("find_customer", {"name": name, "contact": phone}),
            ToolCall("get_order", {"order_id": "O-91001"}),
            "환불 예정 금액은 58,300원입니다.",
            "네, 맞습니다.",
        ],
        [
            f"환불 금액이 궁금해요. {name}, {phone}입니다.",
            "58,300원 맞죠? 영수증 번호는 7777-1234예요.",
            ChatResponse(text="###STOP###"),
        ],
    )
    assert record["termination"] == "user_stop" and record["verdict"]["success"]
    assert sim_report.invented_values(record, DEV["dev-001"]) == [(1, ["num:77771234"])]
    assert sim_report.stop_before_values(record, DEV["dev-001"]) is False


def test_a_scripted_stop_before_the_value_is_counted():
    name, phone = CONTACTS["C-9101"]
    record = scripted(
        "dev-001",
        [ToolCall("find_customer", {"name": name, "contact": phone}), "확인되었습니다. 잠시만요."],
        [f"환불 금액이요. {name}, {phone}입니다.", "###STOP###"],
    )
    assert sim_report.stop_before_values(record, DEV["dev-001"]) is True
    assert sim_report.invented_values(record, DEV["dev-001"]) == []


# ==================================================================== the record and the rules


def test_the_paired_record_is_the_stage_8_m2b_dev_run_with_the_7b_simulator():
    manifest = json.loads((M2B_DEV / "manifest.json").read_text(encoding="utf-8"))
    agent, user = manifest["agent_provider"], manifest["user_provider"]
    assert (agent["model"], agent["think"], agent["num_ctx"]) == (AGENT, False, 12288)
    assert agent["digest"].startswith("2a654d98e6fb") and agent["ollama_version"] == "0.35.1"
    assert (user["model"], user["num_ctx"], user["num_gpu"], user["think"]) == (SIM, 12288, None, None)
    assert user["digest"].startswith("845dbda0ea48")
    assert manifest["tasks_file"] == "dev" and manifest["trials"] == 4
    assert manifest["config"]["user_temperature"] == 0.3 and manifest["config"]["base_seed"] == 1000


def test_the_counts_read_every_episode_of_the_paired_record():
    counts = sim_report.run_counts(M2B_DEV)
    assert counts["episodes"] == 96 and counts["infra"] == 0 and counts["changed_tasks"] == 0
    assert counts["utterances"] == 854  # every simulator call of the record


def test_the_experiments_rules_name_the_record_the_simulator_and_the_options():
    doc = (Path(__file__).resolve().parents[1] / "docs" / "experiments.md").read_text(encoding="utf-8")
    stage = doc[doc.index("## 12단계") :]
    assert M2B_DEV.name in stage
    for value in (U1, "0edcdef34593", "--user-think off", "--user-num-ctx 8192", "sim-check", "sim-counts"):
        assert value in stage, value
    assert f"{sim_report.MIN_DIGITS}자리" in stage
