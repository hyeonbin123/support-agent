"""Stage 9 (voice V4): the caller's number picks the one customer, the name heard may be one jamo off.

Everything here runs without a model: tools on the tiny shop, scripted episodes on the seed data, the runner
with fakes, and the analysis commands on hand-made records.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from conftest import NOW
from factories import extend_tiny_rows
from sqlalchemy.orm import Session
from test_run_voice import FakeOllama
from test_tasks import ALL_TASKS, CONTACTS, say_values
from test_voice import FakeListener, FakeSpeaker

from support_agent import analyze, db, run
from support_agent.agent import build_system_prompt, load_policy, visible_tools
from support_agent.chat import ScriptedProvider, ToolCall
from support_agent.config import RunConfig
from support_agent.episode import run_episode
from support_agent.seed import build_seed_engine
from support_agent.toolkit import ConversationState, ToolContext, execute
from support_agent.tools import NAME_TOLERANCE, build_registry, name_distance
from support_agent.user_sim import ScriptedUser
from support_agent.voice import metrics
from support_agent.voice.channel import SpeechChannel

TEXT_REGISTRY = build_registry()
V4_REGISTRY = build_registry(caller_id=True)
V4 = RunConfig(voice="V4")
# Hashes in the manifests of the development runs so far (stage 5 V1/V2, stage 8 P0·7B): V0..V2 keep them.
RECORDED_AGENT_SYSTEM = "7238ce2ba88a44d37e194701bff80b2224156510e84f61dee6fc2520968dafc7"
RECORDED_TOOLS = "565d3ecc74d8b6cf826ce2c9c74026f3c42542a23628722017367c360e2f937a"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tools_hash(registry, config) -> str:
    return sha(json.dumps(visible_tools(registry, config), ensure_ascii=False, sort_keys=True))


# ------------------------------------------------------------------------------- the name tolerance


def test_the_registered_tolerance_is_one_jamo():
    assert NAME_TOLERANCE == 1


@pytest.mark.parametrize(
    ("said", "registered", "distance"),
    [  # left-hand names are how the recogniser wrote them in the development V1/V2 runs
        ("배성훈", "배성훈", 0),
        ("배성분", "배성훈", 1),  # ㅂ for ㅎ
        ("최민서", "채민서", 1),  # ㅚ for ㅐ
        ("허윤설", "허윤서", 1),  # a final ㄹ added
        ("하도병", "하도경", 1),
        ("안재기", "안재희", 2),  # ㄱㅣ for ㅎㅢ
        ("송민리", "송민규", 2),
        ("빌태윤희", "길태윤", 3),
        ("배성훈 고객님", "배성훈", 0),  # spaces and the honorific go first, as in find_customer
        ("배 성훈님", "배성훈", 0),
        ("Bae Sunghoon", "배성훈", 11),  # "BaeSunghoon" against eight jamo
    ],
)
def test_names_are_compared_jamo_by_jamo(said, registered, distance):
    assert name_distance(said, registered) == distance


# ------------------------------------------------------------------------------- verify_caller


@pytest.fixture
def shop(tiny_engine):
    engine = db.memory_engine(tiny_engine)
    with Session(engine) as session:
        extend_tiny_rows(session)  # C-1 김하준 01000000001, C-2 이서연 01000000002
        session.commit()
    yield engine
    engine.dispose()


def verify(engine, ctx: ToolContext, name: str):
    return execute(V4_REGISTRY, engine, ctx, "verify_caller", {"name": name})


def test_the_caller_number_and_the_name_verify_the_customer(shop):
    ctx = ToolContext(now=NOW, caller_phone="01000000001")
    result = verify(shop, ctx, "김하준")
    assert result.ok, result.content
    assert json.loads(result.content) == {
        "customer_id": "C-1",
        "name": "김하준",
        "grade": "normal",
        "grade_label": "일반",
    }
    assert ctx.state.verified_customer_id == "C-1"
    order = execute(V4_REGISTRY, shop, ctx, "get_order", {"order_id": "O-1"})
    assert order.ok  # verified as with find_customer


def test_one_jamo_off_passes_and_the_registered_name_comes_back(shop):
    ctx = ToolContext(now=NOW, caller_phone="01000000001")
    result = verify(shop, ctx, "김하순")  # ㅅ for ㅈ
    assert result.ok and json.loads(result.content)["name"] == "김하준"


@pytest.mark.parametrize(
    ("caller", "name", "code"),
    [
        ("01000000001", "김하정", "name_mismatch"),  # two jamo off
        ("01000000001", "이서연", "name_mismatch"),  # another customer's name from this phone
        ("01000000002", "김하준", "name_mismatch"),  # the name of the other number's customer
        ("01000000001", "", "name_mismatch"),
        ("01099999999", "김하준", "caller_not_registered"),
        (None, "김하준", "no_caller_number"),
    ],
)
def test_anything_else_is_refused_and_verifies_nobody(shop, caller, name, code):
    ctx = ToolContext(now=NOW, caller_phone=caller)
    result = verify(shop, ctx, name)
    assert not result.ok and result.error_code == code, result.content
    assert ctx.state.verified_customer_id is None


def test_a_number_shared_by_two_customers_is_not_one_to_one(shop):
    with Session(shop) as session:
        session.get(db.Customer, "C-2").phone = "01000000001"
        session.commit()
    ctx = ToolContext(now=NOW, caller_phone="01000000001")
    result = verify(shop, ctx, "김하준")
    assert not result.ok and result.error_code == "caller_shared"


def test_one_customer_per_conversation_still_holds(shop):
    ctx = ToolContext(
        now=NOW, caller_phone="01000000001", state=ConversationState(verified_customer_id="C-2")
    )
    result = verify(shop, ctx, "김하준")
    assert not result.ok and result.error_code == "already_verified"
    again = ToolContext(now=NOW, caller_phone="01000000001")
    assert verify(shop, again, "김하준").ok and verify(shop, again, "김하준").ok  # the same one twice is fine


# ------------------------------------------------------------------------------- what the agent sees


def test_v4_swaps_find_customer_for_verify_caller_and_nothing_else():
    assert "verify_caller" not in TEXT_REGISTRY and "find_customer" in TEXT_REGISTRY  # service, MCP, V0..V2
    assert "verify_caller" in V4_REGISTRY and "find_customer" not in V4_REGISTRY
    others = [name for name in TEXT_REGISTRY if name != "find_customer"]
    assert [name for name in V4_REGISTRY if name != "verify_caller"] == others
    assert list(V4_REGISTRY)[0] == "verify_caller"  # in the place find_customer had


def test_find_customer_is_an_unknown_tool_under_v4(shop):
    ctx = ToolContext(now=NOW, caller_phone="01000000001")
    result = execute(V4_REGISTRY, shop, ctx, "find_customer", {"name": "김하준", "contact": "010-0000-0001"})
    assert not result.ok and result.error_code == "unknown_tool"


def test_v0_to_v2_prompts_and_tools_are_byte_for_byte_the_recorded_ones():
    dev_now = next(t for t in ALL_TASKS if t.id == "dev-001").now
    for voice in ("V0", "V1", "V2"):
        config = RunConfig(voice=voice)
        assert sha(build_system_prompt(load_policy(), dev_now, caller_id=config.caller_id)) == (
            RECORDED_AGENT_SYSTEM
        )
        assert tools_hash(TEXT_REGISTRY, config) == RECORDED_TOOLS
    assert sha(build_system_prompt(load_policy(), dev_now)) == RECORDED_AGENT_SYSTEM


def test_the_v4_prompt_adds_one_rule_line_and_nothing_else():
    dev_now = next(t for t in ALL_TASKS if t.id == "dev-001").now
    text = build_system_prompt(load_policy(), dev_now)
    voice = build_system_prompt(load_policy(), dev_now, caller_id=True)
    added = [line for line in voice.splitlines() if line not in text.splitlines()]
    assert len(added) == 1 and added[0].startswith("- ") and "verify_caller" in added[0]
    assert "발신 번호" in added[0] and voice.replace(added[0] + "\n", "", 1) == text
    assert V4.caller_id and not RunConfig(voice="V2").caller_id
    assert tools_hash(V4_REGISTRY, V4) != RECORDED_TOOLS


@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda task: task.id)
def test_no_required_value_is_given_away_by_the_v4_prompt(task):
    from support_agent.judge import validate_task

    prompt = build_system_prompt(load_policy(), task.now, caller_id=True)
    assert validate_task(task, build_seed_engine(), V4_REGISTRY, prompt_texts=[prompt]) == []


# ------------------------------------------------------------------------------- in an episode


def v4_episode(task, script, *, config=V4, registry=V4_REGISTRY):
    return run_episode(
        task,
        0,
        config=config,
        provider=ScriptedProvider(script),
        user=ScriptedUser(["문의드립니다."]),
        registry=registry,
        seed_engine=build_seed_engine(),
        policy_text=load_policy(),
    )


@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda task: task.id)
def test_the_gold_path_with_verify_caller_passes_under_v4(task):
    name, _ = CONTACTS[task.customer_id]
    script = [ToolCall("verify_caller", {"name": name})]
    script += [ToolCall(a.tool, a.args) for a in task.gold_actions] + [say_values(task)]
    result = v4_episode(task, script, config=RunConfig(voice="V4", policy="P1"))
    assert result.verdict is not None and result.verdict.success, (result.termination, result.verdict)
    assert result.tool_calls[0].ok


def test_the_caller_is_the_task_customer_and_a_misheard_name_still_verifies():
    task = next(t for t in ALL_TASKS if t.id == "dev-002")  # 배성훈, who gave an e-mail address
    result = v4_episode(task, [ToolCall("verify_caller", {"name": "배성분"}), "확인되었습니다."])
    (call,) = result.tool_calls
    assert call.ok and json.loads(call.content)["customer_id"] == task.customer_id == "C-9102"
    assert metrics.identified(json.loads(result.to_json_line()))


def test_v4_needs_its_registry_and_the_other_conditions_refuse_it():
    task = next(t for t in ALL_TASKS if t.id == "dev-001")
    with pytest.raises(ValueError, match="caller_id"):
        v4_episode(task, ["네."], registry=TEXT_REGISTRY)
    with pytest.raises(ValueError, match="caller_id"):
        v4_episode(task, ["네."], config=RunConfig(voice="V2"))


def test_a_text_episode_has_no_caller_number():
    task = next(t for t in ALL_TASKS if t.id == "dev-001")
    name, contact = CONTACTS[task.customer_id]
    result = v4_episode(
        task,
        [ToolCall("find_customer", {"name": name, "contact": contact}), "네."],
        config=RunConfig(voice="V2"),
        registry=TEXT_REGISTRY,
    )
    assert result.tool_calls[0].ok


# ------------------------------------------------------------------------------- the runner


def test_a_v4_run_uses_the_v2_normaliser_the_caller_registry_and_records_k(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OllamaProvider", FakeOllama)
    monkeypatch.setattr(run, "gpu_used_by_others_mib", lambda provider: None)
    monkeypatch.setattr(run, "OUTPUTS", tmp_path)
    built = []

    def build_channel(config, tts_device, stt_device):
        built.append(config.voice)
        return SpeechChannel(FakeSpeaker(), FakeListener(), cache_path=tmp_path / "cache.jsonl")

    monkeypatch.setattr(run, "build_channel", build_channel)
    for voice in ("V2", "V4"):
        argv = ["run", "--tasks", "smoke", "--task-id", "smoke-lookup-01", "--voice", voice]
        monkeypatch.setattr("sys.argv", [*argv, "--label", voice.lower()])
        run.main()
    assert built == ["V2", "V4"]
    v2, v4 = sorted((tmp_path / "runs").iterdir(), key=lambda p: p.name[-2:])
    assert v4.name.endswith("-P0-R0-G0-F0-V4-v4")
    m2, m4 = (json.loads((d / "manifest.json").read_text(encoding="utf-8")) for d in (v2, v4))
    assert m4["config"]["voice"] == "V4" and m4["caller_id"] == {"tool": "verify_caller", "name_tolerance": 1}
    assert m2["caller_id"] is None
    assert m2["prompt_sha256"]["agent_system"] == sha(
        build_system_prompt(load_policy(), next(t for t in ALL_TASKS if t.id == "smoke-lookup-01").now)
    )
    differs = {k for k in m2["prompt_sha256"] if m2["prompt_sha256"][k] != m4["prompt_sha256"][k]}
    assert differs == {"agent_system", "tools"}


def test_the_v4_channel_is_the_v2_channel(tmp_path, monkeypatch):
    import support_agent.voice.speech as speech

    class Speaker(FakeSpeaker):
        def __init__(self, device="cuda"):
            super().__init__()

    class Listener(FakeListener):
        def __init__(self, device="cuda"):
            pass

    monkeypatch.setattr(speech, "MeloSpeaker", Speaker)
    monkeypatch.setattr(speech, "WhisperListener", Listener)
    monkeypatch.setattr(run, "OUTPUTS", tmp_path)
    seen = {
        v: run.build_channel(RunConfig(voice=v), "cpu", "cpu").describe()["normalizer"] for v in ("V2", "V4")
    }
    assert seen == {"V2": "normalize_heard", "V4": "normalize_heard"}


# ------------------------------------------------------------------------------- analysis


def test_identification_counts_verify_caller_too():
    assert metrics.identified({"tool_calls": [{"name": "verify_caller", "ok": True}]})
    assert not metrics.identified({"tool_calls": [{"name": "verify_caller", "ok": False}]})
    assert metrics.identified({"tool_calls": [{"name": "find_customer", "ok": True}]})


def write_manifest(root, name, prompts, voice):
    run_dir = root / name
    run_dir.mkdir()
    manifest = {
        "config": {"voice": voice},
        "trials": 1,
        "task_sha256": {"dev-001": "a"},
        "seed_hash": "s",
        "prompt_sha256": prompts,
        "agent_provider": {"model": "m"},
        "user_provider": {"model": "m"},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    line = {"task_id": "dev-001", "trial": 0, "status": "completed"}
    (run_dir / "episodes.jsonl").write_text(json.dumps(line), encoding="utf-8")
    return run_dir


def test_same_setup_can_let_named_prompt_hashes_differ(tmp_path, monkeypatch, capsys):
    base = {"policy": "p", "agent_system": "a", "user_sim": "u", "tools": "t"}
    v2 = write_manifest(tmp_path, "v2", base, "V2")
    v4 = write_manifest(tmp_path, "v4", {**base, "agent_system": "a4", "tools": "t4"}, "V4")
    other = write_manifest(tmp_path, "v4p", {**base, "agent_system": "a4", "policy": "p4"}, "V4")
    assert analyze.setup_differences(v2, v4) == ["prompt_sha256 differs (agent_system, tools)"]
    assert analyze.setup_differences(v2, v4, ignore_prompts=("tools", "agent_system")) == []
    assert analyze.setup_differences(v2, other, ignore_prompts=("tools", "agent_system")) == [
        "prompt_sha256 differs (policy)"
    ]
    argv = ["analyze", "same-setup", str(v2), str(v4), "--ignore", "voice"]
    monkeypatch.setattr("sys.argv", [*argv, "--ignore-prompt", "tools", "agent_system"])
    analyze.main()
    assert "same setup" in capsys.readouterr().out


def test_the_collision_check_counts_other_names_within_k_and_sets_same_names_apart():
    names = {"C-1": "김하준", "C-2": "김하순", "C-3": "박서연", "C-4": "김하준", "C-5": "김하정"}
    collisions, same = analyze.name_collisions(names, k=1)
    # 김하정 is two jamo from 김하준 and three from 김하순: no collision at k=1
    assert [(a, b) for a, b, *_ in collisions] == [
        ("C-1", "C-2"),
        ("C-2", "C-1"),
        ("C-2", "C-4"),
        ("C-4", "C-2"),
    ]
    assert same == [("C-1", "C-4"), ("C-4", "C-1")]
    assert analyze.name_collisions({"C-1": "김하준", "C-3": "박서연"}, k=1) == ([], [])
    collisions_k2, _ = analyze.name_collisions(names, k=2)
    assert ("C-1", "C-5") in [(a, b) for a, b, *_ in collisions_k2]


def episode_with(calls, task_id="dev-001"):
    return {"task_id": task_id, "status": "completed", "tool_calls": calls, "voice": []}


def call(name, ok, args=None, content="", code=None):
    return {"name": name, "ok": ok, "args": args, "content": content, "error_code": code}


def test_the_caller_id_table_tells_tolerance_passes_apart(tmp_path):
    found = json.dumps({"customer_id": "C-9101", "name": "고은채"}, ensure_ascii=False)
    v4 = [
        episode_with([call("verify_caller", True, {"name": "고은채"}, found)]),
        episode_with(
            [
                call("verify_caller", False, None, code="name_mismatch"),
                call("verify_caller", True, {"name": "고은재"}, found),
            ]
        ),
        episode_with([call("find_customer", False, None, code="unknown_tool")]),
        episode_with([]),
        {**episode_with([call("verify_caller", False, code="no_caller_number")]), "status": "truncated"},
        {**episode_with([]), "status": "infra_error"},  # left out
    ]
    v2 = [episode_with([call("find_customer", True, {"name": "고은채", "contact": "x"}, found)])] * 4
    table = analyze.caller_id_table({"v2": v2, "v4": v4})
    assert "| v2 | 4 | 4 (100.0%) | 0 | 4 / 4 | 0 / 0 | 0 | 0 | 0 |" in table
    assert "| v4 | 5 | 2 (40.0%) | 1 | 0 / 1 | 2 / 4 | 1 | 1 | 1 |" in table


def test_the_recorded_gate_result_comes_back_from_the_seed():
    # docs/experiments.md stage 9: the gate failed at k=1 with 12 ordered pairs, and the stage stopped there
    names = analyze.seed_names()
    collisions, same = analyze.name_collisions(names, k=1)
    assert len(names) == 109 and len(same) == 6 and len(collisions) == 12
    assert {frozenset((a, b)) for a, b, *_ in collisions} == {
        frozenset(pair)
        for pair in [
            ("C-0006", "C-0016"),
            ("C-0024", "C-0028"),
            ("C-0028", "C-0030"),
            ("C-9101", "C-9211"),
            ("C-9110", "C-9203"),
            ("C-9120", "C-9208"),
        ]
    }
    assert analyze.name_collisions(names, k=0)[0] == []
    report, passed = analyze.collision_report(1)
    assert not passed and report.endswith("gate: FAIL")
