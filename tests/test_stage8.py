"""Stage 8 (a new-generation agent model): runner options, smoke gates, the selection rule, the test
verdict and the blind simulator check. No model, no GPU: the runner gets a fake provider and analyze
reads synthetic records.
"""

# [파일 역할] 8단계(새 세대 4B 에이전트 모델) 측정에 새로 넣은 코드를 모델·GPU 없이 검증하는 테스트 묶음이다.
# 네 구역으로 나뉜다: 실행기 옵션(run.py), 스모크 관문 S0~S5(analyze smoke), 개발용 선택 규칙(analyze
# select)과 시험용 판정(analyze verdict), 가린 시뮬레이터 점검(analyze blind·unblind).
# [왜 중요한가] 8단계는 규칙을 재기 전에 커밋하고 그 규칙대로 고르고 판정한다(docs/experiments.md 8단계).
# 관문·선택·판정을 계산하는 코드가 문서의 숫자(+5%p, 3배, P0 참조 이하, S2 1,767 ms 등)와 어긋나면 결과가
# 규칙이 아니라 코드 실수로 정해진다. 실행기 옵션이 잘못되면 Ollama가 턴마다 모델을 다시 읽어 시간 수치가
# 망가진다.
# [방법] 실행기는 가짜 Ollama 제공자로 스모크 과제 하나를 끝까지 돌리고 manifest·summary를 읽는다. analyze
# 함수들은 이 파일이 직접 만든 합성 기록(episodes.jsonl, manifest.json, summary.json)을 읽게 한다.

from __future__ import annotations

import json
import random

import pytest

# analyze: 관문·선택·판정·점검표 계산(검증 대상). run: 실행기 CLI(검증 대상). ChatProvider를 상속해 가짜
# Ollama를 만든다.
from support_agent import analyze, run
from support_agent.chat import ChatProvider, ChatResponse
from support_agent.config import RunConfig

# SIM: 고정된 시뮬레이터 모델(qwen2.5 7B), NEW: 후보 M2a 모델. docs/experiments.md 8단계 후보 표의 이름
# 그대로다.
SIM = "qwen2.5:7b-instruct"
NEW = "qwen3:4b-instruct-2507-q4_K_M"


# 실행기: 시뮬레이터 컨텍스트·GPU 층·think 옵션과 모델 미리 올리기.
# ---------------------------------------------------------------- the runner


# 가짜 OllamaProvider. 만들어질 때의 모델과 옵션을 클래스 목록(made)에 남겨, 실행기가 에이전트·시뮬레이터
# 제공자를 어떤 옵션으로 몇 개 만들었는지 확인하게 한다. 도구 목록이 오면(에이전트 요청) 본인 확인을 묻는
# 답을, 아니면(시뮬레이터 요청) 고객 대본을 차례로 돌려준다. loaded_models는 /api/ps 대신 자기 모델 하나가
# GPU에 다 올라간 것으로 답한다.
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


# 실행기 run.main()을 스모크 과제 하나로 돌리고, 만들어진 가짜 제공자들과 manifest·summary를 돌려준다.
# OllamaProvider를 가짜로, 다른 프로그램의 GPU 사용 검사(gpu_used_by_others_mib)를 '알 수 없음'으로, 출력
# 폴더를 임시 폴더로 바꿔 GPU·서버 없이 실행 경로 전체를 지나게 한다.
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


# --user-num-ctx를 주면 시뮬레이터 제공자만 그 컨텍스트(8192)를 받고 에이전트는 --num-ctx(12288)를 받는지, 둘
# 다 manifest(config, user_provider)에 남는지 본다. 8단계 실행기 옵션 2번(docs/experiments.md)이다.
def test_the_simulator_gets_its_own_context_size(tmp_path, monkeypatch):
    made, manifest, _ = start(
        tmp_path, monkeypatch, "--model", NEW, "--num-ctx", "12288", "--user-num-ctx", "8192"
    )
    agent, user = made
    assert (agent.model, agent.options["num_ctx"]) == (NEW, 12288)
    assert (user.model, user.options["num_ctx"], user.options["num_gpu"]) == (SIM, 8192, None)
    assert manifest["config"]["num_ctx"] == 12288 and manifest["config"]["user_num_ctx"] == 8192
    assert manifest["user_provider"]["num_ctx"] == 8192


# 옵션이 없으면 두 역할이 같은 모델이라 제공자 객체가 하나만 만들어지고, 시뮬레이터 컨텍스트가 에이전트와
# 같은지 본다. 이전 명령이 그대로 돌아야 0.34.x 기록과 같은 설정을 다시 낼 수 있다.
def test_without_the_flag_the_simulator_shares_the_agent_context_and_the_provider(tmp_path, monkeypatch):
    made, manifest, _ = start(tmp_path, monkeypatch)
    assert len(made) == 1  # one model, one set of runner options: Ollama never reloads between the roles
    assert manifest["config"]["user_num_ctx"] == manifest["config"]["num_ctx"] == RunConfig.num_ctx
    assert manifest["user_provider"] == manifest["agent_provider"]


# 같은 모델이 두 역할을 맡는데 실행기 옵션(num_ctx, num_gpu)을 다르게 주면 실행기가 시작하지
# 않는지(SystemExit) 본다. Ollama는 모델 하나에 실행기 하나를 두므로 옵션이 다른 요청이 오면 모델을 다시 읽어,
# 매 턴 다시 읽게 된다(docs/experiments.md 8단계 "재기 전에 고친 것" 1).
@pytest.mark.parametrize("extra", [["--user-num-ctx", "8192"], ["--user-on-cpu"]])
def test_one_model_in_both_roles_cannot_have_two_sets_of_runner_options(tmp_path, monkeypatch, extra):
    # Ollama keeps one runner per model: different num_ctx or num_gpu would reload the model every turn.
    with pytest.raises(SystemExit):
        start(tmp_path, monkeypatch, *extra)


# --user-num-gpu 20이 시뮬레이터 제공자의 num_gpu로 가고 manifest에 남는지 본다. 두 모델이 GPU에 다 올라가지
# 않을 때 시뮬레이터 층 일부만 CPU로 옮기는 '시뮬레이터 자리 사다리' 2단계용 옵션이다.
def test_part_of_the_simulator_can_sit_on_the_cpu(tmp_path, monkeypatch):
    made, manifest, _ = start(tmp_path, monkeypatch, "--model", NEW, "--user-num-gpu", "20")
    assert made[1].options["num_gpu"] == 20 and manifest["user_provider"]["num_gpu"] == 20


# --user-on-cpu(층 0개)와 --user-num-gpu 20을 함께 주면 모순이라 시작하지 않는지 본다.
def test_on_cpu_and_a_layer_count_contradict_each_other(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        start(tmp_path, monkeypatch, "--model", NEW, "--user-on-cpu", "--user-num-gpu", "20")


# 2단계 M1 명령(14B 에이전트, CPU 시뮬레이터, 8192)이 새 옵션을 넣은 뒤에도 같은 제공자 옵션을 만드는지 본다.
# README '다시 재는 방법'의 명령이 바뀌지 않게 지킨다.
def test_the_stage_2_m1_command_still_builds_the_same_simulator(tmp_path, monkeypatch):
    # README: --model qwen2.5:14b-instruct --user-on-cpu --num-ctx 8192 (both roles at 8192, simulator on CPU)
    made, manifest, _ = start(
        tmp_path, monkeypatch, "--model", "qwen2.5:14b-instruct", "--user-on-cpu", "--num-ctx", "8192"
    )
    agent, user = made
    assert agent.options == {"num_ctx": 8192}
    assert user.options == {"num_ctx": 8192, "num_gpu": 0}
    assert manifest["config"]["user_num_ctx"] == 8192


# --think off가 에이전트 제공자에만 think=False로 가고 시뮬레이터에는 가지 않으며 manifest에 남는지 본다. 후보
# 두 모델을 비사고 모드로 명시하되 시뮬레이터 요청은 이전과 같아야 한다(docs/experiments.md 8단계 후보 표
# 아래).
def test_think_goes_to_the_agent_and_never_to_the_simulator(tmp_path, monkeypatch):
    made, manifest, _ = start(tmp_path, monkeypatch, "--model", "qwen3.5:4b", "--think", "off")
    agent, user = made
    assert agent.options["think"] is False and user.options.get("think") is None
    assert manifest["agent_provider"]["think"] is False


# 같은 모델이 두 역할을 맡을 때 --think를 주면 think는 요청 필드라 제공자 객체가 둘로 나뉘지만, 실행기
# 옵션(num_ctx, num_gpu)은 같아 모델을 다시 읽지 않는지 본다.
def test_think_with_one_model_in_both_roles_keeps_the_runner_options_equal(tmp_path, monkeypatch):
    made, _, _ = start(tmp_path, monkeypatch, "--think", "on")
    agent, user = made  # two objects (think is per request), equal runner options (no reload)
    assert agent.options["think"] is True and user.options.get("think") is None
    assert agent.options["num_ctx"] == user.options["num_ctx"]
    assert agent.options.get("num_gpu") == user.options.get("num_gpu")


# --think를 주지 않으면 think 필드를 아예 보내지 않는지 본다. 아래 줄 주석대로 사고 기능이 없는 모델은 think
# 값이 오면 HTTP 400으로 답하기 때문이다.
def test_without_think_nothing_is_sent(tmp_path, monkeypatch):
    made, _, _ = start(tmp_path, monkeypatch, "--model", NEW)
    assert made[0].options.get("think") is None  # a model without thinking answers 400 to any think value


# 서버 하나를 흉내 내는 가짜 제공자: 시뮬레이터를 처음 올릴 때 이미 올라간 에이전트 모델을 내린다(held를
# 비움). 2026-10-04 스모크에서 실제로 qwen2.5 7B를 올리면서 qwen3 4B가 내려가 첫 에이전트 호출이 다시 읽기로
# S1을 어긴 일을 재현한다(docs/experiments.md 8단계 "측정 중 메모").
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


# 미리 올린 뒤 /api/ps에 에이전트 모델이 없으면 실행기가 한 번 더 올리는지(올린 순서 NEW → SIM → NEW),
# manifest의 ollama_ps_after_preload에 두 모델이 다 있는지 본다. 첫 에피소드가 이 다시 읽기 비용을 치르지 않게
# 한 하니스 수정(커밋 80557c2)의 테스트다.
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


# 에이전트 모델이 아직 올라가 있으면 다시 올리지 않는지 본다(제공자는 에이전트·시뮬레이터 두 개만 만들어짐).
def test_a_model_still_held_is_not_loaded_again(tmp_path, monkeypatch):
    start(tmp_path, monkeypatch, "--model", NEW)  # the plain fake always lists the agent model
    assert [m.model for m in RecordingOllama.made] == [NEW, SIM]


# 모델을 올린 직후의 /api/ps가 manifest에, 끝날 때의 /api/ps가 summary의 ollama_ps_at_end에 남는지 본다. 관문
# S1(두 모델이 GPU에 함께 머물렀는가)이 이 기록을 읽는다.
def test_the_records_say_where_the_models_sat(tmp_path, monkeypatch):
    _, manifest, summary = start(tmp_path, monkeypatch, "--model", NEW, "--user-num-ctx", "16384")
    assert manifest["ollama_ps_after_preload"] == [{"name": NEW, "size": 100, "size_vram": 100}]
    assert summary["ollama_ps_at_end"] == [{"name": NEW, "size": 100, "size_vram": 100}]


# 합성 기록: analyze가 읽는 episodes.jsonl·manifest.json·summary.json을 최소 필드로 만든다.
# ---------------------------------------------------------------- synthetic records


# LLM 호출 기록 하나. 시간(wall_ms), 프롬프트 처리 시간(prompt_eval_ms), 모델 적재 시간(load_ms), 형식 오류,
# 버린 호출 수를 바꿔 관문·선택 규칙의 각 조건을 건드린다.
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


# 에피소드 기록 하나. 판정(성공, 정답에 없는 쓰기, 통과된 규정 위반), LLM 호출, 상담원 답, 쓰기 도구 호출 수를
# 정한다. infra_error면 판정이 없다(분모에서 빠짐).
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


# manifest에 남는 제공자 설명. 모델 digest, Ollama 버전, num_ctx, num_gpu, think가 짝짓기 검사(same-setup)와
# 선택 규칙의 기준 확인에 쓰인다.
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


# 실행 폴더 하나를 만든다: manifest(설정, 과제 파일, 과제·프롬프트 해시, 두 제공자), summary(끝날 때의
# /api/ps), episodes.jsonl. 에이전트가 SIM이 아니면 시뮬레이터 제공자는 SIM으로 따로 둔다.
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


# 두 모델이 모두 GPU에 다 올라간 /api/ps(size == size_vram).
ON_GPU = [{"name": NEW, "size": 4_000, "size_vram": 4_000}, {"name": SIM, "size": 5_000, "size_vram": 5_000}]


# 스모크 5과제 × 1회 기록. 기본 호출은 첫 호출만 프롬프트 처리가 길고(캐시 없음) 나머지는 짧아, 모든 관문을
# 통과하는 깨끗한 실행이 된다.
def smoke_episodes(**overrides):
    calls = [call(index=0, prompt_eval=900.0), call(index=1), call(index=2), call("user", index=0)]
    return [ep(f"smoke-{i}", 0, i % 2 == 0, calls=overrides.get("calls", calls)) for i in range(5)]


# analyze.smoke_gates 결과를 관문 이름 → 통과 여부로 줄인다.
def gates(run_dir):
    return {name: ok for name, _detail, ok in analyze.smoke_gates(run_dir)}


# 스모크 관문(측정 아님): S0 infra 오류 0, S1 GPU 자리·다시 읽기 0, S2 호출당 시간, S3 프롬프트 캐시 재사용,
# S4 형식 오류, S5 다른 문자 섞인 답(docs/experiments.md 8단계 "스모크" 표).
# ---------------------------------------------------------------- smoke gates


# 깨끗한 스모크 실행이 S0~S5를 모두 통과하고, 보고문에 GPU 자리가 적히는지 본다.
def test_a_clean_smoke_run_passes_every_gate(tmp_path):
    run_dir = write_run(tmp_path, "smoke", smoke_episodes(), model=NEW, ps=ON_GPU)
    assert gates(run_dir) == {"S0": True, "S1": True, "S2": True, "S3": True, "S4": True, "S5": True}
    assert "GPU" in analyze.smoke_report(run_dir)


# 자리를 지정하지 않은 시뮬레이터가 일부만 GPU에 있으면(size_vram < size) S1을 어기는지 본다. 지정이 없으면
# 100% GPU여야 한다.
def test_a_model_partly_on_the_cpu_fails_the_fit_gate(tmp_path):
    ps = [ON_GPU[0], {"name": SIM, "size": 5_000, "size_vram": 3_000}]
    assert gates(write_run(tmp_path, "smoke", smoke_episodes(), model=NEW, ps=ps))["S1"] is False


# 끝날 때 /api/ps에 시뮬레이터가 없으면(내려감) S1을 어기는지 본다.
def test_a_simulator_that_was_evicted_fails_the_fit_gate(tmp_path):
    assert gates(write_run(tmp_path, "smoke", smoke_episodes(), model=NEW, ps=ON_GPU[:1]))["S1"] is False


# /api/ps 기록이 없으면 확인할 수 없으므로 S1을 통과로 치지 않는지 본다.
def test_without_a_ps_record_the_fit_gate_fails(tmp_path):
    assert gates(write_run(tmp_path, "smoke", smoke_episodes(), model=NEW))["S1"] is False


# 중간 호출 하나가 모델 적재에 1초 넘게(4초) 걸리면 다시 읽은 것으로 보고 S1을 어기는지 본다. 끝날 때 둘 다
# GPU에 있어도 중간에 번갈아 다시 읽으면 시간 수치를 믿을 수 없다.
def test_a_reload_in_the_middle_fails_the_fit_gate(tmp_path):
    calls = [call(index=0), call(index=1, load=4_000.0), call("user", index=0)]
    run_dir = write_run(tmp_path, "smoke", smoke_episodes(calls=calls), model=NEW, ps=ON_GPU)
    assert gates(run_dir)["S1"] is False


# 시뮬레이터를 일부러 CPU에 둔 실행(num_gpu=0)은 시뮬레이터 size_vram 0이 지정과 맞으므로 S1을 통과하고,
# 보고문에 CPU라고 적히는지 본다.
def test_a_simulator_on_the_cpu_on_purpose_passes_the_fit_gate_and_says_so(tmp_path):
    ps = [ON_GPU[0], {"name": SIM, "size": 5_000, "size_vram": 0}]
    user = provider(SIM, num_gpu=0)
    run_dir = write_run(tmp_path, "smoke", smoke_episodes(), model=NEW, ps=ps, user=user)
    assert gates(run_dir)["S1"] is True
    assert "CPU" in analyze.smoke_report(run_dir)


# 에이전트 호출 평균이 1,950 ms로 상한 1,767 ms(0.34.2 기록의 P1·7B 589 ms의 3배)를 넘으면 S2를 어기는지 본다.
def test_slow_agent_calls_fail_the_latency_gate(tmp_path):
    calls = [call(index=0, ms=2_000.0), call(index=1, ms=1_900.0), call("user", index=0)]
    run_dir = write_run(tmp_path, "smoke", smoke_episodes(calls=calls), model=NEW, ps=ON_GPU)
    assert gates(run_dir)["S2"] is False


# 모든 에이전트 호출의 프롬프트 처리가 800 ms면 중앙값이 상한 500 ms를 넘어 S3을 어기는지 본다. 캐시를 다시
# 쓰면 수십 ms, 매번 처음부터 계산하면 수 초라서 중앙값으로 캐시 재사용 여부를 가른다(qwen3.5 구조에서 캐시를
# 다시 쓰지 못한다는 보고가 있어 넣은 관문).
def test_a_prompt_cache_that_is_not_reused_fails_its_gate(tmp_path):
    calls = [call(index=i, prompt_eval=800.0) for i in range(3)] + [call("user", index=0)]
    run_dir = write_run(tmp_path, "smoke", smoke_episodes(calls=calls), model=NEW, ps=ON_GPU)
    assert gates(run_dir)["S3"] is False


# 샌 도구 호출과 본문의 <think> 태그를 함께 세어 에피소드당 2.0이면 S4(상한 1.0)를 어기는지 본다. 가드가
# 붙잡은 답(stall, unbacked_claim)은 형식이 잘못된 것이 아니므로 세지 않는다.
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


# 한 에피소드의 답이 모두 중국어면 한자·가나 2자 이상 섞인 답 비율이 상한 10%를 넘어 S5를 어기는지 본다.
def test_replies_in_another_script_fail_the_language_gate(tmp_path):
    episodes = smoke_episodes()
    episodes[0]["messages"] = [{"role": "assistant", "content": "您好，订单已经取消了。"}] * 3
    assert gates(write_run(tmp_path, "smoke", episodes, model=NEW, ps=ON_GPU))["S5"] is False


# 고객에게 전달되지 않은 답(delivered False)과 도구 호출만 한 메시지는 S5에서 읽지 않는지 본다. S5는 '고객에게
# 전달된 답'만 센다.
def test_replies_that_were_held_back_or_called_a_tool_are_not_read_for_language(tmp_path):
    episodes = smoke_episodes()
    episodes[0]["messages"] = [
        {"role": "assistant", "content": "您好，订单已经取消了。", "delivered": False},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "get_order", "arguments": {}}]},
    ] * 3
    assert gates(write_run(tmp_path, "smoke", episodes, model=NEW, ps=ON_GPU))["S5"] is True


# infra 오류 에피소드가 하나라도 있으면 S0을 어기는지 본다. S0 위반은 후보 탈락이 아니라 하니스를 고치고 다시
# 도는 신호다.
def test_an_infra_error_fails_the_first_gate(tmp_path):
    episodes = smoke_episodes()
    episodes[1] = ep("smoke-1", 0, False, status="infra_error", termination="infra_error")
    assert gates(write_run(tmp_path, "smoke", episodes, model=NEW, ps=ON_GPU))["S0"] is False


# 개발용 선택 규칙: +5%p 이상, 통과된 위반 0이고 정답에 없는 쓰기가 P0·7B 참조 이하, 호출당 시간 3배
# 이하(docs/experiments.md 8단계 "선택 규칙").
# ---------------------------------------------------------------- the selection rule


# 개발용 24과제 × 4회 실행을 만든다. successes개의 성공을 앞 과제부터 채우고, 정답에 없는 쓰기와 위반은 첫
# 에피소드에 몰아 넣는다(실행 전체 합만 보므로).
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


# 마크다운 표에서 첫 칸(실행 이름) → 그 줄 전체를 돌려준다.
def rows(text):
    return {line.split(" | ")[0].strip("| "): line for line in text.splitlines() if line.startswith("| ")}


# 세 조건을 모두 만족하는 후보(+15.6%p, 정답에 없는 쓰기 11 = P0 참조 11, 호출당 1,400 ms < 기준 500 ms의
# 3배)가 선택되고, 다음 단계로 시험용을 말하는지 본다.
def test_a_candidate_that_meets_all_three_conditions_is_selected(tmp_path):
    base = dev_run(tmp_path, "base", 15)
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384, unexpected=11, violations=["x"] * 9)
    cand = dev_run(tmp_path, "m2a", 30, model=NEW, unexpected=11, ms=1_400.0)
    text = analyze.select(base, p0, [cand])
    assert "선택: m2a" in text and "시험용" in text


# +4.2%p(96회 중 4회 더 성공)는 +5%p에 못 미쳐 선택되지 않는지 본다.
def test_a_gain_below_five_points_is_not_selected(tmp_path):
    base = dev_run(tmp_path, "base", 15)
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384, unexpected=11)
    cand = dev_run(tmp_path, "m2a", 19, model=NEW)  # +4 successes of 96 = +4.2%p
    assert "선택: 없음" in analyze.select(base, p0, [cand])


# 성공률이 크게 올라도 정답에 없는 쓰기가 P0·7B 참조(11)보다 하나라도 많으면(12) 선택되지 않는지 본다. 8단계
# 쓰기 조건의 기준은 2단계와 달리 P0·7B 참조다(docs/experiments.md 8단계 "재기 전에 고친 것" 3).
def test_more_unexpected_writes_than_the_p0_reference_are_not_selected(tmp_path):
    base = dev_run(tmp_path, "base", 15)
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384, unexpected=11)
    cand = dev_run(tmp_path, "m2a", 40, model=NEW, unexpected=12)
    assert "선택: 없음" in analyze.select(base, p0, [cand])


# 통과된 규정 위반이 하나라도 있으면 선택되지 않는지 본다. P1에서는 위반이 도구에서 막히므로 구조상 0이고, 0이
# 아니면 도구 버그다.
def test_a_violation_let_through_is_never_selected(tmp_path):
    base = dev_run(tmp_path, "base", 15)
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384, unexpected=11)
    cand = dev_run(tmp_path, "m2a", 40, model=NEW, unexpected=1, violations=["return_window_expired"])
    assert "선택: 없음" in analyze.select(base, p0, [cand])


# 호출당 시간이 기준의 3배(1,500 ms)를 1 ms라도 넘으면 선택되지 않는지 본다. 2단계 14B가 이 조건(7.8배)에
# 걸렸다.
def test_calls_more_than_three_times_slower_are_not_selected(tmp_path):
    base = dev_run(tmp_path, "base", 15, ms=500.0)
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384, unexpected=11)
    cand = dev_run(tmp_path, "m2a", 40, model=NEW, ms=1_501.0)
    assert "선택: 없음" in analyze.select(base, p0, [cand])


# 두 후보가 모두 조건을 만족하면 pass^1이 높은 쪽(m2b)을, 같으면 먼저 준 작은 모델(m2a)을 고르는지 본다(선택
# 규칙 4).
def test_of_two_qualified_candidates_the_higher_wins_and_a_tie_goes_to_the_first(tmp_path):
    base = dev_run(tmp_path, "base", 15)
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384, unexpected=11)
    m2a = dev_run(tmp_path, "m2a", 30, model=NEW)
    m2b = dev_run(tmp_path, "m2b", 35, model="qwen3.5:4b", think=False)
    assert "선택: m2b" in analyze.select(base, p0, [m2a, m2b])
    m2b_tie = dev_run(tmp_path, "m2b-tie", 30, model="qwen3.5:4b", think=False)
    assert "선택: m2a" in analyze.select(base, p0, [m2a, m2b_tie])


# 후보의 에이전트 num_ctx가 기준과 다르면 짝지을 수 없어 선택되지 않고 이유(num_ctx)가 적히는지 본다. 짝짓기는
# 에이전트 모델 말고는 모든 설정이 같아야 한다(선택 규칙 5).
def test_a_candidate_on_another_agent_context_cannot_be_paired(tmp_path):
    base = dev_run(tmp_path, "base", 15)
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384, unexpected=11)
    cand = dev_run(tmp_path, "m2a", 40, model=NEW, num_ctx=16384)
    text = analyze.select(base, p0, [cand])
    assert "선택: 없음" in text and "num_ctx" in text


# 시뮬레이터를 전부 CPU에 둔 후보는 선택돼도 시험용은 재지 않는다고 적히는지 본다. CPU 시뮬레이터는 에피소드당
# 약 140초라 GPU 상한 안에 시험용을 넣을 수 없어서 정한 규칙이다(docs/experiments.md 8단계 "재기 전에 고친 것"
# 4).
def test_a_candidate_with_its_simulator_on_the_cpu_stops_at_the_development_tasks(tmp_path):
    base = dev_run(tmp_path, "base", 15)
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384, unexpected=11)
    cand = dev_run(tmp_path, "m2a", 40, model=NEW, user=provider(SIM, num_gpu=0))
    text = analyze.select(base, p0, [cand])
    assert "선택: m2a" in text and "시험용은 재지 않는다" in text


# 기준이 P1이 아니거나, P0 참조가 다른 Ollama 버전(0.34.2)에서 잰 것이면 ValueError로 거절하는지 본다.
# Ollama가 0.35로 바뀌어 이 단계의 비교·참조는 0.35.1에서 새로 잰 실행끼리만 한다는 규칙(docs/experiments.md
# 8단계 "환경이 바뀌어 기준을 모두 다시 잰다")을 코드가 지킨다.
def test_the_references_must_be_what_the_rule_names(tmp_path):
    base = dev_run(tmp_path, "base", 15, policy="P0")
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384, unexpected=11)
    cand = dev_run(tmp_path, "m2a", 40, model=NEW)
    with pytest.raises(ValueError, match="P1"):
        analyze.select(base, p0, [cand])
    old = dev_run(tmp_path, "p0-old", 19, policy="P0", num_ctx=16384, unexpected=11, version="0.34.2")
    with pytest.raises(ValueError, match="Ollama"):
        analyze.select(dev_run(tmp_path, "base2", 15), old, [cand])


# 선택 표의 후보 줄에 pass^1 차이(+15.6%p), 시간 비율(2.00배), 정답에 없는 쓰기 / P0 참조(3 / 11)가 함께
# 적히는지 본다.
def test_the_selection_table_shows_the_report_only_numbers(tmp_path):
    base = dev_run(tmp_path, "base", 15)
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384, unexpected=11)
    cand = dev_run(tmp_path, "m2a", 30, model=NEW, unexpected=3, ms=1_000.0)
    line = rows(analyze.select(base, p0, [cand]))["m2a"]
    assert "+15.6%p" in line and "2.00배" in line and "3 / 11" in line


# 시험용 판정: 2단계 판정표와 같고, 쓰기 조건의 기준만 P0·7B 시험용 참조다.
# ---------------------------------------------------------------- the test verdict


# 시험용 실행을 만든다. successes_per_task[t]는 과제 t의 4회 중 성공 횟수이고, 정답에 없는 쓰기는 첫
# 에피소드에 넣는다.
def make_test_run(tmp_path, name, successes_per_task, **kwargs):
    episodes = [
        ep(f"test-{t + 1:03d}", trial, trial < successes_per_task[t])
        for t in range(len(successes_per_task))
        for trial in range(4)
    ]
    unexpected = kwargs.pop("unexpected", 0)
    episodes[0]["verdict"]["unexpected_writes"] = unexpected
    return write_run(tmp_path, name, episodes, tasks_file="test", **kwargs)


# 판정의 마지막 줄이 세 경우로 나뉘는지 본다: 구간이 0보다 위이고 정답에 없는 쓰기가 P0 참조 이하(19 ≤ 19)면
# '개선', 하나 많으면(20) '성공률은 올랐지만 위험한 쓰기가 늘었음', 성공률이 같으면 '개선을 확인하지 못함'.
def test_the_verdict_follows_the_stage_2_table_with_the_p0_write_reference(tmp_path):
    base = make_test_run(tmp_path, "base", [1] * 10 + [0] * 30)
    p0 = make_test_run(tmp_path, "p0", [1] * 10 + [0] * 30, policy="P0", num_ctx=16384, unexpected=19)
    better = make_test_run(tmp_path, "m2a", [3] * 30 + [0] * 10, model=NEW, unexpected=19)
    assert analyze.verdict(base, p0, better).splitlines()[-1] == "판정: 개선"
    risky = make_test_run(tmp_path, "m2a-risky", [3] * 30 + [0] * 10, model=NEW, unexpected=20)
    assert analyze.verdict(base, p0, risky).splitlines()[-1] == "판정: 성공률은 올랐지만 위험한 쓰기가 늘었음"
    same = make_test_run(tmp_path, "m2a-same", [1] * 10 + [0] * 30, model=NEW)
    assert analyze.verdict(base, p0, same).splitlines()[-1] == "판정: 개선을 확인하지 못함"


# 판정 함수에 개발용 실행을 주면 거절하는지 본다. 판정은 시험용으로만 하고 개발용 수치로 판정을 내리는 실수를
# 막는다.
def test_the_verdict_reads_only_test_runs(tmp_path):
    base = dev_run(tmp_path, "base", 15)
    p0 = dev_run(tmp_path, "p0", 19, policy="P0", num_ctx=16384)
    cand = dev_run(tmp_path, "m2a", 40, model=NEW)
    with pytest.raises(ValueError, match="test"):
        analyze.verdict(base, p0, cand)


# 가린 시뮬레이터 점검(관찰만): 두 개발용 기록에서 같은 (과제, 시도)를 뽑아 구성 이름과 판정을 가리고 섞어
# 읽게 하고, 읽은 표시를 실행별로 센다.
# ---------------------------------------------------------------- the blind simulator check (P5(a))


# 점검표가 시나리오를 보여 줄 수 있도록 실제 개발용 과제 id 앞의 6개를 쓴다.
def blind_task_ids():
    from support_agent.tasks import load_tasks

    return [t.id for t in load_tasks("dev")][:6]


# 과제 6개 × 4회 에피소드를 만든다. 상담원 말(text)로 실행을 구별할 수 있고, 성공은 첫 시도에만 둔다.
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


# 기준과 후보 두 개발용 실행을 만든다.
def two_dev_runs(tmp_path):
    a = write_run(tmp_path, "run-a", talk_episodes("기준 상담원의 말"))
    b = write_run(tmp_path, "run-b", talk_episodes("후보 상담원의 말"), model=NEW)
    return a, b


# 점검표에 실행 이름, 판정(PASS/fail), 종료 이유, 모델 이름이 나오지 않고 시나리오('알고 있는 것')는 나오는지,
# 열쇠 파일(key.json)에는 표시마다 실행이 남는지 본다. 또 같은 명령은 같은 점검표를 내고(고정 seed), 두
# 실행에서 같은 (과제, 시도)를 뽑는지 확인한다. 같은 대화 상황을 짝지어 읽어야 시뮬레이터 잘못을 실행끼리
# 비교할 수 있다.
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


# 실행 두 개일 때의 뽑기(BOOTSTRAP_SEED)와 섞기(BOOTSTRAP_SEED + 1) 결과가 이 테스트가 같은 seed로 직접 계산한
# 배치와 같은지 본다. 셋 이상을 받게 고친 뒤에도 8단계에 실제로 만든 두 장이 같은 명령으로 다시 만들어져야
# 한다(docs/experiments.md 8단계 날짜 메모).
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


# 실행 세 개를 한 장에 넣으면 표시 15개가 모두 가려지고 세 실행이 같은 (과제, 시도) 5개를 공유하는지 본다.
# 8단계에서 기준을 공유한 두 장을 따로 만들었더니 배치가 같아 둘째 장이 사실상 가려지지 않았고, 그 뒤 한
# 점검의 모든 실행을 한 장에 넣기로 했다(같은 날짜 메모, 커밋 ccfa603).
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


# 실행이 하나뿐이거나 같은 실행을 두 번 주면 거절하는지 본다. 비교할 것이 없거나 가림이 무의미하다.
def test_the_blind_check_needs_two_different_runs(tmp_path):
    a, _ = two_dev_runs(tmp_path)
    with pytest.raises(ValueError, match="two or more"):
        analyze.blind(a, n=1, out_dir=tmp_path / "one")
    with pytest.raises(ValueError, match="same run twice"):
        analyze.blind(a, a, n=1, out_dir=tmp_path / "twice")


# 읽는 사람이 적은 표시(marks.json)를 열쇠로 풀어 실행별 에피소드 수와 '결과를 바꿨을 만한 잘못'·코드 수를
# 세는지 본다.
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


# 점검표 코드에 없는 표시(made_up)나 빠진 표시 번호(B01)가 있으면 세지 않고 거절하는지 본다. 오타나 빠뜨린
# 항목이 조용히 0으로 세어지는 것을 막는다.
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


# 시험용 기록으로는 점검표를 만들지 않는지 본다. 시험용 에피소드를 읽고 무엇을 고치지 않는다는
# 규칙(docs/plan.md "시험용 과제 오염")을 코드로 막는다.
def test_the_blind_check_never_reads_test_records(tmp_path):
    a = write_run(tmp_path, "t-a", [ep("test-001", 0, True)], tasks_file="test")
    b = write_run(tmp_path, "t-b", [ep("test-001", 0, True)], tasks_file="test", model=NEW)
    with pytest.raises(ValueError, match="test"):
        analyze.blind(a, b, n=1, out_dir=tmp_path / "blind")
