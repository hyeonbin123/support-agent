# [테스트 묶음] 실행기(run.py)의 요약(summary.json)과 에피소드 실행의 오류 처리를 확인한다: 성공·실패를 세는
# 법, infra_error를 분모에서 빼는 법, 모델 이름을 폴더 이름으로 바꾸는 법. 끝의 테스트들은 manifest의 하니스
# 글 해시(harness_sha256)를 본다.
# [왜 중요한가] 판정 규칙에서 infra_error(제공자 오류, 도구 버그)는 에이전트의 실패가 아니라 분모에서 뺀다
# (docs/design.md "에이전트 루프 규칙"). 이것이 틀리면 GPU·서버 문제가 성공률을 깎는다. 하니스 글 해시가
# 틀리면 안내문·리마인더·오류 글이 다른 두 실행이 analyze same-setup에서 같은 설정으로 짝지어진다.

import ast
import inspect
import json

from test_run_voice import FakeOllama
from test_tasks import ALL_TASKS, CONTACTS, episode, say_values

from support_agent import agent, analyze, run, tools, user_sim
from support_agent.chat import ScriptedProvider, ToolCall


# 성공 1, 실패 2, infra_error 1인 결과에서 에피소드는 4개로 세되 과제별 성공은 1/3이고 pass^1 = 1/3, pass^3 =
# 0이어야 한다. infra_error 시도는 분자에도 분모에도 들어가지 않는다.
def test_summary_counts_successes_failures_and_leaves_infra_errors_out():
    task = next(t for t in ALL_TASKS if t.id == "smoke-action-01")
    name, contact = CONTACTS[task.customer_id]
    good = [ToolCall("find_customer", {"name": name, "contact": contact})]
    good += [ToolCall(a.tool, a.args) for a in task.gold_actions] + [say_values(task)]
    results = [episode(task, good), episode(task, ["도와드릴 수 없습니다."]), episode(task, ["안 됩니다."])]
    assert [r.status for r in results] == ["completed", "completed", "completed"]

    broken = episode(task, good)
    broken.status, broken.verdict = "infra_error", None
    summary = run.summarise([*results, broken])
    assert summary["episodes"] == 4 and summary["infra_errors"] == 1
    assert summary["by_task"] == {"smoke-action-01": "1/3"}
    assert summary["pass_hat_k"][1] == 1 / 3 and summary["pass_hat_k"][3] == 0.0


# 모델 호출이 제공자 오류(시간 초과)로 실패하면 run_episode가 예외로 죽지 않고 infra_error, 판정 없음(verdict
# None), 오류 문구를 기록해야 한다. 실제 측정에서 Ollama가 멈추면 이렇게 기록되어 분모에서 빠진다.
def test_a_script_that_runs_out_is_an_infra_error_not_a_task_failure():
    class Boom(ScriptedProvider):
        def chat(self, *args, **kwargs):
            from support_agent.chat import ProviderError

            raise ProviderError("timeout")

    from support_agent.agent import load_policy
    from support_agent.config import RunConfig
    from support_agent.episode import run_episode
    from support_agent.seed import build_seed_engine
    from support_agent.tools import build_registry
    from support_agent.user_sim import ScriptedUser

    result = run_episode(
        ALL_TASKS[0],
        0,
        config=RunConfig(),
        provider=Boom([]),
        user=ScriptedUser(["안녕하세요"]),
        registry=build_registry(),
        seed_engine=build_seed_engine(),
        policy_text=load_policy(),
    )
    assert result.status == "infra_error" and result.verdict is None and "timeout" in result.error


# 모델 이름의 ':' 같은 문자는 Windows 폴더 이름에 쓸 수 없으므로 실행 id를 만들 때 '-'로 바꾼다.
def test_run_ids_are_safe_windows_folder_names():
    assert run.safe_name("qwen2.5:7b-instruct") == "qwen2.5-7b-instruct"


# pass^k는 과제마다 유효한 시도 전부(그 과제의 n)로 C(c,k)/C(n,k)를 계산한다. 첫 시도 하나만 보면 0이 되는
# 경우에도 2/3이 나와야 한다. 모든 시도가 infra_error인 과제는 표에서 조용히 사라지지 않고
# tasks_without_valid_trials에 이름이 남는다.
def test_pass_hat_k_uses_every_valid_trial_and_names_tasks_that_were_never_measured():
    task_a = next(t for t in ALL_TASKS if t.id == "smoke-action-01")
    task_b = next(t for t in ALL_TASKS if t.id == "smoke-lookup-01")
    name, contact = CONTACTS[task_a.customer_id]
    good = [ToolCall("find_customer", {"name": name, "contact": contact})]
    good += [ToolCall(a.tool, a.args) for a in task_a.gold_actions] + [say_values(task_a)]
    results = [episode(task_a, ["안 됩니다."]), episode(task_a, good), episode(task_a, good)]
    lost = episode(task_b, ["모르겠습니다."])
    lost.status, lost.verdict = "infra_error", None
    summary = run.summarise([*results, lost])
    assert summary["by_task"] == {"smoke-action-01": "2/3"}
    assert summary["pass_hat_k"][1] == 2 / 3  # 0.0이 아니다: 첫 시도만 보면 0.0이 나온다
    assert summary["tasks_without_valid_trials"] == ["smoke-lookup-01"]


# manifest에 하니스 글의 해시(harness_sha256)가 남는지: 실행기를 가짜 모델로 끝까지 돌려 manifest의 값이 지금
# 코드의 run.harness_sha256()과 같고 등록한 키를 모두 갖는지 본다. prompt_sha256 넷(규정 문서, 시스템
# 프롬프트, 시뮬레이터 프롬프트, 도구 목록)은 모델이 실제로 보는 글 가운데 시뮬레이터 턴 리마인더, 하니스
# 안내문, 고정 인사·이관 문장, 도구 오류 글을 덮지 않아, 그 글이 바뀌어도 analyze same-setup이 알아보지 못했다
# (docs/experiments.md 공통 규칙의 2026-10-29 메모).
def test_the_manifest_records_the_hashes_of_the_harness_texts(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OllamaProvider", FakeOllama)
    monkeypatch.setattr(run, "gpu_used_by_others_mib", lambda provider: None)
    monkeypatch.setattr(run, "OUTPUTS", tmp_path)
    monkeypatch.setattr(
        "sys.argv", ["run", "--tasks", "smoke", "--task-id", "smoke-lookup-01", "--label", "unit"]
    )
    run.main()
    (run_dir,) = (tmp_path / "runs").iterdir()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["harness_sha256"] == run.harness_sha256()
    assert set(manifest["harness_sha256"]) == {
        "turn_reminder",
        "stall_notice",
        "cut_off_notice",
        "format_notice",
        "language_notice",
        "claim_notice",
        "repeat_notice",
        "first_agent_message",
        "handoff_message",
        "nothing_heard",
        "confirm_notice",
        "tool_errors",
        "rules",
    }
    assert all(len(digest) == 64 for digest in manifest["harness_sha256"].values())


# 하니스 글 하나를 바꾸면 그 키의 해시만 달라져야 한다. 상수를 가져다 복사해 두지 않고 그 모듈의 속성을 읽을
# 때만 이렇게 되고, 그래야 안내문이나 리마인더를 고친 코드로 돈 실행이 옛 기록과 same-setup에서 걸린다.
def test_each_harness_hash_follows_its_own_text(monkeypatch):
    before = run.harness_sha256()
    monkeypatch.setattr(agent, "STALL_NOTICE", agent.STALL_NOTICE + " 다시 답하세요.")
    after = run.harness_sha256()
    assert after["stall_notice"] != before["stall_notice"]
    assert {k: v for k, v in after.items() if k != "stall_notice"} == {
        k: v for k, v in before.items() if k != "stall_notice"
    }
    monkeypatch.setattr(user_sim, "TURN_REMINDER", "(고객 역할 지침: 다른 글)")
    assert run.harness_sha256()["turn_reminder"] != before["turn_reminder"]


# 끝에서 끝까지: 같은 명령을 안내문 하나만 바꾼 코드로 다시 돌린 실행은 바꾸기 전 실행과 analyze
# same-setup에서 그 키로 걸린다. 하니스 글 해시가 manifest에 없던 때에는 두 manifest가 같아 '같은 설정'으로
# 짝지어졌다(이 테스트가 고치기 전 코드에서 실패하는 이유). 가짜 모델로 돌려 대화는 두 실행이 같다.
def test_a_run_with_a_changed_notice_does_not_pair_with_the_original(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "OllamaProvider", FakeOllama)
    monkeypatch.setattr(run, "gpu_used_by_others_mib", lambda provider: None)
    monkeypatch.setattr(run, "OUTPUTS", tmp_path)
    argv = ["run", "--tasks", "smoke", "--task-id", "smoke-lookup-01", "--label"]
    monkeypatch.setattr("sys.argv", [*argv, "before"])
    run.main()
    monkeypatch.setattr(agent, "STALL_NOTICE", agent.STALL_NOTICE + " 다시 답하세요.")
    monkeypatch.setattr("sys.argv", [*argv, "after"])
    run.main()
    runs = {path.name.rsplit("-", 1)[-1]: path for path in (tmp_path / "runs").iterdir()}
    assert analyze.setup_differences(runs["before"], runs["after"]) == [
        "harness_sha256 differs (stall_notice)"
    ]


# 도구 오류 글은 tools.py 소스에서 ctx.require·ctx.check_policy의 오류 코드와 메시지만 ast로 모아 모델이 읽는
# `Error: [code] message`의 `[code] message` 틀로 만든다: 위치 인자든 code=·message= 키워드든, f-string이면
# {식} 자리를 남긴 틀로, 코드와 메시지가 모두 같은 것은 한 번만(메시지가 같아도 코드가 다르면 따로), 정렬해서.
# 주석과 다른 문자열은 들어가지 않으므로 한국어 주석을 고쳐도 해시가 흔들리지 않는다. rules.py에서는 모듈 수준
# 숫자 상수(오류 글의 "7일"·"3,000원"을 채우는 값)를 '이름=값'으로 모은다.
def test_tool_error_texts_and_rule_numbers_are_read_from_the_source():
    source = """
# 주석은 해시에 들어가지 않는다
HELP = "이 문자열은 오류 글이 아니다"


def handler(ctx, order, days, kind):
    ctx.require(order is not None, "order_not_found", "주문을 찾을 수 없습니다.")
    ctx.check_policy(
        days <= LIMIT,
        f"{kind}_window_expired",
        f"기한({rules.RETURN_WINDOW_DAYS}일)이 지났습니다.",
        facts=lambda: "(사실)",
    )
    ctx.require(order.status != "cancelled", message="이미 취소된 주문입니다.", code="already_cancelled")
    ctx.require(True, "order_not_found", "주문을 찾을 수 없습니다.")
    ctx.require(True, "line_not_found", "주문을 찾을 수 없습니다.")
    other.require(True, "x", "ctx가 아닌 것의 글은 세지 않는다")
"""
    assert run.error_texts(source) == [
        "[already_cancelled] 이미 취소된 주문입니다.",
        "[line_not_found] 주문을 찾을 수 없습니다.",
        "[order_not_found] 주문을 찾을 수 없습니다.",
        "[{kind}_window_expired] 기한({rules.RETURN_WINDOW_DAYS}일)이 지났습니다.",
    ]
    assert run.numeric_constants("A = 7\nB = 3000  # 원\nNAME = 'x'\nC = A + 1\n") == ["A=7", "B=3000"]
    source = inspect.getsource(tools)
    texts = run.error_texts(source)
    assert "[identity_not_verified] 본인 확인이 되지 않았습니다." in texts
    assert (
        "[{kind}_window_expired] {word} 기한(수령일 다음 날부터 {rules.RETURN_WINDOW_DAYS}일)이 지났습니다."
        in texts
    )
    assert texts == sorted(set(texts))
    # tools.py는 두 검사를 모두 `ctx.`로 부른다. 다른 이름으로 부르는 호출이 생기면 error_texts가 그 글을
    # 놓친다.
    checks = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("require", "check_policy")
    ]
    assert checks and all(isinstance(n.func.value, ast.Name) and n.func.value.id == "ctx" for n in checks)
    # 오류 코드와 메시지도 모두 소스에 적힌 글(문자열이나 f-string)이다. 변수에 담아 넘기는 호출이 생기면
    # error_texts가 그 자리를 <식>으로만 적어, 그 변수의 글이 바뀌어도 해시가 같다.
    written = [
        arg
        for n in checks
        for arg in [*n.args[1:3], *(k.value for k in n.keywords if k.arg in ("code", "message"))]
    ]
    assert len(written) == 2 * len(checks)
    assert all(isinstance(arg, ast.Constant | ast.JoinedStr) for arg in written)
