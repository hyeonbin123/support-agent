# [테스트 묶음] 실행기(run.py)의 요약(summary.json)과 에피소드 실행의 오류 처리를 확인한다: 성공·실패를 세는
# 법, infra_error를 분모에서 빼는 법, 모델 이름을 폴더 이름으로 바꾸는 법.
# [왜 중요한가] 판정 규칙에서 infra_error(제공자 오류, 도구 버그)는 에이전트의 실패가 아니라 분모에서 뺀다
# (docs/design.md "에이전트 루프 규칙"). 이것이 틀리면 GPU·서버 문제가 성공률을 깎는다.

from test_tasks import ALL_TASKS, CONTACTS, episode, say_values

from support_agent import run
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
    assert summary["pass_hat_k"][1] == 2 / 3  # not 0.0: the first trial alone would say so
    assert summary["tasks_without_valid_trials"] == ["smoke-lookup-01"]
