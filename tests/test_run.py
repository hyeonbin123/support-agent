from test_tasks import ALL_TASKS, CONTACTS, episode, say_values

from support_agent import run
from support_agent.chat import ScriptedProvider, ToolCall


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


def test_run_ids_are_safe_windows_folder_names():
    assert run.safe_name("qwen2.5:7b-instruct") == "qwen2.5-7b-instruct"


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
