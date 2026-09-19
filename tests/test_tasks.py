"""Every task file is checked without a GPU: the task itself, a scripted agent that follows the gold path
(must pass) and agents that do nothing useful (must fail)."""

from __future__ import annotations

import pytest

from support_agent import db
from support_agent.agent import load_policy
from support_agent.chat import ScriptedProvider, ToolCall
from support_agent.config import RunConfig
from support_agent.episode import run_episode
from support_agent.judge import validate_task
from support_agent.paths import TASKS
from support_agent.seed import build_seed_engine
from support_agent.tasks import Task, load_tasks
from support_agent.tools import build_registry
from support_agent.user_sim import ScriptedUser

REGISTRY = build_registry()
ALL_TASKS = [task for path in sorted(TASKS.glob("*.yaml")) for task in load_tasks(path)]
CONTACTS = {  # how the scripted agent verifies each fixture customer
    "C-9001": ("김하준", "010-0000-9001"),
    "C-9002": ("이서연", "seoyeon.lee@example.com"),
    "C-9003": ("박도윤", "010-0000-9003"),
    "C-9004": ("최지우", "010-0000-9004"),
    "C-9005": ("정예준", "010-0000-9005"),
}


def say_values(task: Task) -> str:
    parts = []
    for required in task.required_values:
        if required.kind == "number":
            parts.append(f"{required.label}은 {required.value:,}원입니다.")
        elif required.kind == "date":
            parts.append(f"{required.label}은 {required.value.month}월 {required.value.day}일입니다.")
        else:
            parts.append(f"{required.label}: {required.value}")
    return " ".join(parts) or "처리되었습니다."


def episode(task: Task, script, policy: str = "P1"):
    return run_episode(
        task,
        0,
        config=RunConfig(policy=policy),
        provider=ScriptedProvider(script),
        user=ScriptedUser(["문의드립니다."]),
        registry=REGISTRY,
        seed_engine=build_seed_engine(),
        policy_text=load_policy(),
    )


def test_there_are_tasks():
    assert len(ALL_TASKS) >= 5 and len({task.id for task in ALL_TASKS}) == len(ALL_TASKS)


@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda task: task.id)
def test_the_task_is_well_formed(task):
    assert validate_task(task, build_seed_engine(), REGISTRY) == []


@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda task: task.id)
def test_an_agent_that_follows_the_gold_path_passes(task):
    name, contact = CONTACTS[task.customer_id]
    script = [ToolCall("find_customer", {"name": name, "contact": contact})]
    script += [ToolCall(action.tool, action.args) for action in task.gold_actions]
    script.append(say_values(task))
    result = episode(task, script)
    assert result.status == "completed" and result.verdict.success, result.verdict
    assert result.verdict.unexpected_writes == 0 and result.verdict.missing_writes == 0


@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda task: task.id)
def test_an_agent_that_only_talks_fails(task):
    result = episode(task, ["규정상 수령 후 7일이 지나면 반품이 어렵습니다. 도와드릴 수 없습니다."])
    assert result.status == "completed" and not result.verdict.success


@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda task: task.id)
def test_a_needless_handoff_fails(task):
    script = [ToolCall("transfer_to_human", {"reason": "customer_request", "summary": say_values(task)})]
    result = episode(task, script)
    assert result.termination == "handoff" and not result.verdict.success


@pytest.mark.parametrize("task", [t for t in ALL_TASKS if t.forbidden_actions], ids=lambda task: task.id)
def test_the_forbidden_call_passes_under_p0_and_fails_the_task(task):
    name, contact = CONTACTS[task.customer_id]
    forbidden = task.forbidden_actions[0]
    script = [
        ToolCall("find_customer", {"name": name, "contact": contact}),
        ToolCall(forbidden.tool, forbidden.args),
        say_values(task),
    ]
    p0 = episode(task, script, policy="P0")
    assert not p0.verdict.success and not p0.verdict.db_match
    assert p0.verdict.policy_violations == [forbidden.expect_code] and p0.verdict.unexpected_writes == 1
    p1 = episode(task, script, policy="P1")
    assert (
        p1.verdict.success and p1.verdict.blocked_forbidden_attempts == 1
    )  # blocked, then refused correctly


def test_the_record_is_json_and_episodes_do_not_touch_the_seed():
    before = db.state_hash(db.dump_db(build_seed_engine()))
    task = next(t for t in ALL_TASKS if t.gold_actions)
    name, contact = CONTACTS[task.customer_id]
    script = [ToolCall("find_customer", {"name": name, "contact": contact})]
    script += [ToolCall(a.tool, a.args) for a in task.gold_actions] + [say_values(task)]
    result = episode(task, script)
    assert '"task_id"' in result.to_json_line() and result.db_changes
    assert result.final_hash == result.gold_hash != result.seed_hash == before
    assert db.state_hash(db.dump_db(build_seed_engine())) == before
