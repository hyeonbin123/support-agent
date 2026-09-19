"""One episode: a simulated customer talks to the agent over a private DB copy, then it is judged."""

from __future__ import annotations

import time

from sqlalchemy import Engine

from support_agent import db
from support_agent.agent import AgentState, agent_turn, build_system_prompt, new_state
from support_agent.chat import ChatProvider, ProviderError
from support_agent.config import (
    FIRST_AGENT_MESSAGE,
    JUDGED,
    OUT_OF_SCOPE_TOKEN,
    STOP_TOKEN,
    RunConfig,
    Termination,
)
from support_agent.judge import gold_engine, judge
from support_agent.records import EpisodeResult
from support_agent.tasks import Task
from support_agent.toolkit import Registry, ToolBugError, ToolContext, execute
from support_agent.user_sim import SimulatorError, User


def gold_dump_of(task: Task, seed_engine: Engine, registry: Registry) -> db.Dump:
    """The database a correct episode ends with. Computed once per task and reused for every trial."""
    engine = gold_engine(task, seed_engine, registry)
    try:
        return db.dump_db(engine)
    finally:
        engine.dispose()


def _converse(
    state: AgentState, user: User, task: Task, trial: int, config: RunConfig, provider, registry, run_tool
) -> Termination:
    agent_text = FIRST_AGENT_MESSAGE
    for _ in range(config.max_user_turns):
        user_text = user.reply(agent_text)
        if STOP_TOKEN in user_text:
            return "user_stop"
        if OUT_OF_SCOPE_TOKEN in user_text:
            return "out_of_scope"
        turn = agent_turn(
            state,
            user_text,
            provider=provider,
            registry=registry,
            run_tool=run_tool,
            config=config,
            task_id=task.id,
            trial=trial,
        )
        if turn.stop:
            return turn.stop
        agent_text = turn.reply or ""
    return "max_user_turns"


def run_episode(
    task: Task,
    trial: int,
    *,
    config: RunConfig,
    provider: ChatProvider,
    user: User,
    registry: Registry,
    seed_engine: Engine,
    policy_text: str,
    gold_dump: db.Dump | None = None,
    run_id: str = "",
) -> EpisodeResult:
    started = time.perf_counter()
    if gold_dump is None:
        gold_dump = gold_dump_of(task, seed_engine, registry)
    seed_dump = db.dump_db(seed_engine)
    engine = db.memory_engine(seed_engine)  # private copy of this episode
    ctx = ToolContext(now=task.now, enforce_policy=config.enforce_policy)  # the task's fixed clock
    state = new_state(build_system_prompt(policy_text, task.now))

    def run_tool(name: str, arguments: dict):
        return execute(registry, engine, ctx, name, arguments)

    error = ""
    try:
        termination = _converse(state, user, task, trial, config, provider, registry, run_tool)
    except (ProviderError, SimulatorError, ToolBugError) as exc:
        # The model server, the simulator or our own tool code failed: nobody's task failure.
        termination, error = "infra_error", f"{type(exc).__name__}: {exc}"

    final_dump = db.dump_db(engine)
    engine.dispose()
    verdict = None
    if termination != "infra_error":
        verdict = judge(
            task,
            final_dump=final_dump,
            gold_dump=gold_dump,
            messages=state.messages,
            tool_log=state.tool_log,
            termination=termination,
            registry=registry,
        )
    status = "infra_error" if verdict is None else ("completed" if termination in JUDGED else "truncated")
    return EpisodeResult(
        run_id=run_id,
        task_id=task.id,
        task_sha256=task.sha256(),
        trial=trial,
        status=status,
        termination=termination,
        verdict=verdict,
        error=error,
        messages=[m.to_dict() for m in state.messages if m.role != "system"],
        user_messages=[m.to_dict() for m in user.messages if m.role != "system"],
        tool_calls=state.tool_log,
        llm_calls=[*state.llm_log, *user.llm_log],
        db_changes=db.diff_dumps(seed_dump, final_dump),
        seed_hash=db.state_hash(seed_dump),
        final_hash=db.state_hash(final_dump),
        gold_hash=db.state_hash(gold_dump),
        wall_seconds=round(time.perf_counter() - started, 3),
    )
